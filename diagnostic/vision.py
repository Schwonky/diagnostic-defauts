"""Vision : photo → état de la pièce (bon état / endommagé) et type de défaut de surface.

Deux modèles ResNet18, entraînés dans les notebooks :
    - modèle d'état (entrainement_v2.ipynb)  : la pièce est-elle endommagée ?
    - modèle NEU    (entrainement_neu.ipynb) : quel défaut de surface ? (6 classes NEU-DET)
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

from diagnostic import RACINE
from diagnostic.base_connaissances import BaseConnaissances

MODELE_ETAT = RACINE / "modeles" / "v2_etat_resnet18.pt"
MODELE_NEU = RACINE / "modeles" / "vision_resnet18.pt"

# Normalisation ImageNet : celle du ResNet18 préentraîné
MOYENNE, ECART_TYPE = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def transformation_etat(taille=224):
    """Redimensionnement puis carré central de « taille » pixels (même rapport que 256 → 224)."""
    return transforms.Compose([
        transforms.Resize(round(taille * 256 / 224)),
        transforms.CenterCrop(taille),
        transforms.ToTensor(),
        transforms.Normalize(MOYENNE, ECART_TYPE),
    ])


transfo_etat = transformation_etat(224)
transfo_neu = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # images NEU en niveaux de gris → 3 canaux
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART_TYPE),
])


def creer_resnet(nb_classes, preentraine=True):
    """ResNet18 (préentraîné sur ImageNet) dont la dernière couche donne nb_classes scores."""
    modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if preentraine else None)
    modele.fc = nn.Linear(modele.fc.in_features, nb_classes)
    return modele


def charger_resnet(chemin, notebook):
    """Retourne (modèle en mode évaluation, noms des classes, taille d'image) depuis un fichier .pt."""
    if not Path(chemin).exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}\nExécutez d'abord notebooks/{notebook}.")
    sauvegarde = torch.load(chemin, map_location="cpu", weights_only=True)
    modele = creer_resnet(len(sauvegarde["classes"]), preentraine=False)
    modele.load_state_dict(sauvegarde["etat"])
    return modele.eval(), sauvegarde["classes"], sauvegarde.get("taille", 224)


def ouvrir(image):
    """Chemin, objet fichier ou PIL.Image → PIL.Image bien orientée."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return ImageOps.exif_transpose(image)


def preparer_photo(image, recadrage=0.8):
    """Rapproche une photo du format NEU : niveaux de gris, carré central, contraste normalisé, 200×200."""
    image = ouvrir(image).convert("L")
    cote = int(min(image.size) * recadrage)
    gauche, haut = (image.width - cote) // 2, (image.height - cote) // 2
    image = image.crop((gauche, haut, gauche + cote, haut + cote))
    return ImageOps.autocontrast(image, cutoff=1).resize((200, 200), Image.Resampling.LANCZOS)


def grad_cam(modele, tenseur):
    """Zones qui ont le plus influencé la décision : (carte entre 0 et 1, indice de la classe prédite)."""
    sauvegarde = {}

    def capturer(module, entree, sortie):
        sauvegarde["activations"] = sortie
        sortie.register_hook(lambda g: sauvegarde.__setitem__("gradients", g))

    crochet = modele.layer4.register_forward_hook(capturer)
    sortie = modele(tenseur)
    crochet.remove()
    classe = sortie.argmax(1).item()
    modele.zero_grad()
    sortie[0, classe].backward()
    poids = sauvegarde["gradients"].mean(dim=(2, 3), keepdim=True)
    carte = F.relu((poids * sauvegarde["activations"]).sum(dim=1, keepdim=True))
    carte = F.interpolate(carte, size=tenseur.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (carte / (carte.max() + 1e-8)).detach().numpy(), classe


class DetecteurEtat:
    """Modèle v2 : probabilité que la pièce soit endommagée."""

    def __init__(self, chemin=MODELE_ETAT):
        self.modele, self.classes, taille = charger_resnet(chemin, "entrainement_v2.ipynb")
        self.transformation = transformation_etat(taille)  # taille d'image utilisée à l'entraînement

    def proba_endommage(self, image):
        with torch.no_grad():
            p = torch.softmax(self.modele(self.transformation(ouvrir(image).convert("RGB")).unsqueeze(0)), dim=1)[0]
        return p[self.classes.index("endommage")].item()


class ClassifieurDefauts:
    """Modèle NEU : probabilité de chaque défaut de surface (identifiants de la base de connaissances)."""

    def __init__(self, chemin=MODELE_NEU, base=None):
        self.modele, self.classes, _ = charger_resnet(chemin, "entrainement_neu.ipynb")
        self.base = base or BaseConnaissances()
        # Classe NEU (ex. « crazing ») → identifiant de défaut (ex. « microfissures »)
        neu_vers_defaut = {d["classe_neu"]: id_d for id_d, d in self.base.defauts.items()}
        manquantes = set(self.classes) - set(neu_vers_defaut)
        if manquantes:
            raise ValueError(f"Classes absentes de la base de connaissances : {manquantes}")
        self.defauts = [neu_vers_defaut[c] for c in self.classes]

    def predire(self, image):
        """Retourne [{"id", "libelle", "probabilite"}, ...] trié par probabilité décroissante."""
        with torch.no_grad():
            p = torch.softmax(self.modele(transfo_neu(ouvrir(image).convert("RGB")).unsqueeze(0)), dim=1)[0]
        resultats = [{"id": d, "libelle": self.base.defauts[d]["libelle"], "probabilite": round(v, 4)}
                     for d, v in zip(self.defauts, p.tolist())]
        return sorted(resultats, key=lambda r: -r["probabilite"])


class AnalyseurImage:
    """Les deux modèles ensemble, pour la fusion et l'API."""

    def __init__(self, base=None):
        self.classifieur = ClassifieurDefauts(base=base)
        self.detecteur = DetecteurEtat()

    def analyser(self, image, photo_reelle=True):
        """photo_reelle=True : la photo est rapprochée du format NEU avant le modèle de défauts.

        Retourne {"p_endommage": float, "probas_defauts": {identifiant: probabilité}}.
        """
        image = ouvrir(image)
        pour_neu = preparer_photo(image) if photo_reelle else image
        probas = {d["id"]: d["probabilite"] for d in self.classifieur.predire(pour_neu)}
        return {"p_endommage": self.detecteur.proba_endommage(image), "probas_defauts": probas}
