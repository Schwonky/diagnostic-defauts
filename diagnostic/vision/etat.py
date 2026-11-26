"""Modèle version 2 : la pièce est-elle en bon état ou endommagée ?

Le modèle est entraîné par le notebook 07_entrainement_v2.ipynb et enregistré dans modeles/v2_etat_resnet18.pt.
"""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms

RACINE = Path(__file__).resolve().parents[2]
CHEMIN_MODELE_ETAT = RACINE / "modeles" / "v2_etat_resnet18.pt"

transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class DetecteurEtat:
    def __init__(self, chemin=CHEMIN_MODELE_ETAT):
        if not Path(chemin).exists():
            raise FileNotFoundError(f"Modèle introuvable : {chemin}\nExécutez d'abord notebooks/07_entrainement_v2.ipynb.")
        sauvegarde = torch.load(chemin, map_location="cpu", weights_only=True)
        self.classes = sauvegarde["classes"]
        self.modele = models.resnet18(weights=None)
        self.modele.fc = nn.Linear(self.modele.fc.in_features, len(self.classes))
        self.modele.load_state_dict(sauvegarde["etat"])
        self.modele.eval()

    def proba_endommage(self, image):
        """image : chemin, objet fichier ou PIL.Image. Retourne la probabilité que la pièce soit endommagée."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        with torch.no_grad():
            p = torch.softmax(self.modele(transformation(image).unsqueeze(0)), dim=1)[0]
        return p[self.classes.index("endommage")].item()
