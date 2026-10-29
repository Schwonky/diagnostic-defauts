"""Utilisation du modèle entraîné : image → défauts probables (identifiants de la base de connaissances)."""

import torch
from PIL import Image

from diagnostic.base_connaissances import BaseConnaissances
from diagnostic.vision.donnees import transformation_evaluation
from diagnostic.vision.modele import CHEMIN_MODELE, charger


class ClassifieurDefauts:
    def __init__(self, chemin_modele=CHEMIN_MODELE, base=None):
        self.modele, self.classes = charger(chemin_modele)
        self.base = base or BaseConnaissances()
        # Correspondance : nom de classe NEU (ex. « crazing ») → identifiant de défaut (ex. « microfissures »)
        neu_vers_defaut = {d["classe_neu"]: id_d for id_d, d in self.base.defauts.items()}
        manquantes = set(self.classes) - set(neu_vers_defaut)
        if manquantes:
            raise ValueError(f"Classes absentes de la base de connaissances : {manquantes}")
        self.defauts = [neu_vers_defaut[c] for c in self.classes]

    def predire(self, image):
        """image : chemin de fichier, objet fichier ou PIL.Image.

        Retourne la liste des défauts triés par probabilité décroissante :
        [{"id": "rayures", "libelle": "Rayures", "probabilite": 0.93}, ...]
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        tenseur = transformation_evaluation(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probabilites = torch.softmax(self.modele(tenseur), dim=1)[0].tolist()
        resultats = [
            {"id": d, "libelle": self.base.defauts[d]["libelle"], "probabilite": round(p, 4)}
            for d, p in zip(self.defauts, probabilites)
        ]
        return sorted(resultats, key=lambda r: -r["probabilite"])
