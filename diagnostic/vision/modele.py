"""Définition du modèle : ResNet18 préentraîné sur ImageNet, dernière couche remplacée."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models

RACINE = Path(__file__).resolve().parents[2]
CHEMIN_MODELE = RACINE / "modeles" / "vision_resnet18.pt"


def creer_modele(nb_classes, preentraine=True):
    poids = models.ResNet18_Weights.DEFAULT if preentraine else None
    modele = models.resnet18(weights=poids)
    modele.fc = nn.Linear(modele.fc.in_features, nb_classes)
    return modele


def sauvegarder(modele, classes, metriques, chemin=CHEMIN_MODELE):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"etat": modele.state_dict(), "classes": classes, "metriques": metriques}, chemin)


def charger(chemin=CHEMIN_MODELE):
    """Retourne (modele, classes). Le modèle est en mode évaluation."""
    if not Path(chemin).exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}\nEntraînez-le d'abord : python -m diagnostic.vision.entrainer"
        )
    sauvegarde = torch.load(chemin, map_location="cpu", weights_only=True)
    modele = creer_modele(len(sauvegarde["classes"]), preentraine=False)
    modele.load_state_dict(sauvegarde["etat"])
    modele.eval()
    return modele, sauvegarde["classes"]
