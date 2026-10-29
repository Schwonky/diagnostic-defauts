"""Chargement du jeu NEU-DET (classification de 6 défauts de surface).

Structure attendue :
    data/images/NEU-DET/train/images/<classe>/*.jpg
    data/images/NEU-DET/validation/images/<classe>/*.jpg

Découpage utilisé :
    - train/      → 85 % entraînement, 15 % validation (choix du meilleur modèle)
    - validation/ → TEST : jamais utilisé pendant l'entraînement, sert uniquement à l'évaluation finale
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_NEU = RACINE / "data" / "images" / "NEU-DET"
TAILLE_IMAGE = 224
GRAINE = 42

# Statistiques ImageNet : le ResNet18 préentraîné a été entraîné avec cette normalisation
MOYENNE = [0.485, 0.456, 0.406]
ECART_TYPE = [0.229, 0.224, 0.225]

transformation_evaluation = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # images NEU en niveaux de gris → 3 canaux pour ResNet
    transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART_TYPE),
])

# Augmentation de données : uniquement pour l'entraînement
transformation_entrainement = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MOYENNE, ECART_TYPE),
])


def _dossier(partie):
    chemin = DOSSIER_NEU / partie / "images"
    if not chemin.is_dir():
        raise FileNotFoundError(
            f"Dossier introuvable : {chemin}\n"
            "Vérifiez que le jeu NEU-DET est décompressé dans data/images/."
        )
    return chemin


def _decoupage_stratifie(cibles, part_validation):
    """Sépare les indices en entraînement / validation en gardant la même proportion de chaque classe."""
    generateur = torch.Generator().manual_seed(GRAINE)
    indices_entr, indices_val = [], []
    for classe in sorted(set(cibles)):
        indices = [i for i, c in enumerate(cibles) if c == classe]
        melange = torch.randperm(len(indices), generator=generateur).tolist()
        n_val = round(len(indices) * part_validation)
        indices_val += [indices[j] for j in melange[:n_val]]
        indices_entr += [indices[j] for j in melange[n_val:]]
    return indices_entr, indices_val


def charger_donnees(taille_lot=32, part_validation=0.15):
    """Retourne (chargeur_entrainement, chargeur_validation, chargeur_test, noms_classes)."""
    dossier_train = _dossier("train")
    base_entr = datasets.ImageFolder(dossier_train, transform=transformation_entrainement)
    base_val = datasets.ImageFolder(dossier_train, transform=transformation_evaluation)
    test = datasets.ImageFolder(_dossier("validation"), transform=transformation_evaluation)

    if base_entr.classes != test.classes:
        raise ValueError(f"Classes différentes entre train {base_entr.classes} et test {test.classes}")

    idx_entr, idx_val = _decoupage_stratifie(base_entr.targets, part_validation)
    entrainement = Subset(base_entr, idx_entr)
    validation = Subset(base_val, idx_val)

    # num_workers=0 : évite les problèmes de multiprocessus sous Windows
    return (
        DataLoader(entrainement, batch_size=taille_lot, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=taille_lot, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=taille_lot, shuffle=False, num_workers=0),
        base_entr.classes,
    )
