"""
Prépare le jeu de données de pneus pour l'entraînement.

Copie les images de data/raw/.../good et .../defective dans
data/processed/pneus/ en trois ensembles :
    - train (70 %) : pour entraîner le modèle
    - val   (15 %) : pour ajuster le modèle pendant l'entraînement
    - test  (15 %) : pour l'évaluation finale (rappel, précision)

La répartition est stratifiée (classe par classe) et utilise une graine
fixe, pour obtenir le même résultat à chaque exécution.

Utilisation (depuis la racine du projet) :
    python scripts/preparer_donnees_pneus.py
"""

import random
import shutil
from pathlib import Path

SOURCE = Path("data/raw/Digital images of defective and good condition tyres")
DESTINATION = Path("data/processed/pneus")
CLASSES = ["good", "defective"]
PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
GRAINE = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def lister_images(dossier):
    return sorted(
        f for f in dossier.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )


def repartir(images):
    images = images.copy()
    random.Random(GRAINE).shuffle(images)
    n = len(images)
    n_train = round(n * PROPORTIONS["train"])
    n_val = round(n * PROPORTIONS["val"])
    return {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }


def main():
    if not SOURCE.exists():
        raise SystemExit(f"Dossier introuvable : {SOURCE}")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    totaux = {ens: {} for ens in PROPORTIONS}

    for classe in CLASSES:
        images = lister_images(SOURCE / classe)
        if not images:
            raise SystemExit(f"Aucune image dans {SOURCE / classe}")
        for ensemble, fichiers in repartir(images).items():
            cible = DESTINATION / ensemble / classe
            cible.mkdir(parents=True, exist_ok=True)
            for f in fichiers:
                shutil.copy2(f, cible / f.name)
            totaux[ensemble][classe] = len(fichiers)

    print(f"\nImages copiees dans {DESTINATION}\n")
    print(f"{'Ensemble':<10}{'good':>12}{'defective':>12}{'Total':>10}")
    print("-" * 44)
    grand_total = 0
    for ensemble, compte in totaux.items():
        total = sum(compte.values())
        grand_total += total
        print(f"{ensemble:<10}{compte['good']:>12}{compte['defective']:>12}{total:>10}")
    print("-" * 44)
    print(f"{'Total':<10}{'':>24}{grand_total:>10}\n")


if __name__ == "__main__":
    main()
