"""Évaluation finale du modèle sur le jeu de TEST (dossier validation/ de NEU-DET).

Utilisation :
    python -m diagnostic.vision.evaluer

Produit resultats/vision_metriques.json et resultats/vision_matrice_confusion.png.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from diagnostic.vision.donnees import RACINE, charger_donnees
from diagnostic.vision.modele import charger

DOSSIER_RESULTATS = RACINE / "resultats"


def main():
    modele, classes = charger()
    _, _, test, classes_donnees = charger_donnees()
    assert classes == classes_donnees, "Les classes du modèle ne correspondent pas aux données"

    vraies, predites = [], []
    with torch.no_grad():
        for images, cibles in test:
            predites += modele(images).argmax(1).tolist()
            vraies += cibles.tolist()

    rapport = classification_report(vraies, predites, target_names=classes, output_dict=True, zero_division=0)
    print(classification_report(vraies, predites, target_names=classes, digits=3, zero_division=0))

    rappel = rapport["macro avg"]["recall"]
    precision = rapport["macro avg"]["precision"]
    print(f"Rappel moyen    : {rappel:.1%}  (critère ≥ 90 %) → {'ATTEINT' if rappel >= 0.90 else 'NON ATTEINT'}")
    print(f"Précision moyenne : {precision:.1%}  (critère ≥ 85 %) → {'ATTEINT' if precision >= 0.85 else 'NON ATTEINT'}")

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    with open(DOSSIER_RESULTATS / "vision_metriques.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    matrice = confusion_matrix(vraies, predites)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrice, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion – jeu de test NEU-DET")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(DOSSIER_RESULTATS / "vision_matrice_confusion.png", dpi=150)
    print(f"\nRésultats enregistrés dans {DOSSIER_RESULTATS}")


if __name__ == "__main__":
    main()
