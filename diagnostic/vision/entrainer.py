"""Entraînement du classifieur de défauts.

Utilisation :
    python -m diagnostic.vision.entrainer              (8 époques par défaut)
    python -m diagnostic.vision.entrainer --epoques 3  (essai rapide)
"""

import argparse
import time

import torch
from torch import nn

from diagnostic.vision.donnees import GRAINE, charger_donnees
from diagnostic.vision.modele import CHEMIN_MODELE, creer_modele, sauvegarder


def exactitude(modele, chargeur):
    modele.eval()
    corrects = total = 0
    with torch.no_grad():
        for images, cibles in chargeur:
            corrects += (modele(images).argmax(1) == cibles).sum().item()
            total += len(cibles)
    return corrects / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoques", type=int, default=8)
    parser.add_argument("--taux", type=float, default=3e-4, help="taux d'apprentissage")
    args = parser.parse_args()

    torch.manual_seed(GRAINE)
    entr, val, _, classes = charger_donnees()
    print(f"Classes : {classes}")
    print(f"Images : {len(entr.dataset)} entraînement, {len(val.dataset)} validation")

    modele = creer_modele(len(classes))
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=args.taux, weight_decay=1e-4)
    planificateur = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=args.epoques)
    perte_fn = nn.CrossEntropyLoss()
    meilleure = 0.0

    for epoque in range(1, args.epoques + 1):
        debut = time.time()
        modele.train()
        perte_totale = 0.0
        for i, (images, cibles) in enumerate(entr, 1):
            optimiseur.zero_grad()
            perte = perte_fn(modele(images), cibles)
            perte.backward()
            optimiseur.step()
            perte_totale += perte.item()
            print(f"\r  Époque {epoque}/{args.epoques}  lot {i}/{len(entr)}", end="", flush=True)
        planificateur.step()

        exact_val = exactitude(modele, val)
        print(f"\r  Époque {epoque}/{args.epoques}  perte {perte_totale / len(entr):.3f}  "
              f"exactitude validation {exact_val:.1%}  ({time.time() - debut:.0f} s)")

        if exact_val >= meilleure:
            meilleure = exact_val
            sauvegarder(modele, classes, {"exactitude_validation": exact_val, "epoque": epoque})

    print(f"\nMeilleure exactitude en validation : {meilleure:.1%}")
    print(f"Modèle sauvegardé : {CHEMIN_MODELE}")
    print("Évaluez-le sur le jeu de test : python -m diagnostic.vision.evaluer")


if __name__ == "__main__":
    main()
