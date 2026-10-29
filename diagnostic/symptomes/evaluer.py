"""Évaluation du module symptômes sur un corpus annoté.

Utilisation :
    python -m diagnostic.symptomes.evaluer                         (corpus de développement)
    python -m diagnostic.symptomes.evaluer data/corpus_symptomes_test.json
"""

import json
import sys
from pathlib import Path

from diagnostic.symptomes import ExtracteurSymptomes

RACINE = Path(__file__).resolve().parents[2]


def evaluer(chemin_corpus, afficher_erreurs=True):
    with open(chemin_corpus, encoding="utf-8") as f:
        phrases = json.load(f)["phrases"]

    extracteur = ExtracteurSymptomes()
    vp = fp = fn = phrases_parfaites = 0
    erreurs = []

    for p in phrases:
        attendus = set(p["symptomes"])
        predits = {s["id"] for s in extracteur.extraire(p["texte"])["presents"]}
        vp += len(attendus & predits)
        fp += len(predits - attendus)
        fn += len(attendus - predits)
        if attendus == predits:
            phrases_parfaites += 1
        else:
            erreurs.append((p["texte"], sorted(attendus - predits), sorted(predits - attendus)))

    rappel = vp / (vp + fn) if vp + fn else 1.0
    precision = vp / (vp + fp) if vp + fp else 1.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0

    print(f"Corpus : {chemin_corpus}  ({len(phrases)} phrases)")
    print(f"  Rappel (symptômes correctement extraits) : {rappel:.1%}")
    print(f"  Précision                                : {precision:.1%}")
    print(f"  F1                                       : {f1:.1%}")
    print(f"  Phrases entièrement correctes            : {phrases_parfaites}/{len(phrases)}")
    print(f"  Critère visé (≥ 85 % extraits)           : {'ATTEINT' if rappel >= 0.85 else 'NON ATTEINT'}")

    if afficher_erreurs and erreurs:
        print("\nErreurs :")
        for texte, manques, en_trop in erreurs:
            print(f"  « {texte} »")
            if manques:
                print(f"      manqués : {', '.join(manques)}")
            if en_trop:
                print(f"      en trop : {', '.join(en_trop)}")

    return {"rappel": rappel, "precision": precision, "f1": f1}


if __name__ == "__main__":
    chemin = sys.argv[1] if len(sys.argv) > 1 else RACINE / "data" / "corpus_symptomes_dev.json"
    evaluer(chemin)
