"""Démonstration : python -m diagnostic.symptomes "la machine vibre et sent le brûlé" """

import sys

from diagnostic.symptomes import ExtracteurSymptomes

texte = " ".join(sys.argv[1:]) or input("Décrivez les symptômes : ")
resultat = ExtracteurSymptomes().extraire(texte)

print("\nSymptômes détectés :")
for s in resultat["presents"] or []:
    print(f"  - {s['libelle']:<32} confiance {s['confiance']:.2f}   (reconnu : « {s['expression']} »)")
if not resultat["presents"]:
    print("  (aucun)")
if resultat["absents"]:
    print("\nSymptômes explicitement absents :")
    for s in resultat["absents"]:
        print(f"  - {s['libelle']}")
