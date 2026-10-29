"""Démonstration : python -m diagnostic.vision chemin/vers/image.jpg"""

import sys

from diagnostic.vision.classifieur import ClassifieurDefauts

if len(sys.argv) < 2:
    sys.exit("Utilisation : python -m diagnostic.vision chemin/vers/image.jpg")

for d in ClassifieurDefauts().predire(sys.argv[1]):
    barre = "█" * round(d["probabilite"] * 30)
    print(f"  {d['libelle']:<28} {d['probabilite']:6.1%}  {barre}")
