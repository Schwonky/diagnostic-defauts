"""Aide au diagnostic de défauts sur des pièces industrielles.

Modules :
    base_connaissances  symptômes, défauts, composants, causes et vérifications (AMDEC simplifiée)
    symptomes           description en français → symptômes
    vision              photo → état de la pièce et type de défaut
    fusion              symptômes + image → composants suspects
"""

from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
