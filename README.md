# Diagnostic de défauts sur des pièces industrielles

Projet synthèse INF4173, session automne 2026.

Application d'aide au diagnostic qui combine :
- un module de **vision** (détection du type de défaut sur une image, jeu NEU-CLS) ;
- un module d'**analyse des symptômes** (texte en français) ;
- un module de **fusion** qui classe les composants suspects et propose des vérifications.

## Structure

```
data/
  base_connaissances.json   Symptômes, défauts, composants, causes, vérifications
  images/                   Images NEU-CLS (non versionnées)
diagnostic/
  base_connaissances.py     Chargement et validation de la base
  vision/                   Module de vision (étape 4)
  symptomes/                Module textuel (étape 3)
  fusion/                   Moteur de fusion (étape 5)
api/                        API FastAPI (étape 6)
tests/                      Tests unitaires
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest
```
