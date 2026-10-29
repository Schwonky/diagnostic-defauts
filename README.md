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
  corpus_symptomes_dev.json Phrases annotées pour développer le module symptômes
  corpus_symptomes_test.json Phrases annotées pour l'évaluation finale (à compléter)
  images/                   Images NEU-CLS (non versionnées)
diagnostic/
  base_connaissances.py     Chargement et validation de la base
  vision/                   Module de vision : données, modèle, entraînement, évaluation
  symptomes/                Module textuel : extracteur et évaluation
  fusion/                   Moteur de fusion (étape 5)
api/                        API FastAPI (étape 6)
tests/                      Tests unitaires
```

## Module symptômes

```bash
python -m diagnostic.symptomes "la machine vibre et sent le brûlé"
python -m diagnostic.symptomes.evaluer                                  # corpus de développement
python -m diagnostic.symptomes.evaluer data/corpus_symptomes_test.json  # corpus de test
```

## Module vision

```bash
python -m diagnostic.vision.entrainer          # entraîne ResNet18 (modèle dans modeles/)
python -m diagnostic.vision.evaluer            # évalue sur le jeu de test (résultats dans resultats/)
python -m diagnostic.vision chemin/image.jpg   # prédit le défaut d'une image
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pytest
```
