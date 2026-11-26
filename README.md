# Diagnostic de défauts sur des pièces industrielles

Projet synthèse INF4173, session automne 2026.

Application d'aide au diagnostic qui combine :
- un module de **vision** : la pièce est-elle endommagée, et quel type de défaut ? (photo);
- un module de **symptômes** : description en français → symptômes reconnus;
- un module de **fusion** : classe les 3 composants les plus suspects, avec causes et vérifications.

## Structure

```
diagnostic/                  Code utilisé par l'application (prédiction seulement)
    base_connaissances.py    Symptômes, défauts, composants, causes et vérifications (AMDEC simplifiée)
    symptomes.py             Description en français → symptômes (+ évaluation sur un corpus)
    vision.py                Photo → état de la pièce (modèle v2) et type de défaut (modèle NEU)
    fusion.py                Symptômes + image → composants suspects
api/                         Application web (FastAPI)
notebooks/                   Données, entraînement et évaluation
    inventaire_donnees.ipynb     Inventaire des jeux de données
    preparation_donnees_v2.ipynb Construction du jeu bon état / endommagé (data/v2/)
    entrainement_v2.ipynb        Modèle bon état / endommagé
    entrainement_neu.ipynb       Modèle type de défaut (NEU-DET)
    symptomes.ipynb              Exemples et évaluation du module symptômes
    fusion.ipynb                 Évaluation de la fusion
data/                        Base de connaissances, corpus, scénarios (les images ne sont pas versionnées)
resultats/                   Figures et métriques produites par les notebooks
tests/                       Tests unitaires (pytest)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Les modèles (`modeles/*.pt`) ne sont pas versionnés : exécutez `entrainement_v2.ipynb` et `entrainement_neu.ipynb`.
