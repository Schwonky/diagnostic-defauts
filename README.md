# Diagnostic de défauts sur pièces industrielles et automobiles

Projet synthèse INF4173, Jonathan Billings.

Application d'aide au diagnostic qui analyse une photo de pièce et une
description des symptômes pour détecter les défauts, classer les composants
suspects et recommander les étapes à suivre.

## Structure

- `backend/app/` : API FastAPI (vision, symptômes, fusion, base de données)
- `knowledge_base/` : base de connaissances (symptômes, causes, actions)
- `notebooks/` : expériences d'entraînement et d'évaluation
- `frontend/` : interface utilisateur
- `docs/` : cadrage et rapports
