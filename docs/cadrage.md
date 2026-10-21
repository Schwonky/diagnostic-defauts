# Cadrage du projet (v2 : industriel + véhicules)

## 1. Domaines retenus

L'application fonctionne sur deux domaines qui partagent le même moteur de diagnostic :

- **Industriel** : système de transmission mécanique (moteur, roulements, arbre, engrenages, courroie, accouplement, joint).
- **Véhicule** : pièces d'usure d'une automobile (pneus, freins, roulements de roue, courroie, amortisseurs).

Les deux domaines partagent les mêmes types de défauts visuels (fissure, usure, corrosion, déformation). Seule la base de connaissances change d'un domaine à l'autre, ce qui permet d'ajouter un domaine sans réentraîner le modèle de vision.

## 2. Parcours utilisateur

1. L'utilisateur choisit le domaine et le type de pièce (ex. : Véhicule → Disque de frein).
2. Il ajoute une photo de la pièce et/ou une description du problème.
3. L'application **détecte** le défaut visible sur la photo.
4. Elle **diagnostique** : classement des composants et des causes probables, avec un niveau de confiance.
5. Elle **explique les étapes à suivre** : vérifications à faire, action recommandée (surveiller, réparer, remplacer) et niveau d'urgence.

Le choix du type de pièce par l'utilisateur évite au modèle de devoir reconnaître la pièce elle-même, ce qui simplifie fortement la tâche de vision.

## 3. Défauts visuels (communs aux deux domaines)

| Code | Défaut | Exemples industriels | Exemples véhicule |
| --- | --- | --- | --- |
| D0 | Aucun défaut | — | — |
| D1 | Fissure | Arbre, engrenage | Craquelures de pneu, disque fissuré |
| D2 | Usure | Roulement, engrenage, courroie | Bande de roulement usée, plaquette usée, disque rayé |
| D3 | Corrosion / piqûres | Roulement, arbre | Disque rouillé |
| D4 | Déformation | Arbre, accouplement | Hernie de pneu, disque voilé |

## 4. Domaine industriel

### Composants

C1 Moteur électrique, C2 Roulement, C3 Arbre, C4 Engrenage, C5 Courroie, C6 Accouplement, C7 Joint d'étanchéité.

### Symptômes → composants

X = lien fort, o = lien possible.

| | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 Grincement | | X | | o | X | | |
| S2 Vibration anormale | o | X | X | o | | X | |
| S3 Surchauffe | X | X | | o | o | | |
| S4 Cognement | | o | | X | | X | |
| S5 Sifflement | | | | | X | | |
| S6 Blocage / grippage | o | X | o | X | | | |
| S7 Perte de puissance | X | | | | X | o | |
| S8 Fuite de lubrifiant | | o | | | | | X |
| S9 Odeur de brûlé | X | o | | | X | | |
| S10 Jeu excessif | | X | | o | | X | |

## 5. Domaine véhicule

### Composants

| Code | Composant | Pièce de sécurité |
| --- | --- | --- |
| V1 | Pneu | Oui |
| V2 | Disque de frein | Oui |
| V3 | Plaquettes de frein | Oui |
| V4 | Roulement de roue | Oui |
| V5 | Courroie d'accessoires | Non |
| V6 | Amortisseur | Oui |

### Symptômes → composants

| | V1 | V2 | V3 | V4 | V5 | V6 |
| --- | --- | --- | --- | --- | --- | --- |
| VS1 Grincement ou couinement au freinage | | o | X | | | |
| VS2 Vibration dans le volant ou la pédale au freinage | | X | | | | |
| VS3 Grondement qui augmente avec la vitesse | o | | | X | | |
| VS4 Crissement au démarrage ou à froid | | | | | X | |
| VS5 Le véhicule tire d'un côté | X | o | o | | | |
| VS6 Rebonds ou tangage excessifs | | | | | | X |
| VS7 Fuite d'huile près de la roue | | | | | | X |
| VS8 Distance de freinage allongée | | o | X | | | |
| VS9 Vibration à haute vitesse | X | | | o | | |
| VS10 Usure irrégulière des pneus | X | | | | | o |

Exemples de formulations pour le corpus : « ça couine quand je freine », « le volant tremble quand je freine », « bruit sourd qui monte avec la vitesse », « ça crie le matin au démarrage », « l'auto part à droite », « la voiture rebondit sur les bosses ».

## 6. Étapes à suivre (format de la base de connaissances)

Chaque combinaison composant + défaut ou symptôme est associée, dans `knowledge_base/`, à des causes, des vérifications et une action. Exemple :

```yaml
- composant: V2  # Disque de frein
  defaut: D4     # Déformation (disque voilé)
  symptomes: [VS2]
  causes:
    - Surchauffe après freinages répétés
    - Serrage inégal des écrous de roue
  verifications:
    - Mesurer le voile du disque au comparateur
    - Mesurer l'épaisseur et la comparer au minimum gravé sur le disque
  action: Remplacer les disques et les plaquettes de l'essieu
  urgence: élevée
  securite: Pièce de sécurité, faire valider par un mécanicien
```

Niveaux d'urgence : **faible** (surveiller), **moyenne** (planifier la réparation), **élevée** (réparer rapidement), **critique** (ne pas utiliser l'équipement ou le véhicule).

Les étapes proviennent de la base de connaissances et non d'un modèle génératif : chaque recommandation est traçable et vérifiable.

## 7. Données

| Domaine | Source | Contenu |
| --- | --- | --- |
| Véhicule | Mendeley, « Digital images of defective and good condition tyres » | ~1 850 photos de pneus, bon / défectueux |
| Véhicule | Mendeley, TyreNet | ~1 700 photos de pneus, défauts comme craquelures et usure |
| Industriel | NEU Surface Defect, MVTec AD | Surfaces et pièces métalliques |
| Les deux | Photos personnelles (garage, recycleur de pièces auto) | Disques, plaquettes, courroies usagés |

Licences à vérifier avant utilisation. Les images publiques de freins automobiles étant rares, la photo des pièces usagées est la meilleure façon de couvrir ce composant.

## 8. Critères d'évaluation

| Module | Mesure | Cible |
| --- | --- | --- |
| Vision | Rappel / précision sur le jeu de test | ≥ 90 % / ≥ 85 % |
| Symptômes | Symptômes correctement extraits | ≥ 85 % |
| Fusion | Bon composant dans le top 3 | ≥ 80 % des scénarios |
| Fusion | Combiné vs image seule vs texte seul | Combiné supérieur |
| Recommandations | Action recommandée correcte sur les scénarios de test | À définir |

## 9. Limites

L'application est un outil d'aide au diagnostic : pour les pièces de sécurité (freins, pneus, suspension), elle recommande toujours une validation par un professionnel. Les données publiques couvrent bien les pneus mais peu les freins, ce qui sera compensé par des photos personnelles et par l'analyse des symptômes.
