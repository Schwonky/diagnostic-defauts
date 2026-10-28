"""Chargement et validation de la base de connaissances (AMDEC simplifiée)."""

import json
from pathlib import Path

CHEMIN_PAR_DEFAUT = Path(__file__).resolve().parent.parent / "data" / "base_connaissances.json"


class ErreurBaseConnaissances(Exception):
    """Levée quand la base de connaissances est incohérente."""


class BaseConnaissances:
    def __init__(self, chemin=CHEMIN_PAR_DEFAUT):
        with open(chemin, encoding="utf-8") as f:
            self.donnees = json.load(f)
        self.symptomes = self.donnees["symptomes"]
        self.defauts = self.donnees["defauts_visuels"]
        self.composants = self.donnees["composants"]
        self.relations_symptomes = self.donnees["relations_symptomes"]
        self.relations_defauts = self.donnees["relations_defauts"]
        self.valider()

    def valider(self):
        """Vérifie que tous les identifiants utilisés existent et que les poids sont valides."""
        erreurs = []

        for nom_table, relations, cles_valides in [
            ("relations_symptomes", self.relations_symptomes, self.symptomes),
            ("relations_defauts", self.relations_defauts, self.defauts),
        ]:
            for cle, liens in relations.items():
                if cle not in cles_valides:
                    erreurs.append(f"{nom_table} : '{cle}' inconnu")
                for composant, poids in liens.items():
                    if composant not in self.composants:
                        erreurs.append(f"{nom_table}[{cle}] : composant '{composant}' inconnu")
                    if not 0 <= poids <= 1:
                        erreurs.append(f"{nom_table}[{cle}][{composant}] : poids {poids} hors de [0, 1]")

        for id_comp, comp in self.composants.items():
            for cause in comp["causes"]:
                for s in cause["symptomes"]:
                    if s not in self.symptomes:
                        erreurs.append(f"{id_comp}/{cause['id']} : symptôme '{s}' inconnu")
                for d in cause["defauts"]:
                    if d not in self.defauts:
                        erreurs.append(f"{id_comp}/{cause['id']} : défaut '{d}' inconnu")

        if erreurs:
            raise ErreurBaseConnaissances("\n".join(erreurs))

    def composants_pour_symptome(self, symptome):
        """Retourne {composant: poids} pour un symptôme donné."""
        return self.relations_symptomes.get(symptome, {})

    def composants_pour_defaut(self, defaut):
        """Retourne {composant: poids} pour un défaut visuel donné."""
        return self.relations_defauts.get(defaut, {})

    def causes(self, composant, symptomes=(), defauts=()):
        """Retourne les causes d'un composant, triées selon le nombre d'indices observés qui les appuient."""
        observes = set(symptomes) | set(defauts)
        resultat = []
        for cause in self.composants[composant]["causes"]:
            indices = set(cause["symptomes"]) | set(cause["defauts"])
            correspondances = len(indices & observes)
            resultat.append({**cause, "correspondances": correspondances})
        return sorted(resultat, key=lambda c: c["correspondances"], reverse=True)

    def verifications(self, composant):
        """Retourne les vérifications recommandées, par ordre de priorité."""
        return sorted(self.composants[composant]["verifications"], key=lambda v: v["priorite"])
