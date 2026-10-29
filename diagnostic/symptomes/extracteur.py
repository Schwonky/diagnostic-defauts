"""Extraction des symptômes à partir d'une description libre en français.

Méthode :
1. Normalisation du texte (minuscules, apostrophes, ponctuation).
2. Découpage en propositions (virgules, points, « mais », « et »…) pour limiter la portée des négations.
3. Racinisation (stemming) de chaque mot avec l'algorithme Snowball français.
4. Recherche de chaque synonyme de la base de connaissances : correspondance exacte de la racine,
   par préfixe (« grincent » → « grinc ») ou approximative (fautes de frappe) avec RapidFuzz.
5. Détection des négations (« ne vibre pas », « aucune fuite », « sans bruit »).
"""

import re
import unicodedata

import snowballstemmer
from rapidfuzz import fuzz

from diagnostic.base_connaissances import BaseConnaissances

MOTS_VIDES = {
    "le", "la", "les", "l", "de", "du", "des", "d", "un", "une", "au", "aux", "a",
    "en", "y", "il", "elle", "ils", "elles", "on", "ca", "c", "ce", "cet", "cette",
    "est", "sont", "sur", "dans", "que", "qui", "se", "s", "sa", "son", "ses", "j", "je",
    "me", "m", "t", "tres", "vraiment", "beaucoup", "un peu", "peu",
}
NEGATIONS_AVANT = {"pas", "aucun", "aucune", "sans", "jamais", "ni"}  # « aucune fuite », « sans vibration »
NEGATIONS_APRES = {"pas", "jamais", "rien"}                          # « ne vibre pas »
SEPARATEURS = r"[,.;:!?\n]|\bmais\b|\bet\b|\bpuis\b|\bcependant\b|\bpar contre\b"

SEUIL_FLOU = 85          # score RapidFuzz minimal pour une faute de frappe
LONGUEUR_MIN_PREFIXE = 5  # une racine doit faire au moins 5 lettres pour la correspondance par préfixe
ECART_MAX = 1             # nombre de mots tolérés entre deux mots d'un synonyme

_stemmer = snowballstemmer.stemmer("french")


def _sans_accents(texte):
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def _decouper(texte):
    """Découpe en mots (minuscules, avec accents)."""
    texte = texte.lower().replace("’", "'").replace("'", " ")
    mots = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœ]+", texte)
    return ["ne" if m == "n" else m for m in mots]


def _mots(texte):
    """Retourne la racine de chaque mot (racinisation avec accents, puis accents retirés)."""
    return [_sans_accents(_stemmer.stemWord(m)) for m in _decouper(texte)]


def _mot_correspond(racine_texte, racine_synonyme):
    """Retourne un score de confiance (0 à 1) si les deux racines correspondent, sinon 0."""
    if racine_texte == racine_synonyme:
        return 1.0
    if len(racine_synonyme) >= LONGUEUR_MIN_PREFIXE and racine_texte.startswith(racine_synonyme):
        return 0.95
    if min(len(racine_texte), len(racine_synonyme)) >= 4:
        score = fuzz.ratio(racine_texte, racine_synonyme)
        if score >= SEUIL_FLOU:
            return score / 100
    return 0.0


def _contenu(racines):
    """Garde les mots porteurs de sens avec leur position d'origine."""
    return [(i, r) for i, r in enumerate(racines) if r not in MOTS_VIDES]


def _est_negation(mots, debut, fin):
    """Vrai si un mot de négation se trouve près de l'expression (« pas mal » n'est pas une négation)."""
    def negation(i, ensemble):
        pas_mal = mots[i] == "pas" and i + 1 < len(mots) and mots[i + 1] == "mal"
        return mots[i] in ensemble and not pas_mal

    avant = any(negation(i, NEGATIONS_AVANT) for i in range(max(0, debut - 3), debut))
    apres = any(negation(i, NEGATIONS_APRES) for i in range(fin + 1, min(len(mots), fin + 3)))
    return avant or apres


class ExtracteurSymptomes:
    def __init__(self, base=None):
        self.base = base or BaseConnaissances()
        # Pré-calcul des racines de chaque synonyme
        self.patrons = []
        for id_symptome, infos in self.base.symptomes.items():
            formes = set(infos["synonymes"]) | {infos["libelle"]}
            for forme in formes:
                racines = [r for r in _mots(forme) if r not in MOTS_VIDES]
                if racines:
                    self.patrons.append((id_symptome, forme, racines))

    def _chercher(self, contenu, racines_patron):
        """Cherche un patron dans la liste de mots. Retourne (score, début, fin) ou None."""
        meilleur = None
        for depart in range(len(contenu)):
            score = _mot_correspond(contenu[depart][1], racines_patron[0])
            if not score:
                continue
            scores, pos = [score], depart
            for racine in racines_patron[1:]:
                trouve = False
                for suivant in range(pos + 1, min(pos + 2 + ECART_MAX, len(contenu))):
                    s = _mot_correspond(contenu[suivant][1], racine)
                    if s:
                        scores.append(s)
                        pos = suivant
                        trouve = True
                        break
                if not trouve:
                    break
            if len(scores) == len(racines_patron):
                resultat = (min(scores), contenu[depart][0], contenu[pos][0])
                if meilleur is None or resultat[0] > meilleur[0]:
                    meilleur = resultat
        return meilleur

    def extraire(self, texte):
        """Retourne {'presents': [...], 'absents': [...]} avec les symptômes détectés.

        Chaque symptôme est un dict : id, libelle, confiance, expression (synonyme reconnu).
        """
        presents, absents = {}, {}
        for proposition in re.split(SEPARATEURS, texte.lower()):
            mots = [_sans_accents(m) for m in _decouper(proposition)]
            racines = _mots(proposition)
            contenu = _contenu(racines)
            if not contenu:
                continue
            for id_symptome, forme, racines_patron in self.patrons:
                trouve = self._chercher(contenu, racines_patron)
                if not trouve:
                    continue
                confiance, debut, fin = trouve
                cible = absents if _est_negation(mots, debut, fin) else presents
                if id_symptome not in cible or confiance > cible[id_symptome]["confiance"]:
                    cible[id_symptome] = {
                        "id": id_symptome,
                        "libelle": self.base.symptomes[id_symptome]["libelle"],
                        "confiance": round(confiance, 2),
                        "expression": forme,
                    }
        # Un symptôme affirmé ailleurs dans le texte l'emporte sur une négation
        for id_symptome in presents:
            absents.pop(id_symptome, None)
        trier = lambda d: sorted(d.values(), key=lambda s: -s["confiance"])
        return {"presents": trier(presents), "absents": trier(absents)}
