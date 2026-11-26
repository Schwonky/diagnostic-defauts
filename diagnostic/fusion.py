"""Fusion : combine les symptômes (texte) et l'analyse de l'image pour classer les composants suspects.

Principe (volontairement simple et explicable) :

1. Score « symptômes » d'un composant : somme des poids de la base de connaissances
   pour chaque symptôme détecté, puis division par le meilleur score (le premier vaut 1).
2. Score « image » d'un composant :
   - si l'utilisateur indique le composant photographié, ce composant reçoit la probabilité
     que la pièce soit endommagée (ou 1 si cette probabilité n'est pas connue);
   - sinon, chaque défaut visuel probable (modèle NEU) vote pour les composants selon les poids
     de la base de connaissances; le résultat est aussi ramené entre 0 et 1.
3. Score final : moyenne pondérée des deux scores.
   - Les deux sources : 60 % symptômes, 40 % image.
   - Image peu fiable (état incertain, ou confiance du modèle NEU sous le seuil) : 85 % symptômes, 15 % image.
   - Une seule source disponible : 100 % pour celle-ci.

État de la pièce (modèle v2) : « bon état » si p_endommage < SEUIL_BON_ETAT, « endommagé » si
p_endommage ≥ SEUIL_ENDOMMAGE, « incertain » entre les deux. Une pièce jugée en bon état n'accuse aucun
composant, et le type de défaut n'est utilisé que si la pièce n'est pas jugée en bon état.
Les deux seuils sont choisis sur le jeu de validation (notebooks/calibration_seuil.ipynb).
"""

POIDS_SYMPTOMES = 0.6
POIDS_IMAGE = 0.4
POIDS_IMAGE_PEU_FIABLE = 0.15
SEUIL_CONFIANCE_IMAGE = 0.6
SEUIL_BON_ETAT = 0.4
SEUIL_ENDOMMAGE = 0.6

# Regroupement des 6 classes NEU en types de dommages plus parlants pour l'utilisateur
TYPES_DOMMAGE = {
    "corrosion": {"libelle": "Corrosion", "defauts": ["piqures", "plaques"]},
    "rayure_usure": {"libelle": "Rayure / usure", "defauts": ["rayures"]},
    "fissure": {"libelle": "Fissure", "defauts": ["microfissures"]},
    "anomalie_fabrication": {"libelle": "Anomalie de fabrication", "defauts": ["inclusions", "calamine"]},
}


def etat_piece(p_endommage):
    """« bon_etat », « incertain » ou « endommage » selon les seuils; None si la probabilité est inconnue."""
    if p_endommage is None:
        return None
    if p_endommage < SEUIL_BON_ETAT:
        return "bon_etat"
    return "endommage" if p_endommage >= SEUIL_ENDOMMAGE else "incertain"


def _normaliser(scores):
    """Divise chaque score par le plus grand, pour que le premier vaille 1."""
    maximum = max(scores.values(), default=0)
    if maximum <= 0:
        return {}
    return {c: s / maximum for c, s in scores.items() if s > 0}


class MoteurFusion:
    def __init__(self, base, extracteur=None):
        """base : BaseConnaissances. extracteur : ExtracteurSymptomes (nécessaire seulement si on passe du texte)."""
        self.base = base
        self.extracteur = extracteur

    # ---------- Sources ----------

    def scores_symptomes(self, symptomes):
        brut = {}
        for s in symptomes:
            for composant, poids in self.base.relations_symptomes.get(s, {}).items():
                brut[composant] = brut.get(composant, 0) + poids
        return _normaliser(brut)

    def scores_image(self, probas_defauts=None, p_endommage=None, composant_photo=None):
        if etat_piece(p_endommage) == "bon_etat":
            return {}  # une pièce jugée saine n'accuse aucun composant
        if composant_photo:
            return {composant_photo: p_endommage if p_endommage is not None else 1.0}
        if not probas_defauts:
            return {}
        brut = {}
        for defaut, p in probas_defauts.items():
            for composant, poids in self.base.relations_defauts.get(defaut, {}).items():
                brut[composant] = brut.get(composant, 0) + p * poids
        return _normaliser(brut)

    @staticmethod
    def type_dommage(probas_defauts):
        """Retourne (identifiant, libellé, probabilité) du type de dommage le plus probable."""
        if not probas_defauts:
            return None
        totaux = {t: sum(probas_defauts.get(d, 0) for d in info["defauts"]) for t, info in TYPES_DOMMAGE.items()}
        meilleur = max(totaux, key=totaux.get)
        return {"id": meilleur, "libelle": TYPES_DOMMAGE[meilleur]["libelle"], "probabilite": totaux[meilleur]}

    # ---------- Diagnostic ----------

    def diagnostiquer(self, texte=None, symptomes=None, probas_defauts=None, p_endommage=None,
                      composant_photo=None, utiliser_texte=True, utiliser_image=True, top=3):
        """Retourne un dictionnaire avec les symptômes, l'analyse de l'image et le classement des composants.

        texte : description libre (analysée par l'extracteur), ou
        symptomes : liste d'identifiants de symptômes déjà connus.
        probas_defauts : {identifiant de défaut : probabilité} venant du modèle NEU.
        p_endommage : probabilité que la pièce soit endommagée (modèle version 2).
        composant_photo : composant indiqué par l'utilisateur pour la photo (facultatif).
        utiliser_texte / utiliser_image : pour comparer « texte seul », « image seule » et « les deux ».
        """
        avertissements = []
        # Une source désactivée ne doit rien influencer (scores, causes, avertissements)
        if not utiliser_texte:
            texte = symptomes = None
        if not utiliser_image:
            probas_defauts = p_endommage = composant_photo = None

        # 1. Symptômes
        symptomes_detectes, symptomes_nies = [], []
        if symptomes is not None:
            symptomes_detectes = list(symptomes)
        elif texte:
            if self.extracteur is None:
                raise ValueError("Un extracteur de symptômes est nécessaire pour analyser du texte.")
            resultat = self.extracteur.extraire(texte)
            symptomes_detectes = [s["id"] for s in resultat["presents"]]
            symptomes_nies = [s["id"] for s in resultat["absents"]]
        s_sym = self.scores_symptomes(symptomes_detectes)
        if (texte or symptomes is not None) and not symptomes_detectes:
            avertissements.append("Aucun symptôme reconnu dans la description.")

        # 2. Image
        etat = etat_piece(p_endommage)
        if etat == "bon_etat":
            probas_defauts = None  # le type de défaut n'a pas de sens pour une pièce jugée saine
            avertissements.append(f"Le modèle d'image juge la pièce en bon état ({1 - p_endommage:.0%}) : "
                                  "l'image n'accuse aucun composant.")
        elif etat == "incertain":
            avertissements.append(f"État de la pièce incertain ({p_endommage:.0%} endommagé) : l'image compte peu.")
        confiance = max(probas_defauts.values()) if probas_defauts else None
        image_fiable = etat in (None, "endommage") and (
            composant_photo is not None or (confiance is not None and confiance >= SEUIL_CONFIANCE_IMAGE))
        s_img = self.scores_image(probas_defauts, p_endommage, composant_photo)
        if probas_defauts and confiance < SEUIL_CONFIANCE_IMAGE:
            avertissements.append(f"Type de défaut peu fiable (confiance {confiance:.0%}) : "
                                  "la décision repose surtout sur les symptômes.")

        # Désaccord : les deux sources désignent des composants différents
        if s_sym and s_img and not composant_photo:
            top_sym, top_img = max(s_sym, key=s_sym.get), max(s_img, key=s_img.get)
            if top_sym != top_img:
                libelle = lambda c: self.base.composants[c]["libelle"]
                avertissements.append(f"Les symptômes désignent surtout « {libelle(top_sym)} », la photo surtout "
                                      f"« {libelle(top_img)} ». Vérifiez que la photo montre la pièce concernée, "
                                      "ou indiquez le composant photographié.")

        # 3. Pondération
        if s_sym and s_img:
            w_img = POIDS_IMAGE if image_fiable else POIDS_IMAGE_PEU_FIABLE
            w_sym = 1 - w_img
        elif s_sym:
            w_sym, w_img = 1.0, 0.0
        elif s_img:
            w_sym, w_img = 0.0, 1.0
        else:
            w_sym = w_img = 0.0

        composants = set(s_sym) | set(s_img)
        classement = []
        for c in composants:
            score = w_sym * s_sym.get(c, 0) + w_img * s_img.get(c, 0)
            classement.append({"id": c, "libelle": self.base.composants[c]["libelle"], "score": round(score, 3),
                               "score_symptomes": round(s_sym.get(c, 0), 3), "score_image": round(s_img.get(c, 0), 3)})
        # Tri par score, puis par ordre alphabétique pour un résultat stable en cas d'égalité
        classement.sort(key=lambda r: (-r["score"], r["id"]))
        classement = classement[:top]

        # 4. Causes et vérifications pour chaque composant retenu
        defaut_principal = max(probas_defauts, key=probas_defauts.get) if probas_defauts else None
        for r in classement:
            causes = self.base.causes(r["id"], symptomes=symptomes_detectes,
                                      defauts=[defaut_principal] if defaut_principal else [])
            r["causes"] = [c["libelle"] for c in causes if c["correspondances"] > 0][:2] or [causes[0]["libelle"]]
            r["verifications"] = [v["action"] for v in self.base.verifications(r["id"])][:3]

        return {
            "symptomes": symptomes_detectes,
            "symptomes_nies": symptomes_nies,
            "image": {
                "p_endommage": p_endommage,
                "etat": etat,
                "defaut_principal": defaut_principal,
                "type_dommage": self.type_dommage(probas_defauts),
                "confiance": confiance,
                "fiable": image_fiable if (probas_defauts or composant_photo) else None,
                "composant_photo": composant_photo,
            },
            "poids": {"symptomes": w_sym, "image": w_img},
            "classement": classement,
            "avertissements": avertissements,
        }
