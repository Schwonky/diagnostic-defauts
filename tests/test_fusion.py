from diagnostic.base_connaissances import BaseConnaissances
from diagnostic.fusion import MoteurFusion

base = BaseConnaissances()
moteur = MoteurFusion(base)
UNIFORME = {d: 0.01 for d in base.defauts}


def probas(defaut, p=0.95):
    return {**UNIFORME, defaut: p}


def ids(resultat):
    return [r["id"] for r in resultat["classement"]]


def test_fuite_pointe_vers_joint():
    assert ids(moteur.diagnostiquer(symptomes=["fuite"]))[0] == "joint"


def test_classement_limite_a_trois():
    r = moteur.diagnostiquer(symptomes=["vibration", "surchauffe", "bruit_cyclique"])
    assert len(r["classement"]) == 3
    scores = [c["score"] for c in r["classement"]]
    assert scores == sorted(scores, reverse=True)


def test_image_departage_les_symptomes_ambigus():
    # « vibration » seule : roulement et arbre à égalité; des microfissures favorisent l'arbre
    r = moteur.diagnostiquer(symptomes=["vibration"], probas_defauts=probas("microfissures"))
    assert ids(r)[0] == "arbre"


def test_image_seule_avec_composant_indique():
    r = moteur.diagnostiquer(symptomes=[], probas_defauts=probas("piqures"), p_endommage=0.9,
                             composant_photo="roulement")
    assert ids(r) == ["roulement"]


def test_image_peu_fiable_reduit_son_poids():
    faible = {d: 1 / 6 for d in base.defauts}
    r = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=faible)
    assert r["poids"]["image"] == 0.15
    assert any("peu fiable" in a for a in r["avertissements"])


def test_comparaison_des_configurations():
    texte = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=probas("rayures"), utiliser_image=False)
    image = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=probas("rayures"), utiliser_texte=False)
    assert texte["poids"] == {"symptomes": 1.0, "image": 0.0}
    assert image["poids"] == {"symptomes": 0.0, "image": 1.0}


def test_type_de_dommage_regroupe():
    assert MoteurFusion.type_dommage(probas("piqures"))["id"] == "corrosion"
    assert MoteurFusion.type_dommage(probas("rayures"))["id"] == "rayure_usure"


def test_causes_et_verifications_fournies():
    r = moteur.diagnostiquer(symptomes=["grincement", "surchauffe"])
    premier = r["classement"][0]
    assert premier["causes"] and premier["verifications"]


def test_piece_saine_n_accuse_aucun_composant():
    r = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=probas("microfissures"), p_endommage=0.1,
                             composant_photo="arbre")
    assert r["image"]["etat"] == "bon_etat"
    assert r["poids"] == {"symptomes": 1.0, "image": 0.0}
    assert r["image"]["defaut_principal"] is None
    assert all(c["score_image"] == 0 for c in r["classement"])


def test_etat_incertain_reduit_le_poids_de_l_image():
    r = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=probas("rayures"), p_endommage=0.5)
    assert r["image"]["etat"] == "incertain"
    assert r["poids"]["image"] == 0.15


def test_piece_endommagee_garde_le_poids_normal():
    r = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=probas("rayures"), p_endommage=0.9)
    assert r["image"]["etat"] == "endommage"
    assert r["poids"]["image"] == 0.4


def test_image_desactivee_n_influence_pas_les_causes():
    texte_seul = moteur.diagnostiquer(symptomes=["vibration"])
    desactivee = moteur.diagnostiquer(symptomes=["vibration"], probas_defauts=probas("microfissures"),
                                      p_endommage=0.9, utiliser_image=False)
    assert desactivee["classement"] == texte_seul["classement"]
    assert desactivee["avertissements"] == texte_seul["avertissements"]


def test_desaccord_entre_symptomes_et_image_signale():
    r = moteur.diagnostiquer(symptomes=["fuite"], probas_defauts=probas("microfissures"), p_endommage=0.95)
    sym = moteur.scores_symptomes(["fuite"])
    img = moteur.scores_image(probas("microfissures"), 0.95)
    desaccord = max(sym, key=sym.get) != max(img, key=img.get)
    assert any("Les symptômes désignent" in a for a in r["avertissements"]) == desaccord
