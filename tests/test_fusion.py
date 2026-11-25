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
