from diagnostic.base_connaissances import BaseConnaissances


def test_chargement_et_validation():
    kb = BaseConnaissances()
    assert len(kb.composants) == 6
    assert len(kb.symptomes) == 12
    assert len(kb.defauts) == 6


def test_chaque_symptome_mene_a_un_composant():
    kb = BaseConnaissances()
    for s in kb.symptomes:
        assert kb.composants_pour_symptome(s), f"Aucun composant pour {s}"


def test_chaque_defaut_mene_a_un_composant():
    kb = BaseConnaissances()
    for d in kb.defauts:
        assert kb.composants_pour_defaut(d), f"Aucun composant pour {d}"


def test_fuite_pointe_vers_joint():
    kb = BaseConnaissances()
    liens = kb.composants_pour_symptome("fuite")
    assert max(liens, key=liens.get) == "joint"


def test_causes_triees_par_correspondance():
    kb = BaseConnaissances()
    causes = kb.causes("roulement", symptomes=["grincement", "surchauffe", "blocage"])
    assert causes[0]["id"] == "lubrification_roulement"


def test_verifications_triees():
    kb = BaseConnaissances()
    v = kb.verifications("moteur")
    assert [x["priorite"] for x in v] == sorted(x["priorite"] for x in v)
