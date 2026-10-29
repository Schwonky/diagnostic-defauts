import pytest

from diagnostic.symptomes import ExtracteurSymptomes


@pytest.fixture(scope="module")
def extracteur():
    return ExtracteurSymptomes()


def ids(resultat, cle="presents"):
    return {s["id"] for s in resultat[cle]}


def test_symptome_simple(extracteur):
    assert ids(extracteur.extraire("La pompe vibre énormément.")) == {"vibration"}


def test_plusieurs_symptomes(extracteur):
    r = extracteur.extraire("La courroie siffle, patine et sent le brûlé.")
    assert ids(r) == {"sifflement", "perte_puissance", "odeur_brule"}


def test_conjugaison(extracteur):
    assert "grincement" in ids(extracteur.extraire("Les roulements grincent."))


def test_faute_de_frappe(extracteur):
    assert "vibration" in ids(extracteur.extraire("La machine vibrre."))


def test_negation_apres(extracteur):
    r = extracteur.extraire("Ça ne vibre pas mais ça grince.")
    assert ids(r) == {"grincement"}
    assert ids(r, "absents") == {"vibration"}


def test_negation_avant(extracteur):
    r = extracteur.extraire("Aucune fuite, par contre le moteur chauffe.")
    assert ids(r) == {"surchauffe"}
    assert ids(r, "absents") == {"fuite"}


def test_sans_apres_le_symptome_n_est_pas_une_negation(extracteur):
    assert ids(extracteur.extraire("Le convoyeur ralentit sans raison.")) == {"perte_puissance"}


def test_pas_mal_n_est_pas_une_negation(extracteur):
    assert ids(extracteur.extraire("Ça brasse pas mal.")) == {"vibration"}


def test_aucun_symptome(extracteur):
    assert ids(extracteur.extraire("Le technicien a changé le filtre.")) == set()


def test_critere_de_reussite_sur_corpus_dev():
    from diagnostic.symptomes.evaluer import RACINE, evaluer
    resultats = evaluer(RACINE / "data" / "corpus_symptomes_dev.json", afficher_erreurs=False)
    assert resultats["rappel"] >= 0.85
