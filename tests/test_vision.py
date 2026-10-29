import pytest

torch = pytest.importorskip("torch")

from diagnostic.base_connaissances import BaseConnaissances
from diagnostic.vision.modele import CHEMIN_MODELE, creer_modele

CLASSES_NEU = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def test_modele_sort_six_scores():
    modele = creer_modele(6, preentraine=False).eval()
    with torch.no_grad():
        sortie = modele(torch.randn(2, 3, 224, 224))
    assert sortie.shape == (2, 6)


def test_toutes_les_classes_neu_sont_dans_la_base():
    base = BaseConnaissances()
    assert {d["classe_neu"] for d in base.defauts.values()} == set(CLASSES_NEU)


@pytest.mark.skipif(not CHEMIN_MODELE.exists(), reason="modèle pas encore entraîné")
def test_classifieur_retourne_des_probabilites():
    from PIL import Image
    from diagnostic.vision.classifieur import ClassifieurDefauts

    resultats = ClassifieurDefauts().predire(Image.new("L", (200, 200), color=128))
    assert len(resultats) == 6
    assert abs(sum(r["probabilite"] for r in resultats) - 1) < 0.01
