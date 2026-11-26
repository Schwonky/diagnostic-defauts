import pytest

torch = pytest.importorskip("torch")

from diagnostic.base_connaissances import BaseConnaissances
from diagnostic.vision import MODELE_NEU, ClassifieurDefauts, DetecteurEtat, creer_resnet

CLASSES_NEU = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def test_modele_sort_six_scores():
    modele = creer_resnet(6, preentraine=False).eval()
    with torch.no_grad():
        sortie = modele(torch.randn(2, 3, 224, 224))
    assert sortie.shape == (2, 6)


def test_toutes_les_classes_neu_sont_dans_la_base():
    base = BaseConnaissances()
    assert {d["classe_neu"] for d in base.defauts.values()} == set(CLASSES_NEU)


@pytest.mark.skipif(not MODELE_NEU.exists(), reason="modèle pas encore entraîné")
def test_classifieur_retourne_des_probabilites():
    from PIL import Image

    resultats = ClassifieurDefauts().predire(Image.new("L", (200, 200), color=128))
    assert len(resultats) == 6
    assert abs(sum(r["probabilite"] for r in resultats) - 1) < 0.01


def test_detecteur_utilise_la_taille_du_modele(tmp_path):
    from PIL import Image

    chemin = tmp_path / "etat.pt"
    torch.save({"etat": creer_resnet(2, preentraine=False).state_dict(), "classes": ["bon_etat", "endommage"],
                "taille": 96}, chemin)
    detecteur = DetecteurEtat(chemin)
    assert detecteur.transformation(Image.new("RGB", (300, 200))).shape == (3, 96, 96)
    assert 0 <= detecteur.proba_endommage(Image.new("RGB", (300, 200))) <= 1
