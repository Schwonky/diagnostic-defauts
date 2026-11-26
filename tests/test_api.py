from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from diagnostic.vision import MODELE_ETAT, MODELE_NEU

client = TestClient(app)


def test_composants():
    composants = client.get("/composants").json()
    assert len(composants) == 6 and "roulement" in composants


def test_diagnostic_texte_seul():
    r = client.post("/diagnostic", data={"description": "Il y a une flaque d'huile sous le réducteur."})
    assert r.status_code == 200
    assert r.json()["classement"][0]["id"] == "joint"
    assert r.json()["symptomes"] == ["Fuite"]


def test_diagnostic_vide_refuse():
    assert client.post("/diagnostic", data={"description": "  "}).status_code == 400


def test_composant_inconnu_refuse():
    r = client.post("/diagnostic", data={"description": "ça fuit", "composant_photo": "volant"})
    assert r.status_code == 400


def test_fichier_qui_n_est_pas_une_image():
    r = client.post("/diagnostic", data={"description": "ça fuit"},
                    files={"photo": ("note.txt", b"pas une image", "text/plain")})
    assert r.status_code == 400


@pytest.mark.skipif(not (MODELE_ETAT.exists() and MODELE_NEU.exists()), reason="modèles pas encore entraînés")
def test_diagnostic_avec_photo():
    tampon = BytesIO()
    Image.new("RGB", (300, 200), "gray").save(tampon, format="JPEG")
    r = client.post("/diagnostic", data={"description": "ça grince", "composant_photo": "roulement"},
                    files={"photo": ("piece.jpg", tampon.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["image"]["etat"] in ("bon_etat", "incertain", "endommage")


def test_page_web():
    r = client.get("/")
    assert r.status_code == 200 and "Diagnostiquer" in r.text
