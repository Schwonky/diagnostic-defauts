"""API de diagnostic.

Lancement (dans Git Bash, environnement .venv activé) :
    python -m uvicorn api.main:app --reload
Page web : http://127.0.0.1:8000        Documentation interactive : http://127.0.0.1:8000/docs
"""

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from diagnostic.base_connaissances import BaseConnaissances
from diagnostic.fusion import MoteurFusion
from diagnostic.symptomes import ExtracteurSymptomes

app = FastAPI(title="Diagnostic de défauts sur des pièces industrielles")
base = BaseConnaissances()
moteur = MoteurFusion(base, ExtracteurSymptomes(base))
_analyseur = None


def analyseur():
    """Charge les deux modèles d'image au premier usage seulement (quelques secondes)."""
    global _analyseur
    if _analyseur is None:
        from diagnostic.vision import AnalyseurImage
        _analyseur = AnalyseurImage(base)
    return _analyseur


@app.get("/", include_in_schema=False)
def page():
    """La page web de l'application."""
    return FileResponse(Path(__file__).parent / "page.html")


@app.get("/composants")
def composants():
    """Composants que l'utilisateur peut indiquer pour la photo."""
    return {id_c: c["libelle"] for id_c, c in base.composants.items()}


@app.post("/diagnostic")
def diagnostic(description: str = Form(""), composant_photo: str = Form(""), photo: UploadFile | None = File(None)):
    """Description et/ou photo → symptômes, état de la pièce et 3 composants les plus suspects."""
    description = description.strip()
    composant_photo = composant_photo.strip() or None
    if composant_photo and composant_photo not in base.composants:
        raise HTTPException(400, f"Composant inconnu : {composant_photo}")

    analyse = {}
    if photo and photo.filename:
        try:
            image = Image.open(BytesIO(photo.file.read()))
            image.load()
        except OSError:
            raise HTTPException(400, "Le fichier n'est pas une image lisible (JPG ou PNG).")
        analyse = analyseur().analyser(image)  # {"p_endommage": …, "probas_defauts": …}

    if not description and not analyse:
        raise HTTPException(400, "Fournissez une description, une photo, ou les deux.")

    resultat = moteur.diagnostiquer(texte=description or None, composant_photo=composant_photo, **analyse)
    for cle in ("symptomes", "symptomes_nies"):  # identifiants → libellés lisibles
        resultat[cle] = [base.symptomes[s]["libelle"] for s in resultat[cle]]
    return resultat
