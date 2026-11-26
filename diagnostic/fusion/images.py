"""Analyse complète d'une image : état de la pièce (modèle v2) et type de défaut (modèle NEU, version 1)."""

from PIL import Image, ImageOps

from diagnostic.vision.classifieur import ClassifieurDefauts
from diagnostic.vision.etat import DetecteurEtat


def preparer_photo(image, recadrage=0.8):
    """Rapproche une photo du format NEU : niveaux de gris, carré central, contraste normalisé."""
    image = ImageOps.exif_transpose(image).convert("L")
    cote = int(min(image.size) * recadrage)
    gauche, haut = (image.width - cote) // 2, (image.height - cote) // 2
    image = image.crop((gauche, haut, gauche + cote, haut + cote))
    return ImageOps.autocontrast(image, cutoff=1).resize((200, 200), Image.Resampling.LANCZOS)


class AnalyseurImage:
    def __init__(self, base=None):
        self.classifieur = ClassifieurDefauts(base=base)
        self.detecteur = DetecteurEtat()

    def analyser(self, image, photo_reelle=True):
        """image : chemin, objet fichier ou PIL.Image.

        photo_reelle=True : la photo est d'abord rapprochée du format NEU pour le modèle de défauts.
        Retourne {"p_endommage": float, "probas_defauts": {identifiant: probabilité}}.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image.load()
        p_endommage = self.detecteur.proba_endommage(image)
        pour_neu = preparer_photo(image) if photo_reelle else image
        probas = {d["id"]: d["probabilite"] for d in self.classifieur.predire(pour_neu)}
        return {"p_endommage": p_endommage, "probas_defauts": probas}
