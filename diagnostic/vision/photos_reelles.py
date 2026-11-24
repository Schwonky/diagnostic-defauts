"""Test du modèle de vision sur des photos d'objets réels (hors du jeu NEU).

Organisation des photos :
    data/photos_reelles/<défaut attendu>/<photo>.jpg

<défaut attendu> est un identifiant de la base de connaissances :
    microfissures, piqures, rayures, plaques, inclusions, calamine
ou « aucun » pour une pièce sans défaut, ou « inconnu » si on ne sait pas.

Utilisation :
    python -m diagnostic.vision.photos_reelles

Produit aussi la figure resultats/photos_reelles.png (photo, image vue par le modèle, Grad-CAM).
"""

from pathlib import Path

from PIL import Image, ImageOps

from diagnostic.vision.donnees import RACINE

DOSSIER_PHOTOS = RACINE / "data" / "photos_reelles"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preparer_photo(chemin, recadrage=0.8):
    """Ouvre une photo et la rapproche du format NEU : orientation corrigée, niveaux de gris,
    carré central (recadrage = fraction du plus petit côté conservée), contraste normalisé."""
    image = ImageOps.exif_transpose(Image.open(chemin)).convert("L")
    cote = int(min(image.size) * recadrage)
    gauche = (image.width - cote) // 2
    haut = (image.height - cote) // 2
    image = image.crop((gauche, haut, gauche + cote, haut + cote))
    image = ImageOps.autocontrast(image, cutoff=1)
    return image.resize((200, 200), Image.Resampling.LANCZOS)


def lister_photos(dossier=DOSSIER_PHOTOS):
    """Retourne [(chemin, défaut attendu)] pour toutes les photos du dossier."""
    if not dossier.is_dir():
        return []
    return [
        (p, p.parent.name if p.parent != dossier else "inconnu")
        for p in sorted(dossier.rglob("*"))
        if p.suffix.lower() in EXTENSIONS
    ]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from diagnostic.vision.classifieur import ClassifieurDefauts
    from diagnostic.vision.donnees import transformation_evaluation
    from diagnostic.vision.gradcam import grad_cam

    photos = lister_photos()
    if not photos:
        raise SystemExit(f"Aucune photo trouvée dans {DOSSIER_PHOTOS}")

    classifieur = ClassifieurDefauts()
    defauts = classifieur.base.defauts
    corrects = evaluees = 0
    fig, axes = plt.subplots(3, len(photos), figsize=(3.2 * len(photos), 10.5), squeeze=False)

    print(f"{'Photo':<30} {'Attendu':<14} {'Prédit (confiance)':<34} 2e choix")
    for j, (chemin, attendu) in enumerate(photos):
        image = preparer_photo(chemin)
        resultats = classifieur.predire(image)
        premier, second = resultats[0], resultats[1]
        if attendu in defauts:
            evaluees += 1
            corrects += premier["id"] == attendu
        print(f"{chemin.name[:29]:<30} {attendu:<14} "
              f"{premier['libelle'] + ' (' + format(premier['probabilite'], '.0%') + ')':<34} "
              f"{second['libelle']} ({second['probabilite']:.0%})")

        carte, _ = grad_cam(classifieur.modele, transformation_evaluation(image).unsqueeze(0))
        gris = np.array(image.resize((224, 224)))
        nom_attendu = defauts[attendu]["libelle"] if attendu in defauts else attendu
        couleur = "black" if attendu not in defauts else ("green" if premier["id"] == attendu else "red")

        axes[0, j].imshow(ImageOps.exif_transpose(Image.open(chemin)).convert("RGB"))
        axes[0, j].set_title(f"{chemin.stem[:24]}\nAttendu : {nom_attendu}", fontsize=9)
        axes[1, j].imshow(gris, cmap="gray", vmin=0, vmax=255)
        axes[1, j].set_title(f"Prédit : {premier['libelle']} ({premier['probabilite']:.0%})\n"
                             f"2e : {second['libelle']} ({second['probabilite']:.0%})", fontsize=9, color=couleur)
        axes[2, j].imshow(gris, cmap="gray", vmin=0, vmax=255)
        axes[2, j].imshow(carte, cmap="jet", alpha=0.45)
        for ax in axes[:, j]:
            ax.axis("off")

    for i, titre in enumerate(["Photo d'origine", "Image vue par le modèle", "Zones utilisées (Grad-CAM)"]):
        axes[i, 0].text(-0.08, 0.5, titre, transform=axes[i, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=10)
    fig.suptitle("Test du modèle sur des photos d'objets réels", fontsize=13)
    plt.tight_layout()
    sortie = RACINE / "resultats" / "photos_reelles.png"
    sortie.parent.mkdir(exist_ok=True)
    fig.savefig(sortie, dpi=130, bbox_inches="tight")

    if evaluees:
        print(f"\nPhotos de défauts correctement classées : {corrects}/{evaluees} ({corrects / evaluees:.0%})")
    print(f"\nFigure enregistrée : {sortie}")


if __name__ == "__main__":
    main()
