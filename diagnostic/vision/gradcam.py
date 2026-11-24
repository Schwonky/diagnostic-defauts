"""Grad-CAM : zones de l'image qui ont le plus influencé la décision du modèle."""

import torch.nn.functional as F


def grad_cam(modele, tenseur):
    """Retourne (carte entre 0 et 1 de la taille de l'image, indice de la classe prédite)."""
    sauvegarde = {}

    def capturer(module, entree, sortie):
        sauvegarde["activations"] = sortie
        sortie.register_hook(lambda g: sauvegarde.__setitem__("gradients", g))

    crochet = modele.layer4.register_forward_hook(capturer)
    sortie = modele(tenseur)
    crochet.remove()
    classe = sortie.argmax(1).item()
    modele.zero_grad()
    sortie[0, classe].backward()

    poids = sauvegarde["gradients"].mean(dim=(2, 3), keepdim=True)
    carte = F.relu((poids * sauvegarde["activations"]).sum(dim=1, keepdim=True))
    carte = F.interpolate(carte, size=tenseur.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (carte / (carte.max() + 1e-8)).detach().numpy(), classe
