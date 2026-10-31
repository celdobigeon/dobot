# -*- coding: utf-8 -*-
"""
Augmentation par COPIER-COLLER d'instances.

Avec quelques dizaines d'images, un reseau voit toujours les memes pieces
aux memes endroits. Coller des pieces DECOUPEES (avec leur masque) sur
d'autres images multiplie les configurations a bon compte ; c'est l'une des
augmentations les plus efficaces pour la segmentation d'instance sur de
petits jeux de donnees (Ghiasi et al., "Simple Copy-Paste is a Strong Data
Augmentation Method for Instance Segmentation", CVPR 2021 : un collage
aleatoire, sans meme chercher le realisme, suffit).

On garde neanmoins ce qui ne coute rien et rend les images plausibles :
  - ECHELLE PERSPECTIVE : la camera voit la table en biais ; une piece
    deplacee vers le fond doit rapetisser. Le plan de la table (stocke dans
    l'annotation) donne la distance de la table sous chaque pixel : la
    piece est mise a l'echelle par le rapport des distances (point d'appui
    source / point d'appui cible) ;
  - pieces posees SUR LA TABLE (point d'appui cible sur le plan, a distance
    raisonnable), et souvent EN CONTACT avec une piece existante : c'est le
    cas difficile (deux pieces collees font une seule tache) ;
  - petites rotations dans l'image (+-15 deg) et miroir : au-dela, une piece
    vue en biais tournee dans l'image n'est plus plausible ;
  - luminosite et teinte variees par piece, bord adouci ;
  - OCCULTATION : les masques des pieces recouvertes sont rognes ; une piece
    presque entierement cachee est retiree des etiquettes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import cv2


@dataclass
class Morceau:
    """Une piece decoupee."""
    image: np.ndarray          # h x w x 3
    alpha: np.ndarray          # h x w, float 0..1
    classe: str
    ancre: Tuple[float, float]  # point d'appui (bas-centre), dans le morceau
    z: float                   # distance de la table a l'ancre (m), nan si inconnue


def distance_table(plan, K, u, v) -> float:
    """Profondeur z (m) du point de la table vu au pixel (u, v), ou nan.
    Plan n.p + d = 0 dans le repere de l'image, d > 0 cote camera."""
    if plan is None or K is None:
        return float("nan")
    n, d = np.asarray(plan[:3], float), float(plan[3])
    K = np.asarray(K, float)
    r = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    den = float(n @ r)
    if abs(den) < 1e-9:
        return float("nan")
    t = -d / den
    return t if 0.05 < t < 5.0 else float("nan")


def extraire(image: np.ndarray, masques: Sequence[np.ndarray], classes: Sequence[str],
             plan=None, K=None, aire_min: int = 150) -> List[Morceau]:
    morceaux = []
    for m, c in zip(masques, classes):
        if m.sum() < aire_min:
            continue
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        alpha = m[y0:y1, x0:x1].astype(np.float32)
        u_a, v_a = float(np.median(xs)), float(ys.max())
        morceaux.append(Morceau(image[y0:y1, x0:x1].copy(), alpha, c,
                                (u_a - x0, v_a - y0), distance_table(plan, K, u_a, v_a)))
    return morceaux


def _transformer(m: Morceau, echelle: float, angle: float, miroir: bool, rng):
    img, alpha = m.image, m.alpha
    ax, ay = m.ancre
    if miroir:
        img, alpha = img[:, ::-1], alpha[:, ::-1]
        ax = img.shape[1] - 1 - ax
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, echelle)
    coins = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float) @ M.T
    x0, y0 = coins.min(axis=0)
    W, H = int(np.ceil(coins[:, 0].max() - x0)), int(np.ceil(coins[:, 1].max() - y0))
    M[:, 2] -= [x0, y0]
    if W < 3 or H < 3:
        return None
    img2 = cv2.warpAffine(np.ascontiguousarray(img), M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    a2 = cv2.warpAffine(np.ascontiguousarray(alpha), M, (W, H), flags=cv2.INTER_LINEAR)
    a2 = cv2.GaussianBlur(a2, (3, 3), 0.8)            # bord adouci
    ancre = M @ np.array([ax, ay, 1.0])
    # luminosite et teinte de la piece
    gain = rng.uniform(0.75, 1.25) * rng.uniform(0.95, 1.05, size=3)
    img2 = np.clip(img2.astype(np.float32) * gain, 0, 255)
    return img2, a2, ancre


def coller(image: np.ndarray, masques: List[np.ndarray], classes: List[str],
           morceaux: Sequence[Morceau], rng: np.random.Generator, n: int = 4,
           plan=None, K=None, p_contact: float = 0.5, z_max: Optional[float] = None,
           garde_min: float = 0.3) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    Colle `n` morceaux sur une copie de `image`. Rend (image, masques,
    classes), masques des pieces deja presentes rognes par l'occultation.
    """
    out = image.astype(np.float32).copy()
    H, W = image.shape[:2]
    masques = [m.copy() for m in masques]
    classes = list(classes)
    origine = [max(int(m.sum()), 1) for m in masques]
    if z_max is None:
        zs = [mo.z for mo in morceaux if np.isfinite(mo.z)]
        z_max = 1.3 * max(zs) if zs else 1.5
    for _ in range(n):
        if not len(morceaux):
            break
        mo = morceaux[int(rng.integers(len(morceaux)))]
        # point d'appui cible
        cible = None
        for _essai in range(30):
            if masques and rng.random() < p_contact:
                k = int(rng.integers(len(masques)))
                ys, xs = np.nonzero(masques[k])
                if not len(xs):
                    continue
                cote = rng.integers(4)
                largeur = mo.image.shape[1]
                if cote == 0:
                    u, v = xs.min() - largeur * 0.45, ys.max()
                elif cote == 1:
                    u, v = xs.max() + largeur * 0.45, ys.max()
                elif cote == 2:
                    u, v = np.median(xs) + rng.normal(0, 10), ys.min() + 2
                else:
                    u, v = np.median(xs) + rng.normal(0, 10), ys.max() + mo.image.shape[0] * 0.6
            else:
                u, v = rng.uniform(0, W), rng.uniform(H * 0.3, H)
            z = distance_table(plan, K, u, v)
            if plan is not None and (not np.isfinite(z) or z > z_max):
                continue
            if 0 <= u < W and 0 <= v < H:
                cible = (u, v, z)
                break
        if cible is None:
            continue
        u, v, z = cible
        if np.isfinite(z) and np.isfinite(mo.z):
            echelle = float(np.clip(mo.z / z, 0.5, 2.0))
        else:
            echelle = rng.uniform(0.85, 1.15)
        t = _transformer(mo, echelle, rng.uniform(-15, 15), rng.random() < 0.5, rng)
        if t is None:
            continue
        img2, a2, ancre = t
        x0, y0 = int(round(u - ancre[0])), int(round(v - ancre[1]))
        h2, w2 = a2.shape
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w2, W), min(y0 + h2, H)
        if xb - xa < 3 or yb - ya < 3:
            continue
        a = a2[ya - y0:yb - y0, xa - x0:xb - x0, None]
        out[ya:yb, xa:xb] = a * img2[ya - y0:yb - y0, xa - x0:xb - x0] + (1 - a) * out[ya:yb, xa:xb]
        nouveau = np.zeros((H, W), bool)
        nouveau[ya:yb, xa:xb] = a[..., 0] > 0.5
        if nouveau.sum() < 40:
            continue
        for k in range(len(masques)):
            masques[k] &= ~nouveau
        masques.append(nouveau)
        classes.append(mo.classe)
        origine.append(int(nouveau.sum()))
    # pieces trop cachees : plus d'etiquette (le reseau ne doit pas apprendre
    # a voir une piece la ou il n'en reste que quelques pixels)
    garde = [k for k, m in enumerate(masques)
             if m.sum() >= max(40, garde_min * origine[k])]
    return (np.clip(out, 0, 255).astype(np.uint8), [masques[k] for k in garde],
            [classes[k] for k in garde])
