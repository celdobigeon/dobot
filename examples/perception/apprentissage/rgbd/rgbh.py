# -*- coding: utf-8 -*-
"""
Image RGB-H : la couleur et la HAUTEUR AU-DESSUS DE LA TABLE, alignees pixel
a pixel dans l'image couleur brute, pour un reseau de segmentation qui
s'appuie sur la forme 3D (independante du fond).

POURQUOI LA HAUTEUR (et pas la profondeur brute). La profondeur depend de
la pose de la camera (un plan incline est un degrade) ; la hauteur au-dessus
de la table, elle, est la meme partout : 0 sur la table, 8, 17, 19 mm sur
une piece couchee, quels que soient le fond et la camera. C'est l'indice le
plus directement lie a "il y a une piece ici", et il ne voit pas la nappe.

CONSTRUCTION (carte_hauteur) :
  1. la table (perception.analyser_table : plan RANSAC + correction de
     courbure) donne la hauteur de chaque pixel de PROFONDEUR ;
  2. chaque point 3D de profondeur est passe dans le repere couleur
     (T_couleur_profondeur) et projete par K_couleur
     (perception.projeter_profondeur) ;
  3. "splatting" avec tampon de profondeur : chaque point marque les 2 x 2
     pixels couleur autour de sa projection (la couleur a un champ plus
     etroit : un pixel de profondeur couvre ~1.3 pixel couleur, sans cela
     l'image serait criblee de trous), le plus PROCHE de la camera gagne
     (parallaxe de la base stereo : bords des pieces) ;
  4. TROUS (bois clair qui sature l'infrarouge, ombres de parallaxe, bord
     du champ de la profondeur) : rebouches par inpainting (Navier-Stokes)
     a demi-resolution. Un trou au milieu d'une piece prend une hauteur
     intermediaire entre ses bords : la forme reste visible, sa valeur est
     approximative (c'est une limite, voir README).

CODAGE (encoder_hauteur) : uint8 = clip(h_mm x 3, 0, 255) : 1/3 mm par
niveau, 0 a 85 mm (une piece debout fait 68 mm, deux couchees 38 mm) ;
sous la table (sol, decor) : 0.

DEUX VARIANTES d'entree (exporter_rgbh.py) :
  - "rgbh" : 4 canaux B, G, R, H (PNG BGRA). ultralytics 8.4 sait lire et
    entrainer des images a 4 canaux (data.yaml "channels: 4") ; la mosaique,
    les transformations geometriques et le copier-coller marchent ; RandomHSV
    est SAUTE (il ne traite que 3 canaux) : entrainer_rgbh.py le remplace
    par une version qui n'agit que sur la couleur ;
  - "rgh" : 3 canaux (H, G, R) : la hauteur REMPLACE le bleu (le moins
    informatif : bois orange, nappe blanche) ; entree standard, premiere
    couche pre-entrainee utilisable telle quelle ; l'augmentation HSV
    melangerait la hauteur aux couleurs : elle est remplacee de meme.

FOND CHANGE (changer_fond) : sans images reelles d'autres fonds, on SIMULE
le changement : les pixels de TABLE (hauteur < 4 mm, hors des masques
annotes dilates) sont remplaces par une table texturee rendue en
perspective avec la vraie camera (plan de la table de l'annotation),
eclairee par la lumiere lente de la scene ; la hauteur reste intacte (un
autre fond ne change pas la geometrie). Textures : familles procedurales
de synthese_images.py (lu, pas modifie), par defaut celles que l'autre
etude garde pour le test "fonds jamais vus" (bois, carrelage).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
APPRENTISSAGE = os.path.dirname(ICI)
for _d in (APPRENTISSAGE, ICI):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import commun  # noqa: E402
from commun import perception  # noqa: E402

#: niveaux par millimetre du canal de hauteur
NIVEAUX_MM = 3.0
#: hauteur max d'un pixel de "table" pour le changement de fond (m)
SEUIL_TABLE = 0.004


# ---------------------------------------------------------------------------
# Carte de hauteur dans l'image couleur
# ---------------------------------------------------------------------------

def splat(u: np.ndarray, v: np.ndarray, z: np.ndarray, val: np.ndarray, W: int, H: int,
          empreinte: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Valeurs `val` de points vus aux pixels (u, v) a la distance z : image
    H x W (nan : rien) et tampon de profondeur. Chaque point couvre
    empreinte x empreinte pixels a partir de floor(u, v) ; le plus proche
    gagne.
    """
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    z = np.asarray(z, float).ravel()
    val = np.asarray(val, float).ravel()
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    zbuf = np.full(H * W, np.inf)
    idx_l, z_l, val_l = [], [], []
    for dy in range(empreinte):
        for dx in range(empreinte):
            x, y = x0 + dx, y0 + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            idx_l.append(y[ok] * W + x[ok])
            z_l.append(z[ok])
            val_l.append(val[ok])
    idx = np.concatenate(idx_l)
    zz = np.concatenate(z_l)
    vv = np.concatenate(val_l)
    np.minimum.at(zbuf, idx, zz)
    gagne = zz <= zbuf[idx]
    out = np.full(H * W, np.nan)
    out[idx[gagne]] = vv[gagne]
    return out.reshape(H, W), zbuf.reshape(H, W)


def carte_hauteur(capture, vue=None, table=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hauteur au-dessus de la table (m) de chaque pixel de l'image de la vue
    (couleur brute par defaut) : (H x W float, nan sans mesure ; masque des
    pixels mesures). Voir l'en-tete.
    """
    cap = perception.charger_capture(capture)
    vue = vue or perception.vue_couleur(cap)
    K = np.asarray(cap["K"], float)
    table = table or perception.analyser_table(cap["profondeur"], K)
    u, v, ok = perception.projeter_profondeur(cap["profondeur"], K, vue)
    P = table.P[ok]
    Q = P @ vue.T_cp[:3, :3].T + vue.T_cp[:3, 3]
    h = table.hauteur[ok]
    Wc, Hc = vue.taille
    carte, _ = splat(u[ok], v[ok], Q[:, 2], h, Wc, Hc)
    return carte, np.isfinite(carte)


def boucher(carte: np.ndarray, facteur: int = 2) -> np.ndarray:
    """Trous (nan) rebouches par inpainting Navier-Stokes, calcule a
    1/facteur de la resolution (rapide) ; les pixels mesures sont gardes."""
    valides = np.isfinite(carte)
    if valides.all():
        return carte.copy()
    if not valides.any():
        return np.zeros_like(carte)
    H, W = carte.shape
    c = np.where(valides, carte, 0.0).astype(np.float32)
    w, h = max(W // facteur, 1), max(H // facteur, 1)
    # moyenne des seuls pixels mesures (pas de melange avec les zeros des trous)
    somme = cv2.resize(c * valides, (w, h), interpolation=cv2.INTER_AREA)
    poids = cv2.resize(valides.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    petit = np.where(poids > 1e-3, somme / np.maximum(poids, 1e-3), 0.0).astype(np.float32)
    trous = (poids <= 0.25).astype(np.uint8)
    rempli = cv2.inpaint(petit, trous, 5, cv2.INPAINT_NS)
    grand = cv2.resize(rempli, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.where(valides, carte, grand).astype(np.float32)


def encoder_hauteur(h_m: np.ndarray) -> np.ndarray:
    """Hauteur (m) -> uint8 (NIVEAUX_MM niveaux par mm, 0 a 85 mm)."""
    h = np.nan_to_num(np.asarray(h_m, float), nan=0.0) * 1000.0 * NIVEAUX_MM
    return np.clip(np.round(h), 0, 255).astype(np.uint8)


def decoder_hauteur(h8: np.ndarray) -> np.ndarray:
    """uint8 -> hauteur (m)."""
    return np.asarray(h8, float) / NIVEAUX_MM / 1000.0


def composer(couleur_bgr: np.ndarray, h8: np.ndarray, variante: str = "rgbh") -> np.ndarray:
    """Image d'entree du reseau : "rgbh" -> B, G, R, H (4 canaux) ;
    "rgh" -> H, G, R (la hauteur a la place du bleu) ; "rgb" -> la couleur."""
    if variante == "rgbh":
        return np.dstack([couleur_bgr, h8])
    if variante == "rgh":
        return np.dstack([h8, couleur_bgr[..., 1], couleur_bgr[..., 2]])
    if variante == "rgb":
        return couleur_bgr.copy()
    raise ValueError("variante inconnue : " + variante)


def image_rgbh(capture, variante: str = "rgbh", vue=None, table=None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(image d'entree, hauteur en m rebouchee) pour une capture."""
    cap = perception.charger_capture(capture)
    vue = vue or perception.vue_couleur(cap)
    carte, _ = carte_hauteur(cap, vue, table)
    h = boucher(carte)
    return composer(vue.image, encoder_hauteur(h), variante), h


# ---------------------------------------------------------------------------
# Fond change (simulation)
# ---------------------------------------------------------------------------

def zone_table(h_m: np.ndarray, mesuree: np.ndarray, masques: Sequence[np.ndarray] = (),
               dilatation: int = 3) -> np.ndarray:
    """
    Pixels de TABLE a repeindre : hauteur mesuree (pas rebouchee) a moins
    de SEUIL_TABLE du plan ; ouverture morphologique (le decor du fond
    traverse la hauteur 0 le long de fines bandes : retirees), puis les
    grandes composantes seulement (>= 5 % de la plus grande). Les pixels
    SANS mesure situes dans l'enveloppe convexe de la table (bord proche
    hors de portee de la profondeur, reflets) en font partie aussi. Enfin,
    hors des pieces annotees (dilatees de `dilatation` pixels).
    """
    z = mesuree & (np.abs(np.nan_to_num(h_m, nan=1.0)) < SEUIL_TABLE)
    z = cv2.morphologyEx(z.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    z = cv2.morphologyEx(z, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
    nb, lab, st, _ = cv2.connectedComponentsWithStats(z.astype(np.uint8))
    if nb > 1:
        aires = st[1:, cv2.CC_STAT_AREA]
        grands = np.zeros(nb, bool)
        grands[1:] = aires >= max(400, 0.05 * aires.max())
        z = grands[lab]
        pts = cv2.findNonZero(z.astype(np.uint8))
        if pts is not None:
            env = np.zeros(z.shape, np.uint8)
            cv2.fillConvexPoly(env, cv2.convexHull(pts), 1)
            z |= (env > 0) & ~mesuree
    if len(masques):
        pieces = np.any(np.stack([np.asarray(m, bool) for m in masques]), axis=0)
        pieces = cv2.dilate(pieces.astype(np.uint8), np.ones((dilatation, dilatation), np.uint8)) > 0
        z &= ~pieces
    return z


def _texture(rng, famille: str, taille: int = 1024) -> np.ndarray:
    """Texture BGR float 0-1 : synthese_images.texture_famille si elle est
    la (meme generateur que l'etude des images rendues), sinon un bruit
    colore simple."""
    try:
        import synthese_images as si
        return si.texture_famille(rng, famille, taille, taille).astype(np.float32)
    except Exception:
        c = rng.uniform(0.1, 0.9, 3)
        b = cv2.GaussianBlur(rng.random((taille, taille)).astype(np.float32), (0, 0), 6)
        b = (b - b.min()) / max(float(b.max() - b.min()), 1e-6)
        return (c[None, None] * (0.7 + 0.6 * b[..., None])).astype(np.float32)


def table_texturee(K: np.ndarray, W: int, H: int, plan: Sequence[float], rng,
                   familles: Sequence[str] = ("bois", "carrelage")) -> Tuple[np.ndarray, str]:
    """Une table texturee rendue en perspective par la camera (K, plan de la
    table dans son repere) : image BGR float 0-1 et famille tiree."""
    fam = familles[int(rng.integers(len(familles)))]
    tex = _texture(rng, fam)
    try:
        import synthese_images as si
        cam = si.camera_de_plan(K, W, H, plan)
        img, _, dedans = si.rendre_table(cam, tex, rng.uniform(0.4, 1.4), rng.uniform(0, np.pi),
                                         tuple(rng.uniform(0, 1024, 2)), (-10, 10, -10, 10))
        img = np.where(dedans[..., None], img, tex.mean(axis=(0, 1))[None, None])
    except Exception:
        img = cv2.resize(tex, (W, H))
    return img.astype(np.float32), fam


def changer_fond(image: np.ndarray, zone: np.ndarray, nouveau: np.ndarray,
                 garder_lumiere: bool = True) -> np.ndarray:
    """Repeint `zone` de `image` (BGR uint8) avec `nouveau` (BGR float 0-1),
    module par la lumiere LENTE de l'image d'origine (gradient d'eclairage
    de la table, pas son motif), bord adouci d'un pixel."""
    img = image.astype(np.float32) / 255.0
    nv = nouveau
    if garder_lumiere:
        lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        moy = float(lum[zone].mean()) if zone.any() else 128.0
        lent = cv2.GaussianBlur(np.where(zone, lum, moy), (0, 0), 25) / max(moy, 1.0)
        nv = nv * lent[..., None]
    a = cv2.GaussianBlur(zone.astype(np.float32), (3, 3), 0.8)[..., None]
    sortie = a * nv + (1 - a) * img
    return np.clip(sortie * 255.0 + 0.5, 0, 255).astype(np.uint8)
