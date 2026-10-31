# -*- coding: utf-8 -*-
"""
Geometrie des POINTS CLES : coins de la face superieure d'une piece, ordre
canonique, et pose par PnP (Perspective-n-Point).

POURQUOI DES COINS. Un masque dit ou est la piece ; il faut ensuite une
recherche (ajuster_masque, rendre et comparer) pour en tirer une pose. Les
4 coins de la face superieure, eux, donnent la pose presque directement :
avec les dimensions connues de la face et K, un PnP pour 4 points
coplanaires (IPPE, Collins et Bartoli 2014) est exact, rapide (< 0.1 ms) et
ne depend pas du fond. C'est le principe des detecteurs de pose par points
cles (BB8, PVNet, YOLO-pose...).

ORDRE CANONIQUE DES COINS (le choix a documenter). Une face rectangulaire
vue de dessus a une symetrie de 180 deg (et une piece est symetrique : les
deux bouts se ressemblent). Un reseau ne peut pas apprendre "le coin no 0"
si ce coin n'est pas defini par l'image. On definit donc l'ordre DANS
L'IMAGE, a partir de l'axe du grand cote :
  - d = direction du grand cote dans l'image, orientee vers la DROITE
    (d_x > 0 ; si d_x = 0, vers le haut) ;
  - le bout "A" est celui d'ou part d, le bout "B" celui ou il arrive ;
  - n = d tourne de +90 deg (repere image, y vers le bas) ; dans chaque
    bout, le coin "haut" est celui du cote -n ;
  - ordre : 0 = A haut, 1 = B haut, 2 = B bas, 3 = A bas (sens horaire a
    l'ecran : haut-gauche, haut-droite, bas-droite, bas-gauche pour une
    piece horizontale).
Cette regle est COMPATIBLE avec le retournement horizontal (augmentation
fliplr) : le miroir echange 0<->1 et 2<->3 (flip_idx = [1, 0, 3, 2]),
quelle que soit l'orientation. Elle est DISCONTINUE quand le grand cote est
vertical dans l'image (d_x change de signe) : l'ordre y tourne de 180 deg
(0<->2, 1<->3). Aucune regle continue n'existe (l'orientation modulo 180
deg est un cercle ; choisir un bout est une section du revetement double),
d'ou une PERTE INVARIANTE a cette permutation pour l'entrainement
(entrainer.py : minimum sur {identite, rotation de 180 deg}) et une
evaluation qui prend aussi ce minimum. Pour la pose, la permutation de 180
deg est sans effet : la piece est symetrique.

POSE. Deux estimateurs, essayes pour chaque (type, face) candidat :
  - SUR PLAN (le plus sur) : la face superieure est parallele a la table, a
    une hauteur connue (epaisseur du candidat + base) ; les rayons des 4
    coins coupent ce plan, et un rectangle de taille connue s'ajuste a ces
    4 points (Procrustes 2D : rotation + translation, sans echelle). 3
    degres de liberte seulement : robuste au bruit des coins, et la hauteur
    vient de la table (profondeur), pas de l'echelle apparente (1 px
    d'erreur sur un coin = ~5 mm de profondeur par PnP libre a 0.45 m) ;
  - LIBRE (IPPE) : pose 6D, pour une piece inclinee (un bout sur une autre).
Le choix entre candidats : erreur de reprojection et accord avec la hauteur
mesuree par la profondeur dans le quadrilatere (voir proposer.py).

Reperes : ceux de perception.py. Repere de la FACE SUPERIEURE : x le long
du grand cote, y le long du petit, z vers le haut (hors de la piece),
origine au centre de la face. T_table : ce repere dans le repere table ;
la vue couleur voit le repere table par vue.T_cp @ table.T_ct.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import cv2

#: le retournement horizontal echange ces coins (voir l'en-tete)
FLIP_IDX = (1, 0, 3, 2)
#: les permutations sous lesquelles la face est la meme (symetrie de 180 deg)
SYMETRIES = ((0, 1, 2, 3), (2, 3, 0, 1))


# ---------------------------------------------------------------------------
# Coins et ordre canonique
# ---------------------------------------------------------------------------

def coins_face(L: float, l: float) -> np.ndarray:
    """Les 4 coins (4 x 3) d'une face L x l dans son propre repere (z = 0),
    dans l'ordre du contour : (-x,-y), (+x,-y), (+x,+y), (-x,+y) ; l'arete
    0 -> 1 suit le grand cote si L >= l."""
    return np.array([[-L / 2, -l / 2, 0.0], [L / 2, -l / 2, 0.0],
                     [L / 2, l / 2, 0.0], [-L / 2, l / 2, 0.0]])


def ordre_canonique(uv: np.ndarray) -> np.ndarray:
    """
    Indices (4,) qui mettent dans l'ordre canonique (voir l'en-tete) 4 coins
    donnes dans l'ordre du CONTOUR, l'arete 0 -> 1 le long du GRAND cote
    (comme coins_face). Rend `idx` tel que uv[idx] soit canonique.
    """
    p = np.asarray(uv, float).reshape(4, 2)
    d = (p[1] - p[0]) + (p[2] - p[3])          # du bout {0, 3} vers le bout {1, 2}
    if d[0] < 0 or (d[0] == 0 and d[1] > 0):
        A, B = (1, 2), (0, 3)
        d = -d
    else:
        A, B = (0, 3), (1, 2)
    n = np.array([-d[1], d[0]])                # +90 deg (y vers le bas)
    c = p.mean(axis=0)

    def haut_bas(paire):
        i, j = paire
        return (i, j) if (p[i] - c) @ n <= (p[j] - c) @ n else (j, i)

    a_haut, a_bas = haut_bas(A)
    b_haut, b_bas = haut_bas(B)
    return np.array([a_haut, b_haut, b_bas, a_bas])


def canoniser(uv: np.ndarray, coins3d: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Coins image (4 x 2) en ordre de contour -> ordre canonique. Si les coins
    3D correspondants (4 x 3, meme ordre) sont donnes, le grand cote est pris
    en 3D (sur l'image, une face presque carree vue en biais peut tromper) :
    on fait d'abord tourner le contour d'un cran si l'arete 0 -> 1 est la
    plus courte.
    """
    uv = np.asarray(uv, float).reshape(4, 2)
    if coins3d is not None:
        q = np.asarray(coins3d, float).reshape(4, 3)
        if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1]):
            uv = uv[[1, 2, 3, 0]]
    return uv[ordre_canonique(uv)]


def erreur_coins(pred: np.ndarray, vrai: np.ndarray,
                 visibles: Optional[np.ndarray] = None) -> float:
    """Erreur moyenne (px) entre deux jeux de 4 coins canoniques, au mieux
    des permutations de symetrie (la piece est symetrique)."""
    pred = np.asarray(pred, float).reshape(4, 2)
    vrai = np.asarray(vrai, float).reshape(4, 2)
    m = np.ones(4, bool) if visibles is None else np.asarray(visibles, bool)
    if not m.any():
        return float("nan")
    return float(min(np.linalg.norm(pred[list(s)][m] - vrai[m], axis=1).mean()
                     for s in SYMETRIES))


# ---------------------------------------------------------------------------
# Projection (stenope, comme perception.projeter pour la D435)
# ---------------------------------------------------------------------------

def projeter(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q, float).reshape(-1, 3)
    z = np.where(np.abs(Q[:, 2]) < 1e-9, 1e-9, Q[:, 2])
    return np.stack([K[0, 0] * Q[:, 0] / z + K[0, 2], K[1, 1] * Q[:, 1] / z + K[1, 2]], 1)


def coins_image(T_vue_face: np.ndarray, L: float, l: float, K: np.ndarray) -> np.ndarray:
    """Coins (ordre du contour) de la face L x l posee en T_vue_face (repere
    de la face dans celui de la vue), projetes : 4 x 2."""
    q = coins_face(L, l) @ T_vue_face[:3, :3].T + T_vue_face[:3, 3]
    return projeter(q, K)


# ---------------------------------------------------------------------------
# PnP
# ---------------------------------------------------------------------------

def _rt_vers_T(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


def _objets_canoniques(L: float, l: float) -> Tuple[np.ndarray, np.ndarray]:
    """Les deux affectations possibles des coins canoniques aux coins de la
    face (le sens du contour depend de la vue : face vue de dessus ou image
    en miroir) ; la symetrie de 180 deg ne change pas la pose."""
    o = coins_face(L, l)
    return o, o[[3, 2, 1, 0]]


def pnp_ippe(uv: np.ndarray, L: float, l: float, K: np.ndarray,
             coeffs: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float]:
    """
    Pose LIBRE (6D) de la face L x l (m) dont les 4 coins canoniques sont
    vus en `uv` : IPPE (solution exacte pour une cible plane), les deux
    solutions de chaque affectation essayees, gardee celle de plus faible
    erreur de reprojection dont la normale (z de la face) regarde la camera.
    Rend (T_vue_face 4x4, erreur RMS px) ; (None, inf) si echec.
    """
    uv = np.asarray(uv, np.float64).reshape(4, 1, 2)
    D = np.zeros(5) if coeffs is None else np.asarray(coeffs, float)
    meilleur, err_min = None, np.inf
    for obj in _objets_canoniques(L, l):
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj.reshape(4, 1, 3), uv, K, D, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        for k in range(int(n)):
            T = _rt_vers_T(rvecs[k], tvecs[k])
            if T[2, 3] <= 0:
                continue
            # la face superieure est vue de son cote exterieur
            if T[:3, 2] @ T[:3, 3] > 0:
                continue
            e = float(np.asarray(errs).reshape(-1)[k])
            if e < err_min:
                meilleur, err_min = T, e
    return meilleur, err_min


def ajuster_rectangle_2d(xy: np.ndarray, L: float, l: float) -> Tuple[float, float, float, float]:
    """
    Rectangle L x l (m) ajuste (rotation + translation, Procrustes / Kabsch
    2D) a 4 points du plan donnes dans l'ordre canonique ; les deux sens du
    contour sont essayes. Rend (x, y, theta, rms) : centre, angle du grand
    cote (modulo pi), ecart RMS en metres.
    """
    xy = np.asarray(xy, float).reshape(4, 2)
    c = xy.mean(axis=0)
    Y = xy - c
    meilleur = None
    for obj in _objets_canoniques(L, l):
        X = obj[:, :2]
        Hm = X.T @ Y
        U, _, Vt = np.linalg.svd(Hm)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:                 # pas de miroir : l'autre sens
            continue
        rms = float(np.sqrt(np.mean(np.sum((X @ R.T - Y) ** 2, axis=1))))
        if meilleur is None or rms < meilleur[3]:
            th = float(np.arctan2(R[1, 0], R[0, 0]))
            meilleur = (float(c[0]), float(c[1]), th, rms)
    if meilleur is None:                         # degenere : centre seul
        return float(c[0]), float(c[1]), 0.0, float("inf")
    x, y, th, rms = meilleur
    th = (th + np.pi / 2) % np.pi - np.pi / 2
    return x, y, th, rms


def intersection_plan(uv: np.ndarray, K: np.ndarray, T_vue_table: np.ndarray,
                      h: float) -> np.ndarray:
    """Pixels (N x 2, stenope) -> points (N x 2) du repere table sur le plan
    horizontal z = h (m). T_vue_table : repere table vu de la vue."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    r = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1],
                  np.ones(len(uv))], axis=1)
    T = np.linalg.inv(T_vue_table)               # vue -> table
    o = T[:3, 3]
    d = r @ T[:3, :3].T
    dz = np.where(np.abs(d[:, 2]) < 1e-9, -1e-9, d[:, 2])
    t = (h - o[2]) / dz
    return (o + d * t[:, None])[:, :2]


def pose_sur_plan(uv: np.ndarray, L: float, l: float, h_dessus: float, K: np.ndarray,
                  T_vue_table: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Pose de la face superieure L x l (m) CONTRAINTE a etre horizontale a la
    hauteur h_dessus (m, au-dessus de la table) : rayons des coins coupes
    par ce plan, rectangle ajuste. Rend (T_table 4x4 de la face, erreur RMS
    de reprojection en px).
    """
    xy = intersection_plan(uv, K, T_vue_table, h_dessus)
    x, y, th, _ = ajuster_rectangle_2d(xy, L, l)
    T = np.eye(4)
    c, s = np.cos(th), np.sin(th)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, h_dessus]
    return T, erreur_reprojection(uv, T_vue_table @ T, L, l, K)


def erreur_reprojection(uv: np.ndarray, T_vue_face: np.ndarray, L: float, l: float,
                        K: np.ndarray) -> float:
    """RMS (px) entre 4 coins canoniques observes et ceux de la face posee,
    au mieux des affectations (sens du contour et symetrie)."""
    uv = np.asarray(uv, float).reshape(4, 2)
    q = coins_image(T_vue_face, L, l, K)
    q = canoniser(q)
    return float(min(np.sqrt(np.mean(np.sum((q[list(s)] - uv) ** 2, axis=1)))
                     for s in SYMETRIES))


def inclinaison_deg(T_table: np.ndarray) -> float:
    """Angle (deg) entre la normale de la face et la verticale de la table."""
    return float(np.degrees(np.arccos(np.clip(abs(T_table[2, 2]), -1.0, 1.0))))


def lacet(T_table: np.ndarray) -> float:
    """Orientation du grand cote (rad), modulo pi, dans [-pi/2, pi/2)."""
    return float((np.arctan2(T_table[1, 0], T_table[0, 0]) + np.pi / 2) % np.pi - np.pi / 2)


def ecart_pose(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    """(distance des centres de face en mm, ecart de lacet en deg modulo 180)
    entre deux poses de face superieure (repere table)."""
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0)
    da = abs(lacet(T_a) - lacet(T_b)) % np.pi
    da = min(da, np.pi - da)
    return dt, float(np.degrees(da))


def boite_englobante(uv: np.ndarray, W: int, H: int, marge: float = 0.0) -> Sequence[float]:
    uv = np.asarray(uv, float).reshape(-1, 2)
    x0, y0 = uv.min(axis=0) - marge
    x1, y1 = uv.max(axis=0) + marge
    return [float(np.clip(x0, 0, W - 1)), float(np.clip(y0, 0, H - 1)),
            float(np.clip(x1, 0, W - 1)), float(np.clip(y1, 0, H - 1))]
