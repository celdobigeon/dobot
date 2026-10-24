# -*- coding: utf-8 -*-
"""
Reconnaissance et localisation de pieces posees sur une table, a partir de
la profondeur d'une camera RealSense.

Pas d'apprentissage : la scene est GEOMETRIQUE.

  1. chaque pixel mesure devient un point 3D dans le repere de la camera
     (modele du stenope : x = (u - cx) z / fx, y = (v - cy) z / fy) ;
  2. le plan de la table est le plan qui porte le plus de points (RANSAC) ;
  3. tout ce qui depasse de la table de quelques millimetres, A L'INTERIEUR
     de la table, est un objet ; les objets se separent en composantes
     connexes ;
  4. de chaque objet on garde la FACE SUPERIEURE (les points les plus hauts) :
     son rectangle minimal donne longueur, largeur et orientation, sa hauteur
     au-dessus de la table la troisieme dimension ;
  5. les trois dimensions mesurees, comparees aux dimensions connues de la
     piece, disent sur quelle face elle est posee.

Les positions sont rendues dans deux reperes, chacun une matrice homogene :
le repere CAMERA (celui des points 3D) et un repere TABLE (z vers le haut,
origine sous la camera). Le passage au repere du ROBOT demandera un
etalonnage : une troisieme matrice, T_robot_camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:                        # pragma: no cover
    raise ImportError("OpenCV manque : .venv/bin/pip install opencv-python") from exc


#: m : au-dela, une profondeur est tenue pour aberrante
PROFONDEUR_MAX = 4.0


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    """Une capture : profondeur (m), parametres de la camera, images."""
    profondeur: np.ndarray                 # H x W, metres, 0 = pas de mesure
    K: np.ndarray                          # 3 x 3
    infrarouge: Optional[np.ndarray] = None
    couleur: Optional[np.ndarray] = None   # alignee sur la profondeur


def charger(chemin: str) -> Scene:
    """Lit un fichier .npz ecrit par capture_realsense.py."""
    z = np.load(chemin)
    return Scene(profondeur=z["profondeur"].astype(np.float64), K=z["K"],
                 infrarouge=z["infrarouge"] if "infrarouge" in z else None,
                 couleur=z["couleur"] if "couleur" in z else None)


@dataclass
class Piece:
    """Une piece reconnue sur la table."""
    numero: int
    face: str                       # "a plat", "sur la tranche", "debout", "inconnue"
    dimensions: Tuple[float, float, float]   # mesurees, mm : long, large, haut
    ecart: float                    # mm, ecart aux dimensions attendues
    T_camera: np.ndarray            # 4x4 : repere de la face superieure, camera
    T_table: np.ndarray             # 4x4 : le meme, dans le repere table
    contour: np.ndarray             # pixels du rectangle, pour l'affichage
    pixels: int
    type: str = ""                  # nom du type de piece reconnu

    @property
    def position_table(self) -> np.ndarray:
        """Centre de la face superieure dans le repere table, en mm."""
        return self.T_table[:3, 3] * 1000.0

    @property
    def lacet(self) -> float:
        """Orientation du grand cote autour de la verticale, en degres."""
        return float(np.degrees(np.arctan2(self.T_table[1, 0],
                                           self.T_table[0, 0])))


@dataclass
class Resultat:
    pieces: List[Piece]
    T_camera_table: np.ndarray            # repere table vu de la camera
    normale: np.ndarray
    distance_table: float                 # m, de la camera au plan
    hauteur: np.ndarray                   # H x W, metres au-dessus du plan
    trous: List[Tuple[int, int, int, int]] = field(default_factory=list)
    autres: List[str] = field(default_factory=list)   # objets ecartes


# ---------------------------------------------------------------------------
# Geometrie
# ---------------------------------------------------------------------------

def nuage(profondeur: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Points 3D (H x W x 3, metres) dans le repere de la camera."""
    H, W = profondeur.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.mgrid[0:H, 0:W]
    z = profondeur
    return np.dstack([(u - cx) * z / fx, (v - cy) * z / fy, z])


def plan_table(points: np.ndarray, seuil: float = 0.004, essais: int = 500,
               graine: int = 0) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Plan n.p + d = 0 portant le plus de points (RANSAC), affine par
    moindres carres sur ses points. La normale est orientee VERS la camera
    (d > 0) : la hauteur d'un point au-dessus de la table vaut n.p + d.
    Rend (n, d, masque des points du plan).
    """
    rng = np.random.default_rng(graine)
    meilleur, n_meilleur, d_meilleur = 0, None, None
    echant = points if len(points) < 60000 else points[
        rng.choice(len(points), 60000, replace=False)]
    for _ in range(essais):
        a, b, c = echant[rng.choice(len(echant), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -float(n @ a)
        compte = int(np.count_nonzero(np.abs(echant @ n + d) < seuil))
        if compte > meilleur:
            meilleur, n_meilleur, d_meilleur = compte, n, d
    dedans = np.abs(points @ n_meilleur + d_meilleur) < seuil
    # moindres carres : le plan qui passe au mieux par ses points
    q = points[dedans]
    centre = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - centre, full_matrices=False)
    n = vt[2]
    d = -float(n @ centre)
    if d < 0:
        n, d = -n, -d
    dedans = np.abs(points @ n + d) < seuil
    return n, d, dedans


def repere_table(n: np.ndarray, d: float) -> np.ndarray:
    """
    Repere TABLE vu de la camera (matrice 4x4) : z = normale vers le haut
    (vers la camera), origine = pied de la camera sur la table, x = axe x
    de la camera projete sur la table.
    """
    z = n / np.linalg.norm(n)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = -d * z                      # projection de l'origine camera
    return T


def corriger_courbure(hauteur: np.ndarray, P: np.ndarray, T_tc: np.ndarray,
                      utiles: np.ndarray, passes: int = 4) -> np.ndarray:
    """
    Hauteur au-dessus de la table REELLEMENT mesuree, et non du plan ideal.

    La profondeur de la RealSense se deforme de quelques millimetres vers
    les bords de l'image : un plan ne suit pas la table mesuree, et une
    piece fine (8 mm) se perd dans ces ecarts. On ajuste donc a la table une
    surface du second degre, h = f(x, y) dans le repere table, en plusieurs
    passes qui ecartent a chaque fois ce qui depasse (les pieces), puis on
    la retranche.
    """
    q = P[utiles]
    xy = (q @ T_tc[:3, :3].T + T_tc[:3, 3])[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    h = hauteur[utiles]
    garde = np.abs(h) < 0.012
    for _ in range(passes):
        coef, *_ = np.linalg.lstsq(A[garde], h[garde], rcond=None)
        residu = h - A @ coef
        sigma = 1.4826 * float(np.median(np.abs(residu[garde])))
        garde = np.abs(residu) < max(3.0 * sigma, 0.002)
    corrigee = hauteur.copy()
    corrigee[utiles] = residu
    return corrigee


def rayons_sur_plan(us: np.ndarray, vs: np.ndarray, K: np.ndarray,
                    n: np.ndarray, d: float, h: float) -> np.ndarray:
    """
    Points 3D (camera) des pixels (us, vs) PROJETES sur le plan parallele a
    la table, a la hauteur h : n.p + d = h. Le pixel definit un rayon issu
    de la camera, p = t.r avec r = ((u - cx)/fx, (v - cy)/fy, 1) ; le plan
    fixe t = (h - d) / (n.r). C'est le stenope a l'envers : il rend une
    position 3D a un pixel SANS mesure de profondeur, des qu'on sait a
    quelle hauteur il se trouve.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones(len(us))], axis=1)
    t = (h - d) / (r @ n)
    return r * t[:, None]


def classer(mesure: Sequence[float], attendu: Sequence[float]) -> Tuple[str, float]:
    """
    (longueur, largeur, hauteur) mesurees -> face d'appui et ecart en mm.
    La hauteur au-dessus de la table dit quelle dimension est verticale.
    """
    L, l, h = mesure
    a = sorted(attendu)                    # petit, moyen, grand
    meilleur = ("inconnue", float("inf"))
    for i_h in range(3):                   # quelle dimension est verticale
        reste = sorted([a[k] for k in range(3) if k != i_h], reverse=True)
        ecart = float(np.sqrt((L - reste[0]) ** 2 + (l - reste[1]) ** 2
                              + (h - a[i_h]) ** 2))
        # la verticale est la plus GRANDE dimension -> debout, la plus
        # petite -> a plat
        nom = {2: "debout", 1: "sur la tranche", 0: "a plat"}[i_h]
        if ecart < meilleur[1]:
            meilleur = (nom, ecart)
    return meilleur


def classer_types(mesure: Sequence[float],
                  types: Dict[str, Sequence[float]]) -> Tuple[str, str, float]:
    """Le type de piece et la face d'appui les plus proches : (type, face, ecart)."""
    meilleur = ("", "inconnue", float("inf"))
    for nom, dims in types.items():
        face, ecart = classer(mesure, dims)
        if ecart < meilleur[2]:
            meilleur = (nom, face, ecart)
    return meilleur


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detecter(scene: Scene, dimensions=(16, 30, 68),
             hauteur_min: Optional[float] = None, hauteur_max: float = 0.10,
             tolerance: float = 10.0, pixels_min: int = 120,
             seuil_brillant: int = 200) -> Resultat:
    """
    Pieces de dimensions connues (mm) posees sur la table.

    `dimensions` : celles d'un seul type de piece, (a, b, c) ; ou plusieurs
    types, {"epaisse": (20, 20, 45), "fine": (10, 20, 45)} -- chaque objet
    est alors compare a tous les types et a toutes leurs faces d'appui.

    `tolerance` : ecart maximal (mm) entre dimensions mesurees et attendues
    pour qu'une composante soit reconnue comme piece.

    `hauteur_min` : ce qui depasse moins de la table n'est pas un objet. Par
    defaut, il s'adapte au BRUIT de la table dans cette capture (4 fois son
    ecart-type robuste, entre 3 et 6 mm), et reste sous la moitie de la plus
    petite dimension des pieces : une piece fine couchee depasse a peine.

    Une piece claire SATURE souvent l'infrarouge : la profondeur y manque,
    et sa face superieure n'est mesuree qu'en partie. Les pixels sans
    mesure, COLLES a la piece et BRILLANTS dans l'infrarouge
    (>= `seuil_brillant`), sont alors rattaches a sa face superieure : leur
    position 3D se calcule par le rayon du pixel et la hauteur de la face
    (`rayons_sur_plan`).
    """
    types = (dict(dimensions) if isinstance(dimensions, dict)
             else {"": tuple(dimensions)})
    D = scene.profondeur.copy()
    # au-dela de quelques metres, la D435 ne mesure rien d'utile : des
    # valeurs aberrantes (jusqu'a 65 m) y apparaissent ; on les ecarte
    D[D > PROFONDEUR_MAX] = 0.0
    valides = D > 0
    P = nuage(D, scene.K)
    n, d, sur_plan = plan_table(P[valides])
    hauteur = np.zeros(D.shape)
    hauteur[valides] = P[valides] @ n + d

    # la TABLE : la plus grande zone de pixels du plan, bouchee ; seuls
    # comptent les objets qui sont dessus
    masque_plan = np.zeros(D.shape, np.uint8)
    masque_plan[valides] = sur_plan.astype(np.uint8)
    nb, lab, st, _ = cv2.connectedComponentsWithStats(masque_plan)
    plus_grand = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    table = (lab == plus_grand).astype(np.uint8)
    contours, _ = cv2.findContours(table, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    zone = np.zeros_like(table)
    cv2.drawContours(zone, [cv2.convexHull(max(contours, key=cv2.contourArea))],
                     -1, 1, -1)

    T_ct = repere_table(n, d)
    T_tc = np.linalg.inv(T_ct)
    # la table mesuree n'est pas tout a fait plane : on mesure les hauteurs
    # par rapport a ELLE (voir corriger_courbure)
    hauteur = corriger_courbure(hauteur, P, T_tc, valides & (zone > 0))

    if hauteur_min is None:
        # bruit de la table : ecart median absolu des points du plan
        h_table = hauteur[(table > 0) & valides]
        bruit = 1.4826 * float(np.median(np.abs(h_table - np.median(h_table))))
        plus_petite = min(min(dims) for dims in types.values()) / 1000.0
        hauteur_min = float(np.clip(4.0 * bruit, 0.003, 0.006))
        hauteur_min = min(hauteur_min, 0.5 * plus_petite)
    seuil_hauteur = hauteur_min

    objet = (valides & (hauteur > hauteur_min) & (hauteur < hauteur_max)
             & (zone > 0)).astype(np.uint8)
    # pixels sans mesure, sur la table, brillants dans l'infrarouge
    trous_brillants = None
    deja_pris = np.zeros(D.shape, bool)
    if scene.infrarouge is not None:
        trous_brillants = ((~valides) & (zone > 0)
                           & (scene.infrarouge >= seuil_brillant)).astype(np.uint8)
        trous_brillants = cv2.morphologyEx(trous_brillants, cv2.MORPH_OPEN,
                                           np.ones((3, 3), np.uint8))
    objet = cv2.morphologyEx(objet, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nb, lab, st, _ = cv2.connectedComponentsWithStats(objet)

    pieces: List[Piece] = []
    autres: List[str] = []
    for i in range(1, nb):
        if st[i, cv2.CC_STAT_AREA] < pixels_min:
            continue
        m = lab == i
        h_obj = hauteur[m]
        h_haut = float(np.percentile(h_obj, 90))
        # HAUTEUR : les points a moins de 4 mm du sommet
        haut = m & (hauteur > h_haut - 0.004)
        h_face = float(np.median(hauteur[haut]))
        # EMPREINTE : tous les points au-dessus de 60 % de la hauteur, vus
        # de dessus (repere table). Une face verticale, vue de biais, se
        # projette sur le BORD de l'empreinte : la mesure ne depend pas de
        # l'angle de vue. Les bords, lisses par la camera (hauteur
        # intermediaire), sont ainsi gardes -- la face superieure seule
        # sous-estimait la piece de 5 a 10 mm.
        q = P[m & (hauteur > 0.6 * h_haut)]
        if len(q) < 30:
            continue
        # completer par les pixels SANS mesure, colles a la piece et
        # brillants : la partie saturee de la meme face superieure
        complete = 0
        if trous_brillants is not None:
            voisins = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8))
            nb_t, lab_t = cv2.connectedComponents(trous_brillants)
            touches = set(np.unique(lab_t[(voisins > 0) & (trous_brillants > 0)]))
            touches.discard(0)
            if touches:
                ajout = np.isin(lab_t, list(touches))
                vs_, us_ = np.nonzero(ajout)
                q = np.vstack([q, rayons_sur_plan(us_.astype(float),
                                                  vs_.astype(float), scene.K,
                                                  n, d, h_face)])
                complete = len(us_)
                deja_pris[ajout] = True
        # dans le repere table, en mm
        q_t = (T_tc[:3, :3] @ q.T).T + T_tc[:3, 3]
        xy = (q_t[:, :2] * 1000.0).astype(np.float32)
        (cx_, cy_), (a_, b_), ang = cv2.minAreaRect(xy)
        L, l = max(a_, b_), min(a_, b_)
        h_mm = h_face * 1000.0
        type_, face, ecart = classer_types((L, l, h_mm), types)
        if ecart > tolerance:
            autres.append("composante de {:.0f} x {:.0f} mm, {:.0f} mm de "
                          "haut : pas une piece ({:.0f} mm d'ecart)"
                          .format(L, l, h_mm, ecart))
            continue
        # orientation du GRAND cote
        theta = np.radians(ang if a_ >= b_ else ang + 90.0)
        if theta > np.pi / 2:
            theta -= np.pi
        elif theta <= -np.pi / 2:
            theta += np.pi
        T_t = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        T_t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        T_t[:3, 3] = [cx_ / 1000.0, cy_ / 1000.0, h_mm / 1000.0]
        T_c = T_ct @ T_t
        # contour du rectangle, reprojete dans l'image
        coins = cv2.boxPoints(((cx_, cy_), (a_, b_), ang)) / 1000.0
        coins3 = np.c_[coins, np.full(4, h_mm / 1000.0), np.ones(4)]
        cc = (T_ct @ coins3.T)[:3]
        uv = (scene.K @ (cc / cc[2]))[:2].T
        pieces.append(Piece(len(pieces) + 1, face, (L, l, h_mm), ecart,
                            T_c, T_t, uv.astype(np.int32), int(m.sum()),
                            type_))

    # trous en forme d'objet : pas de mesure, A L'INTERIEUR de la table.
    # Souvent une piece claire saturee dans l'infrarouge.
    trous = []
    vide = ((~valides) & (zone > 0) & ~deja_pris).astype(np.uint8)
    vide = cv2.morphologyEx(vide, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nb, lab, st, _ = cv2.connectedComponentsWithStats(vide)
    for i in range(1, nb):
        x, y, w, hh, aire = st[i]
        if 300 <= aire <= 20000 and x > 0 and y > 0 \
                and x + w < D.shape[1] and y + hh < D.shape[0]:
            trous.append((int(x), int(y), int(w), int(hh)))
    return Resultat(pieces, T_ct, n, d, hauteur, trous, autres)


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def dessiner(scene: Scene, res: Resultat) -> np.ndarray:
    """Image annotee : pieces en vert, trous suspects en orange."""
    if scene.couleur is not None:
        img = scene.couleur.copy()
    elif scene.infrarouge is not None:
        img = cv2.cvtColor(scene.infrarouge, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.applyColorMap(cv2.convertScaleAbs(scene.profondeur,
                                                    alpha=255 / 1.2),
                                cv2.COLORMAP_JET)
    img = (img * 0.7).astype(np.uint8)
    for p in res.pieces:
        cv2.polylines(img, [p.contour], True, (60, 220, 60), 2)
        u, v = p.contour.mean(axis=0).astype(int)
        x, y, z = p.position_table
        for k, texte in enumerate((
                " ".join(filter(None, (str(p.numero), p.type, p.face))),
                "x{:.0f} y{:.0f} z{:.0f}".format(x, y, z),
                "{:.0f} deg".format(p.lacet))):
            cv2.putText(img, texte, (u - 40, v - 18 + 14 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
    for (x, y, w, h) in res.trous:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 255), 1)
        cv2.putText(img, "pas de mesure", (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 160, 255), 1,
                    cv2.LINE_AA)
    return img
