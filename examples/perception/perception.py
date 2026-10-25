# -*- coding: utf-8 -*-
"""
Perception des pieces : la bibliotheque a utiliser dans les programmes.

    from perception import Perception
    p = Perception(types={"epaisse": (17, 19, 68), "fine": (8, 17, 68)})
    pieces = p.detecter()                 # capture fraiche par le serveur
    pieces = p.detecter("captures/env2_reglee.npz")   # ou une capture enregistree
    for piece in pieces:
        print(piece.type, piece.face, piece.confiance)
        print(piece.T_table)              # 4x4 : face superieure, repere table
    cv2.imwrite("vue.png", p.dessiner(pieces))

POURQUOI DEUX ETAGES. Un reseau de segmentation (YOLO-seg, entraine sur nos
propres images, voir apprentissage/) dit OU sont les pieces dans l'image
couleur : il voit une piece meme quand la profondeur y est trouee (bois clair
qui sature l'infrarouge), et separe deux pieces qui se touchent. Mais il ne
mesure rien. La GEOMETRIE mesure : on connait les dimensions exactes des
pieces, le plan de la table (profondeur) et la camera. Pour chaque masque, on
cherche la piece de taille connue, posee sur la table, dont la silhouette
projetee dans l'image couvre au mieux le masque. Un masque qu'aucune piece
n'explique (une regle, une main) est rejete : le reseau propose, la geometrie
verifie.

  1. TABLE : plan RANSAC et correction de courbure (vision.py), dans le
     repere de la camera de PROFONDEUR.
  2. COULEUR : l'image couleur BRUTE (entiere) si la capture porte sa
     geometrie (K_couleur, T_couleur_profondeur, voir camera.py) ; sinon la
     couleur recalee, deja dans le repere de la profondeur.
  3. MASQUES : YOLO-seg (modeles/pieces-seg.pt) sur cette image.
  4. HAUTEUR : les points de profondeur qui se projettent dans le masque
     (erode : les bords melangent piece et table) donnent la hauteur de la
     face superieure, donc les faces d'appui plausibles.
  5. POSE : pour chaque (type, face) plausible, recherche locale de
     (x, y, angle) sur la table maximisant l'IoU entre le masque et la
     silhouette projetee du parallelepipede (enveloppe convexe de ses 8
     coins). Les pixels des AUTRES masques sont ignores : une piece cachee
     en partie par une autre n'est pas tiree vers sa partie visible.
  6. CHOIX : IoU x accord de la hauteur mesuree (x 0.8 si le type differe
     de la classe du reseau). IoU trop faible : rejet.
  7. PILES : une piece peut etre posee sur une autre. Les bases essayees
     (bases_autour) : la table, les sommets de pile que la profondeur
     PLATE autour du masque montre (la porteuse depasse), et toutes si la
     hauteur mesuree n'est celle d'aucune piece sur la table. C'est
     l'epaisseur (sommet - base) qui dit le type et la face. Puis chaque
     piece empilee est reliee a sa porteuse (sommet a la hauteur de sa
     base, empreintes qui se recouvrent) ; sans porteuse, elle est
     reposee sur le sommet d'une piece vue sous elle, ou cette piece,
     mal mesuree car cachee, est re-ajustee a la bonne hauteur, ou enfin
     la piece est reposee sur la table (pas de pile fantome). support,
     hauteur_base, dessus ; ordre_de_prise(pieces).

  8. AFFINAGE DE LA SCENE (raffinement.py, actif par defaut ;
     Perception(raffinement=False) pour s'en passer). Les pieces trouvees
     ci-dessus, celles de la detection geometrique (vision_modele : elle
     ne depend pas du fond) et les masques du reseau ne sont plus que des
     PROPOSITIONS : une scene 3D de boites (a plat, empilees ou INCLINEES,
     sans interpenetration, posees de facon stable) est rendue dans les
     vues de la camera et corrigee jusqu'a expliquer la profondeur
     mesuree ; les pieces qui n'expliquent rien sont retirees, les zones
     inexpliquees deviennent de nouvelles pieces. Le reseau ne peut pas
     imposer une piece que la profondeur contredit. Chaque piece porte en
     plus `inclinaison` (deg) et `appuis`.

Sans modele entraine (ou sans image couleur), repli sur la detection
geometrique seule (vision_modele si present, sinon vision), puis affinage.

Les reperes sont ceux de vision.py : T_camera dans le repere de la camera de
PROFONDEUR, T_table dans le repere table (z vers le haut). Avec un
etalonnage camera -> robot (T_robot_camera, repere de la profondeur), chaque
piece porte aussi T_robot.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError as exc:                        # pragma: no cover
    raise ImportError("OpenCV manque : .venv/bin/pip install opencv-python") from exc

ICI = os.path.dirname(os.path.abspath(__file__))
if ICI not in sys.path:
    sys.path.insert(0, ICI)
import vision  # noqa: E402

#: l'ordre dans lequel saisir les pieces (les libres d'abord, du haut vers le bas)
ordre_de_prise = vision.ordre_de_prise

#: types de pieces par defaut (mm) ; toujours configurables
TYPES = {"epaisse": (17.0, 19.0, 68.0), "fine": (8.0, 17.0, 68.0)}
#: le modele entraine par apprentissage/entrainer.py
MODELE = os.path.join(ICI, "modeles", "pieces-seg.pt")
FACES = ("a plat", "sur la tranche", "debout")


def forme(dims: Sequence[float], face: str) -> Tuple[float, float, float]:
    """(long, large, haut) en mm d'un type pose sur une face : la hauteur
    est la petite dimension a plat, la moyenne sur la tranche, la grande
    debout."""
    a, b, c = sorted(float(x) for x in dims)
    return {"a plat": (c, b, a), "sur la tranche": (c, a, b),
            "debout": (b, a, c)}[face]


# ---------------------------------------------------------------------------
# Captures et geometrie couleur
# ---------------------------------------------------------------------------

def charger_capture(source) -> Dict[str, np.ndarray]:
    """Une capture sous forme de dictionnaire, depuis un chemin .npz, un
    dictionnaire (camera.capturer()) ou une vision.Scene."""
    if isinstance(source, (str, os.PathLike)):
        with np.load(source) as z:
            return {k: z[k] for k in z.files}
    if isinstance(source, vision.Scene):
        c = {"profondeur": np.asarray(source.profondeur, np.float32),
             "K": source.K}
        if source.infrarouge is not None:
            c["infrarouge"] = source.infrarouge
        if source.couleur is not None:
            c["couleur"] = source.couleur
        return c
    return dict(source)


def scene_de(capture: Dict[str, np.ndarray]) -> vision.Scene:
    return vision.Scene(profondeur=np.asarray(capture["profondeur"], np.float64),
                        K=capture["K"], infrarouge=capture.get("infrarouge"),
                        couleur=capture.get("couleur"))


@dataclass
class VueCouleur:
    """L'image ou l'on segmente, et comment elle voit l'espace.

    T_cp : passage du repere de la PROFONDEUR a celui de cette image
    (identite pour la couleur recalee). `modele` : le modele de distorsion
    RealSense des coefficients ("" ou "...inverse_brown_conrady"...)."""
    image: np.ndarray
    K: np.ndarray
    coeffs: np.ndarray
    T_cp: np.ndarray
    source: str                       # "couleur_brute", "couleur", "infrarouge"
    modele: str = ""

    @property
    def taille(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


def a_geometrie_brute(capture: Dict[str, np.ndarray]) -> bool:
    return all(k in capture for k in ("couleur_brute", "K_couleur",
                                      "T_couleur_profondeur"))


def vue_couleur(capture: Dict[str, np.ndarray], brute: bool = True,
                infrarouge: bool = False) -> Optional[VueCouleur]:
    """
    La couleur BRUTE si la capture porte sa geometrie (pas de trous, champ
    entier), sinon la couleur RECALEE (trouee la ou la profondeur manque,
    mais deja dans le repere de la profondeur : K et identite). Sans
    couleur : l'infrarouge si `infrarouge`, sinon None.
    """
    K = np.asarray(capture["K"], float)
    if brute and a_geometrie_brute(capture):
        return VueCouleur(np.ascontiguousarray(capture["couleur_brute"]),
                          np.asarray(capture["K_couleur"], float),
                          np.asarray(capture.get("coeffs_couleur", np.zeros(5)), float),
                          np.asarray(capture["T_couleur_profondeur"], float),
                          "couleur_brute", str(capture.get("modele_couleur", "")))
    if "couleur" in capture:
        return VueCouleur(np.ascontiguousarray(capture["couleur"]), K,
                          np.zeros(5), np.eye(4), "couleur")
    if infrarouge and "infrarouge" in capture:
        return VueCouleur(cv2.cvtColor(capture["infrarouge"], cv2.COLOR_GRAY2BGR),
                          K, np.zeros(5), np.eye(4), "infrarouge")
    return None


def projeter(Q: np.ndarray, vue: VueCouleur) -> np.ndarray:
    """
    Points 3D (N x 3) du repere de la VUE -> pixels (N x 2).

    Coefficients nuls (cas habituel de la D435) : stenope. Modele RealSense
    "inverse Brown-Conrady" : comme librealsense (rs2_project_point_to_pixel),
    la projection n'applique PAS ces coefficients, qui servent a la
    deprojection. Autre modele : distorsion de Brown-Conrady (OpenCV).
    """
    Q = np.asarray(Q, float).reshape(-1, 3)
    z = np.where(np.abs(Q[:, 2]) < 1e-9, 1e-9, Q[:, 2])
    if np.any(vue.coeffs != 0) and "inverse" not in vue.modele:
        uv, _ = cv2.projectPoints(Q.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                  vue.K, vue.coeffs)
        return uv.reshape(-1, 2)
    K = vue.K
    return np.stack([K[0, 0] * Q[:, 0] / z + K[0, 2],
                     K[1, 1] * Q[:, 1] / z + K[1, 2]], axis=1)


def rayons(uv: np.ndarray, vue: VueCouleur) -> np.ndarray:
    """Pixels (N x 2) -> directions (N x 3, z = 1) dans le repere de la vue."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    K, c = vue.K, vue.coeffs
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    if np.any(c != 0):
        if "inverse" in vue.modele:            # comme rs2_deproject_pixel_to_point
            r2 = x * x + y * y
            f = 1 + c[0] * r2 + c[1] * r2 * r2 + c[4] * r2 * r2 * r2
            x, y = (x * f + 2 * c[2] * x * y + c[3] * (r2 + 2 * x * x),
                    y * f + 2 * c[3] * x * y + c[2] * (r2 + 2 * y * y))
        else:
            n = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, c).reshape(-1, 2)
            x, y = n[:, 0], n[:, 1]
    return np.stack([x, y, np.ones_like(x)], axis=1)


def projeter_profondeur(profondeur: np.ndarray, K: np.ndarray,
                        vue: VueCouleur) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ou chaque pixel de PROFONDEUR se voit dans l'image de la vue : (u, v, ok),
    tableaux H x W. Le pixel devient un point 3D (stenope, repere de la
    profondeur), passe dans le repere de la vue (T_cp) et s'y projette.
    `ok` : mesure valide, devant la camera, dans l'image.
    """
    P = vision.nuage(np.asarray(profondeur, float), K)
    H, W = profondeur.shape
    Q = P.reshape(-1, 3) @ vue.T_cp[:3, :3].T + vue.T_cp[:3, 3]
    uv = projeter(Q, vue)
    u = uv[:, 0].reshape(H, W)
    v = uv[:, 1].reshape(H, W)
    Wc, Hc = vue.taille
    ok = ((profondeur > 0) & (profondeur < vision.PROFONDEUR_MAX)
          & (Q[:, 2].reshape(H, W) > 0)
          & (u >= 0) & (u <= Wc - 1) & (v >= 0) & (v <= Hc - 1))
    return u, v, ok


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

@dataclass
class Table:
    """Le plan de la table vu de la camera de profondeur."""
    T_ct: np.ndarray              # repere table vu de la camera (4x4)
    n: np.ndarray
    d: float
    hauteur: np.ndarray           # H x W, m au-dessus de la table (corrigee)
    valides: np.ndarray           # H x W, pixels mesures
    P: np.ndarray                 # H x W x 3, repere camera (m)
    bruit: float                  # m, ecart-type robuste de la table

    @property
    def T_tc(self) -> np.ndarray:
        return np.linalg.inv(self.T_ct)

    def plan_dans(self, vue: VueCouleur) -> np.ndarray:
        """Le plan (n, d) exprime dans le repere de la vue : n.p + d = 0."""
        R, t = vue.T_cp[:3, :3], vue.T_cp[:3, 3]
        n = R @ self.n
        return np.r_[n, self.d - n @ t]


def analyser_table(profondeur: np.ndarray, K: np.ndarray) -> Table:
    """Plan RANSAC + correction de courbure, comme vision.detecter."""
    D = np.asarray(profondeur, float).copy()
    D[D > vision.PROFONDEUR_MAX] = 0.0
    valides = D > 0
    P = vision.nuage(D, K)
    n, d, sur_plan = vision.plan_table(P[valides])
    hauteur = np.zeros(D.shape)
    hauteur[valides] = P[valides] @ n + d
    masque_plan = np.zeros(D.shape, np.uint8)
    masque_plan[valides] = sur_plan.astype(np.uint8)
    nb, lab, st, _ = cv2.connectedComponentsWithStats(masque_plan)
    table = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone = np.zeros_like(table)
    cv2.drawContours(zone, [cv2.convexHull(max(contours, key=cv2.contourArea))],
                     -1, 1, -1)
    T_ct = vision.repere_table(n, d)
    hauteur = vision.corriger_courbure(hauteur, P, np.linalg.inv(T_ct),
                                       valides & (zone > 0))
    h_t = hauteur[(table > 0) & valides]
    bruit = 1.4826 * float(np.median(np.abs(h_t - np.median(h_t)))) if h_t.size else 0.002
    return Table(T_ct, n, d, hauteur, valides, P, bruit)


# ---------------------------------------------------------------------------
# Silhouette d'une piece et ajustement a un masque
# ---------------------------------------------------------------------------

def pose_table(x: float, y: float, theta: float, h: float) -> np.ndarray:
    """Repere de la face superieure (x le long du grand cote), repere table ;
    x, y, h en metres."""
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, h]
    return T


def coins(T_t: np.ndarray, L: float, l: float, h: float) -> np.ndarray:
    """Les 8 coins (m, repere table) du parallelepipede dont T_t est le
    repere de la face superieure ; L, l, h en metres."""
    a = np.array([[sx * L / 2, sy * l / 2, sz] for sx in (-1, 1)
                  for sy in (-1, 1) for sz in (0.0, -h)])
    return a @ T_t[:3, :3].T + T_t[:3, 3]


def silhouette(T_t: np.ndarray, dims_mm: Sequence[float], table: Table,
               vue: VueCouleur) -> np.ndarray:
    """Contour (polygone convexe, pixels int32) de la piece dans l'image de la
    vue : enveloppe convexe des 8 coins projetes (un parallelepipede est
    convexe, sa projection aussi)."""
    L, l, h = (float(v) / 1000.0 for v in dims_mm)
    T = vue.T_cp @ table.T_ct
    q = coins(T_t, L, l, h)
    q = q @ T[:3, :3].T + T[:3, 3]
    uv = projeter(q, vue)
    return cv2.convexHull(np.round(uv).astype(np.int32)).reshape(-1, 2)


@dataclass
class Ajustement:
    type: str
    face: str
    dims: Tuple[float, float, float]      # mm, du modele
    T_table: np.ndarray
    iou: float
    h_mesure: float                       # mm, nan si pas de profondeur
    n_points: int
    score: float
    contour: np.ndarray
    base: float = 0.0                     # mm, hauteur de la face d'appui


class _Noteur:
    """IoU entre le masque et la silhouette d'une pose, dans une fenetre
    autour du masque. Les pixels `ignorer` (autres pieces) ne comptent ni
    pour l'intersection ni pour l'union."""

    def __init__(self, masque: np.ndarray, ignorer: Optional[np.ndarray],
                 table: Table, vue: VueCouleur, marge: int = 40):
        ys, xs = np.nonzero(masque)
        H, W = masque.shape
        self.x0, self.y0 = max(int(xs.min()) - marge, 0), max(int(ys.min()) - marge, 0)
        x1, y1 = min(int(xs.max()) + marge + 1, W), min(int(ys.max()) + marge + 1, H)
        self.M = masque[self.y0:y1, self.x0:x1].astype(bool)
        self.garde = np.ones_like(self.M) if ignorer is None else \
            ~ignorer[self.y0:y1, self.x0:x1].astype(bool) | self.M
        self.aire = int(np.count_nonzero(self.M & self.garde))
        self.toile = np.zeros(self.M.shape, np.uint8)
        self.T = vue.T_cp @ table.T_ct
        self.vue = vue

    def iou(self, x, y, th, L, l, h, base: float = 0.0) -> float:
        q = coins(pose_table(x, y, th, base + h), L, l, h)
        q = q @ self.T[:3, :3].T + self.T[:3, 3]
        if np.any(q[:, 2] <= 0):
            return 0.0
        uv = projeter(q, self.vue) - [self.x0, self.y0]
        if not np.all(np.isfinite(uv)) or np.abs(uv).max() > 1e5:
            return 0.0
        self.toile[:] = 0
        cv2.fillConvexPoly(self.toile, cv2.convexHull(np.round(uv * 4).astype(np.int32)),
                           1, cv2.LINE_8, 2)
        S = self.toile.astype(bool) & self.garde
        inter = int(np.count_nonzero(S & self.M))
        union = self.aire + int(np.count_nonzero(S)) - inter
        return inter / union if union else 0.0

    def chercher(self, x, y, th, L, l, h, pas=(0.006, 0.006, np.radians(10)),
                 fin=(0.0003, 0.0003, np.radians(0.5)), base: float = 0.0):
        """Recherche par motifs (coordonnee par coordonnee, pas divises par
        deux quand rien ne s'ameliore) : sans gradient, robuste aux plateaux
        d'un score en nombre de pixels."""
        p = np.array([x, y, th], float)
        pas = np.array(pas, float)
        fin = np.array(fin, float)
        meilleur = self.iou(*p, L, l, h, base)
        for _ in range(200):
            ameliore = False
            for k in range(3):
                for sgn in (1, -1):
                    q = p.copy()
                    q[k] += sgn * pas[k]
                    s = self.iou(*q, L, l, h, base)
                    if s > meilleur + 1e-6:
                        p, meilleur, ameliore = q, s, True
                        break
            if not ameliore:
                pas /= 2
                if np.all(pas < fin):
                    break
        return p, meilleur


def _sur_plan(uv: np.ndarray, vue: VueCouleur, table: Table, h: float) -> np.ndarray:
    """Pixels de la vue -> points (x, y) du repere table, sur le plan parallele
    a la table a la hauteur h (m)."""
    r = rayons(uv, vue)
    T = np.linalg.inv(vue.T_cp @ table.T_ct)      # vue -> table
    o = T[:3, 3]
    dirs = r @ T[:3, :3].T
    t = (h - o[2]) / np.where(np.abs(dirs[:, 2]) < 1e-9, -1e-9, dirs[:, 2])
    return (o + dirs * t[:, None])[:, :2]


def _rectangle(xy: np.ndarray) -> Tuple[float, float, float]:
    """Centre et orientation du grand cote du rectangle minimal (m, rad)."""
    (cx, cy), (a, b), ang = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
    th = np.radians(ang if a >= b else ang + 90.0)
    return cx / 1000.0, cy / 1000.0, th


def bases_de_pile(types: Dict[str, Sequence[float]], etages: int = 2) -> List[float]:
    """Hauteurs ou une piece peut etre posee (mm) : la table, et le sommet
    d'une pile de 1 a `etages` pieces couchees (a plat ou sur la tranche).
    Voir vision_modele.bases_de_pile."""
    hs = sorted({round(forme(d, f)[2], 1) for d in types.values()
                 for f in ("a plat", "sur la tranche")})
    niveaux, courant = {0.0}, {0.0}
    for _ in range(etages):
        courant = {round(b + h, 1) for b in courant for h in hs}
        niveaux |= courant
    return sorted(niveaux)


def bases_autour(masque: np.ndarray, table: Table, proj, possibles: Sequence[float],
                 ignorer: Optional[np.ndarray] = None, part: float = 0.10) -> List[float]:
    """
    Les bases (mm) que la profondeur AUTOUR du masque rend plausibles : une
    piece posee sur une autre est entouree, en partie, par la face
    superieure de sa porteuse, qui depasse. On lit la hauteur des points de
    profondeur dans un anneau de 4 a 14 pixels autour du masque (les
    autres masques compris : la porteuse en a souvent un), seulement les
    points d'une surface plate, et une base est retenue si au moins `part`
    de ces points sont a sa hauteur, a 2.5 mm pres. La table (0) est
    toujours possible.
    """
    u, v, ok = proj
    m8 = masque.astype(np.uint8)
    anneau = (cv2.dilate(m8, np.ones((29, 29), np.uint8)) > 0) & \
        ~(cv2.dilate(m8, np.ones((9, 9), np.uint8)) > 0)
    dedans = np.zeros_like(ok)
    dedans[ok] = anneau[np.round(v[ok]).astype(np.int32), np.round(u[ok]).astype(np.int32)]
    # seulement des points PLATS (hauteur a 3 mm pres sur 5 x 5 pixels) : le
    # bord lisse de toute piece passe par toutes les hauteurs
    H = np.where(table.valides, table.hauteur, np.nan).astype(np.float32)
    noyau = np.ones((5, 5), np.uint8)
    etendue = (cv2.dilate(np.nan_to_num(H, nan=-1.0), noyau)
               - cv2.erode(np.nan_to_num(H, nan=1.0), noyau))
    dedans &= table.valides & (etendue < 0.003)
    h = table.hauteur[dedans] * 1000.0
    sortie = [0.0]
    if len(h) < 30:
        return sortie
    for b in possibles:
        if b > 0 and np.mean(np.abs(h - b) < 2.5) >= part:
            sortie.append(float(b))
    return sortie


def ajuster_masque(masque: np.ndarray, table: Table, vue: VueCouleur,
                   proj: Tuple[np.ndarray, np.ndarray, np.ndarray],
                   types: Dict[str, Sequence[float]], classe: Optional[str] = None,
                   ignorer: Optional[np.ndarray] = None,
                   debout: bool = True,
                   bases: Optional[Sequence[float]] = None,
                   sommet: Optional[float] = None) -> Optional[Ajustement]:
    """
    La piece de taille connue qui explique le mieux un masque de l'image de
    la vue (voir l'en-tete du module). `proj` : projeter_profondeur(...).
    `classe` : le type annonce par le reseau (simple preference).

    `bases` : hauteurs (mm) ou la piece peut etre posee. None : la table,
    et les sommets de piles que la profondeur autour du masque rend
    plausibles (bases_autour). La hauteur mesuree de la face superieure
    vaut base + epaisseur : c'est l'EPAISSEUR qui dit le type et la face.
    A egalite, la table (une pile coute 3 % du score). Une hauteur mesuree
    qu'aucune piece posee sur la table n'atteint a 5 mm pres (25, 34 mm)
    s'explique sans doute par une pile : toutes les bases sont alors essayees,
    meme si la porteuse ne depasse pas. `sommet` (mm) : n'essayer que les
    poses dont la face superieure est a cette hauteur (la porteuse d'une
    piece deja vue, dont on connait la base).
    """
    masque = masque.astype(bool)
    if masque.sum() < 30:
        return None
    u, v, ok = proj
    # points de profondeur DANS le masque erode : hauteur de la piece
    erode = cv2.erode(masque.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if erode.sum() < 20:
        erode = masque
    if ignorer is not None:
        # pas les points des autres pieces (une piece posee dessus, ou celle
        # qui la porte, dont le bord du masque deborde un peu)
        garde = erode & ~(cv2.dilate(ignorer.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if garde.sum() >= 20:
            erode = garde
    dedans = np.zeros_like(ok)
    dedans[ok] = erode[np.round(v[ok]).astype(np.int32), np.round(u[ok]).astype(np.int32)]
    h_pts = table.hauteur[dedans]
    sel = (h_pts > max(0.002, 2.5 * table.bruit)) & (h_pts < 0.10)
    h_pts = h_pts[sel]
    P_t = table.P[dedans][sel] @ table.T_tc[:3, :3].T + table.T_tc[:3, 3]
    h_mes = float(np.percentile(h_pts, 90)) * 1000.0 if len(h_pts) >= 15 else float("nan")
    # PLAT : la profondeur est mesuree dans le masque, mais au niveau de la
    # table (motif de la nappe, papier, ombre, reflet) : pas une piece. Une
    # piece SATUREE, elle, n'a pas de mesure du tout : elle n'est pas rejetee.
    n_mesures = int(dedans.sum())
    if n_mesures >= 30 and len(h_pts) < 0.25 * n_mesures:
        return None

    if bases is None:
        possibles = bases_de_pile(types)
        bases = bases_autour(masque, table, proj, possibles, ignorer)
        if np.isfinite(h_mes) and min(abs(forme(d, f)[2] - h_mes) for d in types.values()
                                      for f in FACES) >= 5.0:
            bases = sorted(set(bases) | set(possibles))
    if sommet is not None:
        bases = bases_de_pile(types)
    modeles = []
    for b in bases:
        for nom, dims in types.items():
            for face in FACES if debout else FACES[:2]:
                if b > 0 and face == "debout":
                    continue
                modeles.append((nom, face, forme(dims, face), float(b)))
    if sommet is not None:
        modeles = [m for m in modeles if abs(m[3] + m[2][2] - sommet) < 0.6]
    elif np.isfinite(h_mes):
        # la piece sur la table a 8 mm pres (la profondeur lisse les pieces
        # etroites) ; empilee, a 5 mm pres (les hauteurs sont serrees)
        proches = [m for m in modeles
                   if abs(m[3] + m[2][2] - h_mes) < (8.0 if m[3] == 0 else 5.0)]
        modeles = proches or [m for m in modeles if m[3] == 0]
    else:
        modeles = [m for m in modeles if m[3] == 0]

    noteur = _Noteur(masque, ignorer, table, vue)
    ys, xs = np.nonzero(masque)
    pix = np.stack([xs, ys], axis=1).astype(float)
    if len(pix) > 3000:
        pix = pix[np.linspace(0, len(pix) - 1, 3000).astype(int)]
    meilleur: Optional[Ajustement] = None
    for nom, face, (L, l, h), b in modeles:
        Lm, lm, hm, bm = L / 1000.0, l / 1000.0, h / 1000.0, b / 1000.0
        departs = []
        haut = P_t[P_t[:, 2] > bm + 0.5 * hm] if len(P_t) else P_t
        if len(haut) >= 20:
            departs.append(_rectangle(haut[:, :2]))
        departs.append(_rectangle(_sur_plan(pix, vue, table, bm + hm / 2)))
        essais = []
        for (x0, y0, t0) in departs:
            # quatre orientations de depart : un depart a 90 degres de la
            # bonne reste coince dans un optimum local (piece debout surtout)
            for dt in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
                essais.append(noteur.chercher(x0, y0, t0 + dt, Lm, lm, hm,
                                              fin=(0.002, 0.002, np.radians(3)), base=bm))
        p, _ = max(essais, key=lambda e: e[1])
        p, iou = noteur.chercher(*p, Lm, lm, hm, pas=(0.002, 0.002, np.radians(4)), base=bm)
        accord = 1.0
        if np.isfinite(h_mes):
            # la profondeur lisse les pieces etroites : tolerance large
            accord = float(np.exp(-0.5 * ((h_mes - (b + h)) / 5.0) ** 2))
        score = iou * (0.5 + 0.5 * accord)
        if classe and nom != classe:
            score *= 0.8
        if b > 0:
            score *= 0.97
        if meilleur is None or score > meilleur.score:
            th = (p[2] + np.pi / 2) % np.pi - np.pi / 2
            T_t = pose_table(p[0], p[1], th, bm + hm)
            meilleur = Ajustement(nom, face, (L, l, h), T_t, iou, h_mes,
                                  int(len(h_pts)), score,
                                  silhouette(T_t, (L, l, h), table, vue), base=float(b))
    return meilleur


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

@dataclass
class PieceVue(vision.Piece):
    """vision.Piece, plus : confiance (0 a 1), masque (image de la vue),
    T_robot (si l'etalonnage T_robot_camera est fourni), IoU masque /
    silhouette. `contour` est ici dans l'image de la VUE (couleur brute ou
    recalee), et `dimensions` sont celles du modele, sauf la hauteur,
    mesuree par la profondeur quand il y en a (epaisseur : sommet moins
    base). Empilement (champs de vision.Piece) : `support` (numero de la
    porteuse, None : la table, ou un objet non identifie si hauteur_base >
    0), `hauteur_base` (mm), `dessus` (numeros des pieces posees dessus) ;
    ordre_de_prise(pieces) dit dans quel ordre les saisir."""
    confiance: float = 0.0
    masque: Optional[np.ndarray] = None
    T_robot: Optional[np.ndarray] = None
    iou: float = float("nan")
    origine: str = ""                   # "reseau", "geometrie", "raffinement:..."
    # remplis par l'affinage (raffinement.py) : inclinaison du grand cote
    # sur la table (deg ; 0 : a plat), et numeros de TOUTES les pieces sur
    # lesquelles elle s'appuie (`support` : la principale)
    inclinaison: float = 0.0
    appuis: List[int] = field(default_factory=list)

    @property
    def position_robot(self) -> Optional[np.ndarray]:
        return None if self.T_robot is None else self.T_robot[:3, 3] * 1000.0


def _detecteur_geometrique():
    """vision_modele s'il existe et s'importe, sinon vision."""
    try:
        import vision_modele
        return vision_modele
    except Exception:
        return vision


class Perception:
    """
    Detecte et localise les pieces. Charge le modele entraine
    (modeles/pieces-seg.pt) s'il existe ; sinon, ou si la capture n'a pas
    d'image couleur, detection geometrique seule.

    types           {"nom": (a, b, c) mm} ; les classes du reseau doivent
                    porter ces noms
    modele          chemin d'un autre poids YOLO-seg ; False : jamais de reseau
    T_robot_camera  4x4 (m), repere de la camera de profondeur -> robot
    confiance_min   seuil du reseau ; iou_min : seuil de la verification
    raffinement     etape finale "rendre et comparer" de toute la scene
                    (raffinement.py : pieces inclinees, occultations,
                    physique) ; False : les detections telles quelles
    propositions_geometriques
                    avec le reseau, ajouter aussi les pieces de la detection
                    geometrique (vision_modele) aux propositions : la scene
                    ne depend plus du reseau seul (fond inconnu)
    points_cles     True (ou chemin de poids) : ajouter aux propositions de
                    l'affinage les poses du reseau de points cles
                    (apprentissage/points_cles, modeles/pieces-cles.pt)
    reglages        raffinement.Reglages (poids, seuils), ou None
    """

    def __init__(self, types: Optional[Dict[str, Sequence[float]]] = None,
                 modele: Union[str, bool, None] = None,
                 T_robot_camera: Optional[np.ndarray] = None,
                 confiance_min: float = 0.35, iou_min: float = 0.45,
                 device: Optional[str] = None, imgsz: Optional[int] = None,
                 raffinement: bool = True, propositions_geometriques: bool = True,
                 points_cles: Union[str, bool] = False, reglages=None):
        self.types = {k: tuple(float(x) for x in v) for k, v in (types or TYPES).items()}
        self.T_robot_camera = None if T_robot_camera is None else np.asarray(T_robot_camera, float)
        self.confiance_min, self.iou_min = confiance_min, iou_min
        self.device = device
        self.imgsz = imgsz
        self.reseau = None
        self.chemin_modele = None
        chemin = MODELE if modele is None else modele
        if chemin and os.path.exists(chemin):
            from ultralytics import YOLO
            self.reseau = YOLO(chemin)
            self.chemin_modele = chemin
            inconnues = [n for n in self.reseau.names.values() if n not in self.types]
            if inconnues:
                print("perception : classes du modele absentes des types : {} "
                      "(ignorees)".format(inconnues), file=sys.stderr)
            if self.imgsz is None:
                self.imgsz = int(getattr(self.reseau, "overrides", {}).get("imgsz", 864) or 864)
        elif modele:
            raise FileNotFoundError("modele introuvable : {}".format(modele))
        self.derniere_vue: Optional[VueCouleur] = None
        self.derniere_table: Optional[Table] = None
        self.durees: Dict[str, float] = {}
        self.raffinement = raffinement
        self.propositions_geometriques = propositions_geometriques
        self.points_cles = points_cles
        self._proposeur = None
        self.reglages = reglages
        self.derniers_masques: Optional[list] = None
        self.dernier_affinage = None

    @property
    def mode(self) -> str:
        return "reseau + geometrie" if self.reseau is not None else "geometrie seule"

    def _device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def detecter(self, capture=None, images: Optional[int] = None) -> List[PieceVue]:
        """Pieces vues. `capture` : None (capture fraiche par le serveur de
        camera), un chemin .npz, un dictionnaire ou une vision.Scene."""
        pieces: List[PieceVue] = []
        for _etape, resultat in self.detecter_par_etapes(capture, images):
            if resultat is not None:
                pieces = resultat
        return pieces

    #: part du temps de calcul de chaque etape, pour une barre de progression
    PROGRESSION = {"rapide": 0.15, "propositions": 0.60, "consolidee": 1.0}

    def detecter_par_etapes(self, capture=None, images: Optional[int] = None):
        """
        La meme detection, PAR ETAPES, pour un affichage qui se consolide :

            for etape, pieces in p.detecter_par_etapes(capture):
                ...   # pieces : None si l'etape n'a rien d'affichable

        "rapide"        reseau + ajustement des masques (< 1 s) ;
        "propositions"  pieces de la detection geometrique ajoutees aux
                        propositions (rien d'affichable : brutes) ;
        "consolidee"    apres l'affinage de la scene : le resultat final,
                        celui que rend detecter().
        Sans reseau, la detection geometrique tient lieu d'etape rapide.
        PROGRESSION donne la part du temps ecoulee a la fin de chaque etape.
        """
        t0 = time.perf_counter()
        if capture is None:
            import camera
            capture = camera.capturer(images or camera.N_IMAGES)
        capture = charger_capture(capture)
        t1 = time.perf_counter()
        vue = vue_couleur(capture)
        self.durees = {"capture": t1 - t0}
        self.derniers_masques = None
        if self.reseau is None or vue is None:
            pieces = self._geometrie(capture, vue)
            table = None
            yield "rapide", list(pieces)
        else:
            pieces = self._reseau(capture, vue)
            table = self.derniere_table
            yield "rapide", list(pieces)
            if self.raffinement and self.propositions_geometriques:
                # la geometrie propose aussi : une piece que le reseau
                # manque (fond inconnu) n'est pas perdue
                pieces = list(pieces) + self._geometrie(capture, vue)
                self.derniere_vue = vue
                yield "propositions", None
        if self.raffinement and self.points_cles and vue is not None:
            pieces = list(pieces) + self._propositions_cles(capture, vue, table)
        if self.raffinement:
            pieces = self._raffiner(capture, pieces, vue, table)
        self.durees["total"] = time.perf_counter() - t0
        yield "consolidee", pieces

    def _propositions_cles(self, capture, vue, table) -> List[PieceVue]:
        """Poses du reseau de points cles, en propositions pour l'affinage."""
        t = time.perf_counter()
        if self._proposeur is None:
            dossier = os.path.join(ICI, "apprentissage", "points_cles")
            if dossier not in sys.path:
                sys.path.insert(0, dossier)
            from proposer import POIDS, ProposeurCles
            chemin = self.points_cles if isinstance(self.points_cles, str) else POIDS
            self._proposeur = ProposeurCles(chemin, types=self.types, device=self.device)
        K = np.asarray(capture["K"], float)
        if table is None:
            table = analyser_table(capture["profondeur"], K)
        proj = projeter_profondeur(capture["profondeur"], K, vue)
        hyps = self._proposeur.proposer(vue.image, vue=vue, table=table, proj=proj)
        out = []
        for k, h in enumerate(hyps):
            try:
                out.append(h.piece(k + 1, table=table, vue=vue))
            except ValueError:
                pass
        self.durees["points_cles"] = time.perf_counter() - t
        return out

    def _raffiner(self, capture, pieces, vue, table) -> List[PieceVue]:
        """Etape finale : raffinement.raffiner (voir son en-tete)."""
        import raffinement as rf
        t = time.perf_counter()
        vue = vue or vue_couleur(capture, infrarouge=True)
        res, aff = rf.raffiner(capture, pieces, self.types, masques=self.derniers_masques,
                               vue=vue, reglages=self.reglages,
                               T_robot_camera=self.T_robot_camera, table=table,
                               rendre_affinage=True)
        self.dernier_affinage = aff
        self.derniere_vue = vue
        self.durees["raffinement"] = time.perf_counter() - t
        return res

    # -- avec le reseau ---------------------------------------------------
    def _reseau(self, capture, vue: VueCouleur) -> List[PieceVue]:
        t = time.perf_counter()
        table = analyser_table(capture["profondeur"], np.asarray(capture["K"], float))
        proj = projeter_profondeur(capture["profondeur"], np.asarray(capture["K"], float), vue)
        self.derniere_vue, self.derniere_table = vue, table
        self.durees["table"] = time.perf_counter() - t
        t = time.perf_counter()
        res = self.reseau.predict(vue.image, conf=self.confiance_min, imgsz=self.imgsz,
                                  device=self._device(), retina_masks=True,
                                  verbose=False)[0]
        self.durees["reseau"] = time.perf_counter() - t
        t = time.perf_counter()
        if res.masks is None:
            return []
        masques = res.masks.data.cpu().numpy() > 0.5
        if not len(masques):
            return []
        classes = [self.reseau.names[int(c)] for c in res.boxes.cls.cpu().numpy()]
        confs = res.boxes.conf.cpu().numpy()
        self.derniers_masques = [(masques[i], classes[i], float(confs[i]))
                                 for i in range(len(masques)) if classes[i] in self.types]
        ordre = np.argsort(-confs)
        pieces: List[PieceVue] = []
        ajustements = []
        for i in ordre:
            if classes[i] not in self.types:
                continue
            autres = (masques[np.arange(len(masques)) != i].any(axis=0)
                      if len(masques) > 1 else None)
            aj = ajuster_masque(masques[i], table, vue, proj, self.types,
                                classe=classes[i], ignorer=autres)
            if aj is None or aj.iou < self.iou_min:
                continue
            # doublon : deux masques pour la meme piece (au meme endroit ET a
            # la meme hauteur : une piece en travers sur une autre a presque
            # le meme centre vu de dessus)
            if any(np.linalg.norm(p.T_table[:2, 3] - aj.T_table[:2, 3]) < 0.008
                   and abs(p.T_table[2, 3] - aj.T_table[2, 3]) < 0.005 for p in pieces):
                continue
            pieces.append(self._piece(len(pieces) + 1, aj, float(confs[i]) * aj.iou,
                                      masques[i], table, "reseau"))
            ajustements.append((aj, i))
        pieces = self._piles(pieces, ajustements, masques, classes, confs, table, vue, proj)
        self.durees["ajustement"] = time.perf_counter() - t
        return pieces

    def _piles(self, pieces, ajustements, masques, classes, confs, table, vue, proj):
        """
        PILES. Une piece ajustee sur une base > 0 doit reposer sur une autre
        piece detectee, dont le sommet nominal est a cette hauteur et dont
        l'empreinte recouvre la sienne (vision_modele.relier_piles). Sinon :
        si sa hauteur s'explique aussi par une piece sur la table, elle est
        re-ajustee sur la table (pas de pile fantome) ; si non (34 mm), elle
        reste empilee, sur un objet non identifie (support None, hauteur_base
        > 0 : la porteuse est cachee, ou le reseau ne l'a pas vue). Enfin
        support et dessus sont remplis (numeros).
        """
        import vision_modele as vm
        h_table = [forme(d, f)[2] for d in self.types.values() for f in FACES]

        def decrire(p):
            L, l, _ = forme(self.types[p.type], p.face)
            h = round(p.T_table[2, 3] * 1000.0 - p.hauteur_base, 1)
            return (p.T_table[0, 3] * 1000, p.T_table[1, 3] * 1000,
                    np.arctan2(p.T_table[1, 0], p.T_table[0, 0]),
                    vm.Modele(p.type, p.face, L, l, h, p.hauteur_base))

        for k, p in enumerate(pieces):
            if p.hauteur_base <= 0:
                # sur la table, mais nettement plus haute que son modele :
                # peut-etre posee sur une piece vue sous elle
                if abs(p.dimensions[2] - forme(self.types[p.type], p.face)[2]) <= 4.0:
                    continue
            sup = vm.relier_piles([decrire(q) for q in pieces])[k]
            if p.hauteur_base <= 0:
                sup = -1
            dessous = []
            if sup == -1:
                a = decrire(p)
                for j, q in enumerate(pieces):
                    if j == k or q.T_table[2, 3] >= p.T_table[2, 3] - 0.004:
                        continue
                    b = decrire(q)
                    large = (b[0], b[1], b[2], vm.Modele(b[3].type, b[3].face, b[3].L + 10,
                                                         b[3].l + 10, b[3].h, b[3].base))
                    if vm.recouvrement(a, large) >= 0.08 * a[3].L * a[3].l:
                        dessous.append(round(q.T_table[2, 3] * 1000.0, 1))
            if sup == -1 and dessous:
                # d'abord : la poser sur le sommet d'une piece vue sous elle
                aj, i = ajustements[k]
                autres = (masques[np.arange(len(masques)) != i].any(axis=0)
                          if len(masques) > 1 else None)
                aj2 = ajuster_masque(masques[i], table, vue, proj, self.types,
                                     classe=classes[i], ignorer=autres,
                                     bases=sorted(set(dessous)))
                if aj2 is not None and aj2.iou >= self.iou_min and (
                        p.hauteur_base > 0 or aj2.score >= 0.9 * aj.score):
                    pieces[k] = p = self._piece(p.numero, aj2, float(confs[i]) * aj2.iou,
                                                masques[i], table, "reseau")
                    ajustements[k] = (aj2, i)
                sup = vm.relier_piles([decrire(q) for q in pieces])[k]
                if p.hauteur_base <= 0:
                    continue
            if sup == -1 and p.hauteur_base > 0:
                # sinon : une piece detectee sous elle, a une autre hauteur,
                # est la porteuse, en grande partie cachee, mal mesuree ; on
                # la re-ajuste avec son sommet impose (la base de celle-ci)
                a = decrire(p)
                for j, q in enumerate(pieces):
                    if j == k or q.T_table[2, 3] >= p.T_table[2, 3]:
                        continue
                    b = decrire(q)
                    large = (b[0], b[1], b[2], vm.Modele(b[3].type, b[3].face, b[3].L + 10,
                                                         b[3].l + 10, b[3].h, b[3].base))
                    if vm.recouvrement(a, large) < 0.08 * a[3].L * a[3].l:
                        continue
                    _, i = ajustements[j]
                    autres = (masques[np.arange(len(masques)) != i].any(axis=0)
                              if len(masques) > 1 else None)
                    aj2 = ajuster_masque(masques[i], table, vue, proj, self.types,
                                         classe=classes[i], ignorer=autres,
                                         sommet=p.hauteur_base)
                    if aj2 is not None and aj2.iou >= 0.8 * self.iou_min:
                        pieces[j] = self._piece(q.numero, aj2, float(confs[i]) * aj2.iou,
                                                masques[i], table, "reseau")
                        break
                sup = vm.relier_piles([decrire(q) for q in pieces])[k]
            haut = p.T_table[2, 3] * 1000.0
            if sup == -1 and p.hauteur_base > 0 and min(abs(haut - h) for h in h_table) < 4.0:
                aj, i = ajustements[k]
                autres = (masques[np.arange(len(masques)) != i].any(axis=0)
                          if len(masques) > 1 else None)
                aj2 = ajuster_masque(masques[i], table, vue, proj, self.types,
                                     classe=classes[i], ignorer=autres, bases=(0.0,))
                if aj2 is not None and aj2.iou >= self.iou_min:
                    pieces[k] = self._piece(p.numero, aj2, float(confs[i]) * aj2.iou,
                                            masques[i], table, "reseau")
        sup = vm.relier_piles([decrire(q) for q in pieces])
        for k, p in enumerate(pieces):
            p.support = None if sup[k] in (None, -1) else pieces[sup[k]].numero
            p.dessus = [pieces[j].numero for j in range(len(pieces)) if sup[j] == k]
        return pieces

    def _piece(self, numero, aj: Ajustement, confiance, masque, table: Table,
               origine) -> PieceVue:
        L, l, h = aj.dims
        # epaisseur mesuree : hauteur du sommet moins celle de la base
        h_mes = aj.h_mesure - aj.base if np.isfinite(aj.h_mesure) else h
        T_c = table.T_ct @ aj.T_table
        return PieceVue(numero, aj.face, (L, l, h_mes), abs(h_mes - h), T_c,
                        aj.T_table, aj.contour, int(masque.sum()), aj.type,
                        confiance=float(np.clip(confiance, 0, 1)), masque=masque,
                        T_robot=(None if self.T_robot_camera is None
                                 else self.T_robot_camera @ T_c),
                        iou=aj.iou, origine=origine, hauteur_base=aj.base)

    # -- repli geometrique --------------------------------------------------
    def _geometrie(self, capture, vue: Optional[VueCouleur]) -> List[PieceVue]:
        t = time.perf_counter()
        module = _detecteur_geometrique()
        scene = scene_de(capture)
        res = module.detecter(scene, self.types)
        self.durees["geometrie"] = time.perf_counter() - t
        vue = vue or vue_couleur(capture, infrarouge=True)
        table = Table(res.T_camera_table, res.normale, res.distance_table,
                      res.hauteur, scene.profondeur > 0,
                      vision.nuage(scene.profondeur, scene.K), 0.0)
        self.derniere_vue, self.derniere_table = vue, table
        pieces = []
        for p in res.pieces:
            dims = forme(self.types[p.type], p.face) if p.type in self.types else p.dimensions
            contour = (silhouette(p.T_table, dims, table, vue) if vue is not None
                       else p.contour)
            masque = None
            if vue is not None:
                masque = np.zeros(vue.image.shape[:2], np.uint8)
                cv2.fillConvexPoly(masque, contour.astype(np.int32), 1)
                masque = masque > 0
            conf = getattr(p, "score", None)
            if conf is None:
                conf = max(0.0, 1.0 - p.ecart / 10.0)
            pieces.append(PieceVue(p.numero, p.face, p.dimensions, p.ecart,
                                   p.T_camera, p.T_table, contour, p.pixels,
                                   p.type, support=getattr(p, "support", None),
                                   hauteur_base=getattr(p, "hauteur_base", 0.0),
                                   dessus=list(getattr(p, "dessus", [])),
                                   confiance=float(conf), masque=masque,
                                   T_robot=(None if self.T_robot_camera is None
                                            else self.T_robot_camera @ p.T_camera),
                                   origine="geometrie"))
        return pieces

    def dessiner(self, pieces: Sequence[PieceVue], image: Optional[np.ndarray] = None) -> np.ndarray:
        """Image annotee de la derniere detection (masques, silhouettes,
        type, face, position table en mm, angle, confiance)."""
        if image is None:
            if self.derniere_vue is None:
                raise ValueError("pas de detection a dessiner")
            image = self.derniere_vue.image
        return dessiner(image, pieces)


#: couleur de chaque type (BGR), dans l'ordre des types
PALETTE = [(40, 170, 255), (255, 200, 40), (90, 220, 90), (220, 90, 220),
           (60, 60, 255), (200, 200, 200)]


def couleur_type(nom: str, noms: Sequence[str]) -> Tuple[int, int, int]:
    noms = list(noms)
    i = noms.index(nom) if nom in noms else len(noms)
    return PALETTE[i % len(PALETTE)]


def dessiner(image: np.ndarray, pieces: Sequence[vision.Piece]) -> np.ndarray:
    """Masques (si presents) et silhouettes sur une copie de l'image."""
    img = image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    noms = sorted({p.type for p in pieces})
    for p in pieces:
        coul = np.array(couleur_type(p.type, noms), np.uint8)
        m = getattr(p, "masque", None)
        if m is not None and m.shape == img.shape[:2]:
            img[m] = (0.55 * img[m] + 0.45 * coul).astype(np.uint8)
    for p in pieces:
        coul = couleur_type(p.type, noms)
        cv2.polylines(img, [p.contour.astype(np.int32)], True, coul, 1, cv2.LINE_AA)
        u, v = p.contour.mean(axis=0).astype(int)
        x, y, z = p.position_table
        lignes = ["{} {} {}".format(p.numero, p.type, p.face),
                  "x{:.0f} y{:.0f} z{:.0f} {:.0f}deg".format(x, y, z, p.lacet)]
        if hasattr(p, "confiance"):
            lignes.append("conf {:.2f}".format(p.confiance))
        if getattr(p, "support", None) is not None:
            lignes.append("sur {}".format(p.support))
        elif getattr(p, "hauteur_base", 0.0) > 0:
            lignes.append("sur ? ({:.0f} mm)".format(p.hauteur_base))
        for k, texte in enumerate(lignes):
            org = (int(u) - 50, int(v) - 14 + 13 * k)
            cv2.putText(img, texte, org, cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 0, 0),
                        3, cv2.LINE_AA)
            cv2.putText(img, texte, org, cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detection des pieces (bibliotheque perception)")
    ap.add_argument("capture", nargs="?", help="fichier .npz (sinon : capture par le serveur)")
    ap.add_argument("--sortie", default=os.path.join(ICI, "captures", "derniere_perception.png"))
    ap.add_argument("--sans-reseau", action="store_true")
    a = ap.parse_args()
    p = Perception(modele=False if a.sans_reseau else None)
    pieces = p.detecter(a.capture)
    print("mode : {} ({:.2f} s : {})".format(p.mode, p.durees["total"], ", ".join(
        "{} {:.2f}".format(k, v) for k, v in p.durees.items() if k != "total")))
    for pc in pieces:
        x, y, z = pc.position_table
        print("  {} {:8s} {:15s} x {:6.1f} y {:6.1f} z {:5.1f} mm  {:5.0f} deg  "
              "conf {:.2f}".format(pc.numero, pc.type, pc.face, x, y, z, pc.lacet,
                                   pc.confiance))
    if p.derniere_vue is not None:
        os.makedirs(os.path.dirname(a.sortie), exist_ok=True)
        cv2.imwrite(a.sortie, p.dessiner(pieces))
        print("image :", a.sortie)
