# -*- coding: utf-8 -*-
"""
Scenes SYNTHETIQUES avec verite terrain, pour mesurer une methode de
detection.

Sur une capture reelle on ne sait pas ou sont les pieces au millimetre
pres : on ne peut que compter et juger a l'oeil. Ici on POSE des pieces
connues (type, face, position, orientation) sur une table virtuelle, on
calcule ce qu'une camera de profondeur en verrait, et on abime cette
image comme le fait une RealSense :

  - rendu exact par lancer de rayons : un rayon par pixel, intersection
    avec le plan de la table et avec chaque boite (methode des "dalles") ;
  - legere COURBURE de la profondeur vers les bords de l'image ;
  - LISSAGE des bords : la stereo agrege une fenetre de pixels, une arete
    franche devient une rampe (des "pixels volants" entre piece et table) ;
  - TROUS d'occultation : un point vu par la camera de gauche mais cache a
    la camera de droite (a `base` metres) n'a pas de mesure ; c'est la
    bande noire qui borde les objets d'un cote ;
  - BRUIT proportionnel a z^2 (l'erreur de disparite d'une stereo se
    traduit en erreur de profondeur proportionnelle au carre de la
    distance), en partie correle dans l'espace comme le vrai ;
  - quelques trous aleatoires, et eventuellement un reflet sature.

Les pieces peuvent etre EMPILEES (placer(..., empilees=k), poser_sur) :
une piece posee a plat sur la face superieure d'une autre, en travers,
dans le meme sens, decalee de cote, ou en tas de trois. Elle reste
horizontale (son centre est au-dessus de sa porteuse : elle ne bascule
pas). Elles peuvent aussi etre INCLINEES (placer(..., inclinees=k),
incliner) : un bout sur la table, l'autre en appui sur l'arete d'une
piece ("sur"), ou contre son flanc ("contre") ; c'est le cas courant d'un
tas reel. La verite dit, pour chaque piece, la hauteur de sa base (0 pour
une inclinee : son bout bas est sur la table), l'indice de sa porteuse
(None : la table ; pour une inclinee, la piece ou elle s'appuie) et son
inclinaison (tangage, degres).

Les images infrarouge et couleur sont grossieres (des aplats ombres) : elles
servent a exercer le code, pas a juger un indice de couleur fin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                # pragma: no cover
    cv2 = None

import vision

#: K des captures 640 x 480 (scene9) et 848 x 480 (env2)
K_640 = np.array([[379.2, 0.0, 319.0], [0.0, 379.2, 240.5], [0.0, 0.0, 1.0]])
K_848 = np.array([[418.7, 0.0, 422.9], [0.0, 418.7, 240.6], [0.0, 0.0, 1.0]])

#: types de pieces des captures (mm)
TYPES = {"epaisse": (17.0, 19.0, 68.0), "fine": (8.0, 17.0, 68.0)}

FACES = ("a plat", "sur la tranche", "debout")


def forme(dims: Sequence[float], face: str) -> Tuple[float, float, float]:
    """(long, large, haut) en mm d'un type pose sur une face : la hauteur
    est la petite dimension a plat, la moyenne sur la tranche, la grande
    debout."""
    a, b, c = sorted(dims)
    return {"a plat": (c, b, a), "sur la tranche": (c, a, b),
            "debout": (b, a, c)}[face]


@dataclass
class PieceVraie:
    type: str
    face: str
    x: float                # mm, centre de l'empreinte, repere monde (table)
    y: float
    lacet: float            # degres, direction du grand cote
    dims: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # long, large, haut
    contact: bool = False   # touche une autre piece
    z: float = 0.0          # mm, hauteur de la face d'appui (0 : sur la table)
    support: Optional[int] = None   # indice de la piece porteuse, None : la table
    # INCLINEE : tangage (deg, le bout +x leve), et hauteur du CENTRE de la
    # boite (mm) ; x, y sont alors la projection de ce centre
    tangage: float = 0.0
    centre_z: Optional[float] = None

    @property
    def inclinee(self) -> bool:
        return self.tangage != 0.0

    def rotation(self) -> np.ndarray:
        """Boite -> monde : lacet autour de z, puis tangage (x leve vers z)."""
        c, s = np.cos(np.radians(self.lacet)), np.sin(np.radians(self.lacet))
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if not self.inclinee:
            return Rz
        a = np.radians(self.tangage)
        Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
        return Rz @ Ry

    def centre(self) -> np.ndarray:
        """Centre de la boite (mm, monde)."""
        if self.inclinee:
            return np.array([self.x, self.y, self.centre_z])
        return np.array([self.x, self.y, self.z + self.dims[2] / 2.0])

    def centre_dessus(self) -> np.ndarray:
        """Centre de la face superieure (mm, monde)."""
        return self.centre() + self.rotation()[:, 2] * self.dims[2] / 2.0

    def coins3d(self) -> np.ndarray:
        """Les 8 coins de la boite (mm, monde)."""
        L, l, h = self.dims
        a = np.array([[sx * L / 2, sy * l / 2, sz * h / 2] for sx in (-1, 1)
                      for sy in (-1, 1) for sz in (-1, 1)])
        return a @ self.rotation().T + self.centre()

    @property
    def sommet(self) -> float:
        """Hauteur de la face superieure au-dessus de la table (mm) ; pour une
        piece inclinee, celle du centre de cette face."""
        if self.inclinee:
            return float(self.centre_dessus()[2])
        return self.z + self.dims[2]

    def coins(self) -> np.ndarray:
        """Les 4 coins de l'empreinte (mm, repere monde)."""
        L, l, _ = self.dims
        c, s = np.cos(np.radians(self.lacet)), np.sin(np.radians(self.lacet))
        R = np.array([[c, -s], [s, c]])
        loc = np.array([[-L, -l], [L, -l], [L, l], [-L, l]]) / 2.0
        return loc @ R.T + [self.x, self.y]


@dataclass
class Verite:
    """Ce que la detection devrait trouver, dans le repere CAMERA (m)."""
    pieces: List[PieceVraie]
    centres_camera: np.ndarray      # N x 3 : centre de la face superieure
    axes_camera: np.ndarray         # N x 3 : direction du grand cote
    R_cw: np.ndarray                # monde -> camera : p_c = R (p_w - C)
    C: np.ndarray                   # position de la camera, monde (m)
    objet: Optional[np.ndarray] = None   # H x W : indice de la piece vue par chaque pixel (< 0 : table, sol)

    def dessus(self, i: int) -> List[int]:
        """Indices des pieces posees directement sur la piece i."""
        return [k for k, p in enumerate(self.pieces) if p.support == i]


@dataclass
class Camera:
    K: np.ndarray = field(default_factory=lambda: K_640.copy())
    taille: Tuple[int, int] = (640, 480)     # largeur, hauteur
    distance: float = 0.50                   # m, de la camera au point vise
    inclinaison: float = 35.0                # deg, axe optique / normale
    roulis: float = 0.0                      # deg, autour de l'axe optique
    base: float = 0.050                      # m, seconde camera (occultations)


def pose_camera(cam: Camera) -> Tuple[np.ndarray, np.ndarray]:
    """(R_cw, C) : la camera vise l'origine de la table, inclinee de
    `inclinaison` par rapport a la verticale."""
    a = np.radians(cam.inclinaison)
    C = cam.distance * np.array([0.0, -np.sin(a), np.cos(a)])
    z = -C / np.linalg.norm(C)
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z])                  # lignes : axes camera, monde
    r = np.radians(cam.roulis)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    return Rz @ R, C


def rendre(K: np.ndarray, taille: Tuple[int, int], R_cw: np.ndarray,
           C: np.ndarray, pieces: Sequence[PieceVraie],
           table: Tuple[float, float, float, float] = (-0.4, 0.4, -0.3, 0.3),
           sol: float = -0.75) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Profondeur exacte (m) par lancer de rayons, l'indice de l'objet touche
    (-2 : sol, -1 : table, i : piece i) et la normale de la face touchee
    (monde). La table est un rectangle fini (`table`, m, repere monde) ;
    au-dela, le sol a `sol` m : la detection doit trouver le BORD de la
    table comme sur une vraie image.
    """
    W, H = taille
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    r_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                    np.ones_like(u)], axis=-1).reshape(-1, 3)
    d_w = r_c @ R_cw                         # R^T r : direction monde
    # r_c a une composante z = 1 : le parametre t du rayon EST la profondeur
    t = np.full(len(d_w), np.inf)
    objet = np.full(len(d_w), -3, np.int32)
    normale = np.zeros((len(d_w), 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_tab = -C[2] / d_w[:, 2]
        p = C + t_tab[:, None] * d_w
        sur_table = ((t_tab > 0) & (p[:, 0] >= table[0]) & (p[:, 0] <= table[1])
                     & (p[:, 1] >= table[2]) & (p[:, 1] <= table[3]))
        t[sur_table] = t_tab[sur_table]
        objet[sur_table] = -1
        normale[sur_table] = [0, 0, 1]
        t_sol = (sol - C[2]) / d_w[:, 2]
        m = (t_sol > 0) & (t_sol < t)
        t[m] = t_sol[m]
        objet[m] = -2
        normale[m] = [0, 0, 1]
    for i, pc in enumerate(pieces):
        L, l, h = (x / 1000.0 for x in pc.dims)
        if getattr(pc, "inclinee", False):
            # boite quelconque : repere centre sur la boite
            Rb = pc.rotation()
            o = Rb.T @ (C - pc.centre() / 1000.0)
            d = d_w @ Rb
            bas = np.array([-L / 2, -l / 2, -h / 2])
            haut = -bas
        else:
            c, s = np.cos(np.radians(pc.lacet)), np.sin(np.radians(pc.lacet))
            Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # boite -> monde
            o = Rb.T @ (C - [pc.x / 1000.0, pc.y / 1000.0, 0.0])
            d = d_w @ Rb                                          # Rb^T d
            z0 = pc.z / 1000.0
            bas = np.array([-L / 2, -l / 2, z0])
            haut = np.array([L / 2, l / 2, z0 + h])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (bas - o) / d
            t2 = (haut - o) / d
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        tmin = np.where(np.isnan(tmin), -np.inf, tmin)
        tmax = np.where(np.isnan(tmax), np.inf, tmax)
        t_entree = tmin.max(axis=1)
        axe = tmin.argmax(axis=1)
        t_sortie = tmax.min(axis=1)
        touche = (t_entree <= t_sortie) & (t_entree > 0) & (t_entree < t)
        t[touche] = t_entree[touche]
        objet[touche] = i
        n_loc = np.zeros((touche.sum(), 3))
        k = axe[touche]
        n_loc[np.arange(len(k)), k] = -np.sign(d[touche, k])
        normale[touche] = n_loc @ Rb.T
    t[~np.isfinite(t)] = 0.0
    return (t.reshape(H, W), objet.reshape(H, W), normale.reshape(H, W, 3))


@dataclass
class Bruit:
    """Defauts d'une RealSense, a regler pour durcir ou adoucir la scene."""
    k_z2: float = 0.0030          # ecart-type = k z^2 (m) : 0.75 mm a 0.5 m
    correle: float = 0.5          # part du bruit correlee dans l'espace
    lissage: int = 5              # fenetre (pixels) du lissage des aretes
    courbure: float = 0.004       # relative, au coin de l'image
    occultation: bool = True
    trous: float = 0.004          # part de l'image en petits trous aleatoires
    reflet: bool = False          # tache saturee (lampe) sur la table


def degrader(D: np.ndarray, K: np.ndarray, bruit: Bruit,
             rng: np.random.Generator,
             masque_occulte: Optional[np.ndarray] = None) -> np.ndarray:
    """Applique a la profondeur exacte les defauts d'une RealSense."""
    H, W = D.shape
    valide = D > 0
    Z = D.copy()
    if bruit.lissage > 1:
        # moyenne normalisee sur une fenetre : une arete devient une rampe
        k = (bruit.lissage, bruit.lissage)
        num = cv2.blur(Z * valide, k)
        den = cv2.blur(valide.astype(np.float64), k)
        Z = np.where(den > 0.3, num / np.maximum(den, 1e-9), 0.0)
        Z[~valide] = 0.0
    if bruit.courbure:
        v, u = np.mgrid[0:H, 0:W]
        r2 = ((u - K[0, 2]) ** 2 + (v - K[1, 2]) ** 2) / (K[0, 2] ** 2 + K[1, 2] ** 2)
        Z = Z * (1.0 + bruit.courbure * r2)
    sigma = bruit.k_z2 * Z ** 2
    blanc = rng.standard_normal((H, W))
    corr = cv2.GaussianBlur(rng.standard_normal((H, W)), (0, 0), 4)
    corr /= corr.std() + 1e-12
    Z = Z + sigma * (np.sqrt(1 - bruit.correle) * blanc
                     + np.sqrt(bruit.correle) * corr)
    Z[~valide] = 0.0
    if masque_occulte is not None:
        Z[masque_occulte] = 0.0
    if bruit.trous > 0:
        graines = rng.random((H, W)) < bruit.trous / 9.0
        trous = cv2.dilate(graines.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        Z[trous] = 0.0
    return Z.astype(np.float32)


def occultation(D: np.ndarray, K: np.ndarray, R_cw: np.ndarray, C: np.ndarray,
                pieces, base: float, taille) -> np.ndarray:
    """
    Pixels vus par la camera principale mais caches a la seconde, decalee de
    `base` le long de l'axe x de la camera : la stereo ne peut pas les
    apparier, ils n'ont pas de mesure.
    """
    C2 = C + base * R_cw[0]
    D2, _, _ = rendre(K, taille, R_cw, C2, pieces)
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    pc = np.dstack([(u - K[0, 2]) * D / K[0, 0], (v - K[1, 2]) * D / K[1, 1], D])
    pc2 = pc - np.array([base, 0.0, 0.0])          # repere de la seconde
    z2 = pc2[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u2 = np.round(K[0, 0] * pc2[..., 0] / z2 + K[0, 2]).astype(np.int64)
        v2 = np.round(K[1, 1] * pc2[..., 1] / z2 + K[1, 2]).astype(np.int64)
    dedans = (D > 0) & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    cache = np.zeros_like(dedans)
    z_vu = D2[v2[dedans], u2[dedans]]
    cache[dedans] = (z_vu > 0) & (z_vu < z2[dedans] - 0.003)
    return cache


def images(objet: np.ndarray, normale: np.ndarray, R_cw: np.ndarray,
           D: np.ndarray, rng: np.random.Generator, couleur: bool,
           motif: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Infrarouge (motif de points du projecteur) et couleur grossieres."""
    H, W = objet.shape
    # eclairage : une lumiere au-dessus, un peu de cote
    lum = np.array([0.3, -0.4, 0.86])
    lum /= np.linalg.norm(lum)
    ombre = np.clip(normale @ lum, 0.2, 1.0)
    ir = np.where(objet >= 0, 170.0, np.where(objet == -1, 95.0, 40.0)) * ombre
    points = (rng.random((H, W)) < 0.012).astype(np.float32)
    points = cv2.GaussianBlur(points, (0, 0), 1.0) * 600
    ir = np.clip(ir + points + rng.normal(0, 4, (H, W)), 0, 255).astype(np.uint8)
    if not couleur:
        return ir, None
    fond = np.array([215.0, 212.0, 222.0])          # BGR : nappe claire
    bois = np.array([150.0, 195.0, 228.0])
    img = np.zeros((H, W, 3))
    img[objet == -1] = fond
    img[objet == -2] = [70, 80, 90]
    img[objet >= 0] = bois
    if motif:
        # des "fleurs" : taches roses et vertes sur la nappe
        taches = np.zeros((H, W), np.float32)
        for _ in range(25):
            c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            cv2.ellipse(taches, c, (int(rng.integers(8, 30)), int(rng.integers(5, 15))),
                        float(rng.uniform(0, 180)), 0, 360,
                        float(rng.integers(1, 3)), -1)
        rose = (taches == 1) & (objet == -1)
        vert = (taches == 2) & (objet == -1)
        img[rose] = [120, 90, 210]
        img[vert] = [120, 170, 130]
    img = img * ombre[..., None] + rng.normal(0, 5, (H, W, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[D == 0] = 0                                   # comme le recalage reel
    return ir, img


#: facons d'empiler (voir empiler)
MODES_PILE = ("travers", "meme sens", "decalee", "tas")


def placer(rng: np.random.Generator, n: int, types: Dict[str, Sequence[float]],
           zone: Tuple[float, float, float, float] = (-110, 110, -80, 80),
           touchantes: int = 0, faces: Sequence[str] = ("a plat", "sur la tranche"),
           essais: int = 400, empilees: int = 0,
           modes: Sequence[str] = MODES_PILE, inclinees: int = 0,
           modes_inclines: Sequence[str] = ("sur", "contre")) -> List[PieceVraie]:
    """
    `n` pieces au hasard dans `zone` (mm, repere monde), sans chevauchement ;
    `touchantes` d'entre elles sont posees CONTRE une piece deja placee (cote
    a cote, bout a bout, ou en T), sans jeu. `empilees` d'entre elles sont
    posees SUR une autre (voir empiler) : les n - empilees premieres sont
    tirees exactement comme sans empilement, une scene sans empilement ne
    change donc pas. `inclinees` d'entre elles, tirees en dernier (meme
    principe), sont INCLINEES contre une piece deja posee (voir incliner).
    """
    n_table = n - empilees - inclinees
    noms = sorted(types)
    pieces: List[PieceVraie] = []
    occupe = np.zeros((int(zone[3] - zone[2]) + 80, int(zone[1] - zone[0]) + 80),
                      np.uint8)

    def masque(p: PieceVraie, marge: float) -> np.ndarray:
        L, l, h = p.dims
        q = PieceVraie(p.type, p.face, p.x, p.y, p.lacet, (L + 2 * marge, l + 2 * marge, h))
        m = np.zeros_like(occupe)
        pts = q.coins() - [zone[0] - 40, zone[2] - 40]
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        return m

    def dans_zone(p: PieceVraie) -> bool:
        c = p.coins()
        return bool((c[:, 0] >= zone[0]).all() and (c[:, 0] <= zone[1]).all()
                    and (c[:, 1] >= zone[2]).all() and (c[:, 1] <= zone[3]).all())

    a_toucher = touchantes
    for _ in range(essais):
        if len(pieces) >= n_table:
            break
        t = noms[int(rng.integers(len(noms)))]
        f = faces[int(rng.integers(len(faces)))]
        dims = forme(types[t], f)
        if a_toucher > 0 and pieces:
            # contre une piece existante
            a = pieces[int(rng.integers(len(pieces)))]
            mode = int(rng.integers(3))
            ca, sa = np.cos(np.radians(a.lacet)), np.sin(np.radians(a.lacet))
            ax, no = np.array([ca, sa]), np.array([-sa, ca])
            signe = 1 if rng.random() < 0.5 else -1
            if mode == 0:        # cote a cote
                lac = a.lacet
                c = (np.array([a.x, a.y]) + signe * no * (a.dims[1] + dims[1]) / 2
                     + ax * rng.uniform(-15, 15))
            elif mode == 1:      # bout a bout
                lac = a.lacet
                c = (np.array([a.x, a.y]) + signe * ax * (a.dims[0] + dims[0]) / 2
                     + no * rng.uniform(-4, 4))
            else:                # en T : le bout de la nouvelle contre le flanc
                lac = a.lacet + 90.0
                c = (np.array([a.x, a.y]) + signe * no * (a.dims[1] + dims[0]) / 2
                     + ax * rng.uniform(-20, 20))
            p = PieceVraie(t, f, float(c[0]), float(c[1]), float(lac), dims)
            marge = -0.6        # contact autorise
        else:
            p = PieceVraie(t, f, float(rng.uniform(zone[0], zone[1])),
                           float(rng.uniform(zone[2], zone[3])),
                           float(rng.uniform(-90, 90)), dims)
            marge = 6.0         # pieces bien separees
        if not dans_zone(p):
            continue
        m = masque(p, marge)
        if (m & occupe).any():
            continue
        if marge < 0:
            a_toucher -= 1
            p.contact = True
            a.contact = True
        occupe |= masque(p, 0.0)
        pieces.append(p)
    if empilees > 0:
        empiler(rng, pieces, empilees, types, zone, modes, faces, essais)
    if inclinees > 0:
        incliner(rng, pieces, inclinees, types, zone, modes_inclines, faces, essais)
    return pieces


def _se_coupent(a: PieceVraie, b: PieceVraie, marge: float = 0.5) -> bool:
    """Les empreintes (rectangles, vue de dessus) de a et b se recouvrent-elles
    de plus de `marge` mm^2 ?"""
    pa = a.coins().astype(np.float32)
    pb = b.coins().astype(np.float32)
    aire, _ = cv2.intersectConvexConvex(pa, pb)
    return aire > marge


def poser_sur(pieces: Sequence[PieceVraie], i: int, type_: str, face: str,
              types: Dict[str, Sequence[float]], dx: float = 0.0, dy: float = 0.0,
              lacet: Optional[float] = None) -> PieceVraie:
    """Une piece posee sur la piece d'indice i : centre decale de (dx, dy) mm
    (repere monde), lacet en degres (par defaut celui de la porteuse + 90,
    en travers). Sa base est le sommet de la porteuse."""
    a = pieces[i]
    lac = a.lacet + 90.0 if lacet is None else lacet
    return PieceVraie(type_, face, a.x + dx, a.y + dy, lac, forme(types[type_], face),
                      z=a.sommet, support=i)


def empiler(rng: np.random.Generator, pieces: List[PieceVraie], nombre: int,
            types: Dict[str, Sequence[float]],
            zone: Tuple[float, float, float, float] = (-110, 110, -80, 80),
            modes: Sequence[str] = MODES_PILE,
            faces: Sequence[str] = ("a plat", "sur la tranche"),
            essais: int = 400) -> int:
    """
    Pose `nombre` pieces SUR des pieces deja placees (ajoutees a `pieces`),
    horizontales, le centre au-dessus de la porteuse pour qu'elles ne
    basculent pas. Modes :
      travers   : a peu pres perpendiculaire, n'importe ou le long de la
                  porteuse ;
      meme sens : dans l'axe, glissee le long de la porteuse (elle en cache
                  une grande partie) ;
      decalee   : dans l'axe, glissee de COTE (la porteuse depasse d'un cote) ;
      tas       : en travers sur une piece qui est deja elle-meme sur une
                  autre (tas de trois).
    Une porteuse ne porte qu'une piece ; une piece debout ne porte rien. Pas
    d'interpenetration : deux pieces dont les empreintes se coupent ont des
    tranches de hauteur disjointes. Rend le nombre de pieces posees.
    """
    noms = sorted(types)
    pose = 0
    for _ in range(essais):
        if pose >= nombre:
            break
        mode = modes[int(rng.integers(len(modes)))]
        portees = {p.support for p in pieces if p.support is not None}
        libres = [k for k, p in enumerate(pieces) if k not in portees
                  and p.face != "debout"
                  and ((p.support is not None) == (mode == "tas"))
                  and (p.support is None or pieces[p.support].support is None)]
        if not libres:
            continue
        i = libres[int(rng.integers(len(libres)))]
        a = pieces[i]
        t = noms[int(rng.integers(len(noms)))]
        f = faces[int(rng.integers(len(faces)))]
        dims = forme(types[t], f)
        ca, sa = np.cos(np.radians(a.lacet)), np.sin(np.radians(a.lacet))
        ax, no = np.array([ca, sa]), np.array([-sa, ca])
        signe = 1 if rng.random() < 0.5 else -1
        if mode in ("travers", "tas"):
            lac = a.lacet + 90.0 + rng.uniform(-25, 25)
            cl, sl = np.cos(np.radians(lac)), np.sin(np.radians(lac))
            # le long de la porteuse, et le long de la piece posee : son
            # centre reste au-dessus de la porteuse
            d = ax * rng.uniform(-18, 18) + np.array([cl, sl]) * rng.uniform(-6, 6)
        elif mode == "meme sens":
            lac = a.lacet + rng.uniform(-8, 8)
            d = ax * signe * rng.uniform(5, 22) + no * rng.uniform(-2, 2)
        else:                                              # decalee
            lac = a.lacet + rng.uniform(-10, 10)
            d = (no * signe * rng.uniform(0.45, 0.85) * a.dims[1] / 2
                 + ax * rng.uniform(-12, 12))
        p = PieceVraie(t, f, float(a.x + d[0]), float(a.y + d[1]), float(lac), dims,
                       z=a.sommet, support=i)
        c = p.coins()
        if not ((c[:, 0] >= zone[0]).all() and (c[:, 0] <= zone[1]).all()
                and (c[:, 1] >= zone[2]).all() and (c[:, 1] <= zone[3]).all()):
            continue
        collision = False
        for k, q in enumerate(pieces):
            if k == i:
                continue
            # tranches de hauteur qui se recouvrent, et empreintes aussi
            if q.z < p.sommet - 0.5 and q.sommet > p.z + 0.5 and _se_coupent(p, q, 0.5):
                collision = True
                break
        if collision:
            continue
        pieces.append(p)
        pose += 1
    return pose


#: facons d'incliner (voir incliner)
MODES_INCLINES = ("sur", "contre")


def _boite_m(p: PieceVraie):
    """(centre, rotation, demi-dimensions) en metres, pour le test SAT."""
    return p.centre() / 1000.0, p.rotation(), np.asarray(p.dims, float) / 2000.0


def _penetre(p: PieceVraie, q: PieceVraie) -> float:
    """Profondeur d'interpenetration (mm) des boites p et q (0 : disjointes)."""
    from raffinement import penetration
    c1, R1, h1 = _boite_m(p)
    c2, R2, h2 = _boite_m(q)
    return 1000.0 * float(penetration(c1[None], R1[None], h1, c2, R2, h2)[0])


def _inclinee(t, f, dims, E, u, alpha, support) -> PieceVraie:
    """La piece dont l'arete basse du bout bas est en E (mm, sur la table),
    l'axe horizontal u (unitaire), inclinee de alpha (rad)."""
    L, _, h = dims
    a1 = np.array([u[0] * np.cos(alpha), u[1] * np.cos(alpha), np.sin(alpha)])
    n1 = np.array([-u[0] * np.sin(alpha), -u[1] * np.sin(alpha), np.cos(alpha)])
    c = np.array([E[0], E[1], 0.0]) + a1 * L / 2 + n1 * h / 2
    lac = float(np.degrees(np.arctan2(u[1], u[0])))
    return PieceVraie(t, f, float(c[0]), float(c[1]), lac, dims, z=0.0,
                      support=support, tangage=float(np.degrees(alpha)),
                      centre_z=float(c[2]))


def incliner(rng: np.random.Generator, pieces: List[PieceVraie], nombre: int,
             types: Dict[str, Sequence[float]],
             zone: Tuple[float, float, float, float] = (-110, 110, -80, 80),
             modes: Sequence[str] = MODES_INCLINES,
             faces: Sequence[str] = ("a plat", "sur la tranche"),
             essais: int = 400) -> int:
    """
    Pose `nombre` pieces INCLINEES (ajoutees a `pieces`), le bout bas sur la
    table. Modes :
      sur    : le dessous de la piece s'appuie sur l'arete superieure d'une
               piece couchee (en travers, ou sur son bout) ; l'inclinaison
               est la plus petite sans interpenetration (elle repose) ;
      contre : le bout haut s'appuie contre le FLANC d'une piece assez haute
               (le bout reste sous son sommet) ; la piece glisse vers la
               porteuse jusqu'au contact.
    Stabilite : le centre de gravite est entre les deux appuis (mode sur :
    le contact est au-dela de 55 % de la longueur). Pas d'interpenetration
    (test SAT entre boites). Rend le nombre de pieces posees.
    """
    noms = sorted(types)
    pose = 0
    for _ in range(essais):
        if pose >= nombre:
            break
        mode = modes[int(rng.integers(len(modes)))]
        porteuses = [k for k, p in enumerate(pieces) if not p.inclinee
                     and p.face != "debout" and p.sommet >= 7.5]
        if mode == "contre":
            porteuses = [k for k in porteuses if pieces[k].sommet >= 16.5]
        if not porteuses:
            continue
        i = porteuses[int(rng.integers(len(porteuses)))]
        a = pieces[i]
        t = noms[int(rng.integers(len(noms)))]
        f = faces[int(rng.integers(len(faces)))]
        dims = forme(types[t], f)
        L, l, h = dims
        ca, sa = np.cos(np.radians(a.lacet)), np.sin(np.radians(a.lacet))
        ax, no = np.array([ca, sa]), np.array([-sa, ca])
        signe = 1 if rng.random() < 0.5 else -1
        bout = mode == "sur" and rng.random() < 0.25
        if bout:
            # sur le BOUT de la porteuse, a peu pres dans son axe
            lac = np.radians(a.lacet + rng.uniform(-20, 20))
            u = -signe * np.array([np.cos(lac), np.sin(lac)])
            Q = np.array([a.x, a.y]) + signe * ax * a.dims[0] / 2
        else:
            lac = np.radians(a.lacet + 90.0 + rng.uniform(-30, 30))
            u = -signe * np.array([np.cos(lac), np.sin(lac)])
            Q = (np.array([a.x, a.y]) + signe * no * a.dims[1] / 2
                 + ax * rng.uniform(-0.3, 0.3) * a.dims[0])
        H = a.sommet
        if mode == "sur":
            s_c = rng.uniform(0.6, 0.85) * L
            alpha0 = np.arcsin(min(H / s_c, 0.95))
            E = Q - u * s_c * np.cos(alpha0)
            # plus petite inclinaison sans interpenetration : la piece repose
            lo, hi = np.radians(1.0), np.radians(65.0)
            if _penetre(_inclinee(t, f, dims, E, u, hi, i), a) > 0.05:
                continue
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if _penetre(_inclinee(t, f, dims, E, u, mid, i), a) > 0.05:
                    lo = mid
                else:
                    hi = mid
            alpha = hi
            if alpha < np.radians(3.0):
                continue
            # contact a la hauteur H le long du dessous : au-dela du milieu
            if H / np.sin(alpha) < 0.55 * L:
                continue
        else:
            amax = np.arcsin(min((H - 2.0) / L, 0.95))
            if amax < np.radians(5.0):
                continue
            alpha = rng.uniform(np.radians(5.0), 0.9 * amax)
            # le bout haut glisse vers le flanc jusqu'au contact
            lo, hi = 0.0, L
            # depart : le bout haut 6 mm DANS la porteuse
            E0 = Q - u * (L * np.cos(alpha) - 6.0)
            if _penetre(_inclinee(t, f, dims, E0 - u * hi, u, alpha, i), a) > 0.05:
                continue
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if _penetre(_inclinee(t, f, dims, E0 - u * mid, u, alpha, i), a) > 0.05:
                    lo = mid
                else:
                    hi = mid
            E = E0 - u * hi
            if hi < 0.5:
                continue
        p = _inclinee(t, f, dims, E, u, alpha, i)
        c = p.coins3d()
        if not ((c[:, 0] >= zone[0]).all() and (c[:, 0] <= zone[1]).all()
                and (c[:, 1] >= zone[2]).all() and (c[:, 1] <= zone[3]).all()):
            continue
        if c[:, 2].min() < -0.05:
            continue
        if any(_penetre(p, q) > 0.05 for q in pieces):
            continue
        pieces.append(p)
        pose += 1
    return pose


def scene(pieces: Sequence[PieceVraie], cam: Optional[Camera] = None,
          bruit: Optional[Bruit] = None, graine: int = 0,
          couleur: bool = False, motif: bool = False,
          table: Tuple[float, float, float, float] = (-0.4, 0.4, -0.3, 0.3)
          ) -> Tuple[vision.Scene, Verite]:
    """La capture simulee (meme format que vision.charger) et sa verite."""
    cam = cam or Camera()
    bruit = bruit or Bruit()
    rng = np.random.default_rng(graine)
    R_cw, C = pose_camera(cam)
    D, objet, normale = rendre(cam.K, cam.taille, R_cw, C, pieces, table)
    occ = occultation(D, cam.K, R_cw, C, pieces, cam.base, cam.taille) \
        if bruit.occultation else None
    Z = degrader(D, cam.K, bruit, rng, occ)
    ir, img = images(objet, normale, R_cw, Z, rng, couleur, motif)
    if bruit.reflet:
        H, W = Z.shape
        tache = np.zeros((H, W), np.uint8)
        cv2.ellipse(tache, (W // 2, int(H * 0.9)), (W // 5, H // 10), 0, 0, 360, 1, -1)
        Z[tache > 0] = 0.0
        ir[tache > 0] = 255
    centres, axes = [], []
    for p in pieces:
        if getattr(p, "inclinee", False):
            pw = p.centre_dessus() / 1000.0
            axes.append(R_cw @ p.rotation()[:, 0])
        else:
            pw = np.array([p.x / 1000.0, p.y / 1000.0, p.sommet / 1000.0])
            a = np.radians(p.lacet)
            axes.append(R_cw @ np.array([np.cos(a), np.sin(a), 0.0]))
        centres.append(R_cw @ (pw - C))
    sc = vision.Scene(profondeur=Z.astype(np.float64), K=cam.K.copy(),
                      infrarouge=ir, couleur=img)
    return sc, Verite(list(pieces), np.array(centres).reshape(-1, 3),
                      np.array(axes).reshape(-1, 3), R_cw, C, objet)


# ---------------------------------------------------------------------------
# Comparaison a la verite
# ---------------------------------------------------------------------------

@dataclass
class Bilan:
    vrais: int = 0              # pieces posees
    trouves: int = 0            # detections appariees a une piece
    faux: int = 0               # detections sans piece
    err_pos: List[float] = field(default_factory=list)   # mm, dans le plan
    err_z: List[float] = field(default_factory=list)     # mm, en hauteur (signe)
    err_ang: List[float] = field(default_factory=list)   # deg
    bon_type: int = 0
    bonne_face: int = 0
    vrais_fines: int = 0        # pieces etroites (8 mm de large ou de haut)
    trouves_fines: int = 0
    vrais_contact: int = 0      # pieces qui en touchent une autre
    trouves_contact: int = 0
    # empilements (voir comparer)
    vrais_empilees: int = 0     # pieces posees sur une autre
    trouves_empilees: int = 0
    vrais_porteuses: int = 0    # pieces qui en portent une autre
    trouves_porteuses: int = 0
    bon_support: int = 0        # appariees dont le support annonce est le bon
    bon_support_empilees: int = 0
    bon_type_face_empilees: int = 0
    err_pos_empilees: List[float] = field(default_factory=list)
    err_pos_porteuses: List[float] = field(default_factory=list)
    scenes: int = 0
    ordre_ok: int = 0           # scenes ou l'ordre de prise respecte les piles
    premiere_libre: int = 0     # scenes ou la premiere piece a prendre est libre
    # pieces INCLINEES, et angles separes (lacet autour de la verticale,
    # inclinaison du grand cote sur la table)
    vrais_inclinees: int = 0
    trouves_inclinees: int = 0
    bon_support_inclinees: int = 0
    err_lacet: List[float] = field(default_factory=list)       # deg
    err_incl: List[float] = field(default_factory=list)        # deg
    err_incl_inclinees: List[float] = field(default_factory=list)
    err_pos_inclinees: List[float] = field(default_factory=list)

    def ajouter(self, b: "Bilan") -> None:
        self.vrais += b.vrais
        self.trouves += b.trouves
        self.faux += b.faux
        self.err_pos += b.err_pos
        self.err_z += b.err_z
        self.vrais_fines += b.vrais_fines
        self.trouves_fines += b.trouves_fines
        self.vrais_contact += b.vrais_contact
        self.trouves_contact += b.trouves_contact
        self.err_ang += b.err_ang
        self.bon_type += b.bon_type
        self.bonne_face += b.bonne_face
        for nom in ("vrais_empilees", "trouves_empilees", "vrais_porteuses",
                    "trouves_porteuses", "bon_support", "bon_support_empilees",
                    "bon_type_face_empilees", "scenes", "ordre_ok", "premiere_libre",
                    "vrais_inclinees", "trouves_inclinees", "bon_support_inclinees"):
            setattr(self, nom, getattr(self, nom) + getattr(b, nom))
        self.err_pos_empilees += b.err_pos_empilees
        self.err_pos_porteuses += b.err_pos_porteuses
        self.err_lacet += b.err_lacet
        self.err_incl += b.err_incl
        self.err_incl_inclinees += b.err_incl_inclinees
        self.err_pos_inclinees += b.err_pos_inclinees

    def resume(self) -> Dict[str, float]:
        t = max(self.trouves, 1)
        return {"rappel": self.trouves / max(self.vrais, 1),
                "faux": self.faux,
                "pos_med": float(np.median(self.err_pos)) if self.err_pos else np.nan,
                "pos_p90": float(np.percentile(self.err_pos, 90)) if self.err_pos else np.nan,
                "ang_med": float(np.median(self.err_ang)) if self.err_ang else np.nan,
                "ang_p90": float(np.percentile(self.err_ang, 90)) if self.err_ang else np.nan,
                "z_med": float(np.median(self.err_z)) if self.err_z else np.nan,
                "type": self.bon_type / t, "face": self.bonne_face / t,
                "rappel_fines": self.trouves_fines / max(self.vrais_fines, 1),
                "rappel_contact": self.trouves_contact / max(self.vrais_contact, 1),
                "rappel_empilees": self.trouves_empilees / max(self.vrais_empilees, 1),
                "rappel_porteuses": self.trouves_porteuses / max(self.vrais_porteuses, 1),
                "support": self.bon_support / t,
                "support_empilees": self.bon_support_empilees / max(self.trouves_empilees, 1),
                "type_face_empilees": self.bon_type_face_empilees / max(self.trouves_empilees, 1),
                "pos_med_empilees": (float(np.median(self.err_pos_empilees))
                                     if self.err_pos_empilees else np.nan),
                "pos_med_porteuses": (float(np.median(self.err_pos_porteuses))
                                      if self.err_pos_porteuses else np.nan),
                "ordre_ok": self.ordre_ok / max(self.scenes, 1),
                "premiere_libre": self.premiere_libre / max(self.scenes, 1),
                "rappel_inclinees": self.trouves_inclinees / max(self.vrais_inclinees, 1),
                "support_inclinees": self.bon_support_inclinees / max(self.trouves_inclinees, 1),
                "lacet_med": float(np.median(self.err_lacet)) if self.err_lacet else np.nan,
                "incl_med": float(np.median(self.err_incl)) if self.err_incl else np.nan,
                "incl_med_inclinees": (float(np.median(self.err_incl_inclinees))
                                       if self.err_incl_inclinees else np.nan),
                "pos_med_inclinees": (float(np.median(self.err_pos_inclinees))
                                      if self.err_pos_inclinees else np.nan)}


def comparer(pieces, verite: Verite, distance_max: float = 15.0) -> Bilan:
    """
    Apparie detections et pieces vraies (centre de la face superieure, dans
    le repere CAMERA : ainsi la comparaison ne depend pas de la facon dont
    chaque methode construit son repere table). Appariement glouton par
    distance croissante, au plus `distance_max` mm. L'erreur de position est
    separee en une erreur DANS le plan de la table (ce qui compte pour
    saisir la piece) et une erreur de hauteur.
    """
    b = Bilan(vrais=len(verite.pieces), scenes=1)
    etroite = [min(p.dims[1], p.dims[2]) < 10 for p in verite.pieces]
    b.vrais_fines = sum(etroite)
    b.vrais_contact = sum(p.contact for p in verite.pieces)
    porteuses = {p.support for p in verite.pieces if p.support is not None}
    b.vrais_empilees = sum(p.support is not None and not getattr(p, "inclinee", False)
                           for p in verite.pieces)
    b.vrais_porteuses = len(porteuses)
    b.vrais_inclinees = sum(bool(getattr(p, "inclinee", False)) for p in verite.pieces)
    if not pieces:
        b.ordre_ok = b.premiere_libre = int(not verite.pieces)
        return b
    det = np.array([p.T_camera[:3, 3] for p in pieces])
    axe_det = np.array([p.T_camera[:3, 0] for p in pieces])
    if len(verite.pieces) == 0:
        b.faux = len(pieces)
        b.ordre_ok = 1
        return b
    dist = np.linalg.norm(det[:, None, :] - verite.centres_camera[None], axis=2) * 1000
    paires = sorted((dist[i, j], i, j) for i in range(len(det))
                    for j in range(len(verite.pieces)))
    pris_d, pris_v = set(), set()
    apparie: Dict[int, int] = {}
    for dd, i, j in paires:
        if dd > distance_max or i in pris_d or j in pris_v:
            continue
        pris_d.add(i)
        pris_v.add(j)
        b.trouves += 1
        e = (det[i] - verite.centres_camera[j]) * 1000
        n = verite.R_cw[:, 2]                 # verticale du monde, vue de la camera
        b.err_z.append(float(e @ n))
        b.err_pos.append(float(np.linalg.norm(e - (e @ n) * n)))
        b.trouves_fines += int(etroite[j])
        b.trouves_contact += int(verite.pieces[j].contact)
        cosang = abs(float(axe_det[i] @ verite.axes_camera[j]))
        b.err_ang.append(float(np.degrees(np.arccos(min(1.0, cosang)))))
        # lacet (axes projetes sur la table) et inclinaison, separement
        a_d, a_v = axe_det[i], verite.axes_camera[j]
        incl_d = np.degrees(np.arcsin(min(1.0, abs(float(a_d @ n)))))
        incl_v = np.degrees(np.arcsin(min(1.0, abs(float(a_v @ n)))))
        b.err_incl.append(float(abs(incl_d - incl_v)))
        h_d, h_v = a_d - (a_d @ n) * n, a_v - (a_v @ n) * n
        if np.linalg.norm(h_d) > 1e-6 and np.linalg.norm(h_v) > 1e-6:
            c = abs(float(h_d @ h_v)) / (np.linalg.norm(h_d) * np.linalg.norm(h_v))
            b.err_lacet.append(float(np.degrees(np.arccos(min(1.0, c)))))
        pv = verite.pieces[j]
        if getattr(pv, "inclinee", False):
            b.trouves_inclinees += 1
            b.err_incl_inclinees.append(b.err_incl[-1])
            b.err_pos_inclinees.append(b.err_pos[-1])
        b.bon_type += int(pieces[i].type == pv.type)
        b.bonne_face += int(pieces[i].face == pv.face)
        if pv.support is not None and not getattr(pv, "inclinee", False):
            b.trouves_empilees += 1
            b.err_pos_empilees.append(b.err_pos[-1])
            b.bon_type_face_empilees += int(pieces[i].type == pv.type
                                            and pieces[i].face == pv.face)
        if j in porteuses:
            b.trouves_porteuses += 1
            b.err_pos_porteuses.append(b.err_pos[-1])
        apparie[i] = j
    b.faux = len(det) - len(pris_d)

    # SUPPORT : celui annonce (numero de la piece porteuse, None : la table)
    # doit etre la detection appariee a la vraie porteuse
    par_numero = {getattr(p, "numero", k + 1): k for k, p in enumerate(pieces)}
    for i, j in apparie.items():
        pv = verite.pieces[j]
        s_det = getattr(pieces[i], "support", None)
        if pv.support is None:
            juste = s_det is None and getattr(pieces[i], "hauteur_base", 0.0) < 4.0
        else:
            k = par_numero.get(s_det)
            juste = k is not None and apparie.get(k) == pv.support
        b.bon_support += int(juste)
        if pv.support is not None:
            if getattr(pv, "inclinee", False):
                b.bon_support_inclinees += int(juste)
            else:
                b.bon_support_empilees += int(juste)

    # ORDRE DE PRISE : chaque vraie piece portee doit etre prise avant sa
    # porteuse ; une porteuse prise alors que la piece qu'elle porte n'a pas
    # ete vue est une faute (on tirerait une piece coincee)
    ordre = vision.ordre_de_prise(list(pieces))
    rang = {}
    for r, p in enumerate(ordre):
        i = next(k for k, q in enumerate(pieces) if q is p)
        if i in apparie:
            rang[apparie[i]] = r
    ok = True
    for j, pv in enumerate(verite.pieces):
        if pv.support is None or pv.support not in rang:
            continue
        if j not in rang or rang[j] > rang[pv.support]:
            ok = False
    b.ordre_ok = int(ok)
    if ordre:
        i0 = next(k for k, q in enumerate(pieces) if q is ordre[0])
        j0 = apparie.get(i0)
        b.premiere_libre = int(j0 is not None and j0 not in porteuses)
    return b
