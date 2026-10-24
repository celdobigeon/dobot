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
        c, s = np.cos(np.radians(pc.lacet)), np.sin(np.radians(pc.lacet))
        Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # boite -> monde
        o = Rb.T @ (C - [pc.x / 1000.0, pc.y / 1000.0, 0.0])
        d = d_w @ Rb                                          # Rb^T d
        bas = np.array([-L / 2, -l / 2, 0.0])
        haut = np.array([L / 2, l / 2, h])
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


def placer(rng: np.random.Generator, n: int, types: Dict[str, Sequence[float]],
           zone: Tuple[float, float, float, float] = (-110, 110, -80, 80),
           touchantes: int = 0, faces: Sequence[str] = ("a plat", "sur la tranche"),
           essais: int = 400) -> List[PieceVraie]:
    """
    `n` pieces au hasard dans `zone` (mm, repere monde), sans chevauchement ;
    `touchantes` d'entre elles sont posees CONTRE une piece deja placee (cote
    a cote, bout a bout, ou en T), sans jeu.
    """
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
        if len(pieces) >= n:
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
    return pieces


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
        pw = np.array([p.x / 1000.0, p.y / 1000.0, p.dims[2] / 1000.0])
        centres.append(R_cw @ (pw - C))
        a = np.radians(p.lacet)
        axes.append(R_cw @ np.array([np.cos(a), np.sin(a), 0.0]))
    sc = vision.Scene(profondeur=Z.astype(np.float64), K=cam.K.copy(),
                      infrarouge=ir, couleur=img)
    return sc, Verite(list(pieces), np.array(centres).reshape(-1, 3),
                      np.array(axes).reshape(-1, 3), R_cw, C)


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
                "rappel_contact": self.trouves_contact / max(self.vrais_contact, 1)}


def comparer(pieces, verite: Verite, distance_max: float = 15.0) -> Bilan:
    """
    Apparie detections et pieces vraies (centre de la face superieure, dans
    le repere CAMERA : ainsi la comparaison ne depend pas de la facon dont
    chaque methode construit son repere table). Appariement glouton par
    distance croissante, au plus `distance_max` mm. L'erreur de position est
    separee en une erreur DANS le plan de la table (ce qui compte pour
    saisir la piece) et une erreur de hauteur.
    """
    b = Bilan(vrais=len(verite.pieces))
    etroite = [min(p.dims[1], p.dims[2]) < 10 for p in verite.pieces]
    b.vrais_fines = sum(etroite)
    b.vrais_contact = sum(p.contact for p in verite.pieces)
    if not pieces:
        return b
    det = np.array([p.T_camera[:3, 3] for p in pieces])
    axe_det = np.array([p.T_camera[:3, 0] for p in pieces])
    if len(verite.pieces) == 0:
        b.faux = len(pieces)
        return b
    dist = np.linalg.norm(det[:, None, :] - verite.centres_camera[None], axis=2) * 1000
    paires = sorted((dist[i, j], i, j) for i in range(len(det))
                    for j in range(len(verite.pieces)))
    pris_d, pris_v = set(), set()
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
        pv = verite.pieces[j]
        b.bon_type += int(pieces[i].type == pv.type)
        b.bonne_face += int(pieces[i].face == pv.face)
    b.faux = len(det) - len(pris_d)
    return b
