# -*- coding: utf-8 -*-
"""
PROPOSITIONS DE POSE par points cles : le reseau (YOLO11-pose, modeles/
pieces-cles.pt) donne pour chaque piece sa classe et les 4 coins de sa face
superieure ; la geometrie (coins.py) en tire une pose 3D par PnP.

    import sys; sys.path.insert(0, "examples/perception/apprentissage/points_cles")
    from proposer import ProposeurCles
    pc_ = ProposeurCles()                        # modeles/pieces-cles.pt
    hyps = pc_.proposer_capture("dataset/brut/0042.npz")
    for h in hyps:
        print(h.type, h.face, h.T_table[:3, 3] * 1000, h.confiance, h.erreur_px)
    pieces = [h.piece(k + 1) for k, h in enumerate(hyps)]   # perception.PieceVue

INTERFACE POUR raffinement.py (non branchee ici, a faire la-bas) :
raffinement.raffiner(capture, propositions, types, masques, vue) prend des
vision.Piece dont T_camera est dans le repere de la PROFONDEUR, avec type
et face ; Hypothese.piece() rend exactement cela (perception.PieceVue,
origine "points_cles"). Il suffit donc d'AJOUTER ces pieces a la liste des
propositions :

    props = pieces_reseau + [h.piece(k) for k, h in enumerate(
        proposeur.proposer(vue.image, vue=vue, table=table, proj=proj))]
    raffinement.raffiner(capture, props, types, masques, vue, table=table)

Le doublon eventuel (meme piece proposee par YOLO-seg et par les coins) est
elimine par l'affinage (_doublon : penetration mutuelle). Chaque Hypothese
porte aussi ses VARIANTES (autres (type, face, base) tries par score) :
utiles si l'affinage veut essayer plusieurs interpretations discretes.

CHOIX DU CANDIDAT. Pour chaque (type, face, base) plausible, la face
superieure (dimensions connues) est posee HORIZONTALE a la hauteur base +
epaisseur et ajustee aux coins (coins.pose_sur_plan) ; score :
  exp(-err^2 / 2 sigma^2)          erreur de reprojection des 4 coins
                                   (sigma 8 px) : la TAILLE apparente de
                                   la face dit son type et sa face ;
  x (0.5 + 0.5 accord)             accord avec la hauteur mesuree par la
                                   profondeur DANS le quadrilatere (comme
                                   perception.ajuster_masque, 5 mm) ;
  x 0.8 si le type differe de la classe du reseau ; x 0.97 si empilee.
Une piece INCLINEE ne s'explique pas par une face horizontale : la pose
libre (IPPE) est retenue (methode "ippe") si elle reprojette nettement
mieux (erreur libre < 0.35 x erreur plane, plane > 1.5 px), avec une
inclinaison > 8 deg, ni sous la table ni en l'air, ET si la PROFONDEUR
confirme la pente : la difference de hauteur mesuree entre les deux bouts
de la face (hauteurs_bouts) vaut celle de la pose libre a 5 mm pres, et
depasse 8 mm. Les coins seuls ne suffisent pas : sur la validation reelle,
un PnP libre (6 degres de liberte pour 8 mesures, vue presque affine)
"explique" le bruit des coins predits presque parfaitement et inventait
des inclinaisons (erreurs de 30 a 100 mm) ; sans profondeur, la pose
reste donc horizontale.

Sans profondeur (table = None), il faut au moins le repere de la table vu
par la camera (T_vue_table) : pose plane sans accord de hauteur. Sans
rien : seulement la pose libre dans le repere de la camera couleur
(T_vue), T_table et T_camera None.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
APPRENTISSAGE = os.path.dirname(ICI)
for _d in (APPRENTISSAGE, ICI):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import commun  # noqa: E402
from commun import perception  # noqa: E402
import coins as kc  # noqa: E402

POIDS = os.path.join(commun.MODELES, "pieces-cles.pt")
#: tolerance (px) de l'erreur de reprojection dans le choix du candidat ; 8 px
#: plutot que 3 : les coins predits ont ~4 px d'erreur (mediane), le choix
#: s'appuie alors davantage sur la hauteur mesuree et la classe (sur la
#: validation reelle : bon (type, face) 71 % a 3 px, 78 % a 8 px)
SIGMA_PX = 8.0


@dataclass
class Detection:
    """Sortie brute du reseau pour une piece."""
    classe: str
    conf: float
    boite: np.ndarray                 # xyxy, pixels de la vue
    coins: np.ndarray                 # 4 x 2, ordre canonique
    conf_coins: np.ndarray            # 4, confiance de chaque coin (0 a 1)


@dataclass
class Hypothese:
    """Une pose proposee pour une piece (voir l'en-tete)."""
    type: str
    face: str
    dims: Tuple[float, float, float]  # mm : long, large, epaisseur (perception.forme)
    T_table: Optional[np.ndarray]     # face superieure, repere table
    T_vue: np.ndarray                 # face superieure, repere de la camera couleur
    T_camera: Optional[np.ndarray]    # face superieure, repere de la PROFONDEUR
    confiance: float
    erreur_px: float                  # RMS de reprojection des 4 coins
    h_mesure: float                   # mm, nan sans profondeur
    base: float                       # mm, hauteur de la face d'appui
    methode: str                      # "plan" ou "ippe"
    coins: np.ndarray                 # 4 x 2 predits (ordre canonique)
    classe_reseau: str = ""
    conf_reseau: float = 0.0
    variantes: List[Tuple[str, str, float, float, float]] = field(default_factory=list)
    # (type, face, base mm, score, erreur px), la meilleure en tete

    @property
    def inclinaison(self) -> float:
        return kc.inclinaison_deg(self.T_table) if self.T_table is not None else float("nan")

    def piece(self, numero: int = 1, table=None, vue=None, T_robot_camera=None):
        """perception.PieceVue (contour : silhouette projetee si table et vue
        sont donnees, sinon le quadrilatere des coins)."""
        if self.T_table is None or self.T_camera is None:
            raise ValueError("pas de table : pose seulement dans le repere couleur (T_vue)")
        if table is not None and vue is not None:
            contour = perception.silhouette(self.T_table, self.dims, table, vue)
        else:
            contour = np.round(self.coins).astype(np.int32)
        return perception.PieceVue(
            numero, self.face, tuple(self.dims), 0.0, self.T_camera, self.T_table, contour,
            int(abs(cv2.contourArea(self.coins.astype(np.float32)))), self.type,
            hauteur_base=self.base, confiance=float(np.clip(self.confiance, 0, 1)),
            T_robot=None if T_robot_camera is None else np.asarray(T_robot_camera) @ self.T_camera,
            origine="points_cles", inclinaison=self.inclinaison)


# ---------------------------------------------------------------------------
# Geometrie : d'une detection a une pose
# ---------------------------------------------------------------------------

def hauteur_dans(coins_uv: np.ndarray, table, proj, retrait: float = 0.25,
                 ignorer: Optional[np.ndarray] = None, centile: float = 90.0,
                 n_min: int = 15) -> float:
    """Hauteur (mm) de la face superieure mesuree par la profondeur : 90e
    centile des points dont l'image tombe dans le quadrilatere des coins,
    retreci vers son centre (les bords melangent piece et table) ; nan si
    moins de n_min points (bois sature : pas de mesure)."""
    u, v, ok = proj
    q = np.asarray(coins_uv, float).reshape(4, 2)
    c = q.mean(axis=0)
    q = c + (1.0 - retrait) * (q - c)
    H, W = ok.shape
    x0, y0 = np.floor(q.min(axis=0)).astype(int)
    x1, y1 = np.ceil(q.max(axis=0)).astype(int)
    sel = ok & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    if sel.sum() < n_min:
        return float("nan")
    pts = np.stack([u[sel], v[sel]], 1).astype(np.float32)
    poly = q.astype(np.float32).reshape(-1, 1, 2)
    dedans = np.array([cv2.pointPolygonTest(poly, (float(a), float(b)), False) >= 0
                       for a, b in pts])
    if ignorer is not None:
        Hi, Wi = ignorer.shape
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, Wi - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, Hi - 1)
        dedans &= ~ignorer[yi, xi]
    h = table.hauteur[sel][dedans]
    h = h[(h > max(0.002, 2.5 * table.bruit)) & (h < 0.10)]
    if len(h) < n_min:
        return float("nan")
    return float(np.percentile(h, centile)) * 1000.0


def quads_bouts(coins_uv: np.ndarray, part: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """Les deux bouts de la face (coins canoniques) : quadrilateres couvrant
    `part` du grand cote, cote des coins 0-3 (bout A) et 1-2 (bout B)."""
    q = np.asarray(coins_uv, float).reshape(4, 2)
    a = np.array([q[0], q[0] + part * (q[1] - q[0]), q[3] + part * (q[2] - q[3]), q[3]])
    b = np.array([q[1] - part * (q[1] - q[0]), q[1], q[2], q[2] - part * (q[2] - q[3])])
    return a, b


def hauteurs_bouts(coins_uv: np.ndarray, table, proj) -> Tuple[float, float]:
    """Hauteur mediane (mm) mesuree a chaque bout de la face : une piece
    inclinee a un bout plus haut que l'autre (voir estimer_pose)."""
    uv = kc.canoniser(np.asarray(coins_uv, float).reshape(4, 2))
    a, b = quads_bouts(uv)
    return (hauteur_dans(a, table, proj, 0.15, centile=50, n_min=8),
            hauteur_dans(b, table, proj, 0.15, centile=50, n_min=8))


def _bouts_predits(T_table: np.ndarray, L: float, uv: np.ndarray, K: np.ndarray,
                   T_vue_table: np.ndarray) -> Tuple[float, float]:
    """Hauteur (mm) de la face posee en T_table aux centres des bouts A et B
    de l'image (L en m)."""
    a, b = quads_bouts(uv)
    P = np.array([[s * 0.33 * L, 0.0, 0.0, 1.0] for s in (-1, 1)]) @ T_table.T
    q = kc.projeter((P @ T_vue_table.T)[:, :3], K)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    if np.linalg.norm(q[0] - ca) + np.linalg.norm(q[1] - cb) > \
            np.linalg.norm(q[1] - ca) + np.linalg.norm(q[0] - cb):
        P = P[::-1]
    return float(P[0, 2] * 1000.0), float(P[1, 2] * 1000.0)


def candidats(types: Dict[str, Sequence[float]], h_mes: float,
              debout: bool = True) -> List[Tuple[str, str, Tuple[float, float, float], float]]:
    """(type, face, (L, l, h) mm, base mm) plausibles, comme
    perception.ajuster_masque : la piece sur la table a 8 mm pres de la
    hauteur mesuree, empilee a 5 mm pres ; sans mesure, sur la table."""
    faces = perception.FACES if debout else perception.FACES[:2]
    bases = perception.bases_de_pile(types)
    tous = [(n, f, perception.forme(d, f), float(b)) for b in bases for n, d in types.items()
            for f in faces if not (b > 0 and f == "debout")]
    if not np.isfinite(h_mes):
        return [c for c in tous if c[3] == 0]
    proches = [c for c in tous if abs(c[3] + c[2][2] - h_mes) < (8.0 if c[3] == 0 else 5.0)]
    return proches or [c for c in tous if c[3] == 0]


def estimer_pose(coins_uv: np.ndarray, K: np.ndarray, T_vue_table: Optional[np.ndarray],
                 types: Dict[str, Sequence[float]], classe: str = "", conf: float = 1.0,
                 h_mes: float = float("nan"), sigma_px: float = SIGMA_PX,
                 debout: bool = True,
                 h_bouts: Optional[Tuple[float, float]] = None) -> Hypothese:
    """La meilleure pose (et ses variantes) pour 4 coins canoniques (voir
    l'en-tete). T_vue_table : repere table vu de la camera couleur ;
    h_mes : hauteur mesuree de la face (mm) ; h_bouts : hauteurs mesurees a
    ses deux bouts (hauteurs_bouts), seules a pouvoir CONFIRMER une piece
    inclinee (sans elles, la pose est toujours horizontale)."""
    uv = kc.canoniser(np.asarray(coins_uv, float).reshape(4, 2))
    if T_vue_table is None:
        # pas de table : pose libre seulement, type = classe du reseau
        dims = perception.forme(types[classe], "a plat") if classe in types else (68.0, 17.0, 8.0)
        T, e = kc.pnp_ippe(uv, dims[0] / 1000, dims[1] / 1000, K)
        return Hypothese(classe, "a plat", dims, None, T if T is not None else np.eye(4), None,
                         conf * float(np.exp(-0.5 * (e / sigma_px) ** 2)), e, h_mes, 0.0, "ippe",
                         uv, classe, conf)
    resultats = []
    for nom, face, (L, l, h), b in candidats(types, h_mes, debout):
        T, err = kc.pose_sur_plan(uv, L / 1000, l / 1000, (b + h) / 1000, K, T_vue_table)
        s = float(np.exp(-0.5 * (err / sigma_px) ** 2))
        accord = 1.0
        if np.isfinite(h_mes):
            accord = float(np.exp(-0.5 * ((h_mes - (b + h)) / 5.0) ** 2))
        s *= 0.5 + 0.5 * accord
        if classe and nom != classe:
            s *= 0.8
        if b > 0:
            s *= 0.97
        resultats.append((s, nom, face, (L, l, h), b, T, err))
    resultats.sort(key=lambda r: -r[0])
    s, nom, face, dims, b, T, err = resultats[0]
    methode = "plan"
    # piece inclinee : la pose libre explique nettement mieux les coins
    Tl, el = kc.pnp_ippe(uv, dims[0] / 1000, dims[1] / 1000, K)
    if (Tl is not None and err > 1.5 and el < 0.35 * err and h_bouts is not None
            and np.all(np.isfinite(h_bouts))):
        Tt = np.linalg.inv(T_vue_table) @ Tl
        pa, pb = _bouts_predits(Tt, dims[0] / 1000, uv, K, T_vue_table)
        # inclinee, pas sous la table ni en l'air, et la PROFONDEUR voit la
        # pente : un bout plus haut que l'autre, d'autant que predit
        if (kc.inclinaison_deg(Tt) > 8.0 and -0.005 < Tt[2, 3] < 0.15
                and abs(pa - pb) > 8.0
                and abs((h_bouts[0] - h_bouts[1]) - (pa - pb)) < 5.0):
            T, err, methode = Tt, el, "ippe"
            s = float(np.exp(-0.5 * (el / sigma_px) ** 2)) * (0.8 if classe and nom != classe else 1.0)
    return Hypothese(nom, face, dims, T, T_vue_table @ T, None, conf * s, err, h_mes, b, methode,
                     uv, classe, conf,
                     [(r[1], r[2], r[4], round(r[0], 4), round(r[6], 2)) for r in resultats])


# ---------------------------------------------------------------------------
# Le reseau
# ---------------------------------------------------------------------------

class ProposeurCles:
    """Le reseau de points cles et la mise en pose (voir l'en-tete)."""

    def __init__(self, modele: str = POIDS, types: Optional[Dict[str, Sequence[float]]] = None,
                 device: Optional[str] = None, imgsz: int = 864, conf: float = 0.25,
                 debout: bool = True, erreur_max: float = 12.0):
        from ultralytics import YOLO
        if not os.path.exists(modele):
            raise FileNotFoundError("poids introuvables : {} (points_cles/entrainer_cles.py)"
                                    .format(modele))
        self.reseau = YOLO(modele)
        self.types = {k: tuple(float(x) for x in v)
                      for k, v in (types or commun.charger_types()).items()}
        self.device = device or commun.device_auto()
        if self.device == "0":
            self.device = "cuda:0"
        self.imgsz, self.conf, self.debout = imgsz, conf, debout
        #: au-dela (px RMS), aucune piece de taille connue n'explique les
        #: coins : pas de proposition (le reseau s'est trompe)
        self.erreur_max = erreur_max
        self.durees: Dict[str, float] = {}

    def detecter(self, image: np.ndarray) -> List[Detection]:
        t = time.perf_counter()
        r = self.reseau.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                verbose=False)[0]
        self.durees["reseau"] = time.perf_counter() - t
        if r.keypoints is None or r.boxes is None or not len(r.boxes):
            return []
        k = r.keypoints.data.cpu().numpy()                 # N x 4 x 3
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        s = r.boxes.conf.cpu().numpy()
        return [Detection(self.reseau.names[int(c[i])], float(s[i]), b[i], k[i, :, :2],
                          k[i, :, 2] if k.shape[2] > 2 else np.ones(4)) for i in range(len(b))]

    def proposer(self, image: np.ndarray, vue=None, table=None, proj=None,
                 T_vue_table: Optional[np.ndarray] = None) -> List[Hypothese]:
        """
        Hypotheses de pose pour une image couleur, triees par confiance.
        `vue` (perception.VueCouleur, REQUISE : K de l'image, passage
        profondeur -> couleur) ; `table` (perception.Table) : le plan, et
        T_camera dans le repere de la profondeur ; `proj`
        (perception.projeter_profondeur) : la hauteur mesuree dans chaque
        quadrilatere. Sans table, T_vue_table (repere table vu de la camera
        couleur) suffit pour la pose plane ; sans l'un ni l'autre, pose libre
        dans le repere couleur seulement. Les detections dont l'erreur de
        reprojection depasse erreur_max sont ecartees.
        """
        dets = self.detecter(image)
        t = time.perf_counter()
        if vue is None:
            raise ValueError("vue (K de l'image) requise pour la pose")
        if table is not None:
            T_vue_table = vue.T_cp @ table.T_ct
        sortie = []
        for d in dets:
            h_mes, bouts = float("nan"), None
            if table is not None and proj is not None:
                h_mes = hauteur_dans(d.coins, table, proj)
                bouts = hauteurs_bouts(d.coins, table, proj)
            hyp = estimer_pose(d.coins, vue.K, T_vue_table, self.types, d.classe, d.conf,
                               h_mes, debout=self.debout, h_bouts=bouts)
            if not hyp.erreur_px <= self.erreur_max:
                continue
            if table is not None and hyp.T_table is not None:
                hyp.T_camera = table.T_ct @ hyp.T_table
            sortie.append(hyp)
        sortie.sort(key=lambda h: -h.confiance)
        self.durees["pose"] = time.perf_counter() - t
        return sortie

    def proposer_capture(self, capture) -> List[Hypothese]:
        """Depuis une capture (.npz, dictionnaire) : vue brute, table, profondeur."""
        t = time.perf_counter()
        cap = perception.charger_capture(capture)
        vue = perception.vue_couleur(cap)
        K = np.asarray(cap["K"], float)
        table = perception.analyser_table(cap["profondeur"], K)
        proj = perception.projeter_profondeur(cap["profondeur"], K, vue)
        dt = time.perf_counter() - t
        h = self.proposer(vue.image, vue=vue, table=table, proj=proj)
        self.durees["table"] = dt
        self.derniere = (vue, table)
        return h


_PROPOSEUR: Optional[ProposeurCles] = None


def proposer(image: np.ndarray, vue=None, table=None, proj=None, modele: str = POIDS,
             **kw) -> List[Hypothese]:
    """Raccourci : un ProposeurCles garde en memoire (charge au premier appel)."""
    global _PROPOSEUR
    if _PROPOSEUR is None or _PROPOSEUR.reseau.ckpt_path != modele:
        _PROPOSEUR = ProposeurCles(modele, **kw)
    return _PROPOSEUR.proposer(image, vue=vue, table=table, proj=proj)
