# -*- coding: utf-8 -*-
"""
Etalonnage camera -> robot : le robot pose une piece, la camera la voit.

    .venv/bin/python examples/perception/etalonner_camera_robot.py
    .venv/bin/python examples/perception/etalonner_camera_robot.py --simulation

Principe. Le robot tient une piece (une EPAISSE A PLAT) dans sa pince. A
chaque cycle, il la pose en un point qu'il choisit, s'ecarte du champ, et
la camera la detecte : on obtient la meme piece vue deux fois, dans le
repere du robot (la ou il l'a posee) et dans celui de la camera (la ou la
perception la voit). Il la reprend LA OU IL L'A POSEE -- jamais la ou la
camera la voit : une erreur du recalage provisoire ne se propage pas.

Avec ces paires, on cherche la transformation rigide T_robot_camera qui
les superpose (Kabsch), et en meme temps le DECALAGE de la piece dans la
pince : les mors la centrent sur sa largeur, pas sur sa longueur, et ce
decalage, fixe dans le repere de la pince, tourne avec r. Les poses sont
donc prises a des orientations variees. On estime enfin l'ecart d'angle
entre r et l'orientation de la piece (a 180 deg pres).

Deroule :

  0. SECTEUR (interactif) : amener le bras a l'une, puis a l'autre des deux
     rotations extremes de la base qu'il ne doit pas depasser (pour ne pas
     heurter la camera). Tout le reste se fait dans ce secteur : poses,
     deplacements au clavier, repli pendant les captures ;
  1. ACCOSTAGE (interactif) : amener la pince OUVERTE au contact de la
     table, n'importe ou -- a la main (bouton de deverrouillage de
     l'avant-bras) puis au clavier. Le contact donne la hauteur de la
     table dans le repere du robot ;
  2. le bras remonte de 5 mm ; on place la piece entre les mors, posee sur
     la table ; Entree : la pince se ferme ;
  3. boucle : poser (approche et depart VERTICAUX), remonter en posture
     d'equerre pour degager la vue, voir, reprendre. Les premiers points
     restent pres du depart ; des que le recalage existe, ils couvrent la
     partie de la table que la camera voit et que le bras atteint ;
  4. VALIDATION : quelques saisies guidees par la camera seule ;
  5. le resultat est ecrit dans etalonnage_camera_robot.json, que
     `charger_etalonnage()` relit pour Perception(T_robot_camera=...).

RELANCER SANS RECONFIGURER : --recommencer garde secteur, hauteur de table
et point de depart du fichier, mais efface les mesures (la camera a
bouge, par exemple).

INTERRUPTION ET REPRISE. Ctrl+C a tout moment arrete le robot sur place ;
si la piece est en main, le script propose de la reposer (la pompe
s'arrete). Le fichier est tenu a jour apres chaque pose : --reprendre
repart de la, sans refaire secteur ni accostage, en reprenant la piece
la ou elle a ete laissee -- ou avec une NOUVELLE piece, qui a son propre
decalage dans la pince.

Avant de lancer : le serveur de camera tourne, la table est degagee (une
seule piece en vue), le robot est mis en route (prise d'origine faite).
Ctrl+C arrete tout ; ce qui a ete mesure est enregistre.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
for _d in (ICI, RACINE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: fichier du resultat, relu par charger_etalonnage()
FICHIER = os.path.join(ICI, "etalonnage_camera_robot.json")

#: la piece tenue : une EPAISSE A PLAT (17 mm de haut, saisie sur ses 19 mm)
TYPE_PIECE, FACE_PIECE, HAUTEUR_PIECE = "epaisse", "a plat", 17.0
#: mm : hauteur de la pointe de la pince au-dessus de la table, pour saisir
#: et poser. La piece repose sur la table pendant que la pince se ferme ou
#: s'ouvre : elle ne tombe jamais.
PRISE = 5.0
#: mm : hauteur des deplacements au-dessus de la table. On y monte et on en
#: descend a la VERTICALE (MOVL), au-dessus du point de pose : l'approche
#: et le depart sont verticaux sur TRANSIT - PRISE = 65 mm, bien plus que
#: les APPROCHE mm exiges.
TRANSIT = 70.0
APPROCHE = 20.0
assert TRANSIT - PRISE >= APPROCHE
#: les premiers points restent a cette distance du depart (mm)...
RAYON_DEPART = 30.0
#: ...tant qu'il n'y a pas assez de paires pour un recalage
PAIRES_MIN = 4
#: s : temps laisse aux mors pour se fermer ou s'ouvrir
DELAI_PINCE = 0.8
#: mm : au-dela, une piece vue n'est pas celle qu'on a posee
ECART_MAX = 25.0
#: deg : marge gardee a l'interieur du secteur de la base
MARGE_SECTEUR = 3.0
#: deg : marge gardee aux butees de chaque articulation, aux points de pose
#: et de transit. Le controleur refuse des cibles que le modele accepte de
#: justesse (essai du 25/10 : bras arriere a 83,9 deg pour une butee a 85,
#: alarme 0x12).
MARGE_BUTEE = 8.0
#: posture de degagement : l'equerre (bras arriere vertical, avant-bras
#: horizontal), le bras arriere un peu penche pour ne pas etre SUR sa
#: butee (0 deg)
EQUERRE_J2, EQUERRE_J3 = 5.0, 0.0


# ---------------------------------------------------------------------------
# Calcul : recalage rigide + decalage de la piece dans la pince
# ---------------------------------------------------------------------------

def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0],
                     [math.sin(a), math.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def kabsch(P: np.ndarray, Q: np.ndarray, poids=None) -> Tuple[np.ndarray, np.ndarray]:
    """R, t minimisant somme w |R p + t - q|^2 (points en lignes)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    w = np.ones(len(P)) if poids is None else np.asarray(poids, float)
    w = w / w.sum()
    cp, cq = w @ P, w @ Q
    H = ((P - cp) * w[:, None]).T @ (Q - cq)
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


@dataclass
class Paire:
    """Une pose : la ou le robot a pose la piece, la ou la camera la voit."""
    pince: np.ndarray        # (x, y, z) mm de la pointe de la pince a la pose
    r: float                 # deg, orientation de la pince
    camera: np.ndarray       # (x, y, z) mm, centre de la face superieure, camera
    lacet_camera: np.ndarray  # vecteur unitaire du grand cote, repere camera
    z_table: float           # mm, hauteur de la table (robot)
    #: numero de la piece (une nouvelle piece, ou une piece replacee a la
    #: main, a son propre decalage dans la pince)
    session: int = 0

    @property
    def dessus(self) -> np.ndarray:
        """Centre de la face superieure SOUS la pince, repere robot (sans
        le decalage dans la pince)."""
        return np.array([self.pince[0], self.pince[1], self.z_table + HAUTEUR_PIECE])


@dataclass
class Recalage:
    R: np.ndarray                     # camera -> robot
    t: np.ndarray                     # mm
    decalages: dict                   # session -> (dx, dy) mm, piece dans le repere pince
    ecart_r: float                    # deg : r de la pince - lacet piece (mod 180)
    residus: np.ndarray               # mm, par paire
    gardees: np.ndarray               # masque des paires retenues

    @property
    def decalage(self) -> np.ndarray:
        """Decalage de la piece de la DERNIERE session."""
        return self.decalages[max(self.decalages)] if self.decalages else np.zeros(2)

    @property
    def T_mm(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    @property
    def T_m(self) -> np.ndarray:
        """T_robot_camera en METRES : la forme qu'attend Perception."""
        T = self.T_mm
        T[:3, 3] /= 1000.0
        return T

    @property
    def rms(self) -> float:
        r = self.residus[self.gardees]
        return float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan")

    def vers_robot(self, p_camera) -> np.ndarray:
        return self.R @ np.asarray(p_camera, float) + self.t

    def vers_camera(self, p_robot) -> np.ndarray:
        return self.R.T @ (np.asarray(p_robot, float) - self.t)

    def lacet_robot(self, axe_camera) -> float:
        """Orientation (deg) du grand cote d'une piece, repere robot."""
        a = self.R @ np.asarray(axe_camera, float)
        return math.degrees(math.atan2(a[1], a[0]))

    def prise(self, p_camera, axe_camera) -> Tuple[np.ndarray, float]:
        """
        Ou mettre la pince pour saisir une piece vue par la camera : au
        CENTRE de sa face superieure (repere robot, mm), pince orientee r
        (deg, a 180 deg pres). Le decalage mesure pendant l'etalonnage ne
        s'applique pas : il tient a la facon dont la piece de depart a ete
        placee dans la pince, une saisie visee au centre n'en a pas.
        """
        r = _angle(self.lacet_robot(axe_camera) + self.ecart_r, 180.0)
        return self.vers_robot(p_camera), r


def _angle(a: float, periode: float = 360.0) -> float:
    """a ramene dans [-periode/2 ; periode/2[."""
    return (a + periode / 2.0) % periode - periode / 2.0


def recaler(paires: Sequence[Paire], decalage: bool = True,
            iterations: int = 40) -> Recalage:
    """
    T_robot_camera et decalage dans la pince, par moindres carres alternes :
    Kabsch a decalages fixes, puis decalages a recalage fixe. Chaque piece
    (session) a son propre decalage, identifiable seulement si r a varie
    pendant la session (sinon il est pris nul). Avec 6 paires et plus,
    celles dont le residu depasse 3 fois la mediane (et 3 mm) sont
    ecartees, puis le calcul est refait.
    """
    if len(paires) < 3:
        raise ValueError("au moins 3 paires")
    n = len(paires)
    P = np.array([p.camera for p in paires], float)
    G = np.array([p.dessus for p in paires], float)
    Rz = [rot_z(p.r) for p in paires]
    sessions = np.array([p.session for p in paires])
    estimables = set()
    for k in set(sessions.tolist()):
        rs = np.radians([p.r for p in paires if p.session == k])
        if len(rs) >= 2 and np.hypot(np.cos(rs).std(), np.sin(rs).std()) > 0.15:
            estimables.add(k)

    def decales(dec):
        return G + np.array([Rz[i] @ [dec[sessions[i]][0], dec[sessions[i]][1], 0.0]
                             for i in range(n)])

    gardees = np.ones(n, bool)
    for _passe in range(2):
        dec = {k: np.zeros(2) for k in set(sessions.tolist())}
        for _ in range(iterations):
            R, t = kabsch(P[gardees], decales(dec)[gardees])
            if not (decalage and estimables):
                break
            e = P @ R.T + t - G
            loc = np.array([(Rz[i].T @ e[i])[:2] for i in range(n)])
            bouge = 0.0
            for k in estimables:
                m = gardees & (sessions == k)
                if m.any():
                    nouv = loc[m].mean(axis=0)
                    bouge = max(bouge, float(np.linalg.norm(nouv - dec[k])))
                    dec[k] = nouv
            if bouge < 1e-5:
                break
        R, t = kabsch(P[gardees], decales(dec)[gardees])
        res = np.linalg.norm(P @ R.T + t - decales(dec), axis=1)
        if n < 6:
            break
        seuil = max(3.0 * float(np.median(res)), 3.0)
        nouv = res <= seuil
        if nouv.sum() < 4 or np.array_equal(nouv, gardees):
            break
        gardees = nouv
    # ecart d'angle : r - lacet, a 180 deg pres (moyenne circulaire de 2x)
    ecarts = []
    for p in paires:
        a = R @ p.lacet_camera
        ecarts.append(p.r - math.degrees(math.atan2(a[1], a[0])))
    e2 = np.radians(2.0 * np.array(ecarts))[gardees]
    ecart_r = _angle(math.degrees(math.atan2(np.sin(e2).mean(), np.cos(e2).mean())) / 2.0,
                     180.0)
    return Recalage(R, t, dec, ecart_r, res, gardees)


# ---------------------------------------------------------------------------
# La piece vue par la camera
# ---------------------------------------------------------------------------

@dataclass
class Vue:
    """Une piece detectee, ramenee a ce dont l'etalonnage a besoin."""
    position: np.ndarray        # mm, centre de la face superieure, repere camera
    axe: np.ndarray             # grand cote, repere camera
    type: str = ""
    face: str = ""
    confiance: float = 1.0


#: ou garder les captures de l'etalonnage : chacune montre UNE piece dont
#: le robot connait la pose -- de quoi annoter automatiquement des images
#: d'entrainement pour la perception (dataset/ n'est pas suivi par git)
ARCHIVE = os.path.join(ICI, "apprentissage", "dataset", "etalonnage")


class Camera:
    """La vraie chaine : serveur de camera + Perception (sans etalonnage)."""

    def __init__(self, archive: Optional[str] = ARCHIVE):
        import perception
        self.perception = perception.Perception()
        self.K = None
        self.taille = None
        self.derniere_capture = None
        self.derniere_detection = []
        self.archive = None
        if archive:
            self.archive = os.path.join(archive, time.strftime("%Y%m%d_%H%M%S"))

    def voir(self) -> List[Vue]:
        import camera
        cap = camera.capturer()
        self.K = np.asarray(cap["K"], float)
        self.taille = cap["profondeur"].shape[::-1]
        self.derniere_capture = cap
        pieces = self.perception.detecter(cap)
        self.derniere_detection = pieces
        return [Vue(p.T_camera[:3, 3] * 1000.0, p.T_camera[:3, 0].copy(), p.type,
                    p.face, float(getattr(p, "confiance", 1.0))) for p in pieces]

    def archiver(self, meta: dict) -> Optional[str]:
        """Garde la derniere capture (.npz) et ce qu'on en sait (.json) : la
        pose de la piece cote robot, ce que la perception a detecte."""
        if self.archive is None or self.derniere_capture is None:
            return None
        os.makedirs(self.archive, exist_ok=True)
        n = len([f for f in os.listdir(self.archive) if f.endswith(".npz")])
        base = os.path.join(self.archive, "{:04d}".format(n))
        np.savez_compressed(base + ".npz", **self.derniere_capture)
        meta = dict(meta)
        meta["detections"] = [{"type": p.type, "face": p.face,
                               "confiance": float(getattr(p, "confiance", 1.0)),
                               "T_camera": np.asarray(p.T_camera, float).tolist()}
                              for p in self.derniere_detection]
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        return base


# ---------------------------------------------------------------------------
# Simulation : robot du simulateur, piece et camera simulees
# ---------------------------------------------------------------------------

class PieceSimulee:
    """
    Suit la piece : tenue par la pince, ou posee. Enveloppe un robot
    (Commande, ou un faux) et intercepte pince() : fermer pres de la piece
    la saisit, ouvrir la pose -- avec un DECALAGE dans la pince, comme en
    vrai.
    """

    def __init__(self, robot, decalage=(0.0, 3.0), ecart_r=90.0):
        self.robot = robot
        self.decalage = np.array([decalage[0], decalage[1], 0.0], float)
        self.ecart_r = float(ecart_r)
        self.tenue = False
        self.position = None            # centre du dessus, repere robot
        self.lacet = 0.0
        self.z_table = None

    def __getattr__(self, nom):
        return getattr(self.robot, nom)

    def pince(self, fermee: bool = True, **kw):
        self.robot.pince(fermee, **kw)
        x, y, z, r = self.robot.pose()
        if fermee and not self.tenue:
            if self.z_table is None:
                self.z_table = z - PRISE           # piece glissee a l'accostage
                self.position = np.array([x, y, self.z_table + HAUTEUR_PIECE])
                self.position += rot_z(r) @ self.decalage
                self.lacet = r - self.ecart_r
            ecart = self.position - [x, y, self.z_table + HAUTEUR_PIECE]
            if np.linalg.norm(ecart[:2]) < 10.0:
                # les mors centrent la piece sur sa LARGEUR seulement : il
                # reste le decalage le long de la piece
                self.tenue = True
                u = np.array([math.cos(math.radians(self.lacet)),
                              math.sin(math.radians(self.lacet)), 0.0])
                self.decalage = rot_z(r).T @ (float(ecart @ u) * u)
        elif not fermee and self.tenue:
            self.tenue = False
            self.position = np.array([x, y, self.z_table + HAUTEUR_PIECE]) + rot_z(r) @ self.decalage
            self.lacet = r - self.ecart_r
        return self


def regarder(position, cible) -> np.ndarray:
    """T_robot_camera (mm) d'une camera placee en `position` qui vise
    `cible` : z optique vers la cible, x vers la droite, y vers le bas."""
    p, c = np.asarray(position, float), np.asarray(cible, float)
    z = (c - p) / np.linalg.norm(c - p)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = p
    return T


class CameraSimulee:
    """Une camera posee devant la table, qui voit la piece simulee avec un
    bruit realiste (perception : ~1 mm, ~1 deg)."""

    def __init__(self, piece: PieceSimulee, T_robot_camera_mm=None, bruit=0.8,
                 graine=0):
        self.piece = piece
        if T_robot_camera_mm is None:
            # camera devant le robot, a 550 mm, 120 mm au-dessus de la
            # table, qui regarde le milieu de la zone de travail
            T_robot_camera_mm = regarder([560.0, 30.0, 120.0], [240.0, 0.0, 0.0])
        self.T = np.asarray(T_robot_camera_mm, float)
        self.bruit = float(bruit)
        self.rng = np.random.default_rng(graine)
        self.K = np.array([[420.0, 0, 424.0], [0, 420.0, 240.0], [0, 0, 1]])
        self.taille = (848, 480)

    def voir(self) -> List[Vue]:
        if self.piece.tenue or self.piece.position is None:
            return []
        R, t = self.T[:3, :3], self.T[:3, 3]
        p = R.T @ (self.piece.position - t) + self.rng.normal(0, self.bruit, 3)
        a = math.radians(self.piece.lacet + self.rng.normal(0, 1.0))
        axe = R.T @ np.array([math.cos(a), math.sin(a), 0.0])
        return [Vue(p, axe, TYPE_PIECE, FACE_PIECE, 0.9)]


# ---------------------------------------------------------------------------
# L'etalonnage
# ---------------------------------------------------------------------------

class Arret(Exception):
    pass


def _vider_clavier() -> None:
    """Oublie les touches tapees d'avance (pendant un mouvement, un Entree
    en trop) : sinon la question suivante les lit et repart sans attendre."""
    if sys.stdin.isatty():
        try:
            import termios
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:                           # noqa: BLE001
            pass


def _demander(texte: str) -> str:
    _vider_clavier()
    try:
        return input(texte)
    except EOFError:
        raise Arret("entree fermee")


class Etalonnage:
    """
    Le deroule complet. `robot` : Commande (ou un objet qui en a pose(),
    movj(), movl(), pince(), repli(), attendre()) ; `camera` : un objet a
    voir() -> [Vue], et K / taille une fois une image vue.
    """

    def __init__(self, robot, camera, z_table: Optional[float] = None,
                 garage: Optional[Sequence[float]] = None, graine: int = 0,
                 journal: Callable[[str], None] = print,
                 demander: Callable[[str], str] = _demander,
                 zone_x=(150.0, 320.0), zone_y=(-220.0, 220.0),
                 r_max: float = 60.0,
                 secteur: Optional[Tuple[float, float]] = None):
        self.robot = robot
        self.camera = camera
        self.z_table = z_table
        self.garage = None if garage is None else [float(v) for v in garage]
        self.rng = np.random.default_rng(graine)
        self.journal = journal
        self.demander = demander
        self.zone_x, self.zone_y, self.r_max = zone_x, zone_y, float(r_max)
        #: (J1 min, J1 max) en degres : la base ne sort pas de ce secteur
        self.secteur = None if secteur is None else tuple(sorted(float(v) for v in secteur))
        self.paires: List[Paire] = []
        self.recalage: Optional[Recalage] = None
        self.depart: Optional[Tuple[float, float, float]] = None   # x, y, r
        self.validations: List[dict] = []
        #: decalage de la piece dans la pince, une fois qu'une saisie visee
        #: au centre l'a remis a zero (None : celui du recalage)
        self.decalage_courant: Optional[np.ndarray] = None
        #: numero de la piece en cours (chaque piece placee a la main a son
        #: propre decalage dans la pince)
        self.session = 0
        #: ou est la piece : {"etat": "tenue"} ; {"etat": "posee", "x", "y",
        #: "r"} ; {"etat": "absente"}
        self.piece: dict = {"etat": "absente"}
        #: fichier tenu a jour apres chaque pose (reprise apres interruption)
        self.chemin: Optional[str] = None
        #: points de pose refuses par le controleur : on n'y retourne pas
        self.interdits: List[Tuple[float, float]] = []
        self.vise: Optional[Tuple[float, float, float]] = None
        from dobot_sim import DobotMagician
        self.modele = getattr(robot, "modele", None) or DobotMagician()

    # -- mouvements ---------------------------------------------------------
    def _z(self, h: float) -> float:
        return self.z_table + h

    def _enchainer(self, mouvements):
        """Une suite de mouvements sans arret inutile (Commande.enchainer) ;
        a defaut, l'un apres l'autre."""
        if hasattr(self.robot, "enchainer"):
            self.robot.enchainer(mouvements)
        else:
            for t, kw in mouvements:
                getattr(self.robot, t)(**kw)

    def _approche(self, x, y, r) -> list:
        """
        De la ou est le bras jusqu'a la pince a PRISE en (x, y), orientee r :
        remontee verticale si besoin, transit, puis descente VERTICALE. Le
        transit est rectiligne si l'outil ne tourne pas (tout s'enchaine
        alors sans arret), articulaire sinon (arret au-dessus, le mode
        continu ne tournant pas l'outil).
        """
        xa, ya, za, ra = self.robot.pose()
        m = []
        if za < self._z(TRANSIT) - 1.0:
            m.append(("movl", dict(x=xa, y=ya, z=self._z(TRANSIT), r=ra)))
        tourne = abs(_angle(r - ra)) > 0.5
        m.append(("movj" if tourne else "movl", dict(x=x, y=y, z=self._z(TRANSIT), r=r)))
        m.append(("movl", dict(x=x, y=y, z=self._z(PRISE), r=r)))
        return m

    def _aller(self, x, y, r):
        """Au-dessus de (x, y), pince orientee r, a la hauteur de transit."""
        self._enchainer(self._approche(x, y, r)[:-1])

    def poser(self, x, y, r, remonter: bool = True):
        self._enchainer(self._approche(x, y, r))
        self.robot.pince(False)
        self.piece = {"etat": "posee", "x": float(x), "y": float(y), "r": float(r)}
        self.robot.attendre(DELAI_PINCE)
        if remonter:
            self.robot.movl(x=x, y=y, z=self._z(TRANSIT), r=r)

    def saisir(self, x, y, r, remonter: bool = True):
        self._enchainer(self._approche(x, y, r))
        self.robot.pince(True)
        self.piece = {"etat": "tenue"}
        self.robot.attendre(DELAI_PINCE)
        if remonter:
            self.robot.movl(x=x, y=y, z=self._z(TRANSIT), r=r)

    def degager(self):
        """
        Ecarte le bras du champ de la camera, d'un seul mouvement continu :
        remontee VERTICALE au-dessus de la piece posee, puis posture
        d'EQUERRE (bras arriere presque vertical, avant-bras horizontal),
        base et poignet inchanges -- l'outil ne tourne pas, rien n'oblige a
        s'arreter entre les deux. Ou la place de garage donnee.
        """
        xa, ya, za, ra = self.robot.pose()
        m = []
        if za < self._z(TRANSIT) - 1.0:
            m.append(("movl", dict(x=xa, y=ya, z=self._z(TRANSIT), r=ra)))
        if self.garage is not None:
            x, y, z = self.garage
            m.append(("movj", dict(x=x, y=y, z=z, r=ra)))
        else:
            q = [float(v) for v in self.modele.mgi_pose(xa, ya, self._z(TRANSIT), ra)]
            m.append(("movj", dict(q=[q[0], EQUERRE_J2, EQUERRE_J3, q[3]])))
        self._enchainer(m)

    # -- choix des points ---------------------------------------------------
    def j1(self, x, y, z, r) -> float:
        """Rotation de la base (deg) pour la pose (x, y, z, r)."""
        try:
            return float(self.modele.mgi_pose(x, y, z, r)[0])
        except Exception:                           # noqa: BLE001
            return math.degrees(math.atan2(y, x))

    def dans_secteur(self, x, y, z, r, marge: float = MARGE_SECTEUR) -> bool:
        if self.secteur is None:
            return True
        a = self.j1(x, y, z, r)
        return self.secteur[0] + marge <= a <= self.secteur[1] - marge

    def _atteignable(self, x, y, r) -> bool:
        """Pose et transit atteints avec MARGE_BUTEE sur chaque articulation,
        dans le secteur de la base."""
        for h in (PRISE, TRANSIT):
            z = self._z(h)
            try:
                q = self.modele.mgi_pose(x, y, z, r)
            except Exception:                       # noqa: BLE001
                return False
            if not self.modele.dans_butees(q) or np.min(self.modele.marges(q)) < MARGE_BUTEE:
                return False
            if not self.dans_secteur(x, y, z, r):
                return False
        return True

    def _visible(self, x, y, marge=60) -> bool:
        K, taille = getattr(self.camera, "K", None), getattr(self.camera, "taille", None)
        if self.recalage is None or K is None or taille is None:
            return True
        for dx, dy in ((-40, 0), (40, 0), (0, -40), (0, 40), (0, 0)):
            p = self.recalage.vers_camera([x + dx, y + dy, self._z(HAUTEUR_PIECE)])
            if p[2] <= 50.0:
                return False
            u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
            if not (marge <= u <= taille[0] - marge and marge <= v <= taille[1] - marge):
                return False
        return True

    def point_suivant(self) -> Tuple[float, float, float]:
        """
        Le prochain point de pose : pres du depart tant qu'il n'y a pas de
        recalage, puis le plus loin possible des points deja pris, dans la
        zone vue ET atteinte. L'orientation varie a chaque fois.
        """
        x0, y0, r0 = self.depart
        deja = np.array([p.pince[:2] for p in self.paires] + [[x0, y0]])
        meilleur, score = None, -1.0
        for _ in range(400):
            if self.recalage is None or len(self.paires) < PAIRES_MIN:
                a = self.rng.uniform(0, 2 * math.pi)
                rho = RAYON_DEPART * math.sqrt(self.rng.uniform(0.2, 1.0))
                x, y = x0 + rho * math.cos(a), y0 + rho * math.sin(a)
            else:
                x = self.rng.uniform(*self.zone_x)
                y = self.rng.uniform(*self.zone_y)
            r = _angle(r0 + self.rng.uniform(-self.r_max, self.r_max))
            if not self._atteignable(x, y, r) or not self._visible(x, y):
                continue
            if any(math.hypot(x - u, y - v) < 25.0 for u, v in self.interdits):
                continue
            s = float(np.min(np.linalg.norm(deja - [x, y], axis=1)))
            if s > score:
                meilleur, score = (x, y, r), s
        if meilleur is None:
            raise Arret("aucun point a la fois vu par la camera et atteint par "
                        "le bras")
        return meilleur

    # -- la camera ----------------------------------------------------------
    def observer(self, attendu_robot=None) -> Optional[Vue]:
        """
        La piece posee, parmi ce que voit la camera ; None si on ne peut pas
        la designer (rien de vu, ou plusieurs candidates qu'on ne sait pas
        departager) : l'appelant abandonne la pose et continue. On ne
        demande jamais rien : une detection en trop (reflet, objet au bord)
        ne doit pas arreter l'etalonnage.
        """
        vues = list(self.camera.voir())
        # une meme piece detectee deux fois : on garde la plus sure
        vues.sort(key=lambda v: -v.confiance)
        uniques: List[Vue] = []
        for v in vues:
            if all(np.linalg.norm(v.position - u.position) > 20.0 for u in uniques):
                uniques.append(v)
        vues = uniques
        self.derniere_observation = len(vues)
        if len(vues) <= 1:
            return vues[0] if vues else None
        if attendu_robot is not None and self.recalage is not None:
            prevu = self.recalage.vers_camera(attendu_robot)
            vues.sort(key=lambda v: np.linalg.norm(v.position - prevu))
            d0 = np.linalg.norm(vues[0].position - prevu)
            d1 = np.linalg.norm(vues[1].position - prevu)
            return vues[0] if d0 < ECART_MAX and d1 > d0 + 20.0 else None
        # pas encore de recalage : la bonne espece de piece...
        bonnes = [v for v in vues if v.type == TYPE_PIECE and v.face == FACE_PIECE]
        if len(bonnes) == 1:
            return bonnes[0]
        # ... sinon la plus proche de la piece vue a la pose precedente
        # (les premieres poses restent a quelques centimetres les unes des
        # autres)
        if self.paires:
            ref = self.paires[-1].camera
            vues.sort(key=lambda v: np.linalg.norm(v.position - ref))
            d0 = np.linalg.norm(vues[0].position - ref)
            d1 = np.linalg.norm(vues[1].position - ref)
            if d1 > d0 + 40.0:
                return vues[0]
        return None

    # -- le deroule ---------------------------------------------------------
    def demarrer(self):
        """Apres l'accostage : la pince est ouverte a PRISE au-dessus de la
        table ; on y place la piece, puis on ferme."""
        x, y, z, r = self.robot.pose()
        if self.z_table is None:
            self.z_table = z - PRISE
        self.depart = (x, y, r)
        self._placer_a_la_main(x, y, r)

    def _placer_a_la_main(self, x, y, r):
        """La pince ouverte a PRISE en (x, y) : l'utilisateur y place la
        piece, la pince se ferme, le bras remonte."""
        self.demander("  placez une EPAISSE A PLAT entre les mors, posee sur la "
                      "table, puis Entree ")
        self.robot.pince(True)
        self.piece = {"etat": "tenue"}
        self.decalage_courant = None
        self.robot.attendre(DELAI_PINCE)
        self.robot.movl(x=x, y=y, z=self._z(TRANSIT), r=r)

    def nouvelle_piece(self):
        """
        Une NOUVELLE piece (ou la meme, replacee a la main) : nouvelle
        session, donc nouveau decalage dans la pince. Le bras va au point
        de depart, descend a la verticale, ouvre ; on y place la piece.
        """
        if self.paires:
            self.session = max(p.session for p in self.paires) + 1
        self._recentrer_depart()
        x, y, r = self.depart
        self._aller(x, y, r)
        self.robot.movl(x=x, y=y, z=self._z(PRISE), r=r)
        self.robot.pince(False)
        self.piece = {"etat": "absente"}
        self.robot.attendre(DELAI_PINCE)
        self.journal("  piece n {} : retirez de la table toute autre piece"
                     .format(self.session + 1))
        self._placer_a_la_main(x, y, r)

    def _recentrer_depart(self):
        """Point de depart hors du secteur (secteur releve a nouveau) : on le
        ramene au milieu du secteur, a la meme distance de la base."""
        x, y, r = self.depart
        if self._atteignable(x, y, r) or self.secteur is None:
            return
        rayon = math.hypot(x, y)
        milieu = math.radians(0.5 * (self.secteur[0] + self.secteur[1]))
        for dr in (0, -10, 10, -20, 20, -30, 30, -40, 40):
            nx, ny = (rayon + dr) * math.cos(milieu), (rayon + dr) * math.sin(milieu)
            nr = _angle(r + math.degrees(milieu - math.atan2(y, x)))
            if self._atteignable(nx, ny, nr):
                self.depart = (nx, ny, nr)
                self.journal("  point de depart ramene dans le secteur : x {:.0f} y {:.0f}"
                             .format(nx, ny))
                return
        raise Arret("aucun point de depart atteignable dans le secteur")

    def remettre_piece(self):
        """A la reprise : le robot ouvre la pince au point de depart, et on
        lui remet une piece entre les mors (nouvelle session)."""
        self.nouvelle_piece()

    def reprendre_apres_defaut(self) -> bool:
        """
        Un mouvement refuse par le controleur (cible hors butees...) n'est
        pas grave en soi : on efface le defaut, on bannit ce point, et on
        continue ailleurs. Rend False si le controleur reste en defaut
        (certains defauts ne s'effacent qu'avec une prise d'origine).
        """
        pont = getattr(self.robot, "_pont", None)
        li = getattr(pont, "liaison", None)
        if li is None:
            return False
        try:
            li.arret_urgence()
            li.effacer_alarmes()
            time.sleep(0.3)
            reste = li.alarmes()
        except Exception:                           # noqa: BLE001
            return False
        if self.vise is not None:
            self.interdits.append(self.vise[:2])
            self.journal("  point x {:.0f} y {:.0f} refuse par le robot : on n'y "
                         "retournera pas".format(*self.vise[:2]))
        if reste:
            return False
        if self.piece.get("etat") == "posee":
            self.saisir(self.piece["x"], self.piece["y"], self.piece["r"])
        return True

    def reposer_et_lacher(self):
        """Apres une interruption, piece en main : la reposer sur la table
        (la pompe s'arrete), la ou elle est, sinon au point de depart."""
        x, y, _z, r = self.robot.pose()
        if not self._atteignable(x, y, r):
            x, y, r = self.depart
        self.poser(x, y, r)

    def cycle(self) -> Optional[Paire]:
        x, y, r = self.point_suivant()
        self.vise = (x, y, r)
        self.poser(x, y, r, remonter=False)          # degager remonte, sans arret
        self.degager()
        attendu = np.array([x, y, self._z(HAUTEUR_PIECE)])
        vue = self.observer(attendu)
        self._archiver(x, y, r, vue)
        if vue is None:
            # hors du champ, cachee, ou ambigue : on le dit, et on continue
            n = getattr(self, "derniere_observation", 0)
            if n == 0:
                self.interdits.append((x, y))
                quoi = "piece NON VUE (hors champ ?) : point abandonne"
            else:
                quoi = "{} pieces vues, impossible de savoir laquelle : pose ignoree".format(n)
            self.journal("  --  pose x {:6.1f} y {:6.1f} r {:6.1f}  | {}, on continue"
                         .format(x, y, r, quoi))
            self.saisir(x, y, r, remonter=False)
            self.enregistrer()
            return None
        paire = Paire(np.array([x, y, self._z(PRISE)]), r, vue.position, vue.axe,
                      self.z_table, self.session)
        prevu = None
        if self.recalage is not None:
            prevu = self.recalage.prise(vue.position, vue.axe)
        self.paires.append(paire)
        if len(self.paires) >= 3:
            self.recalage = recaler(self.paires)
        n = len(self.paires)
        texte = "  {:2d}  pose x {:6.1f} y {:6.1f} r {:6.1f}".format(n, x, y, r)
        if vue.type and (vue.type != TYPE_PIECE or vue.face != FACE_PIECE):
            texte += "  (vue : {} {} ?)".format(vue.type, vue.face)
        if prevu is not None:
            ex = np.linalg.norm(prevu[0][:2] - [x, y])
            texte += "  | prevision camera : ecart {:4.1f} mm, {:+5.1f} deg".format(
                ex, _angle(prevu[1] - r, 180.0))
        if self.recalage is not None:
            texte += "  | recalage : {:.2f} mm rms".format(self.recalage.rms)
        self.journal(texte)
        self.enregistrer()
        # la remontee est faite par la pose suivante, enchainee avec le transit
        self.saisir(x, y, r, remonter=False)
        self.enregistrer()
        return paire

    def _archiver(self, x, y, r, vue) -> None:
        """La capture qu'on vient de faire, pour entrainer la perception."""
        archiver = getattr(self.camera, "archiver", None)
        if archiver is None:
            return
        try:
            archiver({"pince": [float(x), float(y), float(self._z(PRISE))], "r": float(r),
                      "z_table": float(self.z_table), "session": int(self.session),
                      "piece": {"type": TYPE_PIECE, "face": FACE_PIECE,
                                "hauteur": HAUTEUR_PIECE},
                      "retenue": vue is not None,
                      "camera_retenue": None if vue is None
                      else [float(v) for v in vue.position]})
        except Exception as exc:                        # noqa: BLE001
            self.journal("  (capture non archivee : {})".format(exc))

    def valider(self, n: int = 3) -> List[dict]:
        """
        Saisies guidees par la CAMERA : poser, regarder, et reprendre la ou
        le recalage dit que la piece est. L'ecart avec la position connue
        mesure ce que vaudra une vraie saisie.
        """
        self.journal("\nVALIDATION : saisies guidees par la camera seule")
        for k in range(n):
            x, y, r = self.point_suivant()
            self.poser(x, y, r, remonter=False)
            self.degager()
            vue = self.observer(np.array([x, y, self._z(HAUTEUR_PIECE)]))
            if vue is None:
                self.interdits.append((x, y))
                self.journal("  {}  piece NON VUE en x {:.0f} y {:.0f} : reprise a la "
                             "position connue".format(k + 1, x, y))
                self.saisir(x, y, r, remonter=False)
                continue
            (px, py, _pz), pr = self.recalage.prise(vue.position, vue.axe)
            # r a 180 deg pres : on garde le plus proche de la pose
            if abs(_angle(pr - r)) > 90:
                pr = _angle(pr + 180.0)
            # le centre de la piece, connu : la pince + son decalage
            d = self.recalage.decalages.get(self.session, np.zeros(2)) \
                if self.decalage_courant is None else self.decalage_courant
            cx, cy, _cz = np.array([x, y, 0.0]) + rot_z(r) @ [d[0], d[1], 0.0]
            e = float(np.hypot(px - cx, py - cy))
            de = _angle(pr - r, 180.0)
            self.validations.append({"pose": [x, y, r], "centre": [cx, cy],
                                     "prise": [px, py, pr],
                                     "ecart_mm": e, "ecart_deg": de})
            self.journal("  {}  posee ({:.1f}, {:.1f}, {:.1f})  saisie camera ({:.1f}, "
                         "{:.1f}, {:.1f})  ecart {:.1f} mm {:+.1f} deg".format(
                             k + 1, x, y, r, px, py, pr, e, de))
            if e > 6.0 or not self._atteignable(px, py, pr):
                self.journal("    {} : reprise a la position connue".format(
                    "ecart trop grand pour saisir sans risque" if e > 6.0
                    else "prise hors d'atteinte"))
                self.saisir(x, y, r, remonter=False)
            else:
                # visee au centre : la piece est desormais centree dans la
                # pince, le decalage ne vaut plus pour les poses suivantes
                self.saisir(px, py, pr, remonter=False)
                self.decalage_courant = np.zeros(2)
        return self.validations

    def resultat(self) -> dict:
        """Tout ce qu'il faut pour utiliser l'etalonnage, et pour le
        REPRENDRE : mesures, secteur, hauteur de table, ou est la piece."""
        out = {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "z_table_mm": None if self.z_table is None else float(self.z_table),
            "secteur_base_deg": None if self.secteur is None else list(self.secteur),
            "depart": None if self.depart is None else [float(v) for v in self.depart],
            "session": int(self.session),
            "piece": self.piece,
            "paires": len(self.paires),
            "validation": self.validations,
            "camera_K": None if getattr(self.camera, "K", None) is None
            else np.asarray(self.camera.K, float).tolist(),
            "camera_taille": None if getattr(self.camera, "taille", None) is None
            else [int(v) for v in self.camera.taille],
            "points_abandonnes": [[float(u), float(v)] for u, v in self.interdits],
            "mesures": [{"pince": [float(v) for v in p.pince], "r": float(p.r),
                         "camera": [float(v) for v in p.camera],
                         "axe_camera": [float(v) for v in p.lacet_camera],
                         "z_table": float(p.z_table), "session": int(p.session)}
                        for p in self.paires],
        }
        rc = self.recalage
        if rc is not None:
            out.update({
                "T_robot_camera": rc.T_m.tolist(),
                "unite": "metres (repere de la camera de PROFONDEUR -> repere robot "
                         "du simulateur dobot_sim)",
                "T_robot_camera_mm": rc.T_mm.tolist(),
                "decalages_pince_mm": {str(k): [float(v) for v in d]
                                       for k, d in rc.decalages.items()},
                "ecart_r_deg": float(rc.ecart_r),
                "rms_mm": rc.rms,
                "paires_retenues": int(rc.gardees.sum()),
                "residus_mm": [float(v) for v in rc.residus],
            })
        return out

    def enregistrer(self, chemin: Optional[str] = None) -> Optional[str]:
        chemin = chemin or self.chemin
        if chemin is None or self.z_table is None:
            return None
        tmp = chemin + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.resultat(), f, indent=2)
        os.replace(tmp, chemin)
        return chemin

    def restaurer(self, data: dict) -> None:
        """Reprend un etalonnage enregistre (voir resultat())."""
        self.z_table = data["z_table_mm"]
        if data.get("secteur_base_deg"):
            self.secteur = tuple(data["secteur_base_deg"])
        if data.get("depart"):
            self.depart = tuple(data["depart"])
        self.paires = [Paire(np.array(m["pince"], float), float(m["r"]),
                             np.array(m["camera"], float), np.array(m["axe_camera"], float),
                             float(m.get("z_table", self.z_table)), int(m.get("session", 0)))
                       for m in data.get("mesures", [])]
        self.session = int(data.get("session", 0))
        self.piece = dict(data.get("piece") or {"etat": "absente"})
        self.validations = list(data.get("validation", []))
        self.interdits = [tuple(p) for p in data.get("points_abandonnes", [])]
        # le champ de la camera, pour ne pas viser hors champ avant la
        # premiere image de la reprise
        if self.camera is not None and data.get("camera_K") is not None \
                and getattr(self.camera, "K", None) is None:
            self.camera.K = np.array(data["camera_K"], float)
            self.camera.taille = tuple(data["camera_taille"])
        self.recalage = recaler(self.paires) if len(self.paires) >= 3 else None


def charger_etalonnage(chemin: str = FICHIER) -> np.ndarray:
    """T_robot_camera (4x4, METRES) : a donner a Perception(T_robot_camera=...)."""
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    if "T_robot_camera" not in data:
        raise ValueError("etalonnage inacheve ({} poses) : terminez-le avec "
                         "etalonner_camera_robot.py --reprendre".format(data.get("paires", 0)))
    return np.array(data["T_robot_camera"], float)


# ---------------------------------------------------------------------------
# Accostage interactif
# ---------------------------------------------------------------------------

AIDE_CLAVIER = """
  Le bras se deplace a la main (bouton de deverrouillage de l'avant-bras,
  puis relachez-le) ou au clavier :
     fleches haut / bas   monter / descendre (z)
     x X  y Y             deplacer en x / y (minuscule : moins)
     b B                  tourner la base
     r R                  tourner la pince
     1  2  3              pas de 10 / 1 / 0,2 mm (5 / 1 / 0,2 deg)
     o                    ouvrir la pince
     Entree               valider la position
     q                    quitter
"""


def _lire_touche() -> str:
    """Une touche, sans Entree (terminal), ou une ligne sinon."""
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip() or "\n"
    import termios
    import tty
    _vider_clavier()
    fd = sys.stdin.fileno()
    ancien = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        c = sys.stdin.read(1)
        if c == "\x1b":
            c += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, ancien)
    if c == "\x03":
        raise KeyboardInterrupt
    return {"\x1b[A": "haut", "\x1b[B": "bas", "\r": "\n"}.get(c, c)


def piloter(robot, lire: Callable[[], str] = _lire_touche,
            journal: Callable[[str], None] = print,
            autorise: Optional[Callable[[float, float, float, float], bool]] = None
            ) -> Tuple[float, float, float, float]:
    """
    Pilotage au clavier (voir AIDE_CLAVIER) jusqu'a Entree ; rend la pose
    a cet instant. `autorise(x, y, z, r)` refuse les deplacements qui
    sortiraient du secteur.
    """
    pas = {"1": (10.0, 5.0), "2": (1.0, 1.0), "3": (0.2, 0.2)}
    p_mm, p_deg = pas["2"]
    while True:
        x, y, z, r = robot.pose()
        base = math.degrees(math.atan2(y, x))
        sys.stdout.write("\r  x {:6.1f}  y {:6.1f}  z {:6.1f}  r {:6.1f}  base {:6.1f} deg"
                         "   pas {:.1f}   ".format(x, y, z, r, base, p_mm))
        sys.stdout.flush()
        c = lire()
        d = {"haut": (0, 0, 1, 0), "bas": (0, 0, -1, 0), "x": (-1, 0, 0, 0),
             "X": (1, 0, 0, 0), "y": (0, -1, 0, 0), "Y": (0, 1, 0, 0),
             "r": (0, 0, 0, -1), "R": (0, 0, 0, 1)}.get(c)
        cible = None
        if d is not None:
            cible = (x + d[0] * p_mm, y + d[1] * p_mm, z + d[2] * p_mm, r + d[3] * p_deg)
        elif c in ("b", "B"):
            # rotation de la base : meme rayon, meme hauteur, angle +/- pas
            a = math.radians((1 if c == "B" else -1) * p_deg)
            cible = (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a),
                     z, r + math.degrees(a))
        if cible is not None:
            if autorise is not None and not autorise(*cible):
                journal("\n  refuse : hors du secteur de la base")
                continue
            try:
                robot.movl(x=cible[0], y=cible[1], z=cible[2], r=cible[3])
            except Exception as exc:                     # noqa: BLE001
                journal("\n  refuse : {}".format(str(exc).splitlines()[0]))
        elif c in pas:
            p_mm, p_deg = pas[c]
        elif c == "o":
            robot.pince(False)
        elif c == "\n":
            journal("")
            return tuple(float(v) for v in robot.pose())
        elif c in ("q", "Q"):
            raise Arret("arret demande")


def _angle_base(robot, modele) -> float:
    """J1 lu sur le robot (deg)."""
    try:
        return float(robot.angles()[0])
    except Exception:                               # noqa: BLE001
        x, y, z, r = robot.pose()
        return float(modele.mgi_pose(x, y, z, r)[0])


def relever_secteur(robot, modele, lire: Callable[[], str] = _lire_touche,
                    journal: Callable[[str], None] = print) -> Tuple[float, float]:
    """Les deux rotations extremes de la base, montrees l'une apres l'autre."""
    journal(AIDE_CLAVIER)
    journal("  Secteur angulaire : la base ne devra pas tourner au-dela de deux limites, "
            "pour ne pas heurter\n  la camera. Amenez le bras (a la main ou "
            "avec b / B) a la PREMIERE limite, puis Entree.")
    piloter(robot, lire, journal)
    a = _angle_base(robot, modele)
    journal("  premiere limite : base a {:.1f} deg. Amenez-le maintenant a la SECONDE "
            "limite, puis Entree.".format(a))
    piloter(robot, lire, journal)
    b = _angle_base(robot, modele)
    secteur = (min(a, b), max(a, b))
    journal("  secteur de la base : de {:.1f} a {:.1f} deg ({:.0f} deg)".format(
        secteur[0], secteur[1], secteur[1] - secteur[0]))
    if secteur[1] - secteur[0] < 2 * MARGE_SECTEUR + 10.0:
        raise Arret("secteur trop etroit ({:.0f} deg)".format(secteur[1] - secteur[0]))
    if secteur[1] - secteur[0] > 180.0:
        journal("  attention : un secteur de plus de 180 deg laisse les lignes droites "
                "en sortir")
    return secteur


def accoster(robot, lire: Callable[[], str] = _lire_touche,
             journal: Callable[[str], None] = print,
             autorise: Optional[Callable[[float, float, float, float], bool]] = None
             ) -> float:
    """Approche interactive ; rend le z de la pointe au contact de la table."""
    journal("  ACCOSTAGE : amenez la pince OUVERTE au contact de la table, n'importe "
            "ou dans le secteur.\n  Glissez une feuille sous les mors : quand elle "
            "est pincee, Entree.")
    robot.pince(False)
    return float(piloter(robot, lire, journal, autorise)[2])


# ---------------------------------------------------------------------------
# Programme
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Etalonnage camera -> robot par poses "
                                             "successives d'une piece")
    ap.add_argument("--points", type=int, default=0,
                    help="nombre de poses ; 0 (defaut) : sans fin, jusqu'a Ctrl+C")
    ap.add_argument("--validations", type=int, default=3,
                    help="saisies guidees par la camera apres les poses (3) ; "
                         "seulement avec --points N")
    ap.add_argument("--vitesse", type=float, default=40.0, help="mm/s (40)")
    ap.add_argument("--longueur-outil", type=float, default=None,
                    help="longueur mesuree de la pince, mm")
    ap.add_argument("--garage", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="ou ecarter le bras pendant la capture (defaut : posture "
                         "d'equerre, base inchangee)")
    ap.add_argument("--fenetre", action="store_true",
                    help="passer par le simulateur ouvert et connecte au robot")
    ap.add_argument("--simulation", action="store_true",
                    help="simulateur seul, piece et camera simulees : pour essayer")
    ap.add_argument("--reprendre", action="store_true",
                    help="reprendre l'etalonnage enregistre : secteur, table et "
                         "mesures gardes ; la piece est reprise ou remplacee")
    ap.add_argument("--recommencer", action="store_true",
                    help="nouvel etalonnage SANS la configuration interactive : "
                         "secteur, hauteur de table et point de depart repris du "
                         "fichier, mesures effacees (apres un deplacement de la camera)")
    ap.add_argument("--secteur", action="store_true",
                    help="avec --reprendre ou --recommencer : relever a nouveau le "
                         "secteur angulaire de la base (les poses sont gardees)")
    ap.add_argument("--sortie", default=FICHIER)
    a = ap.parse_args(argv)
    if a.reprendre and a.recommencer:
        ap.error("--reprendre ou --recommencer, pas les deux")

    donnees = None
    if a.reprendre or a.recommencer:
        if not os.path.exists(a.sortie):
            print("rien a reprendre : {} n'existe pas".format(os.path.relpath(a.sortie)))
            return 1
        with open(a.sortie, encoding="utf-8") as f:
            donnees = json.load(f)
        if a.recommencer:
            piece = donnees.get("piece")
            donnees = {k: donnees.get(k) for k in ("z_table_mm", "secteur_base_deg",
                                                   "depart")}
            donnees["piece"] = piece

    from dobot_sim import Commande, ErreurExecution
    if a.simulation:
        commande = Commande(simulateur=True, robot=False, outil="pince",
                            vitesse=a.vitesse, lancer_simulateur=True,
                            longueur_outil=a.longueur_outil)
    else:
        commande = Commande(simulateur=bool(a.fenetre), robot=True, outil="pince",
                            vitesse=a.vitesse, longueur_outil=a.longueur_outil)
    robot, camera = commande, None
    if a.simulation:
        robot = PieceSimulee(commande)
        camera = CameraSimulee(robot)
    else:
        print("chargement de la perception...")
        camera = Camera()

    et = Etalonnage(robot, camera, garage=a.garage)
    et.chemin = a.sortie
    try:
        if donnees is not None:
            et.restaurer(donnees)
            print("{} : {} poses deja faites, table a z = {:.1f} mm, secteur de la "
                  "base {}".format("NOUVEL ETALONNAGE" if a.recommencer else "REPRISE",
                                   len(et.paires), et.z_table,
                                   "de {:.1f} a {:.1f} deg".format(*et.secteur)
                                   if et.secteur else "libre"))
            if et.recalage is not None:
                print("  recalage actuel : {:.2f} mm rms".format(et.recalage.rms))
            if getattr(camera, "K", None) is None:
                # le champ de la camera n'est pas dans le fichier (ancien) :
                # une image suffit a l'apprendre
                try:
                    camera.voir()
                except Exception as exc:                # noqa: BLE001
                    print("  (champ de la camera inconnu : {})".format(exc))
            if a.secteur:
                et.secteur = relever_secteur(robot, et.modele)
                et.enregistrer()
            et.remettre_piece()
        else:
            if a.simulation:
                # pas de main pour guider le bras : on part d'un point connu
                commande.movj(x=230, y=40, z=60, r=0)
                print("simulation : la table est a z = 0 ; tournez la base avec b / B, "
                      "descendez avec la fleche bas")
            et.secteur = relever_secteur(robot, et.modele)
            if a.garage is not None and not et.dans_secteur(a.garage[0], a.garage[1],
                                                            a.garage[2], 0.0, 0.0):
                raise Arret("le garage {} est hors du secteur".format(a.garage))
            z_contact = accoster(robot, autorise=lambda x, y, z, r:
                                 et.dans_secteur(x, y, z, r, 0.0))
            et.z_table = z_contact
            x, y, _z, r = robot.pose()
            robot.movl(x=x, y=y, z=z_contact + PRISE, r=r)
            print("table a z = {:.1f} mm (repere robot)".format(z_contact))
            et.demarrer()
            et.enregistrer()
        print("\nETALONNAGE : {} (Ctrl+C pour arreter : la piece est reposee, tout "
              "est enregistre)".format("{} poses".format(a.points) if a.points
                                       else "poses sans fin"))
        faits, echecs = 0, 0
        while a.points <= 0 or faits < a.points:
            try:
                et.cycle()
                faits += 1
            except ErreurExecution:
                echecs += 1
                if echecs > 5 or not et.reprendre_apres_defaut():
                    raise
                print("  defaut efface, on continue ailleurs")
        rc = et.recalage
        print("\nrecalage : {:.2f} mm rms sur {} paires ({} retenues) ; decalage dans la "
              "pince ({:.1f}, {:.1f}) mm ; ecart d'angle {:.1f} deg".format(
                  rc.rms, len(et.paires), int(rc.gardees.sum()), rc.decalage[0],
                  rc.decalage[1], rc.ecart_r))
        if a.validations:
            et.valider(a.validations)
        x, y, r = et.point_suivant()
        et.poser(x, y, r, remonter=False)
        et.degager()
    except KeyboardInterrupt:
        interrompu(et)
    except Arret as exc:
        print("\narret : {}".format(exc))
    except ErreurExecution as exc:
        # l'alerte detaillee est deja affichee par Commande
        print("\nETALONNAGE INTERROMPU : {}".format(exc))
        if et.piece.get("etat") == "tenue":
            print("  la piece est sans doute encore dans la pince (pompe en marche).")
        print("  Le controleur reste en defaut. Une alarme de PLANIFICATION (0x1x)\n"
              "  bloque sa file sur ce firmware : ni l'effacement ni la prise d'origine\n"
              "  ne partent. Il faut ETEINDRE ET RALLUMER le robot (bras tenu, mis en\n"
              "  equerre a la main), puis la mise en route complete :\n"
              "    .venv/bin/python dobot.py mise-en-route\n"
              "  (ATTENTION : la base tourne pendant la prise d'origine ; surveillez "
              "la camera),\n  puis --reprendre.")
    finally:
        chemin = et.enregistrer()
        if chemin and et.recalage is not None:
            print("etalonnage enregistre : {} ({} poses, {:.2f} mm rms)".format(
                os.path.relpath(chemin), len(et.paires), et.recalage.rms))
            print("  Perception(T_robot_camera=charger_etalonnage())")
        elif chemin:
            print("etat enregistre : {} ({} poses, pas encore de recalage)".format(
                os.path.relpath(chemin), len(et.paires)))
        if chemin:
            print("  pour continuer : etalonner_camera_robot.py --reprendre")
        commande.fermer()
    return 0


def interrompu(et: Etalonnage) -> None:
    """
    Ctrl+C : le robot s'est arrete sur place (arret d'urgence). Piece en
    main, on propose de la reposer sur la table -- pince ouverte, la pompe
    s'arrete -- pour que la reprise la retrouve. Un second Ctrl+C quitte
    sans rien faire de plus.
    """
    print("\n\nINTERRUPTION : robot arrete. {} poses mesurees.".format(len(et.paires)))
    try:
        if et.piece.get("etat") == "posee":
            print("  la piece est posee sur la table en x {:.0f} y {:.0f}".format(
                et.piece["x"], et.piece["y"]))
            return
        if et.piece.get("etat") != "tenue" or et.z_table is None:
            return
        rep = et.demander("  la piece est dans la pince. [Entree : la reposer sur la table "
                          "(la pompe s'arrete) / g : la garder en main] ")
        if rep.strip().lower().startswith("g"):
            print("  piece gardee en main : la pompe continue de tourner")
            return
        et.reposer_et_lacher()
        print("  piece reposee en x {:.0f} y {:.0f}".format(et.piece["x"], et.piece["y"]))
    except KeyboardInterrupt:
        print("\n  arret immediat")
    except Exception as exc:                            # noqa: BLE001
        print("  le robot n'a pas pu reposer la piece : {}".format(
            str(exc).splitlines()[0] if str(exc) else exc))


if __name__ == "__main__":
    sys.exit(main())
