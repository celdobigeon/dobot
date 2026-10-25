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

  1. ACCOSTAGE (interactif) : amener la pince OUVERTE au contact de la
     table, n'importe ou -- a la main (bouton de deverrouillage de
     l'avant-bras) puis au clavier. Le contact donne la hauteur de la
     table dans le repere du robot ;
  2. le bras remonte de 5 mm ; on place la piece entre les mors, posee sur
     la table ; Entree : la pince se ferme ;
  3. boucle : poser, s'ecarter, voir, reprendre. Les premiers points
     restent pres du depart ; des que le recalage existe, ils couvrent la
     partie de la table que la camera voit et que le bras atteint ;
  4. VALIDATION : quelques saisies guidees par la camera seule ;
  5. le resultat est ecrit dans etalonnage_camera_robot.json, que
     `charger_etalonnage()` relit pour Perception(T_robot_camera=...).

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
#: mm : hauteur des deplacements au-dessus de la table
TRANSIT = 70.0
#: les premiers points restent a cette distance du depart (mm)...
RAYON_DEPART = 30.0
#: ...tant qu'il n'y a pas assez de paires pour un recalage
PAIRES_MIN = 4
#: s : temps laisse aux mors pour se fermer ou s'ouvrir
DELAI_PINCE = 0.8
#: mm : au-dela, une piece vue n'est pas celle qu'on a posee
ECART_MAX = 25.0


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

    @property
    def dessus(self) -> np.ndarray:
        """Centre de la face superieure SOUS la pince, repere robot (sans
        le decalage dans la pince)."""
        return np.array([self.pince[0], self.pince[1], self.z_table + HAUTEUR_PIECE])


@dataclass
class Recalage:
    R: np.ndarray                     # camera -> robot
    t: np.ndarray                     # mm
    decalage: np.ndarray              # (dx, dy) mm, piece dans le repere pince
    ecart_r: float                    # deg : r de la pince - lacet piece (mod 180)
    residus: np.ndarray               # mm, par paire
    gardees: np.ndarray               # masque des paires retenues

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
            iterations: int = 30) -> Recalage:
    """
    T_robot_camera et decalage dans la pince, par moindres carres alternes :
    Kabsch a decalage fixe, puis decalage a recalage fixe. Avec 6 paires et
    plus, celles dont le residu depasse 3 fois la mediane (et 3 mm) sont
    ecartees, puis le calcul est refait.
    """
    if len(paires) < 3:
        raise ValueError("au moins 3 paires")
    P = np.array([p.camera for p in paires], float)
    G = np.array([p.dessus for p in paires], float)
    Rz = [rot_z(p.r) for p in paires]
    # le decalage n'est identifiable que si r varie
    rs = np.radians([p.r for p in paires])
    varie = np.hypot(np.cos(rs).std(), np.sin(rs).std()) > 0.15
    gardees = np.ones(len(paires), bool)
    for _passe in range(2):
        d = np.zeros(2)
        for _ in range(iterations):
            Q = G + np.array([Rz[i] @ [d[0], d[1], 0.0] for i in range(len(paires))])
            R, t = kabsch(P[gardees], Q[gardees])
            if not (decalage and varie):
                break
            e = P @ R.T + t - G
            loc = np.array([Rz[i].T @ e[i] for i in range(len(paires))])
            d_nouv = loc[gardees, :2].mean(axis=0)
            if np.linalg.norm(d_nouv - d) < 1e-4:
                d = d_nouv
                break
            d = d_nouv
        Q = G + np.array([Rz[i] @ [d[0], d[1], 0.0] for i in range(len(paires))])
        res = np.linalg.norm(P @ R.T + t - Q, axis=1)
        if len(paires) < 6:
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
    return Recalage(R, t, d, ecart_r, res, gardees)


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


class Camera:
    """La vraie chaine : serveur de camera + Perception (sans etalonnage)."""

    def __init__(self):
        import perception
        self.perception = perception.Perception()
        self.K = None
        self.taille = None

    def voir(self) -> List[Vue]:
        import camera
        cap = camera.capturer()
        self.K = np.asarray(cap["K"], float)
        self.taille = cap["profondeur"].shape[::-1]
        pieces = self.perception.detecter(cap)
        return [Vue(p.T_camera[:3, 3] * 1000.0, p.T_camera[:3, 0].copy(), p.type,
                    p.face, float(getattr(p, "confiance", 1.0))) for p in pieces]


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


def _demander(texte: str) -> str:
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
                 r_max: float = 60.0):
        self.robot = robot
        self.camera = camera
        self.z_table = z_table
        self.garage = None if garage is None else [float(v) for v in garage]
        self.rng = np.random.default_rng(graine)
        self.journal = journal
        self.demander = demander
        self.zone_x, self.zone_y, self.r_max = zone_x, zone_y, float(r_max)
        self.paires: List[Paire] = []
        self.recalage: Optional[Recalage] = None
        self.depart: Optional[Tuple[float, float, float]] = None   # x, y, r
        self.validations: List[dict] = []
        #: decalage de la piece dans la pince, une fois qu'une saisie visee
        #: au centre l'a remis a zero (None : celui du recalage)
        self.decalage_courant: Optional[np.ndarray] = None
        from dobot_sim import DobotMagician
        self.modele = getattr(robot, "modele", None) or DobotMagician()

    # -- mouvements ---------------------------------------------------------
    def _z(self, h: float) -> float:
        return self.z_table + h

    def _aller(self, x, y, r):
        """Au-dessus de (x, y), pince orientee r, a la hauteur de transit."""
        xa, ya, za, ra = self.robot.pose()
        if za < self._z(TRANSIT) - 1.0:
            self.robot.movl(x=xa, y=ya, z=self._z(TRANSIT), r=ra)
        self.robot.movj(x=x, y=y, z=self._z(TRANSIT), r=r)

    def poser(self, x, y, r):
        self._aller(x, y, r)
        self.robot.movl(x=x, y=y, z=self._z(PRISE), r=r)
        self.robot.pince(False)
        self.robot.attendre(DELAI_PINCE)
        self.robot.movl(x=x, y=y, z=self._z(TRANSIT), r=r)

    def saisir(self, x, y, r):
        self._aller(x, y, r)
        self.robot.movl(x=x, y=y, z=self._z(PRISE), r=r)
        self.robot.pince(True)
        self.robot.attendre(DELAI_PINCE)
        self.robot.movl(x=x, y=y, z=self._z(TRANSIT), r=r)

    def degager(self):
        """Ecarte le bras du champ de la camera."""
        if self.garage is not None:
            x, y, z = self.garage
            xa, ya, za, ra = self.robot.pose()
            self.robot.movj(x=x, y=y, z=z, r=ra)
        else:
            self.robot.repli()

    # -- choix des points ---------------------------------------------------
    def _atteignable(self, x, y, r) -> bool:
        return all(self.modele.atteignable(x, y, self._z(h), r) for h in (PRISE, TRANSIT))

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
            s = float(np.min(np.linalg.norm(deja - [x, y], axis=1)))
            if s > score:
                meilleur, score = (x, y, r), s
        if meilleur is None:
            raise Arret("aucun point a la fois vu par la camera et atteint par "
                        "le bras")
        return meilleur

    # -- la camera ----------------------------------------------------------
    def observer(self, attendu_robot=None) -> Vue:
        """La piece vue. Exactement une piece, sinon on demande."""
        while True:
            vues = self.camera.voir()
            if attendu_robot is not None and self.recalage is not None and vues:
                prevu = self.recalage.vers_camera(attendu_robot)
                vues.sort(key=lambda v: np.linalg.norm(v.position - prevu))
                if len(vues) > 1 and np.linalg.norm(vues[0].position - prevu) < ECART_MAX \
                        and np.linalg.norm(vues[1].position - prevu) > 3 * ECART_MAX:
                    vues = vues[:1]
            if len(vues) == 1:
                return vues[0]
            rep = self.demander(
                "  la camera voit {} piece(s) au lieu d'une : degagez la table, "
                "puis Entree (q : arreter) ".format(len(vues)))
            if rep.strip().lower().startswith("q"):
                raise Arret("arret demande")

    # -- le deroule ---------------------------------------------------------
    def demarrer(self):
        """Apres l'accostage : la pince est ouverte a PRISE au-dessus de la
        table ; on y place la piece, puis on ferme."""
        x, y, z, r = self.robot.pose()
        if self.z_table is None:
            self.z_table = z - PRISE
        self.depart = (x, y, r)
        self.demander("  placez une EPAISSE A PLAT entre les mors, posee sur la "
                      "table, puis Entree ")
        self.robot.pince(True)
        self.robot.attendre(DELAI_PINCE)
        self.robot.movl(x=x, y=y, z=self._z(TRANSIT), r=r)

    def cycle(self) -> Paire:
        x, y, r = self.point_suivant()
        self.poser(x, y, r)
        self.degager()
        attendu = np.array([x, y, self._z(HAUTEUR_PIECE)])
        vue = self.observer(attendu)
        paire = Paire(np.array([x, y, self._z(PRISE)]), r, vue.position, vue.axe,
                      self.z_table)
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
        self.saisir(x, y, r)
        return paire

    def valider(self, n: int = 3) -> List[dict]:
        """
        Saisies guidees par la CAMERA : poser, regarder, et reprendre la ou
        le recalage dit que la piece est. L'ecart avec la position connue
        mesure ce que vaudra une vraie saisie.
        """
        self.journal("\nVALIDATION : saisies guidees par la camera seule")
        for k in range(n):
            x, y, r = self.point_suivant()
            self.poser(x, y, r)
            self.degager()
            vue = self.observer(np.array([x, y, self._z(HAUTEUR_PIECE)]))
            (px, py, _pz), pr = self.recalage.prise(vue.position, vue.axe)
            # r a 180 deg pres : on garde le plus proche de la pose
            if abs(_angle(pr - r)) > 90:
                pr = _angle(pr + 180.0)
            # le centre de la piece, connu : la pince + son decalage
            d = self.recalage.decalage if self.decalage_courant is None \
                else self.decalage_courant
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
                self.saisir(x, y, r)
            else:
                # visee au centre : la piece est desormais centree dans la
                # pince, le decalage ne vaut plus pour les poses suivantes
                self.saisir(px, py, pr)
                self.decalage_courant = np.zeros(2)
        return self.validations

    def resultat(self) -> dict:
        rc = self.recalage
        return {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "T_robot_camera": rc.T_m.tolist(),
            "unite": "metres (repere de la camera de PROFONDEUR -> repere robot "
                     "du simulateur dobot_sim)",
            "T_robot_camera_mm": rc.T_mm.tolist(),
            "decalage_pince_mm": [float(v) for v in rc.decalage],
            "ecart_r_deg": float(rc.ecart_r),
            "z_table_mm": float(self.z_table),
            "rms_mm": rc.rms,
            "paires": len(self.paires),
            "paires_retenues": int(rc.gardees.sum()),
            "residus_mm": [float(v) for v in rc.residus],
            "validation": self.validations,
            "mesures": [{"pince": [float(v) for v in p.pince], "r": float(p.r),
                         "camera": [float(v) for v in p.camera],
                         "axe_camera": [float(v) for v in p.lacet_camera]}
                        for p in self.paires],
        }

    def enregistrer(self, chemin: str = FICHIER) -> Optional[str]:
        if self.recalage is None:
            return None
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self.resultat(), f, indent=2)
        return chemin


def charger_etalonnage(chemin: str = FICHIER) -> np.ndarray:
    """T_robot_camera (4x4, METRES) : a donner a Perception(T_robot_camera=...)."""
    with open(chemin, encoding="utf-8") as f:
        return np.array(json.load(f)["T_robot_camera"], float)


# ---------------------------------------------------------------------------
# Accostage interactif
# ---------------------------------------------------------------------------

AIDE_CLAVIER = """
  ACCOSTAGE : amenez la pince OUVERTE au contact de la table, n'importe ou.
  A la main : bouton de deverrouillage de l'avant-bras, puis relachez-le.
  Au clavier :
     fleches haut / bas   monter / descendre (z)
     x X  y Y             deplacer en x / y (minuscule : moins)
     r R                  tourner la pince
     1  2  3              pas de 10 / 1 / 0,2 mm (5 / 1 / 0,2 deg)
     o                    ouvrir la pince
     Entree               CONTACT : la pince touche la table (glissez une
                          feuille sous les mors : elle doit etre pincee)
     q                    quitter
"""


def _lire_touche() -> str:
    """Une touche, sans Entree (terminal), ou une ligne sinon."""
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip() or "\n"
    import termios
    import tty
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


def accoster(robot, lire: Callable[[], str] = _lire_touche,
             journal: Callable[[str], None] = print) -> float:
    """Approche interactive ; rend le z de la pointe au contact de la table."""
    journal(AIDE_CLAVIER)
    robot.pince(False)
    pas = {"1": (10.0, 5.0), "2": (1.0, 1.0), "3": (0.2, 0.2)}
    p_mm, p_deg = pas["2"]
    while True:
        x, y, z, r = robot.pose()
        sys.stdout.write("\r  x {:6.1f}  y {:6.1f}  z {:6.1f}  r {:6.1f}   pas {:.1f} mm   "
                         .format(x, y, z, r, p_mm))
        sys.stdout.flush()
        c = lire()
        d = {"haut": (0, 0, 1, 0), "bas": (0, 0, -1, 0), "x": (-1, 0, 0, 0),
             "X": (1, 0, 0, 0), "y": (0, -1, 0, 0), "Y": (0, 1, 0, 0),
             "r": (0, 0, 0, -1), "R": (0, 0, 0, 1)}.get(c)
        if d is not None:
            try:
                robot.movl(x=x + d[0] * p_mm, y=y + d[1] * p_mm, z=z + d[2] * p_mm,
                           r=r + d[3] * p_deg)
            except Exception as exc:                     # noqa: BLE001
                journal("\n  refuse : {}".format(str(exc).splitlines()[0]))
        elif c in pas:
            p_mm, p_deg = pas[c]
        elif c == "o":
            robot.pince(False)
        elif c == "\n":
            journal("")
            return float(robot.pose()[2])
        elif c in ("q", "Q"):
            raise Arret("arret demande")


# ---------------------------------------------------------------------------
# Programme
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Etalonnage camera -> robot par poses "
                                             "successives d'une piece")
    ap.add_argument("--points", type=int, default=15, help="poses d'etalonnage (15)")
    ap.add_argument("--validations", type=int, default=3,
                    help="saisies guidees par la camera, a la fin (3)")
    ap.add_argument("--vitesse", type=float, default=40.0, help="mm/s (40)")
    ap.add_argument("--longueur-outil", type=float, default=None,
                    help="longueur mesuree de la pince, mm")
    ap.add_argument("--garage", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="ou ecarter le bras pendant la capture (defaut : repli)")
    ap.add_argument("--fenetre", action="store_true",
                    help="passer par le simulateur ouvert et connecte au robot")
    ap.add_argument("--simulation", action="store_true",
                    help="simulateur seul, piece et camera simulees : pour essayer")
    ap.add_argument("--sortie", default=FICHIER)
    a = ap.parse_args(argv)

    from dobot_sim import Commande
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
    try:
        if a.simulation:
            # pas de main pour guider le bras : on part d'un point connu
            commande.movj(x=230, y=40, z=60, r=0)
            print("simulation : la table est a z = 0 ; descendez avec la fleche bas")
        z_contact = accoster(robot)
        et.z_table = z_contact
        x, y, _z, r = robot.pose()
        robot.movl(x=x, y=y, z=z_contact + PRISE, r=r)
        print("table a z = {:.1f} mm (repere robot)".format(z_contact))
        et.demarrer()
        print("\nETALONNAGE : {} poses".format(a.points))
        for _ in range(a.points):
            et.cycle()
        rc = et.recalage
        print("\nrecalage : {:.2f} mm rms sur {} paires ({} retenues) ; decalage dans la "
              "pince ({:.1f}, {:.1f}) mm ; ecart d'angle {:.1f} deg".format(
                  rc.rms, len(et.paires), int(rc.gardees.sum()), rc.decalage[0],
                  rc.decalage[1], rc.ecart_r))
        if a.validations:
            et.valider(a.validations)
        x, y, r = et.point_suivant()
        et.poser(x, y, r)
        et.degager()
    except (Arret, KeyboardInterrupt) as exc:
        print("\narret : {}".format(exc or "Ctrl+C"))
    finally:
        chemin = et.enregistrer(a.sortie)
        if chemin:
            print("etalonnage enregistre : {}".format(os.path.relpath(chemin)))
            print("  Perception(T_robot_camera=charger_etalonnage())")
        else:
            print("pas assez de mesures : rien d'enregistre")
        commande.fermer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
