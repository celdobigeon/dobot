"""
Generation de trajectoires : lois horaires, mouvements articulaires et
mouvements cartesiens (rectiligne, circulaire) bases sur les matrices
homogenes.

Deux notions distinctes :

  * la LOI HORAIRE (classe Profil) donne l'abscisse curviligne normalisee
    s(t) dans [0, 1] et sa derivee : c'est elle qui fixe l'allure des
    vitesses (echelon, trapeze, polynome, sinus) ;

  * le SEGMENT (classe Segment) donne la geometrie du chemin : dans
    l'espace articulaire on interpole les angles, dans l'espace cartesien
    on interpole la matrice homogene puis on applique le MGI.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from . import homogeneous as hg
from .robot import DobotMagician, ErreurCinematique

#: sert a dedupliquer les avertissements : deux messages qui ne different
#: que par leurs valeurs numeriques signalent la meme chose
_SANS_NOMBRES = re.compile(r"[-+]?\d[\d.]*")

__all__ = [
    "Profil", "ProfilLineaire", "ProfilTrapeze", "ProfilPolynomial5", "ProfilSinus",
    "ProfilDemarrage", "ProfilArret", "ProfilTroncons",
    "creer_profil", "creer_profil_vitesse", "PROFILS",
    "Segment", "SegmentArticulaire", "SegmentLineaire", "SegmentCirculaire",
    "SegmentAttente", "SegmentAction", "Trajectoire",
]


# ===========================================================================
# Lois horaires
# ===========================================================================

class Profil:
    """
    Loi horaire normalisee : s(0) = 0, s(duree) = 1.

    K_V et K_A sont les valeurs maximales de s'(u) et s''(u) pour une duree
    unitaire et une course unitaire. Elles permettent de calculer la duree
    minimale d'un mouvement de longueur L respectant une vitesse v_max et
    une acceleration a_max :

        T >= K_V . L / v_max        et        T >= sqrt(K_A . L / a_max)

    Ces coefficients rendent la comparaison des lois horaires concrete : a
    vitesse maximale egale, un polynome de degre 5 met 1,875 fois plus de
    temps qu'un profil a vitesse constante.
    """

    nom = "profil"
    K_V = 1.0
    K_A = None        # None : acceleration non bornee (discontinuite de vitesse)

    def __init__(self, duree: float):
        self.duree = max(float(duree), 1e-6)

    def s(self, t: float) -> float:
        raise NotImplementedError

    @classmethod
    def duree_pour(cls, longueur: float, v_max: float, a_max: float) -> float:
        """Duree minimale d'un mouvement de longueur `longueur` sous contraintes."""
        L = abs(float(longueur))
        if L <= 1e-12:
            return 0.0
        T = cls.K_V * L / max(float(v_max), 1e-9)
        if cls.K_A:
            T = max(T, np.sqrt(cls.K_A * L / max(float(a_max), 1e-9)))
        return float(T)

    @classmethod
    def adapte(cls, longueur: float, v_max: float, a_max: float,
               duree_min: float = 0.05) -> "Profil":
        """Instancie le profil avec la duree deduite de v_max et a_max."""
        return cls(max(cls.duree_pour(longueur, v_max, a_max), duree_min))

    def ds(self, t: float) -> float:
        """Derivee de s par rapport au temps (1/s)."""
        h = self.duree * 1e-4
        t0 = min(max(t, h), self.duree - h)
        return (self.s(t0 + h) - self.s(t0 - h)) / (2.0 * h)

    def _u(self, t: float) -> float:
        return float(np.clip(t / self.duree, 0.0, 1.0))

    def __repr__(self):
        return "{}(duree={:.3f}s)".format(type(self).__name__, self.duree)


class ProfilLineaire(Profil):
    """Vitesse constante. Discontinuite de vitesse aux extremites."""

    nom = "lineaire"
    K_V = 1.0
    K_A = None          # echelons de vitesse : acceleration theoriquement infinie

    def s(self, t: float) -> float:
        return self._u(t)

    def ds(self, t: float) -> float:
        return 0.0 if (t < 0 or t > self.duree) else 1.0 / self.duree


class ProfilTrapeze(Profil):
    """
    Profil de vitesse trapezoidal : acceleration constante, palier, freinage.

    `fraction` est la part de la duree consacree a l'acceleration
    (0 < fraction <= 0.5). fraction = 0.5 donne un profil triangulaire.
    """

    nom = "trapeze"

    def __init__(self, duree: float, fraction: float = 0.25):
        super().__init__(duree)
        self.fraction = float(np.clip(fraction, 1e-3, 0.5))

    def s(self, t: float) -> float:
        u = self._u(t)
        f = self.fraction
        # vitesse de palier normalisee telle que l'integrale vaille 1
        vp = 1.0 / (1.0 - f)
        if u < f:
            return 0.5 * vp * u * u / f
        if u <= 1.0 - f:
            return 0.5 * vp * f + vp * (u - f)
        w = 1.0 - u
        return 1.0 - 0.5 * vp * w * w / f

    def ds(self, t: float) -> float:
        u = self._u(t)
        f = self.fraction
        vp = 1.0 / (1.0 - f)
        if u < f:
            v = vp * u / f
        elif u <= 1.0 - f:
            v = vp
        else:
            v = vp * (1.0 - u) / f
        return v / self.duree

    @classmethod
    def duree_pour(cls, longueur: float, v_max: float, a_max: float) -> float:
        """
        Duree exacte du profil trapezoidal saturant a la fois v_max et a_max.

        Palier atteint si L >= v^2 / a, sinon le profil devient triangulaire.
        """
        L = abs(float(longueur))
        v = max(float(v_max), 1e-9)
        a = max(float(a_max), 1e-9)
        if L <= 1e-12:
            return 0.0
        if L >= v * v / a:
            return L / v + v / a          # trapeze : montee + palier + descente
        return 2.0 * float(np.sqrt(L / a))  # triangle : v_max jamais atteinte

    @classmethod
    def adapte(cls, longueur: float, v_max: float, a_max: float,
               duree_min: float = 0.05) -> "ProfilTrapeze":
        """
        Instancie le trapeze dont la duree ET la part d'acceleration
        decoulent de v_max et a_max.
        """
        T = max(cls.duree_pour(longueur, v_max, a_max), duree_min)
        t_acc = min(max(float(v_max), 1e-9) / max(float(a_max), 1e-9), T / 2.0)
        return cls(T, t_acc / T)


class ProfilPolynomial5(Profil):
    """
    Polynome de degre 5 : vitesse ET acceleration nulles aux extremites.

        s(u) = 10u^3 - 15u^4 + 6u^5

    Maximums : s'(1/2) = 15/8 et |s''| = 10/racine(3).
    """

    nom = "polynome5"
    K_V = 1.875                  # 15/8
    K_A = 5.773502691896258      # 10/sqrt(3)

    def s(self, t: float) -> float:
        u = self._u(t)
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))

    def ds(self, t: float) -> float:
        u = self._u(t)
        return (30.0 * u * u * (1.0 - u) ** 2) / self.duree


class ProfilSinus(Profil):
    """Loi en cosinus surleve : s(u) = (1 - cos(pi u)) / 2. Acceleration continue."""

    nom = "sinus"
    K_V = 1.5707963267948966     # pi/2
    K_A = 4.934802200544679      # pi^2/2

    def s(self, t: float) -> float:
        u = self._u(t)
        return 0.5 * (1.0 - np.cos(np.pi * u))

    def ds(self, t: float) -> float:
        u = self._u(t)
        return 0.5 * np.pi * np.sin(np.pi * u) / self.duree


class ProfilDemarrage(Profil):
    """
    Demarrage puis vitesse constante : on part de l'arret, on n'y revient pas.

    C'est le premier segment d'un mouvement CONTINU (trajectoire CP du
    controleur) : le robot accelere, puis enchaine sur le segment suivant
    sans ralentir. Enchaine avec `lineaire` puis `arret`, il donne un
    trapeze etale sur plusieurs segments -- ce que fait la machine.

    `fraction` est la part de la duree consacree a l'acceleration
    (0 < fraction <= 1). La vitesse de palier normalisee vaut
    1 / (1 - fraction / 2).
    """

    nom = "demarrage"
    K_V = 1.0 / (1.0 - 0.125)        # palier, pour fraction = 0.25
    K_A = K_V / 0.25

    def __init__(self, duree: float, fraction: float = 0.25):
        super().__init__(duree)
        self.fraction = float(np.clip(fraction, 1e-3, 1.0))

    def _vp(self) -> float:
        return 1.0 / (1.0 - 0.5 * self.fraction)

    def _s(self, u: float) -> float:
        f, vp = self.fraction, self._vp()
        if u < f:
            return 0.5 * vp * u * u / f
        return 0.5 * vp * f + vp * (u - f)

    def _v(self, u: float) -> float:
        f, vp = self.fraction, self._vp()
        return vp * u / f if u < f else vp

    def s(self, t: float) -> float:
        return self._s(self._u(t))

    def ds(self, t: float) -> float:
        return self._v(self._u(t)) / self.duree

    @classmethod
    def duree_pour(cls, longueur: float, v_max: float, a_max: float) -> float:
        """
        Une seule rampe : L / v + v / (2a) si le palier est atteint
        (L >= v^2 / 2a), sinon la rampe seule, racine(2 L / a).
        """
        L = abs(float(longueur))
        v = max(float(v_max), 1e-9)
        a = max(float(a_max), 1e-9)
        if L <= 1e-12:
            return 0.0
        if L >= v * v / (2.0 * a):
            return L / v + v / (2.0 * a)
        return float(np.sqrt(2.0 * L / a))

    @classmethod
    def adapte(cls, longueur: float, v_max: float, a_max: float,
               duree_min: float = 0.05) -> "ProfilDemarrage":
        T = max(cls.duree_pour(longueur, v_max, a_max), duree_min)
        t_acc = min(max(float(v_max), 1e-9) / max(float(a_max), 1e-9), T)
        return cls(T, t_acc / T)


class ProfilArret(ProfilDemarrage):
    """
    Vitesse constante puis arret : le dernier segment d'un mouvement
    continu. C'est le `demarrage` joue a l'envers dans le temps.
    """

    nom = "arret"

    def s(self, t: float) -> float:
        return 1.0 - self._s(1.0 - self._u(t))

    def ds(self, t: float) -> float:
        return self._v(1.0 - self._u(t)) / self.duree


class ProfilTroncons(Profil):
    """
    Loi horaire d'un morceau de trajectoire CONTINUE (mode CP du robot).

    Le chemin est decoupe en troncons (d, w0, w1, c) : longueur d en mm,
    vitesse d'entree w0, de sortie w1, de croisiere visee c. Sur chacun, le
    robot accelere a `a` constante vers c, garde c, puis freine vers w1 --
    sans atteindre c si le troncon est trop court. Les vitesses aux
    jonctions viennent du modele du controleur (`commande.prevoir_cp`) :
    le simulateur ralentit donc aux coins comme le robot.
    """

    nom = "troncons"

    def __init__(self, troncons, acceleration: float):
        a = max(float(acceleration), 1e-9)
        self._phases = []                  # (t0, s0, duree, v0, acc)
        t = s = 0.0
        for d, w0, w1, c in troncons:
            d = max(float(d), 0.0)
            w0, w1 = max(float(w0), 0.0), max(float(w1), 0.0)
            c = max(float(c), w0, w1, 1e-6)
            if d <= 1e-9:
                continue
            da = (c * c - w0 * w0) / (2.0 * a)
            dd = (c * c - w1 * w1) / (2.0 * a)
            if da + dd > d:
                c = max(np.sqrt(max((2.0 * a * d + w0 * w0 + w1 * w1) / 2.0,
                                    0.0)), w0, w1)
                da = max((c * c - w0 * w0) / (2.0 * a), 0.0)
                dd = max(d - da, 0.0)
            for v0, v1, longueur, acc in ((w0, c, da, a),
                                          (c, c, d - da - dd, 0.0),
                                          (c, w1, dd, -a)):
                if longueur <= 1e-12:
                    continue
                duree = (2.0 * longueur / (v0 + v1) if v0 + v1 > 1e-12
                         else 0.0)
                self._phases.append((t, s, duree, v0, acc))
                t += duree
                s += longueur
        self.longueur = s
        super().__init__(t)

    def _abscisse(self, t: float) -> Tuple[float, float]:
        """(distance parcourue, vitesse) a l'instant t, en mm et mm/s."""
        if not self._phases:
            return 0.0, 0.0
        t = float(np.clip(t, 0.0, self.duree))
        for t0, s0, duree, v0, acc in reversed(self._phases):
            if t >= t0:
                u = min(t - t0, duree)
                return s0 + v0 * u + 0.5 * acc * u * u, v0 + acc * u
        return 0.0, 0.0

    def s(self, t: float) -> float:
        if self.longueur <= 1e-12:
            return self._u(t)
        return float(np.clip(self._abscisse(t)[0] / self.longueur, 0.0, 1.0))

    def ds(self, t: float) -> float:
        if self.longueur <= 1e-12:
            return 1.0 / self.duree
        return self._abscisse(t)[1] / self.longueur


PROFILS = {
    "lineaire": ProfilLineaire,
    "trapeze": ProfilTrapeze,
    "polynome5": ProfilPolynomial5,
    "sinus": ProfilSinus,
    "demarrage": ProfilDemarrage,
    "arret": ProfilArret,
}


def creer_profil(nom: str, duree: float) -> Profil:
    """Fabrique un profil de duree imposee a partir de son nom (voir PROFILS)."""
    cls = PROFILS.get(str(nom).lower(), ProfilPolynomial5)
    return cls(duree)


def creer_profil_vitesse(nom: str, longueur: float, v_max: float, a_max: float,
                         duree_min: float = 0.05) -> Profil:
    """
    Fabrique un profil dont la DUREE est deduite de la longueur du chemin,
    de la vitesse maximale et de l'acceleration maximale.

    `longueur` s'exprime dans l'unite de l'espace ou l'on interpole :
    millimetres pour un mouvement cartesien, degres pour un mouvement
    articulaire. v_max et a_max suivent la meme unite par seconde.
    """
    cls = PROFILS.get(str(nom).lower(), ProfilPolynomial5)
    return cls.adapte(longueur, v_max, a_max, duree_min)


# ===========================================================================
# Segments de trajectoire
# ===========================================================================

class Segment:
    """Portion elementaire de trajectoire."""

    type_mouvement = "segment"

    def __init__(self, robot: DobotMagician, profil: Profil, nom: str = ""):
        self.robot = robot
        self.profil = profil
        self.nom = nom
        self.avertissements: List[str] = []

    #: unite de `longueur` : "mm" en cartesien, "deg" en articulaire
    unite = "mm"

    @property
    def duree(self) -> float:
        return self.profil.duree

    @property
    def longueur(self) -> float:
        """
        Longueur du chemin, dans l'unite de l'espace ou l'on interpole.

        Sert a convertir une consigne de vitesse en duree.
        """
        return 0.0

    def q(self, t: float) -> np.ndarray:
        """Configuration articulaire (degres) a l'instant t."""
        raise NotImplementedError

    def pose(self, t: float) -> np.ndarray:
        """Matrice homogene de l'outil a l'instant t."""
        return self.robot.mgd(self.q(t))

    @property
    def q_final(self) -> np.ndarray:
        return self.q(self.duree)

    def echantillonner(self, dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Renvoie (temps (N,), configurations (N, 4))."""
        n = max(int(np.ceil(self.duree / dt)) + 1, 2)
        ts = np.linspace(0.0, self.duree, n)
        Q = np.array([self.q(t) for t in ts])
        return ts, Q


class SegmentArticulaire(Segment):
    """
    Mouvement articulaire (MOVJ) : interpolation directe des angles.

    Le chemin de l'outil dans l'espace n'est pas rectiligne, mais le
    mouvement est rapide et ne traverse jamais de singularite du MGI.
    """

    type_mouvement = "MOVJ"
    unite = "deg"

    def __init__(self, robot, q_depart, q_arrivee, profil: Profil, nom: str = ""):
        super().__init__(robot, profil, nom)
        self.q0 = np.asarray(q_depart, dtype=float).copy()
        self.q1 = np.asarray(q_arrivee, dtype=float).copy()
        for msg in robot.defauts(self.q1):
            self.avertissements.append("cible : " + msg)

    @property
    def longueur(self) -> float:
        """Plus grande excursion articulaire, en degres : c'est elle qui borne."""
        return float(np.max(np.abs(self.q1 - self.q0)))

    def q(self, t: float) -> np.ndarray:
        return self.q0 + (self.q1 - self.q0) * self.profil.s(t)


class SegmentLineaire(Segment):
    """
    Mouvement cartesien rectiligne (MOVL).

    La pose est interpolee dans SE(3) (position lineaire + SLERP sur
    l'orientation) puis convertie en angles par le MGI a chaque instant.
    """

    type_mouvement = "MOVL"

    def __init__(self, robot, q_depart, T_arrivee, profil: Profil, nom: str = ""):
        super().__init__(robot, profil, nom)
        self.q0 = np.asarray(q_depart, dtype=float).copy()
        self.T0 = robot.mgd(self.q0)
        self.T1 = np.asarray(T_arrivee, dtype=float).copy()
        self._verifier()

    @property
    def longueur(self) -> float:
        """Longueur de la droite parcourue par le TCP, en millimetres."""
        return float(np.linalg.norm(hg.position_de(self.T1)
                                    - hg.position_de(self.T0)))

    def _verifier(self, n: int = 25) -> None:
        for s in np.linspace(0.0, 1.0, n):
            T = hg.interpoler(self.T0, self.T1, s)
            p = hg.position_de(T)
            try:
                q = self.robot.mgi(T)
            except ErreurCinematique as exc:
                self.avertissements.append(
                    "s={:.2f} ({:.0f}, {:.0f}, {:.0f}) : {}".format(s, *p, exc))
                continue
            for msg in self.robot.defauts(q):
                self.avertissements.append("s={:.2f} : {}".format(s, msg))
        # deduplique en conservant l'ordre
        vus = set()
        uniques = []
        for a in self.avertissements:
            cle = a.split(":")[-1].strip()
            if cle not in vus:
                vus.add(cle)
                uniques.append(a)
        self.avertissements = uniques[:8]

    def pose(self, t: float) -> np.ndarray:
        return hg.interpoler(self.T0, self.T1, self.profil.s(t))

    def q(self, t: float) -> np.ndarray:
        return self.robot.mgi(self.pose(t), tolerant=True)


class SegmentCirculaire(Segment):
    """
    Mouvement cartesien circulaire (MOVC) : arc de cercle passant par trois
    points (depart, point de passage, arrivee).

    L'orientation est interpolee par SLERP entre le depart et l'arrivee.
    """

    type_mouvement = "MOVC"

    def __init__(self, robot, q_depart, T_passage, T_arrivee, profil: Profil, nom: str = ""):
        super().__init__(robot, profil, nom)
        self.q0 = np.asarray(q_depart, dtype=float).copy()
        self.T0 = robot.mgd(self.q0)
        self.Tv = np.asarray(T_passage, dtype=float).copy()
        self.T1 = np.asarray(T_arrivee, dtype=float).copy()
        self._degenere = False
        self._construire_arc()

    @property
    def longueur(self) -> float:
        """Longueur de l'arc parcouru par le TCP, en millimetres."""
        if self._degenere:
            return float(np.linalg.norm(hg.position_de(self.T1)
                                        - hg.position_de(self.T0)))
        return float(self.longueur_arc)

    def _construire_arc(self) -> None:
        A = hg.position_de(self.T0)
        B = hg.position_de(self.Tv)
        C = hg.position_de(self.T1)
        ab, ac, bc = B - A, C - A, C - B
        n = np.cross(ab, ac)
        nn = float(np.dot(n, n))
        if nn < 1e-9:
            self._degenere = True
            self.avertissements.append(
                "les trois points sont alignes ou confondus : "
                "le mouvement est execute en ligne droite")
            return
        alpha = np.dot(bc, bc) * np.dot(ab, ac) / (2.0 * nn)
        beta = np.dot(ac, ac) * np.dot(-ab, bc) / (2.0 * nn)
        gamma = np.dot(ab, ab) * np.dot(-ac, -bc) / (2.0 * nn)
        self.centre = alpha * A + beta * B + gamma * C
        self.rayon = float(np.linalg.norm(A - self.centre))
        self.normale = n / np.sqrt(nn)

        u = (A - self.centre) / self.rayon
        w = np.cross(self.normale, u)
        self._u, self._w = u, w

        def angle(p):
            d = p - self.centre
            return float(np.arctan2(np.dot(d, w), np.dot(d, u))) % (2.0 * np.pi)

        th_b, th_c = angle(B), angle(C)
        # on doit passer par B en allant de A (theta = 0) vers C
        self.balayage = th_c if th_b < th_c else th_c - 2.0 * np.pi
        self.longueur_arc = abs(self.balayage) * self.rayon

    def pose(self, t: float) -> np.ndarray:
        s = self.profil.s(t)
        if self._degenere:
            return hg.interpoler(self.T0, self.T1, s)
        th = self.balayage * s
        p = self.centre + self.rayon * (np.cos(th) * self._u + np.sin(th) * self._w)
        R = hg.rotation_de(hg.slerp(self.T0, self.T1, s))
        return hg.depuis_rp(R, p)

    def q(self, t: float) -> np.ndarray:
        return self.robot.mgi(self.pose(t), tolerant=True)


class SegmentAttente(Segment):
    """Temporisation : le robot reste immobile."""

    type_mouvement = "ATTENTE"

    def __init__(self, robot, q_courant, duree: float, nom: str = ""):
        super().__init__(robot, ProfilLineaire(duree), nom)
        self.q0 = np.asarray(q_courant, dtype=float).copy()

    def q(self, t: float) -> np.ndarray:
        return self.q0.copy()


class SegmentAction(SegmentAttente):
    """
    Commande logique instantanee : ventouse, pince, laser, sortie TOR.

    Le robot y est immobile comme dans une temporisation, mais la nature
    est differente et la Trajectoire les traite differemment. Une ATTENTE
    est un temps DEMANDE : le lissage ne doit pas l'absorber. Une ACTION
    ne demande aucun temps, elle part a l'instant ou la file l'atteint --
    donc, si le mouvement precedent est lisse, AVANT que la cible soit
    atteinte et robot en marche.

    C'est volontaire : garantir qu'une prise se fasse a l'arret releve de
    celui qui programme, pas du modele. On ne lisse pas un point ou l'on
    agit, et c'est la reflexion a mener.
    """

    type_mouvement = "ACTION"


# ===========================================================================
# Trajectoire complete
# ===========================================================================

def _adoucir(u: float) -> float:
    """Fonction de melange quintique : 0 et 1 aux bords, derivees nulles."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


class Trajectoire:
    """
    Suite de segments parcourue de maniere continue, avec lissage optionnel
    des raccords.

    Sans lissage, les segments sont mis bout a bout et le robot marque un
    arret a chaque point de passage. Avec lissage, deux segments consecutifs
    se RECOUVRENT pendant une duree tb : sur cette fenetre la consigne est le
    melange pondere des deux segments,

        q(t) = (1 - w) . q_i(t) + w . q_i+1(t),   w passant de 0 a 1,

    la ponderation w etant quintique, donc de derivee nulle aux deux bords :
    la vitesse reste continue et le robot coupe le coin au lieu de s'arreter.
    C'est le principe des zones d'ABB ou du CNT de Fanuc. Le point de passage
    n'est alors plus atteint exactement : c'est le prix du lissage, et la
    duree totale diminue d'autant.
    """

    #: fraction maximale d'un segment consommable par un recouvrement
    RECOUVREMENT_MAX = 0.49

    def __init__(self, robot: DobotMagician,
                 segments: Optional[List[Segment]] = None,
                 lissages: Optional[List[float]] = None):
        self.robot = robot
        self.segments: List[Segment] = list(segments or [])
        # fraction de lissage demandee APRES chaque segment, de 0 a 1 ;
        # 1 correspond au recouvrement maximal admissible
        self.lissages: List[float] = list(lissages or [])
        self._recalculer()

    def ajouter(self, segment: Segment, lissage: float = 0.0) -> "Trajectoire":
        self.segments.append(segment)
        self.lissages.append(float(lissage))
        self._recalculer()
        return self

    # -- chronologie --------------------------------------------------------

    def _recalculer(self) -> None:
        """Recalcule recouvrements effectifs, dates de debut et duree totale."""
        n = len(self.segments)
        while len(self.lissages) < n:
            self.lissages.append(0.0)
        del self.lissages[n:]

        self.recouvrements = [0.0] * n
        for i in range(n - 1):
            a, b = self.segments[i], self.segments[i + 1]
            # on ne lisse jamais autour d'une temporisation : une attente
            # demandee -- un delai, un capteur -- doit etre respectee.
            # Une ACTION, elle, se lisse comme le reste : elle ne demande
            # aucun temps, et le modele n'a pas a garantir en douce que la
            # cible est atteinte avant qu'elle parte. C'est au programme de
            # ne pas lisser un point ou il agit.
            if a.type_mouvement == "ATTENTE" or b.type_mouvement == "ATTENTE":
                continue
            # la fraction porte sur le recouvrement MAXIMAL admissible, si
            # bien que toute la plage 0..1 produit un effet croissant
            plafond = self.RECOUVREMENT_MAX * min(a.duree, b.duree)
            self.recouvrements[i] = float(np.clip(self.lissages[i], 0.0, 1.0)) * plafond

        self.debuts = [0.0] * n
        for i in range(1, n):
            self.debuts[i] = (self.debuts[i - 1] + self.segments[i - 1].duree
                              - self.recouvrements[i - 1])
        self._duree = (self.debuts[-1] + self.segments[-1].duree) if n else 0.0

    @property
    def duree(self) -> float:
        return float(self._duree)

    @property
    def duree_sans_lissage(self) -> float:
        return float(sum(s.duree for s in self.segments))

    @property
    def avertissements(self) -> List[str]:
        out = []
        for i, s in enumerate(self.segments):
            for a in s.avertissements:
                out.append("[{} {}] {}".format(i + 1, s.type_mouvement, a))
        return out

    def _actifs(self, t: float) -> List[int]:
        """Indices des segments en cours a l'instant t (un, ou deux si lissage)."""
        eps = 1e-9
        actifs = [i for i, s in enumerate(self.segments)
                  if self.debuts[i] - eps <= t <= self.debuts[i] + s.duree + eps]
        return actifs[-2:]

    def localiser(self, t: float) -> Tuple[int, float]:
        """Renvoie (indice du segment courant, temps local) pour un temps global t."""
        if not self.segments:
            return -1, 0.0
        t = float(np.clip(t, 0.0, self._duree))
        actifs = self._actifs(t)
        if not actifs:
            i = len(self.segments) - 1
            return i, self.segments[i].duree
        i = actifs[-1]
        return i, min(max(t - self.debuts[i], 0.0), self.segments[i].duree)

    def _poids_melange(self, i: int, j: int, t: float) -> float:
        """Poids du segment j (celui qui arrive) dans le recouvrement i/j."""
        tb = self.recouvrements[i]
        if tb <= 1e-12:
            return 1.0
        return _adoucir((t - self.debuts[j]) / tb)

    def q(self, t: float) -> np.ndarray:
        if not self.segments:
            return np.zeros(4)
        t = float(np.clip(t, 0.0, self._duree))
        actifs = self._actifs(t)
        if len(actifs) == 1:
            i = actifs[0]
            return self.segments[i].q(t - self.debuts[i])
        i, j = actifs
        w = self._poids_melange(i, j, t)
        qi = self.segments[i].q(t - self.debuts[i])
        qj = self.segments[j].q(t - self.debuts[j])
        return (1.0 - w) * qi + w * qj

    def pose(self, t: float) -> np.ndarray:
        if not self.segments:
            return hg.identite()
        t = float(np.clip(t, 0.0, self._duree))
        actifs = self._actifs(t)
        if len(actifs) == 1:
            i = actifs[0]
            return self.segments[i].pose(t - self.debuts[i])
        # dans un raccord lisse, la pose reelle est celle du MGD
        return self.robot.mgd(self.q(t))

    @property
    def lissage_actif(self) -> bool:
        return any(r > 1e-9 for r in self.recouvrements[:-1])

    def echantillonner(self, dt: float = 0.02) -> dict:
        """
        Echantillonne toute la trajectoire.

        Renvoie un dictionnaire :
            t   (N,)    temps
            Q   (N, 4)  angles articulaires (deg)
            dQ  (N, 4)  vitesses articulaires (deg/s)
            P   (N, 3)  position du TCP (mm)
            V   (N,)    norme de la vitesse cartesienne (mm/s)
            R   (N,)    orientation r de l'outil (deg)
        """
        duree = self.duree
        n = max(int(np.ceil(duree / dt)) + 1, 2)
        ts = np.linspace(0.0, duree, n)
        Q = np.array([self.q(t) for t in ts])
        P = np.array([hg.position_de(self.pose(t)) for t in ts])
        Rr = np.array([self.robot.pose(q)[3] for q in Q])
        dQ = np.gradient(Q, ts, axis=0) if n > 2 else np.zeros_like(Q)
        dP = np.gradient(P, ts, axis=0) if n > 2 else np.zeros_like(P)
        V = np.linalg.norm(dP, axis=1)
        return {"t": ts, "Q": Q, "dQ": dQ, "P": P, "V": V, "R": Rr}

    def controler(self, dt: float = 0.05) -> List[str]:
        """Verifie butees et vitesses sur toute la trajectoire."""
        msgs = list(self.avertissements)
        if not self.segments:
            return msgs
        ech = self.echantillonner(dt)

        # collisions : contrairement aux butees, elles ne se lisent pas sur
        # les seules cibles -- le bras peut traverser la table EN CHEMIN.
        # On signale au passage les butees FROLEES : le modele les accepte,
        # une machine reelle les refuse souvent.
        vus = set()
        for t, q in zip(ech["t"], ech["Q"]):
            for msg in (self.robot.collisions(q)
                        + self.robot.butees_frolees(q)):
                # on gomme les nombres pour la comparaison : sinon chaque
                # echantillon produit un message "different" et le journal
                # se remplit de la meme alerte a la virgule pres
                cle = _SANS_NOMBRES.sub("#", msg)
                if cle not in vus:
                    vus.add(cle)
                    msgs.append("t={:.2f}s : {}".format(t, msg))

        vmax = np.max(np.abs(ech["dQ"]), axis=0)
        for i, v in enumerate(vmax):
            if v > self.robot.lim.v_max[i] * 1.01:
                msgs.append("vitesse {} = {:.0f} deg/s > limite {:.0f} deg/s"
                            .format(self.robot.NOMS_ARTICULATIONS[i], v,
                                    self.robot.lim.v_max[i]))
        vc = float(np.max(ech["V"]))
        if vc > self.robot.lim.v_cart_max * 1.01:
            msgs.append("vitesse cartesienne {:.0f} mm/s > limite {:.0f} mm/s"
                        .format(vc, self.robot.lim.v_cart_max))
        return msgs

    def __len__(self):
        return len(self.segments)
