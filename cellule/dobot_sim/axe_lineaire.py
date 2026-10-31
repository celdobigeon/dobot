# -*- coding: utf-8 -*-
"""
L'axe lineaire : un septieme axe qui TRANSLATE LA BASE du robot.

C'est ce qui le distingue du convoyeur. Le tapis deplace les pieces, le
rail deplace le robot : toute l'enveloppe de travail glisse avec lui.
Une cible hors d'atteinte depuis une position du chariot peut etre
atteignable depuis une autre, et c'est tout l'interet.

    from dobot_sim import AxeLineaire, DobotMagician, cible

    RAIL = AxeLineaire(course=1000.0)
    robot = DobotMagician()

    T = cible(220, 400, 30)               # hors d'atteinte bras seul
    l = RAIL.position_pour(robot, T)      # ou amener le chariot
    q = robot.mgi(RAIL.ramener(T, l))     # le bras y va depuis la

Le modele
---------
Le chariot porte la base. Si `l` est sa position le long du rail et `R`
la matrice de translation correspondante, alors :

    situation dans l'atelier  =  R(l) . MGD(q)
    MGI depuis le chariot     =  MGI( inverse(R(l)) . cible )

Le bras lui-meme ne change pas : on resout le MGI sur la cible RAMENEE
dans le repere du chariot. Aucun modele geometrique n'est reecrit, ce
qui evite de toucher a ce qui marche.

La direction du rail
--------------------
Par defaut +y, c'est-a-dire que le chariot se deplace lateralement
devant l'operateur. C'est le montage courant du rail Dobot, mais il
n'est PAS verifie sur votre machine : le protocole V1.1.5 documente la
cinquieme coordonnee sans jamais dire dans quelle direction elle pousse.

Comment le mesurer, sans rien casser : mettez le bras a une pose connue,
lisez `GetPose`, deplacez le chariot de 100 mm, relisez. La coordonnee
qui a bouge de 100 donne la direction, et son signe le sens. Si `GetPose`
ne bouge pas du tout, c'est que le controleur rend la pose DANS le
repere du chariot -- passez alors `pose_absolue=False`, et le simulateur
ajoutera lui-meme le deplacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from . import homogeneous as hg

__all__ = ["AxeLineaire", "COURSE_DOBOT"]

#: Course du rail Dobot, en millimetres. Valeur du catalogue.
COURSE_DOBOT = 1000.0


@dataclass
class AxeLineaire:
    """
    Le rail, sa course, sa direction et sa cadence.

    `direction` est un vecteur unitaire du repere de la BASE : le
    chariot avance selon lui quand `l` croit.
    """

    course: float = COURSE_DOBOT
    direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: position du chariot a l'origine du rail, dans le repere de l'atelier
    origine: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vitesse: float = 100.0            # mm/s
    acceleration: float = 200.0       # mm/s2
    #: `GetPose` du controleur inclut-il deja le deplacement du chariot ?
    #: A determiner sur la machine (voir l'en-tete du module).
    pose_absolue: bool = True

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise ValueError("direction du rail nulle")
        self.direction = tuple(d / n)

    # -- geometrie ----------------------------------------------------------

    def borner(self, l: float) -> float:
        """Ramene une consigne dans la course du rail."""
        return float(np.clip(float(l), 0.0, self.course))

    def dans_la_course(self, l: float) -> bool:
        return -1e-9 <= float(l) <= self.course + 1e-9

    def reperage(self) -> np.ndarray:
        """
        Un repere pose sur le rail : x le long de la course, z vers le haut.

        Sert au dessin. La direction du rail ne suffit pas a orienter une
        boite : il faut deux autres axes, et les choisir ici evite que
        chaque vue les reinvente -- mal, le jour ou le rail sera vertical.
        """
        d = np.asarray(self.direction, dtype=float)
        d = d / (np.linalg.norm(d) or 1.0)
        appui = np.array([0.0, 0.0, 1.0])
        if abs(float(d @ appui)) > 0.99:        # rail vertical : autre appui
            appui = np.array([1.0, 0.0, 0.0])
        y = np.cross(appui, d)
        y /= np.linalg.norm(y) or 1.0
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = d, y, np.cross(d, y)
        T[:3, 3] = np.asarray(self.origine, dtype=float)
        return T

    def boites(self, l: float = 0.0) -> list:
        """
        De quoi DESSINER le rail : la poutre et le chariot en `l`.

        Meme convention que `Convoyeur.boites` -- centre, axes, dims,
        role -- et meme raison d'etre ici : cela se verifie sans fenetre.
        """
        T = self.reperage()
        axes = T[:3, :3]

        def boite(centre, dims, role):
            c = (T @ np.append(np.asarray(centre, dtype=float), 1.0))[:3]
            return {"centre": c, "axes": axes, "dims": tuple(dims),
                    "role": role}

        course = float(self.course)
        return [boite((course / 2, 0.0, -30.0), (course, 90.0, 24.0), "poutre"),
                boite((self.borner(l), 0.0, -12.0), (120.0, 110.0, 14.0),
                      "chariot")]

    def matrice(self, l: float) -> np.ndarray:
        """
        La translation de la base quand le chariot est en `l`.

        C'est la seule chose que le rail fait : il deplace le repere dans
        lequel le bras travaille.
        """
        d = np.asarray(self.direction, dtype=float)
        o = np.asarray(self.origine, dtype=float)
        return hg.translation(*(o + d * float(l)))

    def ramener(self, T, l: float) -> np.ndarray:
        """
        Une cible de l'atelier, vue depuis le chariot en `l`.

        C'est elle qu'on donne au MGI : le bras ne connait que son
        propre repere.
        """
        return hg.inverse(self.matrice(l)) @ np.asarray(T, dtype=float)

    def porter(self, T, l: float) -> np.ndarray:
        """L'inverse : une cible du chariot, vue dans l'atelier."""
        return self.matrice(l) @ np.asarray(T, dtype=float)

    # -- choisir une position de chariot ------------------------------------

    def position_pour(self, robot, T, depart: Optional[float] = None,
                      pas: float = 5.0, marge: float = 0.0) -> Optional[float]:
        """
        Ou amener le chariot pour que le bras atteigne cette cible.

        Balaie la course et rend la position ADMISSIBLE la plus proche de
        `depart` -- on ne traverse pas tout l'atelier quand un petit
        deplacement suffit. None si la cible reste hors d'atteinte sur
        toute la course.

        `marge` ecarte les solutions trop proches des butees : le bras y
        serait tendu, et la moindre correction le ferait sortir.
        """
        T = np.asarray(T, dtype=float)
        depart = self.borner(0.0 if depart is None else depart)
        candidats = np.arange(0.0, self.course + pas, pas)
        candidats = sorted(candidats, key=lambda l: abs(l - depart))
        for l in candidats:
            l = self.borner(l)
            try:
                q = robot.mgi(self.ramener(T, l))
            except Exception:
                continue
            if robot.depassements(q):
                continue
            if marge > 0:
                lim = robot.lim
                if any(q[i] < lim.q_min[i] + marge or q[i] > lim.q_max[i] - marge
                       for i in range(len(q))):
                    continue
            return float(l)
        return None

    def atteignable(self, robot, T, l: float) -> bool:
        """La cible est-elle atteignable depuis cette position de chariot ?"""
        try:
            q = robot.mgi(self.ramener(T, l))
        except Exception:
            return False
        return not robot.depassements(q)

    # -- cadence ------------------------------------------------------------

    def duree(self, depart: float, arrivee: float) -> float:
        """
        Temps que met le chariot, profil trapeze borne par la cadence.

        Sert a dater le mouvement dans la simulation : le rail et le bras
        partent ensemble, et c'est le plus lent qui donne la duree.
        """
        d = abs(float(arrivee) - float(depart))
        if d < 1e-9:
            return 0.0
        v, a = max(self.vitesse, 1e-9), max(self.acceleration, 1e-9)
        d_acc = v * v / a                      # montee + descente
        if d <= d_acc:
            return 2.0 * float(np.sqrt(d / a))
        return v / a + d / v

    def resume(self) -> str:
        d = self.direction
        return ("axe lineaire : course {:.0f} mm, direction ({:+.2f}, {:+.2f}, "
                "{:+.2f}), {:.0f} mm/s\n"
                "  pose du controleur : {}"
                .format(self.course, d[0], d[1], d[2], self.vitesse,
                        "absolue (le rail est compte dedans)"
                        if self.pose_absolue
                        else "relative au chariot"))
