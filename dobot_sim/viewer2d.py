"""
Vue de secours en projections 2D.

Utilisee automatiquement lorsque OpenGL n'est pas disponible (machine
virtuelle, pilote graphique minimal, PyOpenGL absent). Elle expose la meme
interface que Vue3D : l'IHM n'a pas besoin de savoir laquelle est active.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from .robot import DobotMagician

__all__ = ["Vue2D"]

#: couleurs des peripheriques, par role -- les memes qu'en 3D
COULEURS_ROLES = {
    "bande":   (70, 70, 80),
    "bati":    (140, 145, 155),
    "capteur": (240, 90, 50),
    "fenetre": (240, 90, 50),
    "champ":   (75, 180, 240),
    "piece":   (245, 205, 65),
    "poutre":  (115, 120, 132),
    "chariot": (215, 215, 225),
}
#: roles traces en pointilles : ce qu'on voit sans le toucher
ROLES_POINTILLES = ("fenetre", "champ")


def sommets_boite(boite) -> np.ndarray:
    """Les huit coins d'une boite {centre, axes, dims}, en (8, 3)."""
    signes = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1)
                       for sx in (-1, 1)], dtype=float)
    c = np.asarray(boite["centre"], dtype=float)
    R = np.asarray(boite["axes"], dtype=float)
    demi = np.asarray(boite["dims"], dtype=float) / 2.0
    return c + (signes * demi) @ R.T


def contour_xy(points: np.ndarray) -> np.ndarray:
    """
    L'empreinte au sol d'un nuage de points : son enveloppe convexe.

    En vue de dessus, une boite quelconque ne donne pas un rectangle --
    elle peut etre tournee, et ses huit coins se projettent en six
    points au plus. L'enveloppe convexe est la seule facon simple d'en
    tirer un contour juste, quelle que soit l'orientation.

    Marche de Jarvis, suffisante pour huit points. Le contour est ferme
    (le premier point est repete a la fin), pret a tracer.
    """
    P = np.unique(np.round(np.asarray(points, dtype=float)[:, :2], 6), axis=0)
    if len(P) < 3:
        return np.vstack([P, P[:1]]) if len(P) else P

    def a_gauche(a, b, p):
        # produit vectoriel en dimension 2, calcule a la main : numpy 2
        # ne l'accepte plus sur des vecteurs plans
        return ((b[0] - a[0]) * (p[1] - a[1])
                - (b[1] - a[1]) * (p[0] - a[0]))

    def demi_tour(pts):
        pile = []
        for p in pts:
            while len(pile) >= 2 and a_gauche(pile[-2], pile[-1], p) <= 1e-12:
                pile.pop()
            pile.append(p)
        return pile

    ordre = P[np.lexsort((P[:, 1], P[:, 0]))]
    bas = demi_tour(ordre)
    haut = demi_tour(ordre[::-1])
    contour = np.array(bas[:-1] + haut[:-1] + [bas[0]])
    return contour


class Vue2D(QtWidgets.QWidget):
    """Deux projections orthogonales du squelette du robot."""

    def __init__(self, robot: DobotMagician, parent=None):
        super().__init__(parent)
        self.robot = robot
        self._q = np.array(robot.q_home, dtype=float)
        self._trace: List[np.ndarray] = []
        self.pieces = []
        self.pieces_outil = []
        #: courbes des peripheriques, par projection
        self._items_tapis: list = []
        self._items_rail: list = []
        #: deplacement de la BASE, quand un rail porte le robot
        self._base = np.eye(4)

        pg.setConfigOptions(antialias=True)
        grille = QtWidgets.QGridLayout(self)
        grille.setContentsMargins(2, 2, 2, 2)

        self.p_cote = pg.PlotWidget(title="Vue meridienne  (rho, z)  en mm")
        self.p_dessus = pg.PlotWidget(title="Vue de dessus  (x, y)  en mm")
        for p in (self.p_cote, self.p_dessus):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setAspectLocked(True)
        grille.addWidget(self.p_cote, 0, 0)
        grille.addWidget(self.p_dessus, 0, 1)

        stylo_bras = pg.mkPen((250, 200, 60), width=4)
        stylo_chemin = pg.mkPen((60, 200, 235), width=2)
        stylo_trace = pg.mkPen((250, 120, 40), width=2)

        self.c_bras = self.p_cote.plot(pen=stylo_bras, symbol="o", symbolSize=7,
                                       symbolBrush=(230, 230, 230))
        self.c_chemin = self.p_cote.plot(pen=stylo_chemin)
        self.c_trace = self.p_cote.plot(pen=stylo_trace)
        self.c_sol = self.p_cote.plot(pen=pg.mkPen((120, 120, 120), width=1))
        self.c_sol.setData([0, 400], [0, 0])

        self.d_bras = self.p_dessus.plot(pen=stylo_bras, symbol="o", symbolSize=7,
                                         symbolBrush=(230, 230, 230))
        self.d_chemin = self.p_dessus.plot(pen=stylo_chemin)
        self.d_trace = self.p_dessus.plot(pen=stylo_trace)

        marqueur = dict(pen=None, symbol="+", symbolSize=11,
                        symbolPen=pg.mkPen((250, 240, 80), width=2))
        self.c_cibles = self.p_cote.plot(**marqueur)
        self.d_cibles = self.p_dessus.plot(**marqueur)

        self._tracer_enveloppe()
        self.maj(self._q)

    # -- interface commune avec Vue3D ---------------------------------------

    def definir_convoyeur(self, tapis, t: float = 0.0) -> None:
        """
        Dessine le tapis dans les deux projections. `None` efface.

        En vue de dessus, chaque morceau donne son EMPREINTE -- son
        enveloppe convexe au sol. En vue meridienne, l'abscisse etant un
        rayon, une boite se projette sur l'intervalle de rayons qu'elle
        occupe : on la trace donc comme un rectangle (rho, z). C'est une
        projection, pas une coupe -- un tapis pose en travers y parait
        large alors qu'il est mince.

        Le maillage 3D, lui, n'a pas d'equivalent ici : on dessine les
        boites du modele, qui suffisent a situer la bande, le capteur et
        les pieces.
        """
        self._effacer(self._items_tapis)
        self._items_tapis = []
        if tapis is not None:
            self._items_tapis = self._peindre(tapis.boites(t))

    def definir_rail(self, axe, l: float = 0.0) -> None:
        """
        Dessine le rail et y pose le robot. `None` efface et remet la
        base a l'origine.
        """
        self._effacer(self._items_rail)
        self._items_rail = []
        self._base = np.eye(4) if axe is None else axe.matrice(float(l))
        if axe is not None:
            self._items_rail = self._peindre(axe.boites(l))
        self.maj(self._q)

    def _effacer(self, items) -> None:
        for trace, courbe in items:
            trace.removeItem(courbe)

    def _peindre(self, formes) -> list:
        """Trace les boites d'un peripherique dans les deux projections."""
        items = []
        for forme in formes:
            role = forme.get("role", "")
            couleur = COULEURS_ROLES.get(role, (160, 160, 160))
            style = (QtCore.Qt.DashLine if role in ROLES_POINTILLES
                     else QtCore.Qt.SolidLine)
            stylo = pg.mkPen(couleur, width=2, style=style)

            S = sommets_boite(forme)
            C = contour_xy(S)
            if len(C):
                courbe = self.p_dessus.plot(C[:, 0], C[:, 1], pen=stylo)
                items.append((self.p_dessus, courbe))

            rho = np.hypot(S[:, 0], S[:, 1])
            r0, r1 = float(rho.min()), float(rho.max())
            z0, z1 = float(S[:, 2].min()), float(S[:, 2].max())
            courbe = self.p_cote.plot([r0, r1, r1, r0, r0],
                                      [z0, z0, z1, z1, z0], pen=stylo)
            items.append((self.p_cote, courbe))
        return items

    def definir_pieces(self, pieces: Sequence) -> None:
        self.pieces = list(pieces)

    def definir_outil(self, pieces: Sequence) -> None:
        self.pieces_outil = list(pieces)

    def invalider_enveloppe(self) -> None:
        """La geometrie a change (outil) : on retrace le nuage."""
        visible = self._env.isVisible()
        self.p_cote.removeItem(self._env)
        self._tracer_enveloppe()
        self._env.setVisible(visible)

    def maj(self, q_deg: Sequence[float], enregistrer_trace: bool = False) -> None:
        self._q = np.asarray(q_deg, dtype=float).copy()
        P = self.robot.positions_articulations(self._q)
        if not np.allclose(self._base, np.eye(4)):
            P = P + self._base[:3, 3]
        rho = np.hypot(P[:, 0], P[:, 1])
        # au-dela de +/-90 deg on rabat le bras du cote negatif pour rester lisible
        signe = -1.0 if np.cos(np.radians(self._q[0])) < 0.0 else 1.0
        self.c_bras.setData(rho * signe, P[:, 2])
        self.d_bras.setData(P[:, 0], P[:, 1])
        if enregistrer_trace:
            self._trace.append(P[-1].copy())
            if len(self._trace) > 4000:
                self._trace = self._trace[-4000:]
            A = np.array(self._trace)
            self.c_trace.setData(np.hypot(A[:, 0], A[:, 1]), A[:, 2])
            self.d_trace.setData(A[:, 0], A[:, 1])

    def effacer_trace(self) -> None:
        self._trace = []
        self.c_trace.setData([], [])
        self.d_trace.setData([], [])

    def definir_chemin(self, points: Optional[np.ndarray]) -> None:
        if points is None or len(points) == 0:
            self.c_chemin.setData([], [])
            self.d_chemin.setData([], [])
            return
        A = np.asarray(points, dtype=float)
        self.c_chemin.setData(np.hypot(A[:, 0], A[:, 1]), A[:, 2])
        self.d_chemin.setData(A[:, 0], A[:, 1])

    def definir_cibles(self, poses: Optional[Sequence[np.ndarray]],
                       taille: float = 26.0) -> None:
        """Marqueurs des cibles ; `poses` est une suite de matrices 4x4."""
        if poses is None or len(poses) == 0:
            self.c_cibles.setData([], [])
            self.d_cibles.setData([], [])
            return
        P = np.array([np.asarray(T, dtype=float)[:3, 3] for T in poses])
        self.c_cibles.setData(np.hypot(P[:, 0], P[:, 1]), P[:, 2])
        self.d_cibles.setData(P[:, 0], P[:, 1])

    def afficher_reperes_cibles(self, actif: bool) -> None:
        """Sans objet en 2D : les trieds ne sont pas lisibles en projection."""

    def afficher_cibles(self, actif: bool) -> None:
        self.c_cibles.setVisible(actif)
        self.d_cibles.setVisible(actif)

    def afficher_reperes(self, actif: bool) -> None:
        pass

    def afficher_triedre_base(self, actif: bool) -> None:
        """En projection, les axes sont ceux des graphiques eux-memes."""

    def afficher_squelette(self, actif: bool) -> None:
        self.c_bras.setVisible(actif)
        self.d_bras.setVisible(actif)

    def afficher_maillages(self, actif: bool) -> None:
        pass

    def afficher_grille(self, actif: bool) -> None:
        self.p_cote.showGrid(x=actif, y=actif, alpha=0.3)
        self.p_dessus.showGrid(x=actif, y=actif, alpha=0.3)

    def afficher_enveloppe(self, actif: bool) -> None:
        self._env.setVisible(actif)

    def vue(self, nom: str) -> None:
        self.recadrer()

    def zoomer(self, facteur: float) -> None:
        """Zoom sur les deux projections, autour de leur centre."""
        for p in (self.p_cote, self.p_dessus):
            p.getViewBox().scaleBy((float(facteur), float(facteur)))

    def recadrer(self) -> None:
        for p in (self.p_cote, self.p_dessus):
            p.enableAutoRange()

    # -- interne -------------------------------------------------------------

    def _tracer_enveloppe(self) -> None:
        r = self.robot
        nuage = r.enveloppe(60, 60)
        self._env = self.p_cote.plot(nuage[:, 0], nuage[:, 1], pen=None,
                                     symbol="o", symbolSize=2,
                                     symbolBrush=(90, 110, 150, 70),
                                     symbolPen=None)
        self._env.setVisible(False)
