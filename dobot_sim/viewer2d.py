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
from pyqtgraph.Qt import QtWidgets

from .robot import DobotMagician

__all__ = ["Vue2D"]


class Vue2D(QtWidgets.QWidget):
    """Deux projections orthogonales du squelette du robot."""

    def __init__(self, robot: DobotMagician, parent=None):
        super().__init__(parent)
        self.robot = robot
        self._q = np.array(robot.q_home, dtype=float)
        self._trace: List[np.ndarray] = []
        self.pieces = []
        self.pieces_outil = []

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
