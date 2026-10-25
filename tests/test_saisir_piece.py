"""saisir_piece.py : saisie guidee par la camera, sur un faux robot."""

import math
import os
import sys
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, ".."))
sys.path.insert(0, os.path.join(ICI, "..", "examples", "perception"))
sys.path.insert(0, ICI)

import etalonner_camera_robot as ec      # noqa: E402
import saisir_piece as sp                # noqa: E402
from test_etalonnage import FauxRobot    # noqa: E402


class FauxRobotAngles(FauxRobot):
    def angles(self):
        x, y, z, r = self._p
        return self.modele.mgi_pose(x, y, z, r)


class Piece:
    def __init__(self, x, y, z, lacet, dims=(68.0, 19.0, 17.0)):
        a = math.radians(lacet)
        self.T_robot = np.eye(4)
        self.T_robot[:3, 0] = [math.cos(a), math.sin(a), 0]
        self.T_robot[:3, 1] = [-math.sin(a), math.cos(a), 0]
        self.T_robot[:3, 3] = np.array([x, y, z]) / 1000.0
        self.dimensions = dims
        self.type, self.face, self.numero = "epaisse", "a plat", 1
        self.dessus, self.support, self.hauteur_base = [], None, 0.0

    @property
    def libre(self):
        return True


class FaussePerception:
    def __init__(self, robot, z_table):
        self.robot, self.z_table = robot, z_table
        self.piece = Piece(240.0, -20.0, z_table + 17.0, 30.0)

    def detecter(self):
        return [] if self.robot.fermee else [self.piece]


class TestSaisie(unittest.TestCase):

    def test_un_tour(self):
        z_table = 25.0
        robot = FauxRobotAngles(x=220.0, y=0.0, z=120.0)
        perc = FaussePerception(robot, z_table)
        donnees = {"z_table_mm": z_table, "secteur_base_deg": [-40.0, 20.0],
                   "ecart_r_deg": 0.0, "depart": [220.0, 0.0, 0.0], "mesures": []}
        essai = sp.Essai(robot, perc, donnees, journal=lambda m: None, graine=1)
        essai.equerre()
        q = robot.angles()
        self.assertAlmostEqual(q[1], ec.EQUERRE_J2, places=3)
        robot.trace.clear()
        x, y, z, r = essai.saisir(perc.piece)
        self.assertAlmostEqual(x, 240.0)
        self.assertAlmostEqual(y, -20.0)
        self.assertAlmostEqual(z, z_table + 5.0)            # 12 mm sous le dessus
        self.assertTrue(robot.fermee)
        # descente verticale sur la piece
        i = [k for k, p in enumerate(robot.trace) if abs(p[2] - z) < 1e-6][0]
        self.assertEqual(robot.trace[i - 1][:2], robot.trace[i][:2])
        vise = essai.deposer((x, y, z, r))
        self.assertFalse(robot.fermee)
        self.assertGreaterEqual(math.hypot(vise[0] - x, vise[1] - y), sp.ECART_DEPOSE)
        self.assertAlmostEqual(vise[2], z)                   # dessous sur la table
        self.assertAlmostEqual(robot.angles()[1], ec.EQUERRE_J2, places=3)


if __name__ == "__main__":
    unittest.main()
