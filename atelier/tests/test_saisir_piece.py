"""saisir_piece.py : saisie guidee par la camera, sur un faux robot."""

import math
import os
import sys
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "..", "cellule"))
sys.path.insert(0, os.path.join(ICI, "..", "..", "cellule", "examples", "perception"))
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
        self.T_table = self.T_robot.copy()
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
        self.assertAlmostEqual(z, z_table + sp.PIED)
        self.assertTrue(robot.fermee)
        # descente verticale sur la piece
        i = [k for k, p in enumerate(robot.trace) if abs(p[2] - z) < 1e-6][0]
        self.assertEqual(robot.trace[i - 1][:2], robot.trace[i][:2])
        vise = essai.deposer((x, y, z, r))
        self.assertFalse(robot.fermee)
        self.assertGreaterEqual(math.hypot(vise[0] - x, vise[1] - y), sp.ECART_DEPOSE)
        self.assertAlmostEqual(vise[2], z)                   # dessous sur la table

    def test_depose_a_la_hauteur_de_la_saisie(self):
        """Le dessous de la piece vu 4 mm trop haut par la camera : la saisie
        prend la table etalonnee, la depose AUSSI (bug du 25/10 : elle
        descendait de 4 mm de trop)."""
        z_table = 25.0
        robot = FauxRobotAngles(x=220.0, y=0.0, z=120.0)
        perc = FaussePerception(robot, z_table)
        perc.piece = Piece(240.0, -20.0, z_table + 4.0 + 17.0, 30.0)
        donnees = {"z_table_mm": z_table, "secteur_base_deg": [-40.0, 20.0],
                   "ecart_r_deg": 0.0, "depart": [220.0, 0.0, 0.0], "mesures": []}
        essai = sp.Essai(robot, perc, donnees, journal=lambda m: None, graine=1)
        prise = essai.saisir(perc.piece)
        vise = essai.deposer(prise)
        self.assertAlmostEqual(prise[2], z_table + sp.PIED)
        self.assertAlmostEqual(vise[2], z_table + sp.PIED)
        self.assertAlmostEqual(robot.angles()[1], ec.EQUERRE_J2, places=3)


class TestHorsAtteinte(unittest.TestCase):

    def test_piece_trop_loin_on_demande_de_la_deplacer(self):
        robot = FauxRobotAngles(x=220.0, y=0.0, z=120.0)
        perc = FaussePerception(robot, 25.0)
        perc.piece = Piece(420.0, 0.0, 42.0, 0.0)            # hors d'atteinte
        messages = []
        donnees = {"z_table_mm": 25.0, "secteur_base_deg": [-40.0, 20.0],
                   "ecart_r_deg": 0.0, "depart": [220.0, 0.0, 0.0], "mesures": []}
        questions = []
        essai = sp.Essai(robot, perc, donnees, journal=messages.append, graine=1,
                         demander=lambda t: questions.append(t) or "")
        self.assertIsNone(essai.un_tour())
        self.assertFalse(robot.fermee)
        self.assertTrue(any("HORS D'ATTEINTE" in q for q in questions))

    def test_essayer_quand_meme(self):
        robot = FauxRobotAngles(x=220.0, y=0.0, z=120.0)
        perc = FaussePerception(robot, 25.0)
        # pres d'une butee : refuse avec marges, possible sans
        donnees = {"z_table_mm": 25.0, "secteur_base_deg": [-40.0, 20.0],
                   "ecart_r_deg": 0.0, "depart": [220.0, 0.0, 0.0], "mesures": []}
        essai = sp.Essai(robot, perc, donnees, journal=lambda m: None, graine=1,
                         demander=lambda t: "f")
        z = 30.0
        candidats = [(x, y) for x in range(150, 340, 4) for y in range(-150, 60, 4)
                     if essai.verifier(x, y, z, 0.0)[0]]
        essai.forcer = True
        ok = [(x, y) for x, y in candidats if not essai.verifier(x, y, z, 0.0)[0]]
        essai.forcer = False
        x, y = ok[0]
        perc.piece = Piece(x, y, 25.0 + 17.0, 0.0)
        essai.un_tour()
        self.assertFalse(essai.forcer)                   # retabli apres l'essai
        self.assertTrue(any(abs(p[0] - x) < 1e-6 and abs(p[1] - y) < 1e-6
                            for p in robot.trace))


class TestForcer(unittest.TestCase):

    def test_forcer_ouvre_des_positions(self):
        donnees = {"z_table_mm": 25.0, "secteur_base_deg": [-40.0, 20.0],
                   "ecart_r_deg": 0.0, "depart": [220.0, 0.0, 0.0], "mesures": []}
        prudent = sp.Essai(FauxRobotAngles(), None, donnees, journal=lambda m: None)
        force = sp.Essai(FauxRobotAngles(), None, donnees, journal=lambda m: None,
                         forcer=True)
        z = 30.0
        pts = [(x, y, 0.0) for x in range(150, 340, 5) for y in range(-200, 120, 5)]
        n_prudent = sum(not prudent.verifier(x, y, z, r)[0] for x, y, r in pts)
        n_force = sum(not force.verifier(x, y, z, r)[0] for x, y, r in pts)
        self.assertGreater(n_force, n_prudent)
        # avec --forcer, le transit laisse toujours une descente verticale
        for x, y, r in pts:
            raison, h = force.verifier(x, y, z, r)
            if not raison:
                self.assertGreaterEqual(h - z, ec.APPROCHE)
        print("\n  positions saisissables : {} prudent, {} avec --forcer".format(
            n_prudent, n_force))


if __name__ == "__main__":
    unittest.main()
