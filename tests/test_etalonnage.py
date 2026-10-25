"""
Etalonnage camera -> robot (examples/perception/etalonner_camera_robot.py) :
recalage de Kabsch avec decalage dans la pince, et deroule complet sur un
faux robot, une piece et une camera simulees.
"""

import math
import os
import sys
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, ".."))
sys.path.insert(0, os.path.join(ICI, "..", "examples", "perception"))

import etalonner_camera_robot as ec      # noqa: E402


class FauxRobot:
    """Le sous-ensemble de Commande dont l'etalonnage se sert."""

    def __init__(self, x=230.0, y=20.0, z=5.0, r=0.0):
        from dobot_sim import DobotMagician
        self.modele = DobotMagician()
        self._p = [x, y, z, r]
        self.fermee = False
        self.postures = []
        self.trace = []

    def pose(self):
        return tuple(self._p)

    def movj(self, x=None, y=None, z=None, r=None, q=None, **kw):
        if q is not None:
            self._p = list(self.modele.pose(q))
            self.postures.append(list(q))
            return
        self.movl(x=x, y=y, z=z, r=r)

    def movl(self, x=None, y=None, z=None, r=None, **kw):
        for i, v in enumerate((x, y, z, r)):
            if v is not None:
                self._p[i] = float(v)
        assert self.modele.atteignable(*self._p), "cible hors d'atteinte {}".format(self._p)
        self.trace.append(tuple(self._p))

    def pince(self, fermee=True, **kw):
        self.fermee = bool(fermee)

    def repli(self):
        self._p = list(self.modele.pose(self.modele.q_home))

    def attendre(self, s):
        pass


def _paires(T, d, n=12, bruit=0.0, graine=1):
    rng = np.random.default_rng(graine)
    out = []
    for _ in range(n):
        x, y, r = rng.uniform(180, 280), rng.uniform(-120, 120), rng.uniform(-60, 60)
        centre = np.array([x, y, ec.HAUTEUR_PIECE]) + ec.rot_z(r) @ [d[0], d[1], 0]
        p = T[:3, :3].T @ (centre - T[:3, 3]) + rng.normal(0, bruit, 3)
        a = math.radians(r - 90.0)
        axe = T[:3, :3].T @ [math.cos(a), math.sin(a), 0]
        out.append(ec.Paire(np.array([x, y, ec.PRISE]), r, p, axe, 0.0))
    return out


class TestRecalage(unittest.TestCase):

    T = ec.regarder([560.0, 30.0, 120.0], [240.0, 0.0, 0.0])

    def test_exact(self):
        rc = ec.recaler(_paires(self.T, (4.0, -2.0)))
        np.testing.assert_allclose(rc.T_mm, self.T, atol=2e-3)
        np.testing.assert_allclose(rc.decalage, [4.0, -2.0], atol=2e-3)
        self.assertAlmostEqual(abs(ec._angle(rc.ecart_r - 90.0, 180.0)), 0.0, places=3)
        self.assertLess(rc.rms, 1e-3)

    def test_bruit(self):
        rc = ec.recaler(_paires(self.T, (4.0, -2.0), n=20, bruit=0.8))
        self.assertLess(np.linalg.norm(rc.t - self.T[:3, 3]), 3.0)
        self.assertLess(np.linalg.norm(rc.decalage - [4.0, -2.0]), 1.0)
        self.assertLess(rc.rms, 2.0)

    def test_point_aberrant_ecarte(self):
        ps = _paires(self.T, (0.0, 0.0), n=12, bruit=0.3)
        ps[5].camera = ps[5].camera + [30.0, 0.0, 0.0]
        rc = ec.recaler(ps)
        self.assertFalse(rc.gardees[5])
        self.assertEqual(int(rc.gardees.sum()), 11)
        self.assertLess(rc.rms, 1.0)

    def test_prise_au_centre(self):
        T = self.T
        rc = ec.recaler(_paires(T, (4.0, -2.0)))
        centre = np.array([250.0, 30.0, ec.HAUTEUR_PIECE])
        a = math.radians(20.0)
        axe = T[:3, :3].T @ [math.cos(a), math.sin(a), 0]
        q, r = rc.prise(T[:3, :3].T @ (centre - T[:3, 3]), axe)
        np.testing.assert_allclose(q, centre, atol=2e-3)
        self.assertAlmostEqual(abs(ec._angle(r - 110.0, 180.0)), 0.0, places=3)

    def test_fichier(self):
        import json
        import tempfile
        et = ec.Etalonnage(FauxRobot(), None, z_table=0.0)
        et.paires = _paires(self.T, (1.0, 0.0))
        et.recalage = ec.recaler(et.paires)
        with tempfile.TemporaryDirectory() as d:
            chemin = et.enregistrer(os.path.join(d, "e.json"))
            T_m = ec.charger_etalonnage(chemin)
            with open(chemin) as f:
                data = json.load(f)
        np.testing.assert_allclose(T_m[:3, 3] * 1000.0, self.T[:3, 3], atol=2e-3)
        self.assertEqual(data["paires"], 12)


class TestDeroule(unittest.TestCase):

    def test_etalonnage_complet(self):
        messages = []
        robot = ec.PieceSimulee(FauxRobot(), decalage=(0.0, 3.0), ecart_r=90.0)
        camera = ec.CameraSimulee(robot, bruit=0.5, graine=3)
        et = ec.Etalonnage(robot, camera, graine=2, journal=messages.append,
                           demander=lambda _t: "")
        et.z_table = 0.0
        et.demarrer()
        for _ in range(15):
            et.cycle()
        rc = et.recalage
        # ce qui compte : l'erreur SUR LA TABLE (la position de la camera,
        # a 50 cm, est sensible au moindre angle)
        T = camera.T
        for x in (180.0, 240.0, 300.0):
            for y in (-100.0, 0.0, 100.0):
                q = np.array([x, y, ec.HAUTEUR_PIECE])
                p = T[:3, :3].T @ (q - T[:3, 3])
                self.assertLess(np.linalg.norm(rc.vers_robot(p) - q), 1.5)
        self.assertLess(rc.rms, 1.5)
        self.assertLess(np.linalg.norm(rc.decalage - [0.0, 3.0]), 1.0)
        # les points ont quitte les abords du depart
        x0, y0, _r = et.depart
        loin = max(np.hypot(p.pince[0] - x0, p.pince[1] - y0) for p in et.paires)
        self.assertGreater(loin, 2 * ec.RAYON_DEPART)
        v = et.valider(3)
        self.assertTrue(all(e["ecart_mm"] < 3.0 for e in v), v)
        self.assertTrue(robot.tenue)

    def test_deux_pieces_vues(self):
        questions = []

        class DeuxPuisUne:
            K, taille = None, None
            n = 0

            def voir(self):
                self.n += 1
                v = ec.Vue(np.zeros(3), np.array([1.0, 0, 0]))
                return [v, v] if self.n == 1 else [v]

        et = ec.Etalonnage(FauxRobot(), DeuxPuisUne(), z_table=0.0,
                           demander=lambda t: questions.append(t) or "")
        et.observer()
        self.assertEqual(len(questions), 1)


class TestSecteur(unittest.TestCase):

    def test_poses_dans_le_secteur(self):
        robot = ec.PieceSimulee(FauxRobot(x=230.0, y=60.0), decalage=(0.0, 2.0))
        camera = ec.CameraSimulee(robot, bruit=0.5, graine=4)
        et = ec.Etalonnage(robot, camera, z_table=0.0, graine=5, journal=lambda m: None,
                           demander=lambda _t: "", secteur=(5.0, 45.0))
        et.demarrer()
        for _ in range(10):
            et.cycle()
        j1 = [et.j1(*p) for p in robot.robot.trace]
        self.assertGreaterEqual(min(j1), 5.0)
        self.assertLessEqual(max(j1), 45.0)
        # le repli s'est fait au milieu du secteur, pas a J1 = 0
        self.assertTrue(robot.robot.postures)
        self.assertTrue(all(abs(q[0] - 25.0) < 1e-9 for q in robot.robot.postures))
        self.assertLess(et.recalage.rms, 1.5)

    def test_relever_secteur_au_clavier(self):
        robot = FauxRobot(x=230.0, y=0.0, z=60.0)
        touches = iter(["1", "b", "b", "\n", "1", "B", "B", "B", "B", "B", "B", "\n"])
        sect = ec.relever_secteur(robot, robot.modele, lire=lambda: next(touches),
                                  journal=lambda m: None)
        self.assertAlmostEqual(sect[0], -10.0, places=3)
        self.assertAlmostEqual(sect[1], 20.0, places=3)

    def test_clavier_refuse_hors_secteur(self):
        robot = FauxRobot(x=230.0, y=0.0, z=60.0)
        et = ec.Etalonnage(robot, None, z_table=0.0, secteur=(-5.0, 5.0))
        touches = iter(["1", "B", "B", "\n"])
        messages = []
        ec.piloter(robot, lambda: next(touches), messages.append,
                   autorise=lambda x, y, z, r: et.dans_secteur(x, y, z, r, 0.0))
        self.assertTrue(any("hors du secteur" in m for m in messages))
        # le premier pas (5 deg) est accepte, le second refuse
        self.assertAlmostEqual(et.j1(*robot.pose()), 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
