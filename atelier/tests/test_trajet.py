"""
Pilote.trajet : trajectoire continue affine par morceaux. Le simulateur la
joue sans arret aux points intermediaires, le robot reel la recoit en
points CP, et le script exporte en SetCPCmd.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cellule"))

from dobot_sim import Pilote, cible                  # noqa: E402
from dobot_sim.program import Programme              # noqa: E402
from dobot_sim.reel import PontReel                  # noqa: E402
from dobot_sim.trajectory import ProfilTroncons      # noqa: E402

CARRE = [cible(200, 60, 50), (260, 60, 50), (260, 0, 50), (200, 0, 50)]


class _Li:
    def __init__(self):
        self.traces = []

    def __getattr__(self, nom):
        def appel(*a, **k):
            self.traces.append((nom, a, k))
            return len(self.traces)
        return appel


class _Pont(PontReel):
    def _deposer(self, li, commandes, progression, texte):
        for c in commandes:
            c()
        return 0


def _pilote(**kw):
    p = Pilote(outil="pince")
    p.movj(cible(200, 0, 50))
    p.trajet(CARRE, vitesse=80, **kw)
    p.movl(z=80)
    return p


def _vitesse(traj, t, h=1e-3):
    a = traj.pose(t - h)[:3, 3]
    b = traj.pose(t + h)[:3, 3]
    return float(np.linalg.norm(b - a) / (2 * h))


class TestTrajetSimule(unittest.TestCase):

    def test_instructions(self):
        ins = _pilote().programme.instructions
        self.assertEqual([i.type for i in ins], ["MOVJ"] + ["MOVL"] * 5)
        self.assertEqual([i.continu for i in ins],
                         [False, True, True, True, False, False])

    def test_chaine_trouvee(self):
        p = _pilote()
        traj = p.trajectoire()
        self.assertEqual(p.programme.chaines, [[1, 2, 3, 4]])
        for k in (1, 2, 3, 4):
            self.assertIsInstance(traj.segments[k].profil, ProfilTroncons)

    def test_passe_par_les_coins_sans_s_arreter(self):
        p = _pilote()
        traj = p.trajectoire()
        for k, coin in zip((2, 3, 4), ([200, 60, 50], [260, 60, 50],
                                       [260, 0, 50])):
            t = traj.debuts[k]
            np.testing.assert_allclose(traj.pose(t)[:3, 3], coin, atol=1e-6)
            v = _vitesse(traj, t)
            self.assertGreater(v, 5.0)        # pas d'arret au coin...
            self.assertLess(v, 80.0)          # ...mais un ralentissement
        # arret au dernier point, comme au premier
        self.assertLess(_vitesse(traj, traj.debuts[5] - 2e-3, 1e-3), 3.0)

    def test_vitesse_de_croisiere_atteinte(self):
        traj = _pilote().trajectoire()
        t = traj.debuts[2] + traj.segments[2].duree / 2
        self.assertAlmostEqual(_vitesse(traj, t), 80.0, delta=1.0)

    def test_vitesse_plafonnee(self):
        p = Pilote()
        p.movj(cible(200, 0, 50))
        p.trajet([(200, 150, 50), (200, 151, 50)], vitesse=300)
        traj = p.trajectoire()
        t = traj.debuts[1] + traj.segments[1].duree / 2
        self.assertLessEqual(_vitesse(traj, t), 125.0 + 1e-6)

    def test_rotation_refusee(self):
        p = Pilote()
        p.movj(cible(200, 0, 50))
        with self.assertRaises(ValueError):
            p.trajet([cible(200, 60, 50, rz=30)])
        self.assertEqual(len(p.programme), 1)
        # la pose memorisee n'a pas bouge
        p.movl(z=60)
        self.assertAlmostEqual(p.programme.instructions[-1].pose[3], 0.0)

    def test_rechargement_json(self):
        p = _pilote()
        q = Programme.depuis_dict(p.programme.vers_dict())
        self.assertEqual([i.continu for i in q.instructions],
                         [i.continu for i in p.programme.instructions])

    def test_anciens_programmes_inchanges(self):
        d = _pilote().programme.vers_dict()
        for i in d["instructions"]:
            del i["continu"]
        q = Programme.depuis_dict(d)
        q.construire(Pilote().robot)
        self.assertEqual(q.chaines, [])


class TestTrajetReel(unittest.TestCase):

    def _traces(self, p):
        pont = _Pont(p.robot)
        pont.liaison = _Li()
        pont.biais = (61.0, 0.0, 0.0)
        pont._exiger = lambda: pont.liaison
        pont.televerser(p.programme, p.q_depart, outil="pince")
        return [(n, a) for n, a, _k in pont.liaison.traces
                if n in ("ptp", "cp", "parametres_cp", "pince", "ventouse")]

    def test_points_cp(self):
        t = self._traces(_pilote())
        noms = [n for n, _a in t]
        self.assertEqual(noms[0], "ptp")
        self.assertEqual(noms[1], "parametres_cp")
        self.assertEqual(noms[-1], "ptp")
        cp = [a for n, a in t if n == "cp"]
        # un point d'amorce a 1 mm, puis les quatre coins
        self.assertEqual(len(cp), 5)
        self.assertEqual(sum(1 for n in noms if n == "ptp"), 2)

    def test_coupure_pompe_apres_la_chaine(self):
        p = Pilote(outil="pince")
        p.movj(cible(200, 0, 50))
        p.pince(True)
        p.pince(False)
        p.trajet(CARRE, vitesse=80)
        t = self._traces(p)
        noms = [n for n, _a in t]
        dernier_cp = max(i for i, n in enumerate(noms) if n == "cp")
        self.assertEqual(noms[dernier_cp + 1:], ["ventouse", "pince"])
        self.assertNotIn("ventouse", noms[:dernier_cp])


class TestTrajetExport(unittest.TestCase):

    def test_script_cp(self):
        p = _pilote()
        script = p.programme.exporter_python(p.robot)
        self.assertEqual(script.count("SetCPCmd"), 4)
        self.assertEqual(script.count("SetCPParams"), 1)
        self.assertEqual(script.count("SetPTPCmd"), 2)


if __name__ == "__main__":
    unittest.main()
