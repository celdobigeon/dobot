"""PontReel.amener_en_posture : equerre en mode manuel (JOG), hors file."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cellule"))

from dobot_sim import DobotMagician          # noqa: E402
from dobot_sim.reel import PontReel          # noqa: E402


class _LiJog:
    """Un controleur dont les angles avancent tant qu'un JOG est actif."""

    def __init__(self, q):
        self.q = np.array(q, float)
        self.cmd = 0
        self.cmds = []

    def parametres_jog_articulaire(self, v, a):
        self.v = v

    def jog(self, cmd, articulaire=True):
        # le bras a bouge pendant l'impulsion qui se termine
        sens = {3: (1, 1), 4: (1, -1), 5: (2, 1), 6: (2, -1)}.get(self.cmd)
        if sens:
            self.q[sens[0]] += sens[1] * 0.3
        self.cmd = cmd
        self.cmds.append(cmd)

    def pose(self):
        return type("P", (), {"angles": tuple(self.q)})()


class TestEquerre(unittest.TestCase):

    def test_amene_en_equerre_sans_tourner_la_base(self):
        import dobot_sim.reel as reel
        li = _LiJog([-5.8, 54.7, 17.4, 9.3])
        pont = PontReel(DobotMagician(), journal=lambda m: None)
        pont._exiger = lambda: li
        ancien = reel.time.sleep
        reel.time.sleep = lambda s: None
        try:
            q = pont.amener_en_posture(1.0, 0.0)
        finally:
            reel.time.sleep = ancien
        self.assertLess(abs(q[1] - 1.0), 0.5)
        self.assertLess(abs(q[2]), 0.5)
        self.assertEqual((q[0], q[3]), (-5.8, 9.3))
        self.assertEqual(li.cmds[-1], 0)                 # finit a l'arret
        self.assertTrue(set(li.cmds) <= {0, 3, 4, 5, 6})  # jamais J1 ni J4


if __name__ == "__main__":
    unittest.main()
