"""
Etalonnage camera -> robot par la pince (etalonner_par_pince.py), de bout en
bout sur une scene SIMULEE : une camera virtuelle rend la profondeur de la
table et du modele de la pince, la ou le faux robot la place.
"""

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
import etalonner_par_pince as ep         # noqa: E402
from test_etalonnage import FauxRobot    # noqa: E402


class FauxRobotAngles(FauxRobot):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.modele.geo.lt = 107.0
        self._p = list(self.modele.pose([0.0, 10.0, 20.0, 0.0]))

    def angles(self):
        return self.modele.mgi_pose(*self._p)

    def movj(self, x=None, y=None, z=None, r=None, q=None, **kw):
        if q is not None:
            self._p = list(self.modele.pose(q))
            return
        self.movl(x=x, y=y, z=z, r=r)


class CameraVirtuelle:
    """Profondeur (m) d'une table a z_table et de la pince, vue de T (mm)."""

    def __init__(self, robot, T, z_table=0.0, W=424, H=240, bruit=0.0007, graine=0):
        self.robot, self.T, self.z_table = robot, np.asarray(T, float), z_table
        self.W, self.H = W, H
        self.K = np.array([[212.0, 0, W / 2], [0, 212.0, H / 2], [0, 0, 1]])
        self.modele = ep.charger_modele(pas=0.8)
        self.bruit = bruit
        self.rng = np.random.default_rng(graine)

    def __call__(self):
        W, H, K, T = self.W, self.H, self.K, self.T
        R, t = T[:3, :3], T[:3, 3]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
        dw = d @ R.T                                   # rayons, repere robot
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.z_table - t[2]) / dw[..., 2]     # profondeur camera = s (z camera = 1)
        z = np.where((s > 0) & (s < 3000), s, 0.0)
        x, y, zz, r = self.robot.pose()
        A = ep.pose_outil(x, y, zz, r)
        P = self.modele @ A[:3, :3].T + A[:3, 3]
        Pc = (P - t) @ R
        ok = Pc[:, 2] > 50
        Pc = Pc[ok]
        uu = np.round(K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]).astype(int)
        vv = np.round(K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]).astype(int)
        for du in (0, 1):
            for dv in (0, 1):
                a, b = uu + du, vv + dv
                m = (a >= 0) & (a < W) & (b >= 0) & (b < H)
                a, b, zc = a[m], b[m], Pc[m, 2]
                ordre = np.argsort(-zc)                # les plus proches ecrits en dernier
                cur = z[b[ordre], a[ordre]]
                zc2 = np.where((cur > 0) & (cur < zc[ordre]), cur, zc[ordre])
                z[b[ordre], a[ordre]] = zc2
        z = z / 1000.0
        z[z > 0] += self.rng.normal(0, self.bruit, int((z > 0).sum()))
        return {"profondeur": z.astype(np.float32), "K": K}


class TestEtalonnagePince(unittest.TestCase):

    def test_de_bout_en_bout(self):
        robot = FauxRobotAngles()
        T_vrai = ec.regarder([150.0, -380.0, 165.0], [250.0, 0.0, 0.0])
        cam = CameraVirtuelle(robot, T_vrai, z_table=0.0)
        messages = []
        et = ep.EtalonnagePince(robot, cam, journal=messages.append, archive=None, graine=1)
        et.relever_decor()
        self.assertAlmostEqual(et.seg.hauteur_camera, 165.0, delta=2.0)
        et.amorcer()
        et.couvrir(n_max=20)
        T = et.T_robot_camera()
        moy, pire = ep.ecart_zone(T, T_vrai)
        self.assertLess(pire, 3.0, "ecart a la vraie camera : {:.1f} mm".format(pire))
        self.assertAlmostEqual(float(et.solution.t[2]), 0.0, delta=2.0)   # table a z = 0
        res = et.resultat(107.0, (moy, pire))
        self.assertIn("T_robot_camera", res)
        self.assertEqual(res["methode"], "pince")


if __name__ == "__main__":
    unittest.main()
