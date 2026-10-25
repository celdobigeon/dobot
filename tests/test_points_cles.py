#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests du reseau de POINTS CLES (examples/perception/apprentissage/
points_cles/), sans GPU, sans poids et sans le jeu de donnees reel :

  - ordre canonique des coins : independant du coin de depart et du sens
    du contour, compatible avec le retournement horizontal (flip_idx),
    saut de 180 deg (symetrie) quand le grand cote passe par la verticale ;
  - PnP sur une piece synthetique connue : pose retrouvee a moins de 1 mm
    et 0.5 deg sans bruit (pose sur plan et IPPE libre), bon (type, face) ;
    piece inclinee retrouvee par IPPE ;
  - format des etiquettes YOLO-pose (ecriture / relecture, visibilite 0),
    etiquettes d'une image rendue (meta de synthese_images.py) ;
  - perte de coins invariante a la symetrie (si torch et ultralytics).
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ImportError:                               # pragma: no cover
    cv2 = None

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCEPTION = os.path.join(RACINE, "examples", "perception")
APPRENTISSAGE = os.path.join(PERCEPTION, "apprentissage")
POINTS_CLES = os.path.join(APPRENTISSAGE, "points_cles")
for d in (PERCEPTION, APPRENTISSAGE, POINTS_CLES):
    if d not in sys.path:
        sys.path.insert(0, d)

K = np.array([[610.0, 0.0, 424.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])
W, H = 848, 480
TYPES = {"epaisse": (17.0, 19.0, 68.0), "fine": (8.0, 17.0, 68.0)}


def T_vue_table(hauteur=0.45, plongee=50.0):
    """Repere table vu d'une camera a `hauteur` m, inclinee de `plongee`
    degres vers le bas (comme sur le poste)."""
    import vision
    a = np.radians(plongee)
    return vision.repere_table(np.array([0.0, -np.cos(a), -np.sin(a)]), hauteur)


def point_vise(T_vt, z=0.6):
    return (np.linalg.inv(T_vt) @ [0.0, 0.0, z, 1.0])[:2]


def besoin_cv2(test):
    return unittest.skipIf(cv2 is None, "OpenCV absent")(test)


@besoin_cv2
class TestOrdreCanonique(unittest.TestCase):

    def setUp(self):
        import coins
        self.kc = coins

    def test_rectangle_horizontal(self):
        # grand cote horizontal : haut-gauche, haut-droite, bas-droite, bas-gauche
        p = np.array([[100, 50], [300, 50], [300, 90], [100, 90]], float)
        np.testing.assert_allclose(self.kc.canoniser(p),
                                   [[100, 50], [300, 50], [300, 90], [100, 90]])

    def test_independant_du_depart_et_du_sens(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            c = rng.uniform(100, 700, 2)
            a = rng.uniform(-np.pi, np.pi)
            if abs(np.cos(a)) < 0.05:
                continue
            u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            p = np.array([c - 60 * u - 15 * v, c + 60 * u - 15 * v,
                          c + 60 * u + 15 * v, c - 60 * u + 15 * v])
            ref = self.kc.canoniser(p)
            # les 4 contours valides (arete 0 -> 1 le long du grand cote)
            for ordre in ([0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2], [3, 2, 1, 0]):
                np.testing.assert_allclose(self.kc.canoniser(p[ordre]), ref)
            # coin 0 a gauche du coin 1, 3 a gauche de 2
            self.assertLess(ref[0, 0], ref[1, 0])
            self.assertLess(ref[3, 0], ref[2, 0])

    def test_petit_cote_en_premier_avec_3d(self):
        p = np.array([[100, 50], [300, 50], [300, 90], [100, 90]], float)
        q3 = np.array([[0, 0, 0], [0.068, 0, 0], [0.068, 0.017, 0], [0, 0.017, 0]])
        # contour donne a partir d'un petit cote
        p2, q2 = p[[1, 2, 3, 0]], q3[[1, 2, 3, 0]]
        np.testing.assert_allclose(self.kc.canoniser(p2, q2), self.kc.canoniser(p))

    def test_retournement_horizontal(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            c = rng.uniform(100, 700, 2)
            a = rng.uniform(-np.pi, np.pi)
            if abs(np.cos(a)) < 0.05:
                continue
            u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            p = np.array([c - 60 * u - 15 * v, c + 60 * u - 15 * v,
                          c + 60 * u + 15 * v, c - 60 * u + 15 * v])
            ref = self.kc.canoniser(p)
            miroir = ref * [-1, 1] + [W - 1, 0]
            # le miroir d'un ordre canonique, permute par flip_idx, est canonique
            np.testing.assert_allclose(miroir[list(self.kc.FLIP_IDX)],
                                       self.kc.canoniser(miroir[[1, 0, 3, 2]]))

    def test_saut_vertical_est_une_symetrie(self):
        def piece(a):
            u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            c = np.array([400.0, 240.0])
            return np.array([c - 60 * u - 15 * v, c + 60 * u - 15 * v,
                             c + 60 * u + 15 * v, c - 60 * u + 15 * v])
        a = self.kc.canoniser(piece(np.pi / 2 - 1e-3))
        b = self.kc.canoniser(piece(np.pi / 2 + 1e-3))
        # presque la meme image, ordre tourne de 180 deg...
        self.assertGreater(np.abs(a - b).max(), 50)
        np.testing.assert_allclose(a[[2, 3, 0, 1]], b, atol=0.5)
        # ... que l'erreur (au mieux de la symetrie) ne compte pas
        self.assertLess(self.kc.erreur_coins(a, b), 0.5)


@besoin_cv2
class TestPnP(unittest.TestCase):

    def setUp(self):
        import coins
        import proposer
        import perception
        self.kc, self.prop, self.pc = coins, proposer, perception
        self.T_vt = T_vue_table()
        self.c = point_vise(self.T_vt)

    def coins_de(self, typ, face, x, y, th, base=0.0):
        L, l, h = (v / 1000.0 for v in self.pc.forme(TYPES[typ], face))
        T = self.pc.pose_table(x, y, th, base + h)
        uv = self.kc.coins_image(self.T_vt @ T, L, l, K)
        return T, self.kc.canoniser(uv)

    def test_pose_sur_plan_exacte(self):
        cas = [("fine", "a plat", 0.01, 0.02, 0.4), ("epaisse", "sur la tranche", -0.03, 0.0, -1.1),
               ("epaisse", "a plat", 0.0, -0.04, 1.5), ("fine", "sur la tranche", 0.05, 0.03, 0.0),
               ("epaisse", "debout", -0.02, 0.05, 0.7)]
        for typ, face, dx, dy, th in cas:
            T, uv = self.coins_de(typ, face, self.c[0] + dx, self.c[1] + dy, th)
            h_mes = T[2, 3] * 1000.0
            hyp = self.prop.estimer_pose(uv, K, self.T_vt, TYPES, typ, 1.0, h_mes)
            dt, da = self.kc.ecart_pose(hyp.T_table, T)
            self.assertLess(dt, 1.0, (typ, face, dt))
            self.assertLess(da, 0.5, (typ, face, da))
            self.assertEqual((hyp.type, hyp.face), (typ, face))
            self.assertEqual(hyp.methode, "plan")
            self.assertLess(hyp.erreur_px, 0.01)

    def test_ippe_libre_exacte(self):
        T, uv = self.coins_de("fine", "a plat", self.c[0], self.c[1] + 0.02, 0.8)
        Tv, e = self.kc.pnp_ippe(uv, 0.068, 0.017, K)
        self.assertIsNotNone(Tv)
        Tt = np.linalg.inv(self.T_vt) @ Tv
        dt, da = self.kc.ecart_pose(Tt, T)
        self.assertLess(dt, 1.0)
        self.assertLess(da, 0.5)
        self.assertLess(self.kc.inclinaison_deg(Tt), 0.5)
        self.assertLess(e, 0.01)

    def test_piece_inclinee(self):
        # un bout sur la table, l'autre sur une piece : 15 deg autour du petit axe
        L, l = 0.068, 0.017
        a = np.radians(15.0)
        T = self.pc.pose_table(self.c[0], self.c[1], 0.3, 0.020)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        T[:3, :3] = T[:3, :3] @ Ry
        uv = self.kc.canoniser(self.kc.coins_image(self.T_vt @ T, L, l, K))
        # sans mesure de la pente par la profondeur : reste horizontale
        hyp = self.prop.estimer_pose(uv, K, self.T_vt, TYPES, "fine", 1.0, float("nan"))
        self.assertEqual(hyp.methode, "plan")
        # profondeur qui contredit la pente (deux bouts a la meme hauteur)
        hyp = self.prop.estimer_pose(uv, K, self.T_vt, TYPES, "fine", 1.0, float("nan"),
                                     h_bouts=(15.0, 15.0))
        self.assertEqual(hyp.methode, "plan")
        # profondeur qui voit la pente : hauteurs vraies aux deux bouts
        bouts = self.prop._bouts_predits(T, L, uv, K, self.T_vt)
        hyp = self.prop.estimer_pose(uv, K, self.T_vt, TYPES, "fine", 1.0, float("nan"),
                                     h_bouts=bouts)
        self.assertEqual(hyp.methode, "ippe")
        self.assertAlmostEqual(hyp.inclinaison, 15.0, delta=0.5)
        dt, _ = self.kc.ecart_pose(hyp.T_table, T)
        self.assertLess(dt, 1.0)

    def test_bruit_reste_raisonnable(self):
        rng = np.random.default_rng(0)
        T, uv = self.coins_de("epaisse", "sur la tranche", self.c[0], self.c[1], 0.2)
        ecarts = []
        for _ in range(30):
            hyp = self.prop.estimer_pose(uv + rng.normal(0, 0.5, uv.shape), K, self.T_vt, TYPES,
                                         "epaisse", 1.0, T[2, 3] * 1000)
            ecarts.append(self.kc.ecart_pose(hyp.T_table, T)[0])
        self.assertLess(np.median(ecarts), 2.0)

    def test_hypothese_vers_piece(self):
        import vision
        T, uv = self.coins_de("fine", "a plat", self.c[0], self.c[1], 0.4)
        hyp = self.prop.estimer_pose(uv, K, self.T_vt, TYPES, "fine", 0.9, 8.0)
        T_ct = np.linalg.inv(np.eye(4))
        hyp.T_camera = T_ct @ hyp.T_table
        p = hyp.piece(3)
        self.assertIsInstance(p, vision.Piece)
        self.assertEqual((p.numero, p.type, p.face, p.origine), (3, "fine", "a plat", "points_cles"))
        self.assertAlmostEqual(p.position_table[2], 8.0, places=3)


@besoin_cv2
class TestEtiquettes(unittest.TestCase):

    def setUp(self):
        import etiquettes
        self.et = etiquettes

    def test_aller_retour(self):
        boite = [100.0, 50.0, 300.0, 120.0]
        uv = np.array([[110, 60], [290, 55], [295, 110], [105, 115]], float)
        lig = self.et.ligne_pose(1, boite, uv, [2, 2, 1, 0], W, H)
        v = lig.split()
        self.assertEqual(len(v), 1 + 4 + 12)
        self.assertEqual(v[0], "1")
        self.assertTrue(all(0.0 <= float(x) <= 2.0 for x in v[1:]))
        c, b, k, vis = self.et.lire_ligne_pose(lig, W, H)
        self.assertEqual(c, 1)
        np.testing.assert_allclose(b, boite, atol=1e-3)
        np.testing.assert_allclose(k[:3], uv[:3], atol=1e-3)
        np.testing.assert_array_equal(vis, [2, 2, 1, 0])
        np.testing.assert_allclose(k[3], [0, 0])          # absent : (0, 0, 0)

    def test_image_rendue(self):
        import coins
        T_vt = T_vue_table()
        c = point_vise(T_vt)
        # deux pieces : coins de la face superieure (table et image), comme
        # synthese_images._une les ecrit
        pieces, lignes_seg = [], []
        for k, (dx, th) in enumerate(((-0.03, 0.3), (0.04, -1.0))):
            T = np.eye(4)
            ca, sa = np.cos(th), np.sin(th)
            T[:3, :3] = [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]
            T[:3, 3] = [c[0] + dx, c[1], 0.004]
            ct = coins.coins_face(0.068, 0.017) @ T[:3, :3].T + T[:3, 3] + [0, 0, 0.004]
            q = ct @ T_vt[:3, :3].T + T_vt[:3, 3]
            uv = coins.projeter(q, K)
            pieces.append({"classe": "fine", "pose": "a plat",
                           "coins_dessus_table": ct.tolist(),
                           "coins_dessus_image": uv.tolist()})
            poly = uv / [W, H]
            lignes_seg.append("1 " + " ".join("{:.6f}".format(x) for x in poly.ravel()))
        with tempfile.TemporaryDirectory() as d:
            lab = os.path.join(d, "s.txt")
            with open(lab, "w") as f:
                f.write("\n".join(lignes_seg) + "\n")
            lignes = self.et.lignes_synthetiques({"pieces": pieces}, lab, list(TYPES), W, H)
            self.assertEqual(len(lignes), 2)
            for lig, p in zip(lignes, pieces):
                cl, b, k, vis = self.et.lire_ligne_pose(lig, W, H)
                self.assertEqual(cl, 1)
                np.testing.assert_array_equal(vis, [2, 2, 2, 2])
                ref = coins.canoniser(np.array(p["coins_dessus_image"]),
                                      np.array(p["coins_dessus_table"]))
                np.testing.assert_allclose(k, ref, atol=1e-3)
            # nombre d'instances different : image ignoree
            with open(lab, "w") as f:
                f.write(lignes_seg[0] + "\n")
            self.assertIsNone(self.et.lignes_synthetiques({"pieces": pieces}, lab,
                                                          list(TYPES), W, H))


class TestPerteSymetrique(unittest.TestCase):

    def setUp(self):
        try:
            import torch  # noqa: F401
            import entrainer_cles
        except Exception as exc:                       # pragma: no cover
            self.skipTest("torch / ultralytics absents : {}".format(exc))
        self.ec = entrainer_cles

    def test_meilleure_permutation(self):
        import torch
        gt = torch.tensor([[[0., 0, 2], [10, 0, 2], [10, 2, 2], [0, 2, 2]]])
        pred = gt[:, [2, 3, 0, 1], :2].clone() + 0.1          # la meme, tournee de 180 deg
        area = torch.tensor([[20.0]])
        sig = torch.ones(4) / 4
        g = self.ec.meilleure_permutation(pred, gt, sig, area)
        torch.testing.assert_close(g[..., :2], gt[:, [2, 3, 0, 1], :2])
        g2 = self.ec.meilleure_permutation(gt[..., :2] + 0.1, gt, sig, area)
        torch.testing.assert_close(g2, gt)

    def test_oks_symetrique(self):
        import torch
        self.ec.installer_symetrie()
        import ultralytics.models.yolo.pose.val as pv
        gt = torch.tensor([[[0., 0, 2], [10, 0, 2], [10, 2, 2], [0, 2, 2]]])
        pred = gt[:, [2, 3, 0, 1]]
        s = pv.kpt_iou(gt, pred, torch.tensor([20.0]), [0.25] * 4)
        self.assertAlmostEqual(float(s[0, 0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
