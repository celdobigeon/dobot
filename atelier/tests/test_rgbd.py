#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de l'entree RGB-H (examples/perception/apprentissage/rgbd/), sans
GPU, sans poids et sans le jeu reel : scene synthetique (une table et une
piece rendues par lancer de rayons dans la camera de PROFONDEUR, une
camera couleur decalee et d'intrinseques differentes).

  - splatting : le point le plus proche gagne ;
  - ALIGNEMENT : le sommet de la piece, point 3D connu, tombe au bon pixel
    de l'image couleur, ou la carte de hauteur vaut l'epaisseur de la piece ;
    la table vaut 0 ;
  - trous rebouches, codage de la hauteur, ordre des canaux (rgbh, rgh) ;
  - zone de table pour le changement de fond (hors piece), changement de
    fond qui ne touche ni la piece ni la hauteur ;
  - augmentation de couleur qui ne modifie jamais le canal de hauteur.
"""

import os
import sys
import unittest

import numpy as np

try:
    import cv2
except ImportError:                               # pragma: no cover
    cv2 = None

# le paquet et les exemples vivent dans cellule/, voisin de atelier/
RACINE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule")
PERCEPTION = os.path.join(RACINE, "examples", "perception")
APPRENTISSAGE = os.path.join(PERCEPTION, "apprentissage")
RGBD = os.path.join(APPRENTISSAGE, "rgbd")
for d in (PERCEPTION, APPRENTISSAGE, RGBD):
    if d not in sys.path:
        sys.path.insert(0, d)

W, H = 848, 480
K_PROF = np.array([[420.0, 0.0, 424.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]])
K_COUL = np.array([[610.0, 0.0, 430.0], [0.0, 608.0, 245.0], [0.0, 0.0, 1.0]])


def rendre(K, T_ct, boites):
    """Profondeur (m) de la table z = 0 portant des boites [(T face
    superieure, L, l, h) m] : un rayon par pixel, methode des dalles."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)],
                 -1).reshape(-1, 3)
    T_tc = np.linalg.inv(T_ct)
    o, d = T_tc[:3, 3], r @ T_tc[:3, :3].T
    t = -o[2] / d[:, 2]
    t[t <= 0] = np.inf
    for (T, L, l, h) in boites:
        Ti = np.linalg.inv(T)
        o2, d2 = Ti[:3, :3] @ o + Ti[:3, 3], d @ Ti[:3, :3].T
        lo, hi = np.array([-L / 2, -l / 2, -h]), np.array([L / 2, l / 2, 0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - o2) / d2, (hi - o2) / d2
        tmin = np.nanmax(np.minimum(a, b), 1)
        tmax = np.nanmin(np.maximum(a, b), 1)
        touche = (tmax >= tmin) & (tmin > 0) & (tmin < t)
        t = np.where(touche, tmin, t)
    return np.where(np.isfinite(t), t, 0.0).reshape(H, W).astype(np.float32)


def besoin_cv2(test):
    return unittest.skipIf(cv2 is None, "OpenCV absent")(test)


@besoin_cv2
class TestRGBH(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import perception
        import vision
        import rgbh
        cls.pc, cls.rgbh = perception, rgbh
        a = np.radians(50.0)
        cls.T_ct = vision.repere_table(np.array([0.0, -np.cos(a), -np.sin(a)]), 0.45)
        z = 0.45 / np.sin(a)
        cx, cy = (np.linalg.inv(cls.T_ct) @ [0.0, 0.0, z, 1.0])[:2]
        cls.h = 0.019
        cls.T_piece = perception.pose_table(cx, cy, 0.4, cls.h)
        D = rendre(K_PROF, cls.T_ct, [(cls.T_piece, 0.068, 0.017, cls.h)])
        # camera couleur : decalee de 15 mm, tournee de 1 deg
        Tcp = np.eye(4)
        c, s = np.cos(np.radians(1.0)), np.sin(np.radians(1.0))
        Tcp[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        Tcp[:3, 3] = [0.015, 0.0, 0.0]
        cls.Tcp = Tcp
        cls.capture = {"profondeur": D, "K": K_PROF, "coeffs": np.zeros(5),
                       "couleur_brute": np.full((H, W, 3), 200, np.uint8),
                       "K_couleur": K_COUL, "coeffs_couleur": np.zeros(5),
                       "T_couleur_profondeur": Tcp}
        cls.vue = perception.vue_couleur(cls.capture)
        cls.carte, cls.mesuree = rgbh.carte_hauteur(cls.capture, cls.vue)

    def pixel_couleur(self, P_table):
        """Point du repere table -> pixel de l'image couleur."""
        P = self.T_ct[:3, :3] @ P_table + self.T_ct[:3, 3]          # profondeur
        Q = self.Tcp[:3, :3] @ P + self.Tcp[:3, 3]                  # couleur
        return K_COUL[0, 0] * Q[0] / Q[2] + K_COUL[0, 2], K_COUL[1, 1] * Q[1] / Q[2] + K_COUL[1, 2]

    def test_splat_le_plus_proche(self):
        img, zb = self.rgbh.splat(np.array([10.2, 10.7]), np.array([5.1, 5.3]),
                                  np.array([1.0, 0.5]), np.array([7.0, 3.0]), 20, 20, 1)
        self.assertEqual(img[5, 10], 3.0)
        self.assertEqual(zb[5, 10], 0.5)
        self.assertTrue(np.isnan(img[0, 0]))

    def test_alignement_sommet(self):
        """Le centre de la face superieure (point 3D connu) tombe au bon pixel
        couleur, et la carte de hauteur y vaut l'epaisseur."""
        u, v = self.pixel_couleur(self.T_piece[:3, 3])
        h = np.nanmedian(self.carte[int(round(v)) - 1:int(round(v)) + 2,
                                    int(round(u)) - 1:int(round(u)) + 2])
        self.assertAlmostEqual(h, self.h, delta=0.0015)
        # la projection de reference de perception.py est la meme
        uv = self.pc.projeter((self.Tcp @ self.T_ct @ self.T_piece)[:3, 3][None], self.vue)[0]
        self.assertLess(np.hypot(uv[0] - u, uv[1] - v), 1e-6)

    def test_bords_alignes(self):
        """Des points de la face superieure proches de ses coins (a 80 % et
        60 % des demi-cotes) : la hauteur y vaut deja l'epaisseur (pas de
        decalage entre couleur et hauteur)."""
        L, l = 0.068, 0.017
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            P = self.T_piece[:3, :3] @ [sx * L * 0.4, sy * l * 0.3, 0.0] + self.T_piece[:3, 3]
            u, v = self.pixel_couleur(P)
            self.assertAlmostEqual(float(self.carte[int(round(v)), int(round(u))]), self.h,
                                   delta=0.002)

    def test_table_a_zero(self):
        P = self.T_piece[:3, 3] + np.array([0.08, 0.0, -self.h])
        u, v = self.pixel_couleur(P)
        self.assertLess(abs(float(self.carte[int(round(v)), int(round(u))])), 0.001)

    def test_couverture_et_bouchage(self):
        # la couleur a un champ plus etroit : presque tout est mesure
        self.assertGreater(self.mesuree.mean(), 0.9)
        carte = self.carte.copy()
        carte[200:230, 300:340] = np.nan
        b = self.rgbh.boucher(carte)
        self.assertTrue(np.isfinite(b).all())
        np.testing.assert_array_equal(b[self.mesuree & np.isfinite(carte)],
                                      carte[self.mesuree & np.isfinite(carte)].astype(np.float32))

    def test_codage_et_canaux(self):
        h8 = self.rgbh.encoder_hauteur(np.array([0.0, 0.010, -0.02, 0.2, np.nan]))
        np.testing.assert_array_equal(h8, [0, 30, 0, 255, 0])
        self.assertAlmostEqual(float(self.rgbh.decoder_hauteur(30)), 0.010)
        coul = np.zeros((4, 5, 3), np.uint8)
        coul[..., 0], coul[..., 1], coul[..., 2] = 10, 20, 30
        h = np.full((4, 5), 99, np.uint8)
        x = self.rgbh.composer(coul, h, "rgbh")
        self.assertEqual(x.shape, (4, 5, 4))
        np.testing.assert_array_equal(x[0, 0], [10, 20, 30, 99])
        np.testing.assert_array_equal(self.rgbh.composer(coul, h, "rgh")[0, 0], [99, 20, 30])

    def test_image_rgbh_png(self):
        img, h = self.rgbh.image_rgbh(self.capture, "rgbh", self.vue)
        self.assertEqual(img.shape, (H, W, 4))
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x.png")
            cv2.imwrite(f, img)
            relu = cv2.imread(f, cv2.IMREAD_UNCHANGED)       # comme ultralytics (channels: 4)
        np.testing.assert_array_equal(relu, img)

    def test_changement_de_fond(self):
        masque = np.zeros((H, W), np.uint8)
        L, l = 0.068, 0.017
        q = np.array([self.pixel_couleur(self.T_piece[:3, :3] @ [sx * L / 2, sy * l / 2, dz]
                                         + self.T_piece[:3, 3])
                      for sx in (-1, 1) for sy in (-1, 1) for dz in (0.0, -self.h)])
        cv2.fillConvexPoly(masque, cv2.convexHull(np.round(q).astype(np.int32)), 1)
        masque = masque > 0
        zone = self.rgbh.zone_table(self.carte, self.mesuree, [masque])
        self.assertFalse((zone & masque).any())
        self.assertGreater(zone.mean(), 0.3)
        image = np.full((H, W, 3), 200, np.uint8)
        image[masque] = (60, 120, 180)
        nouveau = np.zeros((H, W, 3), np.float32)
        nouveau[..., 1] = 0.5
        im2 = self.rgbh.changer_fond(image, zone, nouveau, garder_lumiere=False)
        np.testing.assert_array_equal(im2[masque], image[masque])
        # la zone est repeinte (sauf son bord adouci d'un pixel)
        ecart = np.abs(im2[zone].astype(int) - [0, 128, 0]).max(axis=1)
        self.assertGreater((ecart <= 2).mean(), 0.95)


class TestAugmentationCouleur(unittest.TestCase):

    def setUp(self):
        try:
            import entrainer_rgbh
        except Exception as exc:                       # pragma: no cover
            self.skipTest("ultralytics absent : {}".format(exc))
        self.er = entrainer_rgbh

    def test_hauteur_intacte(self):
        rng = np.random.default_rng(0)
        for var, n in (("rgbh", 4), ("rgh", 3)):
            img = rng.integers(0, 256, (32, 32, n), dtype=np.uint8)
            canal_h = 3 if var == "rgbh" else 0
            h0 = img[..., canal_h].copy()
            avant = img.copy()
            self.er.hsv_couleur(img, 0.015, 0.5, 0.4, var, rng)
            np.testing.assert_array_equal(img[..., canal_h], h0)
            self.assertFalse(np.array_equal(img, avant))       # la couleur a change


if __name__ == "__main__":
    unittest.main()
