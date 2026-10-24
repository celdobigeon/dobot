# -*- coding: utf-8 -*-
"""
Detection des pieces par modele (examples/perception/vision_modele.py).

Les verifications portent sur des scenes SYNTHETIQUES dont on connait la
verite au dixieme de millimetre (examples/perception/synthese.py) : une
camera de profondeur simulee, avec bruit, lissage des aretes et trous
d'occultation. On y teste ce que la detection par taches ne savait pas
faire -- separer deux pieces qui se touchent, trouver une piece etroite --
et la precision de la pose rendue.
"""

import os
import sys
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "examples", "perception"))

try:
    import cv2  # noqa: F401
    import synthese as sy
    import vision
    import vision_modele as vm
except ImportError:                                  # pragma: no cover
    cv2 = None


def piece(type_, face, x, y, lacet):
    return sy.PieceVraie(type_, face, x, y, lacet, sy.forme(sy.TYPES[type_], face))


class TestDetectionParModele(unittest.TestCase):

    def setUp(self):
        if cv2 is None:
            self.skipTest("OpenCV absent")
        self.cam = sy.Camera(distance=0.55, inclinaison=35)

    def detecter(self, pieces, graine=0):
        sc, verite = sy.scene(pieces, self.cam, graine=graine)
        res = vm.detecter(sc, sy.TYPES)
        return res, verite, sc

    def test_deux_pieces_qui_se_touchent(self):
        # deux epaisses a plat, cote a cote sans jeu : une seule tache de
        # 68 x 38 mm, que la detection par taches ne reconnait pas
        pcs = [piece("epaisse", "a plat", 0, 0, 20),
               piece("epaisse", "a plat", 0, 0, 20)]
        a = np.radians(20)
        normale = np.array([-np.sin(a), np.cos(a)])
        for p, signe in zip(pcs, (-1, 1)):
            p.x, p.y = signe * normale * 9.5
        res, verite, sc = self.detecter(pcs)
        bilan = sy.comparer(res.pieces, verite)
        self.assertEqual(bilan.trouves, 2)
        self.assertEqual(bilan.faux, 0)
        self.assertEqual(len(vision.detecter(sc, sy.TYPES).pieces), 0)

    def test_piece_fine_sur_la_tranche(self):
        # face superieure de 8 mm de large seulement
        res, verite, _ = self.detecter([piece("fine", "sur la tranche", 10, -5, -35)])
        self.assertEqual(len(res.pieces), 1)
        p = res.pieces[0]
        self.assertEqual((p.type, p.face), ("fine", "sur la tranche"))
        self.assertLess(sy.comparer(res.pieces, verite).err_pos[0], 2.0)

    def test_precision_de_la_pose(self):
        pcs = [piece("epaisse", "a plat", -60, 30, 10),
               piece("fine", "a plat", 40, 40, -60),
               piece("epaisse", "sur la tranche", 10, -50, 75)]
        res, verite, _ = self.detecter(pcs, graine=1)
        bilan = sy.comparer(res.pieces, verite)
        self.assertEqual(bilan.trouves, 3)
        self.assertEqual(bilan.faux, 0)
        self.assertLess(max(bilan.err_pos), 2.0)       # mm, dans le plan
        self.assertLess(max(bilan.err_ang), 2.0)       # degres
        self.assertEqual(bilan.bon_type, 3)
        for p in res.pieces:
            self.assertGreater(p.score, 0.6)
            # T_camera et T_table decrivent le meme repere
            np.testing.assert_allclose(res.T_camera_table @ p.T_table,
                                       p.T_camera, atol=1e-9)

    def test_table_vide(self):
        res, _, _ = self.detecter([])
        self.assertEqual(res.pieces, [])

    def test_vue_redressee_a_la_bonne_echelle(self):
        # dans la vue de dessus, une piece a sa vraie taille quel que soit
        # l'angle de vue : la face superieure d'une epaisse a plat couvre
        # 68 x 19 mm, soit environ 1292 cases de 1 mm a 17 mm de haut
        self.cam = sy.Camera(distance=0.55, inclinaison=55)
        sc, _ = sy.scene([piece("epaisse", "a plat", 0, 0, 30)], self.cam)
        prep = vm.preparer(sc, sy.TYPES)
        H = prep.vue.hauteur
        haut = np.isfinite(H) & (np.abs(H - 17) < 3)
        self.assertGreater(haut.sum(), 0.6 * 68 * 19)
        self.assertLess(haut.sum(), 1.15 * 68 * 19)


if __name__ == "__main__":
    unittest.main()
