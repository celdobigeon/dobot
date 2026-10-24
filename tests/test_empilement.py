# -*- coding: utf-8 -*-
"""
Pieces EMPILEES (examples/perception) : une piece posee sur une autre.

Toute la geometrie supposait une piece posee sur la table : la hauteur de sa
face superieure disait son type et sa face d'appui. Posee sur une autre, sa
face superieure est a (sommet de la porteuse + son epaisseur). On verifie,
sur des scenes SYNTHETIQUES (synthese.py, verite connue), que :
  - le generateur de scenes empile sans changer les scenes existantes ;
  - la detection par modele (vision_modele) trouve les DEUX pieces d'une
    pile, dit laquelle porte l'autre, classe bien celle du dessus, et que
    l'ordre de prise commence par elle ;
  - une scene sans pile donne le meme resultat avec ou sans la recherche
    d'empilements ;
  - l'etage masques de perception.py (reseau remplace par des masques
    exacts) fait de meme.
"""

import os
import sys
import types
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


def besoin_cv2(test):
    return unittest.skipIf(cv2 is None, "OpenCV absent")(test)


@besoin_cv2
class TestSynthese(unittest.TestCase):

    def test_sans_empilement_inchange(self):
        """Les n premieres pieces sont tirees comme avant : une scene sans
        empilement ne change pas."""
        a = sy.placer(np.random.default_rng(4), 8, sy.TYPES, touchantes=3)
        b = sy.placer(np.random.default_rng(4), 8, sy.TYPES, touchantes=3, empilees=0)
        self.assertEqual([(p.x, p.y, p.lacet, p.type, p.face) for p in a],
                         [(p.x, p.y, p.lacet, p.type, p.face) for p in b])
        self.assertTrue(all(p.z == 0 and p.support is None for p in a))

    def test_piles_coherentes(self):
        for graine in range(5):
            pcs = sy.placer(np.random.default_rng(graine), 8, sy.TYPES,
                            touchantes=1, empilees=3)
            for i, p in enumerate(pcs):
                if p.support is None:
                    self.assertEqual(p.z, 0.0)
                    continue
                s = pcs[p.support]
                self.assertNotEqual(s.face, "debout")
                self.assertAlmostEqual(p.z, s.sommet)
                # le centre est au-dessus de la porteuse : elle ne bascule pas
                c, sn = np.cos(np.radians(s.lacet)), np.sin(np.radians(s.lacet))
                u = (p.x - s.x) * c + (p.y - s.y) * sn
                v = -(p.x - s.x) * sn + (p.y - s.y) * c
                self.assertLessEqual(abs(u), s.dims[0] / 2)
                self.assertLessEqual(abs(v), s.dims[1] / 2)
                # pas d'interpenetration
                for k, q in enumerate(pcs):
                    if k in (i, p.support):
                        continue
                    if q.z < p.sommet - 0.5 and q.sommet > p.z + 0.5:
                        self.assertFalse(sy._se_coupent(p, q, 0.5))

    def test_rendu_d_une_pile(self):
        """La face superieure de la piece du dessus est vue a 34 mm."""
        pcs = [piece("epaisse", "a plat", 0, 0, 0)]
        pcs.append(sy.poser_sur(pcs, 0, "epaisse", "a plat", sy.TYPES))
        cam = sy.Camera(distance=0.55, inclinaison=0)
        sc, ver = sy.scene(pcs, cam, sy.Bruit(k_z2=0, lissage=1, courbure=0,
                                              occultation=False, trous=0))
        self.assertAlmostEqual(ver.centres_camera[1][2], 0.55 - 0.034, places=6)
        H, W = sc.profondeur.shape
        self.assertAlmostEqual(sc.profondeur[H // 2, W // 2], 0.55 - 0.034, places=4)
        self.assertEqual(ver.dessus(0), [1])
        self.assertEqual(ver.objet[H // 2, W // 2], 1)


class TestOrdreDePrise(unittest.TestCase):

    def piece(self, numero, z, support=None, dessus=()):
        T = np.eye(4)
        T[2, 3] = z / 1000.0
        return vision.Piece(numero, "a plat", (68, 19, 17), 0.0, T, T,
                            np.zeros((4, 2)), 0, "epaisse", support=support,
                            dessus=list(dessus))

    def test_libres_d_abord_du_haut_vers_le_bas(self):
        # 1 porte 2, 2 porte 3 (tas de trois) ; 4 et 5 libres, sur la table
        p = [self.piece(1, 17, dessus=[2]), self.piece(2, 34, 1, [3]),
             self.piece(3, 51, 2), self.piece(4, 17), self.piece(5, 8)]
        ordre = [q.numero for q in vision.ordre_de_prise(p)]
        self.assertEqual(ordre, [3, 4, 5, 2, 1])
        self.assertTrue(p[2].libre)
        self.assertFalse(p[0].libre)

    def test_relation_par_support_seul(self):
        # seul `support` est rempli : 2 est sur 1
        p = [self.piece(1, 17), self.piece(2, 34, support=1)]
        self.assertEqual([q.numero for q in vision.ordre_de_prise(p)], [2, 1])

    def test_vide(self):
        self.assertEqual(vision.ordre_de_prise([]), [])


@besoin_cv2
class TestGeometrieDesPiles(unittest.TestCase):

    def test_aire_commune(self):
        m = vm.Modele("e", "a plat", 68, 19, 17)
        f = vm.Modele("f", "a plat", 68, 17, 8)
        self.assertAlmostEqual(vm.recouvrement((0, 0, 0.3, m), (0, 0, 0.3, f)), 68 * 17, 3)
        self.assertAlmostEqual(vm.recouvrement((0, 0, 0, m), (0, 0, np.pi / 2, m)), 19 * 19, 3)
        self.assertEqual(vm.recouvrement((0, 0, 0, m), (100, 0, 0, m)), 0.0)

    def test_bases_de_pile(self):
        mods = vm.modeles(sy.TYPES)
        self.assertEqual(vm.bases_de_pile(mods),
                         [0.0, 8.0, 16.0, 17.0, 19.0, 25.0, 27.0, 34.0, 36.0, 38.0])

    def test_relier(self):
        bas = vm.Modele("epaisse", "a plat", 68, 19, 17)
        haut = bas.sur(17.0)
        a_cote = vm.Modele("fine", "a plat", 68, 17, 8).sur(17.0)
        r = vm.relier_piles([(0, 0, 0, bas), (0, 5, np.pi / 2, haut), (200, 0, 0, a_cote)])
        self.assertEqual(r, [None, 0, -1])


@besoin_cv2
class TestDetectionDesPiles(unittest.TestCase):

    def setUp(self):
        self.cam = sy.Camera(distance=0.55, inclinaison=35)

    def detecter(self, pieces, graine=0, **kw):
        sc, verite = sy.scene(pieces, self.cam, graine=graine)
        return vm.detecter(sc, sy.TYPES, **kw), verite, sc

    def verifier_pile(self, res, verite, i_bas, i_haut):
        bilan = sy.comparer(res.pieces, verite)
        self.assertEqual(bilan.trouves, len(verite.pieces))
        self.assertEqual(bilan.faux, 0)
        self.assertEqual(bilan.bon_support, len(verite.pieces))
        det = np.array([p.T_camera[:3, 3] for p in res.pieces])
        proche = [int(np.argmin(np.linalg.norm(det - c, axis=1)))
                  for c in verite.centres_camera]
        bas, haut = res.pieces[proche[i_bas]], res.pieces[proche[i_haut]]
        self.assertEqual(haut.support, bas.numero)
        self.assertIn(haut.numero, bas.dessus)
        self.assertIsNone(bas.support)
        self.assertEqual(bas.hauteur_base, 0.0)
        self.assertAlmostEqual(haut.hauteur_base, verite.pieces[i_haut].z)
        pv = verite.pieces[i_haut]
        self.assertEqual((haut.type, haut.face), (pv.type, pv.face))
        # la face superieure de celle du dessus, a sa vraie hauteur
        self.assertAlmostEqual(haut.position_table[2], pv.sommet, places=6)
        ordre = vm.ordre_de_prise(res.pieces)
        self.assertLess(ordre.index(haut), ordre.index(bas))
        self.assertEqual(bilan.ordre_ok, 1)
        return bas, haut, bilan

    def test_en_travers(self):
        """Une epaisse a plat en travers sur une autre : deux pieces, la
        bonne porteuse, celle du dessus prise d'abord."""
        pcs = [piece("epaisse", "a plat", 10, 20, 20)]
        pcs.append(sy.poser_sur(pcs, 0, "epaisse", "a plat", sy.TYPES, lacet=110))
        res, verite, _ = self.detecter(pcs)
        bas, haut, bilan = self.verifier_pile(res, verite, 0, 1)
        self.assertLess(max(bilan.err_pos), 2.0)
        # la porteuse est en partie cachee : elle est moins sure
        self.assertGreater(bas.cachee, 0.15)
        self.assertLess(bas.score, haut.score)

    def test_piece_du_dessus_bien_classee(self):
        """Une fine a plat sur une epaisse sur la tranche : sa face
        superieure est a 27 mm, ce qui n'est la hauteur d'aucune piece sur la
        table ; c'est l'epaisseur (27 - 19 = 8 mm) qui dit fine a plat."""
        pcs = [piece("epaisse", "sur la tranche", -20, 10, -30),
               piece("fine", "a plat", 60, -40, 70)]
        pcs.append(sy.poser_sur(pcs, 0, "fine", "a plat", sy.TYPES, dx=3, dy=-4, lacet=55))
        res, verite, _ = self.detecter(pcs, graine=2)
        _, haut, _ = self.verifier_pile(res, verite, 0, 2)
        self.assertAlmostEqual(haut.dimensions[2], 8.0, delta=2.5)

    def test_meme_sens_decalee(self):
        """Dans le meme sens, glissee le long de la porteuse : le bout de la
        porteuse depasse."""
        pcs = [piece("epaisse", "a plat", 0, 0, -15)]
        a = np.radians(-15)
        pcs.append(sy.poser_sur(pcs, 0, "epaisse", "a plat", sy.TYPES,
                                dx=22 * np.cos(a), dy=22 * np.sin(a), lacet=-12))
        res, verite, _ = self.detecter(pcs, graine=3)
        self.verifier_pile(res, verite, 0, 1)

    def test_sans_pile_inchange(self):
        """Sans pile, la recherche d'empilements ne change rien."""
        pcs = [piece("epaisse", "a plat", -60, 30, 10),
               piece("fine", "a plat", 40, 40, -60),
               piece("epaisse", "sur la tranche", 10, -50, 75),
               piece("fine", "sur la tranche", -40, -30, 30)]
        sc, verite = sy.scene(pcs, self.cam, graine=1)
        avec = vm.detecter(sc, sy.TYPES)
        sans = vm.detecter(sc, sy.TYPES, empilement=False)
        self.assertEqual(len(avec.pieces), 4)
        self.assertEqual([(p.type, p.face) for p in avec.pieces],
                         [(p.type, p.face) for p in sans.pieces])
        for p, q in zip(avec.pieces, sans.pieces):
            np.testing.assert_allclose(p.T_table, q.T_table, atol=1e-9)
            self.assertIsNone(p.support)
            self.assertEqual(p.dessus, [])
            self.assertEqual(p.hauteur_base, 0.0)


@besoin_cv2
class TestPerceptionDesPiles(unittest.TestCase):
    """L'etage masques de perception.py, le reseau remplace par les masques
    exacts de la partie visible de chaque piece."""

    def detecter(self, pcs, graine=0):
        import evaluer_detection as ev
        import perception
        sc, verite = sy.scene(pcs, sy.Camera(distance=0.55, inclinaison=35), graine=graine)
        capture = {"profondeur": sc.profondeur.astype(np.float32), "K": sc.K,
                   "couleur": np.full(sc.profondeur.shape + (3,), 128, np.uint8)}
        p = perception.Perception(types=sy.TYPES, modele=False)
        p.reseau, p.imgsz = ev.OracleMasques(verite, sy.TYPES), 864
        return p.detecter(capture), verite, perception

    def test_en_travers(self):
        pcs = [piece("epaisse", "a plat", 10, 20, 20)]
        pcs.append(sy.poser_sur(pcs, 0, "fine", "a plat", sy.TYPES, lacet=105))
        pieces, verite, perception = self.detecter(pcs)
        self.assertEqual(len(pieces), 2)
        bilan = sy.comparer(pieces, verite)
        self.assertEqual((bilan.trouves, bilan.faux, bilan.bon_support), (2, 0, 2))
        haut = max(pieces, key=lambda p: p.position_table[2])
        bas = min(pieces, key=lambda p: p.position_table[2])
        self.assertEqual((haut.type, haut.face), ("fine", "a plat"))
        self.assertEqual(haut.hauteur_base, 17.0)
        self.assertEqual(haut.support, bas.numero)
        self.assertEqual(bas.dessus, [haut.numero])
        self.assertIs(perception.ordre_de_prise(pieces)[0], haut)

    def test_sans_pile(self):
        pcs = [piece("epaisse", "a plat", -40, 10, 10), piece("fine", "sur la tranche", 50, -20, 80)]
        pieces, verite, _ = self.detecter(pcs, graine=1)
        self.assertEqual(len(pieces), 2)
        for p in pieces:
            self.assertIsNone(p.support)
            self.assertEqual(p.hauteur_base, 0.0)
            self.assertEqual(p.dessus, [])


if __name__ == "__main__":
    unittest.main()
