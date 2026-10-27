# -*- coding: utf-8 -*-
"""
Le profil d'une image, et le programme qui le trace.

Les images d'essai sont fabriquees ici, a la geometrie connue : un
disque de rayon 100 doit ressortir a rayon 100. On verifie donc des
nombres, pas une impression visuelle.

Sans OpenCV, tout le fichier est saute : le trace depuis une image est
une commodite, pas une fonction du simulateur.
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import (ErreurImage, Pilote, lire_image, programme_depuis_image,
                       repere_travail)

try:
    import cv2
except ImportError:                                  # pragma: no cover
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV absent : trace depuis une image ignore")
class TestLectureImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dossier = tempfile.mkdtemp()

        # un anneau : disque noir de rayon 100, trou blanc de rayon 30
        anneau = np.full((400, 400), 255, np.uint8)
        cv2.circle(anneau, (200, 200), 100, 0, -1)
        cv2.circle(anneau, (200, 200), 30, 255, -1)
        cls.anneau = cls._ecrire("anneau.png", anneau)

        # un carre de 200 de cote, coin superieur gauche en (100, 50)
        carre = np.full((400, 400), 255, np.uint8)
        cv2.rectangle(carre, (100, 50), (300, 250), 0, -1)
        cls.carre = cls._ecrire("carre.png", carre)

        # le meme carre en blanc sur noir
        cls.inverse = cls._ecrire("inverse.png", 255 - carre)

        # un disque, avec trois poussieres d'un pixel autour
        poussieres = np.full((300, 300), 255, np.uint8)
        cv2.circle(poussieres, (150, 150), 80, 0, -1)
        for xy in ((10, 10), (290, 20), (20, 280)):
            poussieres[xy[1], xy[0]] = 0
        cls.poussieres = cls._ecrire("poussieres.png", poussieres)

        cls.uniforme = cls._ecrire("uniforme.png",
                                   np.full((100, 100), 255, np.uint8))

    @classmethod
    def _ecrire(cls, nom, image):
        chemin = os.path.join(cls.dossier, nom)
        cv2.imwrite(chemin, image)
        return chemin

    # -- geometrie ----------------------------------------------------------

    def test_un_disque_rend_son_rayon(self):
        contour = lire_image(self.anneau, trous=False)[0]
        xy = np.array(contour)
        centre = xy.mean(axis=0)
        rayons = np.hypot(*(xy - centre).T)
        self.assertAlmostEqual(centre[0], 200.0, delta=1.0)
        self.assertAlmostEqual(centre[1], 200.0, delta=1.0)
        self.assertAlmostEqual(rayons.mean(), 100.0, delta=1.5)

    def test_le_trou_est_un_contour_a_part(self):
        avec = lire_image(self.anneau, trous=True)
        sans = lire_image(self.anneau, trous=False)
        self.assertEqual(len(avec), 2)
        self.assertEqual(len(sans), 1)
        rayons = []
        for contour in avec:
            xy = np.array(contour)
            rayons.append(np.hypot(*(xy - xy.mean(axis=0)).T).mean())
        self.assertAlmostEqual(min(rayons), 30.0, delta=1.5)
        self.assertAlmostEqual(max(rayons), 100.0, delta=1.5)

    def test_les_contours_sont_fermes(self):
        for contour in lire_image(self.anneau):
            self.assertEqual(contour[0], contour[-1])

    def test_un_carre_simplifie_garde_ses_quatre_coins(self):
        contour = lire_image(self.carre, simplification_px=3.0)[0]
        self.assertEqual(len(contour), 5, "4 sommets + la fermeture")
        xy = np.array(contour[:-1])
        self.assertAlmostEqual(xy[:, 0].min(), 100.0, delta=2.0)
        self.assertAlmostEqual(xy[:, 0].max(), 300.0, delta=2.0)
        self.assertAlmostEqual(xy[:, 1].min(), 50.0, delta=2.0)
        self.assertAlmostEqual(xy[:, 1].max(), 250.0, delta=2.0)

    def test_blanc_sur_noir_demande_inverser(self):
        """Sans `inverser`, c'est le FOND que l'on suit, donc le cadre."""
        droit = lire_image(self.inverse, inverser=True, simplification_px=3.0)
        self.assertEqual(len(droit[0]), 5)
        xy = np.array(droit[0][:-1])
        self.assertAlmostEqual(xy[:, 0].min(), 100.0, delta=2.0)
        self.assertAlmostEqual(xy[:, 0].max(), 300.0, delta=2.0)

    def test_les_poussieres_sont_ecartees(self):
        self.assertEqual(len(lire_image(self.poussieres)), 1)
        # aire_min a zero : les poussieres d'un pixel passent le filtre
        # d'aire, mais pas celui des trois points
        beaucoup = lire_image(self.poussieres, aire_min=0.0)
        self.assertEqual(len(beaucoup), 1)

    def test_un_seuil_explicite_est_respecte(self):
        gris = np.full((200, 200), 255, np.uint8)
        cv2.rectangle(gris, (50, 50), (150, 150), 200, -1)     # gris clair
        chemin = self._ecrire("gris.png", gris)
        # seuil au-dessus du gris : la forme est vue
        self.assertEqual(len(lire_image(chemin, seuil=220, flou=0)), 1)
        # seuil en dessous : plus rien n'est sombre
        with self.assertRaises(ErreurImage):
            lire_image(chemin, seuil=150, flou=0)

    # -- refus parlants -----------------------------------------------------

    def test_image_uniforme_refusee(self):
        with self.assertRaises(ErreurImage) as ctx:
            lire_image(self.uniforme)
        self.assertIn("aucun contour", str(ctx.exception))

    def test_fichier_absent_ou_illisible(self):
        with self.assertRaises(ErreurImage):
            lire_image(os.path.join(self.dossier, "pas_la.png"))
        faux = os.path.join(self.dossier, "faux.png")
        with open(faux, "w", encoding="utf-8") as f:
            f.write("ceci n'est pas une image")
        with self.assertRaises(ErreurImage) as ctx:
            lire_image(faux)
        self.assertIn("illisible", str(ctx.exception))


@unittest.skipIf(cv2 is None, "OpenCV absent : trace depuis une image ignore")
class TestProgrammeDepuisImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dossier = tempfile.mkdtemp()
        anneau = np.full((400, 400), 255, np.uint8)
        cv2.circle(anneau, (200, 200), 100, 0, -1)
        cv2.circle(anneau, (200, 200), 30, 255, -1)
        cls.chemin = os.path.join(cls.dossier, "anneau.png")
        cv2.imwrite(cls.chemin, anneau)
        cls.feuille = repere_travail(x=235, y=0, z=10)
        cls.robot = programme_depuis_image(cls.chemin, repere=cls.feuille,
                                           largeur=60)

    def _points_sur_le_papier(self, robot=None):
        e = (robot or self.robot).trajectoire().echantillonner(0.02)
        P = e["P"]
        return P[np.abs(P[:, 2] - 10.0) < 0.05]

    def test_un_pilote_pret_a_jouer(self):
        self.assertIsInstance(self.robot, Pilote)
        self.assertEqual(self.robot.controler(), [])

    def test_la_largeur_demandee_est_tenue(self):
        papier = self._points_sur_le_papier()
        self.assertAlmostEqual(np.ptp(papier[:, 0]), 60.0, delta=1.0)

    def test_le_dessin_est_centre_sur_le_repere(self):
        papier = self._points_sur_le_papier()
        cx = (papier[:, 0].min() + papier[:, 0].max()) / 2.0
        cy = (papier[:, 1].min() + papier[:, 1].max()) / 2.0
        self.assertAlmostEqual(cx, 235.0, delta=0.5)
        self.assertAlmostEqual(cy, 0.0, delta=0.5)

    def test_le_stylo_se_pose_a_l_arret_au_point(self):
        ins = self.robot.programme.instructions
        touche = [x for x in ins
                  if x.commentaire in ("pose du stylo", "fin du trait")]
        self.assertTrue(touche)
        for x in touche:
            self.assertEqual(x.lissage, 0.0)

    def test_la_tolerance_est_en_millimetres_du_dessin(self):
        """
        Le meme dessin, a deux definitions : la tolerance etant exprimee
        en mm du trace final, le nombre de points doit rester du meme
        ordre -- ce qui ne serait pas le cas d'une tolerance en pixels.
        """
        grand = np.full((1600, 1600), 255, np.uint8)
        cv2.circle(grand, (800, 800), 400, 0, -1)
        cv2.circle(grand, (800, 800), 120, 255, -1)
        chemin = os.path.join(self.dossier, "anneau_grand.png")
        cv2.imwrite(chemin, grand)

        petit = len(programme_depuis_image(
            self.chemin, repere=self.feuille, largeur=60,
            tolerance=0.5).programme.instructions)
        gros = len(programme_depuis_image(
            chemin, repere=self.feuille, largeur=60,
            tolerance=0.5).programme.instructions)
        self.assertLess(abs(gros - petit), max(8, 0.35 * petit),
                        "{} instructions contre {} : la tolerance ne suit "
                        "pas la taille du dessin".format(gros, petit))

    def test_une_tolerance_plus_large_allege_le_programme(self):
        fin = programme_depuis_image(self.chemin, repere=self.feuille,
                                     largeur=60, tolerance=0.1)
        gros = programme_depuis_image(self.chemin, repere=self.feuille,
                                      largeur=60, tolerance=2.0)
        self.assertLess(len(gros.programme.instructions),
                        len(fin.programme.instructions))


if __name__ == "__main__":
    unittest.main()
