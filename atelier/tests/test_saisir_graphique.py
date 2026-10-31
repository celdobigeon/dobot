"""saisir_piece_graphique : detection continue, epoques, boites affichees."""

import os
import sys
import time
import unittest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "..", "cellule"))
sys.path.insert(0, os.path.join(ICI, "..", "..", "cellule", "examples", "perception"))
sys.path.insert(0, ICI)

import saisir_piece_graphique as g       # noqa: E402
from test_saisir_piece import Piece      # noqa: E402


class FaussePerception:
    types = {"epaisse": (17.0, 19.0, 68.0)}
    PROGRESSION = {"rapide": 0.15, "propositions": 0.6, "consolidee": 1.0}

    def detecter_par_etapes(self, capture=None):
        yield "rapide", self.detecter(capture)
        yield "propositions", None
        yield "consolidee", self.detecter(capture)

    def __init__(self):
        self.n = 0

    def detecter(self, capture=None):
        self.n += 1
        time.sleep(0.02)
        return [Piece(250.0, 0.0, 42.0, 0.0)]


class FauxEssai:
    def choisir(self, pieces):
        return pieces[0]


class TestDetecteur(unittest.TestCase):

    def test_cible_fraiche_apres_mouvement(self):
        affiche = []
        ancien = g.afficher
        g.afficher = lambda b, c=None: affiche.append((b, c))
        try:
            det = g.Detecteur(FaussePerception(), FauxEssai(),
                              capturer=lambda: {"profondeur": []})
            det.start()
            c = det.cible_fraiche(delai=2.0)
            self.assertIsNotNone(c)
            # un mouvement : la detection d'avant ne compte plus
            det.pause.set()
            det.epoque += 1
            self.assertIsNone(det.cible_fraiche(delai=0.2))
            det.pause.clear()
            self.assertIsNotNone(det.cible_fraiche(delai=2.0))
            det.fin.set()
            det.join(1.0)
        finally:
            g.afficher = ancien
        self.assertEqual(det.derniere[3], "consolidee")
        boites, cible = affiche[-1]
        self.assertEqual(cible, 0)
        self.assertAlmostEqual(boites[0]["centre"][2], 42.0 - 17.0 / 2.0)


class TestChangement(unittest.TestCase):

    def test_changement(self):
        a = [Piece(250.0, 0.0, 42.0, 0.0), Piece(200.0, 50.0, 42.0, 0.0)]
        meme = [Piece(205.0, 48.0, 42.0, 0.0), Piece(252.0, 1.0, 42.0, 0.0)]
        self.assertFalse(g.Detecteur.changement(a, meme))
        bouge = [Piece(250.0, 0.0, 42.0, 0.0), Piece(200.0, 90.0, 42.0, 0.0)]
        self.assertTrue(g.Detecteur.changement(a, bouge))
        self.assertTrue(g.Detecteur.changement(a, a[:1]))
        autre = [Piece(250.0, 0.0, 42.0, 0.0), Piece(200.0, 50.0, 42.0, 0.0)]
        autre[1].type = "fine"
        self.assertTrue(g.Detecteur.changement(a, autre))


if __name__ == "__main__":
    unittest.main()
