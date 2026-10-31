# -*- coding: utf-8 -*-
"""
Le tapis et le chariot comme FONCTIONS DU TEMPS.

Dans ce simulateur, tout ce qui s'anime se recalcule a partir de
l'instant : la configuration articulaire, les sorties, l'etat de
l'outil. La bande et le rail devaient rejoindre cette regle, et pas
s'accumuler image par image -- sinon rembobiner ne ramenerait pas a ce
qu'on avait vu, et une animation qui derive au retour arriere ne montre
plus le programme qu'elle est censee montrer.

C'est donc cela qu'on verifie ici : qu'on peut interroger n'importe quel
instant, dans n'importe quel ordre, et toujours obtenir la meme chose.
"""

import os
import sys
import unittest

import numpy as np

# le paquet vit dans cellule/, voisin de atelier/ -- voir
# atelier/cellule_sur_le_chemin.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule"))

from dobot_sim import Convoyeur, DobotMagician, Pilote, cible


def _robot():
    return DobotMagician()


class TestCadenceDuTapis(unittest.TestCase):

    def _programme(self):
        r = Pilote(nom="essai", outil="ventouse", vitesse=120)
        r.tapis = Convoyeur(longueur=1000)
        r.convoyeur(60)
        r.movl(cible(220, 0, 60))
        r.convoyeur(0)
        r.movl(cible(220, 0, 90))
        r.programme.construire(_robot(), r.q_depart)
        return r.programme

    def test_un_programme_sans_tapis_ne_dit_rien(self):
        r = Pilote(nom="essai", vitesse=120)
        r.movl(cible(220, 0, 60))
        r.programme.construire(_robot(), r.q_depart)
        self.assertEqual(r.programme.cadence_convoyeur(), [(0.0, 0.0)],
                         "sans instruction TAPIS, la cadence reste nulle")

    def test_les_changements_sont_dates(self):
        cadence = self._programme().cadence_convoyeur()
        self.assertEqual(len(cadence), 3, "depart implicite, marche, arret")
        self.assertEqual(cadence[0], (0.0, 0.0))
        self.assertAlmostEqual(cadence[1][1], 60.0)
        self.assertAlmostEqual(cadence[2][1], 0.0)
        self.assertLess(cadence[1][0], cadence[2][0], "l'arret vient apres")

    def test_la_cadence_se_donne_telle_quelle_au_tapis(self):
        """C'est tout l'interet de la forme : aucune conversion."""
        programme = self._programme()
        tapis = Convoyeur(longueur=2000)
        tapis.cadence = programme.cadence_convoyeur()
        t_arret = programme.cadence_convoyeur()[2][0]
        parcouru = tapis.deplacement(t_arret)
        self.assertGreater(parcouru, 0.0)
        # apres l'arret, la bande ne bouge plus
        self.assertAlmostEqual(tapis.deplacement(t_arret + 5.0), parcouru,
                               places=6)

    def test_rembobiner_redonne_la_meme_image(self):
        tapis = Convoyeur(longueur=2000)
        tapis.cadence = self._programme().cadence_convoyeur()
        tapis.poser(0.0, t0=0.0)
        aller = [tapis.abscisse(tapis.pieces[0], t)
                 for t in np.linspace(0, 4, 40)]
        retour = [tapis.abscisse(tapis.pieces[0], t)
                  for t in np.linspace(4, 0, 40)]
        np.testing.assert_allclose(aller, retour[::-1], atol=1e-9)


class TestPositionDuChariot(unittest.TestCase):

    def _programme(self):
        r = Pilote(nom="essai", outil="ventouse", vitesse=120)
        r.movl(cible(220, 0, 60))
        r.movj(cible(220, 0, 60), rail=300)
        r.movl(cible(200, 0, 60), rail=300)
        r.programme.construire(_robot(), r.q_depart)
        return r.programme

    def test_sans_rail_le_chariot_ne_bouge_pas(self):
        r = Pilote(nom="essai", vitesse=120)
        r.movl(cible(220, 0, 60))
        r.programme.construire(_robot(), r.q_depart)
        self.assertEqual(r.programme.courses_rail, [])
        self.assertAlmostEqual(r.programme.position_rail(1.0), 0.0)

    def test_le_chariot_part_de_l_origine_et_arrive_a_la_consigne(self):
        p = self._programme()
        self.assertAlmostEqual(p.position_rail(0.0), 0.0)
        self.assertAlmostEqual(p.position_rail(1e6), 300.0)

    def test_le_chariot_est_interpole_pendant_le_mouvement(self):
        """
        Le controleur traite la cinquieme coordonnee comme les quatre
        autres : le chariot avance AVEC le bras. Le faire sauter a
        l'arrivee montrerait un deplacement qui n'existe pas.
        """
        p = self._programme()
        t0, duree, vise = p.courses_rail[0]
        self.assertAlmostEqual(vise, 300.0)
        milieu = p.position_rail(t0 + duree / 2)
        self.assertGreater(milieu, 1.0)
        self.assertLess(milieu, 299.0)
        self.assertAlmostEqual(milieu, 150.0, delta=1.0)

    def test_la_position_est_monotone_et_rejouable(self):
        p = self._programme()
        instants = np.linspace(0, p.courses_rail[-1][0] + 5, 60)
        valeurs = [p.position_rail(t) for t in instants]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(valeurs, valeurs[1:])))
        encore = [p.position_rail(t) for t in instants[::-1]][::-1]
        np.testing.assert_allclose(valeurs, encore, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
