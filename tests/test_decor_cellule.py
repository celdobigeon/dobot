# -*- coding: utf-8 -*-
"""
La FORME du tapis et du rail, verifiee sans ouvrir de fenetre.

Le dessin d'un peripherique est une source d'erreurs silencieuses : une
boite au mauvais endroit ne leve aucune exception, et il faut l'oeil d'un
operateur devant un ecran pour s'en apercevoir. En demandant a l'objet de
DECRIRE sa forme -- `boites()` -- plutot qu'en la peignant directement,
on peut la verifier ici, et la vue n'a plus qu'a appliquer des couleurs.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import AxeLineaire, Convoyeur
from dobot_sim import homogeneous as hg
from dobot_sim.robot import repere_travail


def _roles(formes):
    return [f["role"] for f in formes]


def _un(formes, role):
    trouves = [f for f in formes if f["role"] == role]
    assert len(trouves) == 1, "{} boite(s) '{}'".format(len(trouves), role)
    return trouves[0]


class TestDecorConvoyeur(unittest.TestCase):

    def test_les_morceaux_attendus_sont_la(self):
        formes = Convoyeur().boites()
        for role in ("bande", "bati", "capteur", "fenetre", "champ"):
            self.assertIn(role, _roles(formes))

    def test_la_bande_a_les_cotes_du_tapis(self):
        t = Convoyeur(longueur=500, largeur=60)
        bande = _un(t.boites(), "bande")
        self.assertAlmostEqual(bande["dims"][0], 500.0)
        self.assertAlmostEqual(bande["dims"][1], 60.0)

    def test_la_bande_est_centree_sur_la_course(self):
        """L'origine du tapis est son ENTREE, pas son milieu."""
        t = Convoyeur(repere=hg.translation(100, 0, 0), longueur=400)
        bande = _un(t.boites(), "bande")
        self.assertAlmostEqual(bande["centre"][0], 100 + 200.0)

    def test_le_tapis_suit_son_repere(self):
        """Deplacer le tapis ne doit demander que de changer sa matrice."""
        t = Convoyeur(repere=repere_travail(x=230, y=250, rz=-90),
                      longueur=500)
        bande = _un(t.boites(), "bande")
        # rz=-90 couche l'axe x du tapis sur le -y du robot : le milieu
        # de la bande est donc 250 mm plus loin, vers les y negatifs
        np.testing.assert_allclose(bande["centre"][:2], [230.0, 0.0], atol=1e-6)

    def test_la_fenetre_du_capteur_a_sa_largeur(self):
        t = Convoyeur(capteur_s=300, capteur_fenetre=12)
        fenetre = _un(t.boites(), "fenetre")
        self.assertAlmostEqual(fenetre["dims"][0], 24.0,
                               msg="la fenetre vaut DEUX demi-largeurs")

    def test_sans_capteur_ni_camera_rien_n_est_dessine(self):
        formes = Convoyeur(capteur_s=None, camera_s=None).boites()
        self.assertNotIn("capteur", _roles(formes))
        self.assertNotIn("champ", _roles(formes))

    def test_les_pieces_defilent_avec_le_temps(self):
        t = Convoyeur(longueur=600).regler(50.0, 0.0)
        t.poser(0.0, t0=0.0)
        debut = _un(t.boites(0.0), "piece")["centre"]
        apres = _un(t.boites(2.0), "piece")["centre"]
        self.assertAlmostEqual(float(apres[0] - debut[0]), 100.0, places=3)

    def test_une_piece_sortie_n_est_plus_dessinee(self):
        t = Convoyeur(longueur=200).regler(100.0, 0.0)
        t.poser(0.0, t0=0.0)
        self.assertIn("piece", _roles(t.boites(1.0)))
        self.assertNotIn("piece", _roles(t.boites(5.0)),
                         "la piece est tombee du tapis depuis longtemps")

    def test_les_axes_sont_orthonormes(self):
        for forme in Convoyeur(repere=repere_travail(x=200, rz=37)).boites():
            R = np.asarray(forme["axes"])
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)


class TestDecorAxeLineaire(unittest.TestCase):

    def test_la_poutre_couvre_la_course(self):
        poutre = _un(AxeLineaire(course=1000).boites(), "poutre")
        self.assertAlmostEqual(poutre["dims"][0], 1000.0)

    def test_le_chariot_suit_la_consigne(self):
        axe = AxeLineaire(course=1000, direction=(0, 1, 0))
        a = _un(axe.boites(0), "chariot")["centre"]
        b = _un(axe.boites(300), "chariot")["centre"]
        np.testing.assert_allclose(b - a, [0, 300, 0], atol=1e-6)

    def test_le_chariot_reste_dans_la_course(self):
        axe = AxeLineaire(course=500)
        loin = _un(axe.boites(9000), "chariot")["centre"]
        bout = _un(axe.boites(500), "chariot")["centre"]
        np.testing.assert_allclose(loin, bout, atol=1e-9)

    def test_un_rail_vertical_garde_un_repere_valide(self):
        """
        Montage improbable, mais le produit vectoriel avec z s'annule et
        donnerait des axes degeneres : la vue afficherait n'importe quoi.
        """
        T = AxeLineaire(direction=(0, 0, 1)).reperage()
        R = T[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)

    def test_le_repere_pointe_bien_le_long_du_rail(self):
        for direction in ((0, 1, 0), (0, -1, 0), (1, 0, 0), (0.5, 0.5, 0.0)):
            T = AxeLineaire(direction=direction).reperage()
            attendu = np.asarray(direction, dtype=float)
            attendu /= np.linalg.norm(attendu)
            np.testing.assert_allclose(T[:3, 0], attendu, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
