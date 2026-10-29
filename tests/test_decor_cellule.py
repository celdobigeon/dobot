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


class TestRecalageCAO(unittest.TestCase):
    """
    Le recalage d'un modele CAO sur le repere du Convoyeur.

    C'est l'etape qui se trompe sans rien dire : un maillage mal recale
    s'affiche parfaitement, simplement pas au bon endroit -- et c'est en
    envoyant le bras dans le vide qu'on s'en apercoit. Les conventions
    sont donc verifiees ici, une par une.
    """

    def _boite(self, long_x=700.0, large_y=220.0, haut_z=85.0,
               decalage=(1000.0, -50.0, 300.0)):
        """Les huit coins d'un pave, places n'importe ou dans l'espace."""
        coins = np.array([[x, y, z] for x in (0, long_x)
                          for y in (0, large_y) for z in (0, haut_z)],
                         dtype=float)
        return coins + np.asarray(decalage, dtype=float)

    def _recaler(self, sommets, faces=None, inverser=False):
        import importlib.util
        chemin = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "convertir_cao.py")
        spec = importlib.util.spec_from_file_location("convertir_cao", chemin)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.recaler_tapis(sommets, faces, inverser)

    def test_l_origine_tombe_a_l_entree_de_la_bande(self):
        r = self._recaler(self._boite())
        self.assertAlmostEqual(float(r[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(r[:, 0].max()), 700.0, places=6)

    def test_le_zero_est_a_la_SURFACE_de_la_bande(self):
        """Une piece posee dessus est a z=0, pas a z=+85."""
        r = self._recaler(self._boite())
        self.assertAlmostEqual(float(r[:, 2].max()), 0.0, places=6)
        self.assertAlmostEqual(float(r[:, 2].min()), -85.0, places=6)

    def test_la_bande_est_centree_en_largeur(self):
        r = self._recaler(self._boite())
        self.assertAlmostEqual(float(r[:, 1].min()), -110.0, places=6)
        self.assertAlmostEqual(float(r[:, 1].max()), 110.0, places=6)

    def test_le_sens_de_marche_suit_la_plus_grande_dimension(self):
        """Peu importe l'axe que le dessinateur a choisi."""
        long_selon_y = self._boite(long_x=220.0, large_y=700.0)
        r = self._recaler(long_selon_y)
        self.assertAlmostEqual(float(r[:, 0].max()), 700.0, places=6)
        self.assertAlmostEqual(float(r[:, 1].max()), 110.0, places=6)

    def test_inverser_prend_l_autre_bout_comme_entree(self):
        droit = self._recaler(self._boite())
        inverse = self._recaler(self._boite(), inverser=True)
        self.assertAlmostEqual(float(inverse[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(inverse[:, 0].max()), 700.0, places=6)
        # le meme coin ne se retrouve pas au meme bout
        self.assertNotAlmostEqual(float(droit[0, 0]), float(inverse[0, 0]),
                                  places=3)

    def test_le_recalage_ne_retourne_pas_le_modele(self):
        """
        Un repere indirect retournerait la piece comme un gant : elle
        s'afficherait tres bien, en miroir, et le capteur se retrouverait
        de l'autre cote de la bande sans que rien ne proteste.

        On le verifie sur le PRODUIT MIXTE de trois aretes : son signe ne
        doit pas changer, et sa valeur non plus -- une rotation conserve
        les volumes.
        """
        avant = self._boite()
        apres = self._recaler(avant)

        def mixte(p):
            a, b, c = p[1] - p[0], p[2] - p[0], p[4] - p[0]
            return float(np.dot(np.cross(a, b), c))

        self.assertGreater(mixte(avant) * mixte(apres), 0.0,
                           "le modele a ete mis en miroir")
        self.assertAlmostEqual(abs(mixte(apres)), abs(mixte(avant)),
                               delta=1e-6, msg="le volume a change")

    def test_le_recalage_conserve_les_distances(self):
        """Une rotation et une translation, rien d'autre : pas d'echelle."""
        avant = self._boite()
        apres = self._recaler(avant)
        for i, j in ((0, 1), (0, 2), (0, 4), (3, 5)):
            self.assertAlmostEqual(
                float(np.linalg.norm(apres[i] - apres[j])),
                float(np.linalg.norm(avant[i] - avant[j])), places=6)
