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

# le paquet vit dans cellule/, voisin de atelier/ -- voir
# atelier/cellule_sur_le_chemin.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule"))

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

    def test_le_capteur_est_un_barillet_en_travers_de_la_bande(self):
        """
        Un cube a cote du tapis ne disait pas ce que le capteur fait.
        Un barillet couche en travers, si : il regarde l'axe de la bande.

        On verifie donc la FORME et l'ORIENTATION, pas seulement la
        presence -- c'etait precisement ce qui manquait.
        """
        t = Convoyeur(repere=repere_travail(x=210, y=-400, rz=90),
                      longueur=698, largeur=100, capteur_s=300)
        capteur = _un(t.boites(), "capteur")
        self.assertEqual(capteur.get("forme"), "cylindre")

        R = np.asarray(capteur["axes"], dtype=float)
        # l'axe du cylindre est le TROISIEME, par convention de la vue
        np.testing.assert_allclose(R[:, 2], np.asarray(t.repere)[:3, 1],
                                   atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9,
                               msg="un repere indirect retourne le maillage")
        self.assertAlmostEqual(capteur["dims"][0], capteur["dims"][1],
                               msg="un barillet a une section ronde")

    def test_le_capteur_se_pose_a_cote_de_la_bande_et_au_dessus(self):
        """Il est visse sur l'equerre, pas pose sur la bande."""
        t = Convoyeur(repere=hg.identite(), longueur=698, largeur=100,
                      capteur_s=300, capteur_y=46, capteur_z=16)
        c = _un(t.boites(), "capteur")["centre"]
        np.testing.assert_allclose(c, [300.0, 46.0, 16.0], atol=1e-9)
        self.assertGreater(c[1], 100 / 2 - 10,
                           "le barillet doit border la bande, pas la couvrir")

    def test_le_bras_du_capteur_se_detache_du_maillage(self):
        """
        La vue glisse le bras du capteur jusqu'a `capteur_s`. Encore
        faut-il savoir lequel des 13 734 triangles le compose.

        On verifie ici que la selection par le POINT LE PLUS HAUT
        isole bien un bras -- mince, haut, sur le bord -- et non un
        morceau de bati. Une selection trop large passerait inapercue
        a l'ecran : le tapis se couperait en deux au premier
        glissement.
        """
        import numpy as np
        from dobot_sim.meshes import charger_obj, composante_du_plus_haut

        obj = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "cellule", "Modele3D",
            "convoyeur.obj")
        if not os.path.isfile(obj):
            self.skipTest("le maillage du tapis n'est pas la")

        V, F = charger_obj(obj)
        haut = composante_du_plus_haut(V, F)
        self.assertTrue(haut.any(), "aucune face detachee")
        self.assertLess(haut.sum(), len(F) / 20.0,
                        "la selection a emporte la moitie du tapis")

        S = V[np.unique(F[haut])]
        self.assertLess(S[:, 1].max() - S[:, 1].min(), 10.0,
                        "un bras de capteur est une PLAQUE, pas un bloc")
        self.assertGreater(S[:, 2].max(), 25.0,
                           "il doit depasser franchement la bande")
        self.assertAlmostEqual(float(S[:, 0].min() + S[:, 0].max()) / 2.0,
                               353.15, places=1,
                               msg="l'abscisse ou le constructeur l'a dessine")

    def test_les_deux_postes_se_detachent_du_maillage(self):
        """
        La vue glisse chaque poste jusqu'a sa cote declaree. Encore
        faut-il savoir lequel des 13 734 triangles compose quoi.

        Le bras du capteur est une plaque qui monte a 31 mm ; le socle
        de la camera, un assemblage de quatorze corps -- embase,
        plots, entretoises, plaques -- qui monte a 16. Le carter du
        moteur, lui, ne depasse pas 6 et doit rester avec le tapis :
        le faire voyager couperait la machine en deux.
        """
        import numpy as np
        from dobot_sim.meshes import charger_obj, grappes_laterales

        obj = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "cellule", "Modele3D",
            "convoyeur.obj")
        if not os.path.isfile(obj):
            self.skipTest("le maillage du tapis n'est pas la")

        V, F = charger_obj(obj)
        grappes = grappes_laterales(V, F)
        self.assertEqual(len(grappes), 2,
                         "deux postes, pas un de plus : le bati doit rester")

        (m_capteur, x_capteur), (m_camera, x_camera) = grappes
        self.assertAlmostEqual(x_capteur, 353.15, places=1)
        self.assertAlmostEqual(x_camera, 278.1, places=1)
        self.assertGreater(x_capteur, x_camera,
                           "la camera regarde la piece AVANT le capteur")

        emporte = int(m_capteur.sum() + m_camera.sum())
        self.assertLess(emporte, len(F) / 8.0,
                        "la selection a emporte un morceau du tapis")
        self.assertFalse((m_capteur & m_camera).any(),
                         "une face ne peut pas appartenir aux deux postes")

        # le socle doit partir ENTIER : ses pieds descendent sous la bande
        socle = V[np.unique(F[m_camera])]
        self.assertLess(socle[:, 2].min(), 0.0,
                        "le socle a ete coupe a hauteur de bande")

    def test_la_camera_ne_regarde_pas_le_capteur(self):
        """
        Deux postes, deux endroits. La camera lit la couleur d'une
        piece qui defile, le capteur arrete la bande ; leur mettre la
        meme abscisse ne veut rien dire, et cela s'est vu a l'ecran --
        le champ de la camera couche sur le capteur.

        Le defaut de `camera_s` valait 300, comme le `capteur_s` du
        corrige. Un defaut qui coincide avec une valeur courante est un
        piege : il ne se trahit que par le dessin.
        """
        t = Convoyeur(capteur_s=300)
        self.assertIsNotNone(t.camera_s)
        ecart = abs(float(t.camera_s) - float(t.capteur_s))
        self.assertGreater(ecart, t.camera_champ / 2 + t.capteur_fenetre,
                           "les deux postes se chevauchent")
        self.assertLess(float(t.camera_s), float(t.capteur_s),
                        "la camera regarde la piece AVANT qu'on l'arrete")

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


class TestProjection2D(unittest.TestCase):
    """
    L'empreinte au sol d'une boite, pour la vue de secours.

    Quand OpenGL manque, le simulateur retombe sur deux projections. Le
    tapis et le rail doivent s'y voir aussi : un poste ou les commandes
    marchent sans que rien ne s'affiche est pire qu'un poste qui refuse
    de demarrer, parce qu'il ne previent pas.
    """

    def setUp(self):
        from dobot_sim.viewer2d import contour_xy, sommets_boite
        self.contour_xy = contour_xy
        self.sommets_boite = sommets_boite

    def _boite(self, centre=(100.0, 0.0, 0.0), dims=(200.0, 60.0, 10.0),
               rz=0.0):
        from dobot_sim.robot import repere_travail
        R = repere_travail(x=centre[0], y=centre[1], z=centre[2], rz=rz)
        return {"centre": R[:3, 3], "axes": R[:3, :3], "dims": dims,
                "role": "bande"}

    def test_une_boite_a_huit_coins(self):
        self.assertEqual(self.sommets_boite(self._boite()).shape, (8, 3))

    def test_le_contour_est_ferme(self):
        C = self.contour_xy(self.sommets_boite(self._boite()))
        np.testing.assert_allclose(C[0], C[-1])

    def test_une_boite_droite_donne_son_rectangle(self):
        C = self.contour_xy(self.sommets_boite(self._boite()))
        self.assertAlmostEqual(float(C[:, 0].min()), 0.0, places=6)
        self.assertAlmostEqual(float(C[:, 0].max()), 200.0, places=6)
        self.assertAlmostEqual(float(C[:, 1].min()), -30.0, places=6)
        self.assertAlmostEqual(float(C[:, 1].max()), 30.0, places=6)

    def test_une_boite_tournee_reste_un_rectangle(self):
        """Un quart de tour echange longueur et largeur, rien de plus."""
        C = self.contour_xy(self.sommets_boite(self._boite(rz=90.0)))
        self.assertEqual(len(C), 5, "quatre coins et la fermeture")
        self.assertAlmostEqual(float(C[:, 0].max() - C[:, 0].min()), 60.0,
                               places=6)
        self.assertAlmostEqual(float(C[:, 1].max() - C[:, 1].min()), 200.0,
                               places=6)

    def test_une_boite_de_biais_a_quatre_coins_distincts(self):
        C = self.contour_xy(self.sommets_boite(self._boite(rz=30.0)))
        self.assertEqual(len(C), 5)
        aire = 0.5 * abs(sum(C[i, 0] * C[i + 1, 1] - C[i + 1, 0] * C[i, 1]
                             for i in range(len(C) - 1)))
        self.assertAlmostEqual(aire, 200.0 * 60.0, delta=1.0,
                               msg="une rotation ne change pas l'aire")

    def test_le_contour_ne_depend_pas_de_l_ordre_des_points(self):
        S = self.sommets_boite(self._boite(rz=30.0))
        a = self.contour_xy(S)
        b = self.contour_xy(S[::-1])
        self.assertEqual(len(a), len(b))

    def test_un_tapis_entier_se_projette(self):
        """Le cas reel : toutes les boites du modele, d'un coup."""
        t = Convoyeur(repere=repere_travail(x=230, y=250, rz=-90),
                      longueur=698)
        t.regler(50.0, 0.0)
        t.poser(0.0, t0=0.0)
        for forme in t.boites(1.0):
            C = self.contour_xy(self.sommets_boite(forme))
            self.assertGreaterEqual(len(C), 2, forme["role"])
            self.assertTrue(np.isfinite(C).all(), forme["role"])


class TestDescripteurPasserelle(unittest.TestCase):
    """
    Ce qu'un tapis dit de lui-meme pour traverser la passerelle.

    Un script et le simulateur sont deux processus : l'objet `Convoyeur`
    ne passe pas de l'un a l'autre. Seuls quelques nombres voyagent, et
    il faut qu'ils suffisent -- un pas oublie, et le tapis tourne a une
    vitesse qui n'est pas celle demandee, sans que rien ne proteste.
    """

    def setUp(self):
        from dobot_sim.pilote import _decrire_tapis
        self.decrire = _decrire_tapis

    def test_tout_le_necessaire_voyage(self):
        from dobot_sim.convoyeur import PORTS_GP
        t = Convoyeur(index_moteur=1, impulsions_par_mm=300.0,
                      capteur_port=PORTS_GP["GP4"])
        d = self.decrire(t)
        self.assertEqual(d["index"], 1)
        self.assertAlmostEqual(d["pas"], 300.0)
        self.assertEqual(d["capteur_port"], PORTS_GP["GP4"])
        self.assertIsNone(d["capteur_eio"])

    def test_le_pas_nominal_sert_de_secours(self):
        """Un pas nul enverrait une vitesse nulle, donc un tapis muet."""
        from dobot_sim.convoyeur import IMPULSIONS_PAR_MM
        d = self.decrire(Convoyeur(impulsions_par_mm=0.0))
        self.assertAlmostEqual(d["pas"], IMPULSIONS_PAR_MM)

    def test_un_capteur_sur_entree_tout_ou_rien_voyage_aussi(self):
        d = self.decrire(Convoyeur(capteur_port=None, capteur_eio=5))
        self.assertEqual(d["capteur_eio"], 5)
        self.assertIsNone(d["capteur_port"])

    def test_le_descripteur_est_serialisable(self):
        """Il traverse la passerelle en JSON : rien d'exotique dedans."""
        import json
        d = self.decrire(Convoyeur())
        self.assertEqual(json.loads(json.dumps(d)), d)

    def test_un_objet_quelconque_est_accepte(self):
        """
        Le simulateur reconstruit un descripteur, pas un Convoyeur : le
        pont ne doit exiger que des attributs, pas une classe.
        """
        import types
        faux = types.SimpleNamespace(index_moteur=0, impulsions_par_mm=282.9,
                                     capteur_port=2, capteur_eio=None)
        self.assertEqual(self.decrire(faux)["capteur_port"], 2)
