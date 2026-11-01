# -*- coding: utf-8 -*-
"""
Les pieces que le robot prend et depose, et leur animation.

Ce qui est verifie ici n'est pas un dessin mais un CALCUL : ou se
trouve chaque piece a chaque instant. La regle du simulateur est que
tout soit fonction du temps, sans rien accumuler -- c'est ce qui permet
de rembobiner. On verifie donc, avant tout, qu'une lecture a l'envers
redonne exactement les memes poses.
"""

import os
import sys
import unittest

import numpy as np

# le paquet vit dans cellule/, voisin de atelier/ -- voir
# atelier/cellule_sur_le_chemin.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule"))

from dobot_sim import Pilote, piece_posee                      # noqa: E402
from dobot_sim import homogeneous as hg                        # noqa: E402
from dobot_sim.pieces import Manutention                       # noqa: E402


def cycle(magasin=True, depose=(150.0, 150.0, 10.0)):
    """Un programme minimal : prendre au magasin, poser ailleurs."""
    robot = Pilote(nom="essai", outil="ventouse")
    robot.poser_piece(x=60.0, y=-220.0, dims=(125.0, 30.0, 10.0),
                      magasin=magasin, nom="planche")
    prise = hg.composer(hg.translation(60.0, -220.0, 10.0), hg.rot_x(np.pi))
    pose = hg.composer(hg.translation(*depose), hg.rot_x(np.pi))
    haut = hg.translation(0, 0, -70)

    robot.repli()
    robot.movj(prise @ haut)
    robot.movl(prise)
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(prise @ haut)
    robot.movj(pose @ haut)
    robot.movl(pose)
    robot.ventouse(False)
    robot.attendre(0.3)
    robot.movl(pose @ haut)
    robot.repli()

    traj = robot.programme.construire(robot.robot, robot.q_depart)
    manu = Manutention(robot.programme.pieces).preparer(
        robot.robot, traj, robot.programme)
    return robot, traj, manu


class TestManutention(unittest.TestCase):

    def test_la_piece_part_du_magasin(self):
        _, _, manu = cycle()
        boites = manu.boites(0.0)
        self.assertEqual(len(boites), 1)
        c = boites[0]["centre"]
        self.assertAlmostEqual(c[0], 60.0, places=6)
        self.assertAlmostEqual(c[1], -220.0, places=6)
        # posee SUR le plan : son centre est une demi-epaisseur plus haut
        self.assertAlmostEqual(c[2], 5.0, places=6)

    def test_le_magasin_se_recharge(self):
        """Des que la piece est emportee, une autre apparait a sa place."""
        _, traj, manu = cycle(magasin=True)
        fin = manu.boites(traj.duree)
        self.assertEqual(len(fin), 2, "il faut la deposee ET la neuve")
        places = sorted(round(b["centre"][1]) for b in fin)
        self.assertEqual(places, [-220, 150])

    def test_sans_magasin_elle_ne_revient_pas(self):
        _, traj, manu = cycle(magasin=False)
        self.assertEqual(len(manu.boites(traj.duree)), 1)

    def test_elle_arrive_ou_le_programme_la_depose(self):
        _, traj, manu = cycle(depose=(150.0, 150.0, 10.0))
        deposee = [b for b in manu.boites(traj.duree)
                   if b["centre"][1] > 0][0]
        self.assertAlmostEqual(deposee["centre"][0], 150.0, places=3)
        self.assertAlmostEqual(deposee["centre"][1], 150.0, places=3)
        # la ventouse visait le DESSUS : le centre est dessous d'une
        # demi-epaisseur, et la piece a garde son epaisseur
        self.assertAlmostEqual(deposee["centre"][2], 5.0, places=3)

    def test_elle_suit_la_ventouse_quand_elle_est_portee(self):
        robot, traj, manu = cycle()
        inst = [i for i in manu._instances if i["etapes"][0][0] == 0.0][0]
        portee = [e for e in inst["etapes"] if e[1] == "portee"]
        self.assertTrue(portee, "la piece n'a jamais ete prise")
        t0 = portee[0][0]
        t1 = [e[0] for e in inst["etapes"] if e[0] > t0][0]
        # a mi-parcours du transport, la piece est sous la ventouse
        t = (t0 + t1) / 2.0
        P = manu.pose(inst, t)
        outil = robot.robot.mgd(traj.q(t))
        ecart = float(np.linalg.norm(P[:3, 3] - outil[:3, 3]))
        self.assertLess(ecart, 10.0, "la piece ne suit pas l'outil")

    def test_rembobiner_redonne_la_meme_chose(self):
        """
        LA PROPRIETE QUI COMPTE. Tout est fonction du temps : lire le
        programme a l'endroit puis a l'envers doit donner, instant par
        instant, exactement les memes poses. Une simulation pas a pas
        echouerait ici.
        """
        _, traj, manu = cycle()
        instants = [traj.duree * k / 40.0 for k in range(41)]
        aller = [manu.boites(t) for t in instants]
        retour = [manu.boites(t) for t in reversed(instants)]
        for a, r in zip(aller, reversed(retour)):
            self.assertEqual(len(a), len(r))
            for ba, br in zip(a, r):
                self.assertTrue(np.allclose(ba["centre"], br["centre"]))
                self.assertTrue(np.allclose(ba["axes"], br["axes"]))

    def test_sans_piece_declaree_il_ne_se_passe_rien(self):
        robot = Pilote(nom="vide", outil="ventouse")
        robot.repli()
        traj = robot.programme.construire(robot.robot, robot.q_depart)
        manu = Manutention([]).preparer(robot.robot, traj, robot.programme)
        self.assertEqual(manu.boites(0.0), [])

    def test_les_pieces_voyagent_avec_le_programme(self):
        from dobot_sim.program import Programme

        robot, _, _ = cycle()
        relu = Programme.depuis_dict(robot.programme.vers_dict())
        self.assertEqual(len(relu.pieces), 1)
        self.assertTrue(relu.pieces[0].magasin)
        self.assertTrue(np.allclose(relu.pieces[0].pose,
                                    robot.programme.pieces[0].pose))

    def test_une_piece_hors_de_portee_n_est_pas_prise(self):
        """La ventouse n'emporte que ce qui est SOUS elle."""
        robot = Pilote(nom="loin", outil="ventouse")
        robot.poser_piece(x=200.0, y=200.0)          # a l'autre bout
        prise = hg.composer(hg.translation(60.0, -220.0, 10.0),
                            hg.rot_x(np.pi))
        robot.repli()
        robot.movl(prise)
        robot.ventouse(True)
        robot.attendre(0.3)
        robot.movl(prise @ hg.translation(0, 0, -70))
        traj = robot.programme.construire(robot.robot, robot.q_depart)
        manu = Manutention(robot.programme.pieces).preparer(
            robot.robot, traj, robot.programme)
        debut = manu.boites(0.0)[0]["centre"]
        fin = manu.boites(traj.duree)[0]["centre"]
        self.assertTrue(np.allclose(debut, fin), "elle a bouge toute seule")


class TestPieceSurLeTapis(unittest.TestCase):
    """
    Le chainon : une piece POSEE SUR LA BANDE avance avec elle, puis
    passe a la ventouse.

    Sans lui, l'etudiant voit une piece apparaitre au capteur -- et non
    le capteur arreter quelque chose qui bougeait.
    """

    def cellule(self):
        from dobot_sim import Convoyeur, repere_travail

        robot = Pilote(nom="tapis", outil="ventouse")
        tapis = Convoyeur(repere=repere_travail(x=210, y=-400, z=60, rz=90),
                          longueur=698, largeur=100, capteur_s=300)
        robot.tapis = tapis
        robot.poser_piece_sur_tapis(s=0.0, dims=(25.0, 25.0, 25.0),
                                    magasin=True)

        # la cible de prise : au droit du capteur, sur le dessus du cube
        prise = tapis.cible(300.0, 0.0, 25.0)
        haut = hg.translation(0, 0, -70)
        depose = hg.composer(hg.translation(15.0, 219.0, 92.0),
                             hg.rot_x(np.pi))

        robot.repli()
        robot.convoyeur(60.0)                     # le tapis part
        robot.attendre(300.0 / 60.0)              # la piece arrive
        robot.convoyeur(0.0)                      # il s'arrete
        robot.movj(prise @ haut)
        robot.movl(prise)
        robot.ventouse(True)
        robot.attendre(0.3)
        robot.movl(prise @ haut)
        robot.movj(depose @ haut)
        robot.movl(depose)
        robot.ventouse(False)
        robot.attendre(0.3)
        robot.repli()

        traj = robot.programme.construire(robot.robot, robot.q_depart)
        tapis.cadence = robot.programme.cadence_convoyeur()
        manu = Manutention(robot.programme.pieces).preparer(
            robot.robot, traj, robot.programme, tapis=tapis)
        return robot, tapis, traj, manu

    def test_elle_avance_avec_la_bande(self):
        _, tapis, traj, manu = self.cellule()
        debut = manu.boites(0.0)[0]["centre"]
        milieu = manu.boites(2.5)[0]["centre"]
        bouge = float(np.linalg.norm(np.array(milieu) - np.array(debut)))
        self.assertGreater(bouge, 100.0,
                           "la piece n'a pas suivi le tapis")

    def test_elle_s_arrete_quand_la_bande_s_arrete(self):
        """
        Apres l'arret du tapis, la piece ne bouge plus -- jusqu'a ce que
        la ventouse la prenne.

        L'INSTANT DE L'ARRET SE LIT DANS LA CADENCE, il ne se devine
        pas : le tapis ne part qu'apres le repli, et ma premiere version
        de ce test supposait t = 5 s. C'est le test qui etait faux, pas
        le code.
        """
        robot, _, traj, manu = self.cellule()
        cadence = robot.programme.cadence_convoyeur()
        arrets = [t for t, v in cadence if abs(v) < 1e-9 and t > 0.0]
        self.assertTrue(arrets, "le programme n'arrete jamais le tapis")
        t_arret = arrets[0]
        # entre l'arret et la prise, elle est immobile
        inst = manu._instances[0]
        t_prise = [e[0] for e in inst["etapes"] if e[1] == "portee"][0]
        self.assertLess(t_arret, t_prise)
        a = manu.pose(inst, t_arret + 0.05)
        b = manu.pose(inst, (t_arret + t_prise) / 2.0)
        self.assertTrue(np.allclose(a, b, atol=1e-6),
                        "elle avance encore apres l'arret de la bande")

    def test_puis_elle_passe_a_la_ventouse(self):
        robot, _, traj, manu = self.cellule()
        inst = manu._instances[0]
        modes = [e[1] for e in inst["etapes"]]
        self.assertEqual(modes[:3], ["tapis", "portee", "posee"],
                         "la piece n'a pas fait tapis -> outil -> table")

    def test_le_magasin_lache_une_neuve_a_l_entree(self):
        _, _, traj, manu = self.cellule()
        fin = manu.boites(traj.duree)
        self.assertEqual(len(fin), 2)
        # la neuve est repartie de l'entree et a avance depuis
        self.assertEqual(len(manu._instances), 2)
        self.assertEqual(manu._instances[1]["etapes"][0][1], "tapis")

    def test_sans_tapis_elle_reste_a_sa_pose(self):
        """Repli : un programme relu sans cellule declaree."""
        robot, _, traj, _ = self.cellule()
        sans = Manutention(robot.programme.pieces).preparer(
            robot.robot, traj, robot.programme, tapis=None)
        a = sans.boites(0.0)[0]["centre"]
        b = sans.boites(3.0)[0]["centre"]
        self.assertTrue(np.allclose(a, b), "elle a bouge sans tapis")


class TestLeTapisVoyage(unittest.TestCase):
    """
    Le convoyeur doit voyager AVEC le programme.

    Sans cela, un script qui declare un tapis ne le montre pas : la
    fenetre n'a de convoyeur que celui qu'on lui coche a la main, et une
    piece posee sur la bande reste a l'entree. Constate a l'ecran.
    """

    def tapis(self):
        from dobot_sim import Convoyeur, repere_travail
        return Convoyeur(repere=repere_travail(x=210, y=-400, z=60, rz=90),
                         longueur=698, largeur=100, capteur_s=300, sens=-1)

    def test_robot_tapis_ecrit_dans_le_programme(self):
        robot = Pilote(nom="t", outil="ventouse")
        t = self.tapis()
        robot.tapis = t
        self.assertIs(robot.programme.tapis, t)
        self.assertIs(robot.tapis, t)

    def test_il_survit_au_voyage(self):
        from dobot_sim.program import Programme

        robot = Pilote(nom="t", outil="ventouse")
        robot.tapis = self.tapis()
        relu = Programme.depuis_dict(robot.programme.vers_dict())
        self.assertIsNotNone(relu.tapis)
        self.assertTrue(np.allclose(relu.tapis.repere, robot.tapis.repere))
        self.assertEqual(relu.tapis.capteur_s, 300.0)
        # LE CABLAGE AUSSI : il ne se deduit d'aucune cote
        self.assertEqual(relu.tapis.sens, -1)

    def test_sans_tapis_le_programme_n_en_invente_pas(self):
        from dobot_sim.program import Programme

        robot = Pilote(nom="t", outil="ventouse")
        relu = Programme.depuis_dict(robot.programme.vers_dict())
        self.assertIsNone(relu.tapis)


class TestLesDeuxVues(unittest.TestCase):
    """
    Les deux vues doivent offrir LA MEME INTERFACE.

    L'interface demande `hasattr(self.vue, "definir_objets")` avant
    d'appeler : une vue qui ne la porte pas est ignoree SANS RIEN DIRE.
    Les pieces ont ainsi manque a la vue 2D pendant une heure -- sur un
    poste sans OpenGL, l'etudiant aurait vu le bras bouger et aucune
    planche, et aurait cru sa declaration fausse.

    Un `hasattr` silencieux se verifie donc ici, et non a l'ecran.
    """

    #: le plancher : ces methodes-la, quoi qu'il arrive
    METHODES = ("definir_objets", "definir_convoyeur", "definir_rail",
                "definir_chemin", "definir_cibles", "maj")

    #: les seules dispensees, et il faut dire pourquoi. Y ajouter un nom
    #: est une DECISION : sans cette liste, on l'aurait prise sans le
    #: savoir, en oubliant d'ecrire la methode.
    #:
    #: Ces deux-la sont des MANQUES CONNUS, pas des impossibilites :
    #: l'IHM les appelle derriere un `hasattr`, donc sur un poste sans
    #: OpenGL le champ de la camera et la piece tenue par la pince ne
    #: s'affichent pas, en silence. A ecrire le jour ou cela genera.
    SANS_OBJET_EN_2D = {
        "definir_camera": "pas encore ecrite pour la vue 2D",
        "definir_tenue": "pas encore ecrite pour la vue 2D",
    }

    def _attendues(self, V3) -> list:
        """
        Tout ce que la vue 3D expose en `afficher_*` et `definir_*`.

        Derivee, et non recopiee : une liste ecrite a la main ne protege
        que des oublis qu'on a deja faits. `afficher_trace` a ete ajoutee
        a la 3D sans l'etre a la 2D, et la liste d'alors ne l'a pas vue.
        """
        noms = {n for n in dir(V3)
                if n.startswith(("afficher_", "definir_"))
                and callable(getattr(V3, n))}
        return sorted((noms | set(self.METHODES)) - set(self.SANS_OBJET_EN_2D))

    def test_meme_interface_de_decor(self):
        import inspect

        from dobot_sim import viewer2d
        try:
            from dobot_sim import viewer3d
        except Exception:                      # pragma: no cover
            self.skipTest("vue 3D indisponible sur ce poste")

        V2, V3 = viewer2d.Vue2D, viewer3d.Vue3D
        for nom in self._attendues(V3):
            with self.subTest(methode=nom):
                self.assertTrue(hasattr(V2, nom), "manque a la vue 2D")
                self.assertTrue(hasattr(V3, nom), "manque a la vue 3D")
                a2 = inspect.signature(getattr(V2, nom)).parameters
                a3 = inspect.signature(getattr(V3, nom)).parameters
                self.assertEqual(list(a2)[:2], list(a3)[:2],
                                 "les signatures divergent : " + nom)


if __name__ == "__main__":
    unittest.main(verbosity=2)
