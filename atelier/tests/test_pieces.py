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


if __name__ == "__main__":
    unittest.main(verbosity=2)
