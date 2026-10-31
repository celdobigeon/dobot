# -*- coding: utf-8 -*-
"""
L'axe lineaire : un septieme axe qui translate la base.

Ce qui compte et qui se verifie : une cible hors d'atteinte bras seul
doit le devenir une fois le chariot place, l'aller-retour
atelier <-> chariot doit etre exact, et la cinquieme coordonnee doit
arriver au controleur dans la bonne trame.
"""

import math
import os
import struct
import sys
import unittest

import numpy as np

# le paquet vit dans cellule/, voisin de atelier/ -- voir
# atelier/cellule_sur_le_chemin.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_sim import AxeLineaire, DobotMagician, Pilote, cible
from dobot_sim.liaison import CommandeDobot
from dobot_sim.reel import PontReel
from test_protocole import SerieSimulee, _liaison_simulee


def _robot():
    return DobotMagician().avec_outil(0.0)


class TestModele(unittest.TestCase):

    def setUp(self):
        self.rail = AxeLineaire(course=1000.0)
        self.robot = _robot()

    def test_la_direction_est_normalisee(self):
        axe = AxeLineaire(direction=(0.0, 3.0, 0.0))
        np.testing.assert_allclose(axe.direction, (0.0, 1.0, 0.0), atol=1e-12)
        with self.assertRaises(ValueError):
            AxeLineaire(direction=(0.0, 0.0, 0.0))

    def test_la_course_est_bornee(self):
        self.assertEqual(self.rail.borner(-50), 0.0)
        self.assertEqual(self.rail.borner(5000), 1000.0)
        self.assertTrue(self.rail.dans_la_course(0.0))
        self.assertFalse(self.rail.dans_la_course(1000.1))

    def test_aller_retour_atelier_chariot(self):
        """ramener() puis porter() doivent redonner la cible, exactement."""
        T = cible(220, 120, 40, rz=30)
        for l in (0.0, 250.0, 1000.0):
            retour = self.rail.porter(self.rail.ramener(T, l), l)
            self.assertLess(float(np.abs(retour - T).max()), 1e-12)

    def test_le_rail_translate_bien_la_base(self):
        T = self.rail.matrice(400.0)
        np.testing.assert_allclose(T[:3, 3], (0.0, 400.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12,
                                   err_msg="le rail translate, il ne tourne pas")

    def test_une_cible_hors_datteinte_le_devient_grace_au_chariot(self):
        T = cible(220, 400, 30)
        from dobot_sim import ErreurCinematique
        with self.assertRaises(ErreurCinematique):
            self.robot.mgi(T)
        l = self.rail.position_pour(self.robot, T)
        self.assertIsNotNone(l, "aucune position de chariot trouvee")
        q = self.robot.mgi(self.rail.ramener(T, l))
        self.assertEqual(self.robot.depassements(q), [])
        # et le TCP retombe bien sur la cible, dans l'atelier
        atelier = self.rail.porter(self.robot.mgd(q), l)
        self.assertLess(float(np.abs(atelier - T).max()), 1e-9)

    def test_la_position_retenue_est_la_plus_proche_du_depart(self):
        T = cible(220, 400, 30)
        loin = self.rail.position_pour(self.robot, T, depart=0.0)
        pres = self.rail.position_pour(self.robot, T, depart=300.0)
        self.assertIsNotNone(loin)
        self.assertIsNotNone(pres)
        self.assertLess(abs(pres - 300.0), abs(loin - 300.0))

    def test_une_cible_vraiment_hors_datteinte_rend_None(self):
        self.assertIsNone(self.rail.position_pour(self.robot,
                                                  cible(900, 0, 30), pas=20.0))

    def test_atteignable_dit_vrai_et_faux(self):
        T = cible(220, 400, 30)
        l = self.rail.position_pour(self.robot, T)
        self.assertTrue(self.rail.atteignable(self.robot, T, l))
        self.assertFalse(self.rail.atteignable(self.robot, T, 0.0))

    def test_duree_du_chariot(self):
        """Profil trapeze : court = triangulaire, long = plateau."""
        rail = AxeLineaire(vitesse=100.0, acceleration=200.0)
        self.assertAlmostEqual(rail.duree(0, 0), 0.0, places=12)
        # distance d'acceleration complete : v^2/a = 50 mm
        self.assertAlmostEqual(rail.duree(0, 50), 2 * math.sqrt(50 / 200.0),
                               places=9)
        self.assertAlmostEqual(rail.duree(0, 500), 100 / 200.0 + 500 / 100.0,
                               places=9)
        self.assertAlmostEqual(rail.duree(500, 0), rail.duree(0, 500), places=12)


class TestInstruction(unittest.TestCase):

    def test_sans_rail_l_instruction_n_en_parle_pas(self):
        r = Pilote(nom="rail")
        r.movj(x=200, y=0, z=60)
        ins = r.programme.instructions[-1]
        self.assertFalse(ins.avec_rail,
                         "un programme sans rail ne doit pas ramener le "
                         "chariot a zero")

    def test_la_coordonnee_de_chariot_est_portee(self):
        r = Pilote(nom="rail")
        r.rail = AxeLineaire(course=1000.0)
        r.movl(x=220, y=0, z=20, rail=400)
        ins = r.programme.instructions[-1]
        self.assertTrue(ins.avec_rail)
        self.assertAlmostEqual(ins.rail, 400.0, places=9)

    def test_la_consigne_est_bornee_par_la_course(self):
        r = Pilote(nom="rail")
        r.rail = AxeLineaire(course=1000.0)
        r.movj(x=200, y=0, z=60, rail=5000)
        self.assertAlmostEqual(r.programme.instructions[-1].rail, 1000.0,
                               places=9)

    def test_aller_rail_ne_bouge_que_le_chariot(self):
        r = Pilote(nom="rail")
        r.rail = AxeLineaire(course=1000.0)
        r.movj(x=210, y=10, z=55)
        avant = list(r.programme.instructions[-1].pose)
        r.aller_rail(400)
        ins = r.programme.instructions[-1]
        self.assertTrue(ins.avec_rail)
        self.assertAlmostEqual(ins.rail, 400.0, places=9)
        np.testing.assert_allclose(ins.pose, avant, atol=1e-9,
                                   err_msg="le bras ne doit pas bouger")

    def test_l_instruction_survit_au_json(self):
        from dobot_sim import Programme
        r = Pilote(nom="rail")
        r.rail = AxeLineaire(course=1000.0)
        r.movl(x=220, y=0, z=20, rail=250)
        copie = Programme.depuis_dict(r.programme.vers_dict())
        ins = copie.instructions[-1]
        self.assertTrue(ins.avec_rail)
        self.assertAlmostEqual(ins.rail, 250.0, places=9)


class TestRobotReel(unittest.TestCase):

    def setUp(self):
        self.faux = SerieSimulee()
        self.pont = PontReel(_robot())
        self.pont.liaison = _liaison_simulee(self.faux)
        self.pont.biais = (61.0, 0.0, 0.0)

    def _file(self, identifiant):
        return [p for i, p in self.faux.file if i == identifiant]

    def _immediates(self, identifiant):
        return [p for i, p in self.faux.immediates if i == identifiant]

    def test_un_mouvement_sans_rail_reste_un_ptp_ordinaire(self):
        r = Pilote(nom="cycle", vitesse=120)
        r.movl(cible(220, 0, 60))
        self.pont.televerser(r.programme, r.q_depart)
        self.assertEqual(len(self._file(CommandeDobot.PTP)), 1)
        self.assertEqual(self._file(CommandeDobot.PTP_AVEC_RAIL), [])

    def test_la_cinquieme_coordonnee_part_dans_la_bonne_trame(self):
        r = Pilote(nom="cycle", vitesse=120)
        r.rail = AxeLineaire(course=1000.0)
        r.movl(cible(220, 0, 60), rail=400)
        self.pont.televerser(r.programme, r.q_depart)
        trames = self._file(CommandeDobot.PTP_AVEC_RAIL)
        self.assertEqual(len(trames), 1)
        mode, x, y, z, rr, l = struct.unpack("<B5f", trames[0])
        self.assertAlmostEqual(l, 400.0, places=3)
        self.assertEqual(self._file(CommandeDobot.PTP), [],
                         "le PTP ordinaire ne doit pas etre utilise en plus")

    def test_le_rail_est_declare_une_seule_fois(self):
        r = Pilote(nom="cycle", vitesse=120)
        r.rail = AxeLineaire(course=1000.0)
        r.movl(cible(220, 0, 60), rail=100)
        r.movl(cible(220, 0, 80), rail=200)
        self.pont.televerser(r.programme, r.q_depart)
        self.assertEqual(len(self._immediates(CommandeDobot.RAIL_PRESENT)), 1)
        self.assertEqual(self._immediates(CommandeDobot.RAIL_PRESENT)[0][0], 1)

    def test_un_programme_sans_rail_ne_le_declare_pas(self):
        r = Pilote(nom="cycle", vitesse=120)
        r.movl(cible(220, 0, 60))
        self.pont.televerser(r.programme, r.q_depart)
        self.assertEqual(self._immediates(CommandeDobot.RAIL_PRESENT), [])

    def test_lecture_de_la_position_du_chariot(self):
        self.faux.rail = 321.5
        self.assertAlmostEqual(self.pont.position_rail(), 321.5, places=3)

    def test_cadence_du_rail(self):
        li = self.pont.liaison
        li.definir_cadence_rail(120.0, 240.0)
        trames = self._file(CommandeDobot.PARAMS_PTP_RAIL)
        self.assertEqual(len(trames), 1)
        v, a = struct.unpack("<2f", trames[0])
        self.assertAlmostEqual(v, 120.0, places=3)
        self.assertAlmostEqual(a, 240.0, places=3)


if __name__ == "__main__":
    unittest.main()
