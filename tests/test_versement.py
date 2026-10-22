"""
Versement d'un programme long dans la file du controleur.

La file du Magician ne tient que 32 commandes (mesure, firmware 3.7.0) ;
au-dela, il renvoie le meme index et JETTE la commande, sans erreur. Ces
tests rejouent ce comportement avec un faux controleur.
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dobot_sim import Pilote, cible                               # noqa: E402
from dobot_sim.liaison import CommandeDobot as C, LiaisonDobot   # noqa: E402
from dobot_sim.reel import PontReel, _Enregistreur               # noqa: E402


class FauxDobot(LiaisonDobot):
    """File de `capacite` places, execute une commande par tick."""

    def __init__(self, capacite=32):
        super().__init__(port="faux")
        self.capacite = capacite
        self.file, self.executees = [], []
        self.courant = self.dernier = 88
        self.demarree = False

    @property
    def ouverte(self):
        return True

    def echanger(self, ident, params=b"", ecriture=False, en_file=False,
                 essais=3):
        if ident == C.FILE_INDEX_COURANT:
            return struct.pack("<Q", self.courant)
        if ident == C.FILE_VIDER:
            self.file, self.dernier = [], self.courant
            return b""
        if ident in (C.FILE_ARRETER, C.FILE_ARRET_FORCE):
            self.demarree = False
            return b""
        if ident == C.FILE_DEMARRER:
            self.demarree = True
            return b""
        if en_file:
            if len(self.file) < self.capacite:       # sinon : jetee
                self.dernier += 1
                self.file.append((self.dernier, ident, params))
            self._dernier_index = self.dernier
            return struct.pack("<Q", self.dernier)
        return b""

    def tick(self):
        if self.demarree and self.file:
            idx, ident, params = self.file.pop(0)
            self.courant = idx
            self.executees.append((ident, params))


def programme_long(n=40):
    robot = Pilote(nom="long")
    robot.repli()
    for i in range(n):
        robot.movj(cible(200, -60 + 3 * i, 60))
        robot.attendre(0.1)
    return robot


def brancher(faux, robot):
    pont = PontReel(robot, journal=lambda m: None)
    pont.liaison = faux
    faux.sur_vidage = pont._abandonner_envois
    pont.biais = (61.0, 0.0, 0.0)
    return pont


def attendu(pilote):
    """Commandes de file que le programme doit produire, dans l'ordre."""
    pont = PontReel(pilote.robot, journal=lambda m: None)
    pont.biais = (61.0, 0.0, 0.0)
    e = _Enregistreur()
    for ins in pilote.programme.instructions:
        pont._emettre(e, ins, pilote.programme, "ventouse")
    return [(i, p) for i, p, _w, f in e.commandes if f]


class TestVersement(unittest.TestCase):

    def executer(self, capacite, urgence_au_tour=None):
        pilote = programme_long()
        faux = FauxDobot(capacite)
        pont = brancher(faux, pilote.robot)
        pont.televerser(pilote.programme, pilote.q_depart, outil="ventouse")
        pont.demarrer()
        for t in range(2000):
            if t == urgence_au_tour:
                pont.arret_urgence()
            faux.tick()
            pont.alimenter()
        return pilote, faux, pont

    def test_programme_plus_long_que_la_file(self):
        pilote, faux, pont = self.executer(32)
        self.assertGreater(len(attendu(pilote)), 32)
        self.assertEqual(faux.executees, attendu(pilote))
        self.assertTrue(pont.termine())
        self.assertEqual(pont.en_reserve, 0)

    def test_commande_jetee_puis_renvoyee(self):
        # file plus petite que prevu : le controleur jette, rien ne se perd
        pilote, faux, pont = self.executer(10)
        self.assertEqual(faux.executees, attendu(pilote))

    def test_arret_urgence_abandonne_la_reserve(self):
        pilote, faux, pont = self.executer(32, urgence_au_tour=15)
        self.assertEqual(len(faux.executees), 15)
        self.assertEqual(pont.en_reserve, 0)
        self.assertFalse(faux.file)


if __name__ == "__main__":
    unittest.main()
