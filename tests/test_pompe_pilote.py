"""
Pilote / executer_sur_robot : apres une ouverture de pince (ou un
relachement de ventouse), la pompe doit etre coupee dans la file -- apres
le mouvement suivant, ou apres 1 s en fin de programme.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dobot_sim import Pilote, cible          # noqa: E402
from dobot_sim.reel import PontReel          # noqa: E402


class _Li:
    def __init__(self):
        self.traces = []

    def __getattr__(self, nom):
        def appel(*a, **k):
            self.traces.append((nom, a, k))
            return len(self.traces)
        return appel


class _Pont(PontReel):
    def _deposer(self, li, commandes, progression, texte):
        for c in commandes:
            c()
        return 0


def _envoyer(pilote, outil):
    pont = _Pont(pilote.robot)
    pont.liaison = _Li()
    pont.biais = (61.0, 0.0, 0.0)
    pont._exiger = lambda: pont.liaison
    pont.televerser(pilote.programme, pilote.q_depart, outil=outil)
    return [(n, a, k) for n, a, k in pont.liaison.traces
            if n in ("pince", "ventouse", "attendre", "ptp")]


COUPURE = [("ventouse", (False,), {"active": False}),
           ("pince", (False,), {"active": False})]


class TestPompePilote(unittest.TestCase):

    def cycle(self, outil, fin=False):
        p = Pilote(outil=outil)
        p.repli()
        p.movj(cible(220, 0, 80))
        p.actionner(True)
        p.actionner(False)
        if not fin:
            p.repli()
        return _envoyer(p, outil)

    def test_coupure_apres_le_mouvement_suivant(self):
        t = self.cycle("pince")
        i = t.index(("pince", (False,), {}))
        self.assertEqual(t[i + 1][0], "ptp")          # le mouvement d'abord
        self.assertEqual(t[i + 2:i + 4], COUPURE)      # puis la coupure

    def test_coupure_en_fin_de_programme(self):
        t = self.cycle("pince", fin=True)
        self.assertEqual(t[-3], ("attendre", (1.0,), {}))
        self.assertEqual(t[-2:], COUPURE)

    def test_ventouse_aussi(self):
        t = self.cycle("ventouse")
        self.assertEqual(t[-2:], COUPURE)

    def test_pas_de_coupure_si_la_pince_reste_fermee(self):
        p = Pilote(outil="pince")
        p.repli()
        p.pince(True)
        p.repli()
        self.assertNotIn(COUPURE[1], _envoyer(p, "pince"))


if __name__ == "__main__":
    unittest.main()
