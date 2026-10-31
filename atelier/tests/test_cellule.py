# -*- coding: utf-8 -*-
"""
Plusieurs robots dans la meme cellule.

Chaque poste a son propre controleur simule : on verifie que chacun
recoit SON programme et pas celui du voisin, que les departs sont
groupes, et surtout qu'un arret d'urgence ou un defaut vaut pour toute
la cellule -- arreter un bras pendant que l'autre avance vers lui ne
protege de rien.
"""

import os
import struct
import sys
import time
import unittest

import numpy as np

# le paquet vit dans cellule/, voisin de atelier/ -- voir
# atelier/cellule_sur_le_chemin.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_sim import Cellule, DobotMagician, ErreurCellule, cible
from dobot_sim.liaison import CommandeDobot
from test_protocole import SerieSimulee, _liaison_simulee


def _brancher(cellule, **faux_par_nom):
    """Donne a chaque poste son controleur simule."""
    faux = {}
    for nom, poste in cellule.postes.items():
        f = faux_par_nom.get(nom) or SerieSimulee()
        poste.pont.liaison = _liaison_simulee(f)
        poste.pont.biais = (61.0, 0.0, 0.0)
        poste.port = poste.port or "SIMULE"
        faux[nom] = f
    return faux


def _cellule(**kw):
    c = Cellule(**kw)
    c.ajouter("alpha", port="SIM1", outil="ventouse", vitesse=120)
    c.ajouter("beta", port="SIM2", outil="pince", vitesse=120)
    return c


class TestComposition(unittest.TestCase):

    def test_les_robots_sont_distincts(self):
        c = _cellule()
        self.assertEqual(c.noms, ["alpha", "beta"])
        self.assertEqual(len(c), 2)
        self.assertIsNot(c["alpha"], c["beta"])
        self.assertIsNot(c["alpha"].robot, c["beta"].robot,
                         "chaque robot a son propre modele")

    def test_deux_robots_de_meme_nom_refuses(self):
        c = _cellule()
        with self.assertRaises(ValueError):
            c.ajouter("alpha", port="SIM3")

    def test_le_journal_dit_qui_parle(self):
        lignes = []
        c = Cellule(journal=lignes.append)
        c.ajouter("alpha", port="SIM1")
        _brancher(c)
        c.postes["alpha"].pont.journal("essai")
        self.assertTrue(any("[alpha]" in l for l in lignes))


class TestExecution(unittest.TestCase):

    def setUp(self):
        self.c = _cellule()
        self.faux = _brancher(self.c)

    def _ptp(self, nom):
        return [p for i, p in self.faux[nom].file if i == CommandeDobot.PTP]

    def test_chacun_recoit_son_programme(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        self.c["beta"].movl(cible(200, 40, 90))
        self.c.lancer()
        self.assertEqual(len(self._ptp("alpha")), 1)
        self.assertEqual(len(self._ptp("beta")), 2)

    def test_un_poste_sans_programme_reste_tranquille(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c.lancer()
        self.assertEqual(self._ptp("beta"), [])

    def test_le_decalage_de_depart_est_mesure(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        decalage = self.c.lancer()
        self.assertGreaterEqual(decalage, 0.0)
        self.assertLess(decalage, 1.0, "les demarrages doivent etre groupes")
        self.assertEqual(decalage, self.c.decalage_demarrage)

    def test_les_televersements_precedent_les_demarrages(self):
        """
        L'ordre compte : televerser tout le monde PUIS demarrer tout le
        monde ramene l'ecart de depart de quelques secondes a quelques
        millisecondes. On le verifie sur la trace du controleur.
        """
        for nom in ("alpha", "beta"):
            for _ in range(5):
                self.c[nom].movl(cible(220, 0, 60))
        self.c.lancer()
        for nom in ("alpha", "beta"):
            idents = [i for i, _ in self.faux[nom].immediates]
            demarrage = idents.index(CommandeDobot.FILE_DEMARRER)
            # toutes les commandes en file sont parties avant le demarrage
            self.assertEqual(len(self.faux[nom].file), 5)
            self.assertGreater(demarrage, 0)

    def test_un_programme_fautif_n_envoie_rien(self):
        self.c["alpha"].movl(cible(900, 0, 60))        # hors d'atteinte
        self.c["beta"].movl(cible(200, 40, 70))
        with self.assertRaises(ErreurCellule) as ctx:
            self.c.lancer()
        self.assertIn("alpha", str(ctx.exception))
        self.assertEqual(self._ptp("beta"), [],
                         "le robot sain ne doit pas partir non plus")

    def test_controler_designe_le_coupable(self):
        self.c["alpha"].movl(cible(900, 0, 60))
        defauts = self.c.controler()
        self.assertIn("alpha", defauts)
        self.assertNotIn("beta", defauts)

    def test_lancer_exige_la_connexion(self):
        c = _cellule()                                  # non branchee
        c["alpha"].movl(cible(220, 0, 60))
        with self.assertRaises(ErreurCellule) as ctx:
            c.lancer()
        self.assertIn("alpha", str(ctx.exception))

    def test_attendre_tous(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        self.c.lancer()
        self.c.attendre_tous(limite=5.0)
        self.assertTrue(self.c.termine())


class TestSecurite(unittest.TestCase):

    def setUp(self):
        self.c = _cellule()
        self.faux = _brancher(self.c)

    def test_l_arret_d_urgence_vaut_pour_toute_la_cellule(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        self.c.lancer()
        self.c.arret_urgence()
        for nom in ("alpha", "beta"):
            idents = [i for i, _ in self.faux[nom].immediates]
            self.assertIn(CommandeDobot.FILE_ARRET_FORCE, idents,
                          "le robot '{}' n'a pas ete coupe".format(nom))

    def test_un_defaut_arrete_les_autres(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        self.c.lancer()
        self.faux["beta"].alarmes = [0x40]              # beta tombe en defaut
        with self.assertRaises(ErreurCellule) as ctx:
            self.c.attendre_tous(limite=2.0)
        self.assertIn("beta", str(ctx.exception))
        idents = [i for i, _ in self.faux["alpha"].immediates]
        self.assertIn(CommandeDobot.FILE_ARRET_FORCE, idents,
                      "alpha devait etre arrete aussi")

    def test_une_cellule_non_solidaire_laisse_les_autres(self):
        c = _cellule(solidaire=False)
        faux = _brancher(c)
        c["alpha"].movl(cible(220, 0, 60))
        c["beta"].movl(cible(200, 40, 70))
        c.lancer()
        faux["beta"].alarmes = [0x40]
        with self.assertRaises(ErreurCellule):
            c.attendre_tous(limite=2.0)
        idents = [i for i, _ in faux["alpha"].immediates]
        self.assertNotIn(CommandeDobot.FILE_ARRET_FORCE, idents)

    def test_la_sortie_du_bloc_coupe_et_ferme(self):
        with _cellule() as c:
            faux = _brancher(c)
            c["alpha"].movl(cible(220, 0, 60))
            c.lancer()
        idents = [i for i, _ in faux["alpha"].immediates]
        self.assertIn(CommandeDobot.FILE_ARRET_FORCE, idents)
        self.assertFalse(c.postes["alpha"].connecte)

    def test_une_connexion_partielle_est_annulee(self):
        """Une cellule a moitie connectee ferait croire qu'on peut lancer."""
        c = _cellule()
        c.postes["beta"].port = ""                      # pas de port
        with self.assertRaises(ErreurCellule):
            c.connecter()
        self.assertFalse(c.postes["alpha"].connecte,
                         "la liaison deja ouverte devait etre refermee")


class TestRendezVous(unittest.TestCase):

    def setUp(self):
        self.c = _cellule()
        self.faux = _brancher(self.c)

    def test_une_etape_vide_les_programmes(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        self.c.etape(limite=5.0)
        self.assertEqual(self.c["alpha"].programme.instructions, [])
        self.assertEqual(self.c["beta"].programme.instructions, [])

    def test_l_etape_suivante_repart_de_la_pose_atteinte(self):
        alpha = self.c["alpha"]
        alpha.movl(cible(220, 0, 60))
        self.c.etape(limite=5.0)
        depart = np.asarray(alpha.q_depart, dtype=float)
        atteint = alpha.robot.mgi(cible(220, 0, 60))
        np.testing.assert_allclose(depart, atteint, atol=1e-6)

    def test_deux_etapes_enchainent_les_mouvements(self):
        """
        Chaque etape televerse, donc VIDE la file du controleur : elle ne
        garde pas l'historique. On compare donc ce qui est parti a
        chaque etape, pas un cumul.
        """
        def z_envoye():
            trames = [p for i, p in self.faux["alpha"].file
                      if i == CommandeDobot.PTP]
            self.assertEqual(len(trames), 1)
            return struct.unpack("<B4f", trames[0])[3]

        self.c["alpha"].movl(cible(220, 0, 60))
        self.c.etape(limite=5.0)
        premier = z_envoye()

        self.c["alpha"].movl(cible(220, 0, 90))
        self.c.etape(limite=5.0)
        second = z_envoye()

        self.assertAlmostEqual(second - premier, 30.0, places=3,
                               msg="la seconde etape doit viser 30 mm plus haut")

    def test_l_etape_rend_le_decalage(self):
        self.c["alpha"].movl(cible(220, 0, 60))
        self.c["beta"].movl(cible(200, 40, 70))
        decalage = self.c.etape(limite=5.0)
        self.assertGreaterEqual(decalage, 0.0)
        self.assertLess(decalage, 1.0)


if __name__ == "__main__":
    unittest.main()
