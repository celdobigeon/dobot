# -*- coding: utf-8 -*-
"""
Le convoyeur : son modele, son instruction, et les trames qui partent.

Le tapis n'avance pas « a peu pres » : une consigne de 60 mm/s doit
arriver au controleur en impulsions par seconde, et le capteur doit
declencher a l'abscisse ou il est pose. On verifie donc des nombres.
"""

import math
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_sim import Convoyeur, Pilote, cible
from dobot_sim import homogeneous as hg
from dobot_sim.convoyeur import IMPULSIONS_PAR_MM, PORTS_GP
from dobot_sim.liaison import CommandeDobot
from dobot_sim.reel import PontReel
from dobot_sim import DobotMagician
from test_protocole import SerieSimulee, _liaison_simulee


def _tapis(**kw):
    """Un tapis le long de +y, entree a (180, -180)."""
    base = dict(repere=hg.composer(hg.translation(180.0, -180.0, 0.0),
                                   hg.rot_z(math.pi / 2)),
                longueur=600.0, largeur=60.0)
    base.update(kw)
    return Convoyeur(**base)


class TestModele(unittest.TestCase):

    def test_le_pas_du_tapis(self):
        """
        200 pas par tour, 16 micro-pas, reducteur 10, poulie de 36 mm :
        32 000 impulsions pour 113,10 mm de tapis.
        """
        self.assertAlmostEqual(IMPULSIONS_PAR_MM,
                               200 * 16 * 10 / (math.pi * 36), places=6)
        self.assertAlmostEqual(IMPULSIONS_PAR_MM, 282.942, places=3)

    def test_conversion_reversible(self):
        t = _tapis()
        for v in (0.0, 10.0, 60.0, -25.0):
            i = t.impulsions_par_seconde(v)
            self.assertAlmostEqual(t.millimetres_par_seconde(i), v, places=9)

    def test_le_sens_inverse_la_consigne(self):
        avant = _tapis(sens=1).impulsions_par_seconde(60)
        arriere = _tapis(sens=-1).impulsions_par_seconde(60)
        self.assertAlmostEqual(avant, -arriere, places=9)

    def test_un_pas_mesure_remplace_le_nominal(self):
        t = _tapis(impulsions_par_mm=300.0)
        self.assertAlmostEqual(t.impulsions_par_seconde(10), 3000.0, places=6)

    # -- le deplacement, par paliers ----------------------------------------

    def test_deplacement_integre_les_paliers(self):
        t = _tapis()
        t.regler(0, 0.0).regler(50, 1.0).regler(0, 5.0).regler(-30, 7.0)
        self.assertAlmostEqual(t.deplacement(0.5), 0.0, places=9)
        self.assertAlmostEqual(t.deplacement(3.0), 100.0, places=9)
        self.assertAlmostEqual(t.deplacement(5.0), 200.0, places=9)
        self.assertAlmostEqual(t.deplacement(6.0), 200.0, places=9)
        self.assertAlmostEqual(t.deplacement(8.0), 170.0, places=9)

    def test_le_deplacement_ne_derive_pas_en_reculant(self):
        """On interroge dans le desordre : la valeur ne doit pas bouger."""
        t = _tapis()
        t.regler(0, 0.0).regler(40, 0.5).regler(0, 4.0)
        avant = t.deplacement(3.0)
        for instant in (10.0, 0.0, 2.0, 3.0, 7.0, 3.0):
            t.deplacement(instant)
        self.assertAlmostEqual(t.deplacement(3.0), avant, places=12)

    def test_regler_deux_fois_au_meme_instant(self):
        t = _tapis()
        t.regler(50, 2.0).regler(20, 2.0)
        self.assertAlmostEqual(t.vitesse(3.0), 20.0, places=9)

    # -- les pieces ---------------------------------------------------------

    def test_une_piece_avance_avec_le_tapis(self):
        t = _tapis()
        t.regler(0, 0.0).regler(60, 1.0)
        p = t.poser(s=0.0, t0=1.0)
        self.assertAlmostEqual(t.abscisse(p, 1.0), 0.0, places=9)
        self.assertAlmostEqual(t.abscisse(p, 6.0), 300.0, places=9)

    def test_une_piece_posee_en_retard_ne_rattrape_pas(self):
        t = _tapis()
        t.regler(60, 0.0)
        tot = t.poser(s=0.0, t0=0.0)
        tard = t.poser(s=0.0, t0=2.0)
        self.assertAlmostEqual(t.abscisse(tot, 4.0) - t.abscisse(tard, 4.0),
                               120.0, places=9)

    def test_la_piece_se_situe_dans_le_repere_de_la_base(self):
        t = _tapis()
        t.regler(60, 0.0)
        p = t.poser(s=0.0, t0=0.0)
        T = t.situation(p, 5.0)             # 300 mm parcourus
        np.testing.assert_allclose(T[:3, 3], (180.0, 120.0, 0.0), atol=1e-9)

    def test_sortie_de_tapis(self):
        t = _tapis()
        t.regler(100, 0.0)
        p = t.poser(s=0.0, t0=0.0)
        self.assertTrue(t.sur_le_tapis(p, 3.0))
        self.assertFalse(t.sur_le_tapis(p, 7.0), "600 mm depasses")
        p.emportee = True
        self.assertFalse(t.sur_le_tapis(p, 3.0), "prise par le robot")

    # -- capteur et camera --------------------------------------------------

    def test_le_capteur_declenche_a_son_abscisse(self):
        t = _tapis(capteur_s=450.0, capteur_fenetre=12.0)
        t.regler(0, 0.0).regler(60, 1.0)
        p = t.poser(s=0.0, t0=1.0)
        attendu = 1.0 + (450.0 - 12.0) / 60.0
        trouve = t.instant_au_capteur(p, depart=1.0, pas=0.005)
        self.assertIsNotNone(trouve)
        self.assertAlmostEqual(trouve, attendu, delta=0.02)
        self.assertFalse(t.capteur(3.0))
        self.assertTrue(t.capteur(trouve))

    def test_le_capteur_est_muet_tapis_arrete(self):
        t = _tapis()
        p = t.poser(s=0.0, t0=0.0)
        self.assertIsNone(t.instant_au_capteur(p, limite=5.0, pas=0.05))

    def test_la_camera_voit_son_champ(self):
        t = _tapis(camera_s=300.0, camera_champ=80.0)
        t.regler(60, 0.0)
        p = t.poser(s=0.0, t0=0.0)
        self.assertEqual(t.devant_camera(5.0), [p], "pile sous la camera")
        self.assertEqual(t.devant_camera(1.0), [])

    def test_une_cible_au_dessus_du_tapis_a_l_axe_vers_le_bas(self):
        t = _tapis()
        C = t.cible(450.0, 0.0, 5.0)
        np.testing.assert_allclose(C[:3, 2], (0.0, 0.0, -1.0), atol=1e-9)
        np.testing.assert_allclose(C[:3, 3], (180.0, 270.0, 5.0), atol=1e-9)


class TestInstruction(unittest.TestCase):

    def test_le_programme_porte_le_pas_du_tapis(self):
        t = _tapis(impulsions_par_mm=300.0, index_moteur=1)
        r = Pilote(nom="tapis")
        r.convoyeur(60, t)
        ins = r.programme.instructions[-1]
        self.assertEqual(ins.type, "CONVOYEUR")
        self.assertAlmostEqual(ins.vitesse, 60.0, places=9)
        self.assertTrue(ins.etat)
        self.assertEqual(ins.port, 1)
        self.assertAlmostEqual(ins.impulsions_par_mm, 300.0, places=9)

    def test_vitesse_nulle_vaut_arret(self):
        r = Pilote(nom="tapis")
        r.convoyeur(0, _tapis())
        self.assertFalse(r.programme.instructions[-1].etat)

    def test_le_tapis_du_pilote_sert_de_defaut(self):
        r = Pilote(nom="tapis")
        r.tapis = _tapis(impulsions_par_mm=250.0)
        r.convoyeur(30)
        self.assertAlmostEqual(
            r.programme.instructions[-1].impulsions_par_mm, 250.0, places=9)

    def test_l_instruction_survit_au_json(self):
        from dobot_sim import Programme
        r = Pilote(nom="tapis")
        r.convoyeur(45, _tapis(impulsions_par_mm=300.0))
        copie = Programme.depuis_dict(r.programme.vers_dict())
        ins = copie.instructions[-1]
        self.assertEqual(ins.type, "CONVOYEUR")
        self.assertAlmostEqual(ins.vitesse, 45.0, places=9)
        self.assertAlmostEqual(ins.impulsions_par_mm, 300.0, places=9)

    def test_le_convoyeur_ne_neutralise_pas_le_lissage(self):
        """
        Comme les autres actions : le tapis ne demande aucun temps, il ne
        doit donc pas proteger en douce le mouvement qui le precede.
        """
        r = Pilote(nom="tapis", vitesse=120)
        r.movl(cible(220, 0, 60), lissage=100)
        r.convoyeur(60, _tapis())
        r.movl(cible(220, 60, 60), lissage=100)
        traj = r.programme.construire(r.robot, r.q_depart)
        self.assertEqual(traj.segments[1].type_mouvement, "ACTION")
        self.assertGreater(traj.recouvrements[0], 0.0)

    def test_le_resume_est_lisible(self):
        r = Pilote(nom="tapis")
        r.convoyeur(60, _tapis())
        r.convoyeur(0, _tapis())
        texte = r.resume()
        self.assertIn("TAPIS", texte)
        self.assertIn("+60 mm/s", texte)
        self.assertIn("arret", texte)


class TestRobotReel(unittest.TestCase):
    """Ce qui part vraiment sur le fil."""

    def setUp(self):
        self.faux = SerieSimulee()
        self.pont = PontReel(DobotMagician().avec_outil(62.0))
        self.pont.liaison = _liaison_simulee(self.faux)
        self.pont.biais = (61.0, 0.0, 0.0)

    def _trames(self, identifiant, en_file=True):
        source = self.faux.file if en_file else self.faux.immediates
        return [p for i, p in source if i == identifiant]

    def test_la_consigne_part_en_impulsions_par_seconde(self):
        t = _tapis()
        self.pont.convoyeur(60.0, t)
        trames = self._trames(CommandeDobot.MOTEUR_ETENDU, en_file=False)
        self.assertEqual(len(trames), 1)
        index, actif, vitesse = struct.unpack("<2Bf", trames[0])
        self.assertEqual(index, 0)
        self.assertEqual(actif, 1)
        self.assertAlmostEqual(vitesse, 60.0 * IMPULSIONS_PAR_MM, places=2)

    def test_arret_immediat(self):
        self.pont.arreter_convoyeur(_tapis())
        index, actif, vitesse = struct.unpack(
            "<2Bf", self._trames(CommandeDobot.MOTEUR_ETENDU, en_file=False)[-1])
        self.assertEqual(actif, 0)
        self.assertAlmostEqual(vitesse, 0.0, places=6)

    def test_le_sens_arriere_part_en_negatif(self):
        self.pont.convoyeur(-40.0, _tapis())
        _, actif, vitesse = struct.unpack(
            "<2Bf", self._trames(CommandeDobot.MOTEUR_ETENDU, en_file=False)[-1])
        self.assertEqual(actif, 1)
        self.assertLess(vitesse, 0.0)

    def test_le_second_moteur_est_adressable(self):
        self.pont.convoyeur(20.0, _tapis(index_moteur=1))
        index, _, _ = struct.unpack(
            "<2Bf", self._trames(CommandeDobot.MOTEUR_ETENDU, en_file=False)[-1])
        self.assertEqual(index, 1)

    def test_un_programme_televerse_son_convoyeur(self):
        t = _tapis(impulsions_par_mm=300.0)
        r = Pilote(nom="cycle", outil="ventouse", vitesse=120)
        r.tapis = t
        r.convoyeur(60)
        r.movl(cible(220, 0, 60))
        r.convoyeur(0)
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        trames = self._trames(CommandeDobot.MOTEUR_ETENDU)
        self.assertEqual(len(trames), 2, "deux reglages de tapis attendus")
        _, actif, vitesse = struct.unpack("<2Bf", trames[0])
        self.assertEqual(actif, 1)
        self.assertAlmostEqual(vitesse, 60.0 * 300.0, places=1)
        _, actif, vitesse = struct.unpack("<2Bf", trames[1])
        self.assertEqual(actif, 0)

    def test_le_pas_nominal_sert_de_secours(self):
        """Un programme sans pas de tapis ne doit pas rester muet."""
        r = Pilote(nom="cycle", vitesse=120)
        r.convoyeur(50)                      # aucun tapis : pas inconnu
        self.assertEqual(r.programme.instructions[-1].impulsions_par_mm, 0.0)
        self.pont.televerser(r.programme, r.q_depart)
        _, _, vitesse = struct.unpack(
            "<2Bf", self._trames(CommandeDobot.MOTEUR_ETENDU)[-1])
        self.assertAlmostEqual(vitesse, 50.0 * IMPULSIONS_PAR_MM, places=1)

    def test_le_capteur_infrarouge_est_allume_une_fois(self):
        t = _tapis(capteur_port=PORTS_GP["GP2"])
        self.faux.ir = 1
        self.pont.capteur_piece(t)
        self.pont.capteur_piece(t)
        activations = self._trames(CommandeDobot.CAPTEUR_IR, en_file=False)
        allumages = [p for p in activations if len(p) == 3]
        self.assertEqual(len(allumages), 1, "une seule activation attendue")
        self.assertEqual(allumages[0][0], 1)
        self.assertEqual(allumages[0][1], PORTS_GP["GP2"])

    def test_le_capteur_cable_en_tout_ou_rien(self):
        t = _tapis(capteur_port=None, capteur_eio=5)
        self.assertIsInstance(self.pont.capteur_piece(t), bool)
        self.assertEqual(self._trames(CommandeDobot.CAPTEUR_IR,
                                      en_file=False), [])

    def test_sans_capteur_declare_le_refus_est_parlant(self):
        from dobot_sim.reel import ErreurPont
        with self.assertRaises(ErreurPont) as ctx:
            self.pont.capteur_piece(_tapis(capteur_port=None,
                                           capteur_eio=None))
        self.assertIn("capteur", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
