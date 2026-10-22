#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests du protocole Dobot et du pont vers la machine reelle.

Aucun materiel n'est necessaire : un port serie SIMULE rejoue le
comportement du controleur, ce qui permet d'exercer toute la pile --
cadrage, retransmission, index de file, traduction d'un programme.

Les valeurs de reference viennent du protocole officiel V1.1.5 et d'un
releve sur une machine reelle (firmware 3.7.0), consigne dans le README.
"""

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import DobotMagician, Pilote, cible
from dobot_sim.liaison import (CommandeDobot, ErreurLiaison, ErreurTrame, PTP,
                               LiaisonDobot, Pose, construire_trame,
                               decoder_trame, resume_alarmes)
from dobot_sim.reel import ErreurPont, PontReel


class TestCadrage(unittest.TestCase):
    """Trame : en-tete, longueur, Ctrl, checksum, endianness."""

    def test_exemple_du_protocole(self):
        # GetPose : ID 10, aucun parametre. La somme vaut 0x0A et son
        # complement a deux 0xF6 -- c'est l'exemple de la section 1.2.2.
        self.assertEqual(construire_trame(CommandeDobot.POSE).hex().upper(),
                         "AAAA020A00F6")

    def test_checksum_est_un_complement_a_deux(self):
        for ident, params, rw, q in ((10, b"", False, False),
                                     (84, struct.pack("<B4f", 1, 200., 0., 50., 0.),
                                      True, True),
                                     (131, bytes([3, 1]), True, True),
                                     (245, b"", True, False)):
            t = construire_trame(ident, params, rw, q)
            charge = t[3:-1]
            self.assertEqual((sum(charge) + t[-1]) % 256, 0)

    def test_ctrl_bit0_rw_bit1_queued(self):
        for rw, q, attendu in ((False, False, 0), (True, False, 1),
                               (False, True, 2), (True, True, 3)):
            self.assertEqual(construire_trame(1, b"", rw, q)[4], attendu)

    def test_longueur_vaut_deux_plus_n(self):
        for n in (0, 1, 2, 4, 17, 32):
            t = construire_trame(84, b"\x00" * n, True, True)
            self.assertEqual(t[2], 2 + n)
            self.assertEqual(len(t), 3 + (2 + n) + 1)

    def test_little_endian(self):
        t = construire_trame(110, struct.pack("<I", 1000), True, True)
        self.assertEqual(t[5:9].hex().upper(), "E8030000")

    def test_aller_retour(self):
        params = struct.pack("<B4f", PTP.MOVL_XYZ, 1., 2., 3., 4.)
        ident, ctrl, p = decoder_trame(construire_trame(84, params, True, True))
        self.assertEqual((ident, ctrl, p), (84, 3, params))

    def test_trames_invalides_refusees(self):
        bonne = construire_trame(10)
        mauvaises = (b"\xAB\xAA" + bonne[2:],                       # en-tete
                     bonne[:-1] + bytes([(bonne[-1] + 1) & 0xFF]),  # checksum
                     b"\xAA\xAA\x02",                               # trop courte
                     b"\xAA\xAA\x09\x0A\x00\xF4")                   # longueur
        for m in mauvaises:
            with self.assertRaises(ErreurTrame):
                decoder_trame(m)

    def test_charge_utile_trop_grande(self):
        with self.assertRaises(ValueError):
            construire_trame(10, b"\x00" * 254)

    def test_decodage_de_pose(self):
        brut = struct.pack("<8f", 220.5, -30.25, 10.0, 45.0, 1.0, 2.0, 3.0, 4.0)
        p = Pose.depuis_octets(brut)
        self.assertEqual((p.x, p.y, p.z, p.r), (220.5, -30.25, 10.0, 45.0))
        self.assertEqual(p.angles, (1.0, 2.0, 3.0, 4.0))
        with self.assertRaises(ErreurTrame):
            Pose.depuis_octets(b"\x00" * 8)

    def test_resume_alarmes(self):
        self.assertEqual(resume_alarmes([]), [])
        self.assertIn("J2 en butee negative", resume_alarmes([0x43])[0])
        self.assertIn("J3 en butee positive", resume_alarmes([0x44])[0])
        self.assertIn("hors de l'espace", resume_alarmes([0x11])[0])
        # un code inconnu renvoie vers le bon document plutot que d'etre devine
        self.assertIn("ALARM", resume_alarmes([0xFE])[0])


# ---------------------------------------------------------------------------

class SerieSimulee:
    """
    Controleur Dobot de laboratoire.

    Repond a toute commande comme le ferait la machine : reponse vide pour
    une commande immediate, index sur 64 bits pour une commande en file.
    Conserve la file pour que les tests puissent l'inspecter.
    """

    is_open = True

    def __init__(self, pose=(208.9, 205.5, 73.9, 44.5),
                 angles=(44.5, 40.6, 11.2, 0.0), biais=(61.0, 0.0, 0.0),
                 erreur=(0.0, 0.0)):
        self.pose = pose
        self.angles = angles
        self.biais = biais
        self.erreur = erreur
        self.file = []              # (identifiant, parametres)
        self.immediates = []
        self.index = 0
        self.demarree = False
        self._sortie = b""
        self.octets_ecrits = 0

    # -- interface pyserial minimale
    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        self._sortie = b""

    def flush(self):
        pass

    def close(self):
        self.is_open = False

    def read(self, n=1):
        morceau, self._sortie = self._sortie[:n], self._sortie[n:]
        return morceau

    def write(self, trame):
        self.octets_ecrits += len(trame)
        ident, ctrl, params = decoder_trame(trame)
        en_file = bool(ctrl & 2)
        self._sortie += construire_trame(
            ident, self._traiter(ident, params, en_file), bool(ctrl & 1), en_file)
        return len(trame)

    # -- comportement du controleur
    def _traiter(self, ident, params, en_file):
        if en_file:
            self.index += 1
            self.file.append((ident, params))
            return struct.pack("<Q", self.index)

        self.immediates.append((ident, params))
        if ident == CommandeDobot.POSE:
            return struct.pack("<8f", *self.pose, *self.angles)
        if ident == CommandeDobot.VERSION:
            return bytes((3, 7, 0))
        if ident == CommandeDobot.PARAMS_EFFECTEUR:
            return struct.pack("<3f", *self.biais)
        if ident == CommandeDobot.ERREUR_CAPTEURS:
            if params:                       # ecriture
                self.erreur = struct.unpack("<2f", params[:8])
                return b""
            return struct.pack("<2f", *self.erreur)
        if ident == CommandeDobot.REINIT_POSE and params:
            # ResetPose manuel : le controleur adopte les angles annonces
            manuel, bras, avant = struct.unpack("<B2f", params[:9])
            if manuel:
                self.angles = (self.angles[0], bras, avant, self.angles[3])
            return b""
        if ident == CommandeDobot.FILE_INDEX_COURANT:
            return struct.pack("<Q", self.index if self.demarree else 0)
        if ident == CommandeDobot.ALARMES:
            return b"\x00" * 16
        if ident == CommandeDobot.ES_ENTREE:
            return bytes((params[0] if params else 1, 1))
        if ident == CommandeDobot.FILE_DEMARRER:
            self.demarree = True
        elif ident == CommandeDobot.FILE_ARRETER:
            self.demarree = False
        elif ident == CommandeDobot.FILE_VIDER:
            self.file.clear()
        return b""


def _liaison_simulee(faux=None):
    """LiaisonDobot branchee sur un controleur simule."""
    li = LiaisonDobot("SIMULE")
    li._serie = faux or SerieSimulee()
    return li


class TestLiaison(unittest.TestCase):

    def setUp(self):
        self.faux = SerieSimulee()
        self.li = _liaison_simulee(self.faux)

    def test_lectures(self):
        self.assertEqual(self.li.version(), "3.7.0")
        p = self.li.pose()
        self.assertAlmostEqual(p.x, 208.9, places=3)
        self.assertEqual(len(p.angles), 4)
        self.assertEqual(self.li.biais_effecteur()[0], 61.0)
        self.assertEqual(self.li.alarmes(), [])

    def test_commande_en_file_renvoie_un_index(self):
        i1 = self.li.ptp(PTP.MOVL_XYZ, 200.0, 0.0, 50.0, 0.0)
        i2 = self.li.attendre(0.5)
        self.assertEqual((i1, i2), (1, 2))
        self.assertEqual(self.li.dernier_index, 2)
        self.assertEqual([f[0] for f in self.faux.file],
                         [CommandeDobot.PTP, CommandeDobot.ATTENTE])

    def test_parametres_ptp_bien_encodes(self):
        self.li.ptp(PTP.MOVJ_XYZ, 1.5, -2.5, 3.5, 4.5)
        ident, params = self.faux.file[-1]
        self.assertEqual(ident, CommandeDobot.PTP)
        mode, x, y, z, r = struct.unpack("<B4f", params)
        self.assertEqual(mode, PTP.MOVJ_XYZ)
        self.assertAlmostEqual(x, 1.5, places=5)
        self.assertAlmostEqual(r, 4.5, places=5)

    def test_attente_en_millisecondes(self):
        self.li.attendre(1.25)
        self.assertEqual(struct.unpack("<I", self.faux.file[-1][1])[0], 1250)

    def test_arret_urgence_coupe_et_vide(self):
        self.li.ptp(PTP.MOVL_XYZ, 0, 0, 0, 0)
        self.li.arret_urgence()
        idents = [i for i, _ in self.faux.immediates]
        self.assertIn(CommandeDobot.FILE_ARRET_FORCE, idents)
        self.assertIn(CommandeDobot.FILE_VIDER, idents)
        self.assertEqual(self.faux.file, [])

    def test_liaison_fermee_refusee(self):
        li = LiaisonDobot("SIMULE")
        with self.assertRaises(ErreurLiaison):
            li.pose()

    def test_reponse_incoherente_signalee(self):
        class Menteur(SerieSimulee):
            def write(self, trame):
                # repond systematiquement avec un autre identifiant
                self._sortie += construire_trame(99, b"")
                return len(trame)

        li = _liaison_simulee(Menteur())
        li.delai = 0.05
        with self.assertRaises(ErreurLiaison):
            li.version()


class TestCalibration(unittest.TestCase):
    """
    Recalage des capteurs d'angle.

    Le protocole (1.16) : les capteurs du bras arriere et de l'avant-bras
    portent un offset statique. Deux commandes le traitent, et AUCUNE ne
    fait bouger le bras.
    """

    def setUp(self):
        self.faux = SerieSimulee(erreur=(0.25, -0.5))
        self.li = _liaison_simulee(self.faux)

    def test_lecture_de_l_erreur_statique(self):
        e = self.li.erreur_capteurs()
        self.assertAlmostEqual(e[0], 0.25, places=5)
        self.assertAlmostEqual(e[1], -0.5, places=5)

    def test_ecriture_de_l_erreur_statique(self):
        self.li.definir_erreur_capteurs(-3.5, 1.25)
        e = self.li.erreur_capteurs()
        self.assertAlmostEqual(e[0], -3.5, places=5)
        self.assertAlmostEqual(e[1], 1.25, places=5)

    def test_l_erreur_statique_est_immediate_pas_en_file(self):
        # une commande en file ne prendrait effet qu'a l'execution
        self.li.definir_erreur_capteurs(1.0, 2.0)
        self.assertEqual(self.faux.file, [])
        self.assertIn(CommandeDobot.ERREUR_CAPTEURS,
                      [i for i, _ in self.faux.immediates])

    def test_reset_pose_manuel_encode_bien_ses_champs(self):
        self.li.redefinir_pose(0.0, 0.0)
        ident, params = self.faux.immediates[-1]
        self.assertEqual(ident, CommandeDobot.REINIT_POSE)
        manuel, bras, avant = struct.unpack("<B2f", params[:9])
        self.assertEqual(manuel, 1)
        self.assertAlmostEqual(bras, 0.0, places=5)
        self.assertAlmostEqual(avant, 0.0, places=5)

    def test_reset_pose_auto_met_manual_a_zero(self):
        self.li.redefinir_pose_auto()
        _ident, params = self.faux.immediates[-1]
        self.assertEqual(struct.unpack("<B2f", params[:9])[0], 0)

    def test_reset_pose_change_les_angles_annonces(self):
        self.li.redefinir_pose(0.0, 0.0)
        p = self.li.pose()
        self.assertAlmostEqual(p.angles[1], 0.0, places=4)
        self.assertAlmostEqual(p.angles[2], 0.0, places=4)

    def test_aucune_de_ces_commandes_ne_bouge_le_bras(self):
        # rien ne part dans la file : aucun mouvement n'est programme
        self.li.definir_erreur_capteurs(1.0, 2.0)
        self.li.redefinir_pose(0.0, 0.0)
        self.li.redefinir_pose_auto()
        self.assertEqual(self.faux.file, [])


class TestPont(unittest.TestCase):
    """Traduction d'un programme et conversion de repere."""

    def setUp(self):
        self.modele = DobotMagician().avec_outil(62.0)
        self.pont = PontReel(self.modele)
        self.faux = SerieSimulee()
        self.pont.liaison = _liaison_simulee(self.faux)
        self.pont.biais = (61.0, 0.0, 0.0)

    # -- conversion de repere

    def test_ecart_en_z_vaut_d1_moins_lt(self):
        x, y, z, r = self.pont.vers_controleur(220.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(z, 62.0 - 139.0, places=9)

    def test_ecart_radial_vaut_xbias_moins_l3(self):
        x, y, _z, _r = self.pont.vers_controleur(220.0, 0.0, 50.0, 0.0)
        self.assertAlmostEqual(float(np.hypot(x, y)) - 220.0,
                               61.0 - self.modele.geo.l3, places=9)

    def test_aller_retour_de_repere(self):
        for pose in ((220.0, 0.0, 50.0, 0.0), (150.0, -120.0, 10.0, 20.0),
                     (0.0, 240.0, 100.0, -30.0), (-100.0, 180.0, 0.0, 45.0)):
            retour = self.pont.depuis_controleur(*self.pont.vers_controleur(*pose))
            np.testing.assert_allclose(retour, pose, atol=1e-9)

    def test_conversion_coherente_avec_le_mgd(self):
        # le controleur annonce une pose ET des angles ; les deux doivent
        # designer le meme point une fois la conversion appliquee
        nu = DobotMagician()
        pont = PontReel(nu)
        pont.biais = (61.0, 0.0, 0.0)
        q = np.array([44.524, 40.587, 11.241, 0.0])
        rho = 135 * np.sin(np.radians(q[1])) + 147 * np.cos(np.radians(q[2])) + 61.0
        a = np.radians(q[0])
        ctrl = (rho * np.cos(a), rho * np.sin(a),
                135 * np.cos(np.radians(q[1])) - 147 * np.sin(np.radians(q[2])),
                q[0] + q[3])
        np.testing.assert_allclose(pont.depuis_controleur(*ctrl), nu.pose(q),
                                   atol=1e-6)

    # -- traduction d'un programme

    def _programme(self):
        r = Pilote(nom="t", outil="ventouse", vitesse=60, acceleration=300,
                   vitesse_articulaire=30)
        r.repli()
        r.movj(cible(220, 0, 60))
        r.movl(cible(220, 0, 20))
        r.ventouse(True)
        r.attendre(0.4)
        r.sortie(1, True)
        r.movc(cible(200, 60, 60), cible(160, 110, 60))
        r.repli()
        return r

    def test_televersement_produit_les_bonnes_commandes(self):
        r = self._programme()
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        idents = [i for i, _ in self.faux.file]
        for attendu in (CommandeDobot.PTP, CommandeDobot.ARC,
                        CommandeDobot.ATTENTE, CommandeDobot.ES_SORTIE,
                        CommandeDobot.VENTOUSE):
            self.assertIn(attendu, idents)

    def test_la_file_est_arretee_avant_d_etre_remplie(self):
        # sinon les commandes partiraient des leur arrivee
        r = self._programme()
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        idents = [i for i, _ in self.faux.immediates]
        self.assertLess(idents.index(CommandeDobot.FILE_ARRETER),
                        idents.index(CommandeDobot.FILE_VIDER))
        self.assertNotIn(CommandeDobot.FILE_DEMARRER, idents)

    def test_movj_articulaire_passe_en_mode_angle(self):
        r = Pilote(nom="t", outil="ventouse")
        r.repli()                                   # MOVJ articulaire
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        modes = [struct.unpack("<B4f", p)[0]
                 for i, p in self.faux.file if i == CommandeDobot.PTP]
        self.assertIn(PTP.MOVJ_ANGLE, modes)

    def test_movl_passe_en_mode_cartesien(self):
        r = Pilote(nom="t", outil="ventouse")
        r.movl(cible(220, 0, 60))
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        modes = [struct.unpack("<B4f", p)[0]
                 for i, p in self.faux.file if i == CommandeDobot.PTP]
        self.assertIn(PTP.MOVL_XYZ, modes)

    def test_entree_devient_un_trig(self):
        r = Pilote(nom="t", outil="ventouse")
        r.attendre_entree(5, True)
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        self.assertIn(CommandeDobot.TRIG, [i for i, _ in self.faux.file])

    def test_outil_suit_l_effecteur_monte(self):
        for nom, attendu in (("ventouse", CommandeDobot.VENTOUSE),
                             ("pince", CommandeDobot.PINCE),
                             ("laser", CommandeDobot.LASER)):
            faux = SerieSimulee()
            pont = PontReel(self.modele)
            pont.liaison = _liaison_simulee(faux)
            pont.biais = (61.0, 0.0, 0.0)
            r = Pilote(nom="t", outil=nom)
            r.actionner(True)
            pont.televerser(r.programme, r.q_depart, outil=nom)
            self.assertIn(attendu, [i for i, _ in faux.file])

    def test_programme_fautif_refuse(self):
        r = Pilote(nom="fautif", outil="ventouse")
        r.movl(cible(220, 0, -40))          # 40 mm sous le plan de pose
        with self.assertRaises(ErreurPont):
            self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        self.assertEqual(self.faux.file, [])

    def test_forcer_passe_outre(self):
        r = Pilote(nom="fautif", outil="ventouse")
        r.movl(cible(220, 0, -40))
        self.pont.televerser(r.programme, r.q_depart, forcer=True,
                             outil="ventouse")
        self.assertTrue(self.faux.file)

    def test_les_coordonnees_envoyees_sont_celles_du_controleur(self):
        r = Pilote(nom="t", outil="ventouse")
        r.movl(cible(220, 0, 20))
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        _mode, x, y, z, _r = struct.unpack(
            "<B4f", [p for i, p in self.faux.file if i == CommandeDobot.PTP][-1])
        attendu = self.pont.vers_controleur(220.0, 0.0, 20.0, 0.0)
        np.testing.assert_allclose((x, y, z), attendu[:3], atol=1e-3)
        # et surtout : ce n'est PAS la coordonnee du simulateur
        self.assertGreater(abs(z - 20.0), 50.0)

    def test_ports_es_configures_une_seule_fois(self):
        # le multiplexage est une commande IMMEDIATE : la configuration du
        # port doit etre en place avant que la file ne s'execute, pas au
        # moment ou le mouvement y arrive
        r = Pilote(nom="t", outil="ventouse")
        r.sortie(1, True)
        r.sortie(1, False)
        r.sortie(1, True)
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        multiplexages = [p for i, p in self.faux.immediates
                         if i == CommandeDobot.ES_MULTIPLEXAGE]
        self.assertEqual(len(multiplexages), 1)
        self.assertEqual(multiplexages[0][0], 1)        # port 1
        self.assertEqual(len([p for i, p in self.faux.file
                              if i == CommandeDobot.ES_SORTIE]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
