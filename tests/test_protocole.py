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
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import DobotMagician, Pilote, cible
from dobot_sim.liaison import (CommandeDobot, ErreurLiaison, ErreurTrame,
                               FilePleine, PTP, LiaisonDobot, Pose,
                               construire_trame,
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

    Avec `capacite`, la file refuse ce qui depasse, par une reponse vide,
    tant qu'elle est arretee. Demarree, elle execute tout instantanement.
    """

    is_open = True

    def __init__(self, pose=(208.9, 205.5, 73.9, 44.5),
                 angles=(44.5, 40.6, 11.2, 0.0), biais=(61.0, 0.0, 0.0),
                 erreur=(0.0, 0.0), capacite=None):
        self.pose = pose
        self.angles = angles
        self.biais = biais
        self.erreur = erreur
        self.capacite = capacite
        self.file = []              # (identifiant, parametres)
        self.immediates = []
        self.index = 0
        self.execute = 0            # index de la derniere commande executee
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
            if (self.capacite is not None
                    and self.index - self.execute >= self.capacite):
                return b""                  # file pleine : commande perdue
            self.index += 1
            self.file.append((ident, params))
            if self.demarree:
                self.execute = self.index
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
        if ident == CommandeDobot.CAPTEUR_IR:
            # lecture : params = (port, 0) ; ecriture : (actif, port, version)
            return bytes((getattr(self, "ir", 0),)) if len(params) == 2 else b""
        if ident == CommandeDobot.CAPTEUR_COULEUR:
            return bytes(getattr(self, "couleur", (0, 0, 0)))                 if not params else b""
        if ident == CommandeDobot.ES_ENTREE:
            return bytes((params[0] if params else 1, 1))
        if ident == CommandeDobot.FILE_DEMARRER:
            self.demarree = True
            self.execute = self.index
        elif ident == CommandeDobot.FILE_ARRETER:
            self.demarree = False
        elif ident == CommandeDobot.FILE_VIDER:
            self.file.clear()
            self.execute = self.index
        return b""


class ControleurFidele(SerieSimulee):
    """
    Controleur a file BORNEE, qui se vide au fil du temps.

    `SerieSimulee` est commode pour tester le protocole, mais sa file
    parait toujours vide : elle repond a GetQueuedCmdCurrentIndex par le
    dernier index accepte, et considere tout execute des le demarrage.
    Aucun probleme de debit ne peut s'y voir.

    Celui-ci se comporte comme la machine :
      - l'index courant est celui de la commande EN COURS ;
      - la file se consomme a `cadence` secondes par commande ;
      - pleine, elle refuse franchement (reponse vide) ou, si
        `silencieux`, repond un index valide et JETTE la commande ;
      - avec `muette_tous_les`, une commande sur N reste sans reponse,
        comme lorsque le lien serie rate une trame.
    """

    def __init__(self, capacite_file=12, cadence=0.002, silencieux=False,
                 muette_tous_les=0, **kw):
        super().__init__(capacite=None, **kw)
        self.capacite_file = capacite_file
        self.cadence = cadence
        self.silencieux = silencieux
        self.muette_tous_les = muette_tous_les
        self.jetees = 0
        self.muettes = 0
        self.occupation_max = 0
        self._t0 = None
        self._recues = 0

    def avancer(self, tout=False):
        """Le temps passe : le controleur execute sa file."""
        if tout:
            self.execute = len(self.file)
            return
        if not self.demarree:
            return
        if self._t0 is None:
            self._t0 = time.time()
        self.execute = min(len(self.file),
                           int((time.time() - self._t0) / self.cadence))

    def _traiter(self, ident, params, en_file):
        self.avancer()
        if en_file:
            self._recues += 1
            if (self.muette_tous_les
                    and self._recues % self.muette_tous_les == 0):
                self.muettes += 1
                return None                      # le lien a rate la reponse
            occupation = len(self.file) - self.execute
            self.occupation_max = max(self.occupation_max, occupation)
            if occupation >= self.capacite_file:
                if not self.silencieux:
                    return b""                   # refus visible
                # Mesure sur firmware 4.0.3 : la reponse fait bien huit
                # octets, mais l'index N'AVANCE PAS et la commande est
                # jetee. C'est exactement ce que fait la machine.
                self.jetees += 1
                return struct.pack("<Q", self.index)
            self.index += 1
            self.file.append((ident, params))
            return struct.pack("<Q", self.index)

        if ident == CommandeDobot.FILE_PLACE_RESTANTE:
            libre = self.capacite_file - (len(self.file) - self.execute)
            return struct.pack("<I", max(0, libre))
        if ident == CommandeDobot.FILE_INDEX_COURANT:
            return struct.pack("<Q", self.execute)
        if ident == CommandeDobot.FILE_DEMARRER:
            self.demarree = True
            self._t0 = time.time()
            return b""
        if ident == CommandeDobot.FILE_VIDER:
            self.file.clear()
            self.execute = 0
            self._t0 = None
            return b""
        return super()._traiter(ident, params, en_file)

    def write(self, trame):
        self.octets_ecrits += len(trame)
        ident, ctrl, params = decoder_trame(trame)
        en_file = bool(ctrl & 2)
        reponse = self._traiter(ident, params, en_file)
        if reponse is None:
            return len(trame)                    # muet : rien a lire
        self._sortie += construire_trame(ident, reponse, bool(ctrl & 1),
                                         en_file)
        return len(trame)


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

    def test_cadence_identique_non_renvoyee(self):
        r = Pilote(nom="t", outil="ventouse", vitesse=60, acceleration=300)
        for _ in range(3):
            r.movl(cible(220, 0, 60))
            r.movl(cible(220, 0, 20))
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        reglages = [i for i, _ in self.faux.file
                    if i == CommandeDobot.PARAMS_PTP_CARTESIEN]
        self.assertEqual(len(reglages), 1)

    # -- file du controleur trop petite pour le programme

    def _cycles(self, n):
        r = Pilote(nom="t", outil="ventouse", vitesse=60, acceleration=300,
                   vitesse_articulaire=30)
        for _ in range(n):
            r.repli()
            r.movj(cible(220, 0, 60))
            r.movl(cible(220, 0, 20), vitesse=30)
            r.ventouse(True)
            r.attendre(0.3)
            r.movl(cible(220, 0, 60))
            r.ventouse(False)
        r.repli()
        return r

    def _pont_capacite(self, capacite):
        faux = SerieSimulee(capacite=capacite)
        pont = PontReel(self.modele)
        pont.liaison = _liaison_simulee(faux)
        pont.biais = (61.0, 0.0, 0.0)
        return pont, faux

    def test_un_index_qui_n_avance_pas_est_une_file_pleine(self):
        """
        Releve sur la machine (firmware 4.0.3) : pleine, elle ne repond
        pas vide. Elle repond huit octets d'allure normale, mais l'index
        reste le meme -- et la commande est jetee.

        Sans cette detection, le pont croyait televerser un programme
        dont les trois quarts n'existaient nulle part : c'est la panne
        des 21 %.
        """
        faux = ControleurFidele(capacite_file=4, silencieux=True)
        li = _liaison_simulee(faux)
        for _ in range(4):
            li.ptp(PTP.MOVJ_XYZ, 200, 0, 0, 0)
        avant = li.dernier_index
        with self.assertRaises(FilePleine):
            li.ptp(PTP.MOVJ_XYZ, 200, 0, 0, 0)
        self.assertEqual(li.dernier_index, avant, "l'index ne doit pas bouger")
        self.assertEqual(len(faux.file), 4)

    def test_la_place_restante_est_lue_quand_le_firmware_repond(self):
        """
        GetQueuedCmdLeftSpace a ete RETIRE du protocole V1.1.5, mais le
        firmware 4.0.3 le sert encore : 32 file vide, 0 file pleine. On
        s'en sert quand il repond, et on s'en passe sinon -- sans le
        redemander a chaque commande.
        """
        faux = ControleurFidele(capacite_file=6)
        li = _liaison_simulee(faux)
        self.assertEqual(li.place_restante(), 6)
        li.ptp(PTP.MOVJ_XYZ, 200, 0, 0, 0)
        self.assertEqual(li.place_restante(), 5)

        muet = SerieSimulee()                    # firmware qui l'ignore
        li2 = _liaison_simulee(muet)
        self.assertIsNone(li2.place_restante())
        self.assertTrue(li2._place_restante_absente)
        avant = len(muet.immediates)
        self.assertIsNone(li2.place_restante())  # plus jamais redemande
        self.assertEqual(len(muet.immediates), avant)

    def test_file_pleine_signalee_par_la_liaison(self):
        faux = SerieSimulee(capacite=2)
        li = _liaison_simulee(faux)
        li.ptp(PTP.MOVJ_XYZ, 200, 0, 0, 0)
        li.ptp(PTP.MOVJ_XYZ, 200, 0, 0, 0)
        with self.assertRaises(FilePleine):
            li.ptp(PTP.MOVJ_XYZ, 200, 0, 0, 0)
        self.assertEqual(len(faux.file), 2)

    def test_le_surplus_part_pendant_l_execution(self):
        # Reference : le meme programme, joue jusqu'au bout. Le
        # televersement ne remplit plus la file au-dela de la marge -- il
        # faut donc demarrer pour que le reste s'ecoule, file illimitee ou
        # non.
        r = self._cycles(10)
        self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        self.pont.demarrer()
        if self.pont._alimentation is not None:
            self.pont._alimentation.join(5.0)
        attendu = list(self.faux.file)
        self.assertGreater(len(attendu), PontReel.MARGE_FILE,
                           "le programme de reference doit deborder la marge")

        pont, faux = self._pont_capacite(8)
        final = pont.televerser(r.programme, r.q_depart, outil="ventouse")
        self.assertEqual(len(faux.file), 8)          # rien n'est perdu...
        self.assertFalse(pont.termine())
        self.assertEqual(final, len(attendu))        # ...et la fin est connue
        pont.demarrer()
        pont._alimentation.join(5.0)
        self.assertEqual(faux.file, attendu)         # tout, dans l'ordre
        self.assertTrue(pont.termine())

    def test_le_televersement_ne_deborde_pas_la_file(self):
        """
        Le televersement s'arrete a la marge, sans rien demander au
        controleur.

        C'est la correction de la panne des 21 % : on ne peut pas se fier
        a un refus, car la machine, pleine, repond un index valide et
        JETTE la commande. On compte donc, au lieu de demander.
        """
        r = self._cycles(10)
        n = self.pont.televerser(r.programme, r.q_depart, outil="ventouse")
        self.assertGreater(n, PontReel.MARGE_FILE)
        self.assertEqual(len(self.faux.file), PontReel.MARGE_FILE)
        self.assertEqual(len(self.pont._reste), n - PontReel.MARGE_FILE)
        self.assertLessEqual(PontReel.MARGE_FILE, PontReel.CAPACITE_FILE)

    def test_un_long_programme_passe_en_entier(self):
        """
        La panne des 21 %, et sa non-regression.

        Un programme de 150 commandes s'arretait a 32 sur la machine du
        labo -- la profondeur de sa file. Le pont detectait alors une file
        pleine a une reponse VIDE ; la machine, pleine, repond un index
        valide et jette la commande. Le pont croyait donc avoir tout
        envoye, ne gardait rien a renvoyer, et le robot s'arretait au bout
        de ce que la file contenait vraiment.

        Le controleur simule des autres tests ne pouvait pas le montrer :
        il repond a GetQueuedCmdCurrentIndex par le dernier index ACCEPTE
        et considere tout execute des le demarrage -- sa file parait
        toujours vide. Celui-ci se comporte comme la machine, et le test
        est joue sous les DEUX hypotheses : qu'elle refuse franchement, ou
        qu'elle jette en silence. Dans les deux cas, tout doit passer.
        """
        for silencieux in (False, True):
            r = self._cycles(10)
            # la profondeur relevee sur la machine : la marge se tient en
            # dessous, et c'est ce qui protege du controleur silencieux
            faux = ControleurFidele(capacite_file=PontReel.CAPACITE_FILE,
                                    silencieux=silencieux)
            pont = PontReel(self.modele)
            pont.liaison = _liaison_simulee(faux)
            pont.biais = (61.0, 0.0, 0.0)

            attendu = pont.televerser(r.programme, r.q_depart, outil="ventouse")
            pont.demarrer()
            pont._alimentation.join(20.0)
            faux.avancer(tout=True)

            cas = "silencieux" if silencieux else "franc"
            self.assertEqual(faux.jetees, 0,
                             "controleur %s : %d commande(s) jetee(s)"
                             % (cas, faux.jetees))
            self.assertEqual(len(faux.file), attendu,
                             "controleur %s : %d commandes recues sur %d"
                             % (cas, len(faux.file), attendu))
            # La propriete qui compte n'est pas « on remplit peu », c'est
            # « on ne deborde jamais ». Quand le firmware annonce sa place
            # restante, le pont l'utilise et remplit donc plus franchement
            # que la marge : c'est voulu, et sans danger tant que la garde
            # est respectee.
            self.assertLessEqual(faux.occupation_max,
                                 PontReel.CAPACITE_FILE - PontReel.GARDE_FILE,
                                 "controleur %s : la file a ete remplie a %d"
                                 % (cas, faux.occupation_max))
            self.assertEqual(pont._incident, "")

    def test_une_file_plus_courte_que_prevu_fait_reculer_la_marge(self):
        """
        La profondeur de file n'est pas publiee par le protocole, et rien
        ne garantit qu'elle vaille 32 sur tous les firmwares. Quand le
        controleur REFUSE, il nous renseigne : on en tient compte au lieu
        de retomber sur le meme mur.

        Face a un controleur qui jette en silence, en revanche, aucun
        repli n'est possible : la marge doit alors etre choisie sous la
        profondeur reelle. C'est tout l'objet de CAPACITE_FILE.
        """
        r = self._cycles(10)
        courte = 9
        self.assertLess(courte, PontReel.MARGE_FILE)
        faux = ControleurFidele(capacite_file=courte, silencieux=False)
        pont = PontReel(self.modele)
        pont.liaison = _liaison_simulee(faux)
        pont.biais = (61.0, 0.0, 0.0)

        attendu = pont.televerser(r.programme, r.q_depart, outil="ventouse")
        self.assertLess(pont.marge, PontReel.MARGE_FILE, "marge non reduite")
        pont.demarrer()
        pont._alimentation.join(20.0)
        faux.avancer(tout=True)
        self.assertEqual(faux.jetees, 0)
        self.assertEqual(len(faux.file), attendu)

    def test_une_trame_perdue_n_abandonne_pas_le_programme(self):
        """
        Un lien serie rate une reponse de temps en temps. Abandonner le
        reste du programme pour autant laisserait le robot s'arreter au
        milieu du cycle, sans rien dire.
        """
        r = self._cycles(10)
        faux = ControleurFidele(capacite_file=12, muette_tous_les=11)
        pont = PontReel(self.modele)
        pont.liaison = _liaison_simulee(faux)
        pont.liaison.delai = 0.05
        pont.biais = (61.0, 0.0, 0.0)

        attendu = pont.televerser(r.programme, r.q_depart, outil="ventouse")
        pont.demarrer()
        pont._alimentation.join(30.0)
        faux.avancer(tout=True)
        self.assertGreater(faux.muettes, 0, "aucune reponse perdue : test vain")
        self.assertEqual(len(faux.file), attendu)
        self.assertEqual(pont._incident, "")

    def test_vidage_abandonne_le_surplus(self):
        r = self._cycles(10)
        pont, faux = self._pont_capacite(8)
        pont.televerser(r.programme, r.q_depart, outil="ventouse")
        pont.liaison.vider_file()                    # pilotage, urgence...
        self.assertFalse(pont.programme_en_file())
        pont.demarrer()
        if pont._alimentation is not None:
            pont._alimentation.join(5.0)
        self.assertEqual(faux.file, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
