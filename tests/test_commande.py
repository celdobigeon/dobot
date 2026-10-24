#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de la commande temps reel (dobot_sim.Commande).

Aucun materiel : le robot est un controleur SIMULE qui bouge vraiment --
il execute sa file au fil du temps, file bornee a 32 commandes dont l'index
se fige quand elle est pleine, et ses angles suivent les commandes
executees. La fenetre est soit une fausse passerelle, soit la vraie
FenetrePrincipale, hors ecran, branchee sur ce meme controleur simule.

PAR SECURITE, ce module :
  * isole la passerelle (DOBOT_SIM_SERVICE propre au test) : aucune
    commande ne peut atteindre un simulateur ouvert sur le poste, qui
    tiendrait peut-etre un vrai robot ;
  * interdit l'ouverture de tout port serie pendant ses tests.
"""

import io
import json
import math
import os
import struct
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

os.environ["DOBOT_SIM_SERVICE"] = "test_commande_{}".format(os.getpid())
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_sim import Commande, DobotMagician, ErreurExecution, Pilote, cible  # noqa: E402
from dobot_sim import commande as cmd                                    # noqa: E402
from dobot_sim.commande import (commandes_robot, echantillonner_course,  # noqa: E402
                                planifier, prevoir_cp, vitesse_pour_duree)
from dobot_sim.liaison import PTP, CommandeDobot, LiaisonDobot           # noqa: E402
from dobot_sim.reel import PontReel                                      # noqa: E402
from dobot_sim.trajectory import ProfilArret, ProfilDemarrage            # noqa: E402

from test_protocole import SerieSimulee, _liaison_simulee                # noqa: E402

VENTOUSE = 62.0
BIAIS = (61.0, 0.0, 0.0)
_patch_port = None


def setUpModule():
    global _patch_port

    def interdit(self):
        raise AssertionError("ouverture d'un vrai port serie interdite "
                             "dans les tests ({})".format(self.port))
    _patch_port = mock.patch.object(LiaisonDobot, "ouvrir", interdit)
    _patch_port.start()


def tearDownModule():
    _patch_port.stop()


def _modele():
    m = DobotMagician()
    m.geo.lt = VENTOUSE
    return m


# ---------------------------------------------------------------------------
# Un controleur simule qui bouge
# ---------------------------------------------------------------------------

class ControleurMobile(SerieSimulee):
    """
    Controleur a file bornee qui EXECUTE ses commandes : ses angles suivent.

      - file de `capacite` commandes ; pleine, elle repond un index valide
        mais FIGE et jette la commande (firmware 3.7.0 et 4.0.3) ;
      - GetQueuedCmdLeftSpace (247) repond la place restante ;
      - demarree, elle consomme une commande toutes les `cadence` s ;
      - l'index courant est celui de la derniere commande executee ;
      - PTP, ARC et CP deplacent le bras : les coordonnees du controleur
        sont reconverties en angles du modele.
    """

    def __init__(self, modele=None, q=(0.0, 15.0, 30.0, 0.0), biais=BIAIS,
                 capacite=32, cadence=0.0005):
        super().__init__(biais=biais)
        self.modele = modele or _modele()
        self.q = [float(v) for v in q]
        self.capacite_file = capacite
        self.cadence = cadence
        self.attente = []            # (index, ident, params) a executer
        self.execute = 0
        self.alarmes_actives = []
        self.jetees = 0
        self._t = time.time()

    # -- conversions de repere, comme le vrai controleur
    def pose_controleur(self):
        x, y, z, r = self.modele.pose(self.q)
        g = self.modele.geo
        rho = math.hypot(x, y)
        k = (rho - g.l3 + self.biais[0]) / rho
        return x * k, y * k, z + g.lt - g.d1 + self.biais[2], r

    def _vers_angles(self, x, y, z, r):
        g = self.modele.geo
        rho = math.hypot(x, y)
        k = (rho + g.l3 - self.biais[0]) / rho
        return list(self.modele.mgi_pose(x * k, y * k,
                                         z - g.lt + g.d1 - self.biais[2], r,
                                         tolerant=True))

    def _appliquer(self, ident, params):
        if ident == CommandeDobot.PTP:
            mode, x, y, z, r = struct.unpack("<B4f", params[:17])
            if mode in (PTP.MOVJ_ANGLE, PTP.MOVL_ANGLE):
                self.q = [x, y, z, r]
            else:
                self.q = self._vers_angles(x, y, z, r)
        elif ident == CommandeDobot.CP:
            _m, x, y, z, _v = struct.unpack("<B4f", params[:17])
            r = self.q[0] + self.q[3]
            self.q = self._vers_angles(x, y, z, r)
        elif ident == CommandeDobot.ARC:
            v = struct.unpack("<8f", params[:32])
            self.q = self._vers_angles(*v[4:])

    def avancer(self):
        maintenant = time.time()
        if not self.demarree:
            self._t = maintenant
            return
        while self.attente and maintenant - self._t >= self.cadence:
            idx, ident, params = self.attente.pop(0)
            self._appliquer(ident, params)
            self.execute = idx
            self._t += self.cadence
        if not self.attente:
            self._t = maintenant

    def _traiter(self, ident, params, en_file):
        self.avancer()
        if en_file:
            if len(self.attente) >= self.capacite_file:
                self.jetees += 1
                return struct.pack("<Q", self.index)       # index fige
            self.index += 1
            self.file.append((ident, params))
            self.attente.append((self.index, ident, params))
            return struct.pack("<Q", self.index)
        if ident == CommandeDobot.FILE_PLACE_RESTANTE:
            return struct.pack("<I", self.capacite_file - len(self.attente))
        if ident == CommandeDobot.FILE_INDEX_COURANT:
            return struct.pack("<Q", self.execute)
        if ident == CommandeDobot.POSE:
            return struct.pack("<8f", *self.pose_controleur(), *self.q)
        if ident == CommandeDobot.ALARMES:
            octets = bytearray(16)
            for a in self.alarmes_actives:
                octets[a // 8] |= 1 << (a % 8)
            return bytes(octets)
        if ident == CommandeDobot.EFFACER_ALARMES:
            self.alarmes_actives = []
            return b""
        self.immediates.append((ident, params))
        if ident == CommandeDobot.FILE_DEMARRER:
            self.demarree = True
            self._t = time.time()
            return b""
        if ident in (CommandeDobot.FILE_ARRETER, CommandeDobot.FILE_ARRET_FORCE):
            self.demarree = False
            return b""
        if ident == CommandeDobot.FILE_VIDER:
            self.attente.clear()
            self.execute = self.index
            return b""
        self.immediates.pop()
        return super()._traiter(ident, params, False)

    # -- inspection
    def ids_file(self, depuis=0):
        return [i for i, _p in self.file[depuis:]]


def _pont_simule(faux, modele):
    pont = PontReel(modele)
    pont.liaison = _liaison_simulee(faux)
    pont.biais = BIAIS
    return pont


class _DirectMixin:
    """Commande(robot=True, simulateur=False) branchee sur ControleurMobile."""

    def ouvrir(self, q=(0.0, 15.0, 30.0, 0.0), **kw):
        self.faux = ControleurMobile(q=q)
        modele_pont = _modele()
        self.pont = _pont_simule(self.faux, modele_pont)

        def liaison(robot, port):
            self.pont.robot = robot
            return self.pont
        p = mock.patch("dobot_sim.pilote._liaison_directe", liaison)
        p.start()
        self.addCleanup(p.stop)
        f = mock.patch("dobot_sim.pilote._fermer_liaison_directe", lambda: None)
        f.start()
        self.addCleanup(f.stop)
        with redirect_stdout(io.StringIO()):
            self.robot = Commande(robot=True, simulateur=False, **kw)
        self.addCleanup(self.robot.fermer)
        return self.robot


# ---------------------------------------------------------------------------
# Fonctions de calcul : vitesses, echantillonnage, prevision
# ---------------------------------------------------------------------------

class TestCalculs(unittest.TestCase):

    def test_vitesse_pour_duree_inverse_le_trapeze(self):
        for L, a, arrets in ((120.0, 400.0, 2), (50.0, 200.0, 1), (80.0, 400.0, 0)):
            for v in (20.0, 45.0, 80.0):
                T = L / v + arrets * v / (2 * a)
                self.assertAlmostEqual(vitesse_pour_duree(L, T, a, arrets), v,
                                       places=6)

    def test_vitesse_pour_duree_intenable(self):
        # 120 mm en 0,5 s, depart et arrivee arretes, 400 mm/s2 : impossible
        self.assertIsNone(vitesse_pour_duree(120.0, 0.5, 400.0, 2))
        self.assertIsNone(vitesse_pour_duree(0.0, 1.0, 400.0, 2))

    def test_profils_demarrage_et_arret(self):
        for cls, v0, v1 in ((ProfilDemarrage, 0.0, 1.0), (ProfilArret, 1.0, 0.0)):
            p = cls.adapte(100.0, 50.0, 400.0)
            self.assertAlmostEqual(p.duree, 100.0 / 50.0 + 50.0 / 800.0)
            self.assertAlmostEqual(p.s(0.0), 0.0)
            self.assertAlmostEqual(p.s(p.duree), 1.0)
            # vitesse aux bouts : nulle d'un cote, palier de l'autre
            self.assertAlmostEqual(100.0 * p.ds(0.0), 50.0 * v0, places=6)
            self.assertAlmostEqual(100.0 * p.ds(p.duree), 50.0 * v1, places=6)
            # s croissant
            ts = np.linspace(0, p.duree, 200)
            self.assertTrue(np.all(np.diff([p.s(t) for t in ts]) >= -1e-12))

    def _traj(self, *etapes, depart=(220, -60, 80)):
        m = _modele()
        q0 = m.mgi(cible(*depart))
        p = Pilote(robot=m, vitesse=50, acceleration=400, profil="trapeze",
                   depart=q0)
        p.regler_longueur_outil(VENTOUSE)
        for f in etapes:
            f(p)
        return m, q0, p

    def test_echantillonnage_d_une_droite(self):
        m, q0, p = self._traj(lambda p: p.movl(cible(220, 60, 80), duree=3.0))
        depart, portions = echantillonner_course(p.trajectoire(), 0, 0)
        self.assertEqual(len(portions), 1)
        pts = portions[0]
        # droite : le point a 1 mm, puis la seule extremite
        self.assertEqual(len(pts), 2)
        np.testing.assert_allclose(pts[0], (220, -59, 80), atol=1e-6)
        np.testing.assert_allclose(pts[-1], (220, 60, 80), atol=1e-6)

    def test_echantillonnage_d_un_arc_tous_les_5_mm(self):
        m, q0, p = self._traj(lambda p: p.movc(cible(280, 0, 80),
                                               cible(220, 60, 80), duree=3.0))
        depart, portions = echantillonner_course(p.trajectoire(), 0, 0)
        pts = np.array(portions[0])
        d = np.linalg.norm(np.diff(np.vstack([depart, pts]), axis=0), axis=1)
        self.assertAlmostEqual(d[0], 1.0, places=2)           # premier a 1 mm
        self.assertTrue(np.all(d[1:-1] <= 5.0 + 1e-6))
        self.assertTrue(np.all(d[1:-1] >= 3.5))
        # tous sur le cercle de centre (220, 0), rayon 60
        rayons = np.hypot(pts[:, 0] - 220.0, pts[:, 1])
        np.testing.assert_allclose(rayons, 60.0, atol=0.05)
        np.testing.assert_allclose(pts[-1], (220, 60, 80), atol=1e-6)

    def test_un_coin_tombe_exactement_sur_la_cible(self):
        m, q0, p = self._traj(lambda p: p.movl(cible(220, 0, 80), duree=1.0),
                              lambda p: p.movl(cible(260, 0, 80), duree=1.0))
        _d, portions = echantillonner_course(p.trajectoire(), 0, 1)
        np.testing.assert_allclose(portions[0][-1], (220, 0, 80), atol=1e-6)
        np.testing.assert_allclose(portions[1][-1], (260, 0, 80), atol=1e-6)

    def test_prevision_d_une_droite(self):
        # droite de 120 mm a 80 mm/s, 400 mm/s2 : trapeze exact
        depart = np.array([0.0, 0.0, 0.0])
        pts = [[np.array([1.0, 0, 0]), np.array([120.0, 0, 0])]]
        durees, courbe = prevoir_cp(depart, pts, [80.0], 400.0)
        self.assertAlmostEqual(durees[0], 120 / 80 + 80 / 400, places=6)
        self.assertFalse(courbe[0])

    def test_prevision_calee_sur_l_arc_mesure(self):
        """Arc de rayon 60 tous les 5 mm, 80 demandes, 200 mm/s2 : ~62 mesures."""
        n = int(round(math.pi * 60 / 5.0))
        a = np.linspace(0, math.pi, n + 1)
        P = [np.array([60 * math.sin(t), 60 * math.cos(t), 0.0]) for t in a]
        durees, courbe = prevoir_cp(P[0], [P[1:]], [80.0], 200.0)
        self.assertTrue(courbe[0])
        v_moy = math.pi * 60 / durees[0]
        self.assertGreater(v_moy, 55.0)
        self.assertLess(v_moy, 62.5)


# ---------------------------------------------------------------------------
# Le plan : point a point, continu, repli, durees
# ---------------------------------------------------------------------------

class TestPlan(unittest.TestCase):

    def setUp(self):
        self.m = _modele()
        self.q0 = self.m.mgi(cible(220, -60, 80))

    def _pilote(self):
        p = Pilote(robot=self.m, vitesse=50, acceleration=400,
                   vitesse_articulaire=50, acceleration_articulaire=400,
                   profil="trapeze", depart=self.q0)
        p.regler_longueur_outil(VENTOUSE)
        return p

    def test_une_seule_droite_part_en_point_a_point(self):
        p = self._pilote()
        p.movl(cible(220, 60, 80), duree=3.0)
        plan = planifier(p.programme, self.m, self.q0)
        self.assertEqual(plan.etapes, [{"type": "ptp", "index": 0}])
        ins = plan.programme.instructions[0]
        self.assertEqual((ins.mode_temps, ins.profil), ("vitesse", "trapeze"))
        # la vitesse choisie fait durer le trapeze du controleur 3 s
        self.assertAlmostEqual(120 / ins.vitesse + ins.vitesse / 400, 3.0, places=6)
        self.assertAlmostEqual(plan.duree, 3.0, places=3)

    def test_plusieurs_cibles_en_continu(self):
        p = self._pilote()
        p.movl(cible(220, 60, 80), duree=4.0)
        p.movl(cible(260, 60, 80), duree=1.0)
        p.movl(cible(260, -60, 80))
        plan = planifier(p.programme, self.m, self.q0)
        self.assertEqual([e["type"] for e in plan.etapes], ["cp"])
        portions = plan.etapes[0]["portions"]
        self.assertEqual(len(portions), 3)
        # aller de 120 mm en 4 s : ~30 mm/s (cf. mesure : 4,03 s pour 4,00)
        self.assertAlmostEqual(portions[0]["vitesse"], 30.0, delta=1.5)
        # derniere : duree omise, vitesse par defaut
        self.assertAlmostEqual(portions[2]["vitesse"], 50.0, places=6)
        profils = [i.profil for i in plan.programme.instructions]
        self.assertEqual(profils, ["demarrage", "lineaire", "arret"])
        self.assertFalse(plan.avertissements)
        self.assertAlmostEqual(plan.programme.instructions[0].duree, 4.0, delta=0.05)

    def test_continu_false_s_arrete_partout(self):
        p = self._pilote()
        p.movl(cible(220, 60, 80), duree=2.0, lissage=50)
        p.movl(cible(260, 60, 80), duree=1.0)
        plan = planifier(p.programme, self.m, self.q0, continu=False)
        self.assertEqual(plan.etapes, [{"type": "ptp", "index": 0},
                                       {"type": "ptp", "index": 1}])
        self.assertTrue(any("lissage ignore" in a for a in plan.annonces))
        self.assertTrue(all(i.lissage == 0 for i in plan.programme.instructions))

    def test_rotation_de_l_outil_repasse_en_point_a_point(self):
        p = self._pilote()
        p.movl(cible(220, 0, 80), duree=1.0)
        p.movl(cible(220, 60, 80), duree=1.0)
        p.movl(cible(260, 60, 80, rz=45), duree=1.0)      # l'outil tourne
        p.movl(cible(260, 0, 80, rz=45), duree=1.0)
        p.movl(cible(260, -40, 80, rz=45), duree=1.0)
        plan = planifier(p.programme, self.m, self.q0)
        self.assertEqual([(e["type"], e.get("index")) for e in plan.etapes],
                         [("cp", None), ("ptp", 2), ("cp", None)])
        self.assertTrue(any("3 MOVL" in a and "tourne" in a
                            for a in plan.annonces))

    def test_duree_intenable_annoncee_avec_la_duree_realiste(self):
        p = self._pilote()
        p.movl(cible(220, 60, 80), duree=0.4)        # 120 mm en 0,4 s
        p.movl(cible(260, 60, 80), duree=1.0)
        plan = planifier(p.programme, self.m, self.q0)
        self.assertTrue(plan.avertissements)
        self.assertIn("1 MOVL", plan.avertissements[0])
        self.assertIn("demandees", plan.avertissements[0])
        # le simulateur joue la duree realiste, pas l'impossible
        self.assertGreater(plan.programme.instructions[0].duree, 0.9)

    def test_arc_trop_rapide_limite_par_la_courbure(self):
        p = self._pilote()
        p.movc(cible(280, 0, 80), cible(220, 60, 80), duree=1.2)   # ~157 mm/s
        plan = planifier(p.programme, self.m, self.q0)
        self.assertEqual(plan.etapes[0]["type"], "cp")
        self.assertTrue(any("courbure" in a or "plafonnee" in a
                            for a in plan.avertissements))

    def test_point_a_point_jamais_au_dela_des_vitesses_mesurees(self):
        for kw in ({"duree": 0.1}, {"vitesse": 300}):
            p = self._pilote()
            p.movl(cible(220, 60, 80), **kw)
            plan = planifier(p.programme, self.m, self.q0)
            self.assertEqual(plan.programme.instructions[0].vitesse, 125.0)
            self.assertTrue(plan.avertissements)

    def test_le_lissage_garde_la_duree_des_portions(self):
        p = self._pilote()
        p.movl(cible(220, 60, 80), duree=2.0, lissage=60)
        p.movl(cible(280, 60, 80), duree=1.0, lissage=60)
        p.movl(cible(280, -20, 80), duree=1.5)
        plan = planifier(p.programme, self.m, self.q0)
        # les portions durent ce qui est demande, recouvrements compris :
        # a la tolerance de la prevision pres, la somme est conservee
        self.assertAlmostEqual(plan.duree, 4.5, delta=0.35)
        self.assertTrue(plan.trajectoire.lissage_actif)

    def test_movj_en_continu_suit_son_chemin(self):
        p = self._pilote()
        p.movj(cible(240, 40, 80), duree=1.0)
        p.movj(cible(200, 80, 60), duree=1.0)
        plan = planifier(p.programme, self.m, self.q0)
        self.assertEqual(plan.etapes[0]["type"], "cp")
        tous = [pt for po in plan.etapes[0]["portions"] for pt in po["points"]]
        np.testing.assert_allclose(tous[-1], (200, 80, 60), atol=1e-4)

    def test_vitesse_des_movj_en_continu(self):
        # la vitesse articulaire d'un MOVJ doit compter en continu aussi
        durees = []
        for v in (10, 100):
            p = self._pilote()
            p.movj(cible(240, 40, 80), vitesse=v)
            p.movj(cible(200, 80, 60), vitesse=v)
            plan = planifier(p.programme, self.m, self.q0)
            self.assertEqual(plan.etapes[0]["type"], "cp")
            durees.append(plan.programme.construire(self.m, self.q0).duree)
        self.assertGreater(durees[0], 1.5 * durees[1])

    def test_auto_collision_bloquante(self):
        from dobot_sim.reel import BLOQUANTS_TELEVERSEMENT
        message = "auto-collision : bras arriere contre deport du poignet"
        self.assertTrue(any(m in message for m in BLOQUANTS_TELEVERSEMENT))

    def test_commandes_robot_dans_le_bon_ordre(self):
        p = self._pilote()
        p.movl(cible(220, 60, 80), duree=4.0)
        p.movl(cible(260, 60, 80), duree=1.0)
        p.actionner(True)
        p.movl(cible(260, 60, 40), duree=1.0)
        plan = planifier(p.programme, self.m, self.q0)
        faux = ControleurMobile()
        pont = _pont_simule(faux, self.m)
        pont.televerser_commandes(lambda: commandes_robot(
            pont, plan.programme, plan.etapes, "ventouse"))
        ids = faux.ids_file()
        # CPParams en file, CP..., CPParams (nouvelle vitesse), CP..., ventouse,
        # vitesse PTP, PTP
        self.assertEqual(ids[0], CommandeDobot.PARAMS_CP)
        i_ventouse = ids.index(CommandeDobot.VENTOUSE)
        self.assertEqual(ids[1:i_ventouse].count(CommandeDobot.PARAMS_CP), 1)
        self.assertTrue(all(i in (CommandeDobot.CP, CommandeDobot.PARAMS_CP)
                            for i in ids[:i_ventouse]))
        self.assertEqual(ids[i_ventouse + 1:], [CommandeDobot.PARAMS_PTP_CARTESIEN,
                                                CommandeDobot.PTP])
        # junctionVel mis en file : 30 puis ~40 mm/s, realTimeTrack = 0
        params = [struct.unpack("<3fB", p_) for i, p_ in faux.file
                  if i == CommandeDobot.PARAMS_CP]
        self.assertAlmostEqual(params[0][1], plan.etapes[0]["portions"][0]["vitesse"],
                               places=3)
        self.assertAlmostEqual(params[0][0], 400.0)
        self.assertEqual({p_[3] for p_ in params}, {0})
        # les points CP sont en coordonnees du CONTROLEUR, mode absolu
        cps = [struct.unpack("<B4f", p_) for i, p_ in faux.file
               if i == CommandeDobot.CP]
        self.assertEqual({c[0] for c in cps}, {1})
        attendu = pont.vers_controleur(260, 60, 80, 0)
        np.testing.assert_allclose(cps[-1][1:4], attendu[:3], atol=1e-3)


# ---------------------------------------------------------------------------
# Resolution des cibles
# ---------------------------------------------------------------------------

class TestCibles(unittest.TestCase):

    def test_liste_ou_cible(self):
        E = Commande._est_liste
        A = cible(220, 0, 80)
        self.assertFalse(E(A))
        self.assertFalse(E((220, 0, 80)))
        self.assertFalse(E([220, 0, 80, 10]))
        self.assertFalse(E(A.tolist()))             # matrice ecrite en listes
        self.assertTrue(E([A, A]))
        self.assertTrue(E([(A, 1.0), A]))
        self.assertTrue(E([(220, 0, 80), (220, 50, 80)]))
        self.assertTrue(E([((220, 0, 80), 0.5)]))

    def test_element(self):
        A = cible(220, 0, 80)
        self.assertEqual(Commande._element((A, 1.5))[1], 1.5)
        self.assertIsNone(Commande._element(A)[1])
        c, d = Commande._element(((220, 0, 80), 2))
        self.assertEqual((tuple(c), d), ((220, 0, 80), 2.0))
        c, d = Commande._element((220, 0, 80))
        self.assertIsNone(d)

    def test_liste_d_arcs(self):
        A, B = cible(260, 0, 80), cible(220, 40, 80)
        self.assertTrue(Commande._est_liste_arcs([(A, B, 1.5), (A, B)]))
        self.assertFalse(Commande._est_liste_arcs((260, 0, 80)))
        self.assertFalse(Commande._est_liste_arcs(A))


# ---------------------------------------------------------------------------
# Chemin direct : le script tient la liaison
# ---------------------------------------------------------------------------

class TestDirect(_DirectMixin, unittest.TestCase):

    def test_les_deux_cibles_absentes_refusees(self):
        with self.assertRaises(ValueError):
            Commande(simulateur=False, robot=False)

    def test_connexion_regle_le_ratio_et_lit_la_pose(self):
        robot = self.ouvrir()
        ratio = [p for i, p in self.faux.immediates
                 if i == CommandeDobot.PARAMS_PTP_COMMUNS]
        self.assertEqual(struct.unpack("<2f", ratio[-1]), (100.0, 100.0))
        np.testing.assert_allclose(robot.angles(), (0, 15, 30, 0), atol=1e-4)
        attendu = _modele().pose([0, 15, 30, 0])
        np.testing.assert_allclose(robot.pose(), attendu, atol=1e-3)

    def test_movl_est_synchrone_et_arrive(self):
        robot = self.ouvrir()
        sortie = io.StringIO()
        with redirect_stdout(sortie):
            robot.movl(cible(220, 30, 80))
        # rend la main une fois arrive, sans afficher de progression
        self.assertEqual(sortie.getvalue(), "")
        self.assertFalse(self.faux.attente)
        np.testing.assert_allclose(robot.pose(), (220, 30, 80, 0), atol=0.01)
        ids = self.faux.ids_file()
        self.assertEqual(ids[-2:], [CommandeDobot.PARAMS_PTP_CARTESIEN,
                                    CommandeDobot.PTP])
        mode = struct.unpack("<B4f", self.faux.file[-1][1])[0]
        self.assertEqual(mode, PTP.MOVL_XYZ)

    def test_composantes_omises_reprises(self):
        robot = self.ouvrir()
        robot.movl(cible(220, 30, 80))
        robot.movl(z=50)
        np.testing.assert_allclose(robot.pose(), (220, 30, 50, 0), atol=0.01)
        robot.movj(x=240)
        np.testing.assert_allclose(robot.pose(), (240, 30, 50, 0), atol=0.01)

    def test_repli_en_angles(self):
        robot = self.ouvrir(q=_modele().mgi(cible(220, 30, 80)))
        robot.repli()
        mode, *q = struct.unpack("<B4f", self.faux.file[-1][1])
        self.assertEqual(mode, PTP.MOVJ_ANGLE)
        np.testing.assert_allclose(q, (0, 15, 30, 0), atol=1e-5)
        np.testing.assert_allclose(robot.angles(), (0, 15, 30, 0), atol=1e-4)

    def test_plusieurs_cibles_continu_en_cp(self):
        robot = self.ouvrir()
        robot.movl(cible(220, -60, 80))
        n0 = len(self.faux.file)
        robot.movl([(cible(220, 60, 80), 2.0), (cible(260, 60, 80), 0.8),
                    cible(260, 0, 80)])
        ids = self.faux.ids_file(n0)
        self.assertNotIn(CommandeDobot.PTP, ids)
        self.assertEqual(ids[0], CommandeDobot.PARAMS_CP)
        # premier point a 1 mm du depart, en coordonnees du controleur
        p1 = struct.unpack("<B4f", self.faux.file[n0 + 1][1])[1:4]
        conv = PontReel(_modele())
        conv.biais = BIAIS
        x, y, z, _r = conv.depuis_controleur(*p1, 0)
        self.assertAlmostEqual(math.dist((x, y, z), (220, -60, 80)), 1.0,
                               delta=0.01)
        np.testing.assert_allclose(robot.pose()[:3], (260, 0, 80), atol=0.01)

    def test_continu_false_reste_en_point_a_point(self):
        robot = self.ouvrir()
        n0 = len(self.faux.file)
        robot.movl([cible(220, -40, 80), cible(220, 40, 80)], continu=False)
        ids = self.faux.ids_file(n0)
        self.assertEqual(ids.count(CommandeDobot.PTP), 2)
        self.assertNotIn(CommandeDobot.CP, ids)

    def test_rotation_annoncee_et_point_a_point(self):
        robot = self.ouvrir()
        robot.movl(cible(220, -60, 80))
        n0 = len(self.faux.file)
        sortie = io.StringIO()
        with redirect_stdout(sortie):
            robot.movl([cible(220, 0, 80), cible(220, 60, 80),
                        cible(220, 60, 80, rz=30), cible(250, 60, 80, rz=30),
                        cible(250, 0, 80, rz=30)])
        self.assertIn("tourne", sortie.getvalue())
        ids = self.faux.ids_file(n0)
        i_ptp = ids.index(CommandeDobot.PTP)
        self.assertIn(CommandeDobot.CP, ids[:i_ptp])
        self.assertIn(CommandeDobot.CP, ids[i_ptp:])
        self.assertAlmostEqual(robot.pose()[3], 30.0, delta=0.05)

    def test_arc_et_liste_d_arcs(self):
        robot = self.ouvrir()
        robot.movl(cible(220, -60, 80))
        robot.movc(cible(280, 0, 80), cible(220, 60, 80), duree=3.0)
        np.testing.assert_allclose(robot.pose()[:3], (220, 60, 80), atol=0.01)
        n0 = len(self.faux.file)
        robot.movc([(cible(160, 0, 80), cible(220, -60, 80), 3.0),
                    (cible(250, -30, 80), cible(220, 0, 80))])
        ids = self.faux.ids_file(n0)
        self.assertGreater(ids.count(CommandeDobot.CP), 30)
        self.assertNotIn(CommandeDobot.ARC, ids)
        np.testing.assert_allclose(robot.pose()[:3], (220, 0, 80), atol=0.01)

    def test_longue_trajectoire_passe_entiere_par_la_file_de_32(self):
        robot = self.ouvrir()
        robot.movl(cible(220, -60, 80))
        n0 = len(self.faux.file)
        robot.movc([(cible(280, 0, 80), cible(220, 60, 80), 2.5),
                    (cible(160, 0, 80), cible(220, -60, 80), 2.5)])
        envoyees = self.faux.ids_file(n0)
        self.assertGreater(len(envoyees), 64)
        self.assertEqual(self.faux.jetees, 0)
        self.assertFalse(self.faux.attente)
        np.testing.assert_allclose(robot.pose()[:3], (220, -60, 80), atol=0.01)

    def test_ventouse_sortie_entree(self):
        robot = self.ouvrir()
        robot.ventouse(True)
        self.assertEqual(self.faux.file[-1],
                         (CommandeDobot.VENTOUSE, bytes((1, 1))))
        robot.sortie(3, True)
        self.assertEqual(self.faux.file[-1], (CommandeDobot.ES_SORTIE, bytes((3, 1))))
        self.assertIn((CommandeDobot.ES_MULTIPLEXAGE, bytes((3, 2))),
                      self.faux.immediates)
        self.assertTrue(robot.entree(5))
        self.assertIn((CommandeDobot.ES_MULTIPLEXAGE, bytes((5, 3))),
                      self.faux.immediates)

    def test_couper_pompe(self):
        robot = self.ouvrir(outil="pince")
        robot.pince(True)
        self.assertEqual(self.faux.file[-1], (CommandeDobot.PINCE, bytes((1, 1))))
        robot.couper_pompe()
        # commandes IMMEDIATES (hors file), ventouse et pince a la fois
        self.assertIn((CommandeDobot.VENTOUSE, bytes((0, 0))), self.faux.immediates)
        self.assertIn((CommandeDobot.PINCE, bytes((0, 0))), self.faux.immediates)

    def _coupures(self):
        return self.faux.immediates.count((CommandeDobot.PINCE, bytes((0, 0))))

    def test_ouvrir_la_pince_coupe_la_pompe_une_seconde_apres(self):
        robot = self.ouvrir(outil="pince")
        with mock.patch("dobot_sim.commande.DELAI_COUPURE_POMPE", 0.1):
            debut = time.time()
            robot.pince(False)
            # le script n'a PAS attendu la coupure
            self.assertLess(time.time() - debut, 0.1 + 5.0)
            self.assertEqual(self._coupures(), 0)
            time.sleep(0.4)
        self.assertEqual(self.faux.file[-1], (CommandeDobot.PINCE, bytes((1, 0))))
        self.assertEqual(self._coupures(), 1)

    def test_refermer_avant_la_coupure_l_annule(self):
        robot = self.ouvrir(outil="pince")
        with mock.patch("dobot_sim.commande.DELAI_COUPURE_POMPE", 0.3):
            robot.pince(False)
            robot.pince(True)            # la piece est reprise : ne pas couper
            time.sleep(0.6)
        self.assertEqual(self._coupures(), 0)

    def test_fermer_coupe_la_pompe_encore_attendue(self):
        robot = self.ouvrir(outil="pince")
        with mock.patch("dobot_sim.commande.DELAI_COUPURE_POMPE", 0.2):
            robot.pince(False)
            robot.fermer()
        self.assertEqual(self._coupures(), 1)

    def test_ouvrir_sans_couper_laisse_la_pompe(self):
        robot = self.ouvrir(outil="pince")
        with mock.patch("dobot_sim.commande.DELAI_COUPURE_POMPE", 0.05):
            robot.pince(False, couper=False)
            time.sleep(0.2)
        self.assertEqual(self._coupures(), 0)

    def test_regler(self):
        robot = self.ouvrir()
        robot.regler(vitesse=20, continu=False, lissage=30)
        self.assertEqual((robot.vitesse, robot.continu, robot.lissage),
                         (20.0, False, 30.0))
        robot.regler_longueur_outil(70.0)
        self.assertEqual(robot.modele.geo.lt, 70.0)
        robot.movl(cible(220, 0, 80))
        p = struct.unpack("<B4f", self.faux.file[-1][1])
        # z du controleur = z + lt - d1 : la nouvelle longueur est prise
        self.assertAlmostEqual(p[3], 80 + 70 - 139, places=3)

    def test_with_et_fermer(self):
        self.ouvrir()
        with self.robot as r:
            r.movl(cible(220, 0, 80))
        with self.assertRaises(RuntimeError):
            self.robot.movl(cible(220, 0, 60))


class TestGardeFous(_DirectMixin, unittest.TestCase):

    def assertRefuse(self, f, *mots):
        n0 = len(self.faux.file)
        pose0 = list(self.robot._pose)
        sortie = io.StringIO()
        with redirect_stdout(sortie):
            with self.assertRaises(ErreurExecution):
                f()
        self.assertEqual(len(self.faux.file), n0, "une commande est partie")
        self.assertEqual(self.robot._pose, pose0, "la cible memorisee a bouge")
        for m in mots:
            self.assertIn(m, sortie.getvalue())
        return sortie.getvalue()

    def test_hors_d_atteinte(self):
        self.ouvrir()
        self.assertRefuse(lambda: self.robot.movj(cible(500, 0, 80)),
                          "hors d'atteinte")

    def test_collision_avec_la_table(self):
        self.ouvrir()
        self.assertRefuse(lambda: self.robot.movl(cible(220, 0, -30)), "REFUSE")

    def test_un_refus_ne_touche_pas_la_memoire_des_cibles(self):
        self.ouvrir()
        self.robot.movl(cible(220, 30, 80))
        self.assertRefuse(lambda: self.robot.movl(x=500))
        self.robot.movl(z=60)                         # repart de (220, 30)
        np.testing.assert_allclose(self.robot.pose(), (220, 30, 60, 0), atol=0.01)

    def test_zone(self):
        self.ouvrir(zone=((180, 260), (-80, 80), (20, 140)))
        self.robot.movl(cible(220, 0, 80))
        self.assertRefuse(lambda: self.robot.movl(cible(220, 100, 80)),
                          "zone", "y = 100.0")
        self.assertRefuse(lambda: self.robot.movl(z=10), "z = 10.0")
        # un arc dont les bouts sont dans la zone mais qui en sort
        self.assertRefuse(lambda: self.robot.movc(cible(280, 0, 80),
                                                  cible(220, 0, 80)),
                          "zone")
        self.robot.movl(cible(240, 50, 100))

    def test_zone_invalide(self):
        with self.assertRaises(ValueError):
            Commande._verifier_zone(((0, 1), (2, 1), (0, 1)))

    def test_pas_max(self):
        self.ouvrir(pas_max=30)                       # depart : le repli
        self.assertRefuse(lambda: self.robot.movl(cible(220, 50, 80)),
                          "pas_max")
        self.robot.movl(z=110)                        # 24 mm : accepte
        # chaque segment d'une liste est juge
        x, y, z, _r = self.robot.pose()
        self.assertRefuse(lambda: self.robot.movl([cible(x, y + 20, z),
                                                   cible(x, y + 60, z)]),
                          "pas_max")

    def test_le_repli_echappe_a_pas_max_mais_pas_a_la_zone(self):
        self.ouvrir(pas_max=30, zone=((170, 270), (-90, 90), (30, 140)))
        for z in (110, 85):
            self.robot.movl(z=z)
        self.robot.movl(y=25)
        self.robot.movl(y=50)
        self.robot.repli()                      # 70 mm d'un coup : accepte
        np.testing.assert_allclose(self.robot.angles(), (0, 15, 30, 0), atol=1e-4)
        self.robot.zone = ((170, 270), (-90, 90), (30, 120))
        self.robot.movl(z=110)
        self.assertRefuse(lambda: self.robot.repli(), "zone")

    def test_robot_en_defaut_ouvre_la_mise_en_route(self):
        self.ouvrir()
        self.faux.alarmes_actives = [0x43]
        appels = []

        def mise_en_route(pont, robot):
            appels.append(1)
            self.faux.alarmes_actives = []
            return True
        with mock.patch("dobot_sim.ui.mise_en_route_autonome", mise_en_route), \
                redirect_stdout(io.StringIO()):
            self.robot.movl(cible(220, 0, 80))
        self.assertEqual(appels, [1])

    def test_defaut_persistant_rien_ne_part(self):
        self.ouvrir()
        self.faux.alarmes_actives = [0x43]
        n0 = len(self.faux.file)
        with mock.patch("dobot_sim.ui.mise_en_route_autonome",
                        lambda p, r: False), redirect_stdout(io.StringIO()):
            with self.assertRaises(ErreurExecution):
                self.robot.movl(cible(220, 0, 80))
        self.assertEqual(len(self.faux.file), n0)

    def test_defaut_pendant_le_mouvement(self):
        self.ouvrir()
        self.faux.cadence = 10.0                       # le robot traine...
        orig = self.faux.avancer

        def avancer():
            orig()
            if self.faux.demarree:
                self.faux.alarmes_actives = [0x22]     # ...et se met en defaut
        self.faux.avancer = avancer
        with redirect_stdout(io.StringIO()) as s:
            with self.assertRaises(ErreurExecution):
                self.robot.movl(cible(220, 0, 80))
        self.assertIn("EN DEFAUT", s.getvalue())

    def test_ctrl_c_arret_d_urgence(self):
        self.ouvrir()
        self.faux.cadence = 10.0                       # mouvement interminable

        def sommeil(_s):
            raise KeyboardInterrupt
        with mock.patch.object(cmd.time, "sleep", sommeil), \
                redirect_stdout(io.StringIO()) as s:
            with self.assertRaises(KeyboardInterrupt):
                self.robot.movl(cible(220, 0, 80))
        self.assertIn((CommandeDobot.FILE_ARRET_FORCE, b""), self.faux.immediates)
        self.assertIn("ARRET D'URGENCE", s.getvalue())

    def test_avertissement_jaune_et_on_continue(self):
        self.ouvrir()
        self.robot.movl(cible(220, -60, 80))
        with redirect_stdout(io.StringIO()) as s:
            self.robot.movl([(cible(220, 60, 80), 0.3), cible(250, 60, 80)])
        self.assertIn("non tenable", s.getvalue())
        np.testing.assert_allclose(self.robot.pose()[:3], (250, 60, 80), atol=0.01)


# ---------------------------------------------------------------------------
# Chemin simulateur, contre une fausse fenetre
# ---------------------------------------------------------------------------

class FausseFenetre:
    """Repond comme la fenetre, sans Qt. Termine apres deux interrogations."""

    def __init__(self, q=(0.0, 15.0, 30.0, 0.0), connu=True):
        self.q = list(q)
        self.connu = connu
        self.recu = []
        self.etat = {"etat": "aucune"}
        self.polls = 0
        self.reponses = None          # suite d'etats imposee

    def __call__(self, charge, attente=None):
        charge = json.loads(json.dumps(charge))       # passe-t-il en JSON ?
        self.recu.append(charge)
        if charge.get("ping"):
            return {}
        if "commande" not in charge or not self.connu:
            return None
        d = charge["commande"]
        if d.get("lire"):
            return {"q": self.q, "reel": d.get("reel"), "entree": True}
        if d.get("etat"):
            self.polls += 1
            if self.reponses:
                self.etat = self.reponses.pop(0)
            elif self.polls >= 2 and self.etat["etat"] == "en_cours":
                self.etat = {"etat": "termine"}
                self.q = self._fin
            return dict(self.etat)
        if d.get("arret_urgence"):
            self.etat = {"etat": "refuse", "raison": "urgence"}
            return dict(self.etat)
        if "connecter" in d:
            self.etat = {"etat": "pret"}
            return dict(self.etat)
        if "executer" in d:
            e = d["executer"]
            from dobot_sim.program import Programme
            prog = Programme.depuis_dict(e["programme"])
            m = _modele()
            traj = prog.construire(m, np.array(e["q_depart"]))
            self._fin = [float(v) for v in traj.q(traj.duree)]
            self.etat = {"etat": "en_cours"}
            self.polls = 0
            return {"etat": "preparation"}
        return None


class TestSimulateur(unittest.TestCase):

    def ouvrir(self, fen=None, **kw):
        self.fen = fen or FausseFenetre()
        p = mock.patch.object(cmd, "demander", self.fen)
        p.start()
        self.addCleanup(p.stop)
        with redirect_stdout(io.StringIO()):
            return Commande(simulateur=True, **kw)

    def test_aucun_simulateur(self):
        with mock.patch.object(cmd, "demander", lambda c, attente=None: None), \
                redirect_stdout(io.StringIO()) as s:
            with self.assertRaises(ErreurExecution):
                Commande(simulateur=True)
        self.assertIn("lancer_simulateur=True", s.getvalue())

    def test_simulateur_trop_ancien(self):
        fen = FausseFenetre(connu=False)
        with mock.patch.object(cmd, "demander", fen), \
                redirect_stdout(io.StringIO()) as s:
            with self.assertRaises(ErreurExecution):
                Commande(simulateur=True)
        self.assertIn("NE CONNAIT PAS", s.getvalue())

    def test_commande_animee_et_attendue(self):
        robot = self.ouvrir()
        robot.movl([(cible(220, 0, 80), 1.0), cible(220, 40, 80)])
        e = [c["commande"]["executer"] for c in self.fen.recu
             if "executer" in c.get("commande", {})][-1]
        self.assertFalse(e["reel"])
        self.assertEqual(e["q_depart"], [0.0, 15.0, 30.0, 0.0])
        self.assertEqual(e["etapes"][0]["type"], "cp")
        np.testing.assert_allclose(robot.pose()[:3], (220, 40, 80), atol=1e-6)
        self.assertGreaterEqual(self.fen.polls, 2)

    def test_lecture_de_l_entree(self):
        robot = self.ouvrir()
        self.assertTrue(robot.entree(4))
        lu = self.fen.recu[-1]["commande"]
        self.assertEqual((lu["entree"], lu["reel"]), (4, False))

    def test_refus_de_la_fenetre(self):
        robot = self.ouvrir()
        self.fen.reponses = [{"etat": "refuse", "raison": "Arret de securite"}]
        with redirect_stdout(io.StringIO()) as s:
            with self.assertRaises(ErreurExecution):
                robot.movl(cible(220, 0, 80))
        self.assertIn("Arret de securite", s.getvalue())

    def test_mise_en_route_annoncee_une_fois(self):
        robot = self.ouvrir(robot=True)
        self.fen.reponses = [{"etat": "mise_en_route"}] * 3 + [{"etat": "termine"}]
        with redirect_stdout(io.StringIO()) as s:
            robot.movl(cible(220, 0, 80))
        self.assertEqual(s.getvalue().count("MISE EN ROUTE"), 1)
        e = [c["commande"]["executer"] for c in self.fen.recu
             if "executer" in c.get("commande", {})][-1]
        self.assertTrue(e["reel"])
        connecter = [c for c in self.fen.recu
                     if "connecter" in c.get("commande", {})]
        self.assertEqual(len(connecter), 1)

    def test_ctrl_c_demande_l_arret_a_la_fenetre(self):
        robot = self.ouvrir()

        def sommeil(_s):
            raise KeyboardInterrupt
        with mock.patch.object(cmd.time, "sleep", sommeil), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                robot.movl(cible(220, 0, 80))
        self.assertTrue(self.fen.recu[-1]["commande"].get("arret_urgence"))
        self.assertFalse(self.fen.recu[-1]["commande"].get("reel"))

    def test_lancer_simulateur_dans_un_autre_processus(self):
        fen = FausseFenetre()
        etat = {"lance": False}

        def demander(charge, attente=None):
            return fen(charge) if etat["lance"] else None

        class Processus:
            returncode = None

            def __init__(self, args, **kw):
                Processus.args, Processus.kw = args, kw
                etat["lance"] = True

            def poll(self):
                return None
        with mock.patch.object(cmd, "demander", demander), \
                mock.patch.object(cmd.subprocess, "Popen", Processus), \
                redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, lancer_simulateur=True)
        self.assertEqual(Processus.args[0], sys.executable)
        self.assertTrue(Processus.args[1].endswith("main.py"))
        if os.name != "nt":
            self.assertTrue(Processus.kw.get("start_new_session"))
        self.assertIsNotNone(robot.processus_simulateur)

    def test_lancement_qui_echoue(self):
        class Processus:
            returncode = 1

            def __init__(self, *a, **kw):
                pass

            def poll(self):
                return 1
        with mock.patch.object(cmd, "demander", lambda c, attente=None: None), \
                mock.patch.object(cmd.subprocess, "Popen", Processus), \
                redirect_stdout(io.StringIO()) as s:
            with self.assertRaises(ErreurExecution):
                Commande(simulateur=True, lancer_simulateur=True)
        self.assertIn("ARRETE AU DEMARRAGE", s.getvalue())


# ---------------------------------------------------------------------------
# Chemin simulateur, contre la VRAIE fenetre (hors ecran)
# ---------------------------------------------------------------------------

def _qt():
    try:
        from pyqtgraph.Qt import QtWidgets
        return QtWidgets
    except Exception:                                  # pragma: no cover
        return None


@unittest.skipIf(_qt() is None, "Qt indisponible")
class TestFenetre(unittest.TestCase):
    """
    La fenetre recoit les demandes comme par la passerelle -- en JSON --,
    mais sans socket : un meme processus ne peut pas a la fois attendre sa
    reponse et la servir. Chaque demande laisse tourner la boucle Qt, ce
    qui fait avancer l'animation et les rappels differes.
    """

    @classmethod
    def setUpClass(cls):
        QtWidgets = _qt()
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from dobot_sim.ui import FenetrePrincipale
        with redirect_stdout(io.StringIO()):
            cls.fen = FenetrePrincipale()
        cls.fen.sp_facteur.setValue(5.0)              # animation acceleree

    @classmethod
    def tearDownClass(cls):
        cls.fen.minuteur_reel.stop()
        cls.fen._serveur.fermer()
        cls.fen.pont.liaison = None
        cls.fen.close()

    def setUp(self):
        self.fen.pont.liaison = None
        self.fen.minuteur_reel.stop()
        self.fen._cmd_script = {"etat": "aucune"}

        def demander(charge, attente=None):
            self.app.processEvents()
            try:
                rep = self.fen._serveur.rappel(json.loads(json.dumps(charge)))
            except Exception:
                return None
            self.app.processEvents()
            return {} if rep is None else json.loads(json.dumps(rep))
        p = mock.patch.object(cmd, "demander", demander)
        p.start()
        self.addCleanup(p.stop)

    def _brancher(self, q=(0.0, 15.0, 30.0, 0.0)):
        self.faux = ControleurMobile(q=q, modele=_modele())
        self.fen.pont.liaison = _liaison_simulee(self.faux)
        self.fen.pont.biais = BIAIS

    def test_simulateur_seul(self):
        with redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=False, vitesse=80)
            robot.movl(cible(220, 0, 80))
            robot.movl([(cible(220, 40, 80), 0.5), cible(240, 40, 80)])
        np.testing.assert_allclose(self.fen.robot.pose(self.fen.q)[:3],
                                   (240, 40, 80), atol=1e-3)
        np.testing.assert_allclose(robot.pose()[:3], (240, 40, 80), atol=1e-3)
        self.assertEqual(self.fen.programme.nom, "commande")
        # le simulateur joue la course continue, sans arret au coin
        profils = [i.profil for i in self.fen.programme.instructions]
        self.assertEqual(profils, ["demarrage", "arret"])

    def test_la_ventouse_reste_affichee_d_une_commande_a_l_autre(self):
        with redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=False)
            robot.movl(cible(220, 0, 80))
            robot.ventouse(True)
            robot.sortie(5, True)
            robot.movl(cible(220, 30, 80))
        self.assertTrue(self.fen.outil_actif)
        self.assertTrue(self.fen.es.sorties[4])
        with redirect_stdout(io.StringIO()):
            robot.ventouse(False)
        self.assertFalse(self.fen.outil_actif)

    def test_simulateur_et_robot(self):
        self._brancher()
        self.fen.sp_reel_ratio.setValue(30.0)
        with redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=True)
            robot.movl(cible(220, 0, 80))
            robot.movl([cible(220, 40, 80), cible(250, 40, 80)])
            robot.ventouse(True)
        self.assertEqual(self.fen.sp_reel_ratio.value(), 100.0)
        self.assertTrue(self.fen.chk_miroir.isChecked())
        self.assertIn(CommandeDobot.CP, self.faux.ids_file())
        self.assertEqual(self.faux.file[-1][0], CommandeDobot.VENTOUSE)
        self.assertTrue(self.fen.outil_actif)    # affichee, sans animation
        np.testing.assert_allclose(robot.pose()[:3], (250, 40, 80), atol=0.01)

    def test_connexion_demandee_par_le_script(self):
        def connecter(port):
            self._brancher()
            return "3.7.0"
        with mock.patch.object(self.fen.pont, "connecter", connecter), \
                mock.patch.object(self.fen, "_reel_chercher_ports", lambda: None), \
                mock.patch.object(self.fen.cb_port, "currentData",
                                  lambda: "/dev/SIMULE"), \
                redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=True)
            robot.movl(cible(220, 0, 80))
        self.fen.minuteur_reel.stop()
        self.assertTrue(self.fen.pont.connecte)
        np.testing.assert_allclose(robot.pose()[:3], (220, 0, 80), atol=0.01)

    def test_mise_en_route_dans_la_fenetre(self):
        self._brancher()
        self.faux.alarmes_actives = [0x43]
        ouverts = []

        class Dialogue:
            def __init__(d, fen):
                ouverts.append(fen)

            def exec_(d):
                self.faux.alarmes_actives = []
        with mock.patch("dobot_sim.ui.DialogueMiseEnRoute", Dialogue), \
                redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=True)
            robot.movl(cible(220, 0, 80))
        self.assertEqual(len(ouverts), 1)

    def test_defaut_persistant_refuse(self):
        self._brancher()

        class Dialogue:
            def __init__(d, fen):
                pass

            def exec_(d):
                pass                                  # l'operateur abandonne
        with mock.patch("dobot_sim.ui.DialogueMiseEnRoute", Dialogue), \
                redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=True)
            self.faux.alarmes_actives = [0x43]
            n0 = len(self.faux.file)
            with self.assertRaises(ErreurExecution):
                robot.movl(cible(220, 0, 80))
        self.assertEqual(len(self.faux.file), n0)

    def test_robot_non_connecte_refuse(self):
        with redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=False)
        robot.robot_reel = True                   # la fenetre n'a pas de robot
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ErreurExecution):
                robot.movl(cible(220, 0, 80))

    def test_arret_d_urgence_par_le_script(self):
        self._brancher()
        with redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=True)
        self.faux.cadence = 10.0

        appels = {"n": 0}
        vrai = time.sleep

        def sommeil(s):
            appels["n"] += 1
            if appels["n"] > 3:
                raise KeyboardInterrupt
            vrai(s)
        with mock.patch.object(cmd.time, "sleep", sommeil), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                robot.movl(cible(220, 0, 80))
        self.assertIn((CommandeDobot.FILE_ARRET_FORCE, b""), self.faux.immediates)
        self.assertEqual(self.fen._cmd_script["etat"], "refuse")

    def test_arret_de_securite_remonte_au_script(self):
        self.fen._cmd_script = {"etat": "en_cours", "reel": False}
        with mock.patch.object(self.fen, "_boite_securite", create=True):
            self.fen._arret_securite(["plan de pose traverse de 3 mm"])
        self.assertEqual(self.fen._cmd_script["etat"], "refuse")
        self.assertIn("plan de pose", self.fen._cmd_script["raison"])
        b = getattr(self.fen, "_boite_securite", None)
        if b is not None and hasattr(b, "close"):
            b.close()

    def test_lecture_robot_et_conversion(self):
        self._brancher(q=_modele().mgi(cible(230, 20, 70)))
        with redirect_stdout(io.StringIO()):
            robot = Commande(simulateur=True, robot=True)
        np.testing.assert_allclose(robot.pose()[:3], (230, 20, 70), atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
