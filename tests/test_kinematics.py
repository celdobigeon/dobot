#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires du coeur mathematique (aucune dependance a Qt).

    python -m unittest discover -s tests -v
ou  python tests/test_kinematics.py
"""

import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import homogeneous as hg
from dobot_sim.robot import DobotMagician, ErreurCinematique
from dobot_sim.trajectory import (ProfilLineaire, ProfilPolynomial5,
                                  ProfilSinus, ProfilTrapeze,
                                  SegmentArticulaire, SegmentCirculaire,
                                  SegmentLineaire, Trajectoire)


def _q_aleatoires(robot, n, graine=1234):
    rng = np.random.default_rng(graine)
    return np.column_stack([rng.uniform(a, b, n)
                            for a, b in zip(robot.lim.q_min, robot.lim.q_max)])


class TestMatricesHomogenes(unittest.TestCase):

    def test_inverse(self):
        T = hg.composer(hg.translation(12, -5, 30), hg.rot_z(0.7),
                        hg.rot_y(-0.3), hg.rot_x(1.1))
        self.assertTrue(np.allclose(T @ hg.inverse(T), np.eye(4), atol=1e-12))
        self.assertTrue(np.allclose(hg.inverse(T), np.linalg.inv(T), atol=1e-10))

    def test_validite(self):
        T = hg.composer(hg.rot_x(0.4), hg.rot_y(-1.2), hg.translation(3, 4, 5))
        self.assertTrue(hg.est_homogene(T))
        faux = T.copy()
        faux[:3, :3] *= 1.5
        self.assertFalse(hg.est_homogene(faux))

    def test_rotations_elementaires(self):
        self.assertTrue(np.allclose(
            hg.transformer_point(hg.rot_z(np.pi / 2), [1, 0, 0]), [0, 1, 0],
            atol=1e-12))
        self.assertTrue(np.allclose(
            hg.transformer_point(hg.rot_y(np.pi / 2), [0, 0, 1]), [1, 0, 0],
            atol=1e-12))
        self.assertTrue(np.allclose(
            hg.transformer_point(hg.rot_x(np.pi / 2), [0, 1, 0]), [0, 0, 1],
            atol=1e-12))

    def test_rot_axe_equivaut_aux_rotations_simples(self):
        for axe, f in (([1, 0, 0], hg.rot_x), ([0, 1, 0], hg.rot_y),
                       ([0, 0, 1], hg.rot_z)):
            for a in (-2.0, 0.3, 2.9):
                self.assertTrue(np.allclose(hg.rot_axe(axe, a), f(a), atol=1e-12))

    def test_rpy_aller_retour(self):
        rng = np.random.default_rng(7)
        for _ in range(200):
            r, p, y = rng.uniform(-np.pi / 2 + 0.05, np.pi / 2 - 0.05, 3)
            T = hg.rpy_vers_rotation(r, p, y)
            r2, p2, y2 = hg.rotation_vers_rpy(T)
            self.assertTrue(np.allclose([r, p, y], [r2, p2, y2], atol=1e-9))

    def test_quaternion_aller_retour(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            axe = rng.normal(size=3)
            angle = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
            T = hg.rot_axe(axe, angle)
            T2 = hg.quaternion_vers_rotation(hg.rotation_vers_quaternion(T))
            self.assertTrue(np.allclose(hg.rotation_de(T), hg.rotation_de(T2),
                                        atol=1e-9))

    def test_axe_angle_aller_retour(self):
        rng = np.random.default_rng(13)
        for _ in range(200):
            axe = rng.normal(size=3)
            axe = axe / np.linalg.norm(axe)
            angle = rng.uniform(0.02, np.pi - 0.02)
            a2, t2 = hg.rotation_vers_axe_angle(hg.rot_axe(axe, angle))
            if np.dot(a2, axe) < 0:
                a2, t2 = -a2, t2
            self.assertAlmostEqual(t2, angle, places=8)
            self.assertTrue(np.allclose(a2, axe, atol=1e-7))

    def test_slerp_extremites_et_milieu(self):
        T0 = hg.rot_z(0.0)
        T1 = hg.rot_z(1.2)
        self.assertTrue(np.allclose(hg.rotation_de(hg.slerp(T0, T1, 0.0)),
                                    hg.rotation_de(T0), atol=1e-12))
        self.assertTrue(np.allclose(hg.rotation_de(hg.slerp(T0, T1, 1.0)),
                                    hg.rotation_de(T1), atol=1e-12))
        self.assertTrue(np.allclose(hg.rotation_de(hg.slerp(T0, T1, 0.5)),
                                    hg.rotation_de(hg.rot_z(0.6)), atol=1e-12))

    def test_interpolation_lineaire_en_position(self):
        T0 = hg.translation(0, 0, 0)
        T1 = hg.composer(hg.translation(100, 40, -20), hg.rot_z(0.9))
        for s in np.linspace(0, 1, 11):
            p = hg.position_de(hg.interpoler(T0, T1, s))
            self.assertTrue(np.allclose(p, s * np.array([100, 40, -20]), atol=1e-9))

    def test_dh_standard_conforme_a_sa_definition(self):
        th, d, a, al = 0.6, 12.0, 34.0, -0.8
        attendu = hg.composer(hg.rot_z(th), hg.translation(0, 0, d),
                              hg.translation(a, 0, 0), hg.rot_x(al))
        self.assertTrue(np.allclose(hg.dh_standard(th, d, a, al), attendu,
                                    atol=1e-12))

    def test_dh_modifie_conforme_a_sa_definition(self):
        al, a, th, d = 0.4, 21.0, -1.1, 7.0
        attendu = hg.composer(hg.rot_x(al), hg.translation(a, 0, 0),
                              hg.rot_z(th), hg.translation(0, 0, d))
        self.assertTrue(np.allclose(hg.dh_modifie(al, a, th, d), attendu,
                                    atol=1e-12))

    def test_erreur_pose(self):
        T0 = hg.composer(hg.translation(10, 0, 0), hg.rot_z(0.2))
        T1 = hg.composer(hg.translation(13, 4, 0), hg.rot_z(0.5))
        dp, da = hg.erreur_pose(T1, T0)
        self.assertAlmostEqual(dp, 5.0, places=9)
        self.assertAlmostEqual(da, 0.3, places=9)


class TestCinematiqueDobot(unittest.TestCase):

    def setUp(self):
        self.robot = DobotMagician()

    def test_pose_de_reference(self):
        """A q = 0 le bras est vertical et l'avant-bras horizontal."""
        g = self.robot.geo
        p = self.robot.pose([0, 0, 0, 0])
        self.assertAlmostEqual(p[0], g.l2 + g.l3, places=9)
        self.assertAlmostEqual(p[1], 0.0, places=9)
        self.assertAlmostEqual(p[2], g.d1 + g.l1, places=9)

    def test_mgd_coherent_avec_la_chaine_de_matrices(self):
        for q in _q_aleatoires(self.robot, 300):
            T = self.robot.mgd(q)
            p_formule = self.robot.pose(q)
            self.assertTrue(np.allclose(hg.position_de(T), p_formule[:3], atol=1e-9))

    def test_parallelogramme_outil_toujours_vertical(self):
        """L'axe z de l'outil doit rester vertical, dirige vers le bas."""
        for q in _q_aleatoires(self.robot, 300):
            R = hg.rotation_de(self.robot.mgd(q))
            self.assertTrue(np.allclose(R[:, 2], [0, 0, -1], atol=1e-9))

    def test_orientation_outil_vaut_q1_plus_q4(self):
        for q in _q_aleatoires(self.robot, 200):
            R = hg.rotation_de(self.robot.mgd(q))
            r = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            attendu = (q[0] + q[3] + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual((r - attendu + 180.0) % 360.0 - 180.0, 0.0,
                                   places=8)

    def test_mgd_mgi_aller_retour_pose(self):
        """La pose doit toujours etre retrouvee exactement."""
        for q in _q_aleatoires(self.robot, 1000):
            q2 = self.robot.mgi(self.robot.mgd(q), tolerant=True)
            p0 = self.robot.pose(q)
            p1 = self.robot.pose(q2)
            self.assertTrue(np.allclose(p0[:3], p1[:3], atol=1e-7),
                            "q={} -> {}".format(q, q2))
            self.assertAlmostEqual((p0[3] - p1[3] + 180.0) % 360.0 - 180.0, 0.0,
                                   places=7)

    def test_mgd_mgi_aller_retour_angles(self):
        """
        Hors hyperextension, la solution coude haut est unique : on doit
        retrouver exactement les angles de depart, y compris bras replie.
        """
        testes = 0
        for q in _q_aleatoires(self.robot, 1000):
            # gamma = flexion du coude ; les deux solutions se confondent
            # aux extremites (bras tendu gamma = 0, bras replie gamma = 180)
            gamma = 90.0 + q[2] - q[1]
            if gamma < 3.0 or gamma > 177.0:
                continue
            testes += 1
            q2 = self.robot.mgi(self.robot.mgd(q))
            self.assertTrue(np.allclose(q[:3], q2[:3], atol=1e-7),
                            "q={} -> {}".format(q, q2))
        self.assertGreater(testes, 900)

    def test_bras_replie_pas_de_repliement_d_angle(self):
        """
        Bras completement replie (D -> |l1-l2|) : beta vaut pi et le calcul
        brut sortirait de ]-180, 180]. Le resultat doit rester dans les butees.
        """
        for q2, q3 in [(0.0, 90.0), (1.0, 92.0), (4.0, 94.0), (0.5, 93.5)]:
            q = np.array([20.0, q2, q3, 0.0])
            retrouve = self.robot.mgi(self.robot.mgd(q))
            self.assertTrue(np.allclose(q[:3], retrouve[:3], atol=1e-7),
                            "q={} -> {}".format(q, retrouve))
            self.assertTrue(self.robot.dans_butees(retrouve))

    def test_les_deux_solutions_du_triangle(self):
        """Coude haut et coude bas atteignent la meme pose."""
        cible = (240.0, 60.0, 90.0, 15.0)
        q_haut = self.robot.mgi_pose(*cible, coude="haut")
        q_bas = self.robot.mgi_pose(*cible, coude="bas")
        self.assertFalse(np.allclose(q_haut, q_bas, atol=1e-3))
        for q in (q_haut, q_bas):
            p = self.robot.pose(q)
            self.assertTrue(np.allclose(p[:3], cible[:3], atol=1e-7))
        # seule la solution coude haut respecte les butees ici
        self.assertTrue(self.robot.dans_butees(q_haut))
        self.assertFalse(self.robot.dans_butees(q_bas))

    def test_mgi_hors_atteinte(self):
        with self.assertRaises(ErreurCinematique):
            self.robot.mgi_pose(2000.0, 0.0, 0.0, 0.0)
        # le mode tolerant ne doit pas lever
        q = self.robot.mgi_pose(2000.0, 0.0, 0.0, 0.0, tolerant=True)
        self.assertEqual(len(q), 4)

    def test_reperes_positions_remarquables(self):
        g = self.robot.geo
        R = self.robot.reperes([0, 0, 0, 0])
        self.assertTrue(np.allclose(hg.position_de(R[2]), [0, 0, g.d1], atol=1e-9))
        self.assertTrue(np.allclose(hg.position_de(R[3]), [0, 0, g.d1 + g.l1],
                                    atol=1e-9))
        self.assertTrue(np.allclose(hg.position_de(R[4]),
                                    [g.l2, 0, g.d1 + g.l1], atol=1e-9))
        self.assertTrue(np.allclose(hg.position_de(R[5]),
                                    [g.l2 + g.l3, 0, g.d1 + g.l1], atol=1e-9))

    def test_longueurs_des_segments(self):
        g = self.robot.geo
        for q in _q_aleatoires(self.robot, 200):
            P = self.robot.positions_articulations(q)
            self.assertAlmostEqual(np.linalg.norm(P[3] - P[2]), g.l1, places=9)
            self.assertAlmostEqual(np.linalg.norm(P[4] - P[3]), g.l2, places=9)
            self.assertAlmostEqual(np.linalg.norm(P[5] - P[4]), g.l3, places=9)

    def test_jacobienne_analytique_contre_numerique(self):
        for q in _q_aleatoires(self.robot, 300):
            Ja = self.robot.jacobienne(q)
            Jn = self.robot.jacobienne_numerique(q)
            self.assertTrue(np.allclose(Ja, Jn, atol=1e-4),
                            "ecart max {:.2e} en q={}".format(
                                float(np.max(np.abs(Ja - Jn))), q))

    def test_jacobienne_predit_le_deplacement(self):
        """Un petit deplacement articulaire doit suivre J . dq au premier ordre."""
        q = np.array([25.0, 30.0, 40.0, 10.0])
        dq_deg = np.array([0.05, -0.04, 0.03, 0.02])
        attendu = self.robot.jacobienne(q) @ np.radians(dq_deg)
        reel = self.robot.pose(q + dq_deg) - self.robot.pose(q)
        self.assertTrue(np.allclose(attendu[:3], reel[:3], atol=1e-3))

    def test_butees(self):
        self.assertTrue(self.robot.dans_butees([0, 10, 20, 0]))
        self.assertFalse(self.robot.dans_butees([200, 10, 20, 0]))
        # trois butees articulaires, et le couplage J3 - J2 (250 deg)
        self.assertEqual(len(self.robot.depassements([200, -50, 200, 0])), 4)
        # le couplage mesure : J2 69,3 / J3 11,1 est refuse par le controleur
        self.assertFalse(self.robot.dans_butees([32.0, 69.3, 11.1, 60.0]))
        self.assertTrue(self.robot.dans_butees([29.2, 38.6, 14.2, 60.0]))
        q = self.robot.saturer([200, -50, 200, 0])
        self.assertTrue(self.robot.dans_butees(q))

    def test_singularite_sur_l_axe(self):
        """Quand le TCP s'approche de l'axe vertical, la manipulabilite chute."""
        loin = self.robot.manipulabilite(self.robot.mgi_pose(280, 0, 100, 0))
        self.assertGreater(loin, 0.0)
        J = self.robot.jacobienne([0.0, 0.0, 90.0, 0.0])   # portee = l3 seulement
        self.assertLess(abs(np.linalg.det(J)), loin)


class TestProfils(unittest.TestCase):

    def test_conditions_aux_limites(self):
        for cls in (ProfilLineaire, ProfilTrapeze, ProfilPolynomial5, ProfilSinus):
            p = cls(2.0)
            self.assertAlmostEqual(p.s(0.0), 0.0, places=9, msg=cls.__name__)
            self.assertAlmostEqual(p.s(2.0), 1.0, places=6, msg=cls.__name__)

    def test_monotonie(self):
        for cls in (ProfilLineaire, ProfilTrapeze, ProfilPolynomial5, ProfilSinus):
            p = cls(3.0)
            s = np.array([p.s(t) for t in np.linspace(0, 3, 300)])
            self.assertTrue(np.all(np.diff(s) >= -1e-12), cls.__name__)

    def test_vitesse_nulle_aux_extremites(self):
        for cls in (ProfilTrapeze, ProfilPolynomial5, ProfilSinus):
            p = cls(2.0)
            self.assertAlmostEqual(p.ds(0.0), 0.0, places=6, msg=cls.__name__)
            self.assertAlmostEqual(p.ds(2.0), 0.0, places=6, msg=cls.__name__)

    def test_integrale_du_trapeze(self):
        # np.trapz a ete retire dans numpy 2 au profit de np.trapezoid
        integre = getattr(np, "trapezoid", None) or np.trapz
        p = ProfilTrapeze(4.0, 0.3)
        t = np.linspace(0, 4, 20001)
        v = np.array([p.ds(x) for x in t])
        self.assertAlmostEqual(float(integre(v, t)), 1.0, places=4)


class TestSegments(unittest.TestCase):

    def setUp(self):
        self.robot = DobotMagician()
        self.q0 = np.array([0.0, 15.0, 30.0, 0.0])

    def test_segment_articulaire_atteint_la_cible(self):
        q1 = np.array([40.0, 30.0, 45.0, -20.0])
        s = SegmentArticulaire(self.robot, self.q0, q1, ProfilPolynomial5(2.0))
        self.assertTrue(np.allclose(s.q(0.0), self.q0, atol=1e-9))
        self.assertTrue(np.allclose(s.q(2.0), q1, atol=1e-9))

    def test_segment_lineaire_est_rectiligne(self):
        T1 = self.robot.pose_vers_matrice(250, 60, 30, 25)
        s = SegmentLineaire(self.robot, self.q0, T1, ProfilPolynomial5(2.0))
        p0 = hg.position_de(s.T0)
        p1 = hg.position_de(s.T1)
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        for t in np.linspace(0, 2, 41):
            p = hg.position_de(self.robot.mgd(s.q(t)))   # position reelle par MGD
            d = p - p0
            self.assertLess(np.linalg.norm(d - np.dot(d, u) * u), 1e-6)
        self.assertTrue(np.allclose(hg.position_de(self.robot.mgd(s.q(2.0))), p1,
                                    atol=1e-6))

    def test_segment_circulaire_reste_sur_le_cercle(self):
        T_via = self.robot.pose_vers_matrice(265, 0, 120, 0)
        T1 = self.robot.pose_vers_matrice(210, 90, 100, 0)
        s = SegmentCirculaire(self.robot, self.q0, T_via, T1, ProfilPolynomial5(3.0))
        self.assertFalse(s._degenere)
        for t in np.linspace(0, 3, 41):
            p = hg.position_de(s.pose(t))
            self.assertAlmostEqual(float(np.linalg.norm(p - s.centre)), s.rayon,
                                   places=6)
        self.assertTrue(np.allclose(hg.position_de(s.pose(3.0)),
                                    hg.position_de(T1), atol=1e-6))

    def test_segment_circulaire_passe_par_le_point_intermediaire(self):
        T_via = self.robot.pose_vers_matrice(265, 0, 120, 0)
        T1 = self.robot.pose_vers_matrice(210, 90, 100, 0)
        s = SegmentCirculaire(self.robot, self.q0, T_via, T1, ProfilPolynomial5(3.0))
        p_via = hg.position_de(T_via)
        distances = [np.linalg.norm(hg.position_de(s.pose(t)) - p_via)
                     for t in np.linspace(0, 3, 400)]
        self.assertLess(min(distances), 1.0)

    def test_points_alignes_bascule_en_ligne_droite(self):
        T_via = self.robot.pose_vers_matrice(240, 0, 60, 0)
        T1 = self.robot.pose_vers_matrice(260, 0, 60, 0)
        q0 = self.robot.mgi_pose(220, 0, 60, 0)
        s = SegmentCirculaire(self.robot, q0, T_via, T1, ProfilPolynomial5(2.0))
        self.assertTrue(s._degenere)
        self.assertTrue(np.allclose(hg.position_de(s.pose(2.0)),
                                    hg.position_de(T1), atol=1e-6))

    def test_trajectoire_continue_et_echantillonnage(self):
        q1 = self.robot.mgi_pose(230, -70, 60, 0)
        t1 = SegmentArticulaire(self.robot, self.q0, q1, ProfilPolynomial5(1.5))
        t2 = SegmentLineaire(self.robot, t1.q_final,
                             self.robot.pose_vers_matrice(230, -70, 10, 0),
                             ProfilTrapeze(1.0))
        traj = Trajectoire(self.robot, [t1, t2])
        self.assertAlmostEqual(traj.duree, 2.5, places=9)
        # continuite a la jonction
        self.assertTrue(np.allclose(traj.q(1.5 - 1e-9), traj.q(1.5 + 1e-9),
                                    atol=1e-5))
        e = traj.echantillonner(0.02)
        self.assertEqual(e["Q"].shape[1], 4)
        self.assertEqual(len(e["t"]), len(e["V"]))
        self.assertTrue(np.all(np.isfinite(e["Q"])))


class TestVitesseAcceleration(unittest.TestCase):
    """La duree doit decouler de la longueur, de v_max et de a_max."""

    def setUp(self):
        self.robot = DobotMagician()

    def test_chaque_profil_atteint_exactement_la_vitesse_demandee(self):
        L, v, a = 200.0, 100.0, 400.0
        for cls in (ProfilLineaire, ProfilTrapeze, ProfilSinus, ProfilPolynomial5):
            p = cls.adapte(L, v, a)
            ts = np.linspace(0, p.duree, 4001)
            vitesses = np.array([p.ds(t) for t in ts]) * L
            self.assertAlmostEqual(float(vitesses.max()), v, delta=0.5,
                                   msg=cls.__name__)

    def test_le_trapeze_sature_aussi_l_acceleration(self):
        L, v, a = 200.0, 100.0, 400.0
        p = ProfilTrapeze.adapte(L, v, a)
        ts = np.linspace(0, p.duree, 20001)
        vitesses = np.array([p.ds(t) for t in ts]) * L
        accels = np.gradient(vitesses, ts)
        self.assertAlmostEqual(float(np.abs(accels[20:-20]).max()), a, delta=5.0)
        self.assertAlmostEqual(p.duree, L / v + v / a, places=6)

    def test_profil_triangulaire_quand_la_course_est_courte(self):
        v, a = 100.0, 400.0
        L = 0.5 * v * v / a               # trop court pour atteindre v
        p = ProfilTrapeze.adapte(L, v, a, duree_min=0.0)
        self.assertAlmostEqual(p.duree, 2.0 * np.sqrt(L / a), places=9)
        ts = np.linspace(0, p.duree, 4001)
        self.assertLess(float(np.max([p.ds(t) for t in ts])) * L, v)

    def test_ordre_des_durees_entre_lois_horaires(self):
        """A vitesse maximale egale, les lois lisses sont plus lentes."""
        d = {cls.nom: cls.duree_pour(200, 100, 1e9)
             for cls in (ProfilLineaire, ProfilTrapeze, ProfilSinus,
                         ProfilPolynomial5)}
        self.assertLess(d["lineaire"], d["trapeze"])
        self.assertLess(d["trapeze"], d["sinus"])
        self.assertLess(d["sinus"], d["polynome5"])

    def test_longueur_des_segments(self):
        q0 = np.array([0.0, 15.0, 30.0, 0.0])
        s = SegmentArticulaire(self.robot, q0, q0 + np.array([10, 5, -20, 3]),
                               ProfilLineaire(1.0))
        self.assertAlmostEqual(s.longueur, 20.0, places=9)   # plus grand ecart
        self.assertEqual(s.unite, "deg")

        T1 = self.robot.pose_vers_matrice(250, 0, 100, 0)
        s = SegmentLineaire(self.robot, q0, T1, ProfilLineaire(1.0))
        attendu = np.linalg.norm(hg.position_de(T1)
                                 - hg.position_de(self.robot.mgd(q0)))
        self.assertAlmostEqual(s.longueur, attendu, places=9)
        self.assertEqual(s.unite, "mm")

    def test_movl_respecte_la_vitesse_cartesienne(self):
        from dobot_sim.program import Instruction, Programme
        prog = Programme([
            Instruction(type="MOVJ", mode="cartesien", pose=[150, 0, 150, 0],
                        mode_temps="duree", duree=2.0),
            Instruction(type="MOVL", mode="cartesien", pose=[260, 0, 150, 0],
                        mode_temps="vitesse", vitesse=80, acceleration=300,
                        profil="trapeze"),
        ])
        traj = prog.construire(self.robot)
        e = traj.echantillonner(0.002)
        m = e["t"] >= traj.debuts[1]
        self.assertAlmostEqual(float(e["V"][m].max()), 80.0, delta=1.0)

    def test_movj_utilise_les_degres_meme_avec_une_cible_cartesienne(self):
        from dobot_sim.program import Instruction
        ins = Instruction(type="MOVJ", mode="cartesien")
        self.assertEqual(ins.unite, "deg")
        ins = Instruction(type="MOVL", mode="articulaire")
        self.assertEqual(ins.unite, "mm")


class TestLissage(unittest.TestCase):
    """Arrondi des raccords entre segments consecutifs."""

    def setUp(self):
        from dobot_sim.program import Instruction, Programme
        self.Instruction, self.Programme = Instruction, Programme
        self.robot = DobotMagician()
        self.coin = np.array([240.0, 0.0, 120.0])

    def _programme(self, pct):
        I = self.Instruction
        commun = dict(mode="cartesien", mode_temps="vitesse",
                      vitesse=120, acceleration=500, profil="trapeze")
        return self.Programme([
            I(type="MOVL", pose=[140, 0, 120, 0], **commun),
            I(type="MOVL", pose=[240, 0, 120, 0], lissage=pct, **commun),
            I(type="MOVL", pose=[240, 0, 30, 0], **commun),
        ])

    def _mesure(self, pct):
        traj = self._programme(pct).construire(self.robot)
        e = traj.echantillonner(0.002)
        coupe = float(np.linalg.norm(e["P"] - self.coin, axis=1).min())
        return traj, coupe

    def test_sans_lissage_le_point_de_passage_est_atteint(self):
        traj, coupe = self._mesure(0.0)
        self.assertLess(coupe, 1e-3)
        self.assertAlmostEqual(traj.duree, traj.duree_sans_lissage, places=9)
        self.assertFalse(traj.lissage_actif)

    def test_le_lissage_coupe_le_coin_et_raccourcit(self):
        traj0, coupe0 = self._mesure(0.0)
        traj1, coupe1 = self._mesure(100.0)
        self.assertGreater(coupe1, 1.0)
        self.assertGreater(coupe1, coupe0)
        self.assertLess(traj1.duree, traj0.duree)
        self.assertTrue(traj1.lissage_actif)

    def test_effet_monotone_sur_toute_la_plage(self):
        coupes, durees = [], []
        for pct in (0, 20, 40, 60, 80, 100):
            traj, coupe = self._mesure(pct)
            coupes.append(coupe)
            durees.append(traj.duree)
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(coupes, coupes[1:])),
                        "coupe non monotone : {}".format(coupes))
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(durees, durees[1:])),
                        "duree non monotone : {}".format(durees))
        self.assertGreater(coupes[-1], coupes[1])

    def test_continuite_de_la_consigne_dans_le_raccord(self):
        traj, _ = self._mesure(100.0)
        dt = 0.001
        e = traj.echantillonner(dt)
        sauts = np.abs(np.diff(e["Q"], axis=0)).max(axis=1)
        # aucun saut ne doit depasser ce qu'autorise la vitesse maximale
        self.assertLess(float(sauts.max()),
                        max(self.robot.lim.v_max) * dt * 3.0)

    def test_le_lissage_ne_traverse_pas_une_attente(self):
        I = self.Instruction
        commun = dict(mode="cartesien", mode_temps="vitesse", vitesse=120,
                      acceleration=500)
        prog = self.Programme([
            I(type="MOVL", pose=[140, 0, 120, 0], lissage=100, **commun),
            I(type="ATTENTE", duree=1.0),
            I(type="MOVL", pose=[240, 0, 120, 0], lissage=100, **commun),
        ])
        traj = prog.construire(self.robot)
        self.assertEqual(traj.recouvrements[0], 0.0)
        self.assertEqual(traj.recouvrements[1], 0.0)
        self.assertAlmostEqual(traj.duree, traj.duree_sans_lissage, places=9)

    def test_dates_des_evenements_outil_tiennent_compte_du_lissage(self):
        I = self.Instruction
        commun = dict(mode="cartesien", mode_temps="vitesse", vitesse=120,
                      acceleration=500)
        instrs = [I(type="MOVL", pose=[140, 0, 120, 0], **commun),
                  I(type="MOVL", pose=[240, 0, 120, 0], lissage=100, **commun),
                  I(type="MOVL", pose=[240, 0, 30, 0], **commun),
                  I(type="OUTIL", etat=True)]
        prog = self.Programme(instrs)
        traj = prog.construire(self.robot)
        t_outil = prog.evenements_outil[0][0]
        self.assertAlmostEqual(t_outil, traj.debuts[3], places=9)
        self.assertLess(t_outil, sum(s.duree for s in traj.segments[:3]))


class TestCiblesMatricielles(unittest.TestCase):
    """Les cibles sont des matrices homogenes, pas des triplets."""

    def setUp(self):
        from dobot_sim import cible, decomposer_cible, repere_travail
        self.cible = cible
        self.repere_travail = repere_travail
        self.decomposer = decomposer_cible
        self.robot = DobotMagician()

    def test_cible_est_une_matrice_homogene_valide(self):
        T = self.cible(215, -95, 25, rz=30)
        self.assertEqual(T.shape, (4, 4))
        self.assertTrue(hg.est_homogene(T))

    def test_axe_outil_dirige_vers_le_bas(self):
        for rz in (-180, -45, 0, 37, 180):
            T = self.cible(200, 0, 50, rz=rz)
            self.assertTrue(np.allclose(T[:3, 2], [0, 0, -1], atol=1e-12))

    def test_repere_travail_a_son_axe_z_vers_le_haut(self):
        R = self.repere_travail(100, 50, 25, rz=20)
        self.assertTrue(np.allclose(R[:3, 2], [0, 0, 1], atol=1e-12))
        self.assertTrue(np.allclose(hg.position_de(R), [100, 50, 25]))

    def test_aller_retour_cible_decomposer(self):
        for x, y, z, r in [(215, -95, 25, 0), (180, 60, 120, 45),
                           (-100, 140, 60, -170), (250, 0, 5, 179)]:
            p = self.decomposer(self.cible(x, y, z, r))
            self.assertTrue(np.allclose(p[:3], [x, y, z], atol=1e-9))
            self.assertAlmostEqual((p[3] - r + 180) % 360 - 180, 0.0, places=9)

    def test_coordonnees_locales_dans_un_repere_de_travail(self):
        PALETTE = self.repere_travail(205, 75, 25, rz=20)
        T = self.cible(0, 48, 0, repere=PALETTE)
        a = np.radians(20.0)
        attendu = np.array([205 - 48 * np.sin(a), 75 + 48 * np.cos(a), 25])
        self.assertTrue(np.allclose(hg.position_de(T), attendu, atol=1e-9))
        # le repere de travail transmet sa rotation a la cible
        self.assertAlmostEqual(self.decomposer(T)[3], 20.0, places=9)
        self.assertTrue(np.allclose(T[:3, 2], [0, 0, -1], atol=1e-12))

    def test_post_multiplication_deplace_dans_le_repere_outil(self):
        """T @ Trans : le decalage tourne avec l'outil."""
        T = self.cible(200, 0, 100, rz=90)
        p = hg.position_de(T @ hg.translation(30, 0, 0))
        # x de l'outil tourne de 90 deg = +y de la base
        self.assertTrue(np.allclose(p, [200, 30, 100], atol=1e-9))

    def test_pre_multiplication_deplace_dans_le_repere_base(self):
        """Trans @ T : le decalage reste selon les axes de la base."""
        T = self.cible(200, 0, 100, rz=90)
        p = hg.position_de(hg.translation(30, 0, 0) @ T)
        self.assertTrue(np.allclose(p, [230, 0, 100], atol=1e-9))

    def test_degagement_selon_l_axe_outil_monte(self):
        """L'axe z de l'outil pointant vers le bas, un z negatif monte."""
        T = self.cible(200, 0, 40, rz=35)
        haut = T @ hg.translation(0, 0, -85)
        self.assertTrue(np.allclose(hg.position_de(haut), [200, 0, 125],
                                    atol=1e-9))
        self.assertAlmostEqual(self.decomposer(haut)[3], 35.0, places=9)

    def test_orientation_inatteignable_refusee(self):
        from dobot_sim.robot import ErreurCinematique
        T = self.cible(200, 0, 100) @ hg.rot_x(np.radians(30))
        with self.assertRaises(ErreurCinematique) as ctx:
            self.decomposer(T)
        self.assertIn("verticale", str(ctx.exception))

    def test_matrice_non_homogene_refusee(self):
        from dobot_sim.robot import ErreurCinematique
        with self.assertRaises(ErreurCinematique):
            self.decomposer(np.eye(3))
        abimee = self.cible(200, 0, 100)
        abimee[:3, :3] *= 2.0
        with self.assertRaises(ErreurCinematique):
            self.decomposer(abimee)

    def test_cible_coherente_avec_le_mgd(self):
        """Le MGI d'une cible construite par cible() redonne cette cible."""
        T = self.cible(215, -95, 60, rz=25)
        q = self.robot.mgi(T)
        self.assertTrue(np.allclose(self.robot.mgd(q), T, atol=1e-7))

    def test_instruction_reconstruit_sa_matrice(self):
        from dobot_sim.program import Instruction
        T = self.cible(215, -95, 60, rz=25)
        ins = Instruction(type="MOVL", mode="cartesien",
                          pose=list(self.decomposer(T)))
        self.assertTrue(np.allclose(ins.matrice(), T, atol=1e-9))

        art = Instruction(type="MOVJ", mode="articulaire", q=[10, 20, 30, 5])
        self.assertTrue(np.allclose(art.matrice(self.robot),
                                    self.robot.mgd(art.q), atol=1e-12))
        with self.assertRaises(ValueError):
            art.matrice()


class TestRepereOutil(unittest.TestCase):
    """Repere outil general : deport lateral et rotation propre."""

    def setUp(self):
        from dobot_sim.robot import verifier_repere_outil
        self.verifier = verifier_repere_outil
        self.robot = DobotMagician()
        self.T = hg.composer(hg.translation(18, -6, -78),
                             hg.rot_z(np.radians(25)), hg.rot_x(np.pi))

    def test_defaut_deduit_de_la_longueur_d_outil(self):
        r = DobotMagician()
        self.assertTrue(r.outil_est_par_defaut)
        r.geo.lt = 62.0
        attendu = hg.composer(hg.translation(0, 0, -62.0), hg.rot_x(np.pi))
        self.assertTrue(np.allclose(r.repere_outil, attendu, atol=1e-12))

    def test_verification_extrait_le_deport(self):
        d = self.verifier(self.T)
        self.assertTrue(np.allclose(d, [18, -6, -78, 25], atol=1e-9))

    def test_inclinaison_refusee(self):
        from dobot_sim.robot import ErreurCinematique
        with self.assertRaises(ErreurCinematique):
            self.verifier(self.T @ hg.rot_y(np.radians(20)))

    def test_aller_retour_mgd_mgi_avec_outil_deporte(self):
        self.robot.definir_repere_outil(self.T)
        for q in _q_aleatoires(self.robot, 400):
            p0 = self.robot.pose(q)
            p1 = self.robot.pose(self.robot.mgi(self.robot.mgd(q), tolerant=True))
            self.assertTrue(np.allclose(p0[:3], p1[:3], atol=1e-7),
                            "q={} : {} vs {}".format(q, p0, p1))

    def test_la_pointe_atteint_la_consigne(self):
        self.robot.definir_repere_outil(self.T)
        for consigne in [(220, -40, 30, 15), (180, 60, 90, -70), (250, 0, 60, 0)]:
            q = self.robot.mgi_pose(*consigne)
            self.assertTrue(np.allclose(self.robot.pose(q), consigne, atol=1e-7),
                            "{} -> {}".format(consigne, self.robot.pose(q)))

    def test_le_deport_lateral_change_la_configuration(self):
        """Un outil deporte n'est pas equivalent a un simple deport vertical."""
        nu = DobotMagician()
        nu.geo.lt = 78.0
        q_nu = nu.mgi_pose(220, -40, 30, 15)
        self.robot.definir_repere_outil(self.T)
        q_dep = self.robot.mgi_pose(220, -40, 30, 15)
        self.assertGreater(float(np.max(np.abs(q_nu - q_dep))), 1.0)

    def test_retour_au_defaut(self):
        self.robot.definir_repere_outil(self.T)
        self.assertFalse(self.robot.outil_est_par_defaut)
        self.robot.definir_repere_outil(None)
        self.assertTrue(self.robot.outil_est_par_defaut)


class TestRepereObjet(unittest.TestCase):
    """Cibles exprimees dans un repere de travail."""

    def setUp(self):
        from dobot_sim import repere_travail
        from dobot_sim.program import Instruction, Programme
        from dobot_sim.robot import verifier_repere_objet
        self.Instruction, self.Programme = Instruction, Programme
        self.verifier = verifier_repere_objet
        self.repere_travail = repere_travail
        self.robot = DobotMagician()
        self.OBJET = repere_travail(205, 75, 25, rz=20)

    def test_verification_extrait_la_pose(self):
        self.assertTrue(np.allclose(self.verifier(self.OBJET),
                                    [205, 75, 25, 20], atol=1e-9))

    def test_repere_incline_refuse(self):
        from dobot_sim.robot import ErreurCinematique
        with self.assertRaises(ErreurCinematique):
            self.verifier(self.OBJET @ hg.rot_x(0.3))

    def test_coordonnees_locales_resolues(self):
        prog = self.Programme([
            self.Instruction(type="MOVL", mode="cartesien",
                             pose=[0, 48, 0, 0], repere="objet")])
        prog.definir_repere_objet(self.OBJET)
        T = prog.instructions[0].matrice(self.robot, prog.repere_objet)
        a = np.radians(20.0)
        attendu = [205 - 48 * np.sin(a), 75 + 48 * np.cos(a), 25]
        self.assertTrue(np.allclose(hg.position_de(T), attendu, atol=1e-9))

    def test_repere_base_insensible(self):
        prog = self.Programme([
            self.Instruction(type="MOVL", mode="cartesien",
                             pose=[240, 0, 60, 0], repere="base")])
        prog.definir_repere_objet(self.OBJET)
        T = prog.instructions[0].matrice(self.robot, prog.repere_objet)
        self.assertTrue(np.allclose(hg.position_de(T), [240, 0, 60], atol=1e-9))

    def test_deplacer_le_repere_deplace_la_trajectoire(self):
        I, P = self.Instruction, self.Programme
        commun = dict(type="MOVL", mode="cartesien", repere="objet",
                      mode_temps="vitesse", vitesse=120)
        prog = P([I(pose=[0, 0, 0, 0], **commun), I(pose=[0, 48, 0, 0], **commun)])
        prog.definir_repere_objet(self.OBJET)
        p0 = hg.position_de(prog.construire(self.robot).segments[0].T1)
        prog.definir_repere_objet(self.repere_travail(235, 75, 25, rz=20))
        p1 = hg.position_de(prog.construire(self.robot).segments[0].T1)
        self.assertTrue(np.allclose(p1 - p0, [30, 0, 0], atol=1e-9))

    def test_aller_retour_json(self):
        import tempfile
        from dobot_sim.program import Programme
        prog = self.Programme([
            self.Instruction(type="MOVL", mode="cartesien",
                             pose=[0, 48, 0, 0], repere="objet")])
        prog.definir_repere_objet(self.OBJET)
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "p.json")
            prog.enregistrer(chemin)
            relu = Programme.charger(chemin)
        self.assertTrue(np.allclose(relu.repere_objet, self.OBJET, atol=1e-12))
        self.assertEqual(relu.instructions[0].repere, "objet")

    def test_export_en_coordonnees_absolues(self):
        prog = self.Programme([
            self.Instruction(type="MOVL", mode="cartesien",
                             pose=[0, 0, 0, 0], repere="objet")])
        prog.definir_repere_objet(self.OBJET)
        src = prog.exporter_python(self.robot)
        compile(src, "<export>", "exec")
        self.assertIn("205.00, 75.00, 25.00", src)
        self.assertIn("coordonnees ABSOLUES", src)

    def test_pilote_expose_les_deux_reperes(self):
        from dobot_sim import Pilote, cible as _cible
        p = Pilote(vitesse=120)
        p.definir_repere_objet(self.OBJET)
        p.movl(_cible(0, 48, 0), repere="objet")
        ins = p.programme.instructions[0]
        self.assertEqual(ins.repere, "objet")
        T = ins.matrice(p.robot, p.repere_objet)
        self.assertTrue(np.allclose(hg.position_de(T)[2], 25.0, atol=1e-9))
        T_outil = hg.composer(hg.translation(18, 0, -78), hg.rot_x(np.pi))
        p.definir_repere_outil(T_outil)
        self.assertFalse(p.robot.outil_est_par_defaut)


class TestPilote(unittest.TestCase):
    """API de script, celle utilisee depuis VS Code."""

    def setUp(self):
        from dobot_sim import Pilote
        self.Pilote = Pilote

    def test_commandes_acceptent_une_matrice_homogene(self):
        from dobot_sim import cible, repere_travail
        from dobot_sim import homogeneous as hg_

        MAGASIN = repere_travail(215, -95, 25)
        PRISE = cible(repere=MAGASIN)
        HAUT = hg_.translation(0, 0, -85)

        p = self.Pilote(vitesse=120, acceleration=500)
        p.movj(PRISE @ HAUT)
        p.movl(PRISE)
        p.movc(via=cible(265, 0, 135), T=cible(205, 75, 110, rz=20))

        a, b, c = p.programme.instructions
        self.assertTrue(np.allclose(a.pose, [215, -95, 110, 0], atol=1e-9))
        self.assertTrue(np.allclose(b.pose, [215, -95, 25, 0], atol=1e-9))
        self.assertTrue(np.allclose(c.via, [265, 0, 135, 0], atol=1e-9))
        self.assertTrue(np.allclose(c.pose, [205, 75, 110, 20], atol=1e-9))

    def test_matrice_puis_surcharge_par_mot_cle(self):
        from dobot_sim import cible
        p = self.Pilote()
        p.movl(cible(215, -95, 25), z=60)
        self.assertTrue(np.allclose(p.programme.instructions[0].pose,
                                    [215, -95, 60, 0], atol=1e-9))

    def test_le_point_de_passage_ne_devient_pas_la_cible_memorisee(self):
        """Apres un MOVC, la memoire doit retenir la cible, pas le via."""
        from dobot_sim import cible
        p = self.Pilote()
        p.movl(cible(200, 0, 100))
        p.movc(via=cible(265, 0, 135), T=cible(205, 75, 110))
        p.movl(z=25)                      # x et y doivent venir de la cible
        self.assertTrue(np.allclose(p.programme.instructions[2].pose,
                                    [205, 75, 25, 0], atol=1e-9))

    def test_cible_inatteignable_refusee_a_l_ecriture(self):
        from dobot_sim import cible
        from dobot_sim.robot import ErreurCinematique
        from dobot_sim import homogeneous as hg_
        p = self.Pilote()
        with self.assertRaises(ErreurCinematique):
            p.movl(cible(200, 0, 100) @ hg_.rot_y(np.radians(20)))

    def test_construction_chainee_et_coordonnees_omises(self):
        p = self.Pilote(vitesse=100, acceleration=400)
        p.movj(x=220, y=-90, z=60, r=0)
        p.movl(z=25)                      # x, y, r conserves
        ins = p.programme.instructions[1]
        self.assertEqual(ins.type, "MOVL")
        self.assertEqual(ins.pose, [220.0, -90.0, 25.0, 0.0])
        self.assertEqual(ins.mode_temps, "vitesse")
        self.assertEqual(ins.vitesse, 100.0)

    def test_duree_explicite_bascule_en_mode_duree(self):
        p = self.Pilote()
        p.movl(x=220, y=0, z=60, duree=3.0)
        ins = p.programme.instructions[0]
        self.assertEqual(ins.mode_temps, "duree")
        self.assertAlmostEqual(ins.duree, 3.0)
        self.assertAlmostEqual(p.trajectoire().segments[0].duree, 3.0, places=6)

    def test_reglages_par_defaut_et_surcharge_locale(self):
        p = self.Pilote(vitesse=100, acceleration=400, lissage=30)
        p.movl(x=220, y=0, z=60)
        p.regler(vitesse=50)
        p.movl(z=30, lissage=0)
        a, b = p.programme.instructions
        self.assertEqual((a.vitesse, a.lissage), (100.0, 30.0))
        self.assertEqual((b.vitesse, b.lissage), (50.0, 0.0))

    def test_profil_inconnu_refuse(self):
        with self.assertRaises(ValueError):
            self.Pilote().regler(profil="magique")

    def test_cycle_complet_simulable(self):
        p = self.Pilote(vitesse=120, acceleration=500)
        p.repli()
        p.movj(x=215, y=-90, z=110, r=0, vitesse=55, acceleration=250)
        p.movl(z=30, vitesse=50, lissage=50)
        p.ventouse(True)
        p.attendre(0.3)
        p.movl(z=110, lissage=50)
        p.movc(via=(255, 0, 130), x=215, y=90, z=110, r=45)
        p.ventouse(False)
        p.repli()
        traj = p.trajectoire()
        self.assertEqual(len(traj.segments), len(p.programme.instructions))
        self.assertGreater(traj.duree, 0.0)
        e = traj.echantillonner(0.01)
        self.assertTrue(np.all(np.isfinite(e["Q"])))
        self.assertIn("programme", p.resume())
        # aucune butee violee sur ce cycle
        butees = [a for a in p.controler() if "hors de" in a]
        self.assertEqual(butees, [], "\n".join(butees))

    def test_export_du_script_robot_reel(self):
        import tempfile
        p = self.Pilote(vitesse=90, acceleration=300)
        p.movl(x=220, y=0, z=60)
        p.ventouse(True)
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "reel.py")
            p.exporter_python(chemin)
            with open(chemin, encoding="utf-8") as f:
                source = f.read()
        compile(source, "<export>", "exec")
        self.assertIn("SetPTPCoordinateParams", source)


class TestOutils(unittest.TestCase):
    """Catalogue d'outils : maillage, longueur, effet sur le modele."""

    def setUp(self):
        from dobot_sim import outils as ou
        self.ou = ou
        self.robot = DobotMagician()

    def test_tous_les_outils_produisent_un_maillage(self):
        for nom in self.ou.noms_outils():
            o = self.ou.outil(nom)
            for actif in (False, True):
                pieces = o.pieces(actif)
                self.assertGreaterEqual(len(pieces), 1, nom)
                for p in pieces:
                    self.assertEqual(p.repere, 5, nom)   # repere de l'axe outil
                    self.assertGreater(p.n_faces, 0, nom)
                    self.assertTrue(np.all(np.isfinite(p.sommets)), nom)
                    self.assertTrue((p.faces < len(p.sommets)).all(), nom)

    def test_le_maillage_descend_sous_la_bride(self):
        """L'outil occupe les z negatifs du repere 5, et pas au-dela."""
        for nom in self.ou.noms_outils():
            o = self.ou.outil(nom)
            if o.longueur <= 0:
                continue
            zmin = min(float(p.sommets[:, 2].min()) for p in o.pieces(False))
            self.assertLessEqual(zmin, 0.0, nom)
            self.assertGreaterEqual(zmin, -o.longueur - 25.0, nom)

    def test_outil_inconnu_refuse(self):
        with self.assertRaises(ValueError):
            self.ou.outil("tournevis")

    def test_la_longueur_devient_le_deport_lt(self):
        """Avec un outil, z designe la pointe : les angles changent."""
        cible_xyz = (220.0, 0.0, 60.0)
        q_nu = DobotMagician().mgi_pose(*cible_xyz, 0.0)
        for nom in ("ventouse", "pince", "stylo"):
            o = self.ou.outil(nom)
            r = self.robot.avec_outil(o.longueur)
            q = r.mgi_pose(*cible_xyz, 0.0)
            # la pointe est bien a l'altitude demandee
            self.assertAlmostEqual(r.pose(q)[2], cible_xyz[2], places=7)
            # et la configuration differe de celle de la bride nue
            self.assertGreater(float(np.max(np.abs(q[:3] - q_nu[:3]))), 3.0, nom)

    def test_la_pince_bouge_en_se_fermant(self):
        """Seuls les DOIGTS bougent : on les isole par leur altitude."""
        o = self.ou.outil("pince")

        def ecartement(actif):
            V = np.vstack([p.sommets for p in o.pieces(actif)])
            doigts = V[V[:, 2] < -40.0]           # sous le corps de la pince
            self.assertGreater(len(doigts), 0)
            # les doigts se ferment selon y (axe de serrage mesure)
            return float(doigts[:, 1].max() - doigts[:, 1].min())

        ouverte, fermee = ecartement(False), ecartement(True)
        self.assertGreater(ouverte, fermee + 10.0,
                           "ouverte {:.1f} / fermee {:.1f}".format(ouverte, fermee))


class TestEntreesSorties(unittest.TestCase):

    def setUp(self):
        from dobot_sim.es import NB_PORTS, EtatES
        self.EtatES, self.NB_PORTS = EtatES, NB_PORTS
        self.es = EtatES()

    def test_etat_initial_et_acces(self):
        self.assertEqual(len(self.es.sorties), self.NB_PORTS)
        self.assertFalse(any(self.es.sorties))
        self.es.definir_sortie(3, True)
        self.assertTrue(self.es.sortie(3))
        self.assertEqual(self.es.sorties_actives(), [3])
        self.assertTrue(self.es.basculer_entree(7))
        self.assertEqual(self.es.entrees_actives(), [7])

    def test_port_hors_bornes(self):
        for mauvais in (0, 21, -1):
            with self.assertRaises(ValueError):
                self.es.definir_sortie(mauvais, True)

    def test_notification_uniquement_sur_changement(self):
        appels = []
        self.es.abonner(lambda: appels.append(1))
        self.es.definir_sortie(2, True)
        self.es.definir_sortie(2, True)      # sans effet
        self.assertEqual(len(appels), 1)
        self.es.definir_sortie(2, False)
        self.assertEqual(len(appels), 2)

    def test_reinitialisation_selective(self):
        self.es.definir_sortie(1, True)
        self.es.definir_entree(2, True)
        self.es.reinitialiser(entrees=False, sorties=True)
        self.assertFalse(any(self.es.sorties))
        self.assertTrue(self.es.entree(2))


class TestInstructionsES(unittest.TestCase):
    """Instructions SORTIE et ENTREE dans un programme."""

    def setUp(self):
        from dobot_sim.program import Instruction, Programme
        self.Instruction, self.Programme = Instruction, Programme
        self.robot = DobotMagician()

    def _programme(self):
        I = self.Instruction
        return self.Programme([
            I(type="MOVJ", mode="cartesien", pose=[220, -60, 80, 0],
              mode_temps="vitesse", vitesse=60),
            I(type="SORTIE", port=3, etat=True),
            I(type="ENTREE", port=5, etat=True, duree=4.0),
            I(type="OUTIL", etat=True),
            I(type="SORTIE", port=3, etat=False),
        ], nom="es", outil="pince")

    def test_chronologie_des_evenements(self):
        prog = self._programme()
        traj = prog.construire(self.robot)
        types = [e.type for e in prog.evenements]
        self.assertEqual(types, ["SORTIE", "ENTREE", "OUTIL", "SORTIE"])
        self.assertTrue(all(e.t >= 0 for e in prog.evenements))
        # les evenements sont dates sur la trajectoire compilee
        for e in prog.evenements:
            self.assertAlmostEqual(e.t, traj.debuts[e.segment], places=9)

    def test_etat_des_sorties_en_fonction_du_temps(self):
        prog = self._programme()
        prog.construire(self.robot)
        t_on = prog.evenements[0].t
        t_off = prog.evenements[3].t
        self.assertFalse(prog.etat_sorties(t_on - 0.01)[2])
        self.assertTrue(prog.etat_sorties(t_on + 0.01)[2])
        self.assertFalse(prog.etat_sorties(t_off + 0.01)[2])

    def test_delai_d_attente_reserve_dans_la_duree(self):
        prog = self._programme()
        traj = prog.construire(self.robot)
        seg = traj.segments[2]
        self.assertAlmostEqual(seg.duree, 4.0, places=9)
        self.assertEqual(seg.instruction.type, "ENTREE")

    def test_une_attente_neutralise_le_lissage_voisin(self):
        I = self.Instruction
        commun = dict(mode="cartesien", mode_temps="vitesse", vitesse=120)
        prog = self.Programme([
            I(type="MOVL", pose=[200, 0, 100, 0], lissage=100, **commun),
            I(type="ENTREE", port=1, etat=True, duree=2.0),
            I(type="MOVL", pose=[240, 0, 100, 0], lissage=100, **commun),
        ])
        traj = prog.construire(self.robot)
        self.assertEqual(traj.recouvrements[0], 0.0)
        self.assertEqual(traj.recouvrements[1], 0.0)

    def test_une_action_ne_neutralise_PAS_le_lissage_voisin(self):
        """
        Une action -- ventouse, pince, sortie -- ne demande aucun temps :
        elle ne doit donc pas proteger en douce le mouvement qui la
        precede. Lisser un point ou l'on agit, c'est agir avant d'y etre,
        et c'est au programme de ne pas le faire.
        """
        I = self.Instruction
        commun = dict(mode="cartesien", mode_temps="vitesse", vitesse=120)
        for type_action, extra in (("OUTIL", dict(etat=True)),
                                   ("SORTIE", dict(port=3, etat=True))):
            prog = self.Programme([
                I(type="MOVL", pose=[200, 0, 100, 0], lissage=100, **commun),
                I(type=type_action, **extra),
                I(type="MOVL", pose=[240, 0, 100, 0], lissage=100, **commun),
            ])
            traj = prog.construire(self.robot)
            self.assertEqual(traj.segments[1].type_mouvement, "ACTION",
                             type_action + " doit compiler en SegmentAction")
            self.assertGreater(traj.recouvrements[0], 0.0,
                               type_action + " ne doit pas annuler le lissage")

    def test_une_action_lissee_part_avant_la_cible(self):
        """La consequence concrete : l'action part en mouvement."""
        import numpy as np
        I = self.Instruction
        commun = dict(mode="cartesien", mode_temps="vitesse", vitesse=150)
        cible = [220, -60, 25, 0]
        instructions = lambda liss: [
            I(type="MOVL", pose=[220, -60, 110, 0], **commun),
            I(type="MOVL", pose=cible, lissage=liss, **commun),
            I(type="OUTIL", etat=True),
            I(type="MOVL", pose=[220, -60, 110, 0], lissage=liss, **commun),
        ]
        ecarts = {}
        for liss in (0, 100):
            prog = self.Programme(instructions(liss))
            traj = prog.construire(self.robot)
            e = traj.echantillonner(0.001)
            t = prog.evenements[0].t
            i = int(np.searchsorted(e["t"], t))
            ecarts[liss] = float(np.linalg.norm(e["P"][i] - np.array(cible[:3])))
        self.assertLess(ecarts[0], 1e-6, "sans lissage, l'action part sur la cible")
        self.assertGreater(ecarts[100], 0.5, "avec lissage, elle part avant")

    def test_ports_utilises(self):
        prog = self._programme()
        self.assertEqual(prog.ports_utilises(), ([3], [5]))

    def test_export_des_es_et_de_l_outil(self):
        prog = self._programme()
        src = prog.exporter_python(self.robot)
        compile(src, "<export>", "exec")
        self.assertIn("SetIOMultiplexing(api, 3, 2", src)   # 2 = sortie
        self.assertIn("SetIOMultiplexing(api, 5, 3", src)   # 3 = entree
        self.assertIn("SetIODO(api, 3, 1", src)
        self.assertIn("GetIODI(api, 5)", src)
        self.assertIn("SetEndEffectorGripper", src)         # outil = pince
        self.assertIn("import time", src)                   # scrutation ENTREE

    def test_import_time_seulement_si_attente_sur_entree(self):
        from dobot_sim.program import Instruction, Programme
        sans = Programme([Instruction(type="SORTIE", port=1, etat=True)])
        self.assertNotIn("import time", sans.exporter_python(self.robot))
        avec = Programme([Instruction(type="ENTREE", port=1, etat=True)])
        self.assertIn("import time", avec.exporter_python(self.robot))

    def test_export_suit_l_outil_choisi(self):
        from dobot_sim.program import Instruction, Programme
        for nom, attendu in (("ventouse", "SetEndEffectorSuctionCup"),
                             ("pince", "SetEndEffectorGripper"),
                             ("laser", "SetEndEffectorLaser")):
            prog = Programme([Instruction(type="OUTIL", etat=True)],
                             outil=nom)
            self.assertIn(attendu, prog.exporter_python(self.robot), nom)

    def test_aller_retour_json_conserve_outil_et_ports(self):
        import tempfile
        from dobot_sim.program import Programme
        prog = self._programme()
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "p.json")
            prog.enregistrer(chemin)
            relu = Programme.charger(chemin)
        self.assertEqual(relu.outil, "pince")
        self.assertEqual(relu.ports_utilises(), ([3], [5]))

    def test_pilote_expose_les_es_et_l_outil(self):
        from dobot_sim import Pilote
        p = Pilote(outil="pince")
        self.assertAlmostEqual(p.robot.geo.lt, 78.0, places=6)
        p.monter_outil("ventouse")
        self.assertAlmostEqual(p.robot.geo.lt, 62.0, places=6)
        self.assertEqual(p.programme.outil, "ventouse")
        p.sortie(4, True).attendre_entree(6, True, timeout=3.0).pince(True)
        types = [i.type for i in p.programme.instructions]
        self.assertEqual(types, ["SORTIE", "ENTREE", "OUTIL"])
        self.assertEqual(p.programme.instructions[1].duree, 3.0)
        self.assertEqual(p.programme.ports_utilises(), ([4], [6]))


def _chemin_urdf():
    import glob
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trouves = glob.glob(os.path.join(racine, "**", "magician.urdf.xacro"),
                        recursive=True)
    return trouves[0] if trouves else None


@unittest.skipUnless(_chemin_urdf(), "description URDF absente du projet")
class TestURDF(unittest.TestCase):
    """
    Description ROS officielle du constructeur.

    Ces tests valident bien plus que le chargeur : ils confrontent notre
    chaine cinematique a celle publiee par Dobot.
    """

    def setUp(self):
        from dobot_sim import urdf
        self.urdf = urdf
        self.chemin = _chemin_urdf()
        self.d = urdf.charger_urdf(self.chemin)
        self.robot = DobotMagician()

    def test_lecture_des_liaisons_et_des_corps(self):
        for nom in ("magician_joint1", "magician_joint2", "magician_joint3"):
            self.assertIn(nom, self.d.par_nom, nom)
            self.assertEqual(self.d.par_nom[nom]["type"], "revolute")
        avec_maillage = [n for n, s in self.d.corps.items() if s["maillage"]]
        self.assertGreaterEqual(len(avec_maillage), 9)

    def test_lecture_stl_en_millimetres(self):
        import os as _os
        chemin = _os.path.join(_os.path.dirname(_os.path.dirname(self.chemin)),
                               "meshes", "magician_base.STL")
        V, F = self.urdf.lire_stl(chemin)
        self.assertEqual(len(V), 3 * len(F))
        self.assertTrue(np.all(np.isfinite(V)))
        # le socle fait une centaine de millimetres, pas 0,1 (metres)
        etendue = float(np.max(V.max(axis=0) - V.min(axis=0)))
        self.assertGreater(etendue, 50.0)
        self.assertLess(etendue, 400.0)

    def test_geometrie_conforme_a_l_urdf(self):
        """Longueurs de bras lues dans la chaine URDF, en mm."""
        R = self.d.reperes({})                       # toutes liaisons a zero
        epaule = hg.position_de(R["magician_link2"])
        coude = hg.position_de(R["magician_fake_link"])
        poignet = hg.position_de(R["magician_link4"])
        socle = hg.position_de(R["magician_base"])
        g = self.robot.geo
        self.assertAlmostEqual(float(np.linalg.norm(coude - epaule)), g.l1,
                               places=6)
        self.assertAlmostEqual(float(np.linalg.norm(poignet - coude)), g.l2,
                               places=6)
        self.assertAlmostEqual(float(epaule[2] - socle[2]), g.d1, places=6)

    def test_pose_de_reference_identique(self):
        """A q = 0 : bras arriere vertical, avant-bras horizontal."""
        R = self.d.reperes({})
        epaule = hg.position_de(R["magician_link2"])
        coude = hg.position_de(R["magician_fake_link"])
        poignet = hg.position_de(R["magician_link4"])
        self.assertTrue(np.allclose(coude - epaule, [0, 0, self.robot.geo.l1],
                                    atol=1e-9))
        self.assertTrue(np.allclose(poignet - coude, [self.robot.geo.l2, 0, 0],
                                    atol=1e-9))

    def test_convention_articulaire_identique(self):
        """Les angles URDF sont exactement les notres."""
        d2r = np.pi / 180.0
        for q in ([20.0, 0, 0, 0], [0, 25.0, 0, 0], [0, 0, 35.0, 0],
                  [-30.0, 40.0, 55.0, 0]):
            q = np.array(q, dtype=float)
            R = self.d.reperes({"magician_joint1": q[0] * d2r,
                                "magician_joint2": q[1] * d2r,
                                "magician_joint3": q[2] * d2r})
            # poignet URDF, ramene dans notre repere de base
            p_urdf = hg.position_de(R["magician_link4"]) + [0, 0, self.robot.geo.d1]
            p_nous = self.robot.positions_articulations(q)[4]
            self.assertTrue(np.allclose(p_urdf, p_nous, atol=1e-9),
                            "q={} : {} vs {}".format(q, p_urdf, p_nous))

    def test_chaque_corps_est_solidaire_d_un_repere(self):
        """
        Le coeur du raccordement : pour chaque corps il existe un repere de
        notre chaine tel que la transformation entre les deux soit CONSTANTE.
        C'est la preuve que les deux modeles decrivent le meme robot.
        """
        messages = []
        pieces = self.urdf.pieces_depuis_urdf(self.robot, self.chemin,
                                              journal=messages.append)
        self.assertGreaterEqual(len(pieces), 9)
        for m in messages:
            self.assertNotIn("n'est solidaire", m, m)
        # tous les residus annonces doivent etre negligeables
        for m in messages:
            trouve = re.search(r"ecart ([0-9.e+-]+)\)", m)
            if trouve:
                self.assertLess(float(trouve.group(1)), 1e-9, m)
        reperes = sorted({p.repere for p in pieces})
        self.assertEqual(reperes, [0, 1, 2, 3, 4])

    def test_pieces_utilisables_pour_l_affichage(self):
        pieces = self.urdf.pieces_depuis_urdf(self.robot, self.chemin)
        for p in pieces:
            self.assertGreater(p.n_faces, 0, p.nom)
            self.assertTrue(np.all(np.isfinite(p.sommets)), p.nom)
            self.assertTrue((p.faces < len(p.sommets)).all(), p.nom)
            self.assertEqual(len(p.couleur), 4, p.nom)

    def test_le_chargeur_prend_l_urdf_par_defaut(self):
        from dobot_sim import meshes as mm
        pieces = mm.charger_pieces(self.robot)
        self.assertTrue(any(p.nom.startswith("magician_") for p in pieces),
                        [p.nom for p in pieces])


class TestCollisions(unittest.TestCase):
    """
    Enveloppe de collision : plan de pose, embase, colonne.

    Les trois interferences sont reellement atteignables a l'interieur des
    butees -- verifie par balayage de l'espace articulaire -- ce qui est la
    raison d'etre de ce controle.
    """

    def setUp(self):
        self.nu = DobotMagician()                    # bride nue
        self.vent = DobotMagician().avec_outil(62.0)  # ventouse
        self.pince = DobotMagician().avec_outil(78.0)

    def test_poses_usuelles_sans_collision(self):
        for r in (self.nu, self.vent, self.pince):
            for q in ([0, 15, 30, 0], [0, 45, 45, 0], [90, 30, 60, 0],
                      [-90, 30, 60, 0]):
                self.assertEqual(r.collisions(q), [], "{} {}".format(r, q))
                self.assertTrue(r.sans_collision(q))

    def test_plan_de_pose_traverse(self):
        msgs = self.vent.collisions([-135, 85, 90, -90])
        self.assertTrue(any("plan de pose" in m for m in msgs), msgs)

    def test_bride_nue_ne_peut_pas_toucher_la_table(self):
        # sans outil, le point le plus bas du bras reste au-dessus de z = 0
        # sur tout l'espace articulaire : la table est hors d'atteinte
        for q in _q_aleatoires(self.nu, 300, graine=7):
            msgs = self.nu.collisions(q)
            self.assertFalse(any("plan de pose" in m for m in msgs),
                             "{} -> {}".format(np.round(q, 1), msgs))

    def test_colonne_et_embase(self):
        self.assertTrue(any("colonne" in m
                            for m in self.vent.collisions([-117, 20, 95, -90])))
        self.assertTrue(any("embase" in m
                            for m in self.pince.collisions([-135, 0, 95, -90])))

    def test_marge_declenche_avant_le_contact(self):
        q = self.vent.mgi_pose(220.0, 0.0, 4.0, 0.0)   # TCP a 4 mm du plan
        self.assertEqual(self.vent.collisions(q), [])
        msgs = self.vent.collisions(q, marge=10.0)
        self.assertTrue(any("plan de pose" in m for m in msgs), msgs)
        # le libelle ne doit PAS parler de traversee : le plan n'est pas franchi
        self.assertTrue(all("traverse" not in m for m in msgs), msgs)

    def test_profondeur_de_penetration_coherente(self):
        # TCP commande 20 mm sous le plan : la penetration doit valoir 20 mm
        q = self.vent.mgi_pose(220.0, 0.0, -20.0, 0.0)
        msgs = [m for m in self.vent.collisions(q) if "plan de pose" in m]
        self.assertEqual(len(msgs), 1, msgs)
        valeur = float(re.search(r"traverse de ([\d.]+) mm", msgs[0]).group(1))
        self.assertAlmostEqual(valeur, 20.0, places=3)

    def test_defauts_enchaine_butees_et_collisions(self):
        d = self.vent.defauts([-200, 85, 90, -90])
        self.assertTrue(any("hors de [" in m for m in d), d)
        self.assertTrue(any("plan de pose" in m for m in d), d)

    def test_points_mobiles_partent_du_coude(self):
        q = [0.0, 15.0, 30.0, 0.0]
        P = self.vent.points_mobiles(q)
        coude = self.vent.reperes(q)[3][:3, 3]
        tcp = self.vent.reperes(q)[-1][:3, 3]
        np.testing.assert_allclose(P[0], coude, atol=1e-9)
        np.testing.assert_allclose(P[-1], tcp, atol=1e-9)

    def test_programme_de_demonstration_sans_collision(self):
        from dobot_sim.program import programme_demo

        traj = programme_demo().construire(self.vent,
                                           np.array([0.0, 15.0, 30.0, 0.0]))
        chocs = [a for a in traj.controler(0.02)
                 if any(m in a for m in ("colonne", "embase", "plan de pose"))]
        self.assertEqual(chocs, [])


class TestButeesFrolees(unittest.TestCase):
    """
    Une pose peut etre admissible pour le modele et refusee par la machine.

    Cas reel : la cible (175, 0, 0) avec la ventouse exige J3 = 90,05 deg,
    soit 4,95 deg de la butee a 95. Le simulateur l'acceptait sans un mot,
    le vrai controleur l'a refusee et est parti en defaut.
    """

    def setUp(self):
        self.r = DobotMagician().avec_outil(62.0)

    def test_pose_au_large_ne_declenche_rien(self):
        q = self.r.mgi_pose(220.0, 0.0, 50.0, 0.0)
        self.assertEqual(self.r.butees_frolees(q), [])

    def test_le_cas_qui_a_fait_caler_la_machine(self):
        q = self.r.mgi_pose(175.0, 0.0, 0.0, 0.0)
        self.assertEqual(self.r.depassements(q), [])      # dans les butees
        msgs = self.r.butees_frolees(q)                   # mais tout pres
        self.assertTrue(any("avant-bras" in m for m in msgs), msgs)
        marges = self.r.marges(q)
        self.assertAlmostEqual(float(marges[2]), 4.95, places=1)

    def test_marges_coherentes_avec_les_butees(self):
        q = [0.0, 0.0, -10.0, 90.0]                       # trois sur la butee
        m = self.r.marges(q)
        self.assertAlmostEqual(float(m[1]), 0.0)
        self.assertAlmostEqual(float(m[2]), 0.0)
        self.assertAlmostEqual(float(m[3]), 0.0)

    def test_seuil_reglable_et_desactivable(self):
        q = self.r.mgi_pose(175.0, 0.0, 0.0, 0.0)
        self.assertEqual(self.r.butees_frolees(q, marge=1.0), [])
        self.assertTrue(self.r.butees_frolees(q, marge=10.0))
        self.assertEqual(self.r.butees_frolees(q, marge=0.0), [])

    def test_ce_n_est_pas_un_defaut_bloquant(self):
        # defauts() ne doit PAS s'en emparer : la pose reste realisable
        q = self.r.mgi_pose(175.0, 0.0, 0.0, 0.0)
        self.assertEqual(self.r.defauts(q), [])

    def test_signale_par_le_controle_de_trajectoire(self):
        from dobot_sim import Pilote, cible

        p = Pilote(nom="t", outil="ventouse")
        p.movj(cible(175, 0, 0))
        traj = p.programme.construire(self.r, p.q_depart)
        self.assertTrue(any("de sa butee" in a for a in traj.controler(0.05)))


class TestLongueurOutil(unittest.TestCase):
    """
    Longueur d'outil mesuree, par-dessus la valeur nominale du catalogue.

    Cas reel : la ventouse du catalogue fait 62 mm, celle de la machine
    d'essai s'est revelee 17,3 mm plus longue -- d'ou une descente trop
    basse a la prise.
    """

    def test_surcharge_la_valeur_du_catalogue(self):
        from dobot_sim import Pilote

        p = Pilote(nom="t", outil="ventouse")
        self.assertAlmostEqual(p.robot.geo.lt, 62.0)
        p.regler_longueur_outil(79.3)
        self.assertAlmostEqual(p.robot.geo.lt, 79.3)

    def test_monter_outil_remet_la_valeur_nominale(self):
        # d'ou la consigne : regler la longueur APRES monter_outil
        from dobot_sim import Pilote

        p = Pilote(nom="t", outil="ventouse").regler_longueur_outil(79.3)
        p.monter_outil("ventouse")
        self.assertAlmostEqual(p.robot.geo.lt, 62.0)

    def test_la_pointe_descend_d_autant(self):
        from dobot_sim import Pilote

        p = Pilote(nom="t", outil="ventouse")
        q = p.robot.mgi_pose(220.0, 0.0, 50.0, 0.0)
        z62 = float(p.robot.pose(q)[2])
        p.regler_longueur_outil(62.0 + 17.3)
        z79 = float(p.robot.pose(q)[2])
        # meme configuration articulaire, pointe 17,3 mm plus bas
        self.assertAlmostEqual(z62 - z79, 17.3, places=6)

    def test_la_cible_est_atteinte_avec_la_bonne_longueur(self):
        from dobot_sim import Pilote

        p = Pilote(nom="t", outil="ventouse").regler_longueur_outil(79.3)
        q = p.robot.mgi_pose(220.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(p.robot.pose(q)[:3], (220.0, 0.0, 0.0),
                                   atol=1e-9)

    def test_la_longueur_est_inscrite_dans_le_programme(self):
        # sinon elle serait perdue des que le programme change de main
        from dobot_sim import Pilote

        p = Pilote(nom="t", outil="ventouse")
        self.assertAlmostEqual(p.programme.longueur_outil, 62.0)
        p.regler_longueur_outil(79.3)
        self.assertAlmostEqual(p.programme.longueur_outil, 79.3)

    def test_la_longueur_survit_a_l_aller_retour_json(self):
        import tempfile
        from dobot_sim import Pilote
        from dobot_sim.program import Programme
        from dobot_sim.robot import cible

        p = Pilote(nom="t", outil="ventouse").regler_longueur_outil(79.3)
        p.movl(cible(220, 0, 0))
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "p.json")
            p.programme.enregistrer(chemin)
            relu = Programme.charger(chemin)
        self.assertAlmostEqual(relu.longueur_outil, 79.3)

    def test_un_programme_ancien_sans_longueur_reste_lisible(self):
        from dobot_sim.program import Programme

        relu = Programme.depuis_dict({"nom": "vieux", "outil": "ventouse",
                                      "instructions": []})
        self.assertIsNone(relu.longueur_outil)

    def test_le_catalogue_reste_nominal(self):
        # une longueur mesuree appartient au programme, pas au catalogue :
        # celui-ci sert aussi au modele et aux tests, qui doivent rester
        # reproductibles d'une machine a l'autre
        from dobot_sim import Pilote
        from dobot_sim import outils as ou

        Pilote(nom="t", outil="ventouse").regler_longueur_outil(79.3)
        self.assertAlmostEqual(ou.outil("ventouse").longueur, 62.0)
        self.assertAlmostEqual(ou.CATALOGUE["ventouse"].longueur, 62.0)

    def test_la_longueur_change_la_configuration_commandee(self):
        # c'est le point qui compte : deux longueurs -> deux poses du bras
        from dobot_sim import Pilote

        a = Pilote(nom="a", outil="ventouse")
        b = Pilote(nom="b", outil="ventouse").regler_longueur_outil(79.3)
        qa = a.robot.mgi_pose(220.0, 0.0, 0.0, 0.0)
        qb = b.robot.mgi_pose(220.0, 0.0, 0.0, 0.0)
        self.assertGreater(float(np.abs(np.asarray(qa) - np.asarray(qb)).max()),
                           1.0)


class TestProgramme(unittest.TestCase):

    def test_aller_retour_json(self):
        import tempfile
        from dobot_sim.program import programme_demo, Programme

        prog = programme_demo()
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "p.json")
            prog.enregistrer(chemin)
            relu = Programme.charger(chemin)
        self.assertEqual(len(relu), len(prog))
        for a, b in zip(prog.instructions, relu.instructions):
            self.assertEqual(a.type, b.type)
            self.assertEqual(a.mode, b.mode)
            self.assertAlmostEqual(a.duree, b.duree)

    def test_construction_de_la_trajectoire(self):
        from dobot_sim.program import programme_demo

        robot = DobotMagician()
        prog = programme_demo()
        traj = prog.construire(robot)
        self.assertGreater(len(traj), 0)
        self.assertGreater(traj.duree, 0.0)
        e = traj.echantillonner(0.05)
        self.assertTrue(np.all(np.isfinite(e["Q"])))
        self.assertEqual(len(prog.evenements_outil), 2)
        self.assertTrue(prog.etat_outil(prog.evenements_outil[0][0] + 0.01))

    def test_export_python_syntaxiquement_valide(self):
        from dobot_sim.program import programme_demo

        source = programme_demo().exporter_python(DobotMagician())
        compile(source, "<export>", "exec")     # leve SyntaxError si invalide
        self.assertIn("SetPTPCmd", source)
        self.assertIn("SetEndEffectorSuctionCup", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAutoCollision(unittest.TestCase):
    """
    Le bras contre lui-meme.

    Cas reel : quand l'avant-bras se replie, le deport du poignet revient
    sur le bras arriere. Le simulateur laissait passer ce mouvement, que la
    machine ne peut pas executer.
    """

    def setUp(self):
        self.r = DobotMagician().avec_outil(62.0)

    def test_poses_usuelles_sans_auto_collision(self):
        for q in ([0, 15, 30, 0], [0, 45, 45, 0], [90, 30, 60, 0],
                  [-90, 62, 72, 45]):
            self.assertEqual(self.r.auto_collisions(q), [], q)

    def test_le_poignet_revient_sur_le_bras_arriere(self):
        msgs = self.r.auto_collisions([0.0, 0.0, 95.0, -90.0])
        self.assertTrue(any("bras arriere" in m and "poignet" in m
                            for m in msgs), msgs)

    def test_seuil_a_77_degres(self):
        # le contact se produit a J3 - J2 = 77,03 deg
        self.assertEqual(self.r.auto_collisions([0.0, 0.0, 77.0, -90.0]), [])
        self.assertTrue(self.r.auto_collisions([0.0, 0.0, 78.0, -90.0]))

    def test_ne_depend_que_de_la_difference_j3_moins_j2(self):
        # consequence du parallelogramme : a J3 - J2 constant, meme jeu
        for delta in (70.0, 80.0):
            etats = {bool(self.r.auto_collisions([0.0, j2, j2 + delta, -90.0]))
                     for j2 in (0.0, 5.0, 10.0, 15.0)
                     if -10.0 <= j2 + delta <= 95.0}
            self.assertEqual(len(etats), 1, (delta, etats))

    def test_integre_aux_collisions_et_aux_defauts(self):
        q = [0.0, 0.0, 95.0, -90.0]
        self.assertTrue(any("auto-collision" in m
                            for m in self.r.collisions(q)))
        self.assertTrue(any("auto-collision" in m for m in self.r.defauts(q)))

    def test_la_marge_alerte_avant_le_contact(self):
        q = [0.0, 0.0, 74.0, -90.0]
        self.assertEqual(self.r.auto_collisions(q), [])
        self.assertTrue(self.r.auto_collisions(q, marge=15.0))

    def test_programme_de_demonstration_indemne(self):
        from dobot_sim.program import programme_demo

        traj = programme_demo().construire(
            self.r, np.array([0.0, 15.0, 30.0, 0.0]))
        self.assertEqual([a for a in traj.controler(0.02)
                          if "auto-collision" in a], [])
