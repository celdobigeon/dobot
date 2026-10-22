# -*- coding: utf-8 -*-
"""
Lecture d'un SVG et fabrication du programme de trace.

Les verifications portent sur ce qui se voit sur le papier : la geometrie
doit etre exacte -- un arc de rayon 15 fait 15 -- et le programme doit
poser le stylo a l'arret au point, sinon le trait commence avant d'etre
arrive.
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import (ErreurSVG, Pilote, lire_svg, programme_depuis_svg,
                       repere_travail)
from dobot_sim.svg import (aplatir_arc, appliquer, composer, lire_chemin,
                           lire_transform)

SVG_ESSAI = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
  <defs><rect id="jamais" x="0" y="0" width="999" height="999"/></defs>
  <path d="M 10 10 H 50 V 50 H 10 Z"/>
  <path d="m 60 10 h 40 v 40 h -40 z"/>
  <path d="M 110 50 C 120 10, 140 10, 150 50 S 180 90, 190 50"/>
  <path d="M 10 80 A 15 15 0 0 1 40 80"/>
  <circle cx="60" cy="80" r="12"/>
  <ellipse cx="100" cy="80" rx="16" ry="8"/>
  <rect x="125" y="68" width="30" height="24" rx="6" ry="6"/>
  <line x1="165" y1="68" x2="195" y2="92"/>
  <polyline points="165,92 175,80 185,92"/>
  <polygon points="188,70 196,70 192,78"/>
  <g transform="translate(20,55) scale(0.5) rotate(45)">
    <rect x="0" y="0" width="20" height="20"/>
  </g>
  <text x="100" y="30">ignore</text>
</svg>
"""


def _fichier(contenu, suffixe=".svg"):
    fd, chemin = tempfile.mkstemp(suffix=suffixe)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(contenu)
    return chemin


class TestLectureSVG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chemin = _fichier(SVG_ESSAI)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.chemin)

    # -- geometrie des chemins ---------------------------------------------

    def test_chemin_absolu_et_relatif_donnent_le_meme_carre(self):
        a = lire_chemin("M 10 10 H 50 V 50 H 10 Z")[0]
        b = lire_chemin("m 10 10 h 40 v 40 h -40 z")[0]
        self.assertEqual(len(a), 5)
        for pa, pb in zip(a, b):
            self.assertAlmostEqual(pa[0], pb[0], places=9)
            self.assertAlmostEqual(pa[1], pb[1], places=9)

    def test_un_Z_referme_la_polyligne(self):
        poly = lire_chemin("M 0 0 L 10 0 L 10 10 Z")[0]
        self.assertEqual(poly[0], poly[-1])

    def test_arc_elliptique_exact(self):
        """Un arc de rayon 15 doit faire 15, pas 14,8."""
        pts = aplatir_arc((10, 80), 15, 15, 0, False, True, (40, 80), 0.02)
        rayons = [math.hypot(x - 25, y - 80) for x, y in pts]
        self.assertAlmostEqual(min(rayons), 15.0, places=6)
        self.assertAlmostEqual(max(rayons), 15.0, places=6)

    def test_la_tolerance_commande_la_finesse(self):
        grossier = lire_chemin("M 0 0 C 0 50, 50 50, 50 0", tol=5.0)[0]
        fin = lire_chemin("M 0 0 C 0 50, 50 50, 50 0", tol=0.05)[0]
        self.assertLess(len(grossier), len(fin))

    def test_cubique_lisse_reflete_le_point_de_controle(self):
        """Le S reprend le miroir du dernier controle : la courbe est C1."""
        poly = lire_chemin("M 0 0 C 10 10, 20 10, 30 0 S 50 -10, 60 0",
                           tol=0.01)[0]
        i = min(range(len(poly)), key=lambda k: abs(poly[k][0] - 30.0))
        avant = np.array(poly[i]) - np.array(poly[i - 2])
        apres = np.array(poly[i + 2]) - np.array(poly[i])
        cos = (avant @ apres) / (np.linalg.norm(avant) * np.linalg.norm(apres))
        self.assertGreater(cos, 0.999)

    # -- transformations ----------------------------------------------------

    def test_transform_compose_de_gauche_a_droite(self):
        m = lire_transform("translate(20,55) scale(0.5) rotate(45)")
        self.assertAlmostEqual(appliquer(m, (0, 0))[0], 20.0, places=9)
        self.assertAlmostEqual(appliquer(m, (0, 0))[1], 55.0, places=9)
        x, y = appliquer(m, (20, 0))
        self.assertAlmostEqual(x, 20 + 10 * math.cos(math.radians(45)), places=9)
        self.assertAlmostEqual(y, 55 + 10 * math.sin(math.radians(45)), places=9)

    def test_rotation_autour_d_un_point(self):
        m = lire_transform("rotate(90 10 10)")
        x, y = appliquer(m, (10, 0))
        self.assertAlmostEqual(x, 20.0, places=9)
        self.assertAlmostEqual(y, 10.0, places=9)

    def test_composition_identite(self):
        m = lire_transform("translate(3,4)")
        self.assertEqual(composer(m, (1, 0, 0, 1, 0, 0)), m)

    # -- document complet ---------------------------------------------------

    def test_defs_et_texte_ignores(self):
        polys = lire_svg(self.chemin, 0.2)
        xs = [p[0] for q in polys for p in q]
        self.assertLess(max(xs), 500, "le rect de <defs> a ete trace")
        self.assertGreaterEqual(len(polys), 11)

    def test_formes_de_base_lues(self):
        """Le cercle r=12 centre en (60,80) doit s'y trouver."""
        polys = lire_svg(self.chemin, 0.05)
        cercle = [q for q in polys
                  if all(abs(math.hypot(x - 60, y - 80) - 12) < 0.05 for x, y in q)]
        self.assertEqual(len(cercle), 1)

    def test_fichier_absent_ou_illisible(self):
        with self.assertRaises(ErreurSVG):
            lire_svg(os.path.join(tempfile.gettempdir(), "pas_la_du_tout.svg"))
        casse = _fichier('<svg><path d="M 0 0 L')
        try:
            with self.assertRaises(ErreurSVG):
                lire_svg(casse)
        finally:
            os.unlink(casse)

    def test_document_sans_geometrie(self):
        vide = _fichier('<svg xmlns="http://www.w3.org/2000/svg">'
                        '<text x="1" y="1">a</text></svg>')
        try:
            with self.assertRaises(ErreurSVG) as ctx:
                lire_svg(vide)
            self.assertIn("texte", str(ctx.exception))
        finally:
            os.unlink(vide)


class TestProgrammeDepuisSVG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chemin = _fichier(SVG_ESSAI)
        cls.feuille = repere_travail(x=235, y=0, z=10)
        cls.robot = programme_depuis_svg(cls.chemin, repere=cls.feuille,
                                         largeur=80)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.chemin)

    def test_un_pilote_pret_a_jouer(self):
        self.assertIsInstance(self.robot, Pilote)
        self.assertGreater(len(self.robot.programme.instructions), 20)

    def test_aucune_violation_de_butee(self):
        self.assertEqual(self.robot.controler(), [])

    def test_le_trace_a_la_largeur_demandee(self):
        e = self.robot.trajectoire().echantillonner(0.02)
        P = e["P"]
        papier = P[np.abs(P[:, 2] - 10.0) < 0.05]
        self.assertAlmostEqual(np.ptp(papier[:, 0]), 80.0, delta=1.0)

    def test_les_proportions_sont_conservees(self):
        """Le dessin fait 186 x 82 unites : le rapport doit survivre."""
        polys = lire_svg(self.chemin, 0.2)
        xs = [p[0] for q in polys for p in q]
        ys = [p[1] for q in polys for p in q]
        attendu = (max(ys) - min(ys)) / (max(xs) - min(xs))
        e = self.robot.trajectoire().echantillonner(0.02)
        P = e["P"]
        papier = P[np.abs(P[:, 2] - 10.0) < 0.05]
        obtenu = np.ptp(papier[:, 1]) / np.ptp(papier[:, 0])
        self.assertAlmostEqual(obtenu, attendu, delta=0.05)

    def test_le_stylo_se_pose_a_l_arret_au_point(self):
        """
        Le premier et le dernier point d'un trait ne sont jamais lisses :
        un trait doit commencer et finir ou on l'a dit.
        """
        ins = self.robot.programme.instructions
        poses = [i for i, x in enumerate(ins) if x.commentaire == "pose du stylo"]
        fins = [i for i, x in enumerate(ins) if x.commentaire == "fin du trait"]
        self.assertTrue(poses and fins)
        for i in poses + fins:
            self.assertEqual(ins[i].lissage, 0.0,
                             "instruction %d lissee alors qu'elle touche le papier" % i)

    def test_les_points_interieurs_sont_lisses(self):
        ins = self.robot.programme.instructions
        interieurs = [x for x in ins
                      if x.type == "MOVL" and not x.commentaire]
        self.assertTrue(interieurs)
        self.assertTrue(all(x.lissage > 0 for x in interieurs))

    def test_la_hauteur_de_levee_est_respectee(self):
        """
        On lit les instructions et non la trajectoire : celle-ci se
        termine par un repli, qui monte bien plus haut que la levee.
        """
        robot = programme_depuis_svg(self.chemin, repere=self.feuille,
                                     largeur=60, levee=25.0)
        hauts = [x.pose[2] for x in robot.programme.instructions
                 if x.commentaire in ("liaison", "levee")]
        self.assertTrue(hauts)
        for z in hauts:
            self.assertAlmostEqual(z, 10.0 + 25.0, places=6)

    def test_le_dessin_est_centre_sur_le_repere(self):
        """Centre de la BOITE, pas moyenne des points : l'echantillonnage
        est plus dense la ou le robot ralentit."""
        e = self.robot.trajectoire().echantillonner(0.02)
        P = e["P"]
        papier = P[np.abs(P[:, 2] - 10.0) < 0.05]
        cx = (papier[:, 0].min() + papier[:, 0].max()) / 2.0
        cy = (papier[:, 1].min() + papier[:, 1].max()) / 2.0
        self.assertAlmostEqual(cx, 235.0, delta=0.5)
        self.assertAlmostEqual(cy, 0.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
