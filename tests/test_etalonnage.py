"""
Etalonnage camera -> robot (examples/perception/etalonner_camera_robot.py) :
recalage de Kabsch avec decalage dans la pince, et deroule complet sur un
faux robot, une piece et une camera simulees.
"""

import math
import os
import sys
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, ".."))
sys.path.insert(0, os.path.join(ICI, "..", "examples", "perception"))

import etalonner_camera_robot as ec      # noqa: E402


class FauxRobot:
    """Le sous-ensemble de Commande dont l'etalonnage se sert."""

    def __init__(self, x=230.0, y=20.0, z=5.0, r=0.0):
        from dobot_sim import DobotMagician
        self.modele = DobotMagician()
        self.modele.geo.lt = 78.0          # la pince mesuree
        self._p = [x, y, z, r]
        self.fermee = False
        self.postures = []
        self.trace = []

    def pose(self):
        return tuple(self._p)

    def movj(self, x=None, y=None, z=None, r=None, q=None, **kw):
        if q is not None:
            self._p = list(self.modele.pose(q))
            self.postures.append(list(q))
            return
        self.movl(x=x, y=y, z=z, r=r)

    def movl(self, x=None, y=None, z=None, r=None, **kw):
        for i, v in enumerate((x, y, z, r)):
            if v is not None:
                self._p[i] = float(v)
        assert self.modele.atteignable(*self._p), "cible hors d'atteinte {}".format(self._p)
        self.trace.append(tuple(self._p))

    def pince(self, fermee=True, **kw):
        self.fermee = bool(fermee)

    def repli(self):
        self._p = list(self.modele.pose(self.modele.q_home))

    def attendre(self, s):
        pass


def _paires(T, d, n=12, bruit=0.0, graine=1):
    rng = np.random.default_rng(graine)
    out = []
    for _ in range(n):
        x, y, r = rng.uniform(180, 280), rng.uniform(-120, 120), rng.uniform(-60, 60)
        centre = np.array([x, y, ec.HAUTEUR_PIECE]) + ec.rot_z(r) @ [d[0], d[1], 0]
        p = T[:3, :3].T @ (centre - T[:3, 3]) + rng.normal(0, bruit, 3)
        a = math.radians(r - 90.0)
        axe = T[:3, :3].T @ [math.cos(a), math.sin(a), 0]
        out.append(ec.Paire(np.array([x, y, ec.PRISE]), r, p, axe, 0.0))
    return out


class TestRecalage(unittest.TestCase):

    T = ec.regarder([560.0, 30.0, 120.0], [240.0, 0.0, 0.0])

    def test_exact(self):
        rc = ec.recaler(_paires(self.T, (4.0, -2.0)))
        np.testing.assert_allclose(rc.T_mm, self.T, atol=2e-3)
        np.testing.assert_allclose(rc.decalage, [4.0, -2.0], atol=2e-3)
        self.assertAlmostEqual(abs(ec._angle(rc.ecart_r - 90.0, 180.0)), 0.0, places=3)
        self.assertLess(rc.rms, 1e-3)

    def test_bruit(self):
        rc = ec.recaler(_paires(self.T, (4.0, -2.0), n=20, bruit=0.8))
        self.assertLess(np.linalg.norm(rc.t - self.T[:3, 3]), 3.0)
        self.assertLess(np.linalg.norm(rc.decalage - [4.0, -2.0]), 1.0)
        self.assertLess(rc.rms, 2.0)

    def test_point_aberrant_ecarte(self):
        ps = _paires(self.T, (0.0, 0.0), n=12, bruit=0.3)
        ps[5].camera = ps[5].camera + [30.0, 0.0, 0.0]
        rc = ec.recaler(ps)
        self.assertFalse(rc.gardees[5])
        self.assertEqual(int(rc.gardees.sum()), 11)
        self.assertLess(rc.rms, 1.0)

    def test_prise_au_centre(self):
        T = self.T
        rc = ec.recaler(_paires(T, (4.0, -2.0)))
        centre = np.array([250.0, 30.0, ec.HAUTEUR_PIECE])
        a = math.radians(20.0)
        axe = T[:3, :3].T @ [math.cos(a), math.sin(a), 0]
        q, r = rc.prise(T[:3, :3].T @ (centre - T[:3, 3]), axe)
        np.testing.assert_allclose(q, centre, atol=2e-3)
        self.assertAlmostEqual(abs(ec._angle(r - 110.0, 180.0)), 0.0, places=3)

    def test_fichier(self):
        import json
        import tempfile
        et = ec.Etalonnage(FauxRobot(), None, z_table=0.0)
        et.paires = _paires(self.T, (1.0, 0.0))
        et.recalage = ec.recaler(et.paires)
        with tempfile.TemporaryDirectory() as d:
            chemin = et.enregistrer(os.path.join(d, "e.json"))
            T_m = ec.charger_etalonnage(chemin)
            with open(chemin) as f:
                data = json.load(f)
        np.testing.assert_allclose(T_m[:3, 3] * 1000.0, self.T[:3, 3], atol=2e-3)
        self.assertEqual(data["paires"], 12)


class TestDeroule(unittest.TestCase):

    def test_etalonnage_complet(self):
        messages = []
        robot = ec.PieceSimulee(FauxRobot(), decalage=(0.0, 3.0), ecart_r=90.0)
        camera = ec.CameraSimulee(robot, bruit=0.5, graine=3)
        et = ec.Etalonnage(robot, camera, graine=2, journal=messages.append,
                           demander=lambda _t: "")
        et.z_table = 0.0
        et.demarrer()
        for _ in range(15):
            et.cycle()
        rc = et.recalage
        # ce qui compte : l'erreur SUR LA TABLE (la position de la camera,
        # a 50 cm, est sensible au moindre angle)
        T = camera.T
        for x in (180.0, 240.0, 300.0):
            for y in (-100.0, 0.0, 100.0):
                q = np.array([x, y, ec.HAUTEUR_PIECE])
                p = T[:3, :3].T @ (q - T[:3, 3])
                self.assertLess(np.linalg.norm(rc.vers_robot(p) - q), 1.5)
        self.assertLess(rc.rms, 1.5)
        self.assertLess(np.linalg.norm(rc.decalage - [0.0, 3.0]), 1.0)
        # les points ont quitte les abords du depart
        x0, y0, _r = et.depart
        loin = max(np.hypot(p.pince[0] - x0, p.pince[1] - y0) for p in et.paires)
        self.assertGreater(loin, 2 * ec.RAYON_DEPART)
        v = et.valider(3)
        self.assertTrue(all(e["ecart_mm"] < 3.0 for e in v), v)
        self.assertTrue(robot.tenue)

    def test_plusieurs_pieces_vues(self):
        questions = []
        cam = type("C", (), {"K": None, "taille": None})()
        et = ec.Etalonnage(FauxRobot(), cam, z_table=0.0,
                           demander=lambda t: questions.append(t) or "")
        axe = np.array([1.0, 0, 0])
        bonne = ec.Vue(np.array([0.0, 0, 300]), axe, ec.TYPE_PIECE, ec.FACE_PIECE, 0.9)
        # doublon de la meme piece : fondu
        cam.voir = lambda: [bonne, ec.Vue(np.array([5.0, 0, 300]), axe, "fine", "a plat", 0.5)]
        self.assertIs(et.observer(), bonne)
        # une fausse detection loin : la bonne espece de piece l'emporte
        cam.voir = lambda: [bonne, ec.Vue(np.array([200.0, 0, 300]), axe, "fine",
                                          "sur la tranche", 0.6)]
        self.assertIs(et.observer(), bonne)
        # deux epaisses a plat, sans recalage ni pose precedente : on renonce
        autre = ec.Vue(np.array([-150.0, 0, 300]), axe, ec.TYPE_PIECE, ec.FACE_PIECE, 0.8)
        cam.voir = lambda: [bonne, autre]
        self.assertIsNone(et.observer())
        self.assertEqual(questions, [])                  # jamais de question


class TestApproche(unittest.TestCase):

    def test_approche_et_depart_verticaux(self):
        robot = FauxRobot(x=230.0, y=20.0, z=60.0)
        et = ec.Etalonnage(robot, None, z_table=0.0)
        robot.trace.clear()
        et.poser(200.0, -10.0, 10.0)
        t = robot.trace
        # la descente : meme x, y, r, de TRANSIT a PRISE ; la remontee aussi
        bas = [i for i, p in enumerate(t) if abs(p[2] - ec.PRISE) < 1e-9]
        self.assertEqual(len(bas), 1)
        i = bas[0]
        for j in (i - 1, i + 1):
            self.assertEqual(t[j][:2], t[i][:2])
            self.assertGreaterEqual(t[j][2] - t[i][2], ec.APPROCHE)


class TestReprise(unittest.TestCase):

    def _session(self, chemin, graine=6):
        robot = ec.PieceSimulee(FauxRobot(x=200.0, y=0.0), decalage=(0.0, 3.0))
        camera = ec.CameraSimulee(robot, bruit=0.5, graine=graine)
        reponses = []
        et = ec.Etalonnage(robot, camera, graine=graine, journal=lambda m: None,
                           demander=lambda t: reponses.pop(0) if reponses else "")
        et.chemin = chemin
        return et, robot, reponses

    def test_interrompre_puis_nouvelle_piece(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "e.json")
            et, robot, _ = self._session(chemin)
            et.z_table = 0.0
            et.demarrer()
            for _ in range(6):
                et.cycle()
            # Ctrl+C : la piece est en main, on la repose
            ec.interrompu(et)
            self.assertEqual(et.piece["etat"], "posee")
            self.assertFalse(robot.tenue)
            et.enregistrer()
            with open(chemin) as f:
                data = json.load(f)
            self.assertEqual(data["paires"], 6)
            # reprise avec une NOUVELLE piece, decalee autrement dans la pince
            et2, robot2, reponses = self._session(chemin, graine=8)
            robot2.decalage = np.array([0.0, -4.0, 0.0])
            et2.restaurer(data)
            reponses.append("n")
            robot2.robot._p = [et2.depart[0], et2.depart[1], 40.0, et2.depart[2]]
            et2.remettre_piece()
            self.assertEqual(et2.session, 1)
            for _ in range(6):
                et2.cycle()
            rc = et2.recalage
            self.assertEqual(sorted(rc.decalages), [0, 1])
            self.assertLess(np.linalg.norm(rc.decalages[0] - [0.0, 3.0]), 1.5)
            self.assertLess(np.linalg.norm(rc.decalages[1] - [0.0, -4.0]), 1.5)
            self.assertLess(rc.rms, 1.5)

    def test_reprise_piece_remise_a_la_main(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "e.json")
            et, robot, _ = self._session(chemin)
            et.z_table = 0.0
            et.demarrer()
            for _ in range(4):
                et.cycle()
            et.enregistrer()
            with open(chemin) as f:
                data = json.load(f)
            questions = []
            et2 = ec.Etalonnage(robot, et.camera, journal=lambda m: None,
                                demander=lambda t: questions.append(t) or "")
            et2.restaurer(data)
            et2.remettre_piece()
            # la pince s'est ouverte au point de depart, on a demande la piece
            self.assertEqual(et2.session, 1)
            self.assertEqual(et2.piece["etat"], "tenue")
            self.assertTrue(any("entre les mors" in q for q in questions))
            x, y, _z, _r = robot.pose()
            self.assertAlmostEqual(x, et2.depart[0])
            self.assertAlmostEqual(y, et2.depart[1])

    def test_recommencer_garde_la_configuration(self):
        et = ec.Etalonnage(FauxRobot(x=200.0, y=0.0), None, journal=lambda m: None,
                           demander=lambda t: "")
        et.restaurer({"z_table_mm": -40.0, "secteur_base_deg": [-20.0, 30.0],
                      "depart": [200.0, 0.0, 0.0], "piece": {"etat": "absente"}})
        self.assertEqual((et.z_table, et.secteur, len(et.paires)), (-40.0, (-20.0, 30.0), 0))
        self.assertIsNone(et.recalage)

    def test_depart_ramene_dans_un_secteur_plus_etroit(self):
        robot = FauxRobot(x=200.0, y=0.0)
        et = ec.Etalonnage(robot, None, z_table=25.0, journal=lambda m: None,
                           secteur=(10.0, 40.0))
        et.depart = (200.0, -60.0, 0.0)          # J1 ~ -17 deg : hors secteur
        et._recentrer_depart()
        x, y, r = et.depart
        self.assertTrue(et._atteignable(x, y, r))
        self.assertAlmostEqual(math.degrees(math.atan2(y, x)), 25.0, delta=1.0)

    def test_charger_inacheve(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "e.json")
            with open(chemin, "w") as f:
                json.dump({"paires": 2, "z_table_mm": 0.0}, f)
            with self.assertRaises(ValueError):
                ec.charger_etalonnage(chemin)


class TestMargeButees(unittest.TestCase):

    def test_essai_du_25_10(self):
        """Depart au bout de la portee (x 305) : l'essai reel avait vise
        x 333,7, bras arriere a 1,1 deg de sa butee -- alarme 0x12."""
        robot = FauxRobot()
        robot.modele.geo.lt = 78.0
        et = ec.Etalonnage(robot, None, z_table=25.0, graine=0,
                           secteur=(-40.9, 29.0))
        et.depart = (305.0, -1.7, -0.3)
        for _ in range(30):
            x, y, r = et.point_suivant()
            for h in (ec.PRISE, ec.TRANSIT):
                q = robot.modele.mgi_pose(x, y, 25.0 + h, r)
                self.assertGreaterEqual(min(robot.modele.marges(q)), ec.MARGE_BUTEE)
            et.paires.append(ec.Paire(np.array([x, y, 30.0]), r, np.zeros(3),
                                      np.array([1.0, 0, 0]), 25.0))


class TestHorsChamp(unittest.TestCase):

    def test_piece_non_vue_on_continue(self):
        robot = ec.PieceSimulee(FauxRobot(x=200.0, y=0.0), decalage=(0.0, 2.0))
        camera = ec.CameraSimulee(robot, bruit=0.5, graine=4)
        vraie = camera.voir
        n = {"k": 0}

        def voir():
            n["k"] += 1
            return [] if n["k"] == 3 else vraie()       # 3e image : rien de vu
        camera.voir = voir
        messages = []
        et = ec.Etalonnage(robot, camera, z_table=0.0, graine=5,
                           journal=messages.append, demander=lambda t: "q")
        et.demarrer()
        resultats = [et.cycle() for _ in range(6)]
        self.assertIsNone(resultats[2])
        self.assertEqual(len(et.paires), 5)
        self.assertEqual(len(et.interdits), 1)
        self.assertTrue(robot.tenue)                     # la piece a ete reprise
        self.assertTrue(any("NON VUE" in m for m in messages))


class TestDefaut(unittest.TestCase):

    def test_point_refuse_banni_et_on_continue(self):
        class Li:
            def __init__(self):
                self.effacees = 0

            def arret_urgence(self):
                pass

            def effacer_alarmes(self):
                self.effacees += 1

            def alarmes(self):
                return []

        robot = FauxRobot(x=200.0, y=0.0)
        robot._pont = type("P", (), {"liaison": Li()})()
        et = ec.Etalonnage(robot, None, z_table=0.0, journal=lambda m: None)
        et.depart = (200.0, 0.0, 0.0)
        et.vise = (210.0, 10.0, 0.0)
        et.piece = {"etat": "tenue"}
        self.assertTrue(et.reprendre_apres_defaut())
        self.assertEqual(et.interdits, [(210.0, 10.0)])
        for _ in range(20):
            x, y, _r = et.point_suivant()
            self.assertGreaterEqual(math.hypot(x - 210.0, y - 10.0), 25.0)

    def test_defaut_persistant(self):
        class Li:
            def arret_urgence(self):
                pass

            def effacer_alarmes(self):
                pass

            def alarmes(self):
                return [18]

        robot = FauxRobot()
        robot._pont = type("P", (), {"liaison": Li()})()
        et = ec.Etalonnage(robot, None, z_table=0.0, journal=lambda m: None)
        self.assertFalse(et.reprendre_apres_defaut())


class TestFaussesPieces(unittest.TestCase):

    T = ec.regarder([560.0, 30.0, 120.0], [240.0, 0.0, 0.0])

    def test_ransac_ecarte_une_paire_fausse_des_4_paires(self):
        ps = _paires(self.T, (0.0, 0.0), n=4, bruit=0.3, graine=3)
        ps.append(_paires(self.T, (0.0, 0.0), n=1, graine=9)[0])
        ps[1].camera = ps[1].camera + [120.0, -40.0, 10.0]     # une fausse piece
        rc = ec.recaler(ps)
        self.assertFalse(rc.gardees[1])
        self.assertLess(rc.rms, 1.0)

    def test_coherence_des_distances_designe_la_vraie_piece(self):
        robot = ec.PieceSimulee(FauxRobot(x=200.0, y=0.0), decalage=(0.0, 0.0))
        camera = ec.CameraSimulee(robot, bruit=0.3, graine=2)
        vraie = camera.voir
        T = camera.T
        # un objet du decor, fixe, pris pour une piece a chaque image
        decor = ec.Vue(T[:3, :3].T @ (np.array([330.0, -90.0, 17.0]) - T[:3, 3]),
                       np.array([1.0, 0, 0]), ec.TYPE_PIECE, ec.FACE_PIECE, 0.95)
        camera.voir = lambda: vraie() + [decor]
        et = ec.Etalonnage(robot, camera, z_table=0.0, graine=4, journal=lambda m: None,
                           demander=lambda t: "")
        et.demarrer()
        et.degager()
        et.relever_decor()                     # bras en equerre : seul le decor
        self.assertEqual(len(et.fantomes), 1)
        for _ in range(8):
            et.cycle()
        self.assertGreaterEqual(len(et.paires), 7)
        self.assertLess(et.recalage.rms, 1.5)

    def test_sans_releve_du_decor_la_coherence_suffit(self):
        robot = ec.PieceSimulee(FauxRobot(x=200.0, y=0.0), decalage=(0.0, 0.0))
        camera = ec.CameraSimulee(robot, bruit=0.3, graine=5)
        vraie = camera.voir
        T = camera.T
        decor = ec.Vue(T[:3, :3].T @ (np.array([330.0, -90.0, 17.0]) - T[:3, 3]),
                       np.array([1.0, 0, 0]), "fine", "a plat", 0.95)
        camera.voir = lambda: vraie() + [decor]
        et = ec.Etalonnage(robot, camera, z_table=0.0, graine=6, journal=lambda m: None,
                           demander=lambda t: "")
        et.demarrer()
        for _ in range(10):
            et.cycle()
        self.assertGreaterEqual(len(et.paires), 8)
        self.assertLess(et.recalage.rms, 1.5)


class TestSecteur(unittest.TestCase):

    def test_poses_dans_le_secteur(self):
        robot = ec.PieceSimulee(FauxRobot(x=230.0, y=60.0), decalage=(0.0, 2.0))
        camera = ec.CameraSimulee(robot, bruit=0.5, graine=4)
        et = ec.Etalonnage(robot, camera, z_table=0.0, graine=5, journal=lambda m: None,
                           demander=lambda _t: "", secteur=(5.0, 45.0))
        et.demarrer()
        for _ in range(10):
            et.cycle()
        j1 = [et.j1(*p) for p in robot.robot.trace]
        self.assertGreaterEqual(min(j1), 5.0)
        self.assertLessEqual(max(j1), 45.0)
        # degagement en posture d'equerre, sans quitter le secteur
        self.assertTrue(robot.robot.postures)
        for q in robot.robot.postures:
            self.assertEqual((q[1], q[2]), (ec.EQUERRE_J2, ec.EQUERRE_J3))
            self.assertTrue(5.0 <= q[0] <= 45.0)
        self.assertLess(et.recalage.rms, 1.5)

    def test_relever_secteur_au_clavier(self):
        robot = FauxRobot(x=230.0, y=0.0, z=60.0)
        touches = iter(["1", "b", "b", "\n", "1", "B", "B", "B", "B", "B", "B", "\n"])
        sect = ec.relever_secteur(robot, robot.modele, lire=lambda: next(touches),
                                  journal=lambda m: None)
        self.assertAlmostEqual(sect[0], -10.0, places=3)
        self.assertAlmostEqual(sect[1], 20.0, places=3)

    def test_clavier_refuse_hors_secteur(self):
        robot = FauxRobot(x=230.0, y=0.0, z=60.0)
        et = ec.Etalonnage(robot, None, z_table=0.0, secteur=(-5.0, 5.0))
        touches = iter(["1", "B", "B", "\n"])
        messages = []
        ec.piloter(robot, lambda: next(touches), messages.append,
                   autorise=lambda x, y, z, r: et.dans_secteur(x, y, z, r, 0.0))
        self.assertTrue(any("hors du secteur" in m for m in messages))
        # le premier pas (5 deg) est accepte, le second refuse
        self.assertAlmostEqual(et.j1(*robot.pose()), 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
