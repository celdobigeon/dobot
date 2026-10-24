#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests de la chaine d'apprentissage de la perception (examples/perception/
perception.py et examples/perception/apprentissage/), SANS camera ni
reseau ni poids : scenes synthetiques rendues par lancer de rayons.

  - projection profondeur -> couleur (identite et vraie extrinseque) ;
  - ajustement d'un parallelepipede de taille connue a un masque connu ;
  - Perception de bout en bout avec un FAUX reseau (masques connus) ;
  - export YOLO-seg (polygones normalises, data.yaml, decoupage stable) ;
  - copier-coller (occultation, echelle perspective) ;
  - lecture de la geometrie couleur par le serveur (faux pyrealsense2).
"""

import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

try:
    import cv2
except ImportError:                               # pragma: no cover
    cv2 = None

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERCEPTION = os.path.join(RACINE, "examples", "perception")
APPRENTISSAGE = os.path.join(PERCEPTION, "apprentissage")
for d in (PERCEPTION, APPRENTISSAGE):
    if d not in sys.path:
        sys.path.insert(0, d)

K = np.array([[420.0, 0.0, 424.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]])
W, H = 848, 480
TYPES = {"epaisse": (17.0, 19.0, 68.0), "fine": (8.0, 17.0, 68.0)}


def rendre(K, W, H, T_ct, boites):
    """Profondeur (m) d'une table (plan z = 0 du repere table) portant des
    boites [(T_t du sommet, L, l, h) en m] : un rayon par pixel, methode
    des dalles."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)],
                 -1).reshape(-1, 3)
    T_tc = np.linalg.inv(T_ct)
    o, d = T_tc[:3, 3], r @ T_tc[:3, :3].T
    t = -o[2] / d[:, 2]
    t[t <= 0] = np.inf
    for (T, L, l, h) in boites:
        Ti = np.linalg.inv(T)
        o2, d2 = Ti[:3, :3] @ o + Ti[:3, 3], d @ Ti[:3, :3].T
        lo, hi = np.array([-L / 2, -l / 2, -h]), np.array([L / 2, l / 2, 0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - o2) / d2, (hi - o2) / d2
        tmin = np.nanmax(np.minimum(a, b), 1)
        tmax = np.nanmin(np.maximum(a, b), 1)
        touche = (tmax >= tmin) & (tmin > 0) & (tmin < t)
        t = np.where(touche, tmin, t)
    return np.where(np.isfinite(t), t, 0.0).reshape(H, W).astype(np.float32)


def table_vue_de_camera():
    """Repere table vu d'une camera a 45 cm au-dessus de la table, inclinee
    de 50 degres vers le bas (vue en biais, comme sur le poste)."""
    import vision
    a = np.radians(50.0)
    n = np.array([0.0, -np.cos(a), -np.sin(a)])      # normale vers la camera
    return vision.repere_table(n, 0.45)


def besoin_cv2(test):
    return unittest.skipIf(cv2 is None, "OpenCV absent")(test)


@besoin_cv2
class TestGeometrie(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import perception
        cls.pc = perception
        T_ct = table_vue_de_camera()
        # coordonnees table du point vise par l'axe optique : les pieces
        # sont posees autour, dans le champ
        z = 0.45 / np.sin(np.radians(50.0))
        cx, cy = (np.linalg.inv(T_ct) @ [0.0, 0.0, z, 1.0])[:2]
        relatives = [(-0.04, 0.0, 0.5, "fine", "a plat"),
                     (0.05, 0.02, -1.0, "epaisse", "sur la tranche"),
                     (0.00, -0.06, 0.2, "epaisse", "debout"),
                     (0.03, 0.08, 1.3, "fine", "sur la tranche")]
        cls.poses = [(cx + x, cy + y, th, typ, face) for (x, y, th, typ, face) in relatives]
        boites = []
        for (x, y, th, typ, face) in cls.poses:
            L, l, h = (v / 1000.0 for v in perception.forme(TYPES[typ], face))
            boites.append((perception.pose_table(x, y, th, h), L, l, h))
        D = rendre(K, W, H, T_ct, boites)
        rng = np.random.default_rng(0)
        D = D + (rng.normal(0, 0.0006, D.shape) * (D > 0)).astype(np.float32)
        cls.D = D
        cls.table = perception.analyser_table(D, K)
        # une "camera couleur" decalee de 15 mm et tournee de 1 degre
        Tcp = np.eye(4)
        c, s = np.cos(np.radians(1.0)), np.sin(np.radians(1.0))
        Tcp[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        Tcp[:3, 3] = [0.015, 0.0, 0.0]
        Kc = np.array([[610.0, 0, 430.0], [0, 610.0, 245.0], [0, 0, 1]])
        cls.capture = {"profondeur": D, "K": K, "coeffs": np.zeros(5),
                       "couleur": np.full((H, W, 3), 128, np.uint8),
                       "couleur_brute": np.full((H, W, 3), 128, np.uint8),
                       "K_couleur": Kc, "coeffs_couleur": np.zeros(5),
                       "T_couleur_profondeur": Tcp}
        cls.vue = perception.vue_couleur(cls.capture)

    def test_table(self):
        n = self.table.n
        self.assertGreater(abs(n @ self.pc.analyser_table(self.D, K).n), 0.9999)
        self.assertLess(self.table.bruit, 0.002)

    def test_vue_brute_ou_recalee(self):
        self.assertEqual(self.vue.source, "couleur_brute")
        sans = {k: v for k, v in self.capture.items() if k != "T_couleur_profondeur"}
        v2 = self.pc.vue_couleur(sans)
        self.assertEqual(v2.source, "couleur")
        np.testing.assert_allclose(v2.T_cp, np.eye(4))

    def test_projection_identite(self):
        """Couleur recalee : chaque pixel de profondeur tombe sur lui-meme."""
        vue = self.pc.vue_couleur({"profondeur": self.D, "K": K,
                                   "couleur": self.capture["couleur"]})
        u, v, ok = self.pc.projeter_profondeur(self.D, K, vue)
        vv, uu = np.mgrid[0:H, 0:W]
        self.assertTrue(ok.mean() > 0.5)
        np.testing.assert_allclose(u[ok], uu[ok], atol=1e-6)
        np.testing.assert_allclose(v[ok], vv[ok], atol=1e-6)

    def test_projection_extrinseque(self):
        """Un pixel de profondeur -> point 3D -> repere couleur -> pixel :
        compare au calcul a la main, et rayons() inverse projeter()."""
        u, v, ok = self.pc.projeter_profondeur(self.D, K, self.vue)
        r, c = 300, 500
        z = self.D[r, c]
        p = np.array([(c - K[0, 2]) * z / K[0, 0], (r - K[1, 2]) * z / K[1, 1], z])
        q = self.vue.T_cp[:3, :3] @ p + self.vue.T_cp[:3, 3]
        Kc = self.vue.K
        self.assertAlmostEqual(u[r, c], Kc[0, 0] * q[0] / q[2] + Kc[0, 2], places=6)
        self.assertAlmostEqual(v[r, c], Kc[1, 1] * q[1] / q[2] + Kc[1, 2], places=6)
        uv = self.pc.projeter(np.array([q]), self.vue)
        ray = self.pc.rayons(uv, self.vue)[0]
        np.testing.assert_allclose(ray * q[2], q, atol=1e-9)

    def _masque(self, i):
        x, y, th, typ, face = self.poses[i]
        dims = self.pc.forme(TYPES[typ], face)
        T_t = self.pc.pose_table(x, y, th, dims[2] / 1000.0)
        poly = self.pc.silhouette(T_t, dims, self.table, self.vue)
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, poly, 1)
        return m > 0

    def _verifier(self, aj, i, tol_mm=1.5, tol_deg=2.5):
        x, y, th, typ, face = self.poses[i]
        self.assertIsNotNone(aj)
        self.assertEqual((aj.type, aj.face), (typ, face))
        self.assertGreater(aj.iou, 0.9)
        np.testing.assert_allclose(aj.T_table[:2, 3] * 1000, [x * 1000, y * 1000], atol=tol_mm)
        th2 = np.arctan2(aj.T_table[1, 0], aj.T_table[0, 0])
        dth = (th2 - th + np.pi / 2) % np.pi - np.pi / 2
        self.assertLess(abs(np.degrees(dth)), tol_deg)

    def test_ajustement_masque_connu(self):
        proj = self.pc.projeter_profondeur(self.D, K, self.vue)
        for i in range(len(self.poses)):
            with self.subTest(piece=i):
                aj = self.pc.ajuster_masque(self._masque(i), self.table, self.vue, proj, TYPES)
                self._verifier(aj, i)
                self.assertTrue(np.isfinite(aj.h_mesure))

    def test_ajustement_sans_profondeur(self):
        """Piece entierement saturee (aucune mesure dans le masque) : la
        silhouette seule suffit pour une piece a plat."""
        vide = np.zeros_like(self.D)
        proj = self.pc.projeter_profondeur(vide, K, self.vue)
        aj = self.pc.ajuster_masque(self._masque(0), self.table, self.vue, proj, TYPES)
        self._verifier(aj, 0)
        self.assertFalse(np.isfinite(aj.h_mesure))

    def test_masque_qui_n_est_pas_une_piece(self):
        """Un disque (une gomme, un reflet) : aucune piece ne l'explique bien."""
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (420, 330), 45, 1, -1)
        proj = self.pc.projeter_profondeur(self.D, K, self.vue)
        aj = self.pc.ajuster_masque(m > 0, self.table, self.vue, proj, TYPES)
        self.assertTrue(aj is None or aj.iou < 0.45)

    def test_perception_faux_reseau(self):
        """Toute la chaine Perception, le reseau remplace par des masques connus."""
        masques = np.stack([self._masque(i) for i in range(len(self.poses))]
                           + [np.zeros((H, W), bool)])
        disque = np.zeros((H, W), np.uint8)
        cv2.circle(disque, (150, 420), 30, 1, -1)
        masques[-1] = disque > 0
        noms = {0: "epaisse", 1: "fine"}
        cls = np.array([1, 0, 0, 1, 0], float)
        conf = np.array([0.9, 0.8, 0.85, 0.7, 0.6])

        class Tenseur:
            def __init__(self, a):
                self.a = a

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        class FauxReseau:
            names = noms

            def predict(self, image, **kw):
                r = types.SimpleNamespace(
                    masks=types.SimpleNamespace(data=Tenseur(masques.astype(np.float32))),
                    boxes=types.SimpleNamespace(cls=Tenseur(cls), conf=Tenseur(conf)))
                return [r]

        T_rc = np.eye(4)
        T_rc[:3, 3] = [0.2, 0.0, 0.0]
        p = self.pc.Perception(types=TYPES, modele=False, T_robot_camera=T_rc)
        p.reseau, p.imgsz = FauxReseau(), 864
        pieces = p.detecter(self.capture)
        self.assertEqual(len(pieces), len(self.poses))      # le disque est rejete
        for piece in pieces:
            self.assertIsInstance(piece, self.pc.vision.Piece)
            proche = min(range(len(self.poses)), key=lambda i: np.hypot(
                piece.T_table[0, 3] - self.poses[i][0], piece.T_table[1, 3] - self.poses[i][1]))
            x, y, th, typ, face = self.poses[proche]
            self.assertEqual((piece.type, piece.face), (typ, face))
            self.assertLess(np.hypot(piece.T_table[0, 3] - x, piece.T_table[1, 3] - y), 0.002)
            np.testing.assert_allclose(piece.T_camera, self.table.T_ct @ piece.T_table, atol=1e-9)
            np.testing.assert_allclose(piece.T_robot, T_rc @ piece.T_camera, atol=1e-12)
            self.assertTrue(0 < piece.confiance <= 1)
            self.assertEqual(piece.masque.shape, (H, W))
        img = p.dessiner(pieces)
        self.assertEqual(img.shape, (H, W, 3))


@besoin_cv2
class TestExport(unittest.TestCase):

    def setUp(self):
        import commun
        import exporter
        self.commun, self.exporter = commun, exporter
        self.tmp = tempfile.TemporaryDirectory()
        self.racine = self.tmp.name
        commun.charger_types(self.racine, TYPES)
        for n in ("0000", "0001", "0002"):
            img = np.full((100, 200, 3), 90, np.uint8)
            os.makedirs(os.path.join(self.racine, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.racine, "brut"), exist_ok=True)
            cv2.imwrite(os.path.join(self.racine, "images", n + ".png"), img)
            np.savez_compressed(os.path.join(self.racine, "brut", n + ".npz"),
                                profondeur=np.zeros((100, 200), np.float32), K=K)
            ann = {"image": n + ".png", "source": "couleur", "taille": [200, 100],
                   "statut": "valide" if n != "0002" else "a_revoir",
                   "instances": [
                       {"classe": "fine", "polygones": [[[20, 10], [60, 10], [60, 30], [20, 30]]]},
                       {"classe": "epaisse", "polygones": [[[100, 50], [150, 50], [150, 90], [100, 90]]]}]}
            commun.ecrire_annotation(n, ann, self.racine)

    def tearDown(self):
        self.tmp.cleanup()

    def test_polygone_normalise(self):
        ligne = self.exporter.ligne_yolo(1, [[[20, 10], [60, 10], [60, 30], [20, 30]]], 200, 100)
        v = ligne.split()
        self.assertEqual(v[0], "1")
        np.testing.assert_allclose(np.array(v[1:], float),
                                   [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3], atol=1e-6)

    def test_export(self):
        s = self.exporter.exporter(self.racine, copies=0)
        self.assertEqual(s["train"] + s["val"], 2)          # 0002 n'est pas valide
        self.assertGreaterEqual(s["val"], 1)
        with open(s["data"]) as f:
            texte = f.read()
        self.assertIn("0: epaisse", texte)
        self.assertIn("1: fine", texte)
        labels = []
        for part in ("train", "val"):
            d = os.path.join(self.racine, "yolo", "labels", part)
            for f in os.listdir(d):
                with open(os.path.join(d, f)) as fh:
                    labels += [l.split() for l in fh.read().splitlines()]
        self.assertEqual(len(labels), 4)
        for l in labels:
            self.assertIn(l[0], ("0", "1"))
            xy = np.array(l[1:], float)
            self.assertTrue(np.all((xy >= 0) & (xy <= 1)))
        s2 = self.exporter.exporter(self.racine, tout=True, copies=0)
        self.assertEqual(s2["train"] + s2["val"], 3)

    def test_decoupage_stable(self):
        a = [self.exporter.en_validation("{:04d}".format(i), 0.2) for i in range(500)]
        b = [self.exporter.en_validation("{:04d}".format(i), 0.2) for i in range(500)]
        self.assertEqual(a, b)
        self.assertTrue(0.12 < np.mean(a) < 0.28)

    def test_export_copier_coller(self):
        s = self.exporter.exporter(self.racine, copies=3, tout=True)
        self.assertGreater(s["synthetiques"], 0)
        d = os.path.join(self.racine, "yolo", "images", "val")
        self.assertFalse(any(f.startswith("cc_") for f in os.listdir(d)))

    def test_masque_polygone_aller_retour(self):
        m = np.zeros((120, 160), np.uint8)
        cv2.ellipse(m, (80, 60), (50, 18), 25, 0, 360, 1, -1)
        polys = self.commun.masque_vers_polygones(m)
        m2 = self.commun.polygones_vers_masque(polys, (160, 120))
        self.assertGreater(self.commun.iou(m > 0, m2), 0.95)


@besoin_cv2
class TestAnnoter(unittest.TestCase):
    """L'outil de revue sans fenetre, SAM remplace par un faux (un carre
    autour du point clique)."""

    def test_clics_et_touches(self):
        import commun
        import annoter

        class FauxSAM:
            def segmenter_un(self, image, boite=None, points=(), labels=(), reference=None):
                m = np.zeros(image.shape[:2], bool)
                if boite is not None:
                    x0, y0, x1, y1 = (int(v) for v in boite)
                    m[y0:y1, x0:x1] = True
                for (x, y), l in zip(points, labels):
                    m[int(y) - 8:int(y) + 8, int(x) - 8:int(x) + 8] = bool(l)
                return m, 0.9

        with tempfile.TemporaryDirectory() as racine:
            commun.charger_types(racine, TYPES)
            os.makedirs(os.path.join(racine, "images"))
            os.makedirs(os.path.join(racine, "brut"))
            cv2.imwrite(os.path.join(racine, "images", "0000.png"), np.zeros((100, 200, 3), np.uint8))
            np.savez_compressed(os.path.join(racine, "brut", "0000.npz"), K=K)
            a = annoter.Annotateur(["0000"], racine, FauxSAM(), echelle=1.0)
            h = a.H_HAUT
            a.souris(cv2.EVENT_LBUTTONDOWN, 50, 50 + h, 0)
            a.souris(cv2.EVENT_LBUTTONUP, 50, 50 + h, 0)
            self.assertEqual(len(a.masques), 1)
            a.souris(cv2.EVENT_LBUTTONDOWN, 100, 20 + h, 0)            # boite
            a.souris(cv2.EVENT_LBUTTONUP, 160, 80 + h, 0)
            self.assertEqual(len(a.masques), 2)
            self.assertTrue(a.masques[1][50, 130])
            a.touche(ord("2"))
            self.assertEqual(a.ann["instances"][1]["classe"], "fine")
            a.souris(cv2.EVENT_RBUTTONDOWN, 50, 50 + h, 0)               # suppression
            self.assertEqual(len(a.masques), 1)
            a.touche(ord("z"))                                          # annuler
            self.assertEqual(len(a.masques), 2)
            self.assertEqual(a.rendu().shape[1], 200)
            a.touche(ord(" "))                                          # valider
            ann = commun.lire_annotation("0000", racine)
            self.assertEqual(ann["statut"], "valide")
            self.assertEqual(len(ann["instances"]), 2)


@besoin_cv2
class TestCopierColler(unittest.TestCase):

    def test_coller(self):
        import copier_coller as cc
        rng = np.random.default_rng(1)
        img = np.full((200, 300, 3), 60, np.uint8)
        m = np.zeros((200, 300), bool)
        m[120:140, 100:180] = True
        img[m] = (200, 220, 230)
        plan = [0.0, -0.7071, -0.7071, 0.35]          # table vue en biais
        Kp = [[300, 0, 150], [0, 300, 100], [0, 0, 1]]
        morceaux = cc.extraire(img, [m], ["fine"], plan, Kp)
        self.assertEqual(len(morceaux), 1)
        self.assertTrue(np.isfinite(morceaux[0].z))
        img2, masques, classes = cc.coller(img, [m], ["fine"], morceaux, rng, n=5,
                                           plan=plan, K=Kp)
        self.assertEqual(img2.shape, img.shape)
        self.assertEqual(len(masques), len(classes))
        self.assertGreater(len(masques), 1)
        # occultation : aucun pixel n'appartient a deux pieces
        pile = np.sum([mm.astype(int) for mm in masques], axis=0)
        self.assertLessEqual(pile.max(), 1)
        # echelle perspective : plus loin (plus haut dans l'image), plus petit
        z_bas = cc.distance_table(plan, Kp, 150, 190)
        z_haut = cc.distance_table(plan, Kp, 150, 110)
        self.assertGreater(z_haut, z_bas)


class TestServeurGeometrie(unittest.TestCase):
    """serveur_camera.geometrie_couleur avec un faux pyrealsense2."""

    def test_transposee_et_cles(self):
        faux = types.ModuleType("pyrealsense2")
        faux.stream = types.SimpleNamespace(depth="depth", color="color")
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        ex = types.SimpleNamespace(rotation=list(R.T.reshape(-1)),   # colonne d'abord
                                   translation=[0.015, 0.001, -0.002])
        intr = types.SimpleNamespace(fx=610.0, fy=611.0, ppx=420.0, ppy=241.0,
                                     coeffs=[0.0] * 5, model="distortion.inverse_brown_conrady")

        class Prof:
            def as_video_stream_profile(self):
                return self

            intrinsics = intr

            def get_extrinsics_to(self, autre):
                return ex

        profil = types.SimpleNamespace(get_stream=lambda s: Prof())
        ancien = sys.modules.get("pyrealsense2")
        sys.modules["pyrealsense2"] = faux
        try:
            sys.modules.pop("serveur_camera", None)
            import serveur_camera
            g = serveur_camera.geometrie_couleur(profil)
        finally:
            sys.modules.pop("serveur_camera", None)
            if ancien is None:
                sys.modules.pop("pyrealsense2", None)
            else:
                sys.modules["pyrealsense2"] = ancien
        np.testing.assert_allclose(g["T_couleur_profondeur"][:3, :3], R)
        np.testing.assert_allclose(g["T_couleur_profondeur"][:3, 3], [0.015, 0.001, -0.002])
        self.assertEqual(g["K_couleur"][0, 0], 610.0)
        self.assertEqual(g["K_couleur"][1, 2], 241.0)
        self.assertIn("inverse", str(g["modele_couleur"]))


if __name__ == "__main__":
    unittest.main()
