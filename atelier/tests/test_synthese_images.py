#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests du generateur d'images synthetiques
(examples/perception/apprentissage/synthese_images.py), sans camera, sans
reseau et sans le jeu de donnees reel (fonds et textures procedurales) :

  - rendu d'une boite connue : son masque est la silhouette projetee ;
  - occultation : une piece devant une autre lui retire exactement sa
    silhouette ;
  - pas d'interpenetration des scenes tirees (verifie independamment du
    test des axes separateurs, par des points tires DANS les boites) ;
  - pieces inclinees : un bout sur la table, appuyees sans traverser ;
  - export YOLO-seg valide (classes, polygones normalises, data.yaml, meta).
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ImportError:                               # pragma: no cover
    cv2 = None

# le paquet et les exemples vivent dans cellule/, voisin de atelier/
RACINE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cellule")
APPRENTISSAGE = os.path.join(RACINE, "examples", "perception", "apprentissage")
if APPRENTISSAGE not in sys.path:
    sys.path.insert(0, APPRENTISSAGE)

TYPES = {"epaisse": (17.0, 19.0, 68.0), "fine": (8.0, 17.0, 68.0)}
K = np.array([[612.0, 0.0, 426.6], [0.0, 611.3, 247.1], [0.0, 0.0, 1.0]])
W, H = 848, 480


def iou(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 1.0


class TestSyntheseImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if cv2 is None:
            raise unittest.SkipTest("OpenCV absent")
        import synthese_images as S
        cls.S = S
        cls.res = S.Ressources(avec_reel=False)
        # camera a 30 cm, plongeant de 40 degres : les cotes des boites se voient
        cls.cam = S.camera_de_plan(K, W, H, S.plan_tire(None, 0.30, 40.0, 3.0))

    def rendre(self, boites):
        S = self.S
        rng = np.random.default_rng(0)
        fond = np.full((H, W, 3), 0.5, np.float32)
        prof = np.full((H, W), np.inf, np.float32)
        ecl = S.Eclairage(np.array([0.3, 0.2, 0.93]), 0.5, 0.4, 1.0, 0.2)
        img, _, ident, seules = S.rendre_scene(self.cam, fond, prof, boites, ecl,
                                               S.Textures(self.res, rng), rng)
        masques = [m for _, m in S.masques_visibles(ident, len(boites), W, H)]
        return img, masques, seules

    def silhouette(self, b):
        """Enveloppe convexe des 8 coins projetes : la silhouette exacte d'un
        parallelepipede (convexe)."""
        uv, _ = self.cam.projeter(b.coins())
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv * 16).astype(np.int32)), 1, cv2.LINE_8, 4)
        return m > 0

    def point_devant(self, distance):
        """Point de la table vu au centre de l'image, a peu pres a `distance`."""
        P, _ = self.cam.sur_table(np.array([[W / 2, H / 2]]))
        return P[0]

    def test_plan_aller_retour(self):
        S = self.S
        plan = S.plan_tire(None, 0.166, 17.2, -2.8)
        h, p, r = S.angles_de_plan(plan)
        self.assertAlmostEqual(h, 0.166, places=6)
        self.assertAlmostEqual(p, 17.2, places=4)
        self.assertAlmostEqual(r, -2.8, places=4)
        cam = S.camera_de_plan(K, W, H, plan)
        np.testing.assert_allclose(cam.plan(), plan, atol=1e-9)

    def test_boite_connue(self):
        S = self.S
        c = self.point_devant(0.4)
        b = S.boite_posee("epaisse", TYPES["epaisse"], "a plat", c[0], c[1], 0.6)
        self.assertAlmostEqual(b.bas, 0.0, places=9)
        self.assertAlmostEqual(b.haut, 0.017, places=9)
        img, masques, seules = self.rendre([b])
        attendu = self.silhouette(b)
        self.assertGreater(attendu.sum(), 800)
        self.assertGreater(iou(masques[0], attendu), 0.97)
        # la piece est bien dessinee la ou est son masque
        self.assertGreater(np.abs(img[masques[0]] - 0.5).mean(), 0.05)
        self.assertLess(np.abs(img[~cv2.dilate(attendu.astype(np.uint8),
                                               np.ones((15, 15), np.uint8)).astype(bool)] - 0.5).max(),
                        0.02 + 1e-6 + 0.5 * 0.85)   # hors piece : fond (eventuellement ombre)

    def test_debout_et_tranche(self):
        S = self.S
        for face, h in (("sur la tranche", 0.019), ("debout", 0.068)):
            b = S.boite_posee("epaisse", TYPES["epaisse"], face, 0.0, 0.4, 0.3)
            self.assertAlmostEqual(b.haut - b.bas, h, places=9)
            self.assertAlmostEqual(abs(np.linalg.det(b.R)), 1.0, places=9)

    def test_occultation(self):
        S = self.S
        c = self.point_devant(0.4)
        # direction de la camera vers le point, projetee sur la table
        o = self.cam.T_tc[:3, 3]
        d = (c - o)[:2]
        d /= np.linalg.norm(d)
        lacet = math.atan2(d[1], d[0]) + math.pi / 2       # en travers du regard
        lat = np.array([-d[1], d[0]])
        # une piece basse, et devant elle (cote camera) une piece haute,
        # decalee en longueur : elle en cache une partie
        derriere = S.boite_posee("fine", TYPES["fine"], "a plat",
                                 c[0] + 0.0095 * d[0], c[1] + 0.0095 * d[1], lacet)
        p = c[:2] - 0.0095 * d + 0.02 * lat
        devant = S.boite_posee("epaisse", TYPES["epaisse"], "sur la tranche", p[0], p[1], lacet)
        self.assertFalse(S.chevauchent(devant, derriere))
        _, masques, seules = self.rendre([derriere, devant])
        s_der, s_dev = self.silhouette(derriere), self.silhouette(devant)
        self.assertGreater((s_der & s_dev).sum(), 200, "la scene doit avoir une occultation")
        self.assertGreater(iou(masques[1], s_dev), 0.97)
        # partie visible = silhouette moins celle de devant, au pixel de bord pres
        attendu = (s_der & ~s_dev).astype(np.uint8)
        noyau = np.ones((3, 3), np.uint8)
        self.assertFalse((masques[0] & ~(cv2.dilate(attendu, noyau) > 0)).any())
        self.assertFalse(((cv2.erode(attendu, noyau) > 0) & ~masques[0]).any())
        self.assertGreater(masques[0].sum(), 300)
        self.assertEqual(np.count_nonzero(masques[0] & masques[1]), 0)
        # la part visible est bien plus petite que la silhouette seule
        self.assertLess(masques[0].sum(), 0.9 * seules[0] / self.S.SUR ** 2)

    def test_axes_separateurs(self):
        S = self.S
        a = S.boite_posee("epaisse", TYPES["epaisse"], "a plat", 0, 0, 0)
        # cote a cote, en contact : pas de collision
        b = S.boite_posee("epaisse", TYPES["epaisse"], "a plat", 0, 0.019, 0)
        self.assertFalse(S.chevauchent(a, b))
        b = S.boite_posee("epaisse", TYPES["epaisse"], "a plat", 0, 0.017, 0)
        self.assertTrue(S.chevauchent(a, b))
        # en croix, a la meme hauteur : collision ; posee dessus : non
        b = S.boite_posee("fine", TYPES["fine"], "a plat", 0, 0, math.pi / 2)
        self.assertTrue(S.chevauchent(a, b))
        b = S.boite_posee("fine", TYPES["fine"], "a plat", 0, 0, math.pi / 2, base=a.haut)
        self.assertFalse(S.chevauchent(a, b))
        # deux boites dont seule une arete croisee les separe
        c = S.boite_posee("epaisse", TYPES["epaisse"], "a plat", 0.045, 0.045, math.pi / 4)
        self.assertFalse(S.chevauchent(a, c))

    def test_scenes_sans_interpenetration(self):
        """Verification INDEPENDANTE du test des axes separateurs : des points
        tires dans chaque boite (amincie de 1 mm) ne tombent dans aucune autre
        (amincie de 1 mm) ni sous la table."""
        S = self.S
        rng = np.random.default_rng(3)
        centre = self.point_devant(0.4)
        tous_ok = lambda P: np.ones(len(P), bool)  # noqa: E731
        reg = S.ReglagesScene(n_max=10, p_vide=0.0, p_intrus=0.5)
        n_inclinees = n_empilees = 0
        for essai in range(25):
            boites = S.tirer_scene(rng, TYPES, self.cam, tous_ok, centre, reg)
            self.assertGreater(len(boites), 0)
            for i, a in enumerate(boites):
                n_inclinees += a.pose == "inclinee"
                n_empilees += a.droite and a.bas > 1e-3
                self.assertGreater(a.bas, -1e-3)
                s = rng.uniform(-1, 1, (400, 3)) * np.maximum(a.e - 0.001, 0)
                P = s @ a.R.T + a.t
                for j, b in enumerate(boites):
                    if i == j:
                        continue
                    q = np.abs((P - b.t) @ b.R)
                    dedans = np.all(q < np.maximum(b.e - 0.001, 0), axis=1)
                    self.assertFalse(dedans.any(), "boites {} et {} s'interpenetrent".format(i, j))
        self.assertGreater(n_inclinees, 3)
        self.assertGreater(n_empilees, 3)

    def test_inclinee_appuyee(self):
        """Une piece inclinee a un bout sur la table et touche sa porteuse
        (a 1 mm pres), sans la traverser."""
        S = self.S
        rng = np.random.default_rng(1)
        porteuse = S.boite_posee("epaisse", TYPES["epaisse"], "sur la tranche", 0, 0.4, 0.2)
        n = 0
        for _ in range(200):
            b = S.poser_inclinee(rng, "fine", TYPES["fine"], "a plat", porteuse, [porteuse])
            if b is None:
                continue
            n += 1
            self.assertAlmostEqual(b.bas, 0.0, delta=1e-6)
            self.assertFalse(S.chevauchent(b, porteuse, jeu=0.0005))
            # touche : amincie de 0 la boite inclinee, grossie de 1 mm, collisionne
            grosse = S.Boite(b.classe, b.R, b.t, b.e + 0.001)
            self.assertTrue(S.chevauchent(grosse, porteuse, jeu=0.0))
            self.assertGreater(b.R[2, 0] ** 2, math.sin(math.radians(4)) ** 2)
        self.assertGreater(n, 20)

    def test_familles_de_textures(self):
        S = self.S
        rng = np.random.default_rng(0)
        for f in S.FAMILLES:
            t = S.texture_famille(rng, f, 64, 96)
            self.assertEqual(t.shape, (64, 96, 3), f)
            self.assertTrue(np.isfinite(t).all() and t.min() >= 0 and t.max() <= 1, f)

    def test_export_yolo(self):
        S = self.S
        with tempfile.TemporaryDirectory() as tmp:
            racine = os.path.join(tmp, "dataset")
            os.makedirs(racine)
            with open(os.path.join(racine, "config.json"), "w") as f:
                json.dump({"types": TYPES}, f)
            dossier = os.path.join(tmp, "synth")
            st = S.generer_jeu(4, dossier, S.ReglagesGeneration(), graine=5, processus=1,
                               racine=racine, cache=os.path.join(tmp, "cache"), verbeux=False)
            self.assertEqual(st["images"], 4)
            images = sorted(os.listdir(os.path.join(dossier, "images")))
            self.assertEqual(len(images), 4)
            n_lignes = 0
            for im in images:
                base = im[:-4]
                img = cv2.imread(os.path.join(dossier, "images", im))
                self.assertEqual(img.shape, (H, W, 3))
                with open(os.path.join(dossier, "labels", base + ".txt")) as f:
                    for ligne in f.read().split("\n"):
                        if not ligne.strip():
                            continue
                        v = ligne.split()
                        self.assertIn(int(v[0]), (0, 1))
                        xy = np.array(v[1:], float)
                        self.assertEqual(len(xy) % 2, 0)
                        self.assertGreaterEqual(len(xy), 6)
                        self.assertTrue((xy >= 0).all() and (xy <= 1).all())
                        n_lignes += 1
                with open(os.path.join(dossier, "meta", base + ".json")) as f:
                    meta = json.load(f)
                self.assertIn("pieces", meta)
                for p in meta["pieces"]:
                    self.assertEqual(np.array(p["coins_dessus_image"]).shape, (4, 2))
            self.assertGreater(n_lignes, 0)
            with open(os.path.join(dossier, "data.yaml")) as f:
                y = f.read()
            self.assertIn("0: epaisse", y)
            self.assertIn("1: fine", y)
            self.assertTrue(os.path.exists(S.controle(dossier, 4)))
            # reproductible : meme graine, memes etiquettes
            d2 = os.path.join(tmp, "synth2")
            S.generer_jeu(1, d2, S.ReglagesGeneration(), graine=5, processus=1,
                          racine=racine, cache=os.path.join(tmp, "cache"), verbeux=False)
            with open(os.path.join(dossier, "labels", "s_000000.txt")) as f1, \
                    open(os.path.join(d2, "labels", "s_000000.txt")) as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_exporter_avec_rendues(self):
        """exporter.py --synthetiques : les images rendues vont a
        l'entrainement, jamais en validation ; sans l'option, rien ne change."""
        import exporter
        S = self.S
        with tempfile.TemporaryDirectory() as tmp:
            racine = os.path.join(tmp, "dataset")
            os.makedirs(os.path.join(racine, "brut"))
            with open(os.path.join(racine, "config.json"), "w") as f:
                json.dump({"types": TYPES}, f)
            import commun
            for k in range(6):
                nom = "{:04d}".format(k)
                np.savez_compressed(os.path.join(racine, "brut", nom + ".npz"), x=np.zeros(1))
                commun.dossier(racine, "images")
                cv2.imwrite(os.path.join(racine, "images", nom + ".png"), np.zeros((H, W, 3), np.uint8))
                m = np.zeros((H, W), bool)
                m[100:140, 100:300] = True
                commun.ecrire_annotation(nom, {"image": nom + ".png", "taille": [W, H],
                                               "statut": "valide",
                                               "instances": [commun.instance("fine", m)]}, racine)
            dossier = os.path.join(tmp, "synth")
            S.generer_jeu(3, dossier, S.ReglagesGeneration(), graine=1, processus=1,
                          racine=racine, cache=os.path.join(tmp, "cache"), verbeux=False)
            s0 = exporter.exporter(racine, os.path.join(tmp, "y0"), copies=0)
            s1 = exporter.exporter(racine, os.path.join(tmp, "y1"), copies=0,
                                   synthetiques=dossier, n_synth=2)
            self.assertEqual(s0["rendues"], 0)
            self.assertEqual(s1["rendues"], 2)
            self.assertEqual(s0["val"], s1["val"])
            tr = os.listdir(os.path.join(tmp, "y1", "images", "train"))
            va = os.listdir(os.path.join(tmp, "y1", "images", "val"))
            self.assertEqual(sum(f.startswith("sy_") for f in tr), 2)
            self.assertFalse(any(f.startswith("sy_") for f in va))
            self.assertEqual(len(tr), len(os.listdir(os.path.join(tmp, "y0", "images", "train"))) + 2)


if __name__ == "__main__":
    unittest.main()
