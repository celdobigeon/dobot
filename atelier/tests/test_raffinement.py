# -*- coding: utf-8 -*-
"""
Affinage "rendre et comparer" de la scene entiere (examples/perception/
raffinement.py), sur des scenes SYNTHETIQUES (synthese.py, verite connue) :
  - geometrie : rendu d'une boite connue (lancer de rayons), profondeur
    d'interpenetration (axes separateurs), appui et stabilite ;
  - le generateur de pieces INCLINEES : posees, sans interpenetration ;
  - l'affinage retrouve une piece inclinee (inclinaison, porteuse), separe
    deux pieces en contact proposees comme une seule, retrouve par la
    geometrie une piece que le reseau n'a pas vue, retire une piece
    fantome, et un masque du reseau sur le fond vide n'impose rien.
"""

import os
import sys
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(ICI)), "cellule",
                             "examples", "perception"))

try:
    import cv2  # noqa: F401
    import perception as pc
    import raffinement as rf
    import synthese as sy
except ImportError:                                  # pragma: no cover
    cv2 = None


def besoin_cv2(test):
    return unittest.skipIf(cv2 is None, "OpenCV absent")(test)


def piece(type_, face, x, y, lacet):
    return sy.PieceVraie(type_, face, x, y, lacet, sy.forme(sy.TYPES[type_], face))


def scene(pcs, graine=0):
    """Capture simulee (camera a 0.55 m, inclinee de 35 deg) et sa verite."""
    sc, v = sy.scene(pcs, sy.Camera(distance=0.55, inclinaison=35), graine=graine)
    cap = {"profondeur": sc.profondeur.astype(np.float32), "K": sc.K}
    return cap, v


def boite_verite(q, v, obs, type_=None, face=None):
    """La piece vraie q (monde) -> Boite dans le repere table de l'affinage."""
    R = obs.T_tc[:3, :3] @ v.R_cw @ q.rotation()
    c = obs.T_tc[:3, :3] @ (v.R_cw @ (q.centre() / 1000.0 - v.C)) + obs.T_tc[:3, 3]
    p = np.r_[c, np.arctan2(R[1, 0], R[0, 0]), np.arcsin(np.clip(R[2, 0], -1, 1)),
              np.arctan2(R[2, 1], R[2, 2])]
    t, f = type_ or q.type, face or q.face
    return rf.Boite(t, f, rf.dims_m(sy.TYPES, t, f), p, "test")


def apparier(pieces, verite, tol=10.0):
    """Indice de la detection la plus proche de chaque piece vraie (centre de
    la face superieure, mm), ou None."""
    if not pieces:
        return [None] * len(verite.pieces)
    det = np.array([p.T_camera[:3, 3] for p in pieces])
    sortie = []
    for c in verite.centres_camera:
        d = np.linalg.norm(det - c, axis=1) * 1000
        sortie.append(int(np.argmin(d)) if d.min() < tol else None)
    return sortie


@besoin_cv2
class TestGeometrie(unittest.TestCase):

    def test_rendu_d_une_boite(self):
        """Rayons verticaux sur une boite a plat de 17 mm, camera a 0.5 m :
        t = 0.5 - 0.017 dans l'empreinte, inf a cote ; boite inclinee :
        hauteur du dessus en pente."""
        P = np.array([[0.01, -0.02, 0.0085, 0.3, 0.0, 0.0]])
        R = rf.rotations(P)
        demi = np.array([0.034, 0.0095, 0.0085])
        o = np.array([[0.01, -0.02, 0.5], [0.01 + 0.03 * np.cos(0.3), -0.02 + 0.03 * np.sin(0.3), 0.5],
                      [0.2, 0.2, 0.5]])
        d = np.tile([0.0, 0.0, -1.0], (3, 1))
        t = rf.lancer(o, d, P[:, :3], R, demi)[0]
        self.assertAlmostEqual(t[0], 0.5 - 0.017, places=6)
        self.assertAlmostEqual(t[1], 0.5 - 0.017, places=6)
        self.assertTrue(np.isinf(t[2]))
        # inclinee de 20 deg : le dessus monte de tan(20) par metre le long de x
        a = np.radians(20)
        P2 = np.array([[0.0, 0.0, 0.03, 0.0, a, 0.0]])
        t2 = rf.lancer(np.array([[0.01, 0.0, 0.5], [-0.01, 0.0, 0.5]]), d[:2], P2[:, :3],
                       rf.rotations(P2), demi)[0]
        self.assertAlmostEqual((t2[1] - t2[0]), 0.02 * np.tan(a), places=6)

    def test_rendu_par_la_camera(self):
        """Une boite de la scene, rendue dans la grille de profondeur, tombe
        a la profondeur de la capture synthetique (sans bruit)."""
        q = piece("epaisse", "a plat", 0, 0, 20)
        sc, v = sy.scene([q], sy.Camera(distance=0.5, inclinaison=0),
                         sy.Bruit(k_z2=0, lissage=1, courbure=0, occultation=False, trous=0))
        cap = {"profondeur": sc.profondeur.astype(np.float32), "K": sc.K}
        obs = rf.Observation(cap, sy.TYPES)
        S = rf.Scene(obs)
        S.ajouter(boite_verite(q, v, obs))
        zp = S.profondeur_rendue()
        ok = obs.connu
        self.assertGreater(ok.sum(), 500)
        np.testing.assert_allclose(zp[ok], obs.z_obs[ok], atol=0.0005)
        # le centre de l'image voit le dessus de la piece, a 0.5 - 0.017 m
        H, W = sc.profondeur.shape
        self.assertAlmostEqual(sc.profondeur[H // 2, W // 2], 0.5 - 0.017, places=4)

    def test_interpenetration(self):
        I = np.eye(3)
        h = np.array([0.01, 0.01, 0.01])
        pen = rf.penetration(np.array([[0.0, 0, 0], [0.015, 0, 0], [0.025, 0, 0]]),
                             np.stack([I] * 3), h, np.zeros(3), I, h)
        np.testing.assert_allclose(pen, [0.02, 0.005, 0.0], atol=1e-12)
        # une boite tournee de 45 deg dont le coin entre de 2 mm
        Rz = rf.rotations(np.array([[0, 0, 0, np.pi / 4, 0, 0]]))
        c = np.array([[0.01 + 0.01 * np.sqrt(2) - 0.002, 0.0, 0.0]])
        self.assertAlmostEqual(rf.penetration(c, Rz, h, np.zeros(3), I, h)[0], 0.002, places=6)
        # toutes les paires d'un coup
        P = np.array([[0.0, 0, 0, 0, 0, 0], [0.015, 0, 0, 0.3, 0.1, 0]])
        Q = np.array([[0.0, 0.03, 0, 0, 0, 0], [0.01, 0, 0.005, 1.0, 0, 0.2]])
        A = rf.penetrations(P[:, :3], rf.rotations(P), h, Q[:, :3], rf.rotations(Q), np.stack([h, h]))
        for j in range(2):
            np.testing.assert_allclose(A[:, j], rf.penetration(P[:, :3], rf.rotations(P), h, Q[j, :3],
                                                                rf.rotations(Q[j:j + 1])[0], h))

    def test_appui_et_stabilite(self):
        """Posee sur la table : aucune penalite ; en l'air : penalite ;
        inclinee le bout sur la table, sans rien sous l'autre bout : instable."""
        cap, v = scene([piece("epaisse", "a plat", 0, 0, 0)])
        obs = rf.Observation(cap, sy.TYPES)
        S = rf.Scene(obs)
        dims = rf.dims_m(sy.TYPES, "epaisse", "a plat")
        L, _, h = dims
        a = np.radians(15)
        zc = L / 2 * np.sin(a) + h / 2 * np.cos(a)
        P = np.array([[0.0, 0.0, h / 2, 0, 0, 0],           # sur la table
                      [0.0, 0.0, h / 2 + 0.01, 0, 0, 0],    # 10 mm en l'air
                      [0.0, 0.0, zc, 0, a, 0]])             # un bout sur la table
        e = S.physique(None, P, dims)
        self.assertLess(e[0], 1e-6)
        self.assertGreater(e[1], 30)
        self.assertGreater(e[2], 30)
        # la meme inclinee, l'autre bout pose sur une piece : stable
        S.ajouter(rf.Boite("epaisse", "a plat", dims, np.array([0.03, 0.0, h / 2, np.pi / 2, 0, 0])))
        a2 = np.arcsin((h - 0.0) / (L * 0.8))
        # dessous a la hauteur h, a 80 % de la longueur du bout bas
        E = np.array([0.03 - 0.0095 - 0.8 * L * np.cos(a2), 0.0])
        c = np.r_[E + np.array([np.cos(a2), 0]) * L / 2, 0.0] + \
            np.array([-np.sin(a2), 0, np.cos(a2)]) * h / 2
        c[2] += L / 2 * np.sin(a2)
        P3 = np.array([[c[0], c[1], c[2], 0, a2, 0]])
        # on cherche la hauteur de contact (ni en l'air, ni dedans)
        meilleur = min(S.physique(None, P3 + [[0, 0, dz, 0, 0, 0]], dims)[0]
                       for dz in np.linspace(-0.004, 0.004, 33))
        self.assertLess(meilleur, 5.0)


@besoin_cv2
class TestSyntheseInclinees(unittest.TestCase):

    def test_pieces_inclinees_posees(self):
        n = 0
        for graine in range(6):
            pcs = sy.placer(np.random.default_rng(graine), 8, sy.TYPES, touchantes=1,
                            empilees=1, inclinees=3)
            for i, p in enumerate(pcs):
                if not p.inclinee:
                    continue
                n += 1
                c = p.coins3d()
                self.assertAlmostEqual(c[:, 2].min(), 0.0, delta=0.05)     # un bout sur la table
                self.assertIsNotNone(p.support)
                self.assertGreater(c[:, 2].max(), 5.0)
                for k, q in enumerate(pcs):
                    if k != i:
                        self.assertLess(sy._penetre(p, q), 0.1)
        self.assertGreater(n, 10)

    def test_tirage_inchange(self):
        """Sans pieces inclinees, les scenes tirees ne changent pas."""
        a = sy.placer(np.random.default_rng(3), 8, sy.TYPES, touchantes=1, empilees=2)
        b = sy.placer(np.random.default_rng(3), 8, sy.TYPES, touchantes=1, empilees=2, inclinees=0)
        self.assertEqual([(p.x, p.y, p.lacet, p.z) for p in a], [(p.x, p.y, p.lacet, p.z) for p in b])


@besoin_cv2
class TestAffinage(unittest.TestCase):

    def test_piece_inclinee_retrouvee(self):
        """Une fine inclinee, un bout sur la table, l'autre sur une epaisse :
        proposee A PLAT (comme la detection d'avant), elle est retrouvee
        inclinee, appuyee sur l'epaisse, et prise avant elle."""
        pcs = [piece("epaisse", "a plat", 10, 10, 10)]
        sy.incliner(np.random.default_rng(4), pcs, 1, sy.TYPES, modes=("sur",))
        self.assertTrue(pcs[1].inclinee)
        cap, v = scene(pcs)
        obs = rf.Observation(cap, sy.TYPES)
        plate = boite_verite(pcs[1], v, obs)
        plate.p[4] = plate.p[5] = 0.0
        plate.p[2] = plate.dims[2] / 2 + 0.004
        aff = rf.Affinage(obs, [boite_verite(pcs[0], v, obs), plate])
        res = rf.vers_pieces(aff.executer(), obs, None)
        k = apparier(res, v)
        self.assertNotIn(None, k)
        haut, bas = res[k[1]], res[k[0]]
        self.assertAlmostEqual(haut.inclinaison, pcs[1].tangage, delta=3.0)
        self.assertEqual((haut.type, haut.face), (pcs[1].type, pcs[1].face))
        self.assertEqual(haut.support, bas.numero)
        self.assertIn(haut.numero, bas.dessus)
        ordre = pc.ordre_de_prise(res)
        self.assertLess(ordre.index(haut), ordre.index(bas))
        b = sy.comparer(res, v)
        self.assertEqual((b.trouves, b.faux), (2, 0))
        self.assertLess(max(b.err_pos), 2.0)

    def test_deux_pieces_en_contact_separees(self):
        """Deux fines a plat cote a cote, proposees comme UNE epaisse a plat
        entre les deux : l'affinage rend les deux fines."""
        pcs = [piece("fine", "a plat", 0, 0, 30)]
        a = np.radians(30)
        pcs.append(piece("fine", "a plat", -17 * np.sin(a), 17 * np.cos(a), 30))
        cap, v = scene(pcs, graine=1)
        obs = rf.Observation(cap, sy.TYPES)
        b0 = boite_verite(pcs[0], v, obs, "epaisse", "a plat")
        b1 = boite_verite(pcs[1], v, obs)
        b0.p[:2] = (b0.p[:2] + b1.p[:2]) / 2
        b0.p[2] = b0.dims[2] / 2
        aff = rf.Affinage(obs, [b0])
        res = rf.vers_pieces(aff.executer(), obs, None)
        b = sy.comparer(res, v)
        self.assertEqual((b.trouves, b.faux), (2, 0))
        self.assertTrue(all(p.type == "fine" and p.face == "a plat" for p in res))

    def _masque(self, q, v, vue, obs):
        """Masque (vue de la profondeur) de la partie visible de la piece q."""
        lab = rf.masque_visible([boite_verite(q, v, obs)], obs, vue)
        return lab == 0

    def test_piece_absente_des_masques_retrouvee(self):
        """Le reseau ne voit qu'une piece sur deux (fond inconnu), et il n'y
        a aucune autre proposition : la piece manquee est trouvee par la
        profondeur seule."""
        pcs = [piece("epaisse", "sur la tranche", -40, 10, 20), piece("fine", "a plat", 40, -10, -40)]
        cap, v = scene(pcs, graine=2)
        cap["couleur"] = np.full(cap["profondeur"].shape + (3,), 128, np.uint8)
        vue = pc.vue_couleur(cap)
        obs0 = rf.Observation(cap, sy.TYPES)
        masques = [(self._masque(pcs[0], v, vue, obs0), "epaisse", 0.9)]
        res = rf.raffiner(cap, [], sy.TYPES, masques=masques, vue=vue)
        b = sy.comparer(res, v)
        self.assertEqual((b.trouves, b.faux), (2, 0))
        self.assertEqual(b.bon_type, 2)

    def test_fantome_retire(self):
        """Une proposition sur la table vide (fantome), et un masque du reseau
        sur le fond vide (faux positif) : ni l'une ni l'autre ne reste."""
        pcs = [piece("epaisse", "a plat", 0, 0, 0)]
        cap, v = scene(pcs, graine=3)
        cap["couleur"] = np.full(cap["profondeur"].shape + (3,), 128, np.uint8)
        vue = pc.vue_couleur(cap)
        obs = rf.Observation(cap, sy.TYPES)
        vraie = boite_verite(pcs[0], v, obs)
        fantome = vraie.copie()
        fantome.p[:2] += [0.06, 0.04]
        # le masque d'une piece qui serait posee 60 mm plus loin
        faux = fantome.copie()
        m_faux = rf.masque_visible([faux], obs, vue) == 0
        aff = rf.Affinage(rf.Observation(cap, sy.TYPES, [(m_faux, "epaisse", 0.95)], vue),
                          [vraie, fantome])
        res = rf.vers_pieces(aff.executer(), aff.obs, vue)
        b = sy.comparer(res, v)
        self.assertEqual((b.trouves, b.faux), (1, 0))
        self.assertGreater(res[0].confiance, 0.5)


if __name__ == "__main__":
    unittest.main()
