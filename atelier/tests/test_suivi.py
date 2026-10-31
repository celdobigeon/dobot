# -*- coding: utf-8 -*-
"""
Suivi temporel des pieces (examples/perception/suivi.py), sur des sequences
SIMULEES : on connait la verite (poses, identites) et on fabrique des
detections comme Perception.detecter() les rendrait, avec du bruit de
position et d'angle, des detections manquees, des faux positifs isoles,
un ordre quelconque, des pieces deplacees ou retirees. On verifie :
  - l'affectation hongroise (implementation propre) contre la force brute ;
  - des identites stables (aucune permutation, meme entre deux pieces qui
    se touchent) ;
  - une pose lissee plus juste que la pose brute ;
  - qu'un faux positif isole ne cree aucune piste rendue ;
  - le maintien d'une piece cachee quelques images, puis sa disparition ;
  - les evenements deplacee / disparue / saisie_probable ;
  - la periodicite du lacet autour de +-90 deg.
"""

import itertools
import math
import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(ICI)), "cellule",
                             "examples", "perception"))

import suivi as S  # noqa: E402


@dataclass
class FaussePiece:
    """Comme PieceVue : ce que Perception.detecter() rend (sous-ensemble)."""
    numero: int
    face: str
    dimensions: Tuple[float, float, float]
    T_camera: np.ndarray
    T_table: np.ndarray
    type: str = "epaisse"
    confiance: float = 0.9
    support: Optional[int] = None
    hauteur_base: float = 0.0
    dessus: List[int] = field(default_factory=list)
    verite: int = -1                     # identite reelle (pour le test)

    @property
    def position_table(self):
        return self.T_table[:3, 3] * 1000.0

    @property
    def lacet(self):
        return math.degrees(math.atan2(self.T_table[1, 0], self.T_table[0, 0]))


#: repere table vu de la camera (camera a 0.5 m au-dessus, regardant en bas)
T_CT = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.5], [0, 0, 0, 1.0]])


def pose(x, y, z, lacet):
    c, s = math.cos(math.radians(lacet)), math.sin(math.radians(lacet))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


@dataclass
class Vraie:
    ident: int
    x: float
    y: float
    z: float = 17.0
    lacet: float = 0.0
    type: str = "epaisse"
    face: str = "a plat"
    dims: Tuple[float, float, float] = (68.0, 19.0, 17.0)


def detections(vraies, rng, bruit=2.0, bruit_angle=3.0, manque=0.0,
               faux=None, bruit_z=1.0, demi_tours=False):
    """Une image : detections bruitees des pieces vraies, dans un ordre
    quelconque, quelques manquees, plus d'eventuels faux positifs.
    demi_tours : le grand axe est rendu au hasard dans un sens ou l'autre
    (un parallelepipede tourne de 180 deg est le meme)."""
    sortie = []
    for v in vraies:
        if rng.random() < manque:
            continue
        T = pose(v.x + rng.normal(0, bruit), v.y + rng.normal(0, bruit),
                 v.z + rng.normal(0, bruit_z), v.lacet + rng.normal(0, bruit_angle)
                 + (180.0 * rng.integers(2) if demi_tours else 0.0))
        sortie.append(FaussePiece(0, v.face, v.dims, T_CT @ T, T, v.type,
                                  float(np.clip(rng.normal(0.85, 0.05), 0.3, 1.0)),
                                  verite=v.ident))
    for (x, y) in faux or []:
        T = pose(x, y, 17.0, rng.uniform(-90, 90))
        sortie.append(FaussePiece(0, "a plat", (68, 19, 17), T_CT @ T, T,
                                  "epaisse", 0.5, verite=-1))
    rng.shuffle(sortie)
    for k, p in enumerate(sortie):
        p.numero = k
    return sortie


def erreur(p, v):
    """(erreur de position xy en mm, erreur de lacet en deg) d'une piece."""
    x, y, _ = np.asarray(p.T_table)[:3, 3] * 1000.0
    lacet = math.degrees(math.atan2(p.T_table[1, 0], p.T_table[0, 0]))
    return math.hypot(x - v.x, y - v.y), abs(float(S.angle_180(lacet - v.lacet)))


class TestOutils(unittest.TestCase):

    def test_hongrois_contre_force_brute(self):
        rng = np.random.default_rng(0)
        for n, m in [(1, 1), (3, 3), (3, 5), (5, 3), (6, 6), (4, 7)]:
            for _ in range(10):
                C = rng.random((n, m))
                if n <= m:
                    paires = S._hongrois_propre(C)
                else:
                    paires = [(i, j) for j, i in S._hongrois_propre(C.T)]
                cout = sum(C[i, j] for i, j in paires)
                k = min(n, m)
                if n <= m:
                    best = min(sum(C[i, p[i]] for i in range(n))
                               for p in itertools.permutations(range(m), n))
                else:
                    best = min(sum(C[p[j], j] for j in range(m))
                               for p in itertools.permutations(range(n), m))
                self.assertEqual(len(paires), k)
                self.assertAlmostEqual(cout, best, places=9)

    def test_associer_respecte_la_portee(self):
        C = np.array([[0.1, S.INTERDIT], [S.INTERDIT, S.INTERDIT]])
        self.assertEqual(S.associer(C), [(0, 0)])

    def test_angle_180(self):
        self.assertAlmostEqual(float(S.angle_180(89 - (-89))), -2.0)
        self.assertAlmostEqual(float(S.angle_180(270)), -90.0)
        self.assertAlmostEqual(float(S.angle_180(-91)), 89.0)

    def test_vote_hysteresis(self):
        v = S.Vote(oubli=0.85, marge=0.15)
        for _ in range(5):
            v.ajouter("epaisse", 0.9)
        v.ajouter("fine", 0.9)               # une erreur isolee ne fait pas changer
        self.assertEqual(v.choix, "epaisse")
        for _ in range(8):
            v.ajouter("fine", 0.9)           # un changement durable, si
        self.assertEqual(v.choix, "fine")


class TestSuivi(unittest.TestCase):

    def suivre(self, sequence, **reglages):
        """Fait tourner le suivi ; rend (pieces par image, evenements par image)."""
        suivi = S.Suivi(**reglages)
        sorties, evts = [], []
        for k, dets in enumerate(sequence):
            sorties.append(suivi.mettre_a_jour(dets, instant=10.0 * k))
            evts.append(list(suivi.evenements))
        return suivi, sorties, evts

    @staticmethod
    def verite_de(p, vraies):
        """La piece vraie la plus proche d'une piece rendue."""
        x, y, _ = np.asarray(p.T_table)[:3, 3] * 1000.0
        return min(vraies, key=lambda v: math.hypot(v.x - x, v.y - y))

    def identites(self, sorties, vraies_par_image):
        """{identite vraie : ensemble des id_piste qu'on lui a donnes}, sur
        les pieces VUES (une piece rendue de memoire est a son ancienne
        place, qui ne dit rien de la verite actuelle)."""
        ids = {}
        for pieces, vraies in zip(sorties, vraies_par_image):
            for p in pieces:
                if not p.vue:
                    continue
                v = self.verite_de(p, vraies)
                ids.setdefault(v.ident, set()).add(p.id_piste)
        return ids

    def test_identites_stables_et_lissage(self):
        """Six pieces (dont deux qui se touchent, et une a +-90 deg),
        bruit, 10 % de detections manquees, faux positifs isoles, ordre
        quelconque : chaque piece garde son identifiant, la pose lissee est
        plus juste que la brute, aucun faux positif n'est rendu."""
        vraies = [Vraie(0, 0, 0, lacet=10), Vraie(1, 0, 20, lacet=12),   # se touchent
                  Vraie(2, 100, 50, lacet=89.5), Vraie(3, -80, 60, lacet=-45),
                  Vraie(4, 60, -70, z=8, type="fine", face="a plat", lacet=30,
                        dims=(68, 17, 8)),
                  Vraie(5, -60, -60, z=68, face="debout", dims=(19, 17, 68))]
        rng = np.random.default_rng(1)
        n = 40
        seq, faux_poses = [], []
        for k in range(n):
            faux = []
            if k in (5, 13, 22, 31):            # faux positif isole, loin de tout
                faux = [(rng.uniform(150, 200), rng.uniform(-200, 200))]
                faux_poses += faux
            seq.append(detections(vraies, rng, manque=0.1 if k > 0 else 0.0, faux=faux))
        _, sorties, evts = self.suivre(seq)

        ids = self.identites(sorties, [vraies] * n)
        for ident, s in ids.items():
            self.assertEqual(len(s), 1, "piece {} : identifiants {}".format(ident, s))
        self.assertEqual(len({min(s) for s in ids.values()}), len(vraies))
        # jamais de piste rendue loin des vraies pieces (faux positifs)
        for pieces in sorties:
            self.assertEqual(len(pieces), len(vraies))
        # pas d'evenement parasite apres l'amorcage
        for e in itertools.chain(*evts[1:]):
            self.assertNotIn(e.genre, ("apparue", "disparue", "deplacee"), str(e))

        # precision : brute (detections) contre lissee (suivi), apres 5 images
        brut_xy, lisse_xy, brut_a, lisse_a = [], [], [], []
        par_ident = {v.ident: v for v in vraies}
        for k in range(5, n):
            for d in seq[k]:
                if d.verite >= 0:
                    e = erreur(d, par_ident[d.verite])
                    brut_xy.append(e[0])
                    if d.verite != 5:
                        brut_a.append(e[1])
            for p in sorties[k]:
                v = self.verite_de(p, vraies)
                e = erreur(p, v)
                lisse_xy.append(e[0])
                if v.ident != 5:
                    lisse_a.append(e[1])
        rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
        self.rapport = (rms(brut_xy), rms(lisse_xy), rms(brut_a), rms(lisse_a))
        print("\n  suivi simule : position RMS brute {:.2f} mm -> lissee {:.2f} mm ; "
              "lacet RMS brut {:.2f} deg -> lisse {:.2f} deg".format(*self.rapport))
        self.assertLess(rms(lisse_xy), 0.6 * rms(brut_xy))
        self.assertLess(rms(lisse_a), 0.6 * rms(brut_a))

    def test_faux_positif_isole(self):
        """Un faux positif isole ne donne ni piste rendue ni evenement ; s'il
        tombe la ou une piece est cachee, il ne lui vole pas son identite."""
        vraies = [Vraie(0, 0, 0), Vraie(1, 80, 0)]
        rng = np.random.default_rng(2)
        seq = [detections(vraies, rng) for _ in range(3)]
        seq.append(detections(vraies, rng, faux=[(-120, 90)]))
        # la piece 1 cachee (bras), un faux positif a 150 mm de la
        seq.append(detections(vraies[:1], rng, faux=[(200, 100)]))
        seq += [detections(vraies, rng) for _ in range(3)]
        _, sorties, evts = self.suivre(seq)
        for pieces in sorties:
            self.assertEqual(len(pieces), 2)
        self.assertEqual({p.id_piste for p in sorties[-1]}, {p.id_piste for p in sorties[0]})
        self.assertFalse([e for e in itertools.chain(*evts[1:])])
        cachee = [p for p in sorties[4] if not p.vue]
        self.assertEqual(len(cachee), 1)
        self.assertLess(cachee[0].confiance, [p for p in sorties[3]
                                              if p.id_piste == cachee[0].id_piste][0].confiance)

    def test_piece_deplacee_puis_retiree(self):
        for confirmer in (True, False):
            with self.subTest(confirmer_sauts=confirmer):
                self._deplacee_puis_retiree(confirmer)

    def _deplacee_puis_retiree(self, confirmer):
        """Une piece poussee de 15 mm : DEPLACEE (une image plus tard, le
        temps de confirmer le saut ; tout de suite avec
        confirmer_sauts=False), meme identifiant, pose recalee sans lissage.
        Une piece emportee loin et reposee : DEPLACEE aussi, une fois sa
        nouvelle place confirmee. Une piece retiree :
        maintenue quelques images (occultation possible), puis DISPARUE et
        SAISIE PROBABLE (elle etait libre)."""
        a, b, c = Vraie(0, 0, 0), Vraie(1, 100, 0, lacet=40), Vraie(2, -100, 50)
        rng = np.random.default_rng(3)
        seq, verites = [], []
        for k in range(30):
            if k == 8:
                a = Vraie(0, 15, 0)                  # poussee de 15 mm
            if k == 15:
                b = Vraie(1, 100, 180, lacet=-30)    # emportee et reposee
            presentes = [a, b] + ([c] if k < 20 else [])
            seq.append(detections(presentes, rng, bruit=1.5))
            verites.append(presentes)
        suivi, sorties, evts = self.suivre(seq, absences_max=3, confirmer_sauts=confirmer)
        ids = self.identites(sorties, verites)
        for ident, s in ids.items():
            self.assertEqual(len(s), 1, "piece {} : {}".format(ident, s))
        id_a, id_b, id_c = (min(ids[i]) for i in (0, 1, 2))
        genres = lambda k: [(e.genre, e.id_piste) for e in evts[k]]  # noqa: E731
        k = 9 if confirmer else 8
        self.assertIn(("deplacee", id_a), genres(k))
        pa = [p for p in sorties[k] if p.id_piste == id_a][0]
        self.assertLess(erreur(pa, a)[0], 5.0)        # recalee tout de suite
        # b : provisoire a 15, confirmee a 16, identite transmise
        self.assertIn(("deplacee", id_b), genres(16))
        pb = [p for p in sorties[16] if p.id_piste == id_b][0]
        self.assertLess(erreur(pb, b)[0], 5.0)
        # c : absente a partir de 20, maintenue 3 images, supprimee a la 4e
        for k in (20, 21, 22):
            self.assertIn(id_c, [p.id_piste for p in sorties[k] if not p.vue])
        self.assertNotIn(id_c, [p.id_piste for p in sorties[23]])
        self.assertIn(("disparue", id_c), genres(23))
        self.assertIn(("saisie_probable", id_c), genres(23))

    def test_retour_apres_longue_absence(self):
        """Une piece cachee plus longtemps que absences_max : DISPARUE ;
        revenue a sa place peu apres, elle reprend son identifiant
        (evenement apparue, \"de retour\"), meme si une autre piece du
        meme type est absente au meme moment."""
        vraies = [Vraie(0, 0, 0), Vraie(1, 120, 0)]
        rng = np.random.default_rng(8)
        seq = [detections(vraies, rng) for _ in range(3)]
        seq += [detections(vraies[1:], rng) for _ in range(6)]       # 0 cachee
        seq.append(detections([], rng))                              # tout cache
        seq += [detections(vraies, rng) for _ in range(3)]
        _, sorties, evts = self.suivre(seq, absences_max=4)
        id0 = [p.id_piste for p in sorties[0] if self.verite_de(p, vraies).ident == 0][0]
        self.assertIn(("disparue", id0), [(e.genre, e.id_piste) for e in evts[7]])
        self.assertIn(id0, [p.id_piste for p in sorties[-1]])
        retour = [e for e in itertools.chain(*evts[10:]) if e.id_piste == id0]
        self.assertEqual([e.genre for e in retour], ["apparue"])
        self.assertIn("retour", retour[0].details)
        self.assertEqual({p.id_piste for p in sorties[-1]}, {p.id_piste for p in sorties[0]})

    def test_bruit_adaptatif(self):
        """Detections deux fois plus bruitees que les reglages ne le disent :
        le facteur d'echelle appris s'en approche, et, une fois appris (les
        premieres images peuvent prendre une mesure aberrante pour un
        saut), aucun saut parasite ni changement d'identite."""
        vraies = [Vraie(k, 60.0 * k, 0) for k in range(5)]
        rng = np.random.default_rng(9)
        seq = [detections(vraies, rng, bruit=4.0, bruit_angle=6.0) for _ in range(40)]
        suivi, sorties, evts = self.suivre(seq)
        self.assertGreater(suivi.echelles["xy"], 2.0)
        self.assertFalse([e for e in itertools.chain(*evts[10:])])
        self.assertEqual(len({p.id_piste for s in sorties for p in s}), 5)

    def test_saisie_annoncee(self):
        vraies = [Vraie(0, 0, 0), Vraie(1, 80, 0)]
        rng = np.random.default_rng(4)
        suivi = S.Suivi()
        for _ in range(3):
            pieces = suivi.mettre_a_jour(detections(vraies, rng))
        cible = [p for p in pieces if self.verite_de(p, vraies).ident == 0][0].id_piste
        suivi.signaler_saisie(cible)
        pieces = suivi.mettre_a_jour(detections(vraies[1:], rng))    # prise
        self.assertNotIn(cible, [p.id_piste for p in pieces])
        self.assertIn("saisie_probable", [e.genre for e in suivi.evenements])

    def test_voisines_sans_echange(self):
        """Deux pieces cote a cote (20 mm entre centres, elles se touchent),
        fort bruit et detections manquees, l'une ou l'autre seule par
        moments : jamais d'echange d'identite."""
        vraies = [Vraie(0, 0, 0, lacet=0), Vraie(1, 0, 20, lacet=0)]
        rng = np.random.default_rng(5)
        seq = [detections(vraies, rng, bruit=3.0, manque=0.25 if k else 0.0)
               for k in range(60)]
        _, sorties, _ = self.suivre(seq)
        ids = self.identites(sorties, [vraies] * 60)
        self.assertEqual([len(s) for s in ids.values()], [1, 1])
        self.assertNotEqual(ids[0], ids[1])

    def test_periodicite_autour_de_90(self):
        """Lacet vrai 89 deg, mesures bruitees de part et d'autre de +-90 :
        une seule piste, jamais de saut, lacet lisse proche de 89 (modulo 180)."""
        vraies = [Vraie(0, 0, 0, lacet=89.0)]
        rng = np.random.default_rng(6)
        seq = [detections(vraies, rng, bruit_angle=4.0, demi_tours=True) for _ in range(30)]
        lacets = [float(S.angle_180(d[0].lacet)) for d in seq]
        self.assertTrue(any(l < -80 for l in lacets) and any(l > 80 for l in lacets))
        _, sorties, evts = self.suivre(seq)
        self.assertEqual({p.id_piste for s in sorties for p in s}, {1})
        self.assertFalse([e for e in itertools.chain(*evts[1:])])
        for pieces in sorties[5:]:
            self.assertLess(erreur(pieces[0], vraies[0])[1], 3.0)

    def test_sortie_compatible(self):
        """Les pieces rendues sont des copies du meme type, T_camera reste
        coherente avec T_table, numero/support/dessus en identifiants de
        piste, et l'entree n'est pas modifiee."""
        rng = np.random.default_rng(7)
        vraies = [Vraie(0, 0, 0, z=34), Vraie(1, 0, 2, z=17)]
        suivi = S.Suivi()
        for _ in range(3):
            dets = detections(vraies, rng)
            haut = [d for d in dets if d.verite == 0][0]
            bas = [d for d in dets if d.verite == 1][0]
            haut.support, bas.dessus = bas.numero, [haut.numero]
            copie_T = haut.T_table.copy()
            pieces = suivi.mettre_a_jour(dets)
            np.testing.assert_array_equal(haut.T_table, copie_T)
        for p in pieces:
            self.assertIsInstance(p, FaussePiece)
            np.testing.assert_allclose(p.T_camera, T_CT @ p.T_table, atol=1e-9)
            self.assertEqual(p.numero, p.id_piste)
        p_haut = max(pieces, key=lambda p: p.T_table[2, 3])
        p_bas = min(pieces, key=lambda p: p.T_table[2, 3])
        self.assertEqual(p_haut.support, p_bas.id_piste)
        self.assertEqual(p_bas.dessus, [p_haut.id_piste])
        try:
            import vision
        except ImportError:
            return
        self.assertIs(vision.ordre_de_prise(pieces)[0], p_haut)


if __name__ == "__main__":
    unittest.main()
