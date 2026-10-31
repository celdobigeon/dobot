#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Images d'entrainement SYNTHETIQUES par randomisation de domaine, a partir de
la geometrie connue des pieces.

    # banque de textures de bois (une fois, ~2 min) : faces des pieces reelles
    .venv/bin/python examples/perception/apprentissage/synthese_images.py --textures
    # 2000 images d'entrainement (tous les fonds), 8 processus
    .venv/bin/python examples/perception/apprentissage/synthese_images.py --n 2000
    # jeu de TEST "fonds jamais vus" (familles exclues de l'entrainement)
    .venv/bin/python examples/perception/apprentissage/synthese_images.py --test-fonds --n 300
    # puis, pour entrainer avec : entrainer.py --synthetiques synthetique/entrainement

POURQUOI. Le jeu reel n'a qu'une nappe, un eclairage, une pose de camera :
un reseau entraine dessus risque d'apprendre le FOND (une piece = ce qui
n'est pas une fleur rose sur du blanc). La randomisation de domaine (Tobin
et al., 2017 ; Tremblay et al., 2018) renverse le probleme : on rend des
milliers de scenes dont TOUT varie (fond, lumiere, pose, bruit) SAUF ce qui
definit une piece -- sa forme de parallelepipede de dimensions connues et
son aspect de bois clair. Le reseau n'a alors plus d'autre indice que la
piece elle-meme. Et les etiquettes sont exactes par construction (pas
d'annotation a la main), y compris pour les cas difficiles : pieces
empilees, inclinees, cachees en partie.

PRINCIPE (une image) :
  1. une CAMERA : soit la vraie (un fond reel, avec son plan de table mesure
     par la profondeur), soit une pose tiree autour (plus haute, plus
     plongeante, plus rasante, tournee) ;
  2. une TABLE : le fond reel tel quel, ou sa nappe REMPLACEE par une autre
     texture (la piece reelle de la maison, la geometrie reelle, une autre
     nappe), ou une table entierement synthetique devant un decor ; les
     textures viennent de FAMILLES procedurales (uni, bruit/marbre, tissu,
     motifs, papier, bois, carrelage) et de la vraie nappe redressee ;
  3. des PIECES : 0 a 10 parallelepipedes des deux types, poses a plat, sur
     la tranche, debout, en contact, empiles, INCLINES (un bout sur la table,
     l'autre appuye sur une piece : l'angle est celui ou la face inferieure
     touche l'obstacle le plus haut), sans interpenetration (test des axes
     separateurs) ; plus des intrus (regle, crayon, gomme, feuille) ;
  4. le RENDU : lancer de rayons par face (exact, z-buffer), sur-echantillonne
     2 x 2 (bords antialiases) ; textures de bois REELLES (faces des pieces
     annotees, redressees par homographie) ou procedurales ; ombrage de
     Lambert, ombres portees exactes (rayon vers la lumiere, test des dalles)
     adoucies, assombrissement de contact ;
  5. la CAMERA encore : flou, balance des blancs, exposition, gamma,
     vignettage, bruit de capteur, compression JPEG ;
  6. les ETIQUETTES : masque VISIBLE de chaque piece (le z-buffer gere les
     occultations), classe, et dans meta/ : pose, coins de la face
     superieure (pour un futur modele de points cles), profondeur (option).

Sortie (dossier synthetique/, ignore par git), au format YOLO-seg :
    synthetique/<nom>/images/s_000123.jpg
    synthetique/<nom>/labels/s_000123.txt
    synthetique/<nom>/meta/s_000123.json      camera, pieces, coins
    synthetique/<nom>/data.yaml               (train = val = images : pour
                                               EVALUER un modele sur ce jeu)
    synthetique/<nom>/controle.jpg            mosaique image + masques
    synthetique/cache/                        textures et fonds pre-calcules

FONDS JAMAIS VUS. `--exclure bois,carrelage` retire ces familles de
l'entrainement ; `--test-fonds` rend un jeu fait UNIQUEMENT de ces familles :
la mAP sur ce jeu mesure ce qu'on veut savoir (le reseau marche-t-il sur
une table qu'il n'a jamais vue ?) sans avoir de vraies images d'autres
fonds. `--test-collage` fait de meme avec des pieces REELLES (decoupees dans
les images de validation, copier_coller.py) : le reseau entraine sur le
reel seul n'y est pas penalise par l'aspect synthetique des pieces.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun  # noqa: E402
from commun import perception  # noqa: E402

vision = perception.vision

SORTIE = os.path.join(commun.ICI, "synthetique")
CACHE = os.path.join(SORTIE, "cache")

#: familles de textures de table (et de decor)
FAMILLES = ("uni", "bruit", "tissu", "motifs", "papier", "bois", "carrelage",
            "nappe", "photo")
#: familles retenues pour le test "fonds jamais vus" (exclues de
#: l'entrainement de comparaison) : le bois de table est le cas le plus dur
#: (couleur proche des pieces), le carrelage un motif geometrique net
FAMILLES_TEST = ("bois", "carrelage")
#: rendu des pieces sur-echantillonne : bords antialiases, masques precis
SUR = 2
#: densite des textures de bois des pieces (pixels par mm)
PX_MM = 4.0


# ---------------------------------------------------------------------------
# Geometrie : boites, test d'interpenetration, camera
# ---------------------------------------------------------------------------

def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass(eq=False)
class Boite:
    """Un parallelepipede dans le repere TABLE (z vers le haut, m).

    Axes locaux : x = la plus grande dimension de la piece (le fil du bois),
    y et z les deux autres ; R a pour colonnes ces axes, t est le centre,
    e les demi-dimensions le long de x, y, z. classe None : un intrus (rendu,
    ombre, occulte, mais sans etiquette)."""
    classe: Optional[str]
    R: np.ndarray
    t: np.ndarray
    e: np.ndarray
    pose: str = "a plat"           # a plat, sur la tranche, debout, inclinee
    support: Optional[int] = None  # indice de la boite porteuse
    couleur: Optional[Tuple[float, float, float]] = None   # intrus : BGR 0-1

    def coins(self) -> np.ndarray:
        s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return (s * self.e) @ self.R.T + self.t

    @property
    def bas(self) -> float:
        return float(self.coins()[:, 2].min())

    @property
    def haut(self) -> float:
        return float(self.coins()[:, 2].max())

    @property
    def droite(self) -> bool:
        """Une face horizontale (posee, pas inclinee)."""
        return bool(np.max(np.abs(self.R[2])) > 0.999)

    def axe_vertical(self) -> int:
        return int(np.argmax(np.abs(self.R[2])))


def boite_posee(classe: Optional[str], dims_mm: Sequence[float], face: str, x: float,
                y: float, lacet: float, base: float = 0.0) -> Boite:
    """Une piece posee a plat, sur la tranche ou debout (perception.forme), sur
    un plan horizontal a la hauteur `base` (m)."""
    a, b, c = sorted(float(v) / 1000.0 for v in dims_mm)
    # axes locaux : x = grande dimension c ; la dimension verticale selon la face
    if face == "a plat":          # hauteur = a (la plus petite)
        e, R0 = np.array([c, b, a]) / 2, np.eye(3)
    elif face == "sur la tranche":  # hauteur = b
        e, R0 = np.array([c, a, b]) / 2, np.eye(3)
    else:                         # debout : x (le fil) vertical
        e = np.array([c, a, b]) / 2
        R0 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        # colonnes : x local -> z table, y local -> x table, z local -> y table
    R = rot_z(lacet) @ R0
    h = float(np.abs(R[2]) @ e)
    return Boite(classe, R, np.array([x, y, base + h]), e, face)


def demi_largeur(b: Boite, d: np.ndarray) -> float:
    """Demi-etendue de la boite le long de la direction unitaire d."""
    return float(np.abs(b.R.T @ d) @ b.e)


def chevauchent(a: Boite, b: Boite, jeu: float = 0.0006) -> bool:
    """Interpenetration de deux boites orientees (theoreme des axes
    separateurs : 15 axes). `jeu` (m) : les boites sont amincies d'autant,
    si bien que deux pieces qui se TOUCHENT (posee dessus, cote a cote)
    ne sont pas en collision."""
    ea, eb = np.maximum(a.e - jeu, 1e-5), np.maximum(b.e - jeu, 1e-5)
    d = b.t - a.t
    axes = [a.R[:, i] for i in range(3)] + [b.R[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(a.R[:, i], b.R[:, j])
            n = np.linalg.norm(c)
            if n > 1e-6:
                axes.append(c / n)
    for L in axes:
        ra = float(np.abs(a.R.T @ L) @ ea)
        rb = float(np.abs(b.R.T @ L) @ eb)
        if abs(float(d @ L)) > ra + rb:
            return False
    return True


def points_aretes(b: Boite, n: int = 48) -> np.ndarray:
    """Points le long des 12 aretes (repere table)."""
    c = b.coins()
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                s = np.linspace(0, 1, n)[:, None]
                pts.append(c[i] * (1 - s) + c[j] * s)
    return np.concatenate(pts)


@dataclass
class Camera:
    """K (pixels), taille, et T_ct : repere table -> repere camera (couleur)."""
    K: np.ndarray
    W: int
    H: int
    T_ct: np.ndarray

    @property
    def T_tc(self) -> np.ndarray:
        return np.linalg.inv(self.T_ct)

    def plan(self) -> np.ndarray:
        """Le plan de la table dans le repere camera : n.p + d = 0."""
        n = self.T_ct[:3, 2]
        return np.r_[n, -float(n @ self.T_ct[:3, 3])]

    def vers_camera(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, float).reshape(-1, 3) @ self.T_ct[:3, :3].T + self.T_ct[:3, 3]

    def projeter(self, P_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Points table -> (pixels N x 2, profondeur z N)."""
        Q = self.vers_camera(P_t)
        z = Q[:, 2]
        zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
        uv = np.stack([self.K[0, 0] * Q[:, 0] / zz + self.K[0, 2],
                       self.K[1, 1] * Q[:, 1] / zz + self.K[1, 2]], 1)
        return uv, z

    def rayons(self, sur: int = 1) -> np.ndarray:
        """Direction (z = 1, repere camera) du rayon de chaque pixel ; `sur` :
        grille sur-echantillonnee (le pixel j de la grille fine est au point
        (j + 0.5) / sur - 0.5 de l'image)."""
        v, u = np.mgrid[0:self.H * sur, 0:self.W * sur].astype(np.float32)
        u = (u + 0.5) / sur - 0.5
        v = (v + 0.5) / sur - 0.5
        return np.stack([(u - self.K[0, 2]) / self.K[0, 0],
                         (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)

    def sur_table(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixels -> points de la table (N x 3, repere table) et validite."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        r = np.stack([(uv[:, 0] - self.K[0, 2]) / self.K[0, 0],
                      (uv[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(uv))], 1)
        T = self.T_tc
        o, d = T[:3, 3], r @ T[:3, :3].T
        s = -o[2] / np.where(np.abs(d[:, 2]) < 1e-12, -1e-12, d[:, 2])
        return o + d * s[:, None], s > 0


def camera_de_plan(K: np.ndarray, W: int, H: int, plan: Sequence[float]) -> Camera:
    """Camera dont le plan de la table (repere camera) est `plan` : le repere
    table est celui de vision.repere_table (origine au pied de la camera)."""
    n, d = np.asarray(plan[:3], float), float(plan[3])
    nn = np.linalg.norm(n)
    return Camera(np.asarray(K, float), W, H, vision.repere_table(n / nn, d / nn))


def plan_tire(rng: np.random.Generator, hauteur: float, plongee_deg: float,
              roulis_deg: float) -> np.ndarray:
    """Plan de la table pour une camera a `hauteur` (m) dont l'axe optique
    plonge de `plongee` sous l'horizontale, tournee de `roulis` autour de
    l'axe optique. Camera horizontale : la normale de la table est -y."""
    p, r = math.radians(plongee_deg), math.radians(roulis_deg)
    n = np.array([0.0, -math.cos(p), -math.sin(p)])
    c, s = math.cos(r), math.sin(r)
    n = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ n
    return np.r_[n, hauteur]


def angles_de_plan(plan: Sequence[float]) -> Tuple[float, float, float]:
    """(hauteur m, plongee deg, roulis deg) d'un plan de table (inverse de
    plan_tire)."""
    n = np.asarray(plan[:3], float)
    n = n / np.linalg.norm(n)
    p = math.degrees(math.asin(-n[2]))
    r = math.degrees(math.atan2(-n[0], -n[1]))
    return float(plan[3]), p, -r


# ---------------------------------------------------------------------------
# Textures procedurales
# ---------------------------------------------------------------------------

def bruit_fractal(rng, h: int, w: int, echelle: float = 64.0, octaves: int = 4,
                  persistance: float = 0.5, etirement: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Bruit fractal (somme d'octaves de bruit interpole) dans [0, 1].
    `etirement` (ex, ey) : grain plus long en x qu'en y pour du bois."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        pas = max(echelle / (2 ** o), 1.0)
        gh = max(int(h / (pas * etirement[1])) + 2, 2)
        gw = max(int(w / (pas * etirement[0])) + 2, 2)
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistance
    out /= tot
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def couleur_hsv(h: float, s: float, v: float) -> np.ndarray:
    """HSV (h en degres 0-360, s et v 0-1) -> BGR float 0-1."""
    px = np.uint8([[[int(h / 2) % 180, int(np.clip(s, 0, 1) * 255), int(np.clip(v, 0, 1) * 255)]]])
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255.0


def couleur_bois_clair(rng) -> np.ndarray:
    """Une couleur de bois clair (pin, hetre) : le cas ou le fond ressemble
    aux pieces."""
    return couleur_hsv(rng.uniform(24, 42), rng.uniform(0.18, 0.45), rng.uniform(0.62, 0.95))


def couleur_aleatoire(rng, bois_clair: float = 0.15) -> np.ndarray:
    if rng.random() < bois_clair:
        return couleur_bois_clair(rng)
    return couleur_hsv(rng.uniform(0, 360), rng.uniform(0, 1) ** 1.3, rng.uniform(0.12, 0.98))


def texture_bois(rng, h: int, w: int, couleur: Optional[np.ndarray] = None,
                 contraste: Optional[float] = None, planches: bool = False) -> np.ndarray:
    """Bois : fil le long de x (veines etirees, cernes ondules), planches
    optionnelles (une teinte par planche, joint sombre)."""
    couleur = couleur_bois_clair(rng) if couleur is None else couleur
    contraste = rng.uniform(0.05, 0.35) if contraste is None else contraste
    veines = bruit_fractal(rng, h, w, echelle=rng.uniform(8, 40), octaves=4,
                           etirement=(rng.uniform(6, 20), 1.0))
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    deform = bruit_fractal(rng, h, w, echelle=rng.uniform(40, 160), octaves=2) * rng.uniform(5, 40)
    cernes = 0.5 + 0.5 * np.sin((v + deform) * rng.uniform(0.15, 0.6) + rng.uniform(0, 6))
    cernes = cernes ** rng.uniform(1.5, 6)
    motif = 0.6 * veines + 0.4 * cernes
    lum = 1.0 - contraste * (motif - 0.5) * 2
    tex = couleur[None, None, :] * lum[..., None]
    if planches:
        larg = int(rng.uniform(40, 220))
        decal = rng.integers(0, w, size=h // larg + 2)
        for k in range(h // larg + 1):
            y0, y1 = k * larg, min((k + 1) * larg, h)
            tex[y0:y1] *= rng.uniform(0.85, 1.15) * rng.uniform(0.95, 1.05, 3)
            tex[y0:min(y0 + 2, h)] *= 0.55
            x = int(decal[k]) % w
            tex[y0:y1, x:x + 2] *= 0.6
    return np.clip(tex, 0, 1)


def texture_famille(rng, famille: str, h: int, w: int, nappes: Sequence[np.ndarray] = (),
                    photos: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Une texture de table (BGR float 0-1) de la famille demandee."""
    if famille == "uni":
        c = couleur_aleatoire(rng, bois_clair=0.3)
        var = bruit_fractal(rng, h, w, echelle=rng.uniform(80, 400), octaves=3)
        tex = c[None, None] * (1 + rng.uniform(0, 0.12) * (var[..., None] - 0.5))
    elif famille == "bruit":       # marbre, beton, granit, toile ciree
        cs = [couleur_aleatoire(rng, 0.1) for _ in range(3)]
        b = bruit_fractal(rng, h, w, echelle=rng.uniform(10, 200), octaves=int(rng.integers(2, 6)),
                          persistance=rng.uniform(0.4, 0.7))
        if rng.random() < 0.4:    # veines de marbre
            b = np.abs(np.sin(b * rng.uniform(6, 20))) ** rng.uniform(0.3, 1.0)
        b = b[..., None]
        tex = np.where(b < 0.5, cs[0] * (1 - 2 * b) + cs[1] * 2 * b,
                       cs[1] * (2 - 2 * b) + cs[2] * (2 * b - 1))
        if rng.random() < 0.5:    # grain fin (granit)
            tex = tex * (1 + rng.uniform(0.05, 0.25) * (rng.random((h, w, 1)) - 0.5))
    elif famille == "tissu":       # rayures, vichy, tissage
        c0, c1 = couleur_aleatoire(rng, 0.1), couleur_aleatoire(rng, 0.1)
        v, u = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = rng.choice([0.0, math.pi / 2, math.pi / 4, rng.uniform(0, math.pi)])
        x = u * math.cos(ang) + v * math.sin(ang)
        y = -u * math.sin(ang) + v * math.cos(ang)
        p = rng.uniform(8, 80)
        sorte = rng.integers(3)
        if sorte == 0:           # rayures
            m = ((x / p) % 1.0 < rng.uniform(0.2, 0.7)).astype(np.float32)
        elif sorte == 1:         # vichy
            a = ((x / p) % 1.0 < 0.5).astype(np.float32)
            b = ((y / p) % 1.0 < 0.5).astype(np.float32)
            m = (a + b) / 2
        else:                    # tissage serre, chine
            m = bruit_fractal(rng, h, w, echelle=rng.uniform(2, 8), octaves=2)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.6, 2.0))[..., None]
        tex = c0 * (1 - m) + c1 * m
        trame = 0.5 + 0.5 * np.sin(u * rng.uniform(1.5, 3)) * np.sin(v * rng.uniform(1.5, 3))
        tex = tex * (1 + rng.uniform(0.0, 0.12) * (trame[..., None] - 0.5))
        plis = bruit_fractal(rng, h, w, echelle=rng.uniform(100, 400), octaves=2)
        tex = tex * (1 - rng.uniform(0, 0.25) * plis[..., None])
    elif famille == "motifs":      # nappes a fleurs, a pois, a feuillages
        fond = couleur_aleatoire(rng, 0.15)
        if rng.random() < 0.5:
            fond = 0.6 + 0.4 * fond        # fonds pales, comme la vraie nappe
        tex = np.ones((h, w, 3), np.float32) * fond
        n = int(rng.integers(8, 120))
        rmin, rmax = sorted(rng.uniform(4, 70, 2))
        palette = [couleur_aleatoire(rng, 0.05) for _ in range(int(rng.integers(2, 6)))]
        sorte = rng.integers(3)
        for _ in range(n):
            cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            r = rng.uniform(rmin, rmax + 1)
            c = tuple(float(x) for x in palette[int(rng.integers(len(palette)))])
            if sorte == 0:               # pois
                cv2.circle(tex, (cx, cy), int(r), c, -1, cv2.LINE_AA)
            elif sorte == 1:             # fleurs : petales en ellipses
                k = int(rng.integers(4, 9))
                a0 = rng.uniform(0, 360)
                for j in range(k):
                    a = a0 + 360.0 * j / k
                    px = int(cx + 0.6 * r * math.cos(math.radians(a)))
                    py = int(cy + 0.6 * r * math.sin(math.radians(a)))
                    cv2.ellipse(tex, (px, py), (max(int(r * 0.6), 1), max(int(r * 0.3), 1)),
                                a, 0, 360, c, -1, cv2.LINE_AA)
                cv2.circle(tex, (cx, cy), max(int(r * 0.25), 1),
                           tuple(float(x) for x in palette[0]), -1, cv2.LINE_AA)
            else:                        # feuillage : tiges et feuilles
                a = rng.uniform(0, 2 * math.pi)
                x1, y1 = int(cx + 3 * r * math.cos(a)), int(cy + 3 * r * math.sin(a))
                cv2.line(tex, (cx, cy), (x1, y1), c, max(int(r / 10), 1), cv2.LINE_AA)
                cv2.ellipse(tex, (x1, y1), (max(int(r * 0.7), 1), max(int(r * 0.25), 1)),
                            math.degrees(a), 0, 360, c, -1, cv2.LINE_AA)
        tex = cv2.GaussianBlur(tex, (0, 0), rng.uniform(0.5, 2.5))
    elif famille == "papier":
        base = [np.array([0.93, 0.95, 0.96]), np.array([0.80, 0.88, 0.92]),
                np.array([0.45, 0.62, 0.72]), np.array([0.85, 0.85, 0.85])][int(rng.integers(4))]
        tex = np.ones((h, w, 3), np.float32) * base * rng.uniform(0.85, 1.05)
        tex *= 1 + 0.06 * (rng.random((h, w, 1)).astype(np.float32) - 0.5)
        if rng.random() < 0.4:            # papier quadrille ou ligne
            p = int(rng.uniform(10, 40))
            c = tuple(float(x) for x in couleur_aleatoire(rng) * 0.5 + 0.4)
            for y in range(0, h, p):
                cv2.line(tex, (0, y), (w, y), c, 1)
            if rng.random() < 0.5:
                for x in range(0, w, p):
                    cv2.line(tex, (x, 0), (x, h), c, 1)
        for _ in range(int(rng.integers(0, 6))):   # feuilles posees, taches
            x0, y0 = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            pts = np.array([[x0, y0], [x0 + rng.uniform(50, 300), y0 + rng.uniform(-40, 40)],
                            [x0 + rng.uniform(50, 300), y0 + rng.uniform(80, 400)],
                            [x0 + rng.uniform(-40, 40), y0 + rng.uniform(80, 400)]], np.int32)
            cv2.fillConvexPoly(tex, pts, tuple(float(x) for x in base * rng.uniform(0.85, 1.1)),
                               cv2.LINE_AA)
    elif famille == "bois":
        c = couleur_bois_clair(rng) if rng.random() < 0.6 else \
            couleur_hsv(rng.uniform(10, 40), rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.75))
        tex = texture_bois(rng, h, w, c, planches=rng.random() < 0.7)
    elif famille == "carrelage":
        t = int(rng.uniform(30, 200))
        joint = int(rng.uniform(1, 8))
        cj = couleur_aleatoire(rng) * 0.4 + 0.5 * rng.random()
        cs = [couleur_aleatoire(rng, 0.15) for _ in range(int(rng.integers(1, 4)))]
        tex = np.ones((h, w, 3), np.float32) * cj
        damier = rng.random() < 0.4
        for i in range(h // t + 1):
            for j in range(w // t + 1):
                k = (i + j) % len(cs) if damier else int(rng.integers(len(cs)))
                c = cs[k] * rng.uniform(0.93, 1.07)
                tex[i * t + joint:(i + 1) * t, j * t + joint:(j + 1) * t] = c
        tex *= 1 + rng.uniform(0, 0.15) * (bruit_fractal(rng, h, w, echelle=rng.uniform(4, 40))[..., None] - 0.5)
    elif famille == "nappe" and len(nappes):
        src = nappes[int(rng.integers(len(nappes)))]
        tex = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            tex = tex[:, ::-1]
        if rng.random() < 0.3:            # teinte changee : une autre nappe a fleurs
            hsv = cv2.cvtColor((np.clip(tex, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + int(rng.integers(0, 180))) % 180
            tex = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    elif famille == "photo" and len(photos):
        src = photos[int(rng.integers(len(photos)))]
        tex = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            tex = tex[::-1]
    else:
        return texture_famille(rng, "uni", h, w)
    return np.clip(np.ascontiguousarray(tex, dtype=np.float32), 0, 1)


def bout_procedural(rng, h: int, w: int, couleur: np.ndarray) -> np.ndarray:
    """Bois de bout : cernes concentriques (centre hors de la face), plus
    sombre et plus sature que le fil."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(-3, 4) * w, rng.uniform(-3, 4) * h
    r = np.hypot(u - cx, v - cy)
    cernes = 0.5 + 0.5 * np.sin(r * rng.uniform(0.3, 1.0))
    b = bruit_fractal(rng, h, w, echelle=6, octaves=2)
    lum = 1 - rng.uniform(0.05, 0.2) * cernes - 0.08 * b
    return np.clip(couleur[None, None] * lum[..., None], 0, 1)


# ---------------------------------------------------------------------------
# Ressources tirees du jeu reel : textures de bois, fonds, nappe redressee
# ---------------------------------------------------------------------------

def _faces_locales(dims_m) -> List[Tuple[str, np.ndarray, Tuple[float, float]]]:
    """Faces d'une piece dans le repere de sa FACE SUPERIEURE (perception :
    x le long du grand cote, la piece de z = 0 a z = -h) : (sorte, 4 coins
    ordonnes [origine, +premier axe, +les deux, +second axe], (taille du
    premier axe, du second) en m). Le premier axe d'une face longue est le
    fil du bois."""
    L, l, h = dims_m
    x0, x1, y0, y1, z0, z1 = -L / 2, L / 2, -l / 2, l / 2, -h, 0.0
    P = lambda x, y, z: np.array([x, y, z])  # noqa: E731
    return [
        ("dessus", np.array([P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]), (L, l)),
        ("cote", np.array([P(x0, y0, z1), P(x1, y0, z1), P(x1, y0, z0), P(x0, y0, z0)]), (L, h)),
        ("cote", np.array([P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0), P(x0, y1, z0)]), (L, h)),
        ("bout", np.array([P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0), P(x0, y0, z0)]), (l, h)),
        ("bout", np.array([P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0), P(x1, y0, z0)]), (l, h)),
    ]


def construire_textures(racine: str = commun.DATASET, cache: str = CACHE,
                        max_images: Optional[int] = None, verbeux: bool = True) -> dict:
    """
    Banque de textures de bois tirees des pieces ANNOTEES : chaque piece
    validee est ajustee (perception.ajuster_masque : la piece de taille
    connue qui explique le masque), ses faces visibles sont projetees dans
    l'image, et celles qui sont entierement dans le masque (pas cachees, pas
    debordantes) sont redressees par homographie en un rectangle de
    PX_MM pixels par mm. On garde aussi la couleur moyenne de chaque sorte
    de face (dessus, cote, bout) : elle calibre l'ombrage du rendu.
    """
    types = commun.charger_types(racine)
    d_long = commun.dossier(cache, "textures/long")
    d_bout = commun.dossier(cache, "textures/bout")
    for d in (d_long, d_bout):
        for f in glob.glob(os.path.join(d, "*.png")):
            os.remove(f)
    couleurs = {"dessus": [], "cote": [], "bout": []}
    n_long = n_bout = 0
    noms = [n for n in commun.noms_captures(racine)
            if (commun.lire_annotation(n, racine) or {}).get("statut") == "valide"]
    if max_images:
        noms = noms[:max_images]
    t0 = time.time()
    for k, nom in enumerate(noms):
        ann = commun.lire_annotation(nom, racine)
        cap = perception.charger_capture(os.path.join(racine, "brut", nom + ".npz"))
        vue = perception.vue_couleur(cap)
        if vue is None or vue.source != "couleur_brute":
            continue
        try:
            table = perception.analyser_table(cap["profondeur"], np.asarray(cap["K"], float))
        except Exception:
            continue
        proj = perception.projeter_profondeur(cap["profondeur"], np.asarray(cap["K"], float), vue)
        taille = tuple(ann["taille"])
        masques = [commun.polygones_vers_masque(i["polygones"], taille) for i in ann["instances"]]
        image = vue.image
        T_vt = vue.T_cp @ table.T_ct
        for j, inst in enumerate(ann["instances"]):
            if inst.get("douteux") or inst["classe"] not in types or masques[j].sum() < 200:
                continue
            autres = np.zeros_like(masques[j])
            for i2, m2 in enumerate(masques):
                if i2 != j:
                    autres |= m2
            aj = perception.ajuster_masque(masques[j], table, vue, proj,
                                           {inst["classe"]: types[inst["classe"]]},
                                           inst["classe"], ignorer=autres)
            if aj is None or aj.iou < 0.78:
                continue
            dims = np.array(aj.dims) / 1000.0
            T = T_vt @ aj.T_table
            centre_c = T[:3, :3] @ np.array([0, 0, -dims[2] / 2]) + T[:3, 3]
            interieur = cv2.erode(masques[j].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            proche_autres = cv2.dilate(autres.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            for sorte, Q, (a, b) in _faces_locales(dims):
                Qc = Q @ T[:3, :3].T + T[:3, 3]
                c_face = Qc.mean(0)
                n = np.cross(Qc[1] - Qc[0], Qc[3] - Qc[0])
                n /= np.linalg.norm(n)
                if n @ (c_face - centre_c) < 0:
                    n = -n
                cos_vue = -float(n @ c_face) / np.linalg.norm(c_face)
                if cos_vue < 0.25:                # face de biais : trop ecrasee
                    continue
                uv = perception.projeter(Qc, vue).astype(np.float32)
                poly = np.zeros(masques[j].shape, np.uint8)
                cv2.fillConvexPoly(poly, np.round(uv).astype(np.int32), 1)
                aire = int(poly.sum())
                if aire < 120:
                    continue
                dedans = poly.astype(bool)
                if np.mean(interieur[dedans]) < 0.92 or np.mean(proche_autres[dedans]) > 0.02:
                    continue
                tw, th = max(int(a * 1000 * PX_MM), 4), max(int(b * 1000 * PX_MM), 4)
                Hm = cv2.getPerspectiveTransform(uv, np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]))
                rect = cv2.warpPerspective(image, Hm, (tw, th), flags=cv2.INTER_LINEAR)
                # bords : l'ajustement est a un ou deux pixels pres
                mx, my = max(int(0.08 * tw), 1), max(int(0.12 * th), 1)
                rect = rect[my:th - my, mx:tw - mx]
                couleurs[sorte].append(rect.reshape(-1, 3).mean(0).tolist())
                if sorte == "bout":
                    cv2.imwrite(os.path.join(d_bout, "{}_{}_{}.png".format(nom, j, n_bout)), rect)
                    n_bout += 1
                else:
                    cv2.imwrite(os.path.join(d_long, "{}_{}_{}.png".format(nom, j, n_long)), rect)
                    n_long += 1
        if verbeux and (k % 20 == 0 or k == len(noms) - 1):
            print("  {}/{} images, {} faces longues, {} bouts ({:.0f} s)".format(
                k + 1, len(noms), n_long, n_bout, time.time() - t0), flush=True)
    stats = {s: {"n": len(v), "moyenne": np.mean(v, 0).tolist() if v else None,
                 "ecart": np.std(v, 0).tolist() if v else None, "toutes": v}
             for s, v in couleurs.items()}
    with open(os.path.join(cache, "couleurs_bois.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)
    return {"long": n_long, "bout": n_bout,
            "moyennes": {s: v["moyenne"] for s, v in stats.items()}}


@dataclass
class FondReel:
    """Une image reelle de la table sans piece, avec sa geometrie."""
    nom: str
    image: np.ndarray              # BGR uint8
    plan: np.ndarray               # plan de la table, repere de l'image
    K: np.ndarray
    zone: np.ndarray               # H x W bool : pixels de la table (posable)


def analyser_fond(npz: str, image: Optional[np.ndarray] = None) -> Tuple[Optional[FondReel], str]:
    """Geometrie d'une capture : plan de la table (profondeur), ZONE de la
    table dans l'image couleur (points de profondeur au niveau du plan,
    enveloppe convexe), et rejet si des objets depassent du plan (un 'fond'
    ou trainent des pieces etiquetterait faux : la piece serait du fond)."""
    cap = perception.charger_capture(npz)
    vue = perception.vue_couleur(cap)
    if vue is None:
        return None, "pas de couleur"
    K = np.asarray(cap["K"], float)
    table = perception.analyser_table(cap["profondeur"], K)
    u, v, ok = perception.projeter_profondeur(cap["profondeur"], K, vue)
    Wc, Hc = vue.taille
    h = table.hauteur
    plat = ok & (np.abs(h) < 0.006)
    m = np.zeros((Hc, Wc), np.uint8)
    m[np.round(v[plat]).astype(int), np.round(u[plat]).astype(int)] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    nb, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if nb < 2:
        return None, "pas de table"
    m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone = np.zeros_like(m)
    cv2.drawContours(zone, [cv2.convexHull(max(cts, key=cv2.contourArea))], -1, 1, -1)
    # des objets au-dessus de la table, dans la zone ?
    haut = ok & (h > 0.006) & (h < 0.08)
    vh, uh = np.round(v[haut]).astype(int), np.round(u[haut]).astype(int)
    n_objets = int(zone[vh, uh].sum())
    if n_objets > 300:
        return None, "objets sur la table ({} points)".format(n_objets)
    zone = cv2.erode(zone, np.ones((9, 9), np.uint8)) > 0
    img = vue.image if image is None else image
    return FondReel(os.path.basename(npz)[:-4], img, table.plan_dans(vue), vue.K, zone), "ok"


def redresser_nappe(fond: FondReel, mm_px: float = 0.6) -> Optional[np.ndarray]:
    """La nappe vue de dessus : chaque pixel d'un rectangle du repere table
    va chercher sa couleur dans l'image (remap). On garde la plus grande
    bande entierement dans la zone de la table."""
    cam = camera_de_plan(fond.K, fond.image.shape[1], fond.image.shape[0], fond.plan)
    xs = np.arange(-0.5, 0.5, mm_px / 1000)
    ys = np.arange(0.15, 1.2, mm_px / 1000)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)
    uv, z = cam.projeter(P)
    u = uv[:, 0].reshape(X.shape).astype(np.float32)
    v = uv[:, 1].reshape(X.shape).astype(np.float32)
    H, W = fond.zone.shape
    dans = (z.reshape(X.shape) > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui, vi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(v).astype(int), 0, H - 1)
    dans &= fond.zone[vi, ui]
    # plus grand rectangle dans la zone (convexe : un intervalle par ligne)
    n_l = dans.shape[0]
    gauche = np.full(n_l, dans.shape[1])
    droite = np.full(n_l, -1)
    for i in range(n_l):
        c = np.nonzero(dans[i])[0]
        if len(c):
            gauche[i], droite[i] = c[0], c[-1]
    meilleur, choix = 0, None
    pas = 10
    for i0 in range(0, n_l, pas):
        g, d_ = gauche[i0], droite[i0]
        for i1 in range(i0 + pas, n_l, pas):
            g, d_ = max(g, gauche[i1 - pas:i1 + 1].max()), min(d_, droite[i1 - pas:i1 + 1].min())
            if d_ - g < 50:
                break
            if (i1 - i0) * (d_ - g) > meilleur:
                meilleur, choix = (i1 - i0) * (d_ - g), (i0, i1, g, d_)
    if choix is None or choix[1] - choix[0] < 50:
        return None
    y0, y1, x0, x1 = choix
    rect = cv2.remap(fond.image, u[y0:y1, x0:x1], v[y0:y1, x0:x1], cv2.INTER_LINEAR)
    return rect[::-1].astype(np.float32) / 255     # le haut de l'image = le loin


class Ressources:
    """Tout ce que la generation lit dans le jeu reel, charge une fois par
    processus : fonds (avec plan et zone), textures de bois, nappes
    redressees, photos de decor, couleurs du bois, intrinseques."""

    def __init__(self, racine: str = commun.DATASET, cache: str = CACHE,
                 avec_reel: bool = True, verbeux: bool = False):
        self.racine = racine
        self.fonds: List[FondReel] = []
        self.rejets: List[Tuple[str, str]] = []
        self.nappes: List[np.ndarray] = []
        self.photos: List[np.ndarray] = []
        self.textures_long: List[np.ndarray] = []
        self.textures_bout: List[np.ndarray] = []
        self.images_reelles: List[str] = []
        # couleurs mesurees sur les pieces reelles (BGR 0-255), valeurs par
        # defaut si la banque n'est pas construite
        self.couleurs = {"dessus": [[178.0, 203.0, 214.0]], "cote": [[150.0, 178.0, 195.0]],
                         "bout": [[120.0, 150.0, 175.0]]}
        self.K = np.array([[612.07, 0, 426.59], [0, 611.32, 247.11], [0, 0, 1.0]])
        self.W, self.H = 848, 480
        self.plan_reel = np.array([-0.047, -0.953, -0.299, 0.166])
        if not avec_reel:
            return
        self._charger_fonds(cache, verbeux)
        for d, liste in ((os.path.join(cache, "textures", "long"), self.textures_long),
                         (os.path.join(cache, "textures", "bout"), self.textures_bout)):
            for f in sorted(glob.glob(os.path.join(d, "*.png"))):
                t = cv2.imread(f)
                if t is not None and min(t.shape[:2]) >= 4:
                    liste.append(t.astype(np.float32))
            # faces mal ajustees (un bout d'ombre, de nappe ou d'une autre
            # piece dans le rectangle) : trop sombres ou trop contrastees
            if liste:
                lum = np.array([t.mean() for t in liste])
                cv = np.array([t.mean(2).std() / max(t.mean(), 1) for t in liste])
                med = float(np.median(lum))
                garde = [t for t, m, c in zip(liste, lum, cv) if 0.7 * med < m < 1.4 * med and c < 0.2]
                liste[:] = garde
        chemin = os.path.join(cache, "couleurs_bois.json")
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as f:
                st = json.load(f)
            for s in self.couleurs:
                if st.get(s, {}).get("toutes"):
                    self.couleurs[s] = st[s]["toutes"]
        for n in commun.noms_captures(racine):
            a = commun.lire_annotation(n, racine)
            if a and a.get("statut") == "valide" and os.path.exists(
                    os.path.join(racine, "images", n + ".png")):
                self.images_reelles.append(n)

    def _charger_fonds(self, cache: str, verbeux: bool) -> None:
        """Fonds reels (dataset/fonds/*.png + .npz), analyses une fois et mis
        en cache (le plan, la zone)."""
        d = commun.dossier(cache, "fonds")
        for png in sorted(glob.glob(os.path.join(self.racine, "fonds", "*.png"))):
            npz = png[:-4] + ".npz"
            if not os.path.exists(npz):
                continue
            nom = os.path.basename(png)[:-4]
            c = os.path.join(d, nom + ".npz")
            if os.path.exists(c):
                z = np.load(c)
                if str(z["statut"]) != "ok":
                    self.rejets.append((nom, str(z["statut"])))
                    continue
                f = FondReel(nom, cv2.imread(png), z["plan"], z["K"], z["zone"])
            else:
                f, statut = analyser_fond(npz, cv2.imread(png))
                if f is None:
                    np.savez_compressed(c, statut=statut)
                    self.rejets.append((nom, statut))
                    continue
                np.savez_compressed(c, statut="ok", plan=f.plan, K=f.K, zone=f.zone)
            self.fonds.append(f)
            nap = os.path.join(d, nom + "_nappe.png")
            if not os.path.exists(nap):
                r = redresser_nappe(f)
                if r is not None:
                    cv2.imwrite(nap, (r * 255).astype(np.uint8))
            if os.path.exists(nap):
                self.nappes.append(cv2.imread(nap).astype(np.float32) / 255)
        if self.fonds:
            self.K = self.fonds[0].K
            self.H, self.W = self.fonds[0].image.shape[:2]
            self.plan_reel = np.median([f.plan for f in self.fonds], axis=0)
            # decor : le haut des fonds (la piece de la maison, hors table)
            for f in self.fonds:
                lignes = np.nonzero(f.zone.any(1))[0]
                y = int(lignes[0]) if len(lignes) else f.image.shape[0] // 2
                if y > 60:
                    self.photos.append(f.image[:y].astype(np.float32) / 255)
        if verbeux:
            print("fonds reels : {} retenus, rejetes : {}".format(len(self.fonds), self.rejets))

    def image_effacee(self, rng) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Une image reelle dont les pieces annotees sont effacees par
        inpainting, et son plan. Les masques sont dilates (bords, ombre
        proche) ; l'inpainting laisse une zone floue, plausible de loin."""
        if not self.images_reelles:
            return None
        nom = self.images_reelles[int(rng.integers(len(self.images_reelles)))]
        ann = commun.lire_annotation(nom, self.racine)
        if not ann.get("plan") or any(i.get("douteux") for i in ann["instances"]):
            return None
        img = cv2.imread(os.path.join(self.racine, "images", nom + ".png"))
        m = np.zeros(img.shape[:2], np.uint8)
        for inst in ann["instances"]:
            m |= commun.polygones_vers_masque(inst["polygones"], tuple(ann["taille"])).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        return cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA), np.asarray(ann["plan"], float)


# ---------------------------------------------------------------------------
# Scenes : pieces posees, en contact, empilees, inclinees ; intrus
# ---------------------------------------------------------------------------

#: intrus (pas des pieces) : (nom, dimensions mm)
INTRUS = (("regle", (300.0, 30.0, 1.5)), ("crayon", (150.0, 7.0, 7.0)),
          ("gomme", (40.0, 20.0, 10.0)), ("feuille", (210.0, 148.0, 0.3)),
          ("cube", (25.0, 25.0, 25.0)), ("trousse", (190.0, 70.0, 40.0)))


@dataclass
class ReglagesScene:
    n_max: int = 10
    p_vide: float = 0.04
    p_intrus: float = 0.45
    #: modes de placement (isolee, contact, empilee, inclinee)
    modes: Tuple[float, float, float, float] = (0.32, 0.30, 0.16, 0.22)
    #: faces d'appui sur la table (a plat, sur la tranche, debout)
    faces: Tuple[float, float, float] = (0.50, 0.42, 0.08)


def poser_inclinee(rng, classe, dims_mm, face: str, porteuse: Boite,
                   boites: Sequence[Boite]) -> Optional[Boite]:
    """
    Une piece couchee (a plat ou sur la tranche) dont une extremite repose
    sur la table et dont la face inferieure s'appuie sur la porteuse.

    On choisit la direction u (lacet) et le point de la table A ou repose
    l'arete basse ; la piece pivote autour de cette arete jusqu'a toucher
    l'obstacle le plus haut dans sa bande (les points des aretes de TOUTES
    les boites, dans la largeur de la piece) : l'angle est le maximum de
    atan(z / distance). C'est l'equilibre reel d'une planchette appuyee.
    """
    b0 = boite_posee(classe, dims_mm, face, 0, 0, 0)
    L, l2, h = 2 * b0.e[0], b0.e[1], 2 * b0.e[2]
    psi = rng.uniform(0, 2 * math.pi)
    u = np.array([math.cos(psi), math.sin(psi), 0.0])
    w = np.array([-u[1], u[0], 0.0])
    # point vise sur le dessus de la porteuse, puis le pied A
    off = (rng.uniform(-0.7, 0.7, 2) * porteuse.e[:2])
    c0 = porteuse.t + porteuse.R[:, 0] * off[0] + porteuse.R[:, 1] * off[1]
    c0[2] = porteuse.haut
    f = rng.uniform(0.45, 0.85)
    a0 = math.asin(min(c0[2] / (f * L), 0.9))
    A = c0 - u * f * L * math.cos(a0)
    A[2] = 0.0
    obst = np.concatenate([points_aretes(b) for b in boites])
    rel = obst - A
    le_long = rel @ u
    lat = rel @ w
    sel = (le_long > 1e-3) & (le_long < L) & (np.abs(lat) <= l2) & (obst[:, 2] > 5e-4)
    if not sel.any():
        return None
    angles = np.arctan2(obst[sel, 2], le_long[sel])
    k = int(np.argmax(angles))
    alpha = float(angles[k])
    s_contact = le_long[sel][k] / math.cos(alpha)
    if not (math.radians(4) < alpha < math.radians(55)) or not (0.25 * L < s_contact < 0.95 * L):
        return None
    x = math.cos(alpha) * u + math.sin(alpha) * np.array([0, 0, 1.0])
    z = -math.sin(alpha) * u + math.cos(alpha) * np.array([0, 0, 1.0])
    R = np.stack([x, w if face != "debout" else w, z], 1)
    # y = z x x pour un repere direct
    R[:, 1] = np.cross(R[:, 2], R[:, 0])
    t = A + x * L / 2 + z * h / 2
    return Boite(classe, R, t, b0.e.copy(), "inclinee")


def tirer_scene(rng, types: Dict[str, Sequence[float]], cam: Camera,
                zone_ok: Callable[[np.ndarray], np.ndarray], centre: np.ndarray,
                reglages: ReglagesScene = ReglagesScene()) -> List[Boite]:
    """Pieces et intrus d'une scene. `zone_ok(P)` : points (N x 3, repere
    table) ou la table peut porter quelque chose ; `centre` : centre du
    tas (les pieces reelles sont souvent groupees)."""
    noms = list(types)
    if rng.random() < reglages.p_vide:
        n = 0
    else:
        n = int(np.clip(round(rng.gamma(3.0, 1.6)), 1, reglages.n_max))
    etal = rng.uniform(0.03, 0.14)
    boites: List[Boite] = []
    faces = ("a plat", "sur la tranche", "debout")

    def admissible(b: Boite) -> bool:
        c = b.coins()
        if c[:, 2].min() < -5e-4:
            return False
        uv, z = cam.projeter(c)
        if z.min() < 0.06:
            return False
        uvc, _ = cam.projeter(b.t[None])
        if not (-0.05 * cam.W < uvc[0, 0] < 1.05 * cam.W and -0.05 * cam.H < uvc[0, 1] < 1.05 * cam.H):
            return False
        bas = c[c[:, 2] < c[:, 2].min() + 2e-3]
        if not zone_ok(np.vstack([bas, b.t[None] * [1, 1, 0]])).all():
            return False
        return not any(chevauchent(b, o) for o in boites)

    def isolee(classe, dims, face, intrus=False):
        if rng.random() < 0.2 or intrus:
            p = centre[:2] + rng.normal(0, 2.5 * etal, 2)
        else:
            p = centre[:2] + rng.normal(0, etal, 2)
        return boite_posee(classe, dims, face, p[0], p[1], rng.uniform(0, 2 * math.pi))

    for k in range(n):
        for _essai in range(40):
            classe = noms[int(rng.integers(len(noms)))]
            dims = types[classe]
            mode = int(rng.choice(4, p=reglages.modes)) if boites else 0
            face = faces[int(rng.choice(3, p=reglages.faces))]
            b = None
            if mode == 0:
                b = isolee(classe, dims, face)
            elif mode == 1:                      # contact, sur la table
                sur_table = [o for o in boites if o.droite and o.bas < 1e-3
                             and o.axe_vertical() == 2]
                if not sur_table:
                    continue
                P = sur_table[int(rng.integers(len(sur_table)))]
                lacet_P = math.atan2(P.R[1, 0], P.R[0, 0])
                lacet = lacet_P + rng.choice([0.0, 0.0, math.pi / 2]) + rng.normal(0, math.radians(6))
                b = boite_posee(classe, dims, face, 0, 0, lacet)
                d = rot_z(lacet_P)[:, int(rng.integers(2))] * rng.choice([-1, 1])
                g = rng.uniform(0, 0.002)
                dist = demi_largeur(P, d) + demi_largeur(b, d) + g
                lat = np.cross([0, 0, 1.0], d)
                p = P.t[:2] + d[:2] * dist + lat[:2] * rng.normal(0, 0.4 * demi_largeur(P, lat))
                b.t[:2] = p
            elif mode == 2:                      # empilee
                droites = [o for o in boites if o.droite and o.classe is not None
                           and o.axe_vertical() == 2]
                if not droites or face == "debout":
                    continue
                P = droites[int(rng.integers(len(droites)))]
                if P.haut > 0.07:
                    continue
                lacet_P = math.atan2(P.R[1, 0], P.R[0, 0])
                lacet = lacet_P + (rng.choice([0.0, math.pi / 2]) + rng.normal(0, math.radians(8))
                                   if rng.random() < 0.7 else rng.uniform(0, math.pi))
                off = rng.uniform(-0.6, 0.6, 2) * np.array([demi_largeur(P, rot_z(lacet_P)[:, 0]),
                                                            demi_largeur(P, rot_z(lacet_P)[:, 1])])
                p = P.t[:2] + rot_z(lacet_P)[:2, :2] @ off
                b = boite_posee(classe, dims, face, p[0], p[1], lacet, base=P.haut)
                b.support = boites.index(P)
            else:                                # inclinee
                porteuses = [o for o in boites if o.droite]
                if not porteuses or face == "debout":
                    continue
                P = porteuses[int(rng.integers(len(porteuses)))]
                b = poser_inclinee(rng, classe, dims, face, P, boites)
                if b is not None:
                    b.support = boites.index(P)
            if b is not None and admissible(b):
                boites.append(b)
                break
    # intrus : poses a plat sur la table, pas des pieces
    if rng.random() < reglages.p_intrus:
        for _ in range(int(rng.integers(1, 4))):
            nom, dims = INTRUS[int(rng.integers(len(INTRUS)))]
            for _essai in range(20):
                b = isolee(None, dims, "a plat", intrus=True)
                if nom in ("cube", "gomme", "trousse") and rng.random() < 0.5:
                    b = isolee(None, dims, "sur la tranche", intrus=True)
                c = couleur_aleatoire(rng, bois_clair=0.0)
                if nom == "regle":
                    c = np.array([0.75, 0.75, 0.78]) * rng.uniform(0.6, 1.1)
                elif nom == "feuille":
                    c = np.array([0.95, 0.96, 0.97]) * rng.uniform(0.8, 1.0)
                b.couleur = tuple(float(x) for x in np.clip(c, 0, 1))
                if admissible(b):
                    boites.append(b)
                    break
    return boites


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------

def ombre_rayons(P: np.ndarray, lum: np.ndarray, boites: Sequence[Boite],
                 propre: Optional[np.ndarray] = None) -> np.ndarray:
    """1 si le rayon P + s * lum (s > 0) touche une boite (methode des
    dalles), sauf la boite `propre[i]` du point i (pas d'ombre sur soi)."""
    out = np.zeros(len(P), bool)
    for k, b in enumerate(boites):
        o = (P - b.t) @ b.R
        d = b.R.T @ lum
        with np.errstate(divide="ignore", invalid="ignore"):
            a1 = (-b.e - o) / d
            a2 = (b.e - o) / d
        tmin = np.nanmax(np.minimum(a1, a2), 1)
        tmax = np.nanmin(np.maximum(a1, a2), 1)
        hit = (tmax >= np.maximum(tmin, 1e-4))
        if propre is not None:
            hit &= propre != k
        out |= hit
    return out


@dataclass
class Eclairage:
    lum: np.ndarray            # direction vers la lumiere (repere table)
    ambiant: float             # part de lumiere ambiante (0 a 1)
    force_ombre: float         # assombrissement d'une ombre sur la table
    flou_ombre: float          # penombre (pixels)
    contact: float             # assombrissement de contact (occlusion ambiante)


def tirer_eclairage(rng) -> Eclairage:
    elev = math.radians(rng.uniform(25, 88))
    az = rng.uniform(0, 2 * math.pi)
    lum = np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
    return Eclairage(lum, rng.uniform(0.35, 0.75), rng.uniform(0.15, 0.6),
                     rng.uniform(0.6, 4.0), rng.uniform(0.1, 0.45))


def _remap_mip(tex: List[np.ndarray], mx: np.ndarray, my: np.ndarray, niveau: np.ndarray) -> np.ndarray:
    """Echantillonnage d'une texture avec une pyramide (pas de moire sur la
    table vue en biais) : interpolation entre les deux niveaux voisins."""
    out = np.zeros(mx.shape + (3,), np.float32)
    n = len(tex)
    niveau = np.clip(niveau, 0, n - 1)
    for k in range(n):
        poids = np.clip(1 - np.abs(niveau - k), 0, 1)
        if not poids.any():
            continue
        s = 0.5 ** k
        r = cv2.remap(tex[k], (mx + 0.5) * s - 0.5, (my + 0.5) * s - 0.5, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)
        out += r * poids[..., None]
    return out


def rendre_table(cam: Camera, tex: np.ndarray, mm_px: float, angle: float,
                 decalage: Tuple[float, float], rect: Tuple[float, float, float, float],
                 rayons: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image de la table texturee (plan z = 0, rectangle `rect` = x0, x1, y0,
    y1 du repere table), profondeur (inf hors table), et masque table."""
    r = cam.rayons() if rayons is None else rayons
    T = cam.T_tc
    o = T[:3, 3]
    d = r @ T[:3, :3].T.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -o[2] / d[..., 2]
    X = o[0] + d[..., 0] * s
    Y = o[1] + d[..., 1] * s
    dedans = (s > 0) & (X >= rect[0]) & (X <= rect[1]) & (Y >= rect[2]) & (Y <= rect[3])
    s = np.where(dedans, s, np.inf).astype(np.float32)
    c, si = math.cos(angle), math.sin(angle)
    xr = (c * X - si * Y) * 1000.0 / mm_px + decalage[0]
    yr = (si * X + c * Y) * 1000.0 / mm_px + decalage[1]
    xr = np.where(dedans, xr, 0).astype(np.float32)
    yr = np.where(dedans, yr, 0).astype(np.float32)
    # empreinte d'un pixel sur la table (pixels de texture) -> niveau de pyramide
    nr = np.linalg.norm(r, axis=-1)
    cos_i = np.abs(d[..., 2]) / nr
    with np.errstate(divide="ignore", invalid="ignore"):
        empreinte = s * nr / (cam.K[0, 0] * np.maximum(cos_i, 0.05)) * 1000.0 / mm_px
    niveau = np.log2(np.maximum(np.where(dedans, empreinte, 1.0), 1.0)) * 0.8
    pyr = [tex]
    while len(pyr) < 6 and min(pyr[-1].shape[:2]) > 16:
        pyr.append(cv2.pyrDown(pyr[-1]))
    img = _remap_mip(pyr, xr, yr, niveau.astype(np.float32))
    return img, s, dedans


def echantillonner(tex: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Couleurs (N x 3) de la texture aux points (x, y) (interpolation
    bilineaire) ; cv2.remap veut des cartes 2D de moins de 32767 lignes."""
    n = len(x)
    larg = 1024
    lignes = (n + larg - 1) // larg
    mx = np.zeros(lignes * larg, np.float32)
    my = np.zeros(lignes * larg, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(tex, mx.reshape(lignes, larg), my.reshape(lignes, larg), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    return out.reshape(-1, 3)[:n]


@dataclass
class Rendu:
    image: np.ndarray                    # BGR uint8
    masques: List[np.ndarray]            # masques visibles des pieces etiquetees
    classes: List[str]
    boites: List[Boite]                  # les pieces etiquetees (meme ordre)
    profondeur: np.ndarray               # m (z camera), inf hors scene
    camera: Camera
    infos: dict = field(default_factory=dict)


class Textures:
    """Choix des textures de bois d'une piece : reelles (banque) ou
    procedurales, couleur de base tiree parmi les couleurs reelles."""

    def __init__(self, res: Ressources, rng, p_reelle: float = 0.7):
        self.res, self.rng, self.p_reelle = res, rng, p_reelle

    def albedo(self) -> np.ndarray:
        cs = self.res.couleurs["dessus"]
        c = np.asarray(cs[int(self.rng.integers(len(cs)))], np.float32) / 255
        c = c * self.rng.uniform(0.9, 1.12) * self.rng.uniform(0.96, 1.04, 3)
        return np.clip(c, 0.05, 1.0)

    def face(self, sorte: str, w: int, h: int, albedo: np.ndarray) -> np.ndarray:
        rng = self.rng
        banque = self.res.textures_bout if sorte == "bout" else self.res.textures_long
        if banque and rng.random() < self.p_reelle:
            t = banque[int(rng.integers(len(banque)))]
            if rng.random() < 0.5:
                t = t[:, ::-1]
            if rng.random() < 0.5:
                t = t[::-1]
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
            rel = t / np.maximum(t.reshape(-1, 3).mean(0), 1.0)
            # le relief du grain, atenue ou accentue
            rel = np.clip(1 + (rel - 1) * rng.uniform(0.6, 1.4), 0.6, 1.4)
            tex = rel * albedo
        elif sorte == "bout":
            tex = bout_procedural(rng, h, w, albedo)
        else:
            tex = texture_bois(rng, h, w, albedo, contraste=rng.uniform(0.03, 0.18))
        if sorte == "bout":
            cs = self.res.couleurs["bout"]
            cd = self.res.couleurs["dessus"]
            ratio = (np.mean(cs, 0) + 1) / (np.mean(cd, 0) + 1)
            tex = tex * np.clip(ratio * rng.uniform(0.85, 1.1), 0.4, 1.1)
        return np.clip(tex, 0, 1.2).astype(np.float32)


def rendre_scene(cam: Camera, fond: np.ndarray, prof_fond: np.ndarray, boites: Sequence[Boite],
                 ecl: Eclairage, textures: Textures, rng,
                 surface_table: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Les boites sur le fond (float BGR 0-1, profondeur `prof_fond`). Rend
    (image float, profondeur, identifiants sur-echantillonnes, aires
    seules) : id = indice de la boite visible (-1 : fond), et pour chaque
    boite l'aire (pixels fins) de sa silhouette SANS occultation, pour
    savoir quelle part en reste visible.

    Chaque face visible (normale vers la camera) est rendue par lancer de
    rayons : intersection du rayon de chaque pixel fin de sa boite
    englobante avec le plan de la face, test d'appartenance au rectangle,
    z-buffer. Les coordonnees dans la face donnent la texture.
    """
    H, W = cam.H, cam.W
    Hs, Ws = H * SUR, W * SUR
    rays = cam.rayons(SUR)
    z = np.full((Hs, Ws), np.inf, np.float32)
    ident = np.full((Hs, Ws), -1, np.int32)
    code = np.full((Hs, Ws), -1, np.int32)
    s1 = np.zeros((Hs, Ws), np.float32)
    s2 = np.zeros((Hs, Ws), np.float32)
    seules = []
    faces = []      # (boite, axe, signe, n_t, tangentes)
    Rct, tct = cam.T_ct[:3, :3], cam.T_ct[:3, 3]
    for k, b in enumerate(boites):
        seule = np.zeros((Hs, Ws), bool)
        for a in range(3):
            for sg in (-1.0, 1.0):
                n_t = sg * b.R[:, a]
                c_t = b.t + n_t * b.e[a]
                n_c = Rct @ n_t
                c_c = Rct @ c_t + tct
                if n_c @ c_c >= 0:          # face de dos
                    continue
                ia, ib = [(1, 2), (0, 2), (0, 1)][a]
                ta, tb = Rct @ b.R[:, ia], Rct @ b.R[:, ib]
                ea, eb = b.e[ia], b.e[ib]
                cor = np.array([c_c + sa * ea * ta + sb * eb * tb for sa in (-1, 1) for sb in (-1, 1)])
                if cor[:, 2].min() < 0.01:
                    continue
                u = cam.K[0, 0] * cor[:, 0] / cor[:, 2] + cam.K[0, 2]
                v = cam.K[1, 1] * cor[:, 1] / cor[:, 2] + cam.K[1, 2]
                x0 = max(int(math.floor((u.min() + 0.5) * SUR)) - 1, 0)
                x1 = min(int(math.ceil((u.max() + 0.5) * SUR)) + 1, Ws)
                y0 = max(int(math.floor((v.min() + 0.5) * SUR)) - 1, 0)
                y1 = min(int(math.ceil((v.max() + 0.5) * SUR)) + 1, Hs)
                if x1 <= x0 or y1 <= y0:
                    continue
                Rg = rays[y0:y1, x0:x1]
                den = Rg @ n_c.astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = float(n_c @ c_c) / den
                P = Rg * t[..., None]
                dP = P - c_c.astype(np.float32)
                q1 = (dP @ ta.astype(np.float32)) / ea
                q2 = (dP @ tb.astype(np.float32)) / eb
                dedans = (t > 0) & (np.abs(q1) <= 1) & (np.abs(q2) <= 1)
                seule[y0:y1, x0:x1] |= dedans
                maj = dedans & (t < z[y0:y1, x0:x1])
                if not maj.any():
                    continue
                f_id = len(faces)
                faces.append((k, a, sg, n_t, ia, ib))
                z[y0:y1, x0:x1][maj] = t[maj]
                ident[y0:y1, x0:x1][maj] = k
                code[y0:y1, x0:x1][maj] = f_id
                s1[y0:y1, x0:x1][maj] = q1[maj]
                s2[y0:y1, x0:x1][maj] = q2[maj]
        seules.append(int(seule.sum()))

    # couleur des faces : texture * ombrage de Lambert * ombres portees
    couleur = np.zeros((Hs, Ws, 3), np.float32)
    piece = ident >= 0
    T_tc = cam.T_tc
    alb = {}
    norm_top = ecl.ambiant + (1 - ecl.ambiant) * max(ecl.lum[2], 0.05)
    if piece.any():
        ys, xs = np.nonzero(piece)
        Pc = rays[ys, xs] * z[ys, xs, None]
        Pt = Pc @ T_tc[:3, :3].T + T_tc[:3, 3]
        ombre = ombre_rayons(Pt.astype(np.float64), ecl.lum, boites, ident[ys, xs])
        o_img = np.zeros((Hs, Ws), np.float32)
        o_img[ys, xs] = ombre
        o_img = cv2.GaussianBlur(o_img, (0, 0), ecl.flou_ombre * SUR * 0.5) * piece
        for f_id, (k, a, sg, n_t, ia, ib) in enumerate(faces):
            sel = code == f_id
            if not sel.any():
                continue
            b = boites[k]
            if b.couleur is not None:
                base = np.ones((8, 8, 3), np.float32) * np.array(b.couleur, np.float32)
                ea, eb = b.e[ia], b.e[ib]
            else:
                if k not in alb:
                    alb[k] = (textures.albedo(), {})
                albedo, cache = alb[k]
                sorte = "bout" if a == 0 else "long"
                ea, eb = b.e[ia], b.e[ib]
                if (a, sg) not in cache:
                    # face longue : premier axe = x local (le fil)
                    w_px = max(int(2 * ea * 1000 * PX_MM), 4)
                    h_px = max(int(2 * eb * 1000 * PX_MM), 4)
                    cache[(a, sg)] = textures.face(sorte, w_px, h_px, albedo)
                base = cache[(a, sg)]
            th, tw = base.shape[:2]
            yy, xx = np.nonzero(sel)
            col = echantillonner(base, (s1[yy, xx] + 1) / 2 * (tw - 1),
                                 (s2[yy, xx] + 1) / 2 * (th - 1))
            lamb = max(float(n_t @ ecl.lum), 0.0)
            o = o_img[yy, xx]
            f = (ecl.ambiant + (1 - ecl.ambiant) * lamb * (1 - o)) / norm_top
            couleur[yy, xx] = col * f[:, None]

    # ombres portees sur la table (fond) : rayon de chaque point de la table
    # vers la lumiere ; et assombrissement de contact (lumiere zenithale)
    fond = fond.astype(np.float32).copy()
    zone_ombre = np.zeros((H, W), bool)
    if boites:
        pts = []
        for b in boites:
            c = b.coins()
            pts.append(c)
            for direction in (ecl.lum, np.array([0, 0, 1.0])):
                s = c[:, 2] / max(direction[2], 0.08)
                pts.append(c - s[:, None] * direction)
        uv, zz = cam.projeter(np.concatenate(pts))
        uv = uv[zz > 0.01]
        if len(uv):
            x0, y0 = np.clip(np.floor(uv.min(0)).astype(int) - 8, 0, [W, H])
            x1, y1 = np.clip(np.ceil(uv.max(0)).astype(int) + 8, 0, [W, H])
            zone_ombre[y0:y1, x0:x1] = True
    zone_ombre &= np.isfinite(prof_fond)
    if surface_table is not None:
        zone_ombre &= surface_table
    if zone_ombre.any():
        ys, xs = np.nonzero(zone_ombre)
        r1 = cam.rayons()[ys, xs] * prof_fond[ys, xs, None]
        Pt = (r1 @ T_tc[:3, :3].T + T_tc[:3, 3]).astype(np.float64)
        Pt[:, 2] = np.maximum(Pt[:, 2], 1e-4)
        o1 = np.zeros((H, W), np.float32)
        o1[ys, xs] = ombre_rayons(Pt, ecl.lum, boites)
        o1 = cv2.GaussianBlur(o1, (0, 0), ecl.flou_ombre)
        o2 = np.zeros((H, W), np.float32)
        o2[ys, xs] = ombre_rayons(Pt, np.array([0, 0.0, 1.0]), boites)
        o2 = cv2.GaussianBlur(o2, (0, 0), 3.0)
        assomb = 1 - np.clip(ecl.force_ombre * o1 + ecl.contact * o2, 0, 0.85)
        assomb = np.where(zone_ombre, assomb, 1.0)
        fond *= assomb[..., None]

    # composition : moyenne des pixels fins (antialiasing)
    cov = cv2.resize(piece.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    col = cv2.resize(couleur, (W, H), interpolation=cv2.INTER_AREA)
    img = col + (1 - cov[..., None]) * fond
    zf = np.where(piece, z, np.inf)
    zmin = cv2.resize(np.where(np.isfinite(zf), zf, 0), (W, H), interpolation=cv2.INTER_NEAREST)
    prof = np.where(cov > 0.5, zmin, prof_fond)
    return img, prof, ident, seules


def effets_camera(img: np.ndarray, rng) -> np.ndarray:
    """Ce que la camera fait a l'image : flou optique, exposition, gamma,
    balance des blancs, vignettage, bruit de capteur (photonique + lecture,
    plus fort a basse lumiere, en partie colore et a basse resolution comme
    apres demosaicage), compression JPEG. Image float BGR 0-1 -> uint8."""
    H, W = img.shape[:2]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.35, 1.4))
    if rng.random() < 0.08:          # flou de bouge
        k = int(rng.integers(3, 8))
        noyau = np.zeros((k, k), np.float32)
        noyau[k // 2] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        noyau = cv2.warpAffine(noyau, M, (k, k))
        img = cv2.filter2D(img, -1, noyau / max(noyau.sum(), 1e-6))
    img = img * rng.uniform(0.6, 1.35)
    if rng.random() < 0.5:           # dominante chaude, comme la lampe du jeu reel
        wb = np.array([rng.uniform(0.75, 0.95), rng.uniform(0.9, 1.0), rng.uniform(1.0, 1.15)])
    else:
        wb = rng.uniform(0.85, 1.15, 3)
    img = img * wb.astype(np.float32)
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((u - W / 2) / (W / 2)) ** 2 + ((v - H / 2) / (W / 2)) ** 2
    img = img * (1 - rng.uniform(0, 0.35) * r2)[..., None]
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)
    sig_p = rng.uniform(0.0, 0.03)
    sig_l = rng.uniform(0.002, 0.02)
    bruit = rng.standard_normal(img.shape).astype(np.float32) * np.sqrt(sig_p ** 2 * img + sig_l ** 2)
    if rng.random() < 0.5:
        bc = rng.standard_normal((H // 2, W // 2, 3)).astype(np.float32) * rng.uniform(0, 0.012)
        bruit += cv2.resize(bc, (W, H))
    img = np.clip((img + bruit) * 255, 0, 255).astype(np.uint8)
    q = int(rng.uniform(35, 96))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@dataclass
class ReglagesGeneration:
    #: familles de textures de table permises
    familles: Tuple[str, ...] = FAMILLES
    #: modes de fond : reel tel quel, reel effacee (inpainting), reel a nappe
    #: changee, entierement synthetique
    modes_fond: Tuple[float, float, float, float] = (0.12, 0.05, 0.28, 0.55)
    #: part des camera synthetiques tirees PRES de la pose reelle
    p_pose_reelle: float = 0.35
    #: collage de pieces REELLES (test_collage) au lieu du rendu
    collage_reel: bool = False
    scene: ReglagesScene = field(default_factory=ReglagesScene)
    #: aire visible minimale (pixels) et part visible minimale d'une piece
    #: pour l'etiqueter (une piece cachee aux 4/5 n'est pas etiquetee : le
    #: reseau ne doit pas apprendre a inventer des pieces sur 3 pixels)
    aire_min: int = 60
    part_min: float = 0.2


def _decor(rng, res: Ressources, H: int, W: int, familles) -> np.ndarray:
    """Ce qu'on voit au-dela de la table : piece reelle, mur, rideau..."""
    x = rng.random()
    if res.photos and x < 0.4:
        p = res.photos[int(rng.integers(len(res.photos)))]
        h, w = p.shape[:2]
        e = rng.uniform(1.0, 2.0)
        cw, ch = int(w / e), int(h / e)
        x0, y0 = int(rng.uniform(0, w - cw + 1)), int(rng.uniform(0, h - ch + 1))
        d = cv2.resize(p[y0:y0 + ch, x0:x0 + cw], (W, H))
        if rng.random() < 0.5:
            d = d[:, ::-1]
        return np.ascontiguousarray(d) * rng.uniform(0.6, 1.3)
    fam = [f for f in familles if f not in ("nappe",)] or ["uni"]
    d = texture_famille(rng, fam[int(rng.integers(len(fam)))], H, W, res.nappes, res.photos)
    d = cv2.GaussianBlur(d, (0, 0), rng.uniform(1, 6))
    for _ in range(int(rng.integers(0, 6))):   # silhouettes de meubles
        x0, y0 = int(rng.uniform(-100, W)), int(rng.uniform(-100, H))
        cv2.rectangle(d, (x0, y0), (x0 + int(rng.uniform(30, 400)), y0 + int(rng.uniform(30, 400))),
                      tuple(float(c) for c in couleur_aleatoire(rng) * rng.uniform(0.3, 0.9)), -1)
    return cv2.GaussianBlur(d, (0, 0), rng.uniform(0.5, 3)) * rng.uniform(0.4, 1.1)


def _texture_table(rng, res: Ressources, familles) -> Tuple[np.ndarray, str]:
    fam = [f for f in familles if (f != "nappe" or res.nappes) and (f != "photo" or res.photos)]
    if not fam:
        fam = ["uni"]
    f = fam[int(rng.integers(len(fam)))]
    taille = 1024
    tex = texture_famille(rng, f, taille, taille, res.nappes, res.photos)
    return tex, f


def generer(rng, res: Ressources, types: Dict[str, Sequence[float]],
            reglages: ReglagesGeneration = ReglagesGeneration()) -> Rendu:
    """Une image synthetique et ses etiquettes."""
    familles = reglages.familles
    W, H = res.W, res.H
    mode = int(rng.choice(4, p=reglages.modes_fond))
    if not res.fonds and mode < 3:
        mode = 3
    infos = {}
    surface = None
    if mode in (0, 2):
        fr = res.fonds[int(rng.integers(len(res.fonds)))]
        cam = camera_de_plan(fr.K, W, H, fr.plan)
        fond = fr.image.astype(np.float32) / 255
        zone_img = fr.zone
        _, prof_fond, _ = rendre_table(cam, np.zeros((8, 8, 3), np.float32), 1.0, 0, (0, 0),
                                       (-10, 10, -10, 10))
        prof_fond = np.where(zone_img, prof_fond, np.inf).astype(np.float32)
        surface = zone_img
        infos["fond"] = fr.nom
        if mode == 2:          # meme piece, meme camera, autre nappe
            tex, fam = _texture_table(rng, res, familles)
            img_t, _, _ = rendre_table(cam, tex, rng.uniform(0.4, 1.5), rng.uniform(0, math.pi),
                                       tuple(rng.uniform(0, 1024, 2)), (-10, 10, -10, 10))
            # l'eclairage reel de la table (gradient lent) garde sur la nouvelle nappe
            lum = cv2.cvtColor(fr.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            moy = float(lum[zone_img].mean()) if zone_img.any() else 128.0
            lent = cv2.GaussianBlur(np.where(zone_img, lum, moy), (0, 0), 25) / max(moy, 1)
            img_t = img_t * lent[..., None] * rng.uniform(0.7, 1.15)
            # bord de la table adouci (la zone est un peu plus petite que la table)
            zf = cv2.GaussianBlur(cv2.dilate(zone_img.astype(np.uint8), np.ones((9, 9), np.uint8))
                                  .astype(np.float32), (0, 0), 1.5)[..., None]
            fond = zf * img_t + (1 - zf) * fond
            infos["famille"] = fam
        else:
            infos["famille"] = "reel"
    elif mode == 1:
        eff = res.image_effacee(rng)
        if eff is None:
            return generer(rng, res, types, reglages)
        img_e, plan = eff
        cam = camera_de_plan(res.K, W, H, plan)
        zone_img = res.fonds[0].zone if res.fonds else np.ones((H, W), bool)
        fond = img_e.astype(np.float32) / 255
        _, prof_fond, _ = rendre_table(cam, np.zeros((8, 8, 3), np.float32), 1.0, 0, (0, 0),
                                       (-10, 10, -10, 10))
        prof_fond = np.where(zone_img, prof_fond, np.inf).astype(np.float32)
        surface = zone_img
        infos["famille"] = "reel_efface"
    else:
        h0, p0, r0 = angles_de_plan(res.plan_reel)
        if rng.random() < reglages.p_pose_reelle:
            plan = plan_tire(rng, h0 * rng.uniform(0.85, 1.2), p0 + rng.normal(0, 4),
                             r0 + rng.normal(0, 2))
        else:
            plan = plan_tire(rng, rng.uniform(0.10, 0.45), rng.uniform(8, 70), rng.normal(0, 5))
        K = res.K.copy()
        K[0, 0] *= rng.uniform(0.95, 1.05)
        K[1, 1] = K[0, 0] * rng.uniform(0.995, 1.005)
        K[:2, 2] += rng.normal(0, 4, 2)
        cam = Camera(K, W, H, camera_de_plan(K, W, H, plan).T_ct)
        # la table : un rectangle, bord lointain parfois visible
        y_loin = rng.uniform(0.35, 1.6)
        rect = (-rng.uniform(0.25, 1.2), rng.uniform(0.25, 1.2), -1.0, y_loin)
        tex, fam = _texture_table(rng, res, familles)
        mm_px = rng.uniform(0.35, 1.6)
        img_t, prof_fond, dedans = rendre_table(cam, tex, mm_px, rng.uniform(0, math.pi),
                                                tuple(rng.uniform(0, 1024, 2)), rect)
        decor = _decor(rng, res, H, W, familles)
        dd = dedans[..., None].astype(np.float32)
        fond = dd * img_t + (1 - dd) * decor
        # lumiere lente sur la table (une lampe pas tres loin)
        g = bruit_fractal(rng, H, W, echelle=300, octaves=2)
        fond = fond * (1 + rng.uniform(0, 0.4) * (g[..., None] - 0.5))
        zone_img = dedans & (prof_fond < rng.uniform(0.6, 1.1))
        zone_img = cv2.erode(zone_img.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        surface = dedans
        infos["famille"] = fam
    infos["mode"] = ["reel", "reel_efface", "reel_nappe_changee", "synthetique"][mode]

    # zone ou poser : pixels de la table, et au-dela du bord de l'image si la
    # table y continue (une piece coupee par le bord est un cas reel)
    def zone_ok(P: np.ndarray) -> np.ndarray:
        uv, zz = cam.projeter(P)
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        dans_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zz > 0)
        ok = np.zeros(len(P), bool)
        ok[dans_img] = zone_img[v[dans_img], u[dans_img]]
        # hors image : sur les cotes et en bas seulement, et pas trop loin
        hors = ~dans_img & (zz > 0) & (v >= 0) & (u > -0.1 * W) & (u < 1.1 * W) & (v < 1.15 * H)
        ok |= hors
        return ok

    ys, xs = np.nonzero(zone_img)
    if len(xs) < 50:
        return generer(rng, res, types, reglages)
    i = int(rng.integers(len(xs)))
    centre, _ = cam.sur_table(np.array([[xs[i], ys[i]]], float))
    centre = centre[0]

    ecl = tirer_eclairage(rng)
    boites = tirer_scene(rng, types, cam, zone_ok, centre, reglages.scene)
    textures = Textures(res, rng)
    img, prof, ident, seules = rendre_scene(cam, fond, prof_fond, boites, ecl, textures, rng, surface)
    image = effets_camera(img, rng)
    masques, classes, gardees = [], [], []
    for k, m in masques_visibles(ident, len(boites), W, H):
        b = boites[k]
        if b.classe is None:
            continue
        n_fin = int(np.count_nonzero(ident == k))
        if m.sum() < reglages.aire_min or n_fin < reglages.part_min * max(seules[k], 1):
            continue
        masques.append(m)
        classes.append(b.classe)
        gardees.append(b)
    infos["n_boites"] = len(boites)
    infos["eclairage"] = {"lum": ecl.lum.round(3).tolist(), "ambiant": round(ecl.ambiant, 3)}
    return Rendu(image, masques, classes, gardees, prof, cam, infos)


def masques_visibles(ident: np.ndarray, n: int, W: int, H: int):
    """Masques visibles (W x H) des boites 0..n-1 a partir des identifiants
    sur-echantillonnes : chaque pixel va a la boite qui en couvre la plus
    grande part, si elle en couvre au moins la moitie (deux masques ne se
    recouvrent jamais)."""
    if n == 0:
        return []
    cov = np.stack([cv2.resize((ident == k).astype(np.float32), (W, H),
                               interpolation=cv2.INTER_AREA) for k in range(n)])
    meilleur = cov.argmax(0)
    ok = cov.max(0) >= 0.5
    return [(k, ok & (meilleur == k)) for k in range(n)]


def coins_dessus(b: Boite) -> np.ndarray:
    """Les 4 coins de la face la plus haute (repere table), dans l'ordre du
    contour : pour un futur modele de points cles (saisie)."""
    a = int(np.argmax(b.R[2] * np.sign(b.R[2]) * 1.0))
    sg = 1.0 if b.R[2, a] > 0 else -1.0
    ia, ib = [(1, 2), (0, 2), (0, 1)][a]
    c = b.t + sg * b.e[a] * b.R[:, a]
    return np.array([c + sa * b.e[ia] * b.R[:, ia] + sb * b.e[ib] * b.R[:, ib]
                     for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


# ---------------------------------------------------------------------------
# Test "collage" : pieces REELLES sur des fonds jamais vus
# ---------------------------------------------------------------------------

def generer_collage(rng, res: Ressources, banque, reglages: ReglagesGeneration) -> Rendu:
    """Un fond synthetique (pose reelle, famille permise) et des pieces
    REELLES collees (copier_coller.coller : echelle perspective, contacts,
    occultations). Pour juger un reseau entraine sur le reel seul sans le
    penaliser par l'aspect des pieces rendues."""
    import copier_coller
    W, H = res.W, res.H
    h0, p0, r0 = angles_de_plan(res.plan_reel)
    plan = plan_tire(rng, h0 * rng.uniform(0.9, 1.1), p0 + rng.normal(0, 2), r0 + rng.normal(0, 1))
    cam = camera_de_plan(res.K, W, H, plan)
    tex, fam = _texture_table(rng, res, reglages.familles)
    rect = (-1.0, 1.0, -1.0, rng.uniform(0.5, 1.2))
    img_t, prof, dedans = rendre_table(cam, tex, rng.uniform(0.4, 1.4), rng.uniform(0, math.pi),
                                       tuple(rng.uniform(0, 1024, 2)), rect)
    decor = _decor(rng, res, H, W, reglages.familles)
    dd = dedans[..., None].astype(np.float32)
    fond = effets_camera(dd * img_t + (1 - dd) * decor, rng)
    img, masques, classes = copier_coller.coller(fond, [], [], banque, rng,
                                                 int(rng.integers(2, 8)), cam.plan(), cam.K)
    return Rendu(img, masques, classes, [], prof, cam, {"famille": fam, "mode": "collage"})


# ---------------------------------------------------------------------------
# Ecriture du jeu et controle
# ---------------------------------------------------------------------------

_RES: Optional[Ressources] = None
_BANQUE = None


def _init_processus(racine, cache, collage):
    global _RES, _BANQUE
    cv2.setNumThreads(1)
    _RES = Ressources(racine, cache)
    if collage:
        _BANQUE = banque_validation(racine)


def banque_validation(racine: str = commun.DATASET):
    """Pieces decoupees dans les images reelles de VALIDATION (exporter.py) :
    jamais vues a l'entrainement."""
    import copier_coller
    import exporter
    banque = []
    for n in commun.noms_captures(racine):
        ann = commun.lire_annotation(n, racine)
        if not ann or ann.get("statut") != "valide" or not exporter.en_validation(n, 0.2):
            continue
        img = commun.charger_image(n, racine)
        ms, cs = exporter.masques_de(ann)
        sures = [j for j, i in enumerate(ann["instances"]) if not i.get("douteux")]
        banque += copier_coller.extraire(img, [ms[j] for j in sures], [cs[j] for j in sures],
                                         ann.get("plan"), ann.get("K"))
    return banque


def _une(args):
    i, graine, dossier, reglages_d, profondeur = args
    import exporter
    rng = np.random.default_rng([graine, i])
    reglages = ReglagesGeneration(**reglages_d)
    types = commun.charger_types(_RES.racine)
    if reglages.collage_reel:
        r = generer_collage(rng, _RES, _BANQUE, reglages)
    else:
        r = generer(rng, _RES, types, reglages)
    nom = "s_{:06d}".format(i)
    cv2.imwrite(os.path.join(dossier, "images", nom + ".jpg"), r.image,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    classes = list(types)
    lignes = []
    for m, c in zip(r.masques, r.classes):
        polys = commun.masque_vers_polygones(m, aire_min=10)
        if polys:
            lignes.append(exporter.ligne_yolo(classes.index(c), polys, r.camera.W, r.camera.H))
    with open(os.path.join(dossier, "labels", nom + ".txt"), "w") as f:
        f.write("\n".join(lignes) + ("\n" if lignes else ""))
    pieces = []
    for b, c in zip(r.boites, r.classes):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = b.R, b.t
        cd = coins_dessus(b)
        uv, _ = r.camera.projeter(cd)
        pieces.append({"classe": c, "pose": b.pose, "T_table": T.round(6).tolist(),
                       "demi_dimensions": b.e.round(6).tolist(),
                       "coins_dessus_table": cd.round(6).tolist(),
                       "coins_dessus_image": uv.round(2).tolist()})
    meta = {"image": nom + ".jpg", "K": r.camera.K.round(4).tolist(),
            "T_camera_table": r.camera.T_ct.round(6).tolist(),
            "plan": r.camera.plan().round(6).tolist(), "pieces": pieces, **r.infos}
    with open(os.path.join(dossier, "meta", nom + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    if profondeur:
        p = np.where(np.isfinite(r.profondeur), r.profondeur * 1000, 0).astype(np.uint16)
        cv2.imwrite(os.path.join(dossier, "meta", nom + "_profondeur.png"), p)
    return len(lignes), r.infos.get("famille", "?")


def controle(dossier: str, n: int = 24, sortie: Optional[str] = None, colonnes: int = 4,
             noms_classes: Sequence[str] = ("epaisse", "fine")) -> str:
    """Mosaique : images et masques (contours colores par classe) lus dans
    labels/ : verifie l'export lui-meme, pas seulement le rendu."""
    imgs = sorted(glob.glob(os.path.join(dossier, "images", "*.jpg")))[:n]
    vignettes = []
    for f in imgs:
        im = cv2.imread(f)
        H, W = im.shape[:2]
        lab = os.path.join(dossier, "labels", os.path.basename(f)[:-4] + ".txt")
        calque = im.copy()
        with open(lab) as f_lab:
            lignes = f_lab.read().split("\n")
        for ligne in lignes:
            if not ligne.strip():
                continue
            v = ligne.split()
            c = int(v[0])
            p = (np.array(v[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
            col = [(40, 170, 255), (255, 200, 40), (90, 220, 90)][c % 3]
            cv2.fillPoly(calque, [p], col)
            cv2.polylines(im, [p], True, col, 1, cv2.LINE_AA)
        im = cv2.addWeighted(im, 0.7, calque, 0.3, 0)
        cv2.putText(im, os.path.basename(f)[:-4], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vignettes.append(cv2.resize(im, (W // 2, H // 2)))
    if not vignettes:
        raise SystemExit("aucune image dans " + dossier)
    while len(vignettes) % colonnes:
        vignettes.append(np.zeros_like(vignettes[0]))
    m = np.vstack([np.hstack(vignettes[i:i + colonnes]) for i in range(0, len(vignettes), colonnes)])
    sortie = sortie or os.path.join(dossier, "controle.jpg")
    cv2.imwrite(sortie, m, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return sortie


def generer_jeu(n: int, dossier: str, reglages: ReglagesGeneration = ReglagesGeneration(),
                graine: int = 0, processus: int = 8, racine: str = commun.DATASET,
                cache: str = CACHE, profondeur: bool = False, debut: int = 0,
                verbeux: bool = True) -> dict:
    """n images dans `dossier` (images/, labels/, meta/, data.yaml). Chaque
    image a sa graine (graine, indice) : le jeu est reproductible et peut
    etre complete (`debut`) sans changer les images deja faites."""
    for s in ("images", "labels", "meta"):
        commun.dossier(dossier, s)
    types = commun.charger_types(racine)
    rd = {k: getattr(reglages, k) for k in ("familles", "modes_fond", "p_pose_reelle",
                                             "collage_reel", "scene", "aire_min", "part_min")}
    taches = [(i, graine, dossier, rd, profondeur) for i in range(debut, debut + n)]
    t0 = time.time()
    resultats = []
    if processus <= 1:
        _init_processus(racine, cache, reglages.collage_reel)
        for k, t in enumerate(taches):
            resultats.append(_une(t))
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with ProcessPoolExecutor(processus, mp_context=mp.get_context("spawn"),
                                 initializer=_init_processus,
                                 initargs=(racine, cache, reglages.collage_reel)) as ex:
            for k, r in enumerate(ex.map(_une, taches, chunksize=4)):
                resultats.append(r)
                if verbeux and (k + 1) % 100 == 0:
                    print("  {}/{} images ({:.1f} s)".format(k + 1, n, time.time() - t0), flush=True)
    duree = time.time() - t0
    with open(os.path.join(dossier, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# genere par synthese_images.py : pour EVALUER un modele sur ce jeu\n")
        f.write("path: {}\ntrain: images\nval: images\nnames:\n".format(os.path.abspath(dossier)))
        for k, c in enumerate(types):
            f.write("  {}: {}\n".format(k, c))
    familles = {}
    for _, fam in resultats:
        familles[fam] = familles.get(fam, 0) + 1
    stats = {"images": n, "pieces": int(sum(r[0] for r in resultats)), "duree_s": round(duree, 1),
             "s_par_image": round(duree / max(n, 1), 3), "processus": processus,
             "familles": familles, "dossier": dossier}
    with open(os.path.join(dossier, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Images synthetiques par randomisation de domaine")
    ap.add_argument("--textures", action="store_true",
                    help="(re)construire la banque de textures de bois (pieces annotees)")
    ap.add_argument("--n", type=int, default=2000, help="nombre d'images")
    ap.add_argument("--nom", default=None, help="sous-dossier de synthetique/ "
                    "(defaut : entrainement, test_fonds, test_collage)")
    ap.add_argument("--familles", default=None,
                    help="familles de tables permises, separees par des virgules ({})".format(
                        ",".join(FAMILLES)))
    ap.add_argument("--exclure", default=None,
                    help="familles a exclure (ex. {} : pour mesurer la robustesse aux "
                    "fonds jamais vus)".format(",".join(FAMILLES_TEST)))
    ap.add_argument("--test-fonds", action="store_true",
                    help="jeu de TEST : seulement les familles {}, pieces rendues".format(
                        ",".join(FAMILLES_TEST)))
    ap.add_argument("--test-collage", action="store_true",
                    help="jeu de TEST : familles {}, pieces REELLES de la validation "
                    "collees".format(",".join(FAMILLES_TEST)))
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--debut", type=int, default=0, help="premier indice (pour completer un jeu)")
    ap.add_argument("--processus", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--profondeur", action="store_true", help="ecrire aussi la profondeur (mm, PNG 16 bits)")
    ap.add_argument("--controle", type=int, default=24, help="images dans la mosaique de controle")
    ap.add_argument("--dataset", default=commun.DATASET)
    a = ap.parse_args()
    if a.textures:
        print("banque de textures de bois (pieces validees) ...")
        s = construire_textures(a.dataset)
        print("faces longues : {long}, bouts : {bout} ; couleurs moyennes (BGR) {moyennes}".format(**s))
        r = Ressources(a.dataset, verbeux=True)
        print("nappes redressees : {}".format(len(r.nappes)))
        return
    familles = list(FAMILLES)
    nom = a.nom or "entrainement"
    reglages = ReglagesGeneration()
    if a.test_fonds or a.test_collage:
        familles = list(FAMILLES_TEST)
        nom = a.nom or ("test_fonds" if a.test_fonds else "test_collage")
        # que des tables synthetiques (ou la vraie piece a nappe changee)
        reglages.modes_fond = (0.0, 0.0, 0.3, 0.7)
        reglages.collage_reel = a.test_collage
    if a.familles:
        familles = [f.strip() for f in a.familles.split(",") if f.strip()]
    if a.exclure:
        ex = {f.strip() for f in a.exclure.split(",")}
        familles = [f for f in familles if f not in ex]
    inconnues = set(familles) - set(FAMILLES)
    if inconnues:
        raise SystemExit("familles inconnues : {}".format(sorted(inconnues)))
    reglages.familles = tuple(familles)
    dossier = os.path.join(SORTIE, nom)
    print("{} images -> {} (familles : {})".format(a.n, dossier, ", ".join(familles)))
    s = generer_jeu(a.n, dossier, reglages, a.graine, a.processus, a.dataset,
                    profondeur=a.profondeur, debut=a.debut)
    print("{images} images, {pieces} pieces, {duree_s} s ({s_par_image} s/image, "
          "{processus} processus) ; familles {familles}".format(**s))
    if a.controle:
        print("controle :", controle(dossier, a.controle))


if __name__ == "__main__":
    main()
