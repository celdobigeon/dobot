# -*- coding: utf-8 -*-
"""
Lecture d'un fichier SVG et fabrication d'un programme de trace.

Le dessin devient une suite de polylignes exprimees dans un repere de
travail, puis un programme : liaison en l'air, descente sur le papier,
trace, levee. Le stylo n'ayant aucun actionneur, la levee est un
mouvement en z.

    from dobot_sim import programme_depuis_svg, repere_travail

    FEUILLE = repere_travail(x=210, y=0, z=12)        # le plan du papier
    robot = programme_depuis_svg("logo.svg", repere=FEUILLE, largeur=120)
    print(robot.resume())
    robot.simuler()

Ce qui est lu
-------------
Les elements geometriques -- `path`, `rect`, `circle`, `ellipse`, `line`,
`polyline`, `polygon` -- les groupes imbriques, les attributs `transform`
(translate, scale, rotate, matrix, skewX, skewY) et le `viewBox`.

Ce qui est ignore
-----------------
Tout ce qui n'est pas une trajectoire : remplissages, epaisseurs,
couleurs, pointilles, opacite. Le texte aussi, faute de contours de
police : convertissez-le en chemin avant d'exporter. Les images
matricielles, degrades et masques le sont egalement.

Le cas du trace
---------------
La descente sur le papier et le dernier point d'un trait se font a
l'ARRET AU POINT : un trait doit commencer et finir ou on l'a dit. Les
points intermediaires, eux, sont lisses -- sans quoi le robot
s'arreterait a chacun des centaines de points d'une courbe aplatie. Le
lissage sert ici exactement a ce pour quoi il existe.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pilote import Pilote
from .robot import cible

__all__ = ["lire_svg", "programme_depuis_svg", "ErreurSVG"]

#: tolerance d'aplatissement des courbes, en unites utilisateur du document
TOLERANCE = 0.2

Point = Tuple[float, float]
#: transformation affine 2D : (a, b, c, d, e, f), x' = a x + c y + e
Affine = Tuple[float, float, float, float, float, float]

IDENTITE: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


class ErreurSVG(ValueError):
    """Fichier illisible, ou vide de toute geometrie."""


# ===========================================================================
# Transformations affines
# ===========================================================================

def composer(m: Affine, n: Affine) -> Affine:
    """m puis n applique au resultat : renvoie la composee m*n."""
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1 * a2 + c1 * b2,
            b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2,
            b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1,
            b1 * e2 + d1 * f2 + f1)


def appliquer(m: Affine, p: Point) -> Point:
    a, b, c, d, e, f = m
    x, y = p
    return (a * x + c * y + e, b * x + d * y + f)


_NOMBRES = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _nombres(texte: str) -> List[float]:
    return [float(x) for x in _NOMBRES.findall(texte or "")]


def lire_transform(texte: str) -> Affine:
    """
    Analyse un attribut `transform` SVG.

    Les operations s'appliquent de gauche a droite, la premiere ecrite
    etant la plus exterieure -- c'est la convention SVG.
    """
    m = IDENTITE
    for nom, args in re.findall(r"(\w+)\s*\(([^)]*)\)", texte or ""):
        v = _nombres(args)
        if nom == "translate":
            tx = v[0] if v else 0.0
            ty = v[1] if len(v) > 1 else 0.0
            op = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif nom == "scale":
            sx = v[0] if v else 1.0
            sy = v[1] if len(v) > 1 else sx
            op = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif nom == "rotate":
            a = math.radians(v[0]) if v else 0.0
            co, si = math.cos(a), math.sin(a)
            op = (co, si, -si, co, 0.0, 0.0)
            if len(v) >= 3:                      # rotation autour d'un point
                cx, cy = v[1], v[2]
                op = composer(composer((1, 0, 0, 1, cx, cy), op),
                              (1, 0, 0, 1, -cx, -cy))
        elif nom == "matrix" and len(v) >= 6:
            op = (v[0], v[1], v[2], v[3], v[4], v[5])
        elif nom == "skewX":
            op = (1.0, 0.0, math.tan(math.radians(v[0] if v else 0.0)),
                  1.0, 0.0, 0.0)
        elif nom == "skewY":
            op = (1.0, math.tan(math.radians(v[0] if v else 0.0)),
                  0.0, 1.0, 0.0, 0.0)
        else:
            continue
        m = composer(m, op)
    return m


# ===========================================================================
# Aplatissement des courbes
# ===========================================================================

def _bezier3(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u * u * u * p0[0] + 3 * u * u * t * p1[0]
            + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1]
            + 3 * u * t * t * p2[1] + t * t * t * p3[1])


def _pas_cubique(p0, p1, p2, p3, tol: float) -> int:
    """
    Nombre de segments pour rester sous la tolerance.

    On majore la fleche de la courbe par celle de son polygone de
    controle, ce qui donne un decoupage sur et peu couteux.
    """
    d = max(math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
            math.hypot(p2[0] - p1[0], p2[1] - p1[1]),
            math.hypot(p3[0] - p2[0], p3[1] - p2[1]))
    if d <= tol:
        return 1
    return max(2, min(400, int(math.ceil(math.sqrt(3.0 * d / tol) * 2))))


def aplatir_cubique(p0, p1, p2, p3, tol: float = TOLERANCE) -> List[Point]:
    n = _pas_cubique(p0, p1, p2, p3, tol)
    return [_bezier3(p0, p1, p2, p3, i / n) for i in range(1, n + 1)]


def aplatir_quadratique(p0, p1, p2, tol: float = TOLERANCE) -> List[Point]:
    """Une quadratique est une cubique : on releve les points de controle."""
    c1 = (p0[0] + 2.0 / 3.0 * (p1[0] - p0[0]),
          p0[1] + 2.0 / 3.0 * (p1[1] - p0[1]))
    c2 = (p2[0] + 2.0 / 3.0 * (p1[0] - p2[0]),
          p2[1] + 2.0 / 3.0 * (p1[1] - p2[1]))
    return aplatir_cubique(p0, c1, c2, p2, tol)


def aplatir_arc(p0: Point, rx: float, ry: float, phi_deg: float,
                grand: bool, sens: bool, p1: Point,
                tol: float = TOLERANCE) -> List[Point]:
    """
    Arc elliptique, du parametrage par extremites a celui par centre.

    Suit l'annexe F.6 de la specification SVG, y compris la correction
    des rayons trop petits pour joindre les deux points.
    """
    x0, y0 = p0
    x1, y1 = p1
    if (x0, y0) == (x1, y1):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:               # degenere : un segment
        return [p1]

    phi = math.radians(phi_deg)
    co, si = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x0 - x1) / 2.0, (y0 - y1) / 2.0
    x1p = co * dx2 + si * dy2
    y1p = -si * dx2 + co * dy2

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:                              # rayons trop petits
        r = math.sqrt(lam)
        rx, ry = rx * r, ry * r

    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den > 0 else 0.0
    if grand == sens:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = co * cxp - si * cyp + (x0 + x1) / 2.0
    cy = si * cxp + co * cyp + (y0 + y1) / 2.0

    def angle(ux, uy, vx, vy):
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        if n < 1e-15:
            return 0.0
        c = max(-1.0, min(1.0, (ux * vx + uy * vy) / n))
        a = math.acos(c)
        return -a if ux * vy - uy * vx < 0 else a

    th0 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sens and dth > 0:
        dth -= 2 * math.pi
    elif sens and dth < 0:
        dth += 2 * math.pi

    rayon = max(rx, ry)
    n = max(2, min(800, int(math.ceil(abs(dth) / (2 * math.acos(
        max(-1.0, min(1.0, 1.0 - tol / max(rayon, tol))))))))) if rayon > tol else 2
    pts = []
    for i in range(1, n + 1):
        th = th0 + dth * i / n
        xe = cx + rx * math.cos(th) * co - ry * math.sin(th) * si
        ye = cy + rx * math.cos(th) * si + ry * math.sin(th) * co
        pts.append((xe, ye))
    return pts


# ===========================================================================
# L'attribut `d` d'un chemin
# ===========================================================================

_JETONS = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])|"
                     r"([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


def lire_chemin(d: str, tol: float = TOLERANCE) -> List[List[Point]]:
    """
    Analyse un attribut `d` et renvoie ses sous-chemins aplatis.

    Un sous-chemin est une polyligne : la liste des points par lesquels
    l'outil passe. Un `Z` referme la polyligne en y repetant le premier
    point.
    """
    jetons: List = []
    for cmd, nombre in _JETONS.findall(d or ""):
        jetons.append(cmd if cmd else float(nombre))

    chemins: List[List[Point]] = []
    courant: List[Point] = []
    pos: Point = (0.0, 0.0)
    depart: Point = (0.0, 0.0)
    ctrl: Optional[Point] = None          # dernier point de controle, pour S/T
    cmd = ""
    i = 0

    def nb(k: int = 1):
        nonlocal i
        v = jetons[i:i + k]
        i += k
        return v

    while i < len(jetons):
        if isinstance(jetons[i], str):
            cmd = jetons[i]
            i += 1
            if cmd in "Zz":
                if courant:
                    if courant[0] != courant[-1]:
                        courant.append(courant[0])
                    chemins.append(courant)
                    courant = []
                pos = depart
                ctrl = None
                continue
        if i >= len(jetons) or isinstance(jetons[i], str):
            continue
        rel = cmd.islower()
        c = cmd.upper()

        if c == "M":
            v = nb(2)
            if len(v) < 2:
                break
            p = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
            if courant:
                chemins.append(courant)
            courant = [p]
            pos = depart = p
            ctrl = None
            cmd = "l" if rel else "L"          # les suivants sont des lignes
        elif c == "L":
            v = nb(2)
            if len(v) < 2:
                break
            p = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
            courant.append(p)
            pos, ctrl = p, None
        elif c == "H":
            v = nb(1)
            if not v:
                break
            p = (pos[0] + v[0], pos[1]) if rel else (v[0], pos[1])
            courant.append(p)
            pos, ctrl = p, None
        elif c == "V":
            v = nb(1)
            if not v:
                break
            p = (pos[0], pos[1] + v[0]) if rel else (pos[0], v[0])
            courant.append(p)
            pos, ctrl = p, None
        elif c in ("C", "S"):
            if c == "C":
                v = nb(6)
                if len(v) < 6:
                    break
                p1 = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
                p2 = (pos[0] + v[2], pos[1] + v[3]) if rel else (v[2], v[3])
                p3 = (pos[0] + v[4], pos[1] + v[5]) if rel else (v[4], v[5])
            else:
                v = nb(4)
                if len(v) < 4:
                    break
                p1 = ((2 * pos[0] - ctrl[0], 2 * pos[1] - ctrl[1])
                      if ctrl else pos)
                p2 = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
                p3 = (pos[0] + v[2], pos[1] + v[3]) if rel else (v[2], v[3])
            if not courant:
                courant = [pos]
            courant.extend(aplatir_cubique(pos, p1, p2, p3, tol))
            pos, ctrl = p3, p2
        elif c in ("Q", "T"):
            if c == "Q":
                v = nb(4)
                if len(v) < 4:
                    break
                p1 = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
                p2 = (pos[0] + v[2], pos[1] + v[3]) if rel else (v[2], v[3])
            else:
                v = nb(2)
                if len(v) < 2:
                    break
                p1 = ((2 * pos[0] - ctrl[0], 2 * pos[1] - ctrl[1])
                      if ctrl else pos)
                p2 = (pos[0] + v[0], pos[1] + v[1]) if rel else (v[0], v[1])
            if not courant:
                courant = [pos]
            courant.extend(aplatir_quadratique(pos, p1, p2, tol))
            pos, ctrl = p2, p1
        elif c == "A":
            v = nb(7)
            if len(v) < 7:
                break
            p = (pos[0] + v[5], pos[1] + v[6]) if rel else (v[5], v[6])
            if not courant:
                courant = [pos]
            courant.extend(aplatir_arc(pos, v[0], v[1], v[2],
                                       bool(v[3]), bool(v[4]), p, tol))
            pos, ctrl = p, None
        else:
            i += 1

    if courant:
        chemins.append(courant)
    return [c for c in chemins if len(c) >= 2]


# ===========================================================================
# Les formes de base
# ===========================================================================

def _att(e, nom, defaut=0.0) -> float:
    v = e.get(nom)
    if v is None:
        return defaut
    n = _nombres(v)
    return n[0] if n else defaut


def _forme(e, balise: str, tol: float) -> List[List[Point]]:
    """Convertit une forme de base en polylignes."""
    if balise == "rect":
        x, y = _att(e, "x"), _att(e, "y")
        w, h = _att(e, "width"), _att(e, "height")
        if w <= 0 or h <= 0:
            return []
        rx, ry = _att(e, "rx"), _att(e, "ry")
        rx = ry if (rx == 0 and ry > 0) else rx
        ry = rx if (ry == 0 and rx > 0) else ry
        rx, ry = min(rx, w / 2), min(ry, h / 2)
        if rx <= 0 or ry <= 0:
            return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]
        d = ("M {} {} H {} A {} {} 0 0 1 {} {} V {} A {} {} 0 0 1 {} {} "
             "H {} A {} {} 0 0 1 {} {} V {} A {} {} 0 0 1 {} {} Z").format(
            x + rx, y, x + w - rx, rx, ry, x + w, y + ry, y + h - ry,
            rx, ry, x + w - rx, y + h, x + rx, rx, ry, x, y + h - ry,
            y + ry, rx, ry, x + rx, y)
        return lire_chemin(d, tol)
    if balise in ("circle", "ellipse"):
        cx, cy = _att(e, "cx"), _att(e, "cy")
        if balise == "circle":
            rx = ry = _att(e, "r")
        else:
            rx, ry = _att(e, "rx"), _att(e, "ry")
        if rx <= 0 or ry <= 0:
            return []
        d = ("M {} {} A {} {} 0 1 0 {} {} A {} {} 0 1 0 {} {} Z").format(
            cx - rx, cy, rx, ry, cx + rx, cy, rx, ry, cx - rx, cy)
        return lire_chemin(d, tol)
    if balise == "line":
        return [[(_att(e, "x1"), _att(e, "y1")),
                 (_att(e, "x2"), _att(e, "y2"))]]
    if balise in ("polyline", "polygon"):
        v = _nombres(e.get("points", ""))
        pts = list(zip(v[0::2], v[1::2]))
        if len(pts) < 2:
            return []
        if balise == "polygon" and pts[0] != pts[-1]:
            pts.append(pts[0])
        return [pts]
    return []


# ===========================================================================
# Lecture du document
# ===========================================================================

def _sans_espace(balise: str) -> str:
    return balise.rsplit("}", 1)[-1]


def lire_svg(chemin: str, tolerance: float = TOLERANCE) -> List[List[Point]]:
    """
    Lit un fichier SVG et renvoie ses polylignes, dans les coordonnees
    du document (y vers le BAS, comme en SVG).

    Chaque polyligne est un trait continu : l'outil s'y pose au premier
    point et s'en releve au dernier.
    """
    if not os.path.exists(chemin):
        raise ErreurSVG("fichier introuvable : {}".format(chemin))
    try:
        racine = ET.parse(chemin).getroot()
    except ET.ParseError as exc:
        raise ErreurSVG("SVG illisible ({}) : {}".format(
            os.path.basename(chemin), exc))

    polylignes: List[List[Point]] = []

    def descendre(noeud, ctm: Affine):
        for e in noeud:
            balise = _sans_espace(e.tag)
            if balise in ("defs", "metadata", "title", "desc", "style",
                          "symbol", "clipPath", "mask", "marker", "pattern"):
                continue
            m = composer(ctm, lire_transform(e.get("transform", "")))
            if balise == "path":
                brut = lire_chemin(e.get("d", ""), tolerance)
            elif balise in ("rect", "circle", "ellipse", "line",
                            "polyline", "polygon"):
                brut = _forme(e, balise, tolerance)
            else:
                brut = []
            for poly in brut:
                polylignes.append([appliquer(m, p) for p in poly])
            if balise in ("g", "a", "svg", "switch"):
                descendre(e, m)

    descendre(racine, lire_transform(racine.get("transform", "")))
    if not polylignes:
        raise ErreurSVG(
            "aucune geometrie dans {} -- le dessin est-il fait de texte ou "
            "d'images ? Convertissez le texte en chemin avant d'exporter."
            .format(os.path.basename(chemin)))
    return polylignes


# ===========================================================================
# Du dessin au programme
# ===========================================================================

def _boite(polylignes: Sequence[Sequence[Point]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for poly in polylignes for p in poly]
    ys = [p[1] for poly in polylignes for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def programme_depuis_svg(chemin: str,
                         repere=None,
                         largeur: float = 100.0,
                         hauteur: Optional[float] = None,
                         levee: float = 10.0,
                         profondeur: float = 0.0,
                         tolerance: float = TOLERANCE,
                         vitesse_trace: float = 60.0,
                         vitesse_liaison: float = 150.0,
                         lissage_trace: float = 60.0,
                         outil: str = "stylo",
                         nom: Optional[str] = None,
                         journal=None) -> Pilote:
    """
    Construit le programme qui trace le dessin d'un fichier SVG.

    Le dessin est mis a l'echelle pour tenir dans `largeur` millimetres
    en conservant ses proportions, puis centre sur l'origine du repere.
    L'axe y du SVG pointant vers le bas et celui du repere vers le haut,
    il est retourne : le trace ressemble au fichier vu a l'ecran.

    Parametres
    ----------
    repere : Mat 4x4
        le plan du papier, cree par `repere_travail`. A defaut, un plan
        a (210 ; 0 ; 0).
    largeur : float
        largeur du dessin, en millimetres.
    hauteur : float, optionnel
        si elle est donnee, le dessin tient dans la boite
        `largeur` x `hauteur` sans se deformer.
    levee : float
        hauteur de degagement entre deux traits, en millimetres.
    profondeur : float
        z du trace dans le repere. Zero = le plan du papier ; une valeur
        legerement negative appuie le stylo.
    lissage_trace : float
        lissage des points INTERIEURS d'un trait. Le premier et le
        dernier point restent a l'arret au point : un trait doit
        commencer et finir ou on l'a dit.

    Renvoie un `Pilote` : il reste a l'inspecter (`resume`, `controler`)
    ou a le jouer (`simuler`).
    """
    log = journal or (lambda *_: None)
    polylignes = lire_svg(chemin, tolerance)

    x0, y0, x1, y1 = _boite(polylignes)
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    echelle = largeur / dx
    if hauteur is not None:
        echelle = min(echelle, hauteur / dy)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def vers_repere(p: Point) -> Point:
        # y du SVG vers le bas, y du repere vers le haut : on retourne
        return ((p[0] - cx) * echelle, -(p[1] - cy) * echelle)

    if repere is None:
        from .robot import repere_travail
        repere = repere_travail(x=210.0, y=0.0, z=0.0)

    robot = Pilote(nom=nom or os.path.splitext(os.path.basename(chemin))[0],
                   outil=outil,
                   vitesse=vitesse_liaison, acceleration=600.0,
                   vitesse_articulaire=60.0)

    points = 0
    for poly in polylignes:
        pts = [vers_repere(p) for p in poly]
        # on ote les points confondus : ils ne tracent rien et alourdissent
        propre = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - propre[-1][0], p[1] - propre[-1][1]) > 1e-6:
                propre.append(p)
        if len(propre) < 2:
            continue
        points += len(propre)

        def vise(p, z):
            return cible(x=p[0], y=p[1], z=z, repere=repere)

        # liaison en l'air, puis descente a l'ARRET AU POINT
        robot.movj(vise(propre[0], levee), lissage=70,
                   commentaire="liaison")
        robot.movl(vise(propre[0], profondeur), vitesse=vitesse_trace,
                   commentaire="pose du stylo")
        for p in propre[1:-1]:
            robot.movl(vise(p, profondeur), vitesse=vitesse_trace,
                       lissage=lissage_trace)
        robot.movl(vise(propre[-1], profondeur), vitesse=vitesse_trace,
                   commentaire="fin du trait")
        robot.movl(vise(propre[-1], levee), vitesse=vitesse_liaison,
                   commentaire="levee")

    robot.repli()
    log("{} : {} traits, {} points, echelle {:.3f} mm/unite"
        .format(os.path.basename(chemin), len(polylignes), points, echelle))
    return robot
