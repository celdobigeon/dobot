# -*- coding: utf-8 -*-
"""
Des polylignes au programme de trace.

La partie commune a tous les dessins, quelle que soit leur provenance :
un fichier SVG (`dobot_sim.svg`) ou une image (`dobot_sim.image`). Elle
met le dessin a l'echelle, le pose sur le plan du papier et en fait un
programme.

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
from typing import List, Optional, Sequence, Tuple

from .pilote import Pilote
from .robot import cible

__all__ = ["programme_depuis_polylignes", "programme_depuis_dessin", "boite"]

Point = Tuple[float, float]


def boite(polylignes: Sequence[Sequence[Point]]) -> Tuple[float, float, float, float]:
    """Rectangle englobant (xmin, ymin, xmax, ymax) de tous les points."""
    xs = [p[0] for poly in polylignes for p in poly]
    ys = [p[1] for poly in polylignes for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def programme_depuis_polylignes(polylignes: Sequence[Sequence[Point]],
                                repere=None,
                                largeur: float = 100.0,
                                hauteur: Optional[float] = None,
                                levee: float = 10.0,
                                profondeur: float = 0.0,
                                vitesse_trace: float = 60.0,
                                vitesse_liaison: float = 150.0,
                                lissage_trace: float = 60.0,
                                outil: str = "stylo",
                                nom: str = "dessin",
                                y_vers_le_bas: bool = True,
                                journal=None) -> Pilote:
    """
    Construit le programme qui trace une suite de polylignes.

    Le dessin est mis a l'echelle pour tenir dans `largeur` millimetres
    en conservant ses proportions, puis centre sur l'origine du repere.

    `y_vers_le_bas` vaut True pour les sources d'images -- SVG comme
    matrices de pixels comptent leurs lignes du haut vers le bas, quand
    l'axe y du repere monte. L'axe est alors retourne, et le trace
    ressemble au fichier vu a l'ecran.
    """
    log = journal or (lambda *_: None)
    if not polylignes:
        raise ValueError("aucune polyligne a tracer")

    x0, y0, x1, y1 = boite(polylignes)
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    echelle = largeur / dx
    if hauteur is not None:
        echelle = min(echelle, hauteur / dy)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sens = -1.0 if y_vers_le_bas else 1.0

    def vers_repere(p: Point) -> Point:
        return ((p[0] - cx) * echelle, sens * (p[1] - cy) * echelle)

    if repere is None:
        from .robot import repere_travail
        repere = repere_travail(x=210.0, y=0.0, z=0.0)

    robot = Pilote(nom=nom, outil=outil, vitesse=vitesse_liaison,
                   acceleration=600.0, vitesse_articulaire=60.0)

    def vise(p, z):
        return cible(x=p[0], y=p[1], z=z, repere=repere)

    points = traits = 0
    for poly in polylignes:
        pts = [vers_repere(p) for p in poly]
        # on ote les points confondus : ils ne tracent rien et alourdissent
        propre: List[Point] = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - propre[-1][0], p[1] - propre[-1][1]) > 1e-6:
                propre.append(p)
        if len(propre) < 2:
            continue
        points += len(propre)
        traits += 1

        # liaison en l'air, puis descente a l'ARRET AU POINT
        robot.movj(vise(propre[0], levee), lissage=70, commentaire="liaison")
        robot.movl(vise(propre[0], profondeur), vitesse=vitesse_trace,
                   commentaire="pose du stylo")
        for p in propre[1:-1]:
            robot.movl(vise(p, profondeur), vitesse=vitesse_trace,
                       lissage=lissage_trace)
        robot.movl(vise(propre[-1], profondeur), vitesse=vitesse_trace,
                   commentaire="fin du trait")
        robot.movl(vise(propre[-1], levee), vitesse=vitesse_liaison,
                   commentaire="levee")

    if not traits:
        raise ValueError("rien a tracer : toutes les polylignes sont "
                         "reduites a un point une fois mises a l'echelle")
    robot.repli()
    log("{} : {} traits, {} points, {:.0f} x {:.0f} mm, echelle {:.3f} "
        "mm/unite".format(nom, traits, points, dx * echelle, dy * echelle,
                          echelle))
    return robot


#: extensions reconnues comme des images matricielles
IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm",
          ".ppm")


def programme_depuis_dessin(chemin: str, **kw) -> Pilote:
    """
    Trace un dessin, qu'il soit vectoriel ou matriciel.

    L'extension decide : `.svg` passe par le lecteur de chemins, le reste
    par l'extraction de contours. Les parametres communs -- repere,
    largeur, levee, profondeur, tolerance, vitesses -- valent pour les
    deux ; ceux qui ne concernent qu'une source (seuil, inverser, flou,
    trous pour une image) sont simplement transmis.

    C'est la porte d'entree a preferer dans un script : changer de
    fichier ne demande alors pas de changer d'appel.
    """
    extension = os.path.splitext(chemin)[1].lower()
    if extension == ".svg":
        from .svg import programme_depuis_svg
        return programme_depuis_svg(chemin, **kw)
    if extension and extension not in IMAGES:
        raise ValueError(
            "extension inconnue : {}\n"
            "attendu : .svg, ou une image ({})".format(
                extension, ", ".join(IMAGES)))
    from .image import programme_depuis_image
    return programme_depuis_image(chemin, **kw)
