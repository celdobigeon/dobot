#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine le profil a tracer au stylo, cote.

    python dessiner_profil.py

Ecrit cellule/assets/profil_stylo.svg, affiche par
cellule/ACTIVITE_PROFIL_STYLO.md.

LA FIGURE EST CALCULEE DEPUIS LE CORRIGE : les memes constantes
tracent le dessin de l'enonce et le programme du robot. Changer
LARGEUR, HAUTEUR ou RAYON change les deux, et ils ne peuvent pas se
contredire.

Ce qu'elle montre, et qu'une cotation ordinaire ne montre pas : les
POINTS DE PASSAGE des quatre arcs, a 45 degres sur chaque conge. C'est
la seule donnee que `movc` reclame et qu'un plan ne porte jamais --
d'ou la croix, et la legende.
"""

import io
import os

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

import corrige_profil_stylo as profil
from dessin_svg import BLEU, ENCRE, GRIS, ROUGE, VERT, texte

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "profil_stylo.svg")

ECHELLE = 2.2                       # pixels par millimetre
TRAIT, COTE, PASSAGE = "#2d3139", "#8a8a8a", "#c8792a"


def u(x):
    """Abscisse du dessin -> abscisse de l'image."""
    return CENTRE[0] + ECHELLE * x


def v(y):
    """Ordonnee du dessin -> ordonnee de l'image (l'ecran compte vers le bas)."""
    return CENTRE[1] - ECHELLE * y


def contour():
    """Le rectangle a conges, en un seul chemin SVG."""
    DX, DY, R = profil.DX, profil.DY, profil.RAYON
    CX, CY = profil.CX, profil.CY
    r = ECHELLE * R
    d = ["M {:.1f} {:.1f}".format(u(-CX), v(-DY)),
         "L {:.1f} {:.1f}".format(u(CX), v(-DY)),
         "A {r:.1f} {r:.1f} 0 0 0 {:.1f} {:.1f}".format(u(DX), v(-CY), r=r),
         "L {:.1f} {:.1f}".format(u(DX), v(CY)),
         "A {r:.1f} {r:.1f} 0 0 0 {:.1f} {:.1f}".format(u(CX), v(DY), r=r),
         "L {:.1f} {:.1f}".format(u(-CX), v(DY)),
         "A {r:.1f} {r:.1f} 0 0 0 {:.1f} {:.1f}".format(u(-DX), v(CY), r=r),
         "L {:.1f} {:.1f}".format(u(-DX), v(-CY)),
         "A {r:.1f} {r:.1f} 0 0 0 {:.1f} {:.1f}".format(u(-CX), v(-DY), r=r),
         "Z"]
    return ('<path d="{}" fill="#f0ece0" fill-opacity="0.5" stroke="{}" '
            'stroke-width="2.4" stroke-linejoin="round"/>'.format(" ".join(d),
                                                                  TRAIT))


def cote_horizontale(y, x0, x1, etiquette, decal=0.0):
    """Une cote avec ses deux fleches et son texte au milieu."""
    vy = v(y) + decal
    out = ['<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" '
           'stroke-width="1" marker-start="url(#f1)" marker-end="url(#f2)"/>'
           .format(u(x0), vy, u(x1), vy, COTE)]
    for x in (x0, x1):
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                   'stroke="{}" stroke-width="0.8" stroke-dasharray="3 3"/>'
                   .format(u(x), v(y), u(x), vy, COTE))
    out.append(texte((u(x0) + u(x1)) / 2, vy - 7, etiquette, 13, ENCRE, "700",
                     "middle"))
    return out


def cote_verticale(x, y0, y1, etiquette, decal=0.0):
    ux = u(x) + decal
    out = ['<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" stroke="{}" '
           'stroke-width="1" marker-start="url(#f1)" marker-end="url(#f2)"/>'
           .format(ux, v(y0), ux, v(y1), COTE)]
    for y in (y0, y1):
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                   'stroke="{}" stroke-width="0.8" stroke-dasharray="3 3"/>'
                   .format(u(x), v(y), ux, v(y), COTE))
    out.append(texte(ux + 8, (v(y0) + v(y1)) / 2 + 4, etiquette, 13, ENCRE,
                     "700"))
    return out


def main():
    global CENTRE
    larg, haut = 760.0, 470.0
    CENTRE = (larg / 2.0 - 20.0, 252.0)
    corps = []

    corps.append(texte(40, 42, "le profil &#224; tracer", 17, ENCRE, "600"))
    corps.append(texte(40, 64, "cot&#233; dans le rep&#232;re de la FEUILLE, "
                       "origine au centre &#183; toutes les cotes en "
                       "millim&#232;tres", 12, "#666"))

    corps.append(contour())

    # le percage
    r = profil.DIAMETRE_PERCAGE / 2.0
    corps.append('<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="none" '
                 'stroke="{}" stroke-width="2.4"/>'
                 .format(u(0), v(0), ECHELLE * r, TRAIT))

    # les axes de la feuille
    corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="1.6"/>'
                 .format(u(-profil.DX - 14), v(0), u(profil.DX + 20), v(0),
                         ROUGE))
    corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="1.6"/>'
                 .format(u(0), v(-profil.DY - 14), u(0), v(profil.DY + 20),
                         VERT))
    corps.append(texte(u(profil.DX + 26), v(0) + 5, "x", 14, ROUGE, "700"))
    corps.append(texte(u(0), v(profil.DY + 28), "y", 14, VERT, "700",
                       "middle"))

    # les quatre points de passage, a 45 degres sur les conges
    B = profil.BIAIS
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = sx * (profil.CX + B), sy * (profil.CY + B)
            for a, b in ((-5, -5, ), (-5, 5)):
                corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" '
                             'y2="{:.1f}" stroke="{}" stroke-width="2.2"/>'
                             .format(u(px) + a, v(py) + b, u(px) - a,
                                     v(py) - b, PASSAGE))
    corps.append(texte(u(profil.CX + B) + 12, v(-profil.CY - B) + 18,
                       "point de passage du movc", 12, PASSAGE, "700"))
    corps.append(texte(u(profil.CX + B) + 12, v(-profil.CY - B) + 34,
                       "(&#224; 45&#176; sur le cong&#233;)", 12, PASSAGE,
                       "400"))

    # les cotes
    corps += cote_horizontale(-profil.DY, -profil.DX, profil.DX,
                              "{:.0f}".format(profil.LARGEUR), 46)
    corps += cote_verticale(-profil.DX, -profil.DY, profil.DY,
                            "{:.0f}".format(profil.HAUTEUR), -40)

    # le rayon d'un conge
    cx, cy = -profil.CX, profil.CY
    ang = np.radians(135.0)
    corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="1"/>'
                 .format(u(cx), v(cy), u(cx + profil.RAYON * np.cos(ang)),
                         v(cy + profil.RAYON * np.sin(ang)), COTE))
    corps.append('<circle cx="{:.1f}" cy="{:.1f}" r="2" fill="{}"/>'
                 .format(u(cx), v(cy), COTE))
    corps.append(texte(u(cx) - 6, v(cy) - 8, "R{:.0f}".format(profil.RAYON),
                       13, ENCRE, "700", "end"))

    # le diametre du percage
    corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="1"/>'
                 .format(u(-r * 0.71), v(-r * 0.71), u(r * 0.71), v(r * 0.71),
                         COTE))
    corps.append(texte(u(r * 0.71) + 8, v(r * 0.71) - 6,
                       "&#216;{:.0f}".format(profil.DIAMETRE_PERCAGE), 13,
                       ENCRE, "700"))

    corps.append(texte(40, haut - 24, "quatre droites, quatre arcs, et un "
                       "cercle fait de DEUX demi-arcs &#8212; "
                       "{} instructions en tout"
                       .format(len(profil.robot.programme.instructions)),
                       12, "#666"))

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           'role="img" aria-label="Le profil a tracer, cote">\n'
           '<title>Le profil &#224; tracer</title>\n'
           '<defs>\n'
           '<marker id="f1" viewBox="0 0 10 10" refX="1" refY="5" '
           'markerWidth="6" markerHeight="6" orient="auto">'
           '<path d="M 10 0 L 0 5 L 10 10 z" fill="{g}"/></marker>\n'
           '<marker id="f2" viewBox="0 0 10 10" refX="9" refY="5" '
           'markerWidth="6" markerHeight="6" orient="auto">'
           '<path d="M 0 0 L 10 5 L 0 10 z" fill="{g}"/></marker>\n'
           '</defs>\n'
           '<rect width="{w:.0f}" height="{h:.0f}" fill="#fdfcf7"/>\n'
           '{corps}\n</svg>\n'
           .format(w=larg, h=haut, g=COTE, corps="\n".join(corps)))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}".format(SORTIE))
    print("{:.0f} x {:.0f} mm, conges R{:.0f}, percage {:.0f}"
          .format(profil.LARGEUR, profil.HAUTEUR, profil.RAYON,
                  profil.DIAMETRE_PERCAGE))


if __name__ == "__main__":
    main()
