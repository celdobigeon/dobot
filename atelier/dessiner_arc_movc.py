#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine ce que `movc` demande, et dans quel ordre.

    python dessiner_arc_movc.py

Ecrit cellule/assets/arc_movc.svg, affiche par
cellule/ACTIVITE_PROFIL_STYLO.md.

TROIS POINTS FONT UN ARC, et deux seulement se declarent : le premier
est celui ou le robot se trouve DEJA. C'est la source de la confusion,
avec le centre du cercle -- qui n'entre nulle part dans l'appel, et
que la figure barre pour cette raison.

La figure montre donc, dans l'ordre : ce qu'on ne declare pas, le
premier argument, le second, et ce qu'on croit devoir declarer et
qu'on ne declare pas.
"""

import io
import os

import numpy as np

from dessin_svg import BLEU, ENCRE, GRIS, ROUGE, VERT, texte

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "arc_movc.svg")

ORANGE, SOURD = "#c8792a", "#9aa2ae"

#: le quart de cercle de la figure : rayon et position a l'ecran
CX, CY, R = 250.0, 330.0, 170.0


def sur_arc(deg):
    a = np.radians(deg)
    return CX + R * np.cos(a), CY - R * np.sin(a)


def pastille(x, y, numero, couleur, nom, dessous=""):
    """Un point numerote, avec son role ecrit a cote."""
    out = ['<circle cx="{:.1f}" cy="{:.1f}" r="7" fill="{}"/>'
           .format(x, y, couleur),
           '<circle cx="{:.1f}" cy="{:.1f}" r="13" fill="none" stroke="{}" '
           'stroke-width="1.6"/>'.format(x, y, couleur),
           texte(x, y - 20, numero, 15, couleur, "700", "middle")]
    out.append(texte(x + 20, y + 1, nom, 13, couleur, "700"))
    if dessous:
        out.append(texte(x + 20, y + 17, dessous, 12, "#666", "400"))
    return out


def main():
    larg, haut = 760.0, 430.0
    corps = []

    corps.append(texte(40, 42, "ce que <tspan font-style=\"italic\">movc</tspan>"
                       " demande, et dans quel ordre", 17, ENCRE, "600"))
    corps.append(texte(40, 64, "trois points font un arc &#183; deux seulement "
                       "se d&#233;clarent", 12, "#666"))

    # l'arc : du depart (180 deg) a la cible (90 deg), passage a 135
    depart, passage, cible = sur_arc(180), sur_arc(135), sur_arc(90)
    corps.append('<path d="M {:.1f} {:.1f} A {r:.1f} {r:.1f} 0 0 1 {:.1f} '
                 '{:.1f}" fill="none" stroke="{}" stroke-width="3.4" '
                 'stroke-linecap="round"/>'
                 .format(depart[0], depart[1], cible[0], cible[1],
                         ENCRE, r=R))

    # le cercle entier, en pointilles : l'arc n'en est qu'un morceau
    corps.append('<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="none" '
                 'stroke="{}" stroke-width="1" stroke-dasharray="4 5"/>'
                 .format(CX, CY, R, SOURD))

    # les rayons vers les trois points
    for p in (depart, passage, cible):
        corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                     'stroke="{}" stroke-width="1" stroke-dasharray="3 4"/>'
                     .format(CX, CY, p[0], p[1], SOURD))

    # LE CENTRE, barre : il n'entre nulle part dans l'appel
    corps.append('<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="none" '
                 'stroke="{}" stroke-width="1.6"/>'.format(CX, CY, SOURD))
    for dx, dy in ((1, 1), (1, -1)):
        corps.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                     'stroke="{}" stroke-width="1.6"/>'
                     .format(CX - 9 * dx, CY - 9 * dy, CX + 9 * dx,
                             CY + 9 * dy, SOURD))
    corps.append(texte(CX + 18, CY + 5, "le centre du cercle", 13, SOURD,
                       "700"))
    corps.append(texte(CX + 18, CY + 21, "ne se d&#233;clare PAS", 12, SOURD,
                       "400"))

    # les trois points
    corps += pastille(depart[0], depart[1], "1", SOURD,
                      "l&#224; o&#249; l&#8217;on est d&#233;j&#224;",
                      "la fin du mouvement pr&#233;c&#233;dent")
    corps += pastille(passage[0], passage[1], "2", ORANGE,
                      "le point de passage",
                      "1<tspan baseline-shift=\"super\" font-size=\"9\">er"
                      "</tspan> argument")
    corps += pastille(cible[0], cible[1], "3", ROUGE, "la cible",
                      "2<tspan baseline-shift=\"super\" font-size=\"9\">e"
                      "</tspan> argument")

    # l'appel, et les deux fleches qui relient
    y0 = haut - 56
    corps.append('<rect x="36" y="{:.0f}" width="470" height="44" rx="6" '
                 'fill="#f2efe6" stroke="{}" stroke-width="1"/>'
                 .format(y0 - 28, "#ddd8c8"))
    corps.append('<text x="56" y="{:.0f}" font-family="Cascadia Mono, '
                 'Consolas, monospace" font-size="15" fill="{}">'
                 'robot.movc('
                 '<tspan fill="{}" font-weight="700">point de passage</tspan>'
                 ', <tspan fill="{}" font-weight="700">cible</tspan>)'
                 '</text>'.format(y0, ENCRE, ORANGE, ROUGE))
    corps.append(texte(524, y0 - 6, "le point 1 n&#8217;y figure pas&#160;:",
                       12, "#666"))
    corps.append(texte(524, y0 + 10, "le robot y est d&#233;j&#224;.", 12,
                       "#666"))

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           'role="img" aria-label="Les trois points d un arc movc">\n'
           '<title>Ce que movc demande</title>\n'
           '<rect width="{w:.0f}" height="{h:.0f}" fill="#fdfcf7"/>\n'
           '{corps}\n</svg>\n'
           .format(w=larg, h=haut, corps="\n".join(corps)))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}".format(SORTIE))


if __name__ == "__main__":
    main()
