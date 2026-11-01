#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine d'ou viennent le cosinus et le sinus d'une cible orientee.

    python dessiner_cible_angle.py

Ecrit cellule/assets/cible_angle.svg, affiche par
cellule/ACTIVITE_PICK_AND_PLACE.md et cellule/ACTIVITE_MUR_KAPLA.md.

CE QUE LA FIGURE REPOND. Un etudiant a qui on donne

    [[c,  s, 0, x],
     [s, -c, 0, y],
     [0,  0, -1, z]]

retient une recette. Avec le dessin il voit trois choses :

  - la colonne 0 est le vecteur unitaire qui porte la LONGUEUR de la
    piece ; ses composantes sont les projections cos a et sin a, et
    rien d'autre ;
  - la colonne 1 n'est pas (-s, c) comme dans Rz(a) seul, parce que le
    demi-tour autour de x a retourne le y ;
  - la colonne 2 pointe vers le bas, pour la meme raison.

Vue DE DESSUS, donc x a droite et y en haut -- et l'ecran compte ses
ordonnees vers le bas, d'ou le signe de `_v`.
"""

import io
import os

import numpy as np

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "cible_angle.svg")

#: l'angle de la figure. Quelconque : ni 0, ni 45, ni 90, pour qu'aucune
#: coincidence ne fasse croire a une regle.
ANGLE = 38.0
LONGUEUR, LARGEUR = 125.0, 30.0

ROUGE, VERT, BLEU = "#c0392b", "#2e9e4f", "#2b6cb0"
ENCRE, GRIS = "#333", "#8a8a8a"
BOIS, BOIS_TRAIT = "#d8b44a", "#5c470d"
POLICE = "IBM Plex Sans, Segoe UI, sans-serif"


def texte(u, v, t, taille=14, couleur=ENCRE, poids="400", ancre="start",
          style=""):
    return ('<text x="{:.1f}" y="{:.1f}" font-family="{}" font-size="{}" '
            'font-weight="{}" fill="{}" text-anchor="{}"{}>{}</text>'
            .format(u, v, POLICE, taille, poids, couleur, ancre,
                    ' font-style="italic"' if style else "", t))


def fleche(u0, v0, u1, v1, couleur, largeur=2.6, pointille=None):
    """Segment flechi, pointe pleine, dans le plan de l'image."""
    du, dv = u1 - u0, v1 - v0
    n = np.hypot(du, dv) or 1.0
    du, dv = du / n, dv / n
    px, py = -dv, du
    p = 10.0
    pts = "{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}".format(
        u1, v1,
        u1 - p * du + 0.36 * p * px, v1 - p * dv + 0.36 * p * py,
        u1 - p * du - 0.36 * p * px, v1 - p * dv - 0.36 * p * py)
    tirets = ' stroke-dasharray="{}"'.format(pointille) if pointille else ""
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="{}" stroke-linecap="round"{}/>\n'
            '<polygon points="{}" fill="{}"/>'
            .format(u0, v0, u1 - p * du * 0.7, v1 - p * dv * 0.7,
                    couleur, largeur, tirets, pts, couleur))


def arc(cx, cy, r, a0, a1, couleur, largeur=1.6):
    """Arc d'angle, de a0 a a1 (degres, sens trigonometrique)."""
    p0 = (cx + r * np.cos(np.radians(a0)), cy - r * np.sin(np.radians(a0)))
    p1 = (cx + r * np.cos(np.radians(a1)), cy - r * np.sin(np.radians(a1)))
    return ('<path d="M {:.1f} {:.1f} A {r:.1f} {r:.1f} 0 0 0 {:.1f} {:.1f}" '
            'fill="none" stroke="{}" stroke-width="{}"/>'
            .format(p0[0], p0[1], p1[0], p1[1], couleur, largeur, r=r))


def planche(cx, cy, a_deg, echelle):
    """La piece vue de dessus, centree, tournee de a."""
    a = np.radians(a_deg)
    u = np.array([np.cos(a), -np.sin(a)])      # x du repere, a l'ecran
    v = np.array([np.sin(a), np.cos(a)])       # la largeur, a l'ecran
    c = np.array([cx, cy])
    pts = [c + su * LONGUEUR / 2 * echelle * u + sv * LARGEUR / 2 * echelle * v
           for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return ('<polygon points="{}" fill="{}" fill-opacity="0.55" stroke="{}" '
            'stroke-width="1.2"/>'
            .format(" ".join("{:.1f},{:.1f}".format(*p) for p in pts),
                   BOIS, BOIS_TRAIT))


def panneau_principal(cx, cy, r):
    """Le cercle, l'angle, et les deux projections."""
    a = np.radians(ANGLE)
    c, s = np.cos(a), np.sin(a)
    out = []

    # les axes du repere de travail
    out.append(fleche(cx - 40, cy, cx + r + 56, cy, ROUGE, 2.2))
    out.append(fleche(cx, cy + 40, cx, cy - r - 56, VERT, 2.2))
    out.append(texte(cx + r + 64, cy + 5, "x", 15, ROUGE, "700"))
    out.append(texte(cx - 4, cy - r - 64, "y", 15, VERT, "700", "middle"))

    # le cercle unite -- unite, donc les projections SONT cos et sin
    out.append('<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="none" '
               'stroke="{}" stroke-width="1" stroke-dasharray="3 4"/>'
               .format(cx, cy, r, GRIS))

    # la piece, puis le vecteur qui la porte
    out.append(planche(cx, cy, ANGLE, r / (LONGUEUR / 2) * 0.80))
    px, py = cx + r * c, cy - r * s
    out.append(fleche(cx, cy, px, py, ROUGE, 3.2))
    out.append(texte(px + 10, py - 10, "colonne 0", 13, ROUGE, "700"))
    out.append(texte(px + 10, py + 6, "(cos a , sin a)", 13, ROUGE, "400"))

    # les projections, en pointilles
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="{}" stroke-width="1.2" stroke-dasharray="4 3"/>'
               .format(px, py, px, cy, GRIS))
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="{}" stroke-width="1.2" stroke-dasharray="4 3"/>'
               .format(px, py, cx, py, GRIS))
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="{}" stroke-width="4"/>'.format(cx, cy + 16, px,
                                                       cy + 16, ROUGE))
    out.append(texte((cx + px) / 2, cy + 34, "cos a", 13, ROUGE, "700",
                     "middle"))
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="{}" stroke-width="4"/>'.format(cx - 16, cy, cx - 16,
                                                       py, VERT))
    out.append(texte(cx - 24, (cy + py) / 2 + 4, "sin a", 13, VERT, "700",
                     "end"))

    # l'angle
    out.append(arc(cx, cy, 46, 0, ANGLE, ENCRE))
    am = np.radians(ANGLE / 2)
    out.append(texte(cx + 60 * np.cos(am), cy - 60 * np.sin(am) + 5, "a",
                     15, ENCRE, "700", "middle", style="i"))
    return out


def panneau_colonne1(cx, cy, r):
    """Pourquoi la colonne 1 vaut (sin a, -cos a) et non (-sin a, cos a)."""
    a = np.radians(ANGLE)
    c, s = np.cos(a), np.sin(a)
    out = []
    out.append(fleche(cx - 30, cy, cx + r + 34, cy, "#bbb", 1.6))
    out.append(fleche(cx, cy + 30, cx, cy - r - 34, "#bbb", 1.6))

    # le y de Rz(a) seul : perpendiculaire a gauche
    out.append(fleche(cx, cy, cx - r * s, cy - r * c, VERT, 2.2, "5 4"))
    out.append(texte(cx - r * s - 8, cy - r * c - 12,
                     "(&#8722;sin a , cos a)", 12, VERT, "400", "end"))
    out.append(texte(cx - r * s - 8, cy - r * c + 4, "Rz(a) seul", 12,
                     "#777", "400", "end"))

    # le y de la cible : retourne par le demi-tour
    out.append(fleche(cx, cy, cx + r * s, cy + r * c, VERT, 3.0))
    out.append(texte(cx + r * s + 8, cy + r * c + 4,
                     "(sin a , &#8722;cos a)", 12, VERT, "700"))
    out.append(texte(cx + r * s + 8, cy + r * c + 20, "apr&#232;s Rx(180&#176;)",
                     12, "#777", "400"))

    # la colonne 0, qui ne bouge pas
    out.append(fleche(cx, cy, cx + r * c, cy - r * s, ROUGE, 2.2))
    out.append(texte(cx + r * c + 8, cy - r * s - 6, "colonne 0", 12, ROUGE,
                     "700"))
    return out


def main():
    larg, haut = 880.0, 560.0
    corps = []

    corps.append(texte(40, 42, "d&#8217;o&#249; viennent le cosinus et le "
                       "sinus", 16, ENCRE, "600"))
    corps.append(texte(40, 64, "vue de dessus &#183; le cercle est de rayon 1, "
                       "donc les projections SONT cos a et sin a",
                       13, "#666"))

    corps += panneau_principal(250, 300, 140)

    corps.append(texte(640, 110, "la colonne 1 est retourn&#233;e", 15, ENCRE,
                       "600", "middle"))
    corps += panneau_colonne1(640, 250, 86)

    # la matrice, colonne par colonne
    y0 = 500
    corps.append(texte(40, y0 - 14, "cible(a) =", 15, ENCRE, "700"))
    colonnes = [("(c , s , 0)", ROUGE, "la longueur de la pi&#232;ce"),
                ("(s , &#8722;c , 0)", VERT, "sa largeur, retourn&#233;e"),
                ("(0 , 0 , &#8722;1)", BLEU, "l&#8217;approche, vers le bas"),
                ("(x , y , z)", ENCRE, "o&#249; elle se pose")]
    for i, (col, couleur, note) in enumerate(colonnes):
        x = 150 + i * 178
        corps.append('<rect x="{:.0f}" y="{:.0f}" width="164" height="30" '
                     'rx="5" fill="{}" fill-opacity="0.10" stroke="{}" '
                     'stroke-width="1"/>'.format(x, y0 - 22, couleur, couleur))
        corps.append(texte(x + 82, y0, col, 14, couleur, "700", "middle"))
        corps.append(texte(x + 82, y0 + 24, note, 12, "#555", "400", "middle"))

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           'role="img" aria-label="Le cosinus et le sinus d une cible '
           'orientee">\n'
           '<title>Le cosinus et le sinus d&#8217;une cible orient&#233;e'
           '</title>\n'
           '<rect width="{w:.0f}" height="{h:.0f}" fill="#fdfcf7"/>\n'
           '{corps}\n</svg>\n'
           .format(w=larg, h=haut, corps="\n".join(corps)))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}  (angle de la figure : {:.0f} deg)"
          .format(SORTIE, ANGLE))


if __name__ == "__main__":
    main()
