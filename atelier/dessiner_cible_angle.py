#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine d'ou viennent le cosinus et le sinus d'une cible orientee.

    python dessiner_cible_angle.py

Ecrit cellule/assets/cible_angle.svg, affiche par
cellule/ACTIVITE_PICK_AND_PLACE.md et
cellule/ACTIVITE_STRUCTURE_KAPLA.md.

CE QUE LA FIGURE REPOND. Un etudiant a qui on donne

    [[c,  s,  0, x],
     [s, -c,  0, y],
     [0,  0, -1, z],
     [0,  0,  0, 1]]

retient une recette. Avec le dessin il voit quatre choses :

  - la colonne 0 est le vecteur unitaire qui porte la LONGUEUR de la
    piece ; ses composantes sont les projections cos a et sin a, et
    rien d'autre ;
  - la colonne 1 n'est pas (-s, c) comme dans Rz(a) seul, parce que le
    demi-tour autour de x a retourne le y ;
  - la colonne 2 pointe vers le bas, pour la meme raison -- et cela,
    une vue de dessus ne peut pas le montrer, d'ou le panneau de cote ;
  - la derniere ligne, 0 0 0 1, qui est celle qu'on oublie d'ecrire.

Vue DE DESSUS, donc x a droite et y en haut -- et l'ecran compte ses
ordonnees vers le bas, d'ou les signes.
"""

import io
import os

import numpy as np

from dessin_svg import BLEU, ENCRE, GRIS, ROUGE, VERT, matrice_4x4, texte

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "cible_angle.svg")

#: l'angle de la figure. Quelconque : ni 0, ni 45, ni 90, pour qu'aucune
#: coincidence ne fasse croire a une regle.
ANGLE = 38.0
LONGUEUR, LARGEUR, EPAISSEUR = 125.0, 30.0, 10.0

BOIS, BOIS_TRAIT = "#d8b44a", "#5c470d"


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


def piece(cx, cy, a_deg, echelle):
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
    """Le cercle, l'angle, et les deux projections. Vue de DESSUS."""
    a = np.radians(ANGLE)
    c, s = np.cos(a), np.sin(a)
    out = []

    out.append(fleche(cx - 40, cy, cx + r + 56, cy, ROUGE, 2.2))
    out.append(fleche(cx, cy + 40, cx, cy - r - 56, VERT, 2.2))
    out.append(texte(cx + r + 64, cy + 5, "x", 15, ROUGE, "700"))
    out.append(texte(cx - 4, cy - r - 64, "y", 15, VERT, "700", "middle"))

    # le cercle unite -- unite, donc les projections SONT cos et sin
    out.append('<circle cx="{:.1f}" cy="{:.1f}" r="{:.1f}" fill="none" '
               'stroke="{}" stroke-width="1" stroke-dasharray="3 4"/>'
               .format(cx, cy, r, GRIS))

    out.append(piece(cx, cy, ANGLE, r / (LONGUEUR / 2) * 0.80))
    px, py = cx + r * c, cy - r * s
    out.append(fleche(cx, cy, px, py, ROUGE, 3.2))
    out.append(texte(px + 10, py - 10, "colonne 0", 13, ROUGE, "700"))
    out.append(texte(px + 10, py + 6, "(cos a , sin a)", 13, ROUGE, "400"))

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

    out.append(arc(cx, cy, 46, 0, ANGLE, ENCRE))
    am = np.radians(ANGLE / 2)
    out.append(texte(cx + 60 * np.cos(am), cy - 60 * np.sin(am) + 5, "a",
                     15, ENCRE, "700", "middle", italique=True))
    return out


def panneau_colonne1(cx, cy, r):
    """Pourquoi la colonne 1 vaut (sin a, -cos a) et non (-sin a, cos a)."""
    a = np.radians(ANGLE)
    c, s = np.cos(a), np.sin(a)
    out = []
    out.append(fleche(cx - 30, cy, cx + r + 34, cy, "#bbb", 1.6))
    out.append(fleche(cx, cy + 30, cx, cy - r - 34, "#bbb", 1.6))

    out.append(fleche(cx, cy, cx - r * s, cy - r * c, VERT, 2.2, "5 4"))
    out.append(texte(cx - r * s - 8, cy - r * c - 12,
                     "(&#8722;sin a , cos a)", 12, VERT, "400", "end"))
    out.append(texte(cx - r * s - 8, cy - r * c + 4, "Rz(a) seul", 12,
                     "#777", "400", "end"))

    out.append(fleche(cx, cy, cx + r * s, cy + r * c, VERT, 3.0))
    out.append(texte(cx + r * s + 8, cy + r * c + 4,
                     "(sin a , &#8722;cos a)", 12, VERT, "700"))
    out.append(texte(cx + r * s + 8, cy + r * c + 20,
                     "apr&#232;s Rx(180&#176;)", 12, "#777", "400"))

    out.append(fleche(cx, cy, cx + r * c, cy - r * s, ROUGE, 2.2))
    out.append(texte(cx + r * c + 8, cy - r * s - 6, "colonne 0", 12, ROUGE,
                     "700"))
    return out


def panneau_de_cote(cx, cy):
    """
    LA TROISIEME COLONNE, que la vue de dessus ne peut pas montrer.

    On regarde ici le long du y de la cible, si bien que son x va a
    droite et son z vers le bas. Le y sort alors de la feuille -- d'ou
    le point dans le cercle, et non la croix.
    """
    out = []
    demi, e = 60.0, 20.0                       # la piece, a l'ecran
    sol = cy + 26.0
    dessus = sol - e

    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="#9a9a9a" stroke-width="2"/>'
               .format(cx - demi - 36, sol, cx + demi + 36, sol))
    out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
               'fill="{}" fill-opacity="0.55" stroke="{}" stroke-width="1.2"/>'
               .format(cx - demi, dessus, 2 * demi, e, BOIS, BOIS_TRAIT))

    # la ventouse qui descend : elle vient de la-haut
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="#9a9a9a" stroke-width="1.4" stroke-dasharray="4 4"/>'
               .format(cx, dessus - 70, cx, dessus - 34))
    out.append('<rect x="{:.1f}" y="{:.1f}" width="16" height="14" rx="2" '
               'fill="#cfd6dd" stroke="{}" stroke-width="1"/>'
               .format(cx - 8, dessus - 34, GRIS))
    out.append('<ellipse cx="{:.1f}" cy="{:.1f}" rx="11" ry="5" '
               'fill="#b03a2e" stroke="{}" stroke-width="1"/>'
               .format(cx, dessus - 16, GRIS))

    # le repere de la cible, pose sur le DESSUS de la piece
    out.append(fleche(cx, dessus, cx + 72, dessus, ROUGE, 2.6))
    out.append(texte(cx + 80, dessus + 5, "x", 14, ROUGE, "700"))
    out.append(fleche(cx, dessus, cx, dessus + 60, BLEU, 2.6))
    out.append(texte(cx + 10, dessus + 52, "z : (0 , 0 , &#8722;1)", 13, BLEU,
                     "700"))
    out.append(texte(cx + 10, dessus + 68, "il descend vers la mati&#232;re",
                     11, "#777"))

    # le y, qui sort de la feuille
    out.append('<circle cx="{:.1f}" cy="{:.1f}" r="8" fill="none" '
               'stroke="{}" stroke-width="2"/>'
               .format(cx - 44, dessus - 14, VERT))
    out.append('<circle cx="{:.1f}" cy="{:.1f}" r="2.6" fill="{}"/>'
               .format(cx - 44, dessus - 14, VERT))
    out.append(texte(cx - 56, dessus - 10, "y", 14, VERT, "700", "end"))
    out.append(texte(cx - 56, dessus + 6, "vers vous", 11, "#777", "400",
                     "end"))
    return out


def main():
    larg, haut = 880.0, 710.0
    corps = []

    corps.append(texte(40, 42, "d&#8217;o&#249; viennent le cosinus et le "
                       "sinus", 16, ENCRE, "600"))
    corps.append(texte(40, 64, "le cercle est de rayon 1 : les projections "
                       "SONT cos a et sin a", 13, "#666"))

    corps.append(texte(240, 112, "vu de dessus", 14, ENCRE, "600", "middle"))
    corps += panneau_principal(240, 288, 132)

    corps.append(texte(664, 112, "la colonne 1 est retourn&#233;e", 14, ENCRE,
                       "600", "middle"))
    corps += panneau_colonne1(664, 198, 74)

    corps.append(texte(664, 330, "vu de c&#244;t&#233;", 14, ENCRE, "600",
                       "middle"))
    corps += panneau_de_cote(664, 392)

    corps += matrice_4x4(
        268, 512, "cible(a) =",
        [["c", "s", "0", "x"],
         ["s", "&#8722;c", "0", "y"],
         ["0", "0", "&#8722;1", "z"],
         ["0", "0", "0", "1"]],
        ["la longueur", "la largeur, retourn&#233;e", "l&#8217;approche",
         "o&#249; elle se pose"])

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
