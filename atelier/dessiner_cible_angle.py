#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D'ou vient la matrice d'une cible orientee, en deux temps.

    python dessiner_cible_angle.py

Ecrit cellule/assets/cible_angle.svg, affiche par
cellule/ACTIVITE_PICK_AND_PLACE.md et
cellule/ACTIVITE_STRUCTURE_KAPLA.md.

LA FIGURE SUIT LA PRESENTATION DU COURS : un repere fixe, un repere
tourne par-dessus, les projections en pointilles, et la matrice a cote
avec ses LIGNES et ses COLONNES NOMMEES. Les colonnes sont les axes du
repere tourne, les lignes les composantes du repere fixe -- lire la
case (Yb, Yo) suffit alors a voir le retournement.

Deux temps, parce que la cible est un PRODUIT :

    cible = Rz(a) . Rx(180 deg)

  - Rz(a) tourne autour de la verticale. C'est la matrice du cours,
    celle qu'on reconnait : (cos, -sin / sin, cos).
  - Rx(180) retourne ensuite y et z, parce que l'outil approche PAR LE
    DESSUS. C'est de la que vient le signe qui surprend : la deuxieme
    colonne devient (sin, -cos) et la troisieme (0, 0, -1).

L'ancienne version montrait trois petits schemas cote a cote ; c'etait
confus. Un schema par etape, et la matiere dans la matrice.
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

#: les axes du repere FIXE, en trait fin ; ceux du repere TOURNE, en gras
PALE = "#b9bec6"
POINTILLE = "#c8792a"


def fleche(u0, v0, u1, v1, couleur, largeur=2.4, pointille=None, tete=9.0):
    """Segment flechi, pointe pleine, dans le plan de l'image."""
    du, dv = u1 - u0, v1 - v0
    n = np.hypot(du, dv) or 1.0
    du, dv = du / n, dv / n
    px, py = -dv, du
    pts = "{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}".format(
        u1, v1,
        u1 - tete * du + 0.36 * tete * px, v1 - tete * dv + 0.36 * tete * py,
        u1 - tete * du - 0.36 * tete * px, v1 - tete * dv - 0.36 * tete * py)
    tirets = ' stroke-dasharray="{}"'.format(pointille) if pointille else ""
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="{}" stroke-linecap="round"{}/>\n'
            '<polygon points="{}" fill="{}"/>'
            .format(u0, v0, u1 - tete * du * 0.7, v1 - tete * dv * 0.7,
                    couleur, largeur, tirets, pts, couleur))


def arc(cx, cy, r, a0, a1, couleur, largeur=1.5):
    """Arc d'angle, de a0 a a1 (degres, sens trigonometrique)."""
    p0 = (cx + r * np.cos(np.radians(a0)), cy - r * np.sin(np.radians(a0)))
    p1 = (cx + r * np.cos(np.radians(a1)), cy - r * np.sin(np.radians(a1)))
    return ('<path d="M {:.1f} {:.1f} A {r:.1f} {r:.1f} 0 0 0 {:.1f} {:.1f}" '
            'fill="none" stroke="{}" stroke-width="{}"/>'
            .format(p0[0], p0[1], p1[0], p1[1], couleur, largeur, r=r))


def tirets(u0, v0, u1, v1, couleur=POINTILLE):
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="1.4" stroke-dasharray="5 4"/>'
            .format(u0, v0, u1, v1, couleur))


def axe_hors_plan(cx, cy, couleur, sortant, nom):
    """
    Le troisieme axe, perpendiculaire a la feuille.

    Point dans le cercle : il vient VERS le lecteur. Croix : il s'en
    va. C'est la convention du cours, et elle se verifie : avec x a
    droite et y en haut, z sort ; apres le demi-tour autour de x, il
    rentre.
    """
    out = ['<circle cx="{:.1f}" cy="{:.1f}" r="9" fill="{}" '
           'fill-opacity="0.08" stroke="{}" stroke-width="2"/>'
           .format(cx, cy, couleur, couleur)]
    if sortant:
        out.append('<circle cx="{:.1f}" cy="{:.1f}" r="2.8" fill="{}"/>'
                   .format(cx, cy, couleur))
    else:
        for dx, dy in ((1, 1), (1, -1)):
            out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                       'stroke="{}" stroke-width="2"/>'
                       .format(cx - 5 * dx, cy - 5 * dy, cx + 5 * dx,
                               cy + 5 * dy, couleur))
    out.append(texte(cx - 14, cy + 5, nom, 13, couleur, "700", "end"))
    return out


def panneau(cx, cy, r, retourne, noms, projections):
    """
    Un repere fixe en trait pale, un repere tourne par-dessus.

    `retourne` : False pour Rz(a) seule, True pour la cible, dont le y
    et le z ont subi le demi-tour autour de x.
    `noms` : (nom du 1er axe tourne, du 2e, du 3e).
    """
    a = np.radians(ANGLE)
    c, s = np.cos(a), np.sin(a)
    out = []

    # Le repere FIXE, en trait pale : x a droite, y en haut.
    #
    # Il ne depasse que de 12 px. Il en depassait 34, et la pointe du y
    # montait dans le titre du panneau -- une fleche qui traverse une
    # ligne de texte, c'est le lecteur qui trebuche.
    out.append(fleche(cx, cy, cx + r + 12, cy, PALE, 1.8, tete=8))
    out.append(fleche(cx, cy, cx, cy - r - 12, PALE, 1.8, tete=8))
    out.append(texte(cx + r + 20, cy + 5, "x", 13, "#8d939c", "700"))
    out.append(texte(cx - 2, cy - r - 20, "y", 13, "#8d939c", "700", "middle"))

    # le repere TOURNE
    u = (cx + r * c, cy - r * s)
    out.append(fleche(cx, cy, u[0], u[1], ROUGE, 2.8))
    out.append(texte(u[0] + 10, u[1] - 8, noms[0], 14, ROUGE, "700"))

    signe = -1.0 if retourne else 1.0          # Rx(180) retourne le y
    if retourne:
        # LE REPERE INTERMEDIAIRE, celui que Rz(a) a laisse. C'est
        # autour de SON x -- autour de u -- que le demi-tour se fait,
        # et c'est pour cela que v bascule de l'autre cote. Le montrer
        # en pointilles repond a la seule question que la figure
        # posait sans y repondre : pourquoi (sin a, -cos a) ?
        vi = (cx - r * s, cy - r * c)
        out.append(fleche(cx, cy, vi[0], vi[1], "#9fd3ae", 2.0, "5 4"))
        out.append(texte(vi[0] - 8, vi[1] - 6, "v", 13, "#6aa87f", "700",
                         "end"))
        out.append(texte(vi[0] - 8, vi[1] + 10, "du panneau 1", 11, "#8aa894",
                         "400", "end"))
    v = (cx - signe * r * s, cy - signe * r * c)
    out.append(fleche(cx, cy, v[0], v[1], VERT, 2.8))
    out.append(texte(v[0] + (10 if retourne else -10),
                     v[1] + (16 if retourne else -8), noms[1], 14, VERT,
                     "700", "start" if retourne else "end"))

    out += axe_hors_plan(cx, cy, BLEU, not retourne, noms[2])

    out.append(arc(cx, cy, 40, 0, ANGLE, ENCRE))
    am = np.radians(ANGLE / 2)
    out.append(texte(cx + 54 * np.cos(am), cy - 54 * np.sin(am) + 5, "a",
                     14, ENCRE, "700", "middle", italique=True))

    if projections:
        # d'ou viennent le cosinus et le sinus : les projections du
        # premier axe tourne sur les deux axes fixes. Le repere est
        # UNITAIRE, donc elles valent cos a et sin a, sans facteur.
        out.append(tirets(u[0], u[1], u[0], cy))
        out.append(tirets(u[0], u[1], cx, u[1]))
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                   'stroke="{}" stroke-width="3.5"/>'
                   .format(cx, cy, u[0], cy, POINTILLE))
        out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                   'stroke="{}" stroke-width="3.5"/>'
                   .format(cx, cy, cx, u[1], POINTILLE))
        out.append(texte((cx + u[0]) / 2, cy + 20, "cos a", 12, POINTILLE,
                         "700", "middle"))
        out.append(texte(cx - 10, (cy + u[1]) / 2 + 4, "sin a", 12, POINTILLE,
                         "700", "end"))
    return out


def main():
    larg, haut = 760.0, 930.0
    corps = []

    corps.append(texte(40, 42, "la cible, en deux temps", 17, ENCRE, "600"))
    corps.append(texte(40, 64, "cible = Rz(a) &#183; Rx(180&#176;) &#183; "
                       "les colonnes sont les axes tourn&#233;s, les lignes "
                       "le rep&#232;re fixe", 12, "#666"))

    # -- 1. la rotation du cours ------------------------------------------
    corps.append(texte(40, 110, "1. Rz(a) : la rotation autour de la "
                       "verticale", 14, ENCRE, "700"))
    corps += panneau(180, 272, 84, False, ("u", "v", "w"), True)
    corps += matrice_4x4(
        428, 224, "Rz(a) =",
        [["cos a", "&#8722;sin a", "0"],
         ["sin a", "cos a", "0"],
         ["0", "0", "1"]],
        largeur_colonne=86, hauteur_ligne=32,
        titres=["u", "v", "w"], cotes=["x", "y", "z"])

    # -- 2. le demi-tour, et la cible --------------------------------------
    corps.append(texte(40, 412, "2. puis Rx(180&#176;), AUTOUR DE u : "
                       "l&#8217;outil approche par le dessus", 14, ENCRE,
                       "700"))
    corps.append(texte(40, 432, "multiplier &#192; DROITE, c&#8217;est "
                       "tourner dans le rep&#232;re INTERM&#201;DIAIRE "
                       "(u, v, w) : l&#8217;axe du demi-tour reste, les deux "
                       "autres se retournent", 12, "#666"))
    corps += panneau(180, 572, 84, True,
                     ("Xo = u", "Yo = &#8722;v", "Zo = &#8722;w"), False)
    corps += matrice_4x4(
        428, 524, "cible =",
        [["cos a", "sin a", "0"],
         ["sin a", "&#8722;cos a", "0"],
         ["0", "0", "&#8722;1"]],
        largeur_colonne=86, hauteur_ligne=32,
        titres=["Xo", "Yo", "Zo"], cotes=["Xb", "Yb", "Zb"])

    # -- 3. et en homogene --------------------------------------------------
    corps.append(texte(40, 722, "3. en matrice homog&#232;ne : la position "
                       "s&#8217;ajoute en quatri&#232;me colonne", 14, ENCRE,
                       "700"))
    corps += matrice_4x4(
        250, 762, "",
        [["cos a", "sin a", "0", "x"],
         ["sin a", "&#8722;cos a", "0", "y"],
         ["0", "0", "&#8722;1", "z"],
         ["0", "0", "0", "1"]],
        largeur_colonne=82, hauteur_ligne=30,
        titres=["Xo", "Yo", "Zo", "Oo"], cotes=["Xb", "Yb", "Zb", ""],
        notes=["la longueur", "la largeur", "l&#8217;approche",
               "o&#249; elle se pose"])

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           'role="img" aria-label="La matrice d une cible orientee, en deux '
           'temps">\n'
           '<title>La matrice d&#8217;une cible orient&#233;e</title>\n'
           '<rect width="{w:.0f}" height="{h:.0f}" fill="#fdfcf7"/>\n'
           '{corps}\n</svg>\n'
           .format(w=larg, h=haut, corps="\n".join(corps)))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}  (angle de la figure : {:.0f} deg)"
          .format(SORTIE, ANGLE))


if __name__ == "__main__":
    main()
