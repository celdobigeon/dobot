#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine le repere de la BASE et le repere de l'OUTIL cote a cote.

    python dessiner_reperes_outil.py

Ecrit cellule/assets/reperes_base_outil.svg, affiche par
cellule/ACTIVITE_PICK_AND_PLACE.md.

POURQUOI CETTE FIGURE. Parce que la matrice OUTIL s'ecrit en trois
secondes et se comprend en une heure. Ce qu'elle dit tient dans un
dessin : les deux reperes ont le MEME x, et leurs y et z sont
retournes -- c'est un demi-tour autour de x, suivi d'une translation
jusqu'a la pointe de la ventouse.

L'etudiant qui a la figure sous les yeux n'a plus a retenir la
matrice : il la relit.
"""

import io
import os

import numpy as np

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "reperes_base_outil.svg")

# -- projection isometrique, la meme que dessiner_structure_kapla ------------------
COS30, SIN30 = np.cos(np.pi / 6), np.sin(np.pi / 6)


def projeter(p):
    x, y, z = p
    return ((x - y) * COS30, (x + y) * SIN30 - z)


from dessin_svg import BLEU, ENCRE, ROUGE, VERT, matrice_4x4, texte

GRIS, TRAIT = "#d8d8d8", "#8a8a8a"


def axe(direction, longueur, couleur, nom, decal=(0, 0)):
    """Une fleche d'axe partant de l'origine, pointe pleine, nom au bout."""
    d = np.asarray(direction, dtype=float)
    u0, v0 = projeter((0.0, 0.0, 0.0))
    u1, v1 = projeter(d * longueur)
    du, dv = u1 - u0, v1 - v0
    n = np.hypot(du, dv) or 1.0
    du, dv = du / n, dv / n
    px, py = -dv, du                       # la normale, pour la base
    p = 9.0
    pointe = "{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}".format(
        u1, v1,
        u1 - p * du + 0.38 * p * px, v1 - p * dv + 0.38 * p * py,
        u1 - p * du - 0.38 * p * px, v1 - p * dv - 0.38 * p * py)
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="2.6" stroke-linecap="round"/>\n'
            '<polygon points="{}" fill="{}"/>\n{}'
            .format(u0, v0, u1 - p * du * 0.75, v1 - p * dv * 0.75, couleur,
                    pointe, couleur,
                    texte(u1 + decal[0], v1 + decal[1], nom, 15, couleur,
                          "700", "middle")))


def boite(centre, demi, remplissage=GRIS):
    """Un pave isometrique : dessus et les deux flancs vus."""
    c = np.asarray(centre, dtype=float)
    dx, dy, dz = demi
    s = [c + np.array([sx * dx, sy * dy, sz * dz])
         for sz in (1, -1) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    out = []
    for idx, ton in (((0, 1, 2, 3), 1.00), ((1, 5, 6, 2), 0.80),
                     ((2, 6, 7, 3), 0.62)):
        pts = " ".join("{:.1f},{:.1f}".format(*projeter(s[i])) for i in idx)
        out.append('<polygon points="{}" fill="{}" fill-opacity="{:.2f}" '
                   'stroke="{}" stroke-width="1"/>'
                   .format(pts, remplissage, ton, TRAIT))
    return out


def panneau_base():
    """Le socle du robot, et son repere : z vers le HAUT."""
    out = boite((0, 0, -24), (46, 46, 24))
    for d, couleur, nom, decal in (((1, 0, 0), ROUGE, "x", (6, 16)),
                                   ((0, 1, 0), VERT, "y", (-8, 16)),
                                   ((0, 0, 1), BLEU, "z", (0, -10))):
        out.append(axe(d, 78, couleur, nom, decal))
    return out


def panneau_outil(longueur_outil=92.0):
    """La bride et la ventouse, et le repere outil : z vers le BAS."""
    out = boite((0, 0, 0), (30, 30, 9), "#cfd6dd")
    u0, v0 = projeter((0, 0, -9))
    u1, v1 = projeter((0, 0, -longueur_outil))
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="#6b7280" stroke-width="5" stroke-linecap="round"/>'
               .format(u0, v0, u1, v1))
    out.append('<ellipse cx="{:.1f}" cy="{:.1f}" rx="14" ry="6" '
               'fill="#b03a2e" stroke="{}" stroke-width="1"/>'
               .format(u1, v1 + 4, TRAIT))
    # la cote L, en pointilles a cote de la tige
    out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
               'stroke="{}" stroke-width="1" stroke-dasharray="4 3"/>'
               .format(u0 + 52, v0, u1 + 52, v1, ENCRE))
    out.append(texte(u0 + 58, (v0 + v1) / 2 + 5, "L", 15, ENCRE, "700"))
    for d, couleur, nom, decal in (((1, 0, 0), ROUGE, "x", (6, 16)),
                                   ((0, -1, 0), VERT, "y", (8, 16)),
                                   ((0, 0, -1), BLEU, "z", (-24, 4))):
        out.append(axe(d, 78, couleur, nom, decal))
    return out


def main():
    larg, haut = 820.0, 470.0
    corps = []

    corps.append('<g transform="translate(180 195)">')
    corps += panneau_base()
    corps.append("</g>")
    corps.append(texte(180, 42, "rep&#232;re de la BASE", 16, ENCRE, "600",
                       "middle"))
    corps.append(texte(180, 63, "z vers le haut", 13, "#666", "400", "middle"))

    corps.append('<g transform="translate(610 150)">')
    corps += panneau_outil()
    corps.append("</g>")
    corps.append(texte(610, 42, "rep&#232;re de l'OUTIL", 16, ENCRE, "600",
                       "middle"))
    corps.append(texte(610, 63, "z vers le bas, y retourn&#233;", 13, "#666",
                       "400", "middle"))

    # ce qui passe de l'un a l'autre
    corps.append('<path d="M 300 150 q 100 -36 200 0" fill="none" '
                 'stroke="#7a6a2a" stroke-width="2.2" '
                 'marker-end="url(#pointe)"/>')
    corps.append(texte(400, 112, "Rx(180&#176;)", 16, "#7a6a2a", "700",
                       "middle"))
    corps.append(texte(400, 132, "puis translation de &#8722;L selon z",
                       12, "#7a6a2a", "400", "middle"))

    # LA MATRICE ENTIERE, et non quatre etiquettes : la ligne du bas
    # -- 0 0 0 1 -- est justement celle qu'on oublie d'ecrire.
    corps += matrice_4x4(
        252, 268, "OUTIL =",
        [["1", "0", "0", "0"],
         ["0", "&#8722;1", "0", "0"],
         ["0", "0", "&#8722;1", "&#8722;L"],
         ["0", "0", "0", "1"]],
        ["x : inchang&#233;", "y : retourn&#233;", "z : vers le bas",
         "la pointe"])

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           'role="img" aria-label="Repere de la base et repere de l outil">\n'
           '<title>Rep&#232;re de la base et rep&#232;re de l&#8217;outil'
           '</title>\n'
           '<defs><marker id="pointe" viewBox="0 0 10 10" refX="9" refY="5" '
           'markerWidth="7" markerHeight="7" orient="auto">'
           '<path d="M 0 0 L 10 5 L 0 10 z" fill="#7a6a2a"/></marker></defs>\n'
           '<rect width="{w:.0f}" height="{h:.0f}" fill="#fdfcf7"/>\n'
           '{corps}\n</svg>\n'
           .format(w=larg, h=haut, corps="\n".join(corps)))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}".format(SORTIE))


if __name__ == "__main__":
    main()
