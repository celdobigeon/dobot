#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine la chaine de reperes du convoyeur, vue de dessus.

    python dessiner_convoyeur.py

Ecrit cellule/assets/chaine_convoyeur.svg, affiche par
cellule/ACTIVITE_CONVOYEUR.md.

LA FIGURE EST CALCULEE, non dessinee : elle lit les matrices de
corrige_convoyeur.py. Changer une cote dans le corrige change le
dessin. Une figure faite a part derive du code des la premiere
modification, et une figure fausse dans un enonce coute plus cher
qu'une figure absente.

LE POINT DE VUE EST CELUI DE L'OPERATEUR, debout devant le robot et
regardant la cellule d'en haut :

    le +x du robot s'eloigne de lui        -> vers le HAUT de la feuille
    le +y du robot part a sa GAUCHE        -> vers la GAUCHE

C'est la seule disposition qui permette de montrer du doigt ce que la
figure nomme.
"""

import io
import os

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

import corrige_convoyeur as cv

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "chaine_convoyeur.svg")

ECHELLE = 0.92
ENCRE, GRIS, ACCENT = "#2a2a2a", "#8a8f98", "#e07a1f"
BANDE, BATI, CAPTEUR = "#4a4f58", "#c9ced6", "#e8482c"


def point(x, y):
    """Point du repere de la base -> point de l'image (u vers la droite)."""
    return (ECHELLE * (-y), ECHELLE * (-x))


def ligne(a, b, couleur, largeur=1.5, tirets=None):
    u0, v0 = point(*a)
    u1, v1 = point(*b)
    style = ' stroke-dasharray="{}"'.format(tirets) if tirets else ""
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="{}"{}/>'
            .format(u0, v0, u1, v1, couleur, largeur, style))


def texte(x, y, s, couleur=ENCRE, taille=13, ancre="middle", gras=False):
    u, v = point(x, y)
    return ('<text x="{:.1f}" y="{:.1f}" text-anchor="{}" '
            'font-family="IBM Plex Sans, Segoe UI, sans-serif" '
            'font-size="{}" font-weight="{}" fill="{}">{}</text>'
            .format(u, v, ancre, taille, "600" if gras else "400",
                    couleur, s))


def rectangle(T, longueur, largeur, remplissage, trait, opacite=1.0):
    """Une boite posee a plat, decrite par sa matrice et ses cotes."""
    u, v = T[:3, 0], T[:3, 1]
    c = T[:3, 3]
    coins = [c + su * longueur / 2 * u + sv * largeur / 2 * v
             for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    pts = " ".join("{:.1f},{:.1f}".format(*point(p[0], p[1])) for p in coins)
    return ('<polygon points="{}" fill="{}" fill-opacity="{:.2f}" '
            'stroke="{}" stroke-width="1.4"/>'
            .format(pts, remplissage, opacite, trait))


def repere(T, nom, taille=46, decal_nom=-26.0):
    """Un trident : l'origine d'un repere et ses axes x (rouge) et y."""
    o = T[:3, 3]
    ax = o + taille * T[:3, 0]
    ay = o + taille * 0.7 * T[:3, 1]
    return [ligne((o[0], o[1]), (ax[0], ax[1]), ACCENT, 2.4),
            ligne((o[0], o[1]), (ay[0], ay[1]), GRIS, 2.0),
            '<circle cx="{:.1f}" cy="{:.1f}" r="3.4" fill="{}"/>'
            .format(*point(o[0], o[1]), ENCRE),
            texte(ax[0], ax[1], "x", ACCENT, 12, gras=True),
            texte(ay[0], ay[1], "y", GRIS, 12),
            texte(o[0] + decal_nom, o[1], nom, ENCRE, 13, gras=True)]


def cote(a, b, etiquette, decal=0.0):
    """Une cote : un trait fin a double fleche, et sa valeur."""
    out = [ligne(a, b, GRIS, 1.0, tirets="4 3")]
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    out.append(texte(mx + decal, my, etiquette, GRIS, 12))
    return out


def main():
    C, K = cv.CONVOYEUR, cv.CAPTEUR
    P = cv.PRISE
    L, l = 698.0, 100.0
    dessin = []

    # -- la bande, le bati, et l'axe moteur ------------------------------
    #
    # le repere du tapis a son origine sur l'AXE MOTEUR : la bande part
    # de la vers les s positifs, donc son centre est a L/2
    centre_bande = C @ np.array([[1, 0, 0, L / 2.0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], dtype=float)
    dessin.append(rectangle(centre_bande, L, 120.0, BATI, "#aab0b8", 0.55))
    dessin.append(rectangle(centre_bande, L, l, BANDE, "#33373d", 0.22))

    o = C[:3, 3]
    bord = o + (l / 2.0 + 18.0) * C[:3, 1]
    bord2 = o - (l / 2.0 + 18.0) * C[:3, 1]
    dessin.append(ligne((bord[0], bord[1]), (bord2[0], bord2[1]),
                        "#333", 4.0))
    dessin.append(texte(o[0] - 108, o[1], "axe moteur", ENCRE, 13,
                        gras=True))

    # -- le capteur, et la piece arretee ---------------------------------
    Tc = C @ K
    dessin.append(rectangle(Tc, 14.0, l + 26.0, CAPTEUR, CAPTEUR, 0.30))
    dessin.append(texte(Tc[0, 3] - 78, Tc[1, 3], "capteur", CAPTEUR, 13,
                        gras=True))

    piece = np.eye(4)
    piece[:3, 3] = [P[0, 3], P[1, 3], 0.0]
    dessin.append(rectangle(piece, cv.COTE, cv.COTE, "#e0b84a", "#8a6b14"))

    # -- les trois reperes de la chaine ----------------------------------
    for item in repere(np.eye(4), "BASE"):
        dessin.append(item)
    for item in repere(C, "CONVOYEUR", decal_nom=-82.0):
        dessin.append(item)
    for item in repere(Tc, "CAPTEUR"):
        dessin.append(item)

    # -- les cotes qui font la chaine ------------------------------------
    dessin += cote((0, 0), (0, o[1]), "y = {:.0f}".format(o[1]), 0)
    dessin += cote((0, o[1]), (o[0], o[1]), "x = {:.0f}".format(o[0]), 0)
    dessin += cote((o[0], o[1]), (Tc[0, 3], Tc[1, 3]),
                   "capteur a {:.0f} mm".format(cv.CAPTEUR_S), -140)

    # -- cadre ------------------------------------------------------------
    xs = [0.0, o[0], Tc[0, 3], P[0, 3], centre_bande[0, 3] + 70]
    ys = [0.0, o[1] - 120, Tc[1, 3], P[1, 3], 320.0]
    coins = [point(x, y) for x in xs for y in ys]
    us = [u for u, _ in coins]
    vs = [v for _, v in coins]
    M = 54.0
    x0, y0 = min(us) - M, min(vs) - M
    larg, haut = max(us) - x0 + M, max(vs) - y0 + M + 34

    legende = (
        '<text x="{:.0f}" y="{:.0f}" font-family="IBM Plex Sans, Segoe UI, '
        'sans-serif" font-size="13" fill="#555">'
        'PRISE = CONVOYEUR &#215; CAPTEUR &#215; DANS_LE_CAPTEUR &#183; '
        'vue de dessus, l\'op&#233;rateur en bas &#183; '
        'cotes en millim&#232;tres</text>'
        .format(x0 + 10, y0 + haut - 12))

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="{:.1f} {:.1f} {:.1f} {:.1f}" width="{:.0f}" '
           'height="{:.0f}" role="img" '
           'aria-label="Chaine de reperes du convoyeur, vue de dessus">\n'
           '<title>Cha&#238;ne de rep&#232;res du convoyeur</title>\n'
           '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
           'fill="#fdfcf7"/>\n{}\n{}\n</svg>\n'
           .format(x0, y0, larg, haut, larg, haut, x0, y0, larg, haut,
                   "\n".join(dessin), legende))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}".format(SORTIE))
    print("capteur a ({:.0f}, {:.0f}), prise a ({:.0f}, {:.0f})"
          .format(Tc[0, 3], Tc[1, 3], P[0, 3], P[1, 3]))


if __name__ == "__main__":
    main()
