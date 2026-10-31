#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dessine le mur de kapla de l'activite, en vue isometrique.

    python dessiner_mur_kapla.py

Ecrit cellule/assets/mur_kapla.svg, affiche par
cellule/ACTIVITE_MUR_KAPLA.md.

POURQUOI UN DESSIN CALCULE, ET NON DESSINE A LA MAIN. Parce qu'une
figure faite a part derive du code des la premiere modification, et
qu'une figure fausse dans un enonce coute plus cher qu'une figure
absente. Ici, les douze planches sont placees par la MEME fonction
`depose` que le programme du robot : changer l'angle dans le corrige
change le dessin.
"""

import io
import os

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

import corrige_mur_kapla as mur

SORTIE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule", "assets", "mur_kapla.svg")

# -- projection isometrique --------------------------------------------------
#
# L'oeil est en haut, a droite et devant. Une cote x ou y s'en va en
# diagonale, une cote z monte tout droit.
COS30, SIN30 = np.cos(np.pi / 6), np.sin(np.pi / 6)
ECHELLE = 1.35


def projeter(p):
    """Point (x, y, z) du chantier -> point (u, v) de l'image."""
    x, y, z = p
    return (ECHELLE * (x - y) * COS30,
            ECHELLE * ((x + y) * SIN30 - z))


#: direction de l'oeil, normalisee. Une face n'est peinte que si sa
#: normale sortante pointe vers lui.
OEIL = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def boite(centre, angle_deg, longueur, largeur, epaisseur):
    """Les huit sommets d'une planche posee a plat, tournee de `angle`."""
    a = np.radians(angle_deg)
    u = np.array([np.cos(a), np.sin(a), 0.0])        # la longueur
    v = np.array([-np.sin(a), np.cos(a), 0.0])       # la largeur
    w = np.array([0.0, 0.0, 1.0])
    c = np.asarray(centre, dtype=float)
    return [c + su * longueur / 2 * u + sv * largeur / 2 * v
            + sw * epaisseur * w
            for sw in (0.0, -1.0)                    # dessus, puis dessous
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))], (u, v, w)


def faces_visibles(s, repere):
    """
    Les faces tournees vers l'oeil, du fond vers l'avant.

    ON NE PEUT PAS CHOISIR LES FACES A L'AVANCE. Une planche a 15
    degres et une planche a 165 ne montrent pas les memes cotes : coder
    en dur "le dessus et deux lateraux" dessine des boites retournees,
    et le mur ressemble alors a un tas. On calcule donc la NORMALE
    SORTANTE de chacune des six faces, et on ne garde que celles qui
    regardent vers l'oeil.
    """
    u, v, w = repere
    #   s[0..3] : le dessus       s[4..7] : le dessous
    faces = [((0, 1, 2, 3), w),          # dessus
             ((7, 6, 5, 4), -w),         # dessous
             ((0, 1, 5, 4), -v),
             ((1, 2, 6, 5), u),
             ((2, 3, 7, 6), v),
             ((3, 0, 4, 7), -u)]
    vues = [(idx, n) for idx, n in faces if float(np.dot(n, OEIL)) > 1e-9]
    # du fond vers l'avant : le centre de la face le plus eloigne d'abord
    vues.sort(key=lambda f: float(np.dot(
        np.mean([s[i] for i in f[0]], axis=0), OEIL)))
    return [([s[i] for i in idx], n) for idx, n in vues]


def polygone(sommets, remplissage, trait):
    pts = " ".join("{:.1f},{:.1f}".format(*projeter(p)) for p in sommets)
    return ('<polygon points="{}" fill="{}" stroke="{}" '
            'stroke-width="1" stroke-linejoin="round"/>'
            .format(pts, remplissage, trait))


# -- teintes : une par assise, pour que la vrille se lise ---------------------
TEINTES = ["#c9a227", "#d8b44a", "#b8912a", "#e0c268", "#a8821f", "#edd392"]
COTE_SOMBRE = "#8a6b14"
TRAIT = "#5c470d"


def panneau_dessus(pieces, x0, y0):
    """
    VUE DE DESSUS. C'est elle qui montre la vrille.

    L'isometrie seule trompe : six assises de 10 mm font 60 mm de haut
    pour 125 de large, donc un objet plat, ou la rotation se devine mal.
    Vue de dessus, chaque assise est un trait net et l'angle se lit.
    """
    out = []
    for assise, centre, angle in pieces:
        a = np.radians(angle)
        u = np.array([np.cos(a), np.sin(a)])
        v = np.array([-np.sin(a), np.cos(a)])
        c = np.array([centre[0], centre[1]])
        coins = [c + su * mur.LONGUEUR / 2 * u + sv * mur.LARGEUR / 2 * v
                 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pts = " ".join("{:.1f},{:.1f}".format(x0 + p[0], y0 - p[1])
                       for p in coins)
        # les assises basses s'effacent, la derniere reste franche
        opacite = 0.25 + 0.75 * (assise + 1) / mur.ASSISES
        out.append('<polygon points="{}" fill="{}" fill-opacity="{:.2f}" '
                   'stroke="{}" stroke-width="1"/>'
                   .format(pts, TEINTES[assise % len(TEINTES)], opacite,
                           TRAIT))
    return out


def main():
    pieces = []
    for assise in range(mur.ASSISES):
        for planche in (0, 1):
            T = mur.depose(assise, planche)
            # la cible designe le DESSUS de la planche, dans le repere
            # de la base ; on repasse dans celui du chantier pour
            # dessiner le mur centre
            centre = T[:3, 3] - mur.CHANTIER[:3, 3]
            pieces.append((assise, centre, mur.angle_outil(assise)))

    # peintre : du plus loin au plus proche. L'oeil etant en (+x, +y, +z),
    # c'est la projection sur sa direction qui ordonne.
    pieces.sort(key=lambda p: float(np.dot(p[1], OEIL)))

    # -- panneau de droite : l'isometrie ---------------------------------
    corps = []
    for assise, centre, angle in pieces:
        s, rep = boite(centre, angle, mur.LONGUEUR, mur.LARGEUR,
                       mur.EPAISSEUR)
        clair = TEINTES[assise % len(TEINTES)]
        for sommets, normale in faces_visibles(s, rep):
            dessus = float(np.dot(normale, (0.0, 0.0, 1.0))) > 0.5
            corps.append(polygone(sommets, clair if dessus else COTE_SOMBRE,
                                  TRAIT))

    tous = [projeter(p)
            for _, c, a in pieces
            for p in boite(c, a, mur.LONGUEUR, mur.LARGEUR,
                           mur.EPAISSEUR)[0]]
    iso_u = [u for u, _ in tous]
    iso_v = [v for _, v in tous]
    iso_larg = max(iso_u) - min(iso_u)
    iso_haut = max(iso_v) - min(iso_v)

    # -- panneau de gauche : la vue de dessus ----------------------------
    demi = mur.LONGUEUR / 2 + 10
    plan_larg = plan_haut = 2 * demi

    MARGE, ENTRE = 30.0, 60.0
    # la legende court sous les deux panneaux : si elle est plus large
    # qu'eux, c'est elle qui fixe la toile. Sans ca, elle se fait couper
    # a droite -- constate.
    larg = max(MARGE + plan_larg + ENTRE + iso_larg + MARGE, 790.0)
    haut = MARGE + 26 + max(plan_haut, iso_haut) + 34

    plan = panneau_dessus(pieces, MARGE + demi, MARGE + 26 + demi)
    decal_u = MARGE + plan_larg + ENTRE - min(iso_u)
    decal_v = MARGE + 26 - min(iso_v)

    def titre(x, t):
        return ('<text x="{:.0f}" y="{:.0f}" font-family="IBM Plex Sans, '
                'Segoe UI, sans-serif" font-size="15" font-weight="600" '
                'fill="#333">{}</text>'.format(x, MARGE + 10, t))

    legende = (
        '<text x="{x:.0f}" y="{y:.0f}" font-family="IBM Plex Sans, Segoe UI, '
        'sans-serif" font-size="14" fill="#555">'
        'planche {L:.0f} &#215; {l:.0f} &#215; {e:.0f} mm &#183; '
        '{n} assises &#183; {a:.0f}&#176; par assise &#183; '
        'd&#233;calage {d:.1f} mm &#183; '
        'emprise {E:.0f} &#215; {E:.0f} mm</text>'
        .format(x=MARGE, y=haut - 12, L=mur.LONGUEUR, l=mur.LARGEUR,
                e=mur.EPAISSEUR, n=mur.ASSISES, a=mur.ANGLE,
                d=mur.DECALAGE, E=mur.LONGUEUR))

    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
           'role="img" aria-label="Mur de kapla vrille : vue de dessus et '
           'vue isometrique">\n'
           '<title>Mur de kapla vrill&#233;</title>\n'
           '<rect width="{w:.0f}" height="{h:.0f}" fill="#fdfcf7"/>\n'
           '{t1}\n{t2}\n{plan}\n'
           '<g transform="translate({du:.1f} {dv:.1f})">\n{iso}\n</g>\n'
           '{leg}\n</svg>\n'
           .format(w=larg, h=haut,
                   t1=titre(MARGE, "vue de dessus"),
                   t2=titre(MARGE + plan_larg + ENTRE, "vue isom&#233;trique"),
                   plan="\n".join(plan), du=decal_u, dv=decal_v,
                   iso="\n".join(corps), leg=legende))

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    io.open(SORTIE, "w", encoding="utf-8").write(svg)
    print("ecrit : {}".format(SORTIE))
    print("{} planches, {} assises, vrille de {:.0f} deg, emprise {:.0f} mm"
          .format(len(pieces), mur.ASSISES, mur.ANGLE, mur.LONGUEUR))


if __name__ == "__main__":
    main()
