# -*- coding: utf-8 -*-
"""
Le peu de SVG que partagent les figures des activites.

Un seul endroit pour la police, les couleurs des axes et le trace
d'une matrice homogene : deux figures qui montrent la meme matrice
doivent la montrer PAREIL, et cela ne tient pas si chacune la
redessine dans son coin.
"""

__all__ = ["POLICE", "ROUGE", "VERT", "BLEU", "ENCRE", "GRIS", "FOND",
           "texte", "matrice_4x4"]

POLICE = "IBM Plex Sans, Segoe UI, sans-serif"

#: les couleurs des axes, celles du simulateur
ROUGE, VERT, BLEU = "#c0392b", "#2e9e4f", "#2b6cb0"
ENCRE, GRIS, FOND = "#333", "#8a8a8a", "#fdfcf7"


def texte(u, v, t, taille=14, couleur=ENCRE, poids="400", ancre="start",
          italique=False):
    return ('<text x="{:.1f}" y="{:.1f}" font-family="{}" font-size="{}" '
            'font-weight="{}" fill="{}" text-anchor="{}"{}>{}</text>'
            .format(u, v, POLICE, taille, poids, couleur, ancre,
                    ' font-style="italic"' if italique else "", t))


def matrice_4x4(x, y, nom, lignes, notes=None, largeur_colonne=86,
                hauteur_ligne=34, titres=None, cotes=None,
                couleur_titres=None, couleur_cotes=None, couleurs=None):
    """
    Une matrice, crochets compris, colonnes teintees.

    Le nom dit 4x4 parce que c'est le cas courant, mais la taille se
    lit sur `lignes` : une rotation 3x3 se dessine pareil.

    `lignes` : quatre suites de quatre chaines, deja echappees pour le
    SVG (&#8722; pour le signe moins). `notes` : une legende par
    colonne, ou None.

    `titres` et `cotes` nomment les colonnes et les lignes -- par
    exemple `Xo Yo Zo` au-dessus et `Xb Yb Zb` a gauche. CE N'EST PAS
    DE LA DECORATION : avec ces etiquettes, on lit d'un coup d'oeil que
    Yb et Yo ne different que par un signe, ce que seize chiffres nus
    ne disent pas.

    POURQUOI UNE VRAIE MATRICE ET NON QUATRE ETIQUETTES. Parce que la
    colonne 3 n'a de sens qu'a cote des trois autres, et que la ligne
    du bas -- 0 0 0 1 -- est precisement ce qu'un etudiant oublie
    d'ecrire. Une figure qui la cache lui apprend a l'oublier.

    `x`, `y` : coin haut gauche du corps de la matrice.
    """
    couleurs = tuple(couleurs or (ROUGE, VERT, BLEU, ENCRE))
    out = []
    n_l = len(lignes)
    n_c = len(lignes[0]) if n_l else 0
    larg = n_c * largeur_colonne
    haut = n_l * hauteur_ligne

    for c, t in enumerate(titres or []):
        if t:
            out.append(texte(x + (c + 0.5) * largeur_colonne, y - 14, t, 14,
                             (couleur_titres or couleurs)[c]
                             if not isinstance(couleur_titres, str)
                             else couleur_titres,
                             "700", "middle"))
    for i, t in enumerate(cotes or []):
        if t:
            out.append(texte(x - 18, y + (i + 0.5) * hauteur_ligne + 6, t, 14,
                             (couleur_cotes or couleurs)[i]
                             if not isinstance(couleur_cotes, str)
                             else couleur_cotes,
                             "700", "end"))

    # les colonnes, teintees, derriere les chiffres
    for c in range(n_c):
        out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                   'height="{:.1f}" rx="6" fill="{}" fill-opacity="0.09"/>'
                   .format(x + c * largeur_colonne, y, largeur_colonne,
                           haut, couleurs[c]))

    # les crochets
    for cote in (0, 1):
        x0 = x - 10 if cote == 0 else x + larg + 10
        pointe = 11 if cote == 0 else -11
        out.append('<path d="M {:.1f} {:.1f} h {:.1f} v {:.1f} h {:.1f}" '
                   'fill="none" stroke="{}" stroke-width="2.2" '
                   'stroke-linecap="round" stroke-linejoin="round"/>'
                   .format(x0 + pointe, y - 6, -pointe, haut + 12, pointe,
                           ENCRE))

    for i, ligne in enumerate(lignes):
        for c, val in enumerate(ligne):
            out.append(texte(x + (c + 0.5) * largeur_colonne,
                             y + (i + 0.5) * hauteur_ligne + 6,
                             val, 17, couleurs[c], "700", "middle"))

    if nom:
        decal = 46 if cotes else 30
        out.append(texte(x - decal, y + haut / 2 + 6, nom, 16, ENCRE, "700",
                         "end"))

    for c, note in enumerate(notes or []):
        if note:
            out.append(texte(x + (c + 0.5) * largeur_colonne, y + haut + 26,
                             note, 12, "#555", "400", "middle"))
    return out
