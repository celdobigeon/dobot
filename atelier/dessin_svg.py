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
                hauteur_ligne=34):
    """
    Une matrice homogene 4x4, crochets compris, colonnes teintees.

    `lignes` : quatre suites de quatre chaines, deja echappees pour le
    SVG (&#8722; pour le signe moins). `notes` : une legende par
    colonne, ou None.

    POURQUOI UNE VRAIE MATRICE ET NON QUATRE ETIQUETTES. Parce que la
    colonne 3 n'a de sens qu'a cote des trois autres, et que la ligne
    du bas -- 0 0 0 1 -- est precisement ce qu'un etudiant oublie
    d'ecrire. Une figure qui la cache lui apprend a l'oublier.

    `x`, `y` : coin haut gauche du corps de la matrice.
    """
    couleurs = (ROUGE, VERT, BLEU, ENCRE)
    out = []
    larg = 4 * largeur_colonne
    haut = 4 * hauteur_ligne

    # les colonnes, teintees, derriere les chiffres
    for c in range(4):
        out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                   'height="{:.1f}" rx="6" fill="{}" fill-opacity="0.09"/>'
                   .format(x + c * largeur_colonne, y, largeur_colonne,
                           haut, couleurs[c]))

    # les crochets
    for cote in (0, 1):
        x0 = x - 12 if cote == 0 else x + larg + 12
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

    out.append(texte(x - 32, y + haut / 2 + 6, nom, 16, ENCRE, "700", "end"))

    for c, note in enumerate(notes or []):
        if note:
            out.append(texte(x + (c + 0.5) * largeur_colonne, y + haut + 26,
                             note, 12, "#555", "400", "middle"))
    return out
