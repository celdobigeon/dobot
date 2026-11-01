#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRIGE de l'activite du profil au stylo -- cellule/ACTIVITE_PROFIL_STYLO.md

    python corrige_profil_stylo.py              simulation
    python corrige_profil_stylo.py --reel       sur la machine
    python corrige_profil_stylo.py --sans-percage   le contour seul

Un profil ferme -- quatre droites et quatre arcs -- plus un cercle au
milieu, traces au porte-stylo sur une feuille posee devant le robot.

NE PAS METTRE DANS cellule/. L'activite consiste a ecrire ce fichier.

Ce que l'activite demande, et ou c'est fait ici
-----------------------------------------------
    1. mise en route                    -- hors du script, dobot.py
    2. monter et REGLER le stylo        -- section 1
    3. le repere de la FEUILLE          -- section 2
    4. les cibles dans ce repere        -- section 3, point()
    5. les DROITES -- movl              -- section 5
    6. les ARCS -- movc et son point    -- section 4 et 5
       de passage
    7. lever et reposer le stylo        -- section 5

CE QUE LE MOVC DEMANDE, ET QU'ON OUBLIE
----------------------------------------
`movc(via, cible)` trace l'arc qui passe par TROIS points : la ou l'on
est, le point de passage, la cible. Le point de passage n'est donc pas
le centre du cercle -- erreur classique -- mais un point DU TRACE.

Ici chaque conge est un quart de cercle, et son point de passage est
pris a 45 degres sur le conge : c'est le seul endroit ou il se calcule
sans effort, et c'est aussi le plus loin de la corde, donc le plus sur
numeriquement. Trois points alignes degenerent en droite, et le modele
le dit au lieu de le faire en silence.

UN CERCLE COMPLET NE SE FAIT PAS EN UN MOVC : depart et arrivee seraient
confondus, et il y a une infinite de cercles qui passent par deux points
confondus et un troisieme. Il en faut DEUX, chacun un demi-tour.

UNE NOTE POUR L'ENSEIGNANT, sur la hauteur du stylo
----------------------------------------------------
`definir_repere_outil(T)` et `regler_longueur_outil(L)` ne commandent
PAS la meme hauteur : seule la seconde ecrit `geo.lt`, qui entre dans
la conversion du pont vers le controleur. Ecart mesure : 27,3 mm. Sur
un trace, cela ne pardonne pas -- le stylo laboure ou n'ecrit pas.

La methode d'atelier reste la bonne : on mesure, on declare, ON
VERIFIE PAR UNE DESCENTE sur la feuille, et on corrige L_OUTIL si
l'ecart est constant. C'est exactement ce que demande la section 1 de
l'activite, et c'est la raison pour laquelle elle le demande.
"""

import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

from dobot_sim import Pilote, repere_travail

REEL = "--reel" in sys.argv
PERCAGE = "--sans-percage" not in sys.argv

# ---------------------------------------------------------------------------
# 1. L'OUTIL : LE PORTE-STYLO
# ---------------------------------------------------------------------------
#
# De l'axe de rotation du poignet a la POINTE DU STYLO, axe z vers le
# BAS -- la direction d'approche, ici celle qui appuie sur le papier.
#
#   colonne 0 : ( 1,  0,  0)     x inchange
#   colonne 1 : ( 0, -1,  0)     y retourne
#   colonne 2 : ( 0,  0, -1)     z vers le bas
#   colonne 3 : ( 0,  0, -L)     la pointe, L millimetres sous la bride
#
# 70 mm est la cote du porte-stylo du kit, stylo monte affleurant. LA
# VOTRE N'EST PAS CELLE-LA : elle depend de la profondeur a laquelle le
# stylo est serre. On la mesure, et SURTOUT on la verifie par une
# descente sur la feuille.
L_OUTIL = 70.0

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot = Pilote(nom="profil-stylo", outil="stylo",
               vitesse=60, acceleration=200, vitesse_articulaire=50)
robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 2. LE REPERE DE LA FEUILLE
# ---------------------------------------------------------------------------
#
# Un repere de TABLE : son z monte. Origine au CENTRE de la feuille,
# x vers la droite de l'operateur... non : vers le +x du robot, c'est-a-
# dire en s'eloignant de lui. On le dit, parce qu'un dessin se lit, et
# qu'un dessin retourne se voit tout de suite.
#
#   x = 235   du centre de la base au centre de la feuille
#   z =  10   l'epaisseur du support plus celle du papier -- A MESURER
FEUILLE = np.array([[1, 0, 0, 235.0],
                    [0, 1, 0,   0.0],
                    [0, 0, 1,  10.0],
                    [0, 0, 0,   1.0]], dtype=float)

assert np.allclose(FEUILLE, repere_travail(x=235, y=0, z=10)), \
    "le repere de travail et la matrice doivent dire la meme chose"

# ---------------------------------------------------------------------------
# 3. UN POINT DU DESSIN
# ---------------------------------------------------------------------------


def point(x, y, h=0.0):
    """
    La cible outil au point (x, y) du dessin, a la hauteur h au-dessus
    du papier.

    LA MATRICE EST TOUJOURS LA MEME, seule la derniere colonne bouge :
    le stylo ne tourne pas, il se promene. C'est le contraire du mur de
    kapla, ou c'est l'orientation qui travaillait.

        colonne 0 : ( 1,  0,  0)
        colonne 1 : ( 0, -1,  0)   le demi-tour de l'outil
        colonne 2 : ( 0,  0, -1)
        colonne 3 : (x, y, h)      ou l'on va, dans le repere FEUILLE

    `h` est compte VERS LE HAUT depuis le papier, parce que c'est ainsi
    qu'un operateur le dit. La matrice, elle, a son z vers le bas : d'ou
    le signe dans la derniere colonne.
    """
    dans_la_feuille = np.array([[1,  0,  0,  float(x)],
                                [0, -1,  0,  float(y)],
                                [0,  0, -1,  float(h)],
                                [0,  0,  0,       1.0]], dtype=float)
    return FEUILLE @ dans_la_feuille


#: hauteur de LEVEE entre deux traits, en mm au-dessus du papier
LEVEE = 12.0

# ---------------------------------------------------------------------------
# 4. LE PROFIL : UN RECTANGLE A CONGES, ET UN PERCAGE
# ---------------------------------------------------------------------------
#
# Cotes du dessin, en millimetres :
#
#     120 x 70 hors tout, conges de R15 aux quatre coins
#     un cercle de diametre 40 au centre
#
#   (-60, +35)                                    (+60, +35)
#        .-----------------------------------------.
#       /                                           \
#      |                      O                      |
#       \                                           /
#        '-----------------------------------------'
#
# LES COTES SE DEDUISENT, elles ne se recopient pas. Changer LARGEUR,
# HAUTEUR ou RAYON doit suffire : si vous devez reprendre un seul des
# huit points a la main, le dessin n'est pas parametre.
LARGEUR, HAUTEUR, RAYON = 120.0, 70.0, 15.0
DIAMETRE_PERCAGE = 40.0

#: demi-cotes, et centres des quatre conges
DX, DY = LARGEUR / 2.0, HAUTEUR / 2.0
CX, CY = DX - RAYON, DY - RAYON

#: le point de passage d'un quart de cercle, pris A 45 DEGRES sur le
#: conge. Ce n'est pas le centre du cercle : c'est un point DU TRACE.
BIAIS = RAYON * np.sqrt(2.0) / 2.0

# ---------------------------------------------------------------------------
# 5. LE TRACE
# ---------------------------------------------------------------------------
#
# Un seul trait ferme : on descend une fois, on remonte une fois. Entre
# les deux, droites et arcs s'enchainent sans lever le stylo -- c'est ce
# qui distingue un PROFIL d'une suite de traits.

robot.repli()

# approche au-dessus du depart, puis descente sur le papier
robot.movj(point(-CX, -DY, LEVEE), lissage=60)
robot.movl(point(-CX, -DY), vitesse=30, commentaire="le stylo se pose")

# le contour, dans le sens trigonometrique
robot.movl(point(CX, -DY), commentaire="bas")
robot.movc(point(CX + BIAIS, -CY - BIAIS), point(DX, -CY),
           commentaire="conge bas droite")
robot.movl(point(DX, CY), commentaire="droite")
robot.movc(point(CX + BIAIS, CY + BIAIS), point(CX, DY),
           commentaire="conge haut droite")
robot.movl(point(-CX, DY), commentaire="haut")
robot.movc(point(-CX - BIAIS, CY + BIAIS), point(-DX, CY),
           commentaire="conge haut gauche")
robot.movl(point(-DX, -CY), commentaire="gauche")
robot.movc(point(-CX - BIAIS, -CY - BIAIS), point(-CX, -DY),
           commentaire="conge bas gauche, et le profil se referme")

robot.movl(point(-CX, -DY, LEVEE), vitesse=30, commentaire="le stylo se leve")

if PERCAGE:
    # LE CERCLE : DEUX arcs, et non un seul. Un movc dont le depart et
    # l'arrivee seraient confondus ne definit aucun cercle.
    rayon = DIAMETRE_PERCAGE / 2.0
    robot.movj(point(rayon, 0.0, LEVEE), lissage=60)
    robot.movl(point(rayon, 0.0), vitesse=30)
    robot.movc(point(0.0, rayon), point(-rayon, 0.0),
               commentaire="premier demi-cercle")
    robot.movc(point(0.0, -rayon), point(rayon, 0.0),
               commentaire="second demi-cercle")
    robot.movl(point(rayon, 0.0, LEVEE), vitesse=30)

robot.repli()

# ---------------------------------------------------------------------------
# 6. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("profil {:.0f} x {:.0f} mm, conges R{:.0f}"
          .format(LARGEUR, HAUTEUR, RAYON))
    if PERCAGE:
        print("percage : cercle de diametre {:.0f}".format(DIAMETRE_PERCAGE))
    print("feuille  : x={:.0f}  z={:.0f} mm".format(FEUILLE[0, 3],
                                                    FEUILLE[2, 3]))
    print("stylo    : L = {:.1f} mm, levee {:.0f} mm".format(L_OUTIL, LEVEE))
    print("")

    T = point(-CX, -DY)
    print("depart du trace, vu de la base : x={:.1f}  y={:.1f}  z={:.1f}"
          .format(T[0, 3], T[1, 3], T[2, 3]))
    print("point de passage d'un conge, a 45 deg : {:.2f} mm du centre"
          .format(BIAIS * np.sqrt(2.0)))
    print("")
    print(robot.resume().splitlines()[0])

    avertissements = robot.controler()
    if avertissements:
        print("")
        print("reserves :")
        for a in avertissements[:8]:
            print("  - " + a)
    else:
        print("tout le trace est dans l'espace de travail")

    if REEL:
        robot.executer_sur_robot(vitesse=30)
    else:
        robot.simuler()
