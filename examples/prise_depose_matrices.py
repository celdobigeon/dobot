#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prise et depose : la cible ECRITE dans le repere du CAPTEUR.

    python examples/prise_depose_matrices.py              simulation
    python examples/prise_depose_matrices.py --reel       sur la machine

L'idee, et pourquoi elle change tout
-------------------------------------
On n'ecrit PAS la cible par rapport a la base du robot. On ecrit :

    1. ou se trouve le CAPTEUR, dans le repere du robot ;
    2. ou se trouve la piece, dans le repere du CAPTEUR.

    PRISE = CAPTEUR @ DANS_LE_CAPTEUR

Deplacez le capteur de trois centimetres, tournez le tapis d'un quart
de tour : vous changez UNE matrice, celle du capteur, et la cible suit
toute seule. C'est la difference entre un programme qui decrit une
cellule et un programme qui decrit un montage.

Un point appris, lui, melange tout -- la place du tapis, celle du
capteur, la taille de la piece -- en trois nombres dont personne ne sait
d'ou ils viennent. Il depanne ; il ne s'entretient pas. Et sur cette
machine il a un defaut de plus : pousser le bras a la main fait perdre
J1 au controleur, qui ne relit que deux capteurs d'angle sur trois.

Le repere du capteur
---------------------
    origine   le faisceau, au ras de la bande
    axe x     le SENS DE MARCHE du tapis
    axe y     en travers de la bande
    axe z     vers le haut

Ce choix n'est pas gratuit : il rend les trois nombres suivants
LISIBLES. Dans ce repere, la piece arretee par le capteur se trouve a

    x = -cote/2    le capteur voit le BORD AVANT, la ventouse vise le
                   CENTRE : une demi-piece en amont, et c'est tout
    y = 0          au milieu de la bande
    z = hauteur    le dessus de la piece

Trois nombres qu'on peut discuter, corriger, expliquer a un etudiant.
Changez de pieces, vous changez `COTE` et rien d'autre.

Comment lire une matrice homogene
----------------------------------
    [ x_outil  y_outil  z_outil  position ]
    [    0        0        0         1    ]

Les trois premieres colonnes donnent l'ORIENTATION, la quatrieme la
POSITION. Une cible de prise verticale a toujours la meme forme : l'axe
z de l'outil vaut (0, 0, -1), il pointe VERS LE BAS, et c'est la
direction d'approche.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import Pilote                                   # noqa: E402

REEL = "--reel" in sys.argv

# ---------------------------------------------------------------------------
# 1. L'EN-TETE
# ---------------------------------------------------------------------------

robot = Pilote(nom="essai-matrices", outil="ventouse",
               vitesse=120, acceleration=400, vitesse_articulaire=50)

# ---------------------------------------------------------------------------
# 2. LE REPERE OUTIL
# ---------------------------------------------------------------------------
#
# De l'axe de rotation de l'outil a sa pointe. Axe z vers le bas.
# Ventouse droite de 79,3 mm, mesuree sur la machine.

OUTIL = np.array([[1,  0,  0,     0],
                  [0, -1,  0,     0],
                  [0,  0, -1, -79.3],
                  [0,  0,  0,     1]], dtype=float)

robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 3. LE REPERE DU CAPTEUR
# ---------------------------------------------------------------------------
#
# LA SEULE CHOSE A MESURER. Origine au faisceau, axe x dans le sens de
# marche du tapis, axe z vers le haut.
#
# Ici le capteur est a 230 mm devant le robot, dans son axe, et la bande
# descend les y -- de la gauche vers la droite de l'operateur. L'axe x
# du capteur vaut donc (0 ; -1 ; 0) dans le repere du robot, et c'est la
# PREMIERE COLONNE de la matrice.
#
#   premiere colonne  ( 0 ; -1 ;  0)   sens de marche du tapis
#   deuxieme colonne  ( 1 ;  0 ;  0)   en travers de la bande
#   troisieme colonne ( 0 ;  0 ;  1)   vers le haut
#   quatrieme colonne (230 ;  0 ;  0)  ou se trouve le faisceau

CAPTEUR = np.array([[ 0, 1, 0, 230],
                    [-1, 0, 0,   0],
                    [ 0, 0, 1,   0],
                    [ 0, 0, 0,   1]], dtype=float)

# ---------------------------------------------------------------------------
# 4. LA PIECE, DANS LE REPERE DU CAPTEUR
# ---------------------------------------------------------------------------

COTE = 25.0            # mm, taille de la piece dans le sens de marche
HAUTEUR = 75.0         # mm, dessus de la piece au-dessus du faisceau

# -COTE/2 : le capteur voit le BORD AVANT, la ventouse vise le CENTRE.
# C'est le decalage qui, laisse implicite, coute une demi-piece a chaque
# prise -- toujours du meme cote, donc facile a prendre pour un reglage.

# L'orientation, elle, porte un QUART DE TOUR autour de z, et il faut
# dire pourquoi. Le repere du capteur est tourne de -90 deg par rapport
# au robot -- le tapis est en travers. Une cible ecrite dedans en herite
# l'orientation, et le poignet doit alors fournir ces 90 deg a lui seul :
# J4 part a -93 deg, hors de sa butee de +/-90. controler() le dit.
#
# Une ventouse se moque de son orientation autour de z. On choisit donc
# celle qui remet le poignet au large, et le quart de tour s'annule avec
# celui du capteur. Avec une PINCE il serait au contraire impose par la
# piece, et c'est le tapis ou le robot qu'il faudrait deplacer.
#
#   premiere colonne  (0 ; 1 ; 0)    l'outil tourne d'un quart de tour
#   deuxieme colonne  (1 ; 0 ; 0)
#   troisieme colonne (0 ; 0 ; -1)   z de l'outil vers le bas

DANS_LE_CAPTEUR = np.array([[0,  1,  0, -12.5],
                            [1,  0,  0,     0],
                            [0,  0, -1,  75.0],
                            [0,  0,  0,     1]], dtype=float)

PRISE = CAPTEUR @ DANS_LE_CAPTEUR

# La depose, elle, ne depend pas du capteur : c'est un endroit de la
# table. On l'ecrit donc dans le repere du ROBOT.

DEPOSE = np.array([[1,  0,  0,  170],
                   [0, -1,  0, -140],
                   [0,  0, -1,   75],
                   [0,  0,  0,    1]], dtype=float)

# un degagement de 70 mm vers le haut, dans le repere de l'OUTIL
HAUT = np.array([[1, 0, 0,   0],
                 [0, 1, 0,   0],
                 [0, 0, 1, -70],
                 [0, 0, 0,   1]], dtype=float)

# PRISE @ HAUT se lit « 70 mm au-dessus de PRISE ». L'ordre compte :
# ecrit a gauche, HAUT @ PRISE deplacerait dans le repere du ROBOT, et
# comme l'axe z de l'outil pointe vers le bas, le bras DESCENDRAIT.

# ---------------------------------------------------------------------------
# 5. LE CYCLE
# ---------------------------------------------------------------------------

robot.repli()
for _ in range(3):
    # prise : ARRET AU POINT, sinon la ventouse aspire avant d'y etre
    robot.movj(PRISE @ HAUT, lissage=70)
    robot.movl(PRISE, vitesse=40)
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(PRISE @ HAUT, lissage=70)

    robot.movj(DEPOSE @ HAUT, lissage=70)
    robot.movl(DEPOSE, vitesse=40)
    robot.ventouse(False)
    robot.attendre(0.3)
    robot.movl(DEPOSE @ HAUT)
robot.repli()

# ---------------------------------------------------------------------------
# 6. CE QU'ON A ECRIT, ET CE QUE CA DONNE
# ---------------------------------------------------------------------------

np.set_printoptions(precision=1, suppress=True)
print("CAPTEUR  (ou est le faisceau, dans le repere du robot)")
print(CAPTEUR)
print("  position x={:.1f}  y={:.1f}  z={:.1f}".format(*CAPTEUR[:3, 3]))
print("  sens de marche du tapis : {}".format(np.round(CAPTEUR[:3, 0], 1)))
print("")
print("PRISE = CAPTEUR @ DANS_LE_CAPTEUR")
print(PRISE)
print("  position x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
print("  axe z de l'outil : {} -- vers le bas".format(np.round(PRISE[:3, 2], 1)))
print("")

# Ce que l'on gagne : deplacer le capteur ne touche qu'UNE matrice.
CAPTEUR_DEPLACE = np.array([[ 0, 1, 0, 230],
                            [-1, 0, 0,  30],
                            [ 0, 0, 1,   0],
                            [ 0, 0, 0,   1]], dtype=float)
print("Si le capteur avance de 30 mm le long du tapis :")
print("  la prise passe de y={:.1f} a y={:.1f} -- sans toucher au reste"
      .format(PRISE[1, 3], (CAPTEUR_DEPLACE @ DANS_LE_CAPTEUR)[1, 3]))
print("")

avertissements = robot.controler()
if avertissements:
    print("avertissements :")
    for a in avertissements[:6]:
        print("  -", a)
else:
    print("tout le cycle est dans l'espace de travail")
print(robot.resume().splitlines()[0])

if REEL:
    print("")
    print("LE ROBOT VA BOUGER.")
    robot.executer_sur_robot(vitesse=30)
else:
    robot.simuler()
