#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prise et depose : la cible ECRITE dans le repere du CAPTEUR.

    python examples/prise_depose_matrices.py              simulation
    python examples/prise_depose_matrices.py --reel       sur la machine

L'idee, et pourquoi elle change tout
-------------------------------------
On n'ecrit PAS la cible par rapport a la base du robot. On ecrit :

    1. ou se trouve le CONVOYEUR, dans le repere du ROBOT ;
    2. ou se trouve le CAPTEUR, sur le convoyeur ;
    3. ou se trouve la piece, dans le repere du CAPTEUR.

    PRISE = CONVOYEUR @ CAPTEUR @ DANS_LE_CAPTEUR

La reference principale reste LA BASE DU ROBOT. Chaque matrice ne decrit
QU'UNE chose, et on sait laquelle corriger : le tapis a bouge ->
CONVOYEUR ; le capteur a glisse le long de la bande -> CAPTEUR ; les
pieces ont change de taille -> COTE.

Deplacez le capteur de trois centimetres, tournez le tapis d'un quart
de tour : vous changez UNE matrice, celle du capteur, et la cible suit
toute seule. C'est la difference entre un programme qui decrit une
cellule et un programme qui decrit un montage.

Un point appris, lui, melange tout -- la place du tapis, celle du
capteur, la taille de la piece -- en trois nombres dont personne ne sait
d'ou ils viennent. Il depanne ; il ne s'entretient pas. Et sur cette
machine il a un defaut de plus : pousser le bras a la main fait perdre
J1 au controleur, qui ne relit que deux capteurs d'angle sur trois.

Le repere du convoyeur
-----------------------
    origine   l'entree de la bande, au ras de la courroie
    axe y     L'AXE DU MOTEUR -- la reference, materielle, qu'on montre
              du doigt, et qui ne depend pas du cablage
    axe z     la verticale, vers le haut
    axe x     le SENS DE MARCHE, et il se DEDUIT : x = y ^ z

On ne choisit donc pas le sens de marche, on le calcule, et le repere
reste direct. D'ou la regle : INVERSER LE TAPIS, c'est prendre l'axe
moteur dans l'autre sens -- une colonne change de signe, sa voisine
suit, et rien d'autre ne bouge.

Le capteur, lui, est POSE SUR LE TAPIS : sa place se mesure le long de
la bande, depuis l'entree. Une seule cote, dans l'unite ou on la mesure
vraiment -- un reglet pose sur le bati.

Ce decoupage rend les trois nombres suivants LISIBLES. Dans le repere du
capteur, la piece arretee se trouve a

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
# --viser : on va AU POINT DE PRISE et on s'y arrete, pour mesurer
# l'ecart a la piece au reglet. Rien ne se referme, rien ne se depose.
VISER = "--viser" in sys.argv

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
#
# 89,3 mm. La ventouse etait declaree a 79,3 -- cote relevee en posant
# la pointe sur le plan de travail -- et le robot descendait 10 mm trop
# bas sur le tapis. Un outil plus LONG que declare fait descendre
# d'autant : le modele croit la pointe plus haut qu'elle n'est.
#
# L'erreur est CONSTANTE, elle ne depend pas de la posture : c'est la
# signature d'une longueur d'outil, et c'est ce qui permet de l'imputer
# a l'outil plutot qu'au repere du tapis ou a la hauteur de la bande.

OUTIL = np.array([[1,  0,  0,     0],
                  [0, -1,  0,     0],
                  [0,  0, -1, -89.3],
                  [0,  0,  0,     1]], dtype=float)

robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 3. LE CONVOYEUR, DANS LE REPERE DU ROBOT
# ---------------------------------------------------------------------------
#
# COTES RELEVEES SUR LA CELLULE le 30/10/2026 :
#
#   x = 210    du centre du robot au CENTRE du convoyeur
#   y = -400   l'axe moteur est a 400 mm sur la DROITE de l'operateur,
#              et la droite de l'operateur est le -y du robot
#   z = 60     hauteur de la bande
#
# L'origine est donc l'intersection de L'AXE MOTEUR et de la LIGNE
# MEDIANE de la bande. Les deux se montrent du doigt, et c'est ce qui
# fait un bon repere : on sait y revenir au reglet.
#
# Mesurer jusqu'au CENTRE plutot que jusqu'au profile supprime une cote
# intermediaire -- il avait d'abord ete releve 150 mm jusqu'au profile,
# ce qui obligeait a y ajouter la demi-largeur, donc a savoir de quel
# bord on etait parti. Une cote de moins est une incertitude de moins.
#
# Pour memoire : bati 120 mm de large, bande utile 100 mm, centre a
# 60 mm du bord du profile.
#
#   premiere colonne  ( 0 ;  1 ; 0)    sens de marche, = y ^ z
#   deuxieme colonne  (-1 ;  0 ; 0)    L'AXE MOTEUR, la reference
#   troisieme colonne ( 0 ;  0 ; 1)    vers le haut
#   quatrieme colonne (210 ; -400 ; 60) axe moteur x ligne mediane

CONVOYEUR = np.array([[ 0, -1, 0,  210],
                      [ 1,  0, 0, -400],
                      [ 0,  0, 1,   60],
                      [ 0,  0, 0,    1]], dtype=float)

# Le meme convoyeur, moteur dans l'autre sens -- pour montrer ce que
# coute une inversion : une colonne, et sa voisine qui suit.
#
# CONVOYEUR = np.array([[0,  1, 0, 230],
#                       [-1, 0, 0, 250],
#                       [0,  0, 1,   0],
#                       [0,  0, 0,   1]], dtype=float)

# ---------------------------------------------------------------------------
# 4. LE CAPTEUR, SUR LE CONVOYEUR
# ---------------------------------------------------------------------------
#
# Une seule cote : sa position le long de la bande, depuis l'entree. Il
# ne tourne pas par rapport au tapis, donc la matrice est une simple
# translation -- et c'est ce qui la rend relisible.

# 300 mm le long de la bande depuis l'axe moteur, vers les y positifs.
#
# CE NOMBRE A ETE CONFIRME PAR UNE CONTRADICTION. Un premier jeu de
# cotes donnait l'axe moteur a y = -250 et le capteur a 300 mm de lui,
# ce qui placait le capteur a y = +50. Or il se trouve a y = -100, et la
# mesure faite par le robot le confirmait : ventouse arretee sur le
# point calcule, l'ecart a la piece valait 145 mm, donc la piece etait a
# y = -107,5 -- a 7,5 mm du capteur.
#
# Les trois cotes ne pouvaient pas etre vraies ensemble. Le releve
# repris a donne l'axe moteur a -400, et tout concorde.
#
# A NOTER, parce que c'est l'interet de la chaine de reperes : corriger
# les deux cotes de +150 et -150 donne EXACTEMENT le meme point de
# prise. Une erreur sur la place du tapis et une erreur egale et opposee
# sur la place du capteur se compensent. Ce sont les cotes qui etaient
# fausses, pas la cible -- et seul un troisieme releve pouvait le dire.
#
# Aucun decalage lateral : le repere du convoyeur est deja sur la ligne
# mediane de la bande, et la piece y circule. La matrice se reduit donc
# a une translation le long de x -- une seule cote, celle qu'on a
# mesuree.

CAPTEUR = np.array([[1, 0, 0, 300],
                    [0, 1, 0,   0],
                    [0, 0, 1,   0],
                    [0, 0, 0,   1]], dtype=float)

# ---------------------------------------------------------------------------
# 5. LA PIECE, DANS LE REPERE DU CAPTEUR
# ---------------------------------------------------------------------------

COTE = 25.0            # mm, cube de 25
HAUTEUR = 25.0         # mm, dessus du cube au-dessus de la BANDE
                       # (la hauteur de la bande est deja dans CONVOYEUR)

# -COTE/2 : le capteur voit le BORD AVANT, la ventouse vise le CENTRE.
# Ce decalage, laisse implicite, coute une demi-piece a chaque prise --
# toujours du meme cote, donc facile a prendre pour un reglage.
#
# ATTENTION : le quart de tour de l'orientation SUIT le repere du
# convoyeur. Inversez l'axe moteur et il doit changer de signe, sinon J4
# saute de 0 a 180 deg. controler() le dit, et c'est la raison de le
# lancer avant toute descente sur la machine.
#
#   premiere colonne  ( 0 ; -1 ;  0)   l'outil tourne d'un quart de tour
#   deuxieme colonne  (-1 ;  0 ;  0)
#   troisieme colonne ( 0 ;  0 ; -1)   z de l'outil vers le bas

DANS_LE_CAPTEUR = np.array([[ 0, -1,  0, -12.5],
                            [-1,  0,  0,     0],
                            [ 0,  0, -1,  25.0],
                            [ 0,  0,  0,     1]], dtype=float)

# ---------------------------------------------------------------------------
# 6. LA CIBLE : LA CHAINE COMPLETE
# ---------------------------------------------------------------------------
#
#     base du robot  ->  convoyeur  ->  capteur  ->  piece

PRISE = CONVOYEUR @ CAPTEUR @ DANS_LE_CAPTEUR

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
if VISER:
    # Une approche, une descente, et on s'arrete la. Lentement : on
    # regarde ou la ventouse arrive, on ne produit pas.
    robot.movj(PRISE @ HAUT)
    robot.movl(PRISE, vitesse=15)
for _ in range(0 if VISER else 3):
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
if not VISER:
    robot.repli()

# ---------------------------------------------------------------------------
# 6. CE QU'ON A ECRIT, ET CE QUE CA DONNE
# ---------------------------------------------------------------------------

np.set_printoptions(precision=1, suppress=True)
print("CONVOYEUR  (l'entree de la bande, dans le repere du robot)")
print(CONVOYEUR)
print("  axe moteur       : {}".format(np.round(CONVOYEUR[:3, 1], 1)))
print("  sens de marche   : {}  (= y ^ z)".format(np.round(CONVOYEUR[:3, 0], 1)))
print("")
print("CAPTEUR  (sur le convoyeur, a {:.0f} mm de l'entree)"
      .format(CAPTEUR[0, 3]))
print("  dans le repere du robot : {}"
      .format(np.round((CONVOYEUR @ CAPTEUR)[:3, 3], 1)))
print("")
print("PRISE = CONVOYEUR @ CAPTEUR @ DANS_LE_CAPTEUR")
print(PRISE)
print("  position x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
print("  axe z de l'outil : {} -- vers le bas".format(np.round(PRISE[:3, 2], 1)))
print("")

# Ce que la chaine fait gagner : chaque defaut se corrige a UN endroit.
CAPTEUR_PLUS_LOIN = np.array([[1, 0, 0, 330],
                              [0, 1, 0,   0],
                              [0, 0, 1,   0],
                              [0, 0, 0,   1]], dtype=float)
CONVOYEUR_RECULE = np.array([[ 0, -1, 0,  240],
                             [ 1,  0, 0, -400],
                             [ 0,  0, 1,   60],
                             [ 0,  0, 0,    1]], dtype=float)
print("Ce qu'il faut changer, et ou :")
print("  capteur glisse de 30 mm le long de la bande  -> CAPTEUR seul")
print("     la prise passe de y={:.1f} a y={:.1f}"
      .format(PRISE[1, 3],
              (CONVOYEUR @ CAPTEUR_PLUS_LOIN @ DANS_LE_CAPTEUR)[1, 3]))
print("  tapis recule de 30 mm                        -> CONVOYEUR seul")
print("     la prise passe de x={:.1f} a x={:.1f}"
      .format(PRISE[0, 3],
              (CONVOYEUR_RECULE @ CAPTEUR @ DANS_LE_CAPTEUR)[0, 3]))
print("  pieces de 40 mm au lieu de 25                -> COTE seul")
print("")

avertissements = robot.controler()
if avertissements:
    print("avertissements :")
    for a in avertissements[:6]:
        print("  -", a)
else:
    print("tout le cycle est dans l'espace de travail")
print(robot.resume().splitlines()[0])

if not REEL:
    robot.simuler()
    sys.exit(0)

print("")
print("LE ROBOT VA BOUGER.")

# OU EST LE BRAS AVANT DE PARTIR. Un repli qui ne produit aucun
# mouvement -- parce que le bras y est deja -- ressemble, vu de
# l'atelier, a un repli qui n'a pas eu lieu. On le dit plutot que de
# laisser deviner.
_q = robot.angles_sur_robot()
_repli = np.asarray(robot.robot.q_home, dtype=float)
_ecart = float(np.max(np.abs(_q - _repli)))
print("  posture actuelle : J = ({:.1f}, {:.1f}, {:.1f}, {:.1f}) deg"
      .format(*_q))
print("  repli            : J = ({:.1f}, {:.1f}, {:.1f}, {:.1f}) deg"
      .format(*_repli))
if _ecart < 1.0:
    print("  -> le bras EST DEJA replie : le premier mouvement ne se verra")
    print("     pas, et c'est normal.")
else:
    print("  -> {:.0f} deg d'ecart sur l'axe le plus loin : le bras VA se"
          .format(_ecart))
    print("     replier, et cela doit se voir.")
print("")

robot.executer_sur_robot(vitesse=30)

if VISER:
    print("")
    print("=" * 60)
    print("  La ventouse est au point de prise CALCULE :")
    print("    x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
    print("")
    print("  Mesurez l'ecart a la piece, LE LONG DU TAPIS -- c'est-a-dire")
    print("  selon y, le sens de marche. Comptez POSITIF si la ventouse")
    print("  est en aval de la piece, vers la sortie du tapis.")
    print("")
    print("  Cet ecart se reporte directement sur UNE cote : les 300 mm")
    print("  du capteur le long de la bande. Rien d'autre a toucher.")
    print("=" * 60)
    try:
        input("  Entree pour degager le bras... ")
    except EOFError:
        pass
    robot.vider_programme()
    robot.movl(PRISE @ HAUT, vitesse=20)
    robot.repli()
    robot.executer_sur_robot(vitesse=30)
    print("  bras degage")
