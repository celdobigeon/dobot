#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRIGE de l'activite pick and place -- cellule/ACTIVITE_PICK_AND_PLACE.md

    python corrige_pick_and_place.py                  simulation
    python corrige_pick_and_place.py --reel           sur la machine
    python corrige_pick_and_place.py --lissage 0      sans arrondi
    python corrige_pick_and_place.py --angle 30       depose en biais

UNE piece, prise a un endroit et posee a un autre. Rien de plus :
l'empilement et la rotation d'une assise a l'autre sont l'objet de
l'activite suivante, le structure de kapla.

NE PAS METTRE DANS cellule/. L'activite consiste a ecrire ce fichier.

PAS DE REPERE DE PROGRAMMATION ICI, ET C'EST VOULU
---------------------------------------------------
Les deux cibles sont ecrites DIRECTEMENT par rapport a la base du
robot. C'est la facon naive, et c'est exactement celle qu'il faut
faire pratiquer d'abord : l'etudiant doit sentir ce qu'elle coute
avant qu'on lui vende la solution.

Ce qu'elle coute se mesure a la derniere question du sujet. Deplacez
le magasin de deux centimetres : ici, deux nombres a reprendre, c'est
indolore. Mais chaque cote de la cible melange DEUX informations --
ou est le magasin, et ou est la piece dans le magasin -- et plus
rien ne dit laquelle on corrige. Avec douze deposes, comme a la
structure de kapla, la facture devient impayable.

L'activite suivante introduit `MAGASIN` et `STRUCTURE`, et la meme
cible s'y ecrit `MAGASIN @ DANS_LE_MAGASIN`. Ne pas anticiper : le
confort ne se comprend que par le manque.

Ce que l'activite demande, et ou c'est fait ici
-----------------------------------------------
    1. mise en route                       -- hors du script, dobot.py
    2. mesure et matrice de l'outil        -- section 1
    3. les deux cibles, depuis la base     -- section 2
    4. approches et retraits par @         -- section 3
    5. LE LISSAGE, et ce qu'il coute       -- section 4, le coeur

LE LISSAGE, PUISQUE C'EST L'OBJET DE L'ACTIVITE
------------------------------------------------
`lissage` est un POURCENTAGE porte par une instruction, et il concerne
le raccord avec la SUIVANTE. A 0 le robot s'arrete au point ; a 100 il
coupe le coin au maximum admissible, soit 49 % du plus court des deux
segments.

Trois choses a ne pas confondre :

  - un point lisse n'est PAS atteint. Donc on ne lisse jamais le point
    ou l'outil agit : ici la prise et la depose restent a 0 ;
  - le modele refuse de lisser autour d'une ATTENTE, parce qu'une
    temporisation demandee doit etre tenue ;
  - SUR CE CONTROLEUR-CI, le lissage n'existe pas en PTP. Le pont le
    dit a l'envoi, et le robot reel marque un arret a chaque point. Le
    simulateur montre ce qu'on gagnerait avec un controleur qui a des
    zones (ABB) ou du CNT (Fanuc).

Le gain se lit dans `robot.resume()`, qui affiche les secondes
economisees. Mesure sur ce cycle-ci :

    lissage    0 %  ->  24,01 s     ecart au point de degagement  0,0 mm
    lissage   40 %  ->  22,13 s                                   0,1 mm
    lissage   70 %  ->  20,73 s                                   0,7 mm
    lissage  100 %  ->  19,32 s                                   2,0 mm

Et le point de prise, lui, est atteint a 0,00 mm dans les quatre cas :
il n'est pas lisse. Le raccord melange deux lois quintiques, donc de
vitesse nulle aux bords ; les deux segments sont tous deux pres du
point pendant le recouvrement, et l'ecart geometrique reste petit. Ce
qu'on achete avec le lissage, ici, c'est 19,5 % de temps de cycle pour
2 mm -- et non un virage large.
"""

import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

from dobot_sim import Pilote
from dobot_sim import homogeneous as hg

REEL = "--reel" in sys.argv


def option(nom, defaut):
    """Valeur numerique passee en ligne de commande, ou le defaut."""
    if nom in sys.argv:
        return float(sys.argv[sys.argv.index(nom) + 1])
    return defaut


#: pourcentage d'arrondi des raccords de DEGAGEMENT. 0 a 100.
LISSAGE = option("--lissage", 70.0)
#: orientation de la piece deposee, en degres. 0 = le long de x.
#:
#: 90 PAR DEFAUT, ET CE N'EST PAS UN GOUT. La structure est droit
#: devant le robot, cap 90 degres ; or J4 = orientation - J1, et J4 ne
#: va que de -90 a +90. Une depose a 0 degre demande donc au poignet
#: d'aller exactement en butee -- `controler()` le signale. A 90 degres
#: la piece pointe dans l'axe du bras et le poignet reste a zero.
#: Essayez `--angle 0` pour voir la reserve apparaitre.
ANGLE = option("--angle", 90.0)

LONGUEUR, LARGEUR, EPAISSEUR = 125.0, 30.0, 10.0

# ---------------------------------------------------------------------------
# 1. L'OUTIL
# ---------------------------------------------------------------------------
#
# De l'axe de rotation du poignet a la pointe de la ventouse, z vers le
# BAS. Voir cellule/assets/reperes_base_outil.svg : les deux reperes
# ont le meme x, et un demi-tour autour de x retourne y et z.
#
#   colonne 0 : ( 1,  0,  0)     x inchange
#   colonne 1 : ( 0, -1,  0)     y retourne
#   colonne 2 : ( 0,  0, -1)     z vers le bas : direction d'approche
#   colonne 3 : ( 0,  0, -L)     la pointe, L millimetres sous la bride
#
# L se MESURE (cellule/mesurer_outil.py) puis se VERIFIE par une
# descente sur un plan de cote connue.
L_OUTIL = 79.3

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot = Pilote(nom="pick-and-place", outil="ventouse",
               vitesse=100, acceleration=300, vitesse_articulaire=50)
robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 2. LES DEUX CIBLES, PAR RAPPORT A LA BASE DU ROBOT
# ---------------------------------------------------------------------------
#
# Chacune est une situation complete : OU va la ventouse, et COMMENT
# elle est orientee. Rien d'autre n'est declare -- pas de repere de
# travail, c'est le sujet de l'activite suivante.
#
# Les cotes se relevent au reglet depuis le CENTRE DE LA BASE.

# le magasin est a droite du robot, hors de l'emprise du convoyeur
# (x = 150 a 270). La piece y est couchee le long de x.
#
#   x = 60, y = -220   la piece, mesuree depuis la base
#   z = 10             le DESSUS de la piece : le plan porte a 0,
#                      la piece fait EPAISSEUR, la ventouse se pose
#                      dessus. Les deux cotes sont deja melangees dans
#                      ce seul nombre -- retenez-le pour la derniere
#                      question.
PRISE = np.array([[1,  0,  0,        60.0],
                  [0, -1,  0,      -220.0],
                  [0,  0, -1,   EPAISSEUR],
                  [0,  0,  0,         1.0]], dtype=float)

# la depose est droit devant le robot, et la piece y est tournee de
# ANGLE. Voir cellule/assets/cible_angle.svg, qui montre d'ou viennent
# c et s :
#
#   colonne 0 : ( cos a,  sin a, 0)   l'axe x de l'outil, donc la
#                                     LONGUEUR de la piece
#   colonne 1 : ( sin a, -cos a, 0)   son y, retourne par Rx(180)
#   colonne 2 : (     0,      0, -1)  son z, vers le BAS
#   colonne 3 : ou elle se pose
#
# A ANGLE = 0 cette matrice redevient celle de la prise, au point pres :
# la prise est le cas particulier ou l'on ne tourne rien.
# ELLE SE COMPOSE, elle ne se recopie pas. Deux rotations elementaires,
# dans cet ordre, et la lecture du programme suit celle de la figure :
#
#   Rz(a)      on tourne la piece autour de la verticale
#   Rx(180)    l'outil approche PAR LE DESSUS -- y et z se retournent
#
# Ecrire les seize nombres a la main donne le meme resultat, et
# l'assertion ci-dessous le prouve ; mais la composition DIT ce qu'on
# fait, la liste de nombres non.
DEPOSE = (hg.translation(0.0, 240.0, EPAISSEUR)
          @ hg.rot_z(np.radians(ANGLE))
          @ hg.rot_x(np.pi))

_c, _s = np.cos(np.radians(ANGLE)), np.sin(np.radians(ANGLE))
assert np.allclose(DEPOSE, [[_c,  _s,  0,         0.0],
                            [_s, -_c,  0,       240.0],
                            [0,    0, -1,   EPAISSEUR],
                            [0,    0,  0,         1.0]])

# la piece du simulateur : du decor, aucune instruction. magasin=True
# en fait reapparaitre une des qu'on emporte celle-ci.
robot.poser_piece(x=float(PRISE[0, 3]), y=float(PRISE[1, 3]),
                  dims=(LONGUEUR, LARGEUR, EPAISSEUR), magasin=True,
                  nom="piece")

# ---------------------------------------------------------------------------
# 3. LES DEGAGEMENTS, PAR @
# ---------------------------------------------------------------------------
#
# Exprimes dans le repere de l'OUTIL : c'est le seul endroit du
# programme ou l'on compose, et c'est deja beaucoup. L'axe z de l'outil
# pointant vers le bas, MONTER s'ecrit avec un z NEGATIF -- on le
# verifie plutot que de le croire.

HAUT = hg.translation(0, 0, -80)
assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"

# ---------------------------------------------------------------------------
# 4. LE CYCLE, ET LE LISSAGE
# ---------------------------------------------------------------------------
#
# LE LISSAGE NE PORTE QUE SUR LES DEGAGEMENTS. Les deux points ou
# l'outil agit -- la prise et la depose -- restent a 0 : un point lisse
# n'est pas atteint, et une ventouse qui aspire a cote n'emporte rien.

robot.repli()

# prise : approche lissee, descente franche, aspiration, retrait
robot.movj(PRISE @ HAUT, lissage=LISSAGE)
robot.movl(PRISE, vitesse=40)                      # lissage 0 par defaut
robot.ventouse(True)
robot.attendre(0.3)
robot.movl(PRISE @ HAUT, lissage=LISSAGE)

# depose : le meme geste, a l'autre bout et dans l'autre orientation
robot.movj(DEPOSE @ HAUT, lissage=LISSAGE)
robot.movl(DEPOSE, vitesse=40)
robot.ventouse(False)
robot.attendre(0.3)
robot.movl(DEPOSE @ HAUT, lissage=LISSAGE)

robot.repli()

# ---------------------------------------------------------------------------
# 5. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("piece {:.0f} x {:.0f} x {:.0f} mm"
          .format(LONGUEUR, LARGEUR, EPAISSEUR))
    print("lissage des degagements : {:.0f} %".format(LISSAGE))
    print("orientation a la depose : {:+.0f} deg".format(ANGLE))
    print("")
    print("prise  : x={:6.1f}  y={:6.1f}  z={:5.1f}   a   +0 deg"
          .format(*PRISE[:3, 3]))
    print("depose : x={:6.1f}  y={:6.1f}  z={:5.1f}   a {:+.0f} deg"
          .format(DEPOSE[0, 3], DEPOSE[1, 3], DEPOSE[2, 3], ANGLE))
    print("")
    print(robot.resume().splitlines()[0])

    avertissements = robot.controler()
    if avertissements:
        print("")
        print("reserves :")
        for a in avertissements[:8]:
            print("  - " + a)
    else:
        print("tout le cycle est dans l'espace de travail")

    if REEL:
        robot.executer_sur_robot(vitesse=30)
    else:
        robot.simuler()
