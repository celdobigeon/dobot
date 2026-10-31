#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 - La ventouse, les sorties et les entrees.

    python examples/controle-temps-reel/05_ventouse_et_es.py [simulateur|robot|les-deux]

Une prise et depose pas a pas : chaque etape attend Entree. Puis une
sortie tout-ou-rien qu'on allume et eteint, et une entree qu'on lit.

Dans le SIMULATEUR, l'entree se bascule a la main dans le panneau
Entrees / Sorties : cliquez sur EIO17, puis relisez.

Sur le ROBOT, verifiez d'abord ce qui est branche sur les ports EIO
utilises ci-dessous : une sortie peut commander un verin ou un convoyeur.
La ventouse, elle, n'est pas sur ces ports : elle a sa propre commande.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dobot_sim import Commande, cible  # noqa: E402
from dobot_sim import homogeneous as hg  # noqa: E402

CIBLE = "simulateur"                     # "simulateur", "robot" ou "les-deux"
if len(sys.argv) > 1:
    CIBLE = sys.argv[1]
if CIBLE not in ("simulateur", "robot", "les-deux"):
    sys.exit("cible inconnue : {} (simulateur, robot ou les-deux)".format(CIBLE))

PORT_SORTIE = 18                         # a adapter a votre cellule
PORT_ENTREE = 17

PRISE = cible(220, 40, 30)               # la piece, 30 mm au-dessus de la table
DEPOSE = cible(220, -40, 30)
# 50 mm au-dessus, selon l'axe de l'OUTIL : son z pointe vers le bas, monter
# s'ecrit donc avec un z negatif
HAUT = hg.translation(0, 0, -50)


def etape(texte):
    input("\n" + texte + "   [Entree] ")


with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"),
              vitesse=40, outil="ventouse") as robot:

    etape("1. au-dessus de la piece")
    robot.movj(PRISE @ HAUT)
    etape("2. descente, puis la ventouse aspire")
    robot.movl(PRISE)
    robot.ventouse(True)                 # rend la main une fois la commande faite
    etape("3. on remonte, on va au-dessus de la depose")
    robot.movl(PRISE @ HAUT)
    robot.movj(DEPOSE @ HAUT)
    etape("4. descente, la ventouse relache, on remonte")
    robot.movl(DEPOSE)
    # Relacher fait SOUFFLER la pompe, qui continuerait de tourner (et de
    # faire du bruit) : couper=True l'arrete 1 s apres, sans bloquer le
    # script -- la remontee part tout de suite. Avec une pince
    # (outil="pince"), pince(False) le fait d'office.
    robot.ventouse(False, couper=True)
    robot.movl(DEPOSE @ HAUT)

    etape("5. sortie EIO{} allumee".format(PORT_SORTIE))
    robot.sortie(PORT_SORTIE, True)
    etape("6. sortie EIO{} eteinte".format(PORT_SORTIE))
    robot.sortie(PORT_SORTIE, False)

    print("\n7. lecture de l'entree EIO{} (q pour finir)".format(PORT_ENTREE))
    while input("   Entree pour lire, q pour finir : ").strip().lower() != "q":
        print("   EIO{} = {}".format(PORT_ENTREE,
                                     1 if robot.entree(PORT_ENTREE) else 0))
