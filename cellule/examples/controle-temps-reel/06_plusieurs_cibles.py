#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 - Plusieurs cibles d'un coup, avec leurs durees : continu ou pas.

    python examples/controle-temps-reel/06_plusieurs_cibles.py [simulateur|robot|les-deux]

Un seul appel peut enchainer plusieurs cibles. Chaque element de la liste
est une cible, ou un couple (cible, duree en secondes) :

    robot.movl([(A, 1.5), (B, 1.0), C])     C : duree deduite de la vitesse

Le meme carre est parcouru deux fois :

  * continu=False : POINT A POINT, le robot s'arrete a chaque coin ;
  * continu=True  : TRAJECTOIRE CONTINUE, il enchaine sans s'arreter, et
    chaque cote est parcouru a la vitesse qui tient sa duree.

Avant de partir, la commande PREVOIT si chaque duree est tenable : un cote
trop court pour sa duree, ou un coin trop vif, donne un avertissement
jaune avec la duree realiste. Le chronometre affiche ensuite la duree
reelle. Sur le robot, regardez (et ecoutez) les coins.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dobot_sim import Commande, cible  # noqa: E402

CIBLE = "simulateur"                     # "simulateur", "robot" ou "les-deux"
if len(sys.argv) > 1:
    CIBLE = sys.argv[1]
if CIBLE not in ("simulateur", "robot", "les-deux"):
    sys.exit("cible inconnue : {} (simulateur, robot ou les-deux)".format(CIBLE))

# un carre de 100 x 70 mm, a 80 mm au-dessus de la table
COINS = [cible(190, -50, 80), cible(190, 50, 80),
         cible(260, 50, 80), cible(260, -50, 80)]
# (cible, duree) : 100 mm en 2 s, 70 mm en 1,5 s...
CARRE = [(COINS[1], 2.0), (COINS[2], 1.5), (COINS[3], 2.0), (COINS[0], 1.5)]
DUREE_DEMANDEE = sum(d for _c, d in CARRE)

with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"), vitesse=40) as robot:

    input("\nLe robot va au premier coin.   [Entree] ")
    robot.movj(COINS[0])

    for continu in (False, True):
        input("\nTour du carre, continu={}.   [Entree] ".format(continu))
        t = time.time()
        robot.movl(CARRE, continu=continu)
        print("   {} : {:.2f} s pour {:.2f} s demandees".format(
            "continu" if continu else "point a point",
            time.time() - t, DUREE_DEMANDEE))

    # Sans durees, chaque cote se parcourt a la vitesse de la commande
    input("\nMeme carre, sans durees, a 60 mm/s, continu.   [Entree] ")
    t = time.time()
    robot.movl(COINS[1:] + COINS[:1], vitesse=60)
    print("   {:.2f} s".format(time.time() - t))

    # Une duree impossible : 100 mm en 0,5 s. L'avertissement jaune donne
    # la duree realiste, et le robot fait au mieux.
    input("\nUne duree trop courte, pour voir l'avertissement.   [Entree] ")
    robot.movl([(COINS[1], 0.5), COINS[2], COINS[3], COINS[0]])
