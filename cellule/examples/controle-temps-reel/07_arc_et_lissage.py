#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 - Arcs de cercle (movc) et coins arrondis (lissage).

    python examples/controle-temps-reel/07_arc_et_lissage.py [simulateur|robot|les-deux]

movc(via, fin) : un arc qui part de la position actuelle, passe par `via`
et finit en `fin`. Deux demi-cercles font un cercle :

    robot.movc([(via1, fin1, 2.5), (via2, fin2, 2.5)])     (via, fin, duree)

Le robot suit l'arc en trajectoire continue : des points tous les 5 mm,
parcourus sans arret. Dans une courbe serree, il ralentit -- c'est mesure,
et l'avertissement jaune le dit.

lissage=N (0 a 100) : le robot "coupe le coin" au lieu de passer
exactement par la cible. Plus N est grand, plus le coin est arrondi, et
moins le robot ralentit. Le passage n'est plus exact : c'est le prix.
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

# cercle de centre (220, 0), de rayon 50 mm, a z = 80
HAUT_CERCLE = cible(220, 50, 80)
BAS_CERCLE = cible(220, -50, 80)
DROITE = cible(270, 0, 80)
GAUCHE = cible(170, 0, 80)

# un zigzag pour le lissage
ZIGZAG = [cible(200, -40, 80), cible(240, -20, 80), cible(200, 0, 80),
          cible(240, 20, 80), cible(200, 40, 80)]

with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"), vitesse=40) as robot:

    input("\nAu depart du cercle.   [Entree] ")
    robot.movj(BAS_CERCLE)

    input("\nUn demi-cercle par la droite, en 3 s.   [Entree] ")
    robot.movc(DROITE, HAUT_CERCLE, duree=3.0)

    input("\nLe cercle complet, deux demi-cercles de 2,5 s chacun.   [Entree] ")
    t = time.time()
    robot.movc([(GAUCHE, BAS_CERCLE, 2.5), (DROITE, HAUT_CERCLE, 2.5)])
    print("   {:.2f} s pour 5,00 demandees".format(time.time() - t))

    input("\nAu depart du zigzag.   [Entree] ")
    robot.movl(ZIGZAG[0])
    for lissage in (0, 60):
        input("\nZigzag, lissage={} (et retour).   [Entree] ".format(lissage))
        t = time.time()
        robot.movl(ZIGZAG[1:], lissage=lissage, vitesse=50)
        print("   aller en {:.2f} s".format(time.time() - t))
        robot.movl(ZIGZAG[-2::-1], lissage=lissage, vitesse=50)
