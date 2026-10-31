#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 - Le simulateur ET le robot en meme temps, simulateur lance par le script.

    python examples/controle-temps-reel/08_simulateur_et_robot.py [les-deux|simulateur]

Un port serie ne s'ouvre qu'une fois : quand on veut voir le robot dans le
simulateur PENDANT qu'il bouge, c'est la FENETRE qui tient la liaison. Le
script, lui :

  1. s'assure qu'un simulateur tourne -- avec `lancer_simulateur=True`, il
     ouvre main.py dans un AUTRE processus s'il n'y en a pas ;
  2. lui demande de se connecter au robot (port trouve tout seul, ou
     `port="COM4"`, `port="/dev/ttyUSB0"`...). Si le robot est en defaut --
     c'est le cas apres chaque mise sous tension --, l'assistant de mise en
     route s'ouvre DANS LA FENETRE : suivez-le, le script attend ;
  3. envoie chaque commande a la fenetre, qui l'execute sur le robot ; le
     modele 3D suit le vrai bras (miroir), et le script attend la fin du
     ROBOT.

Fermer le script ne ferme pas la fenetre : relancez-le autant de fois que
vous voulez, il la retrouve.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dobot_sim import Commande, cible  # noqa: E402

CIBLE = "les-deux"                       # "les-deux", ou "simulateur" pour essayer
if len(sys.argv) > 1:
    CIBLE = sys.argv[1]
if CIBLE not in ("simulateur", "les-deux"):
    sys.exit("ici : les-deux, ou simulateur (le robot seul se passe de fenetre)")

A = cible(220, -50, 80)
B = cible(220, 50, 80)

print("ouverture (ou recherche) du simulateur...")
with Commande(simulateur=True, robot=(CIBLE == "les-deux"),
              lancer_simulateur=True,   # ouvre main.py s'il n'est pas ouvert
              port=None,                # None : premier port serie trouve
              vitesse=40) as robot:
    print("pret. Pose lue :", ["{:.1f}".format(v) for v in robot.pose()])

    input("\nAller-retour entre A et B : la fenetre suit le robot.   [Entree] ")
    robot.movj(A)
    robot.movl([B, A])

    input("\nRepli.   [Entree] ")
    robot.repli()

print("script fini ; la fenetre reste ouverte{}".format(
    ", et garde la liaison avec le robot" if CIBLE == "les-deux" else ""))
