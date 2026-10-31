#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 - Mouvements simples : chaque appel attend la fin du mouvement.

    python examples/controle-temps-reel/02_mouvements_simples.py [simulateur|robot|les-deux]

Avec `Pilote`, on DECRIT un programme, puis on le joue d'un bloc. Avec
`Commande`, chaque ligne bouge le robot TOUT DE SUITE, et ne rend la main
qu'une fois le mouvement FINI. On peut donc lire la pose juste apres, et
decider de la suite.

Chaque etape attend que vous tapiez Entree : rien ne bouge sans vous.
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

# Des cibles sures pour la ventouse (62 mm) : verifiees avec le modele,
# loin des butees et de la table. Une cible est une matrice homogene 4x4.
AU_DESSUS = cible(220, 0, 120)
A = cible(220, -50, 80)
B = cible(220, 50, 80)


def etape(texte):
    """Rien ne bouge tant que l'utilisateur n'a pas tape Entree."""
    input("\n" + texte + "   [Entree] ")


def afficher(robot, debut):
    x, y, z, r = robot.pose()
    print("   fini en {:.2f} s -> pose lue : x={:.1f} y={:.1f} z={:.1f} r={:.1f}"
          .format(time.time() - debut, x, y, z, r))


# vitesse en mm/s pour les mouvements cartesiens (movl, movc),
# vitesse_articulaire en deg/s pour movj
with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"),
              vitesse=40, vitesse_articulaire=40) as robot:

    etape("1. movj : au-dessus du centre (les angles sont interpoles)")
    t = time.time()
    robot.movj(AU_DESSUS)            # le script attend ICI la fin du mouvement
    afficher(robot, t)

    etape("2. movl : vers A, en ligne droite, a 40 mm/s")
    t = time.time()
    robot.movl(A)
    afficher(robot, t)

    etape("3. movl : vers B en 2 s exactement (duree imposee)")
    t = time.time()
    robot.movl(B, duree=2.0)
    afficher(robot, t)

    etape("4. movl(z=50) : descente verticale, x, y et r sont gardes")
    t = time.time()
    robot.movl(z=50)                 # composantes omises : celles de B
    afficher(robot, t)

    etape("5. repli : retour a la position de repos")
    t = time.time()
    robot.repli()
    afficher(robot, t)

print("\ntermine")
