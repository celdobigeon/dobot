#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 - Deplacer le robot pas a pas, au clavier.

    python examples/controle-temps-reel/04_pas_a_pas_clavier.py [simulateur|robot|les-deux]

Chaque ligne tapee (puis Entree) fait un petit deplacement :

    x+  x-  y+  y-  z+  z-     avancer d'un pas dans cette direction
    pas 5                      changer le pas (en mm)
    Entree seule               refaire le dernier deplacement
    r                          retour au repli (pas_max ne s'y applique pas)
    q                          quitter

Deux garde-fous : la zone de travail, et `pas_max`, qui refuse tout
deplacement de plus de 30 mm d'un coup -- une faute de frappe dans le pas
("pas 500") ne lance pas le robot a l'autre bout de la table.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dobot_sim import Commande, ErreurExecution  # noqa: E402

CIBLE = "simulateur"                     # "simulateur", "robot" ou "les-deux"
if len(sys.argv) > 1:
    CIBLE = sys.argv[1]
if CIBLE not in ("simulateur", "robot", "les-deux"):
    sys.exit("cible inconnue : {} (simulateur, robot ou les-deux)".format(CIBLE))

ZONE = ((170, 270), (-90, 90), (30, 140))

# direction de chaque touche : (dx, dy, dz)
DIRECTIONS = {
    "x+": (1, 0, 0), "x-": (-1, 0, 0),
    "y+": (0, 1, 0), "y-": (0, -1, 0),
    "z+": (0, 0, 1), "z-": (0, 0, -1),
}

print(__doc__)

with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"),
              vitesse=30, zone=ZONE, pas_max=30) as robot:
    pas = 10.0                                   # mm
    dernier = None
    while True:
        x, y, z, r = robot.pose()
        ligne = input("\n[pas {:.0f} mm]  x={:.1f} y={:.1f} z={:.1f} > "
                      .format(pas, x, y, z)).strip().lower()
        if ligne == "q":
            break
        if ligne == "":
            ligne = dernier or ""                # Entree seule : on refait
        if ligne.startswith("pas"):
            try:
                pas = float(ligne.split()[1].replace(",", "."))
            except (IndexError, ValueError):
                print("   exemple : pas 5")
            continue
        if ligne == "r":
            try:
                robot.repli()
            except ErreurExecution:
                print("   -> repli refuse")
            continue
        if ligne not in DIRECTIONS:
            print("   touches : x+ x- y+ y- z+ z-, pas N, r, q")
            continue

        dx, dy, dz = DIRECTIONS[ligne]
        try:
            # la nouvelle cible part de la pose LUE ; r n'est pas donne :
            # l'outil garde son orientation
            robot.movl(x=x + dx * pas, y=y + dy * pas, z=z + dz * pas)
            dernier = ligne
        except ErreurExecution:
            print("   -> deplacement refuse, le robot n'a pas bouge")
