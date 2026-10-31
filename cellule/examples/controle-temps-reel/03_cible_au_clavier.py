#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 - Une cible tapee au clavier : le robot y va.

    python examples/controle-temps-reel/03_cible_au_clavier.py [simulateur|robot|les-deux]

Tapez trois nombres, "x y z" (en mm, separes par des espaces), puis
Entree : le robot y va en ligne droite. Un quatrieme nombre donne
l'orientation r de l'outil (en degres). "q" pour quitter.

La ZONE de travail est activee : une cible en dehors est REFUSEE, et le
robot ne bouge pas. Une cible hors d'atteinte, ou qui ferait toucher la
table, l'est aussi. Le refus leve `ErreurExecution` : on l'attrape pour
afficher un message et continuer, au lieu d'arreter le script.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dobot_sim import Commande, ErreurExecution, cible  # noqa: E402

CIBLE = "simulateur"                     # "simulateur", "robot" ou "les-deux"
if len(sys.argv) > 1:
    CIBLE = sys.argv[1]
if CIBLE not in ("simulateur", "robot", "les-deux"):
    sys.exit("cible inconnue : {} (simulateur, robot ou les-deux)".format(CIBLE))

# ((xmin, xmax), (ymin, ymax), (zmin, zmax)) en mm : une boite devant le
# robot, au-dessus de la table. Adaptez-la a votre poste.
ZONE = ((170, 270), (-90, 90), (30, 140))

print(__doc__)
print("zone autorisee : x de {} a {}, y de {} a {}, z de {} a {}"
      .format(*ZONE[0], *ZONE[1], *ZONE[2]))
print("exemple : 220 0 80")

with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"),
              vitesse=40, zone=ZONE) as robot:
    while True:
        ligne = input("\nx y z [r] > ").strip()
        if ligne.lower() == "q":
            break
        if not ligne:
            continue
        # une virgule decimale est acceptee : "220,5 0 80"
        try:
            valeurs = [float(v.replace(",", ".")) for v in ligne.split()]
        except ValueError:
            print("   tapez des nombres, par exemple : 220 0 80")
            continue
        if len(valeurs) not in (3, 4):
            print("   il faut 3 nombres (x y z) ou 4 (x y z r)")
            continue

        try:
            robot.movl(cible(*valeurs))      # rend la main a l'arrivee
        except ErreurExecution:
            # l'alerte rouge au-dessus dit pourquoi ; le robot n'a pas bouge
            print("   -> cible refusee, le robot n'a pas bouge. Essayez-en une autre.")
            continue
        x, y, z, r = robot.pose()
        print("   -> arrive : x={:.1f} y={:.1f} z={:.1f} r={:.1f}".format(x, y, z, r))
