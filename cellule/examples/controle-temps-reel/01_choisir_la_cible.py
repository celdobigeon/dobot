#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 - Choisir ce qui bouge, et lire ou en est le bras.

    python examples/controle-temps-reel/01_choisir_la_cible.py              simulateur
    python examples/controle-temps-reel/01_choisir_la_cible.py robot        robot seul
    python examples/controle-temps-reel/01_choisir_la_cible.py les-deux     les deux

AUCUN MOUVEMENT dans ce script : il ne fait que LIRE.

Une `Commande` parle a l'une de ces trois "cibles" :

  * "simulateur" : la fenetre du simulateur, deja ouverte (python main.py).
    Le script lui envoie chaque commande, elle l'anime ;
  * "robot"      : le vrai Dobot, branche en USB. Le script ouvre lui-meme
    le port serie. Le simulateur doit etre FERME (un port serie ne
    s'ouvre qu'une fois) ;
  * "les-deux"   : le simulateur ouvert tient la liaison avec le robot ; le
    script passe par lui. Le modele 3D suit alors le vrai bras (miroir).

Les lectures sont toujours faites sur la machine : `pose()` et `angles()`
disent ou EST le bras, pas ou le script croit qu'il est.
"""

import os
import sys

# le paquet dobot_sim est trois dossiers plus haut
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dobot_sim import Commande  # noqa: E402

# -- choix de la cible : une constante, ou un argument ---------------------
CIBLE = "simulateur"                     # "simulateur", "robot" ou "les-deux"
if len(sys.argv) > 1:
    CIBLE = sys.argv[1]
if CIBLE not in ("simulateur", "robot", "les-deux"):
    sys.exit("cible inconnue : {} (simulateur, robot ou les-deux)".format(CIBLE))

print("cible choisie :", CIBLE)

# `with` garantit que la liaison est fermee a la fin, meme apres une erreur
with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux")) as robot:
    while True:
        # pose() : (x, y, z, r) de la POINTE de la ventouse, en mm et degres,
        # dans le repere du simulateur (celui du controleur est converti)
        x, y, z, r = robot.pose()
        # angles() : J1 a J4, en degres -- les memes pour le robot et le modele
        j1, j2, j3, j4 = robot.angles()
        print("pose    x = {:7.1f}   y = {:7.1f}   z = {:7.1f} mm   r = {:6.1f} deg"
              .format(x, y, z, r))
        print("angles  J1 = {:6.1f}   J2 = {:6.1f}   J3 = {:6.1f}   J4 = {:6.1f} deg"
              .format(j1, j2, j3, j4))
        # Dans le simulateur, bougez un curseur puis relisez : la pose suit.
        if input("Entree pour relire, q pour quitter : ").strip().lower() == "q":
            break
