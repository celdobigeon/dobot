#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09 - Les garde-fous : zone, pas maximal, refus, Ctrl+C.

    python examples/controle-temps-reel/09_garde_fous.py [simulateur|robot|les-deux]

Une commande "temps reel" part tout de suite : il n'y a pas de programme a
relire avant. D'ou des garde-fous, verifies AVANT chaque mouvement, depuis
la posture LUE sur la machine :

  * le robot peut-il atteindre la cible (MGI) ? sans sortir de ses butees ?
    sans toucher la table, son embase ou lui-meme, sur TOUT le chemin ?
  * `zone=` : la cible, et le trajet, restent-ils dans la boite autorisee ?
  * `pas_max=` : la cible est-elle a moins de N mm de la precedente ?

Un refus affiche une alerte rouge et leve `ErreurExecution` : RIEN n'est
parti. Un avertissement (jaune) laisse partir le mouvement.

Ctrl+C pendant un mouvement : ARRET D'URGENCE, puis le script s'arrete.
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

ZONE = ((170, 270), (-90, 90), (30, 140))


def essayer(texte, mouvement):
    """Tente un mouvement ; un refus est affiche, le script continue."""
    input("\n" + texte + "   [Entree] ")
    try:
        mouvement()
        print("   -> accepte et execute")
    except ErreurExecution as exc:
        print("   -> REFUSE ({}) : le robot n'a pas bouge".format(exc))


with Commande(simulateur=CIBLE in ("simulateur", "les-deux"),
              robot=CIBLE in ("robot", "les-deux"),
              vitesse=40, zone=ZONE, pas_max=100) as robot:

    essayer("1. Une cible normale : (220, 0, 80)",
            lambda: robot.movj(cible(220, 0, 80)))
    # (170, 0, 140) est dans la zone, mais le bras arriere devrait passer
    # derriere la verticale : J2 < 0, hors de ses butees
    essayer("2. Dans la zone, mais hors des butees : (170, 0, 140)",
            lambda: robot.movl(cible(170, 0, 140)))
    # Les regles du script sont verifiees en premier : la zone refuse ce
    # point avant meme qu'on cherche a savoir s'il est atteignable
    essayer("3. Hors d'atteinte, et hors de la zone : (400, 0, 80)",
            lambda: robot.movl(cible(400, 0, 80)))
    essayer("4. Hors de la zone : y = 120",
            lambda: robot.movl(cible(220, 120, 80)))
    essayer("5. Un pas de 50 mm (accepte), vers (220, -50, 80)",
            lambda: robot.movl(cible(220, -50, 80)))
    essayer("6. Trop loin d'un coup : 120 mm, pour pas_max = 100",
            lambda: robot.movl(cible(220, 70, 80)))
    # Cercle qui passe par les trois points : son milieu atteint x = 280,
    # au-dela du bord de la zone (270), alors que les bouts y sont.
    essayer("7. Un arc dont les bouts sont dans la zone, mais pas le milieu",
            lambda: robot.movc(cible(255, -45, 80), cible(220, 45, 80)))

    # Ctrl+C : 90 mm en 12 s, lentement ; interrompez quand vous voulez.
    input("\n8. Mouvement lent de 12 s : appuyez sur Ctrl+C PENDANT qu'il "
          "bouge.   [Entree] ")
    try:
        robot.movl(cible(220, 40, 80), duree=12.0)
        print("   -> pas interrompu : le mouvement est alle au bout")
    except KeyboardInterrupt:
        # L'arret d'urgence est deja parti (alerte rouge au-dessus) : on
        # s'arrete la, le bras n'est pas ou le script le croit.
        print("   -> interrompu. Verifiez la posture avant de relancer.")
        sys.exit(1)
