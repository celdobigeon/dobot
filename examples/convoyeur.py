#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le convoyeur : tapis, capteur de presence, et prise au vol.

    python examples/convoyeur.py              simulation
    python examples/convoyeur.py --reel       sur la machine

Le cycle, dix fois : le tapis avance, le capteur voit arriver une piece,
le tapis s'arrete, le robot la prend et la depose a cote, le tapis
repart.

Le tapis sur la table
---------------------
`repere` place le tapis comme un repere de travail : son origine est son
ENTREE, son axe x suit le sens de marche, son axe y traverse la bande.
Une piece a l'abscisse s se trouve donc en `TAPIS.cible(s, 0, z)`, et
deplacer le tapis ne demande de changer que cette ligne.

Le pas du tapis
---------------
`impulsions_par_mm` vaut par defaut celui du kit Dobot -- 282,9, pour un
moteur de 200 pas, 16 micro-pas, un reducteur 10 et une poulie de 36 mm.
C'est une valeur NOMINALE. Mesurez la votre : marquez la bande, lancez
le tapis une duree connue, mesurez le deplacement, et corrigez ici. Le
nombre voyage ensuite avec le programme jusqu'au robot.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import Convoyeur, Pilote, cible, repere_travail   # noqa: E402
from dobot_sim import homogeneous as hg                          # noqa: E402

REEL = "--reel" in sys.argv

# ---------------------------------------------------------------------------
# 1. LA CELLULE
# ---------------------------------------------------------------------------

# Le tapis, pose le long de l'axe y du robot : il entre a (180 ; -180) et
# s'eloigne vers les y positifs. rz=90 fait tourner son repere d'un quart
# de tour, pour que son axe x -- le sens de marche -- suive +y.
# Le capteur est place la ou le BRAS ATTEINT la bande : plus loin, la
# piece arrive hors de portee. controler() le dit, mais autant poser le
# capteur au bon endroit.
TAPIS = Convoyeur(
    repere=repere_travail(x=170, y=-230, z=0, rz=90),
    longueur=600,          # mm, de l'entree a la sortie
    largeur=60,
    index_moteur=0,        # 0 = Stepper1, 1 = Stepper2
    capteur_s=380,         # abscisse du capteur le long du tapis
    camera_s=250,          # abscisse du champ de la camera
    capteur_eio=5,         # capteur tout-ou-rien sur l'entree EIO 5
)

PRISE = TAPIS.cible(380, 0, 5)             # au droit du capteur, sur la bande
DEPOSE = cible(x=250, y=-60, z=10)         # a cote, sur la table
HAUT = hg.translation(0, 0, -70)           # degagement, repere de l'OUTIL

VITESSE_TAPIS = 60                         # mm/s

# ---------------------------------------------------------------------------
# 2. LE PROGRAMME
# ---------------------------------------------------------------------------

robot = Pilote(nom="convoyeur", outil="ventouse",
               vitesse=140, acceleration=600, vitesse_articulaire=60)
robot.tapis = TAPIS                        # sert de defaut aux commandes
robot.regler_longueur_outil(79.3)          # mesuree sur la machine

robot.repli()
for i in range(10):
    # le tapis amene la piece, le capteur l'arrete
    robot.convoyeur(VITESSE_TAPIS, commentaire="le tapis amene")
    robot.attendre_piece(TAPIS.capteur_eio, timeout=30,
                         commentaire="piece au capteur")
    robot.convoyeur(0, commentaire="arret sous le bras")

    # prise : ARRET AU POINT, sinon la ventouse aspire avant d'y etre
    robot.movj(PRISE @ HAUT, lissage=70)
    robot.movl(PRISE, vitesse=50)
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(PRISE @ HAUT, lissage=80)

    # depose a cote
    robot.movj(DEPOSE @ HAUT, lissage=70)
    robot.movl(DEPOSE, vitesse=50)
    robot.ventouse(False)
    robot.attendre(0.3)
    robot.movl(DEPOSE @ HAUT)
robot.repli()
robot.convoyeur(0, commentaire="fin de cycle")

# ---------------------------------------------------------------------------
# 3. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

print(TAPIS.resume())
print()
print(robot.resume().splitlines()[0])

# Tous les avertissements ne se valent pas : une butee ou une collision
# empechent le cycle, une survitesse ne fait que meriter un coup d'oeil.
# C'est le partage que le pont fait avant de televerser.
from dobot_sim.reel import BLOQUANTS_TELEVERSEMENT       # noqa: E402

avertissements = robot.controler()
bloquants = [a for a in avertissements
             if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
reserves = [a for a in avertissements if a not in bloquants]

if bloquants:
    print("\nCYCLE IMPOSSIBLE :")
    for a in bloquants[:6]:
        print("  -", a)
    print("\nDeplacez le tapis ou revoyez les cibles : le bras n'atteint")
    print("pas tout le cycle depuis cette position.")
    sys.exit(1)
if reserves:
    print("\nreserves (le cycle passe, mais) :")
    for a in reserves[:4]:
        print("  -", a)
else:
    print("Tout le cycle est dans l'espace de travail.")

# Ce que le tapis fait pendant ce temps : on rejoue sa cadence pour
# verifier qu'une piece posee a l'entree arrive bien au capteur.
TAPIS.regler(0, 0.0).regler(VITESSE_TAPIS, 0.0)
piece = TAPIS.poser(s=0, t0=0.0, nom="bloc")
quand = TAPIS.instant_au_capteur(piece)
if quand is not None:
    print("une piece posee a l'entree atteint le capteur en {:.1f} s "
          "({:.0f} mm a {:.0f} mm/s)".format(quand, TAPIS.capteur_s,
                                             VITESSE_TAPIS))

if REEL:
    print("\nLE ROBOT ET LE TAPIS VONT BOUGER.")
    robot.executer_sur_robot(vitesse=30)
else:
    robot.simuler()
