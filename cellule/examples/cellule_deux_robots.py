#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deux robots dans la meme cellule : chacun son cycle, puis rendez-vous.

    python examples/cellule_deux_robots.py                 controle seul
    python examples/cellule_deux_robots.py --reel COM4 COM6

Deux facons de travailler a plusieurs, et la difference compte.

CHACUN SON CYCLE. Les robots sont independants : on leur donne a chacun
son programme, ils partent, on attend qu'ils aient tous fini. C'est le
cas courant -- deux postes en parallele.

RENDEZ-VOUS. Les bras cooperent : on decoupe en ETAPES, et personne ne
passe a la suivante avant que tous aient fini la sienne.

Ce que la synchronisation vaut
------------------------------
La file du controleur n'a AUCUNE synchronisation entre machines : il
n'existe pas de commande « pars quand l'autre part ». Tout se fait cote
PC, ce qui borne la precision :

  - le depart est decale du temps d'envoi d'une trame par robot, soit
    une dizaine de millisecondes. A 100 mm/s, cela fait un millimetre.
    Le programme AFFICHE le decalage mesure, plutot que de le promettre ;
  - a l'interieur d'une etape, les bras derivent librement. Le
    rendez-vous est a la FIN de l'etape, pas en continu.

Decoupez donc en etapes la ou la coordination compte.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import Cellule, ErreurCellule, cible                # noqa: E402
from dobot_sim import homogeneous as hg                            # noqa: E402

REEL = "--reel" in sys.argv
PORTS = [a for a in sys.argv[1:] if a.upper().startswith("COM")
         or a.startswith("/dev/")]

HAUT = hg.translation(0, 0, -60)        # degagement, repere de l'OUTIL

# ---------------------------------------------------------------------------
# 1. LA CELLULE
# ---------------------------------------------------------------------------

cellule = Cellule(journal=print)

alpha = cellule.ajouter("alpha", port=PORTS[0] if PORTS else "",
                        outil="ventouse", vitesse=120, acceleration=400,
                        vitesse_articulaire=60)
beta = cellule.ajouter("beta", port=PORTS[1] if len(PORTS) > 1 else "",
                       outil="pince", vitesse=120, acceleration=400,
                       vitesse_articulaire=60)

alpha.regler_longueur_outil(79.3)       # ventouse mesuree
beta.regler_longueur_outil(107.0)       # pince mesuree

# Chaque robot travaille dans SON repere. Ils sont cote a cote, mais
# chacun raisonne depuis sa propre base : c'est a l'atelier de dire ou
# les postes se trouvent l'un par rapport a l'autre.
POSTE_A = cible(x=220, y=-60, z=20)
POSTE_B = cible(x=220, y=60, z=20)
ECHANGE_A = cible(x=250, y=0, z=40)
ECHANGE_B = cible(x=250, y=0, z=40)

# ---------------------------------------------------------------------------
# 2. CHACUN SON CYCLE
# ---------------------------------------------------------------------------

print("=" * 64)
print("CHACUN SON CYCLE -- les deux travaillent en parallele")
print("=" * 64)

for _ in range(3):
    alpha.movj(POSTE_A @ HAUT, lissage=70)
    alpha.movl(POSTE_A, vitesse=50)
    alpha.ventouse(True)
    alpha.attendre(0.3)
    alpha.movl(POSTE_A @ HAUT, lissage=70)

for _ in range(2):
    beta.movj(POSTE_B @ HAUT, lissage=70)
    beta.movl(POSTE_B, vitesse=50)
    beta.pince(True)
    beta.attendre(0.3)
    beta.movl(POSTE_B @ HAUT, lissage=70)

defauts = cellule.controler()
if defauts:
    print("\nCYCLE REFUSE :")
    for nom, liste in defauts.items():
        print("  {} : {}".format(nom, liste[0]))
    sys.exit(1)
print(cellule.resume())
print("alpha : {} instructions, {:.1f} s"
      .format(len(alpha.programme.instructions), alpha.trajectoire().duree))
print("beta  : {} instructions, {:.1f} s"
      .format(len(beta.programme.instructions), beta.trajectoire().duree))

if not REEL:
    print("\n(simulation : --reel COM4 COM6 pour lancer les machines)")

# ---------------------------------------------------------------------------
# 3. RENDEZ-VOUS
# ---------------------------------------------------------------------------

if REEL:
    if len(PORTS) < 2:
        sys.exit("donnez les deux ports : --reel COM4 COM6")
    with cellule:
        cellule.connecter()
        cellule.definir_ratio(30)

        # les cycles independants prepares ci-dessus
        decalage = cellule.lancer()
        print("\ndepart groupe a {:.0f} ms pres".format(1000 * decalage))
        cellule.attendre_tous()
        print("les deux cycles sont finis")

        # puis une coordination, etape par etape
        print("\n" + "=" * 64)
        print("RENDEZ-VOUS -- personne ne passe a la suite avant les autres")
        print("=" * 64)
        for poste in (alpha, beta):
            poste.vider_programme()

        alpha.movj(ECHANGE_A @ HAUT, lissage=70)
        beta.movj(ECHANGE_B @ HAUT, lissage=70)
        print("etape 1 : les deux montent au point d'echange, "
              "depart a {:.0f} ms pres".format(1000 * cellule.etape()))

        alpha.movl(ECHANGE_A, vitesse=40)
        print("etape 2 : alpha descend, beta attend -- "
              "{:.0f} ms".format(1000 * cellule.etape()))

        alpha.ventouse(False)
        beta.pince(True)
        print("etape 3 : alpha lache, beta serre -- "
              "{:.0f} ms".format(1000 * cellule.etape()))

        alpha.movl(ECHANGE_A @ HAUT)
        beta.movj(POSTE_B @ HAUT)
        print("etape 4 : chacun se degage -- "
              "{:.0f} ms".format(1000 * cellule.etape()))

        for poste in (alpha, beta):
            poste.vider_programme()
            poste.repli()
        cellule.etape()
        print("\ncellule au repos")
