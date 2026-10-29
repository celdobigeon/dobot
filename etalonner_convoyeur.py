#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apprendre au robot OU prendre la piece sur le tapis.

    python etalonner_convoyeur.py COM6

Pourquoi apprendre plutot que mesurer
-------------------------------------
Le point de prise peut se CALCULER : on releve la place du tapis, son
orientation, l'abscisse du capteur, on compose les matrices, et on
obtient une cible. C'est ce que fait l'exemple -- avec des cotes
plausibles mais inventees, ce qui envoie le bras a cent millimetres de
la piece.

On peut aussi l'APPRENDRE : on arrete une piece au capteur, on amene la
ventouse dessus a la main, et on demande au robot ou il est. Le point
obtenu absorbe d'un coup tout ce qu'on ne sait pas mesurer proprement :

  - la place et l'orientation reelles du tapis ;
  - l'abscisse reelle du capteur le long de la bande ;
  - le retard de la chaine de scrutation, quel qu'il soit ;
  - le decalage entre le BORD de la piece, que le capteur voit, et son
    CENTRE, que la ventouse doit viser ;
  - la hauteur de la bande et l'ecrasement de la ventouse.

Aucune de ces grandeurs n'a besoin d'etre connue separement. C'est tout
l'interet, et c'est ainsi qu'on procede sur une vraie cellule.

Ce que le script NE fait pas
----------------------------
Il n'apprend que la PRISE. La depose reste a vous : elle ne depend pas
du tapis, et son endroit est un choix d'implantation, pas une mesure.

Le bras ne bouge QUE par votre main : aucun mouvement n'est commande.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_sim import Convoyeur, Pilote                        # noqa: E402
from dobot_sim.convoyeur import PORTS_GP                       # noqa: E402

ESSAIS = 3
# La MEME que celle du cycle : le depassement du tapis depend de
# la vitesse, donc un point appris ne vaut que pour elle.
VITESSE_TAPIS = 25.0
FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "examples", "convoyeur_etalonnage.json")


def demander(message: str) -> str:
    try:
        return input(message).strip()
    except EOFError:
        return ""


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None

    tapis = Convoyeur(longueur=500, index_moteur=0,
                      capteur_port=PORTS_GP["GP4"])
    robot = Pilote(nom="etalonnage", outil="ventouse", vitesse=120)
    robot.tapis = tapis
    robot.regler_longueur_outil(79.3)

    print(__doc__.split("Ce que le script NE fait pas")[0].strip())
    print("\n" + "=" * 66)
    print("{} essais. A chaque fois :".format(ESSAIS))
    print("  1. le tapis part, vous posez une piece BIEN EN AMONT ;")
    print("  2. le capteur la voit, le tapis s'arrete ;")
    print("  3. vous amenez la ventouse au contact du centre de la piece,")
    print("     a la main -- bouton de la fourche -- puis Entree.")
    print("=" * 66)

    poses = []
    for essai in range(1, ESSAIS + 1):
        demander("\nessai {}/{} - Entree pour lancer le tapis... "
                 .format(essai, ESSAIS))
        robot.convoyeur_sur_robot(VITESSE_TAPIS, port=port)
        print("  posez la piece ; j'attends le capteur (30 s max)")
        vue = robot.attendre_piece_sur_robot(timeout=30.0, port=port)
        robot.convoyeur_sur_robot(0, port=port)
        if not vue:
            print("  aucune piece vue en 30 s -- essai abandonne")
            continue
        print("  piece au capteur, tapis arrete.")
        demander("  Amenez la ventouse sur le CENTRE de la piece, "
                 "puis Entree... ")
        T = robot.pose_sur_robot(port=port)
        p = T[:3, 3]
        poses.append(T)
        print("  appris : x={:.1f}  y={:.1f}  z={:.1f}".format(*p))

    if not poses:
        sys.exit("\naucun point appris : rien n'a ete ecrit.")

    points = np.array([T[:3, 3] for T in poses])
    moyenne = points.mean(axis=0)
    # l'orientation du dernier essai : une ventouse s'en moque, mais la
    # garder evite de la reinventer, et elle sert telle quelle a une pince
    rz = float(np.degrees(np.arctan2(poses[-1][1, 0], poses[-1][0, 0])))

    print("\n" + "=" * 66)
    print("RESULTAT")
    print("=" * 66)
    print("  point moyen : x={:.1f}  y={:.1f}  z={:.1f}  rz={:.1f} deg"
          .format(moyenne[0], moyenne[1], moyenne[2], rz))
    if len(points) > 1:
        etendues = points.max(axis=0) - points.min(axis=0)
        print("  etendue sur {} essais : {:.1f} / {:.1f} / {:.1f} mm"
              .format(len(points), *etendues))
        pire = float(max(etendues))
        if pire > 15:
            print("  -> DISPERSION IMPORTANTE. Elle vient de la cellule, pas")
            print("     du calcul : pieces de tailles differentes, posees")
            print("     trop pres du capteur, ou qui glissent. Un point")
            print("     moyen ne rattrapera pas cela -- reprenez la pose des")
            print("     pieces avant d'aller plus loin.")
        elif pire > 5:
            print("  -> dispersion moyenne : la ventouse pardonne, une pince")
            print("     non. Suffisant pour ce cycle.")
        else:
            print("  -> dispersion faible : la prise sera franche.")

    with open(FICHIER, "w", encoding="utf-8") as f:
        json.dump({"x": round(float(moyenne[0]), 2),
                   "y": round(float(moyenne[1]), 2),
                   "z": round(float(moyenne[2]), 2),
                   "rz": round(rz, 2),
                   "essais": len(points),
                   "vitesse_tapis": VITESSE_TAPIS}, f, indent=2)
    print("\n  ecrit dans {}".format(FICHIER))
    print("  examples/convoyeur.py le lira tout seul au prochain lancement.")
    print("\n  Pour le mettre en dur dans un programme a vous :")
    print("    PRISE = cible(x={:.1f}, y={:.1f}, z={:.1f}, rz={:.1f})"
          .format(moyenne[0], moyenne[1], moyenne[2], rz))
    print("\n  A REFAIRE si le tapis est deplace, si le capteur change de")
    print("  place, ou si les pieces changent de taille : le point appris")
    print("  contient tout cela a la fois.")


if __name__ == "__main__":
    main()
