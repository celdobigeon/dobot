#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apprendre au robot OU prendre la piece sur le tapis.

    python etalonner_convoyeur.py COM6

LE BRAS BOUGE, sous vos ordres, par petits pas. Il ne faut PAS le
pousser a la main : c'est tout l'objet de ce script.

Pourquoi on ne deplace pas le bras a la main
---------------------------------------------
Le controleur du Magician relit DEUX capteurs d'angle -- bras arriere et
avant-bras. Le protocole le dit : « Angle sensors of forearm and rear
arm may have a static offset ». Deux, et pas trois : la ROTATION DE BASE
n'en a pas. Le controleur la compte en pas.

Moteurs relaches, les pas ne sont pas comptes. Pousser le bras a la main
lui fait donc perdre J1 -- silencieusement, et jusqu'a la prochaine
prise d'origine.

Mesure sur cette cellule : trois poses successives sur le MEME point
fixe ont donne le meme rayon a 1,4 mm pres et la meme hauteur a 0,3 mm
pres, mais des angles de base distants de 49 degres -- soit 241 mm a ce
rayon. Une main fidele au millimetre dans deux directions ne peut pas
etre fausse de 241 mm dans la troisieme : c'est le controleur qui ne
sait plus.

Un etalonnage fait a la main est donc juste en rayon et en hauteur, et
faux en angle. C'est-a-dire faux LE LONG DU TAPIS, la ou cela compte.

D'ou ce script : on amene la ventouse par DEPLACEMENTS COMMANDES, pas a
pas. Chaque pas est execute par les moteurs, donc compte, et le
controleur sait ou il est.

Pourquoi apprendre plutot que mesurer
--------------------------------------
Le point obtenu absorbe d'un coup ce qu'on ne sait pas mesurer
separement : la place et l'orientation reelles du tapis, l'abscisse
reelle du capteur, le retard de la scrutation, la hauteur de la bande,
l'ecrasement de la ventouse, et le decalage entre le BORD AVANT de la
piece -- qui coupe le faisceau -- et son CENTRE, que la ventouse vise.
Sur une piece de 25 mm, ce dernier vaut deja une demi-piece.

L'etalonnage doit se faire DANS LES MEMES CONDITIONS que le cycle :
meme vitesse, et meme attente sur FRONT, la piece arrivant en mouvement.
Un point appris sur une piece deja immobile devant le capteur decrit une
situation qui ne se reproduira pas.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_sim import Convoyeur, Pilote                        # noqa: E402
from dobot_sim.convoyeur import PORTS_GP                       # noqa: E402
from dobot_sim.pilote import _liaison_directe                  # noqa: E402

ESSAIS = 3
# La MEME que celle du cycle : le point appris ne vaut que pour elle.
VITESSE_TAPIS = 60.0
FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "examples", "convoyeur_etalonnage.json")

AIDE = """
  COMMANDES  (une par ligne, puis Entree)

    x+  x-     avancer / reculer     (x du robot, vers l'avant)
    y+  y-     gauche / droite
    z+  z-     monter / descendre
    1 5 10 25  changer le pas, en mm
    ou         enregistrer ce point
    non        abandonner cet essai

  Amenez la ventouse AU CONTACT du centre de la piece, puis « ou ».
"""


def demander(message: str) -> str:
    try:
        return input(message).strip().lower()
    except EOFError:
        return "non"


def amener(pont, robot) -> bool:
    """
    Deplace le bras pas a pas, sous les ordres de l'operateur.

    Rend True si le point est valide, False s'il est abandonne.
    """
    pas = 5.0
    print(AIDE)
    while True:
        p = robot.pose_sur_robot()[:3, 3]
        rep = demander("  [pas {:.0f} mm]  x={:.1f} y={:.1f} z={:.1f}  > "
                       .format(pas, p[0], p[1], p[2]))
        if rep in ("ou", "oui", "o", ""):
            return True
        if rep in ("non", "n", "q"):
            return False
        if rep in ("1", "5", "10", "25"):
            pas = float(rep)
            continue
        axes = {"x+": (pas, 0.0, 0.0), "x-": (-pas, 0.0, 0.0),
                "y+": (0.0, pas, 0.0), "y-": (0.0, -pas, 0.0),
                "z+": (0.0, 0.0, pas), "z-": (0.0, 0.0, -pas)}
        if rep not in axes:
            print("    ? -- x+ x- y+ y- z+ z- , 1 5 10 25 , ou , non")
            continue
        dx, dy, dz = axes[rep]
        pont.deplacer_manuel(dx=dx, dy=dy, dz=dz, vitesse=20.0,
                             dire=lambda _m: None)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None

    tapis = Convoyeur(longueur=698, index_moteur=0,
                      capteur_port=PORTS_GP["GP4"])
    robot = Pilote(nom="etalonnage", outil="ventouse", vitesse=120)
    robot.tapis = tapis
    robot.regler_longueur_outil(79.3)

    print(__doc__.split("Pourquoi apprendre")[0].strip())
    print("")
    print("=" * 66)
    print("{} essais. A chaque fois :".format(ESSAIS))
    print("  1. le tapis part, vous posez une piece BIEN EN AMONT ;")
    print("  2. le capteur la voit, le tapis s'arrete ;")
    print("  3. vous amenez la ventouse sur la piece AU CLAVIER.")
    print("=" * 66)

    pont = _liaison_directe(robot.robot, port)
    poses = []
    for essai in range(1, ESSAIS + 1):
        demander("\nessai {}/{} - Entree pour lancer le tapis... "
                 .format(essai, ESSAIS))
        robot.convoyeur_sur_robot(VITESSE_TAPIS, port=port)
        print("  posez la piece ; j'attends le capteur (60 s max)")
        vue = robot.attendre_piece_sur_robot(timeout=60.0, port=port)
        robot.convoyeur_sur_robot(0, port=port)
        if not vue:
            print("  aucune piece vue -- essai abandonne")
            continue
        print("  piece au capteur, tapis arrete.")
        if not amener(pont, robot):
            print("  essai abandonne")
            continue
        T = robot.pose_sur_robot(port=port)
        poses.append(T)
        print("  appris : x={:.1f}  y={:.1f}  z={:.1f}".format(*T[:3, 3]))

    if not poses:
        sys.exit("\naucun point appris : rien n'a ete ecrit.")

    points = np.array([T[:3, 3] for T in poses])
    moyenne = points.mean(axis=0)
    rz = float(np.degrees(np.arctan2(poses[-1][1, 0], poses[-1][0, 0])))

    print("")
    print("=" * 66)
    print("RESULTAT")
    print("=" * 66)
    print("  point moyen : x={:.1f}  y={:.1f}  z={:.1f}  rz={:.1f} deg"
          .format(moyenne[0], moyenne[1], moyenne[2], rz))
    if len(points) > 1:
        etendues = points.max(axis=0) - points.min(axis=0)
        print("  etendue sur {} essais : {:.1f} / {:.1f} / {:.1f} mm"
              .format(len(points), *etendues))
        # Le rayon et l'angle SEPAREMENT : c'est l'angle qui trahit un J1
        # perdu, et il se noie dans une etendue exprimee en x-y.
        rayons = np.hypot(points[:, 0], points[:, 1])
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        ecart_angle = float(angles.max() - angles.min())
        print("  dont rayon {:.1f} mm, angle {:.1f} deg"
              .format(float(rayons.max() - rayons.min()), ecart_angle))
        if ecart_angle > 3.0:
            print("  -> L'ANGLE DISPERSE. Le bras a-t-il ete pousse a la")
            print("     main entre deux essais ? Cela fait perdre J1 au")
            print("     controleur. Reprenez la prise d'origine, puis cet")
            print("     etalonnage, SANS toucher au bras.")
        pire = float(max(etendues))
        if pire > 15:
            print("  -> dispersion importante : reprenez la pose des pieces")
        elif pire > 5:
            print("  -> dispersion moyenne : la ventouse pardonne, une pince non")
        else:
            print("  -> dispersion faible : la prise sera franche")

    with open(FICHIER, "w", encoding="utf-8") as f:
        json.dump({"x": round(float(moyenne[0]), 2),
                   "y": round(float(moyenne[1]), 2),
                   "z": round(float(moyenne[2]), 2),
                   "rz": round(rz, 2),
                   "essais": len(points),
                   "vitesse_tapis": VITESSE_TAPIS}, f, indent=2)
    print("")
    print("  ecrit dans {}".format(FICHIER))
    print("  examples/convoyeur.py le lira tout seul au prochain lancement.")
    print("")
    print("  A REFAIRE apres chaque prise d'origine, apres chaque coupure,")
    print("  et chaque fois que le bras a ete pousse a la main.")


if __name__ == "__main__":
    main()
