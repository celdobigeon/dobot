#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le bras va-t-il la ou on le lui demande ?

    python diag_vise_juste.py COM6

LE BRAS BOUGE : il monte au-dessus du point de prise appris, a 70 mm
au-dessus de la piece. Il ne descend PAS dessus -- rien ne touche rien.

La question, et pourquoi elle se pose maintenant
-------------------------------------------------
Le point de prise a ete appris a 1,5 mm pres, trois fois de suite. Le
tapis arrete la piece au droit du capteur, mesure. Et pourtant le bras
va la chercher ailleurs.

Il ne reste alors qu'une possibilite : entre la cible qu'on ecrit et le
mouvement qui se produit, quelque chose se perd. Ce script le mesure
sans rien supposer -- on commande une pose, on relit la pose atteinte,
on compare. Si l'ecart est nul, le bras obeit et c'est la CIBLE qui est
fausse ; s'il ne l'est pas, c'est la chaine de commande.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim import Pilote, cible                       # noqa: E402
from dobot_sim import homogeneous as hg                   # noqa: E402

FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "examples", "convoyeur_etalonnage.json")
DEGAGEMENT = 70.0          # mm au-dessus de la piece


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else None
    if not os.path.exists(FICHIER):
        sys.exit("pas d'etalonnage : lancez etalonner_convoyeur.py")
    with open(FICHIER, encoding="utf-8") as f:
        p = json.load(f)

    robot = Pilote(nom="visee", outil="ventouse", vitesse=80,
                   acceleration=300, vitesse_articulaire=40)
    robot.regler_longueur_outil(79.3)

    PRISE = cible(x=p["x"], y=p["y"], z=p["z"], rz=p["rz"])
    VISEE = PRISE @ hg.translation(0, 0, -DEGAGEMENT)
    attendu = VISEE[:3, 3]

    print("point de prise appris : x={x:.1f}  y={y:.1f}  z={z:.1f}  "
          "rz={rz:.1f}".format(**p))
    print("on vise {:.0f} mm au-dessus : x={:.1f}  y={:.1f}  z={:.1f}"
          .format(DEGAGEMENT, *attendu))
    print("")

    avant = robot.pose_sur_robot(port=port)
    print("pose actuelle du bras   : x={:.1f}  y={:.1f}  z={:.1f}"
          .format(*avant[:3, 3]))
    try:
        input("Entree pour y aller (le bras va bouger)... ")
    except EOFError:
        pass

    def aller(titre, T, attendu_xyz):
        robot.vider_programme()
        robot.movj(T) if titre == "approche" else robot.movl(T)
        robot.executer_sur_robot(vitesse=20, port=port)
        atteint = robot.pose_sur_robot(port=port)[:3, 3]
        ecart = atteint - attendu_xyz
        print("")
        print("  {} :".format(titre.upper()))
        print("    demande : x={:7.1f}  y={:7.1f}  z={:7.1f}".format(*attendu_xyz))
        print("    atteint : x={:7.1f}  y={:7.1f}  z={:7.1f}".format(*atteint))
        print("    ecart   : {:+.1f} / {:+.1f} / {:+.1f} mm   (norme {:.1f})"
              .format(ecart[0], ecart[1], ecart[2],
                      float(np.linalg.norm(ecart))))
        return float(np.linalg.norm(ecart))

    e1 = aller("approche", VISEE, attendu)
    print("")
    print("  La ventouse est {:.0f} mm AU-DESSUS du point de prise.".format(
        DEGAGEMENT))
    print("  Est-elle au-dessus de la piece ?")
    try:
        rep = input("  DESCENDRE dessus ? (o/n, elle va toucher) ")
    except EOFError:
        rep = "o"
    if rep[:1].lower() == "n":
        print("")
        print("  Descente annulee. La ventouse n'etant PAS au-dessus de la")
        print("  piece alors que le bras atteint sa consigne au dixieme de")
        print("  millimetre, c'est le point appris qui ne decrit plus la")
        print("  realite : lancez diag_derive_origine.py pour mesurer de")
        print("  combien le repere a bouge.")
        return
    e2 = aller("descente", PRISE, PRISE[:3, 3])

    print("")
    print("=" * 60)
    if max(e1, e2) < 5.0:
        print("  LE BRAS OBEIT aux deux consignes.")
        print("  Si la ventouse n'est pas sur la piece, c'est le POINT")
        print("  APPRIS qui ne correspond plus a l'endroit ou la piece")
        print("  s'arrete : refaites etalonner_convoyeur.py, dans les")
        print("  memes conditions que le cycle.")
    else:
        print("  LE BRAS N'OBEIT PAS : {:.0f} mm d'ecart.".format(max(e1, e2)))
        print("  La cible n'est pas en cause, la chaine de commande si.")
    print("=" * 60)


if __name__ == "__main__":
    main()
