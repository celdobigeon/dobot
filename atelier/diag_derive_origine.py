#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le repere du robot a-t-il bouge ?

    python diag_derive_origine.py --poser COM6     enregistre la reference
    python diag_derive_origine.py COM6             la compare

LE BRAS NE BOUGE PAS DE LUI-MEME : c'est vous qui l'amenez a la main.

Ce qu'on mesure, et pourquoi
----------------------------
Le bras atteint ses consignes au dixieme de millimetre -- mesure. Et
pourtant la ventouse ne tombe pas sur la piece. Si la commande est
exacte, c'est que la CIBLE ne decrit plus la realite : entre
l'etalonnage et maintenant, le repere du robot a bouge.

Ce qui peut le faire bouger : une coupure d'alimentation, une prise
d'origine, un bras remis a la main. Sur cette cellule il y a eu les
trois.

IL FAUT UN REPERE QUI NE BOUGE PAS. Pas une piece : une piece se pose
a la main, et on mesurerait alors son placement autant que la derive du
robot. Choisissez un detail FIXE et franc de la cellule -- un coin du
bati du tapis, le boitier du capteur, une vis -- et amenez-y toujours
la ventouse au meme point.

  --poser   enregistre ce point comme reference
  (sans)    y revient et dit de combien le robot s'est deplace

Entre les deux : une prise d'origine, une coupure, ce que vous voulez
tester. Mesurez d'abord DEUX FOIS DE SUITE sans rien changer, pour
connaitre la fidelite de votre propre geste : c'est le bruit au-dessous
duquel aucune conclusion ne tient.

  quelques dixiemes   le repere est stable, cherchez ailleurs
  quelques mm         la prise d'origine n'est pas parfaitement
                      repetable : il faut re-etalonner apres chaque
                      mise sous tension, et le dire en TP
  plus de 10 mm       la prise d'origine n'est pas fiable du tout
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim import Pilote                                # noqa: E402

FICHIER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "examples", "repere_fixe.json")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    poser = "--poser" in sys.argv
    port = args[0] if args else None

    robot = Pilote(nom="derive", outil="ventouse", vitesse=80)
    robot.regler_longueur_outil(79.3)

    if poser:
        print("ENREGISTREMENT DE LA REFERENCE")
        print("")
        print("  Choisissez un detail FIXE de la cellule -- un coin du bati")
        print("  du tapis, le boitier du capteur, une vis -- et amenez-y la")
        print("  ventouse au contact, a la main (bouton de la fourche).")
        print("  C'est a ce point-la qu'on reviendra. Choisissez-le franc :")
        print("  un angle vaut mieux qu'une surface plane.")
        try:
            input("  Puis Entree... ")
        except EOFError:
            pass
        ici = robot.pose_sur_robot(port=port)[:3, 3]
        with open(FICHIER, "w", encoding="utf-8") as f:
            json.dump({"x": round(float(ici[0]), 2),
                       "y": round(float(ici[1]), 2),
                       "z": round(float(ici[2]), 2)}, f, indent=2)
        print("")
        print("  reference enregistree : x={:.1f}  y={:.1f}  z={:.1f}"
              .format(*ici))
        print("  ecrit dans {}".format(FICHIER))
        print("")
        print("  Relancez SANS --poser pour mesurer, d'abord tout de suite")
        print("  (c'est la fidelite de votre geste), puis apres une prise")
        print("  d'origine ou une coupure.")
        return

    if not os.path.exists(FICHIER):
        sys.exit("aucune reference : lancez d'abord avec --poser")
    with open(FICHIER, encoding="utf-8") as f:
        p = json.load(f)
    appris = np.array([p["x"], p["y"], p["z"]], dtype=float)

    print("reference enregistree : x={:.1f}  y={:.1f}  z={:.1f}".format(*appris))
    print("")
    print("  Ramenez la ventouse sur LE MEME detail fixe, au contact.")
    try:
        input("  Puis Entree... ")
    except EOFError:
        pass

    ici = robot.pose_sur_robot(port=port)[:3, 3]
    ecart = ici - appris
    norme = float(np.linalg.norm(ecart))

    print("")
    print("=" * 60)
    print("  reference  : x={:7.1f}  y={:7.1f}  z={:7.1f}".format(*appris))
    print("  maintenant : x={:7.1f}  y={:7.1f}  z={:7.1f}".format(*ici))
    print("  ECART      : {:+.1f} / {:+.1f} / {:+.1f} mm   (norme {:.1f})"
          .format(ecart[0], ecart[1], ecart[2], norme))
    print("=" * 60)
    print("")
    print("  Rappel : cet ecart contient AUSSI l'imprecision de votre")
    print("  geste. Mesurez deux fois de suite sans rien changer pour la")
    print("  connaitre, et ne concluez qu'au-dela.")
    if norme < 2.0:
        print("  -> ici, moins de 2 mm : rien de significatif.")
    elif norme < 10.0:
        print("  -> quelques millimetres : a confirmer contre le bruit de")
        print("     votre geste avant d'accuser la prise d'origine.")
    else:
        print("  -> plus de 10 mm : si votre geste est fidele a 2 mm pres,")
        print("     c'est le repere du robot qui a bouge, et il faudra")
        print("     re-etalonner apres chaque mise sous tension.")


if __name__ == "__main__":
    main()
