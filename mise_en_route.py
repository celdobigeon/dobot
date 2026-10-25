#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mise en route du Dobot reel, sans ouvrir le simulateur.

    .venv/bin/python mise_en_route.py
    .venv/bin/python mise_en_route.py --port /dev/ttyUSB0

Ouvre l'assistant de mise en route -- le meme que celui de la fenetre du
simulateur -- dans une petite fenetre a part :

  1. declarer la posture de reference (equerre ou butees) : capteurs
     d'angle du bras arriere et de l'avant-bras recales, rien ne bouge ;
  2. prise d'origine : la base retrouve son repere, le bras parcourt sa
     course ;
  3. effacer les defauts et verifier que le controleur voit le bras ou il
     est.

A faire apres CHAQUE mise sous tension, avant tout autre script. La
fenetre du simulateur doit etre fermee : elle garderait le port serie.
"""

import argparse
import sys

from dobot_sim import DobotMagician
from dobot_sim.liaison import resume_alarmes
from dobot_sim.pilote import _fermer_liaison_directe, _liaison_directe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mise en route du Dobot reel")
    ap.add_argument("--port", help="port serie (defaut : le premier trouve)")
    a = ap.parse_args(argv)

    robot = DobotMagician()
    try:
        pont = _liaison_directe(robot, a.port)
    except RuntimeError as exc:
        print("CONNEXION IMPOSSIBLE : {}".format(exc))
        print("  le robot est-il allume et branche ? la fenetre du simulateur "
              "est-elle fermee ?")
        return 1
    try:
        alarmes = pont.liaison.alarmes()
        if alarmes:
            print("robot en defaut :")
            for ligne in resume_alarmes(alarmes):
                print("  " + ligne)
        else:
            print("aucun defaut signale ; l'assistant permet quand meme de "
                  "refaire la mise en route")
        from dobot_sim.ui import mise_en_route_autonome
        ok = mise_en_route_autonome(pont, robot)
        x, y, z, r = (float(v) for v in pont.depuis_controleur(*_pose(pont)))
        if ok:
            print("robot pret : x {:.1f}  y {:.1f}  z {:.1f}  r {:.1f}".format(x, y, z, r))
            return 0
        print("le robot signale encore un defaut :")
        for ligne in resume_alarmes(pont.liaison.alarmes()):
            print("  " + ligne)
        return 2
    finally:
        _fermer_liaison_directe()


def _pose(pont):
    p = pont.liaison.pose()
    return p.x, p.y, p.z, p.r


if __name__ == "__main__":
    sys.exit(main())
