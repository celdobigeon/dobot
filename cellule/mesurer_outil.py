#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesure de la longueur reelle de l'outil monte sur le Dobot Magician.

    python mesurer_outil.py                  # mesure guidee, RIEN ne bouge
    python mesurer_outil.py --port COM6
    python mesurer_outil.py --plan 12.5      # plan de contact a z = 12,5 mm
    python mesurer_outil.py --angles 0,45,45,0   # calcul hors ligne, sans robot

Principe. L'outil du Magician reste vertical, pointe vers le bas : sa
longueur est donc simplement la HAUTEUR DE LA BRIDE quand la pointe touche le
plan de travail. Le script lit les angles articulaires, calcule la hauteur de
la bride par le modele geometrique direct SANS outil (lt = 0), et en retranche
la cote du plan :

    L = z_bride(q) - z_plan

Le resultat ne depend donc pas de la longueur du catalogue. C'est la meme
mesure que "lire le z annonce et l'ajouter a la longueur courante", sans
l'etape intermediaire.

Pourquoi PLUSIEURS points. Une longueur d'outil est la meme partout. Si les
releves s'ecartent d'un endroit a l'autre de la table, ce n'est pas l'outil :
c'est un reste d'erreur d'origine articulaire, ou une table qui n'est pas
plane. Il faut alors recaler le robot (calibrer_robot.py) avant de conclure.

Aucun mouvement n'est commande : on deplace le bras A LA MAIN (bouton de
deverrouillage sur l'avant-bras). Le robot doit avoir ete recale ou avoir
fait sa prise d'origine dans la seance, sinon ses angles sont faux.
"""

import argparse
import sys

import numpy as np

from dobot_sim import DobotMagician

#: au-dela, les releves ne sont plus coherents avec un outil rigide
DISPERSION_MAX = 1.0      # mm

CONSIGNES = """\
  Pour chaque releve :
    1. maintenez le bouton de deverrouillage et amenez la POINTE de l'outil
       au contact du plan de travail -- elle touche, sans l'ecraser (une
       ventouse comprimee fausse la mesure de plusieurs millimetres) ;
    2. relachez le bouton, puis appuyez sur Entree.

  Faites au moins trois releves, bien ecartes sur la table : pres et loin de
  la colonne, a gauche et a droite.

  Entree = relever    a = annuler le dernier    f = terminer
"""


def longueur_depuis_angles(q_deg, z_plan: float = 0.0,
                           robot: DobotMagician = None) -> float:
    """Longueur d'outil deduite d'une posture ou la pointe touche le plan."""
    nu = (robot or DobotMagician()).avec_outil(0.0)
    z_bride = float(nu.pose(q_deg)[2])
    return z_bride - z_plan


def bilan(releves, z_plan: float, nominale: float = None) -> int:
    """Moyenne, dispersion et ligne a recopier dans le programme."""
    L = np.array([r[1] for r in releves], dtype=float)
    print()
    print("=== BILAN " + "=" * 60)
    for i, (q, l) in enumerate(releves, 1):
        print("  {:2d}. J = ({:+7.2f} {:+7.2f} {:+7.2f})   L = {:7.2f} mm"
              .format(i, q[0], q[1], q[2], l))
    moyenne = float(L.mean())
    dispersion = float(L.max() - L.min())
    print()
    print("  longueur moyenne : {:.2f} mm".format(moyenne))
    if len(L) > 1:
        print("  dispersion       : {:.2f} mm (ecart-type {:.2f})"
              .format(dispersion, float(L.std(ddof=1))))
    if nominale is not None:
        print("  catalogue        : {:.2f} mm, soit {:+.2f} mm d'ecart"
              .format(nominale, moyenne - nominale))

    if len(L) < 3:
        print()
        print("  ATTENTION : moins de trois releves, la constance de la")
        print("  longueur n'est pas verifiee.")
    if dispersion > DISPERSION_MAX:
        print()
        print("  ATTENTION : les releves s'ecartent de plus de {:.1f} mm."
              .format(DISPERSION_MAX))
        print("  Un outil rigide a la meme longueur partout : l'ecart vient")
        print("  plutot de l'origine articulaire ou d'une table non plane.")
        print("  Recalez le robot (calibrer_robot.py) et recommencez.")
        return 1

    print()
    print("  A recopier dans le programme, APRES le constructeur du Pilote :")
    print()
    print("      robot.regler_longueur_outil({:.1f})".format(moyenne))
    if z_plan:
        print()
        print("  (plan de contact pris a z = {:.2f} mm)".format(z_plan))
    return 0


def mesure_guidee(pont, z_plan: float, nominale: float = None) -> int:
    li = pont.liaison
    print()
    print(CONSIGNES)
    releves = []
    while True:
        try:
            rep = input("  [{} releve(s)] > ".format(len(releves))).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            rep = "f"
        if rep == "f":
            break
        if rep == "a":
            if releves:
                releves.pop()
                print("      dernier releve annule")
            continue
        q = tuple(li.pose().angles)
        L = longueur_depuis_angles(q, z_plan, pont.robot)
        releves.append((q, L))
        print("      J = ({:+.2f} {:+.2f} {:+.2f} {:+.2f})  ->  L = {:.2f} mm"
              .format(*q, L))
        if len(releves) > 1:
            Ls = [r[1] for r in releves]
            print("      dispersion jusqu'ici : {:.2f} mm"
                  .format(max(Ls) - min(Ls)))

    if not releves:
        print("  Aucun releve : rien a conclure.")
        return 1
    return bilan(releves, z_plan, nominale)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mesure la longueur reelle de l'outil, pointe au contact "
                    "du plan de travail. Aucun mouvement commande.")
    ap.add_argument("--port", default="", help="port serie (defaut : le premier "
                                               "pont CH340 / CP210x trouve)")
    ap.add_argument("--plan", type=float, default=0.0,
                    help="cote z du plan de contact dans le repere du "
                         "simulateur, en mm (defaut 0 : la table)")
    ap.add_argument("--outil", default="", choices=["", "ventouse", "pince",
                                                    "stylo", "laser"],
                    help="outil monte, pour comparer a sa longueur catalogue")
    ap.add_argument("--angles", default="",
                    help="calcul hors ligne a partir d'angles J1,J2,J3,J4 "
                         "en degres ; ne se connecte pas au robot")
    args = ap.parse_args()

    nominale = None
    if args.outil:
        from dobot_sim import outils as ou
        nominale = ou.outil(args.outil).longueur

    if args.angles:
        q = [float(v) for v in args.angles.split(",")]
        q += [0.0] * (4 - len(q))
        L = longueur_depuis_angles(q[:4], args.plan)
        return bilan([(tuple(q[:4]), L)], args.plan, nominale)

    from dobot_sim.liaison import ErreurLiaison, ports_disponibles
    from dobot_sim.reel import PontReel

    ports = ports_disponibles()
    if not ports:
        print("Aucun port serie. Branchez le Dobot et allumez-le.")
        return 1
    port = args.port or ports[0][0]
    print("ports vus : " + ", ".join("{} ({})".format(d, n) for d, n in ports))

    pont = PontReel(DobotMagician(), journal=lambda m: print("  [liaison] " + m))
    try:
        pont.connecter(port)
    except ErreurLiaison as exc:
        print("connexion impossible : {}".format(exc))
        return 1
    try:
        ecart, _ = pont.coherence()
        if ecart >= 0.05:
            print()
            print("  ATTENTION : la pose du controleur et celle du modele ne")
            print("  concordent pas ({:.3f} mm/deg) : ils ne decrivent pas la"
                  .format(ecart))
            print("  meme geometrie, la mesure serait douteuse.")
        print()
        print("  RAPPEL : les angles lus ne valent que si le robot a ete")
        print("  recale ou a fait sa prise d'origine depuis la mise sous")
        print("  tension (python calibrer_robot.py --recaler ou --origine).")
        return mesure_guidee(pont, args.plan, nominale)
    finally:
        pont.deconnecter()


if __name__ == "__main__":
    sys.exit(main())
