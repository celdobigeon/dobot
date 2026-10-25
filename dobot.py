#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le Dobot reel en ligne de commande : une commande, une action.

    .venv/bin/python dobot.py etat                  pose, angles, alarmes
    .venv/bin/python dobot.py pince ouvrir          (ou fermer)
    .venv/bin/python dobot.py ventouse on           (ou off)
    .venv/bin/python dobot.py pompe off             couper la pompe
    .venv/bin/python dobot.py repli                 posture de repos
    .venv/bin/python dobot.py equerre               bras arriere vertical,
                                                    avant-bras horizontal
    .venv/bin/python dobot.py aller 220 0 80 [r]    MOVJ vers (x, y, z, r)
    .venv/bin/python dobot.py aller 220 0 80 --lineaire
    .venv/bin/python dobot.py effacer               effacer les alarmes
    .venv/bin/python dobot.py mise-en-route [--sans-posture]
    .venv/bin/python dobot.py etalonner [--reprendre ...]
                                                    etalonnage camera -> robot

Les coordonnees sont celles du simulateur (x, y, z de la pointe d'outil, en
mm ; r en degres). Options communes, AVANT la commande :

    --port /dev/...       port serie (defaut : le premier trouve)
    --vitesse 40          mm/s (et deg/s pour les MOVJ)
    --outil pince         pince, ventouse... (defaut : pince)
    --longueur-outil 78   longueur mesuree de l'outil, mm

La fenetre du simulateur doit etre fermee : elle garderait le port serie.
Un robot en defaut ouvre l'assistant de mise en route avant tout mouvement.
"""

import argparse
import os
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))


def _commande(a):
    from dobot_sim import Commande
    return Commande(simulateur=False, robot=True, port=a.port, outil=a.outil,
                    vitesse=a.vitesse, vitesse_articulaire=a.vitesse,
                    longueur_outil=a.longueur_outil)


def _afficher_pose(c) -> None:
    x, y, z, r = c.pose()
    q = c.angles()
    print("pose    x {:7.1f}  y {:7.1f}  z {:7.1f} mm   r {:7.1f} deg".format(x, y, z, r))
    print("angles  J1 {:+7.2f}  J2 {:+7.2f}  J3 {:+7.2f}  J4 {:+7.2f} deg".format(*q))


def etat(a) -> int:
    from dobot_sim.liaison import resume_alarmes
    c = _commande(a)
    try:
        _afficher_pose(c)
        alarmes = c._pont.liaison.alarmes()
        print("alarmes " + ("aucune" if not alarmes else ""))
        for ligne in resume_alarmes(alarmes):
            print("  " + ligne)
    finally:
        c.fermer()
    return 0


def pince(a) -> int:
    c = _commande(a)
    try:
        c.pince(a.action == "fermer")
        if a.action == "ouvrir":
            print("pince ouverte ; la pompe sera coupee dans 1 s")
        else:
            print("pince fermee (la pompe tourne tant qu'elle tient)")
    finally:
        c.fermer()            # attend la coupure de pompe prevue
    return 0


def ventouse(a) -> int:
    c = _commande(a)
    try:
        c.ventouse(a.action == "on", couper=True)
        print("ventouse " + ("active" if a.action == "on"
                             else "relachee ; la pompe sera coupee dans 1 s"))
    finally:
        c.fermer()
    return 0


def pompe(a) -> int:
    c = _commande(a)
    try:
        c.couper_pompe()
        print("pompe coupee")
    finally:
        c.fermer()
    return 0


def repli(a) -> int:
    c = _commande(a)
    try:
        c.repli()
        _afficher_pose(c)
    finally:
        c.fermer()
    return 0


def equerre(a) -> int:
    """Bras arriere vertical, avant-bras horizontal ; base et poignet gardes."""
    c = _commande(a)
    try:
        q = c.angles()
        c.movj(q=[float(q[0]), 0.0, 0.0, float(q[3])])
        _afficher_pose(c)
    finally:
        c.fermer()
    return 0


def aller(a) -> int:
    c = _commande(a)
    try:
        r = a.r if a.r is not None else c.pose()[3]
        if a.lineaire:
            c.movl(x=a.x, y=a.y, z=a.z, r=r)
        else:
            c.movj(x=a.x, y=a.y, z=a.z, r=r)
        _afficher_pose(c)
    finally:
        c.fermer()
    return 0


def effacer(a) -> int:
    from dobot_sim import DobotMagician
    from dobot_sim.liaison import resume_alarmes
    from dobot_sim.pilote import _fermer_liaison_directe, _liaison_directe
    pont = _liaison_directe(DobotMagician(), a.port)
    try:
        li = pont.liaison
        li.arreter_file()
        li.vider_file()
        li.effacer_alarmes()
        time.sleep(0.3)
        reste = li.alarmes()
        if not reste:
            print("alarmes effacees")
            return 0
        print("certaines alarmes restent :")
        for ligne in resume_alarmes(reste):
            print("  " + ligne)
        print("  -> dobot.py mise-en-route --sans-posture (prise d'origine)")
        return 2
    finally:
        _fermer_liaison_directe()


def mise_en_route(a) -> int:
    import mise_en_route as mer
    args = (["--port", a.port] if a.port else []) + (["--sans-posture"]
                                                      if a.sans_posture else [])
    return mer.main(args)


def etalonner(a) -> int:
    sys.path.insert(0, os.path.join(ICI, "examples", "perception"))
    import etalonner_camera_robot as ec
    args = list(a.reste)
    if a.longueur_outil is not None and "--longueur-outil" not in args:
        args += ["--longueur-outil", str(a.longueur_outil)]
    if "--vitesse" not in args:
        args += ["--vitesse", str(a.vitesse)]
    return ec.main(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Le Dobot reel en ligne de commande",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 2)[1])
    ap.add_argument("--port", help="port serie (defaut : le premier trouve)")
    ap.add_argument("--vitesse", type=float, default=30.0,
                    help="mm/s, et deg/s pour les MOVJ (30)")
    ap.add_argument("--outil", default="pince", help="outil monte (pince)")
    ap.add_argument("--longueur-outil", type=float, default=None,
                    help="longueur mesuree de l'outil, mm")
    sp = ap.add_subparsers(dest="commande", required=True, metavar="commande")

    sp.add_parser("etat", help="pose, angles, alarmes").set_defaults(f=etat)
    p = sp.add_parser("pince", help="ouvrir ou fermer la pince")
    p.add_argument("action", choices=("ouvrir", "fermer"))
    p.set_defaults(f=pince)
    p = sp.add_parser("ventouse", help="activer ou relacher la ventouse")
    p.add_argument("action", choices=("on", "off"))
    p.set_defaults(f=ventouse, outil="ventouse")
    p = sp.add_parser("pompe", help="couper la pompe")
    p.add_argument("action", choices=("off",))
    p.set_defaults(f=pompe)
    sp.add_parser("repli", help="posture de repos (0, 15, 30, 0)").set_defaults(f=repli)
    sp.add_parser("equerre", help="bras arriere vertical, avant-bras horizontal"
                  ).set_defaults(f=equerre)
    p = sp.add_parser("aller", help="aller en (x, y, z, r)")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("z", type=float)
    p.add_argument("r", type=float, nargs="?")
    p.add_argument("--lineaire", action="store_true", help="ligne droite (MOVL)")
    p.set_defaults(f=aller)
    sp.add_parser("effacer", help="effacer les alarmes").set_defaults(f=effacer)
    p = sp.add_parser("mise-en-route", help="assistant de mise en route")
    p.add_argument("--sans-posture", action="store_true",
                   help="passer l'etape 1 (posture deja declaree)")
    p.set_defaults(f=mise_en_route)
    p = sp.add_parser("etalonner", help="etalonnage camera -> robot",
                      add_help=False)
    p.add_argument("reste", nargs=argparse.REMAINDER,
                   help="options de etalonner_camera_robot.py (--reprendre...)")
    p.set_defaults(f=etalonner)

    a = ap.parse_args(argv)
    try:
        return a.f(a)
    except KeyboardInterrupt:
        print("\ninterrompu")
        return 130
    except Exception as exc:                        # noqa: BLE001
        print("ECHEC : {}".format(str(exc).splitlines()[0] if str(exc) else exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
