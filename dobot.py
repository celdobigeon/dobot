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
    .venv/bin/python dobot.py deplacer 0 0 20       deplacement RELATIF de
                                                    l'outil : dx dy dz (mm)
    .venv/bin/python dobot.py deplacer 10 -5 0 --dr 15
    .venv/bin/python dobot.py effacer               effacer les alarmes
    .venv/bin/python dobot.py mise-en-route [--sans-posture] [--equerre]
    .venv/bin/python dobot.py posture [equerre|butees]
                                                    etape 1 seule : declarer la
                                                    posture (rien ne bouge)
    .venv/bin/python dobot.py origine               prise d'origine seule
                                                    (= mise-en-route --sans-posture)
    .venv/bin/python dobot.py etalonner-avec-pieces [--reprendre ...]
                                                    etalonnage camera -> robot
                                                    par poses d'une piece
    .venv/bin/python dobot.py etalonner-pince       etalonnage camera -> robot
                                                    par la pince seule
    .venv/bin/python dobot.py saisir [--boucle]     essai de saisie guidee
                                                    par la camera
    .venv/bin/python dobot.py saisir --visualisation
                                                    la meme, dans le simulateur
                                                    (miroir, pieces affichees)
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
    """Lecture seule : ne bouge rien, n'ouvre aucun assistant."""
    from dobot_sim import DobotMagician
    from dobot_sim.liaison import resume_alarmes
    from dobot_sim.outils import outil
    from dobot_sim.pilote import _fermer_liaison_directe, _liaison_directe
    modele = DobotMagician()
    modele.geo.lt = float(a.longueur_outil if a.longueur_outil is not None
                          else outil(a.outil).longueur)
    pont = _liaison_directe(modele, a.port)
    try:
        p = pont.liaison.pose()
        x, y, z, r = pont.depuis_controleur(p.x, p.y, p.z, p.r)
        q = pont.angles_reels()
        print("pose    x {:7.1f}  y {:7.1f}  z {:7.1f} mm   r {:7.1f} deg".format(x, y, z, r))
        print("angles  J1 {:+7.2f}  J2 {:+7.2f}  J3 {:+7.2f}  J4 {:+7.2f} deg".format(*q))
        alarmes = pont.liaison.alarmes()
        print("alarmes " + ("aucune" if not alarmes else ""))
        for ligne in resume_alarmes(alarmes):
            print("  " + ligne)
    finally:
        _fermer_liaison_directe()
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


def deplacer(a) -> int:
    """
    Deplacement RELATIF de la pointe d'outil, dans le repere de la base.
    Robot en defaut (file bloquee) ou --manuel : en mode manuel, un axe
    apres l'autre, z d'abord.
    """
    from dobot_sim import DobotMagician
    from dobot_sim.outils import outil
    from dobot_sim.pilote import _fermer_liaison_directe, _liaison_directe
    modele = DobotMagician()
    modele.geo.lt = float(a.longueur_outil if a.longueur_outil is not None
                          else outil(a.outil).longueur)
    pont = _liaison_directe(modele, a.port)
    p = pont.liaison.pose()
    x0, y0, z0, r0 = pont.depuis_controleur(p.x, p.y, p.z, p.r)
    cible = (x0 + a.dx, y0 + a.dy, z0 + a.dz, r0 + a.dr)
    print("depuis  x {:7.1f}  y {:7.1f}  z {:7.1f}  r {:7.1f}".format(x0, y0, z0, r0))
    if not modele.atteignable(*cible):
        # jusqu'ou peut-on aller dans cette direction ? (dichotomie)
        bas, haut = 0.0, 1.0
        for _ in range(30):
            m = 0.5 * (bas + haut)
            if modele.atteignable(x0 + m * a.dx, y0 + m * a.dy, z0 + m * a.dz, r0 + m * a.dr):
                bas = m
            else:
                haut = m
        print("HORS D'ATTEINTE : x {:.1f}  y {:.1f}  z {:.1f}  r {:.1f} sort de l'espace de "
              "travail du bras\n  (une articulation passerait sa butee). Rien n'a bouge."
              .format(*cible))
        if not modele.atteignable(x0, y0, z0, r0):
            print("  Le bras est DEJA hors de l'espace de travail du modele (une "
                  "articulation\n  est au-dela de sa butee) : ramenez-le d'abord "
                  "vers le centre, par exemple\n  dobot.py deplacer 0 0 -20")
        elif bas > 0.02:
            print("  Dans cette direction, on peut aller au plus a {:.0f} % du chemin : "
                  "deplacer {:.0f} {:.0f} {:.0f}".format(100 * bas, bas * a.dx, bas * a.dy,
                                                         bas * a.dz))
        _fermer_liaison_directe()
        return 1
    alarmes = pont.liaison.alarmes()
    manuel = a.manuel or bool(alarmes)
    if manuel:
        try:
            if alarmes:
                from dobot_sim.liaison import resume_alarmes
                print("Le robot est en DEFAUT :")
                for ligne in resume_alarmes(alarmes):
                    print("  " + ligne)
                print("Dans cet etat, le controleur refuse tout mouvement programme. On "
                      "deplace donc\nle bras en MODE MANUEL (comme les fleches du boitier), "
                      "un axe apres l'autre,\nz d'abord, lentement. Ctrl+C l'arrete a tout "
                      "moment.")
            if a.dr:
                print("  (--dr ignore en mode manuel)")
            x, y, z, r = pont.deplacer_manuel(a.dx, a.dy, a.dz,
                                              vitesse=min(a.vitesse, 20.0), dire=print)
            print("pose    x {:7.1f}  y {:7.1f}  z {:7.1f} mm   r {:7.1f} deg"
                  .format(x, y, z, r))
            if alarmes:
                print("Le defaut est toujours la : pour l'effacer, prise d'origine\n"
                      "  .venv/bin/python mise_en_route.py --sans-posture")
        finally:
            _fermer_liaison_directe()
        return 0
    c = _commande(a)
    try:
        x, y, z, r = c.pose()
        cible = (x + a.dx, y + a.dy, z + a.dz, r + a.dr)
        print("de      x {:7.1f}  y {:7.1f}  z {:7.1f}  r {:7.1f}".format(x, y, z, r))
        if a.articulaire:
            c.movj(x=cible[0], y=cible[1], z=cible[2], r=cible[3])
        else:
            c.movl(x=cible[0], y=cible[1], z=cible[2], r=cible[3])
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
        if any(0x10 <= k < 0x20 for k in reste):
            print("  Une alarme de PLANIFICATION bloque la file du controleur (firmware\n"
                  "  3.7.0) : rien ne l'efface, pas meme la prise d'origine. Eteignez et\n"
                  "  rallumez le robot (bras tenu, en equerre), puis :\n"
                  "    dobot.py mise-en-route")
        else:
            print("  -> dobot.py origine (prise d'origine)")
        return 2
    finally:
        _fermer_liaison_directe()


def mise_en_route(a) -> int:
    import mise_en_route as mer
    args = (["--port", a.port] if a.port else [])
    args += ["--sans-posture"] if getattr(a, "sans_posture", False) else []
    args += ["--equerre"] if getattr(a, "equerre", False) else []
    return mer.main(args)


def posture(a) -> int:
    """
    Etape 1 de la mise en route SEULE : declarer la posture de reference
    (equerre ou butees). Rien ne bouge : le controleur recale seulement les
    capteurs d'angle du bras arriere et de l'avant-bras.
    """
    import re
    from dobot_sim import DobotMagician
    from dobot_sim.liaison import resume_alarmes
    from dobot_sim.pilote import _fermer_liaison_directe, _liaison_directe
    from dobot_sim.ui import DialogueMiseEnRoute
    p = DialogueMiseEnRoute.POSTURES[a.quelle]
    j2, j3 = p["angles"]
    geste = re.sub(r"<[^>]+>", "", p["geste"])
    pont = _liaison_directe(DobotMagician(), a.port)
    try:
        q = pont.angles_reels()
        print("le controleur croit : J2 {:+.2f}  J3 {:+.2f} deg".format(q[1], q[2]))
        print("POSTURE {} (J2 {:+.2f}, J3 {:+.2f}) : bouton de deverrouillage enfonce "
              "(avant-bras),\n  {}, puis relachez le bouton.".format(
                  p["nom"].upper(), j2, j3, geste))
        print("  La rotation de la base ne compte pas ici. Rien ne va bouger.")
        rep = input("Le bras est en place ? [Entree : declarer / q : annuler] ")
        if rep.strip().lower().startswith("q"):
            print("annule : rien n'a ete declare")
            return 1
        pont.liaison.redefinir_pose(j2, j3)
        time.sleep(0.3)
        q = pont.angles_reels()
        if abs(q[1] - j2) > 0.5 or abs(q[2] - j3) > 0.5:
            print("ECHEC : le controleur n'a pas retenu la posture (J2 {:+.2f}, J3 {:+.2f})"
                  .format(q[1], q[2]))
            return 2
        print("posture declaree : J2 {:+.2f}  J3 {:+.2f} deg".format(q[1], q[2]))
        if p.get("hors_course"):
            print("  Ramenez maintenant le bras dans sa course (bras arriere vers la "
                  "verticale,\n  avant-bras vers l'horizontale) avant la prise d'origine.")
        alarmes = pont.liaison.alarmes()
        if alarmes:
            print("alarmes encore levees (la prise d'origine les efface) :")
            for ligne in resume_alarmes(alarmes):
                print("  " + ligne)
        print("suite : dobot.py origine")
    finally:
        _fermer_liaison_directe()
    return 0


def origine(a) -> int:
    """La mise en route SANS declarer la posture : prise d'origine et
    verification (posture deja declaree depuis la mise sous tension)."""
    a.sans_posture = True
    return mise_en_route(a)


def etalonner(a) -> int:
    sys.path.insert(0, os.path.join(ICI, "examples", "perception"))
    import etalonner_camera_robot as ec
    args = list(a.reste)
    if a.longueur_outil is not None and "--longueur-outil" not in args:
        args += ["--longueur-outil", str(a.longueur_outil)]
    if "--vitesse" not in args:
        args += ["--vitesse", str(a.vitesse)]
    return ec.main(args)


def etalonner_pince(a) -> int:
    sys.path.insert(0, os.path.join(ICI, "examples", "perception"))
    import etalonner_par_pince
    args = ["--vitesse", str(a.vitesse)] + list(a.reste)
    if a.longueur_outil is not None:
        args += ["--longueur-outil", str(a.longueur_outil)]
    if a.port:
        args += ["--port", a.port]
    return etalonner_par_pince.main(args)


def saisir(a) -> int:
    sys.path.insert(0, os.path.join(ICI, "examples", "perception"))
    args = ["--vitesse", str(a.vitesse)]
    if a.longueur_outil is not None:
        args += ["--longueur-outil", str(a.longueur_outil)]
    if a.hauteur_prise is not None:
        args += ["--hauteur-prise", str(a.hauteur_prise)]
    if a.visualisation:
        import saisir_piece_graphique
        if a.port:
            args += ["--port", a.port]
        return saisir_piece_graphique.main(args)
    import saisir_piece
    if a.boucle:
        args.append("--boucle")
    if a.forcer:
        args.append("--forcer")
    return saisir_piece.main(args)


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
    p = sp.add_parser("deplacer", help="deplacement relatif de l'outil (dx dy dz, mm)",
                      description="Deplace la pointe d'outil de (dx, dy, dz) mm, "
                                  "dans le repere de la base, en ligne droite.")
    p.add_argument("dx", type=float)
    p.add_argument("dy", type=float)
    p.add_argument("dz", type=float)
    p.add_argument("--dr", type=float, default=0.0, help="rotation de l'outil, deg")
    p.add_argument("--articulaire", action="store_true",
                   help="mouvement articulaire (MOVJ) plutot qu'en ligne droite")
    p.add_argument("--manuel", action="store_true",
                   help="en mode manuel (JOG), un axe apres l'autre, z d'abord ; "
                        "automatique si le robot est en defaut")
    p.set_defaults(f=deplacer)
    sp.add_parser("effacer", help="effacer les alarmes").set_defaults(f=effacer)
    p = sp.add_parser("mise-en-route", help="assistant de mise en route")
    p.add_argument("--sans-posture", action="store_true",
                   help="passer l'etape 1 (posture deja declaree)")
    p.add_argument("--equerre", action="store_true",
                   help="amener d'abord le bras en equerre en mode manuel")
    p.set_defaults(f=mise_en_route)
    p = sp.add_parser("posture", help="etape 1 seule : declarer la posture (equerre ou "
                                      "butees), rien ne bouge")
    p.add_argument("quelle", nargs="?", default="equerre", choices=("equerre", "butees"))
    p.set_defaults(f=posture)
    p = sp.add_parser("origine", help="prise d'origine (= mise-en-route --sans-posture) : "
                                      "efface les defauts tenaces (0x12)")
    p.add_argument("--equerre", action="store_true",
                   help="amener d'abord le bras en equerre en mode manuel")
    p.set_defaults(f=origine)
    # ancien nom garde en alias : etalonner-camera
    p = sp.add_parser("etalonner-avec-pieces", aliases=["etalonner-camera"],
                      help="etalonnage camera -> robot par poses d'une piece",
                      add_help=False)
    p3 = sp.add_parser("etalonner-pince", help="etalonnage camera -> robot par la pince "
                                               "seule (aucune piece a poser)")
    p3.set_defaults(f=etalonner_pince, reste=[])
    p2 = sp.add_parser("saisir", help="essai : saisir une piece vue par la camera, "
                                      "la reposer ailleurs")
    p2.add_argument("--boucle", action="store_true",
                    help="enchainer sans attendre Entree (jusqu'a Ctrl+C)")
    p2.add_argument("--visualisation", action="store_true",
                    help="version graphique : simulateur en miroir, pieces vues et "
                         "cible affichees en continu")
    p2.add_argument("--hauteur-prise", type=float, default=None,
                    help="mm visees entre le bout des doigts et la table (3)")
    p2.add_argument("--forcer", action="store_true",
                    help="sans marges de securite (plus de positions saisissables)")
    p2.set_defaults(f=saisir)
    # ses options (--reprendre...) sont celles de etalonner_camera_robot.py :
    # elles lui sont passees telles quelles (voir parse_known_args)
    p.set_defaults(f=etalonner, reste=[])

    a, inconnus = ap.parse_known_args(argv)
    if inconnus:
        if a.f not in (etalonner, etalonner_pince):
            ap.error("arguments inconnus : {}".format(" ".join(inconnus)))
        a.reste = inconnus
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
