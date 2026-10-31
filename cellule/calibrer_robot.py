#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prise d'origine du Dobot Magician reel, en ligne de commande.

    python calibrer_robot.py                 # diagnostic seul, RIEN ne bouge
    python calibrer_robot.py --origine       # lance la prise d'origine
    python calibrer_robot.py --port COM4 --origine

Sans `--origine`, le script se contente de LIRE : firmware, alarmes, pose,
point d'origine enregistre, et controle de coherence entre le modele et le
controleur. Aucun mouvement.

Avec `--origine`, le bras PARCOURT TOUTE SA COURSE pour retrouver ses
reperes, puis se parque sur le point enregistre dans HOMEParams. Degagez la
zone de travail avant de lancer.

La sequence suivie est la plus prudente possible :

  1. couper la file, puis la vider -- dans cet ordre, sinon une file restee
     en mode "demarre" ferait partir la commande des son arrivee ;
  2. effacer les alarmes, qui feraient refuser le mouvement ;
  3. brider les vitesses ;
  4. envoyer SetHOMECmd puis demarrer la file ;
  5. suivre l'index et les angles jusqu'a stabilisation ;
  6. verifier que la pose obtenue et les angles obtenus se correspondent.

En cas de delai depasse, arret force et file videe.
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

from dobot_sim import DobotMagician
from dobot_sim.liaison import (CommandeDobot, ErreurLiaison,
                               ports_disponibles, resume_alarmes)
from dobot_sim.reel import PontReel


def diagnostic(pont: PontReel) -> None:
    """Tout ce que la machine sait dire d'elle-meme. Lecture seule."""
    li = pont.liaison
    p = li.pose()
    print("  pose annoncee   x={:.2f} y={:.2f} z={:.2f} r={:.2f} "
          "(repere controleur)".format(p.x, p.y, p.z, p.r))
    xs, ys, zs, rs = pont.depuis_controleur(p.x, p.y, p.z, p.r)
    print("                  x={:.2f} y={:.2f} z={:.2f} r={:.2f} "
          "(repere simulateur)".format(xs, ys, zs, rs))
    print("  angles J1..J4   {}".format(
        tuple(round(a, 2) for a in p.angles)))

    alarmes = li.alarmes()
    print("  alarmes         {}".format(
        [hex(i) for i in alarmes] or "aucune"))
    for m in resume_alarmes(alarmes):
        print("                    " + m)

    hp = li.echanger(CommandeDobot.PARAMS_HOME)
    if len(hp) >= 16:
        x, y, z, r = struct.unpack("<4f", hp[:16])
        print("  point d'origine x={:.2f} y={:.2f} z={:.2f} r={:.2f}"
              .format(x, y, z, r))

    ecart, recalcule = pont.coherence()
    print("  coherence       modele contre controleur : ecart {:.4f} mm/deg"
          .format(ecart))
    print("                  {}".format(
        "COHERENT" if ecart < 0.05 else "INCOHERENT -- le robot a "
        "probablement besoin d'une prise d'origine"))


def prise_origine(pont: PontReel, limite: float = 120.0) -> bool:
    """Lance la prise d'origine et attend qu'elle aboutisse."""
    li = pont.liaison
    print()
    print("  vitesses bridees a 30 %")
    pont.definir_ratio(30.0)

    print("  envoi de SetHOMECmd")
    index = pont.prise_origine()
    print("  commande en file sous l'index {}".format(index))
    print()
    print("  prise d'origine en cours, ne touchez pas au bras...")

    t0 = time.time()
    affiche = ""
    while time.time() - t0 < limite:
        time.sleep(0.5)
        courant = li.index_courant()
        angles = li.pose().angles
        ligne = "J = ({:7.2f}, {:7.2f}, {:7.2f}, {:7.2f})".format(*angles)
        if ligne != affiche:
            print("    t={:5.1f} s  index {}/{}  {}"
                  .format(time.time() - t0, courant, index, ligne))
            affiche = ligne
        if courant >= index:
            break
    else:
        print("  DELAI DEPASSE : arret force")
        pont.arret_urgence()
        return False

    # l'index avance des la prise en charge : on laisse le bras s'immobiliser
    print("  stabilisation...")
    stable, precedent = 0, None
    while stable < 6 and time.time() - t0 < limite + 30.0:
        time.sleep(0.4)
        a = np.asarray(li.pose().angles, dtype=float)
        stable = stable + 1 if (precedent is not None
                                and np.allclose(a, precedent, atol=1e-3)) else 0
        precedent = a
    return True


#: Posture de reference du recalage, dans la convention du controleur --
#: verifiee sur machine reelle : z = l1.cos(J2) - l2.sin(J3).
#:   bras arriere a 0 deg  = VERTICAL
#:   avant-bras   a 0 deg  = HORIZONTAL
REFERENCE = (0.0, 0.0)

POSTURE = """\
  Placez le bras a la posture de reference :

      - bras arriere VERTICAL        (J2 = {:.1f} deg)
      - avant-bras   HORIZONTAL      (J3 = {:.1f} deg)

  Deverrouillez les moteurs en maintenant le bouton de la fourche, placez
  le bras a la main en verifiant a l'equerre ou au niveau, relachez le
  bouton, puis relancez la commande."""


def capteurs(pont: PontReel, reference) -> None:
    """Erreur statique enregistree, et ecart a la posture de reference."""
    li = pont.liaison
    e = li.erreur_capteurs()
    q = li.pose().angles
    print("  erreur statique enregistree : bras {:+.4f} deg, "
          "avant-bras {:+.4f} deg".format(*e))
    print("  angles annonces maintenant  : J2 = {:+.2f} deg, "
          "J3 = {:+.2f} deg".format(q[1], q[2]))
    print("  posture de reference        : J2 = {:+.2f} deg, "
          "J3 = {:+.2f} deg".format(*reference))
    print("  ECART bras arriere          : {:+.2f} deg"
          .format(q[1] - reference[0]))
    print("  ECART avant-bras            : {:+.2f} deg"
          .format(q[2] - reference[1]))


def recaler(pont: PontReel, reference) -> bool:
    """
    ResetPose : declare les angles vrais a la posture actuelle.

    Aucun mouvement. Effet immediat, mais perdu a la mise hors tension.
    """
    li = pont.liaison
    avant = li.pose().angles
    print("  angles avant : J2 = {:+.2f}, J3 = {:+.2f}".format(avant[1], avant[2]))
    li.redefinir_pose(reference[0], reference[1])
    time.sleep(0.3)
    apres = li.pose().angles
    print("  angles apres : J2 = {:+.2f}, J3 = {:+.2f}".format(apres[1], apres[2]))
    ok = (abs(apres[1] - reference[0]) < 0.5
          and abs(apres[2] - reference[1]) < 0.5)
    print("  -> {}".format(
        "recalage pris en compte" if ok else
        "le controleur n'a PAS retenu les valeurs demandees"))
    return ok


def ecrire_erreur(pont: PontReel, reference, signe: float = 1.0) -> bool:
    """
    Ecrit l'erreur statique PERSISTANTE des capteurs d'angle.

    MESURE FAITE SUR FIRMWARE 3.7.0 : cette commande est bien acceptee et
    memorisee -- on la relit telle qu'on l'a ecrite -- mais elle ne change
    RIEN aux angles annonces tant que le controleur n'a pas redemarre. Elle
    n'est donc pas verifiable a chaud : il faut couper l'alimentation, la
    retablir, puis relire.

    D'ou la procedure en deux temps, et l'option `signe` : le protocole ne
    dit pas dans quel sens l'erreur s'applique, et la sonde a chaud ne peut
    pas le reveler. On ecrit, on redemarre, on regarde. Si l'ecart double
    au lieu de s'annuler, on recommence avec --signe -1.

    Aucune de ces ecritures ne fait bouger le bras.
    """
    li = pont.liaison
    initiale = li.erreur_capteurs()
    q0 = li.pose().angles
    ecart = (q0[1] - reference[0], q0[2] - reference[1])

    print("  erreur actuellement enregistree : bras {:+.4f}, "
          "avant-bras {:+.4f}".format(*initiale))
    print("  >>> NOTEZ CES DEUX VALEURS <<<  --restaurer {:.4f},{:.4f} "
          "les remet en place.".format(*initiale))
    print()
    print("  angles annonces  : J2 {:+.3f}, J3 {:+.3f}".format(q0[1], q0[2]))
    print("  reference        : J2 {:+.3f}, J3 {:+.3f}".format(*reference))
    print("  ecart a corriger : bras {:+.3f} deg, avant-bras {:+.3f} deg"
          .format(*ecart))

    if max(abs(v) for v in ecart) < 0.2:
        print()
        print("  L'ecart est deja negligeable : rien a ecrire.")
        return True

    cible = (initiale[0] + signe * ecart[0], initiale[1] + signe * ecart[1])
    print("  valeur ecrite    : bras {:+.4f}, avant-bras {:+.4f}   (signe {:+.0f})"
          .format(cible[0], cible[1], signe))
    li.definir_erreur_capteurs(*cible)
    time.sleep(0.4)

    relu = li.erreur_capteurs()
    print("  relu             : bras {:+.4f}, avant-bras {:+.4f}".format(*relu))
    if max(abs(a - b) for a, b in zip(relu, cible)) > 1e-3:
        print("  -> le controleur n'a pas retenu la valeur : restauration")
        li.definir_erreur_capteurs(*initiale)
        return False

    print()
    print("  " + "=" * 66)
    print("  ECRIT. Le firmware ne l'applique qu'au DEMARRAGE :")
    print("    1. coupez l'alimentation du robot, attendez 5 s, rallumez")
    print("    2. remettez le bras a l'equerre")
    print("    3. python calibrer_robot.py --capteurs")
    print()
    print("  Si l'ecart est retombe vers 0 : c'est gagne, enchainez sur")
    print("  --origine. S'il a DOUBLE (vers {:+.0f} deg), le signe est"
          .format(2 * ecart[0]))
    print("  inverse : relancez avec --erreur-statique --signe -1.")
    print("  " + "=" * 66)
    return True


def suivre(pont: PontReel, reference) -> None:
    """
    Affiche les angles en direct, jusqu'a Ctrl+C.

    Bien plus commode que "placer le bras puis lancer la commande" : on
    tient le bras, on regarde le nombre. Le test le plus parlant est
    d'ailleurs l'inverse du protocole habituel -- amenez J2 a ZERO et
    regardez si le bras a l'air vertical.
    """
    print("  Amenez le bras la ou J2 affiche 0,00 et regardez s'il est")
    print("  VERTICAL. Ctrl+C pour arreter.")
    print()
    try:
        while True:
            q = pont.liaison.pose().angles
            ecart = q[1] - reference[0]
            reperes = ("  <-- VERTICAL si le bras l'est aussi"
                       if abs(ecart) < 0.6 else "")
            sys.stdout.write(
                "\r  J2 = {:+8.2f} deg   J3 = {:+8.2f} deg   "
                "ecart bras {:+7.2f}{:<40}"
                .format(q[1], q[2], ecart, reperes))
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
        print("  arret")


#: releves accumules par --mesure, relus par --bilan
FICHIER_MESURES = "calibration_mesures.csv"


def mesurer(pont: PontReel, vrais) -> None:
    """
    Ajoute un releve (angle VRAI mesure a la main, angle LU par le robot).

    Placez le bras a un angle que vous savez mesurer -- niveau, equerre,
    inclinometre de telephone -- et donnez-le ici. Trois ou quatre points
    bien repartis suffisent.
    """
    q = pont.liaison.pose().angles
    ligne = "{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
        vrais[0], q[1], vrais[1] if len(vrais) > 1 else float("nan"), q[2])
    neuf = not os.path.exists(FICHIER_MESURES)
    with open(FICHIER_MESURES, "a", encoding="utf-8") as f:
        if neuf:
            f.write("J2_vrai,J2_lu,J3_vrai,J3_lu\n")
        f.write(ligne)
    print("  releve ajoute : J2 vrai {:+.2f} / lu {:+.2f}".format(vrais[0], q[1]))
    if len(vrais) > 1:
        print("                  J3 vrai {:+.2f} / lu {:+.2f}"
              .format(vrais[1], q[2]))
    print("  fichier : {}".format(os.path.abspath(FICHIER_MESURES)))


def _ajuster(vrai, lus):
    """
    Confronte trois modeles d'erreur et renvoie leurs residus.

    Ils ne se reparent pas de la meme facon, d'ou l'interet de les
    distinguer AVANT de demonter quoi que ce soit :

      * CONSTANT   : lu = vrai + b. Un cran de trop sur un accouplement
        indexe. Le remontage le corrige.
      * ECHELLE    : lu = a.vrai + b. Le capteur ne couvre pas la bonne
        plage. Le remontage n'y change rien.
      * BIELLETTE  : lu = vrai + A.cos(vrai) + B.sin(vrai) + C. Signature
        d'un capteur entraine par une tringlerie et non monte dans l'axe :
        l'erreur varie de facon sinusoidale. Ni le remontage ni un offset
        ne la corrigent -- il faut la geometrie du renvoi.
    """
    resultats = {}

    b = float(np.mean(lus - vrai))
    resultats["constant"] = (float(np.abs(lus - (vrai + b)).max()),
                             "lu = vrai {:+.3f}".format(b), (b,))

    if len(vrai) >= 2:
        a, b2 = np.polyfit(vrai, lus, 1)
        resultats["echelle"] = (
            float(np.abs(lus - (a * vrai + b2)).max()),
            "lu = {:.4f} x vrai {:+.3f}".format(a, b2), (a, b2))

    if len(vrai) >= 4:
        r = np.radians(vrai)
        M = np.column_stack([np.cos(r), np.sin(r), np.ones_like(r)])
        (A, B, C), *_ = np.linalg.lstsq(M, lus - vrai, rcond=None)
        residu = float(np.abs(lus - (vrai + M @ (A, B, C))).max())
        resultats["biellette"] = (
            residu,
            "lu = vrai {:+.2f}.cos {:+.2f}.sin {:+.2f}".format(A, B, C),
            (A, B, C))
    return resultats


def bilan() -> int:
    """
    Confronte les releves a trois modeles d'erreur et designe le meilleur.

    Trois points suffisent a distinguer un decalage constant d'une erreur
    d'echelle ; il en faut quatre, bien repartis, pour tester en plus le
    modele a biellette.
    """
    if not os.path.exists(FICHIER_MESURES):
        print("Aucun releve. Utilisez --mesure ANGLE d'abord.")
        return 1
    lignes = [l.strip().split(",") for l in
              open(FICHIER_MESURES, encoding="utf-8").read().splitlines()[1:]
              if l.strip()]
    print("  {} releve(s) dans {}".format(len(lignes),
                                          os.path.abspath(FICHIER_MESURES)))
    if len(lignes) < 2:
        print("  Il en faut au moins 2, et de preference 3 ou 4 bien repartis.")
        return 1

    for i, nom in ((0, "J2 bras arriere"), (2, "J3 avant-bras")):
        points = [(float(l[i]), float(l[i + 1])) for l in lignes
                  if l[i] and l[i].lower() != "nan"]
        print()
        print("  --- {} : {} point(s) ---".format(nom, len(points)))
        for v, lu in points:
            print("      vrai {:+8.2f}   lu {:+8.2f}   ecart {:+7.2f}"
                  .format(v, lu, lu - v))
        if len(points) < 2:
            print("      pas assez de points")
            continue
        vrai = np.array([p[0] for p in points])
        lus = np.array([p[1] for p in points])
        if float(vrai.max() - vrai.min()) < 5.0:
            print("      les points sont trop groupes pour juger du gain ;"
                  " ecartez-les d'au moins 30 deg")
            continue
        modeles = _ajuster(vrai, lus)
        print()
        for nom in ("constant", "echelle", "biellette"):
            if nom in modeles:
                residu, formule, _ = modeles[nom]
                print("      {:<10} residu max {:6.2f} deg   {}"
                      .format(nom, residu, formule))
            else:
                print("      {:<10} (il faut 4 points bien repartis pour le "
                      "tester)".format(nom))

        if len(points) < 3:
            print()
            print("      ATTENTION : avec 2 points, la droite passe")
            print("      EXACTEMENT par les deux, quel que soit le defaut")
            print("      reel. Son residu nul ne prouve rien. Une erreur")
            print("      sinusoidale donnerait les memes deux points.")
            print("      Ajoutez au moins deux postures intermediaires.")
            continue

        print()
        if modeles["constant"][0] < 0.6:
            b = modeles["constant"][2][0]
            if abs(b) < 1.0:
                print("      -> CONFORME : rien a corriger sur cet axe.")
                continue
            print("      -> DECALAGE CONSTANT de {:+.2f} deg".format(b))
            for n in (16, 15, 18, 20, 24):
                pas = 360.0 / n
                crans = b / pas
                if abs(crans - round(crans)) < 0.06 and round(crans) != 0:
                    print("         = {:.0f} cran(s) sur un accouplement a "
                          "{} dents ({:.3f} deg)".format(round(crans), n, pas))
                    break
            else:
                print("         ne tombe sur aucun pas rond")
            print("         Erreur d'indexation : le remontage la corrige.")
            continue

        meilleur = min(modeles, key=lambda k: modeles[k][0])
        residu, _formule, params = modeles[meilleur]
        if meilleur == "echelle":
            a = params[0]
            print("      -> ERREUR D'ECHELLE : gain {:.4f}, soit {:.1f} %."
                  .format(a, 100 * abs(a - 1)))
            print("         Le capteur ne couvre pas la bonne plage.")
            print("         NI le remontage NI un offset ne la corrigent.")
        else:
            A, B, _C = params
            print("      -> ERREUR SINUSOIDALE, amplitude {:.1f} deg."
                  .format(float(np.hypot(A, B))))
            print("         Signature d'un capteur entraine par une")
            print("         TRINGLERIE, non monte dans l'axe de la liaison.")
            print("         Ni le remontage ni un offset n'y peuvent rien.")
        print("         (residu du meilleur modele : {:.2f} deg)".format(residu))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diagnostic, recalage des capteurs et prise d'origine "
                    "du Dobot Magician reel.")
    ap.add_argument("--port", default="", help="port serie (defaut : le CH340)")
    ap.add_argument("--origine", action="store_true",
                    help="lance la prise d'origine : LE ROBOT BOUGE")
    ap.add_argument("--capteurs", action="store_true",
                    help="etat des capteurs d'angle, en lecture seule")
    ap.add_argument("--recaler", action="store_true",
                    help="ResetPose a la posture de reference : aucun "
                         "mouvement, mais perdu a la mise hors tension")
    ap.add_argument("--erreur-statique", action="store_true",
                    help="ecrit le recalage PERSISTANT des capteurs ; "
                         "aucun mouvement")
    ap.add_argument("--restaurer", default="",
                    help="remet une erreur statique notee, au format "
                         "bras,avant-bras")
    ap.add_argument("--signe", type=float, default=1.0, choices=[1.0, -1.0],
                    help="sens d'application de l'erreur statique ; passez "
                         "-1 si l'ecart a double apres redemarrage")
    ap.add_argument("--reference", default="",
                    help="angles de la posture de reference, au format "
                         "J2,J3 (defaut 0,0 : bras vertical, avant-bras "
                         "horizontal)")
    ap.add_argument("--mesure", default="",
                    help="ajoute un releve : angle VRAI du bras arriere, "
                         "eventuellement suivi de celui de l'avant-bras "
                         "(format J2 ou J2,J3)")
    ap.add_argument("--bilan", action="store_true",
                    help="ajuste offset et gain sur les releves accumules ; "
                         "ne se connecte pas au robot")
    ap.add_argument("--suivre", action="store_true",
                    help="affiche les angles en direct jusqu'a Ctrl+C ; "
                         "lecture seule")
    ap.add_argument("--outil", default="", choices=["", "ventouse", "pince",
                                                    "stylo", "laser"],
                    help="outil monte, pour le controle de coherence")
    args = ap.parse_args()

    if args.bilan:
        return bilan()

    reference = REFERENCE
    if args.reference:
        reference = tuple(float(v) for v in args.reference.split(",")[:2])

    ports = ports_disponibles()
    if not ports:
        print("Aucun port serie. Branchez le Dobot et allumez-le.")
        return 1
    port = args.port or ports[0][0]
    print("ports vus : " + ", ".join("{} ({})".format(d, n) for d, n in ports))

    robot = DobotMagician()
    if args.outil:
        from dobot_sim import outils as ou
        robot = robot.avec_outil(ou.outil(args.outil).longueur)

    pont = PontReel(robot, journal=lambda m: print("  [liaison] " + m))
    try:
        pont.connecter(port)
    except ErreurLiaison as exc:
        print("connexion impossible : {}".format(exc))
        return 1

    try:
        print()
        print("=== AVANT " + "=" * 60)
        diagnostic(pont)
        print()
        print("--- capteurs d'angle ---")
        capteurs(pont, reference)

        if args.suivre:
            print()
            print("=== LECTURE EN DIRECT " + "=" * 48)
            suivre(pont, reference)
            return 0

        if args.mesure:
            print()
            print("=== RELEVE " + "=" * 59)
            mesurer(pont, [float(v) for v in args.mesure.split(",")[:2]])
            return 0

        if args.restaurer:
            v = tuple(float(x) for x in args.restaurer.split(",")[:2])
            print()
            print("=== RESTAURATION " + "=" * 53)
            pont.liaison.definir_erreur_capteurs(*v)
            print("  erreur statique remise a {:+.4f}, {:+.4f}".format(*v))
            capteurs(pont, reference)
            return 0

        if args.recaler or args.erreur_statique:
            print()
            print("=== RECALAGE " + "=" * 57)
            print(POSTURE.format(*reference))
            print()
            if args.erreur_statique:
                ok = ecrire_erreur(pont, reference, args.signe)
            else:
                ok = recaler(pont, reference)
                print()
                print("--- apres recalage ---")
                capteurs(pont, reference)
                if ok and not args.origine:
                    print()
                    print("Recalage fait, mais NON persistant : il sera perdu "
                          "a la mise hors tension.")
            if not ok:
                return 1

        if not args.origine:
            if not (args.recaler or args.erreur_statique or args.capteurs):
                print()
                print("Diagnostic seul : rien n'a bouge.")
                print("  --capteurs         etat des capteurs d'angle")
                print("  --recaler          recalage immediat (sans mouvement)")
                print("  --erreur-statique  recalage persistant (sans mouvement)")
                print("  --origine          prise d'origine (LE ROBOT BOUGE)")
            return 0

        print()
        print("=== PRISE D'ORIGINE " + "=" * 51)
        if not prise_origine(pont):
            return 1

        print()
        print("=== APRES " + "=" * 60)
        diagnostic(pont)
        print()
        print("Prise d'origine terminee.")
        return 0
    finally:
        try:
            pont.liaison.arreter_file()
        except Exception:
            pass
        pont.deconnecter()


if __name__ == "__main__":
    sys.exit(main())
