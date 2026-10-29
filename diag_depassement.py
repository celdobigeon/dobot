#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
De combien le tapis depasse apres l'ordre d'arret, et selon quelle loi.

    python diag_depassement.py COM6

LE BRAS NE BOUGE PAS. Il vous faut une regle et des pieces.

La mesure se fait A L'ARRET
---------------------------
On ne trace pas un repere sur une bande qui defile : c'est infaisable,
et c'est pourtant ce que demandait la premiere version de ce script. On
se sert du CAPTEUR. Une piece arrive, le capteur la voit, le tapis
s'arrete -- et elle a continue. Il ne reste qu'a mesurer, tranquillement
et a l'arret, la distance entre le capteur et la piece. C'est le
depassement.

Pourquoi chronometrer ne marcherait pas
---------------------------------------
Mesurer la distance TOTALE entre « marche » et « arret » ne montre
rien : la rampe de demarrage perd exactement ce que la rampe d'arret
gagne, et le total tombe juste. C'est ce qui s'est passe ici -- 180 mm
mesures pour 180 mm attendus, alors que la piece depassait le capteur de
145 mm. Une mesure peut etre exacte et ne rien dire.

Ce qu'on en tire, et pourquoi plusieurs vitesses
-------------------------------------------------
Deux mecanismes donnent le meme depassement a une vitesse donnee et
divergent partout ailleurs :

    RETARD FIXE   la commande met un temps t a agir ; depassement = v.t
                  -> deux fois moins vite, deux fois moins loin
    RAMPE         le moteur decelere a taux constant ; d = v^2/(2a)
                  -> deux fois moins vite, QUATRE fois moins loin

Entre 60 et 20 mm/s les deux predictions different d'un facteur trois.
La loi decide si ralentir le tapis resout le probleme ou le deplace.

Le decalage entre le bord de la piece que le capteur voit et l'endroit
d'ou vous mesurez est CONSTANT : il s'ajoute a toutes les mesures de la
meme facon et disparait quand on les compare. Mesurez donc toujours au
meme endroit de la piece, peu importe lequel -- mais le meme.
"""

import sys
import time

from dobot_sim.convoyeur import PORTS_GP, IMPULSIONS_PAR_MM
from dobot_sim.liaison import ErreurLiaison, LiaisonDobot, ports_disponibles

VITESSES = (60.0, 30.0, 20.0)
GP = PORTS_GP["GP4"]           # le capteur de presence
ATTENTE_MAX = 60.0


def demander(message: str) -> str:
    try:
        return input(message).strip()
    except EOFError:
        return ""


def nombre(message: str):
    while True:
        rep = demander(message)
        if not rep:
            return None
        try:
            return float(rep.replace(",", "."))
        except ValueError:
            print("    (un nombre, ou Entree pour passer)")


def mesurer(li, vitesse):
    """Un passage de piece a une vitesse donnee. Rend le depassement."""
    print("")
    print("=" * 64)
    print("VITESSE {:.0f} mm/s".format(vitesse))
    print("=" * 64)
    print("  Le tapis part. Posez une piece BIEN EN AMONT du capteur :")
    print("  il faut que la bande ait atteint sa vitesse, et a la main on")
    print("  pose toujours trop pres. Le capteur l'arretera, et vous")
    print("  mesurerez a l'arret, sans vous presser.")
    demander("  Entree pour lancer... ")

    li.moteur_etendu(0, True, vitesse * IMPULSIONS_PAR_MM, en_file=False)
    print("    tapis en marche, j'attends le capteur ({:.0f} s max)"
          .format(ATTENTE_MAX))
    t0 = time.perf_counter()
    vu = False
    while time.perf_counter() - t0 < ATTENTE_MAX:
        try:
            if li.capteur_ir(GP):
                vu = True
                break
        except ErreurLiaison:
            continue
    li.moteur_etendu(0, False, 0.0, en_file=False)
    if not vu:
        print("    aucune piece vue : vitesse abandonnee")
        return None
    print("    detectee apres {:.1f} s, tapis arrete."
          .format(time.perf_counter() - t0))
    return nombre("    Distance du CAPTEUR a la piece, en mm ? ")


def bilan(mesures):
    print("")
    print("=" * 64)
    print("BILAN")
    print("=" * 64)
    if len(mesures) < 2:
        print("  Moins de deux vitesses : impossible de trancher la loi.")
        if mesures:
            v, d = mesures[0]
            print("  Retenez au moins : depassement={:.0f}, "
                  "vitesse_depassement={:.0f}".format(d, v))
        return

    v0, d0 = mesures[0]
    print("  mesures : {}".format(
        ", ".join("{:.0f} mm a {:.0f} mm/s".format(d, v) for v, d in mesures)))

    err_retard = sum(abs(d - d0 * v / v0) for v, d in mesures[1:])
    err_rampe = sum(abs(d - d0 * (v / v0) ** 2) for v, d in mesures[1:])
    print("  ecart au modele RETARD FIXE : {:.0f} mm".format(err_retard))
    print("  ecart au modele RAMPE       : {:.0f} mm".format(err_rampe))

    loi = "retard" if err_retard <= err_rampe else "rampe"
    print("")
    if loi == "retard":
        print("  -> RETARD FIXE : le depassement suit la vitesse.")
        print("     retard equivalent : {:.2f} s".format(d0 / v0))
        print("     Ralentir aide, mais lineairement seulement. Et un")
        print("     retard de plus d'une seconde n'est pas une rampe")
        print("     mecanique : cherchez du cote du firmware.")
    else:
        print("  -> RAMPE : le depassement suit le CARRE de la vitesse.")
        print("     deceleration : {:.0f} mm/s2".format(v0 ** 2 / (2 * d0)))
        print("     Ralentir aide beaucoup : moitie moins vite, quatre fois")
        print("     moins loin.")

    print("")
    print("  A mettre dans votre Convoyeur :")
    print("      depassement={:.0f},".format(d0))
    print("      vitesse_depassement={:.0f},".format(v0))
    print('      loi_depassement="{}",'.format(loi))
    print("")
    print("  Le modele saura alors ou la piece s'immobilise vraiment :")
    print("  TAPIS.abscisse_arret(vitesse). Le capteur n'est que l'endroit")
    print("  ou on l'a vue passer.")
    print("")
    print("  ATTENTION : le point de prise appris par")
    print("  etalonner_convoyeur.py ne vaut que pour la vitesse a laquelle")
    print("  il a ete appris. Si vous changez VITESSE_TAPIS, refaites-le.")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie")
        port = dispo[0]

    li = LiaisonDobot(port).ouvrir()
    mesures = []
    try:
        print("port {} - firmware {}".format(port, li.version()))
        li.arreter_file()
        li.vider_file()
        li.demarrer_file()
        li.activer_capteur_ir(GP, True)
        time.sleep(0.3)
        print(__doc__.split("Pourquoi chronometrer")[0].strip())

        for vitesse in VITESSES:
            d = mesurer(li, vitesse)
            if d is not None:
                mesures.append((vitesse, d))
        bilan(mesures)
    finally:
        try:
            li.moteur_etendu(0, False, 0.0, en_file=False)
            li.activer_capteur_ir(GP, False)
        except ErreurLiaison:
            pass
        li.arreter_file()
        li.vider_file()
        li.fermer()
        print("")
        print("  tapis arrete, capteur eteint, liaison fermee")


if __name__ == "__main__":
    main()
