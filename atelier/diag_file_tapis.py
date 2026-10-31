#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Un arret de tapis obeit-il quand la FILE DU CONTROLEUR tourne ?

    python diag_file_tapis.py COM6

LE BRAS NE BOUGE PAS : la file ne contiendra qu'une attente.

L'enigme qu'il s'agit de resoudre
----------------------------------
Dans `diag_depassement.py`, le tapis s'arrete net : la piece s'immobilise
au droit du capteur. Dans `examples/convoyeur.py`, elle file 145 a
180 mm au-dela. Or les deux programmes envoient EXACTEMENT la meme
trame, et le chronometre le confirme : une dizaine de millisecondes dans
les deux cas, mesurees sur la machine.

Une seule chose les distingue. L'exemple fait tourner le bras entre deux
pieces, donc la FILE du controleur est en marche quand il commande le
tapis. Le diagnostic, lui, ne fait tourner que le tapis, file a l'arret.

Si un `SetEMotor` immediat est ignore ou differe tant que la file
tourne, tout s'explique d'un coup -- y compris pourquoi ralentir le
tapis de 60 a 25 mm/s n'a rien change : un retard en TEMPS donne une
distance qui ne suit pas la vitesse, et 145 puis 180 mm, c'est a peu
pres constant.

Deux essais, a regarder avec les yeux
--------------------------------------
    A. file A L'ARRET      -- le tapis doit s'arreter net
    B. file EN MARCHE      -- s'arrete-t-il encore ?

Le script attend votre Entree avant chaque essai, et annonce chaque
etape. Ne lachez pas la bande des yeux apres l'ordre d'arret.
"""

import sys
import time

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim.convoyeur import IMPULSIONS_PAR_MM
from dobot_sim.liaison import ErreurLiaison, LiaisonDobot, ports_disponibles

VITESSE = 60.0                 # mm/s : un depassement se voit mieux vite
MARCHE = 4.0
OBSERVATION = 8.0


def demander(message: str) -> str:
    try:
        return input(message).strip()
    except EOFError:
        return ""


def essai(li, titre, explication):
    print("")
    print("=" * 64)
    print(titre)
    print("=" * 64)
    for ligne in explication:
        print("  " + ligne)
    demander("  Entree pour lancer le tapis... ")

    li.moteur_etendu(0, True, VITESSE * IMPULSIONS_PAR_MM, en_file=False)
    for reste in range(int(MARCHE), 0, -1):
        print("    en marche... {}".format(reste))
        time.sleep(1.0)

    t = time.perf_counter()
    li.moteur_etendu(0, False, 0.0, en_file=False)
    envoi = (time.perf_counter() - t) * 1000.0
    print("")
    print("    >>> ORDRE D'ARRET ENVOYE ({:.0f} ms) - REGARDEZ LA BANDE <<<"
          .format(envoi))
    for reste in range(int(OBSERVATION), 0, -1):
        print("    observation... {}".format(reste))
        time.sleep(1.0)

    rep = demander("  Le tapis s'est-il arrete TOUT DE SUITE ? (o/n) ")
    return rep[:1].lower() == "o"


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie")
        port = dispo[0]

    li = LiaisonDobot(port).ouvrir()
    try:
        print("port {} - firmware {}".format(port, li.version()))
        print(__doc__.split("Deux essais")[0].split("L'enigme")[1].strip())
        li.arreter_file()
        li.vider_file()

        a = essai(li, "A. FILE A L'ARRET  (comme diag_depassement.py)",
                  ["La file du controleur est vide et arretee.",
                   "C'est la situation ou le tapis s'arrete net."])

        li.vider_file()
        li.attendre(30.0)              # une attente, pour occuper la file
        li.demarrer_file()
        b = essai(li, "B. FILE EN MARCHE  (comme examples/convoyeur.py)",
                  ["La file tourne, occupee par une attente de 30 s.",
                   "Le bras ne bouge pas : il n'y a aucun mouvement dedans.",
                   "C'est la situation de l'exemple, qui fait tourner le",
                   "bras entre deux pieces."])

        print("")
        print("=" * 64)
        print("BILAN")
        print("=" * 64)
        if a and b:
            print("  Le tapis obeit dans les deux cas.")
            print("  -> La file n'y est pour rien, et mon hypothese tombe.")
            print("     Il reste a regarder la DETECTION : relancez")
            print("     examples/convoyeur.py et lisez le recapitulatif de")
            print("     fin, qui donne le retard entre detection et arret.")
        elif a and not b:
            print("  Le tapis obeit file arretee, PAS file en marche.")
            print("  -> C'est la cause. Un SetEMotor immediat ne passe pas")
            print("     tant que la file tourne : le tapis ne s'arrete qu'a")
            print("     la fin du programme du bras, d'ou un depassement de")
            print("     plusieurs secondes, independant de la vitesse.")
            print("     Remede : couper la file avant d'arreter le tapis,")
            print("     ou passer les commandes de tapis DANS la file.")
        elif not a:
            print("  Le tapis n'obeit meme pas file arretee.")
            print("  -> Inattendu : c'est pourtant ce que fait")
            print("     diag_depassement.py, ou il s'arrete net. Quelque")
            print("     chose a change sur la cellule depuis.")
        else:
            print("  Il obeit file en marche mais pas file arretee :")
            print("  resultat incoherent, a refaire.")
    finally:
        try:
            li.moteur_etendu(0, False, 0.0, en_file=False)
        except ErreurLiaison:
            pass
        li.arreter_file()
        li.vider_file()
        li.fermer()
        print("")
        print("  tapis arrete, file videe, liaison fermee")


if __name__ == "__main__":
    main()
