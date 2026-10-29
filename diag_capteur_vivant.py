#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le capteur, en direct, pendant que vous bougez une piece a la main.

    python diag_capteur_vivant.py COM6

RIEN NE BOUGE : ni le bras, ni le tapis. On ne fait que lire.

Ce qu'il s'agit d'etablir
--------------------------
Le journal du dernier cycle dit que le capteur voyait DEJA quelque
chose au moment ou le tapis demarrait : detection en 8 ms, bande qui
n'a pas eu le temps de tourner, piece qui ne quitte jamais le faisceau.
Trois lectures possibles, et il faut choisir :

  - une piece etait restee devant lui, d'un cycle precedent ;
  - il voit quelque chose en permanence : la bande, un reflet, le bati ;
  - il voit LOIN, et apercoit une piece bien avant qu'elle n'arrive
    devant lui -- ce qui expliquerait d'un coup les 145 a 180 mm, sans
    qu'aucune de nos mesures ne soit fausse.

L'affichage ci-dessous rend l'etat du capteur visible en continu. A vous
de promener une piece le long de la bande, tapis a l'arret, et de voir
OU il bascule.
"""

import sys
import time

from dobot_sim.convoyeur import PORTS_GP
from dobot_sim.liaison import ErreurLiaison, LiaisonDobot, ports_disponibles

GP = PORTS_GP["GP4"]
DUREE = 60.0


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
        li.arreter_file()
        li.vider_file()
        li.demarrer_file()
        li.activer_capteur_ir(GP, True)
        time.sleep(0.3)

        print("")
        print("=" * 64)
        print("1. DEGAGEZ LA BANDE : enlevez toute piece du convoyeur.")
        print("=" * 64)
        print("   Si le capteur reste a VU pendant que la bande est vide,")
        print("   il voit autre chose qu'une piece, et c'est son reglage")
        print("   qu'il faut reprendre.")
        print("")
        print("2. Puis PROMENEZ UNE PIECE le long de la bande, lentement,")
        print("   dans le sens de marche, et regardez OU il bascule.")
        print("   S'il bascule bien avant d'etre en face de lui, nous")
        print("   tenons les 145 a 180 mm.")
        print("")
        print("   {:.0f} secondes. Ctrl-C pour arreter plus tot.".format(DUREE))
        print("")

        t0 = time.time()
        precedent = None
        bascules = 0
        while time.time() - t0 < DUREE:
            try:
                etat = bool(li.capteur_ir(GP))
            except ErreurLiaison:
                continue
            if etat != precedent:
                bascules += (precedent is not None)
                print("  t={:5.1f}s   {}".format(
                    time.time() - t0,
                    "######  VU  ######" if etat else "        rien"))
                precedent = etat
            time.sleep(0.03)

        print("")
        print("  {} basculement(s) en {:.0f} s".format(bascules, DUREE))
        if bascules == 0 and precedent:
            print("  -> le capteur est reste a VU tout du long : il voit")
            print("     quelque chose en permanence. Reglez sa sensibilite")
            print("     (vis sur le capteur) ou son orientation.")
        elif bascules == 0:
            print("  -> le capteur n'a JAMAIS bascule : soit rien n'est")
            print("     passe devant, soit il ne voit pas vos pieces.")
    except KeyboardInterrupt:
        print("\n  interrompu")
    finally:
        try:
            li.activer_capteur_ir(GP, False)
        except ErreurLiaison:
            pass
        li.arreter_file()
        li.vider_file()
        li.fermer()
        print("  capteur eteint, liaison fermee")


if __name__ == "__main__":
    main()
