#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quel port voit le capteur de presence, et sous quel nom ?

    python diag_capteur.py COM6

LE BRAS NE BOUGE PAS, LE TAPIS NON PLUS. On ne fait que lire.

A LANCER VOUS-MEME, en ayant la main libre : le script compte a rebours,
puis vous demande de passer un objet devant le capteur pendant qu'il
scrute. C'est le genre d'essai qui ne se delegue pas -- il faut etre
devant la machine ET devant l'ecran en meme temps.

Les deux familles de capteurs, et pourquoi la difference compte
---------------------------------------------------------------
Le kit convoyeur peut etre cable de deux facons, et le programme ne
peut pas en faire la meme chose.

  - INFRAROUGE SUR UN PORT GP (GP1, GP2, GP4, GP5). Il se lit par une
    commande immediate. La file du controleur ne sait PAS attendre
    dessus : un programme televerse ne peut donc pas s'y synchroniser.
    Il faut piloter le cycle depuis le PC, ou deplacer le capteur.

  - TOUT-OU-RIEN SUR UNE ENTREE EIO. Celui-la, la file sait l'attendre :
    le programme se suspend jusqu'a ce que la piece arrive, et le PC
    peut meme etre debranche.

Le script essaie les deux, et dit lequel a bouge.
"""

import sys
import time

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim.convoyeur import PORTS_GP
from dobot_sim.liaison import (ErreurLiaison, FonctionES, LiaisonDobot,
                               ports_disponibles)

#: entrees EIO balayees. Au-dela, on sort des bornes de la carte.
EIO = range(1, 21)
DUREE = 20.0


def compte_a_rebours(secondes: int = 5) -> None:
    print("\n  PREPAREZ-VOUS : un objet a la main, pres du capteur.")
    for reste in range(secondes, 0, -1):
        print("    {}...".format(reste))
        time.sleep(1.0)
    print("\n  >>> PASSEZ L'OBJET DEVANT LE CAPTEUR, PLUSIEURS FOIS <<<")
    print("  >>> pendant les {:.0f} secondes qui viennent          <<<\n"
          .format(DUREE))


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        ports = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not ports:
            sys.exit("aucun port serie")
        port = ports[0]
    print("port", port)

    noms_gp = {v: k for k, v in PORTS_GP.items()}
    li = LiaisonDobot(port).ouvrir()
    try:
        print("firmware", li.version())
        li.arreter_file()
        li.vider_file()
        li.demarrer_file()

        # on allume les quatre infrarouges : si la correspondance
        # GP4 -> index 2 du code etait fausse, un autre index repondra
        for index in sorted(noms_gp):
            try:
                li.activer_capteur_ir(index, True)
            except ErreurLiaison as exc:
                print("  {} : activation refusee ({})"
                      .format(noms_gp[index], exc))

        # et on declare les entrees EIO en entree, pour pouvoir les lire
        eio_lisibles = []
        for p in EIO:
            try:
                li.configurer_port(p, FonctionES.ENTREE)
                eio_lisibles.append(p)
            except ErreurLiaison:
                pass

        compte_a_rebours()

        vus_gp = {i: set() for i in noms_gp}
        vus_eio = {p: set() for p in eio_lisibles}
        t0 = time.time()
        while time.time() - t0 < DUREE:
            for index in sorted(noms_gp):
                try:
                    etat = bool(li.capteur_ir(index))
                except ErreurLiaison:
                    continue
                if etat not in vus_gp[index]:
                    vus_gp[index].add(etat)
                    if len(vus_gp[index]) > 1:
                        print("  t={:4.1f}s  {} a CHANGE -> {}"
                              .format(time.time() - t0, noms_gp[index], etat))
            for p in eio_lisibles:
                try:
                    etat = bool(li.entree(p))
                except ErreurLiaison:
                    continue
                if etat not in vus_eio[p]:
                    vus_eio[p].add(etat)
                    if len(vus_eio[p]) > 1:
                        print("  t={:4.1f}s  EIO {} a CHANGE -> {}"
                              .format(time.time() - t0, p, etat))

        print("\n" + "=" * 62)
        print("BILAN")
        print("=" * 62)
        bouges_gp = [i for i, v in vus_gp.items() if len(v) > 1]
        bouges_eio = [p for p, v in vus_eio.items() if len(v) > 1]

        for index in sorted(noms_gp):
            etats = sorted(vus_gp[index])
            print("  {} (index {}) : {}".format(
                noms_gp[index], index,
                "A CHANGE" if len(etats) > 1
                else "fige sur {}".format(etats[0] if etats else "rien")))
        if bouges_eio:
            print("  entrees EIO qui ont bouge : {}"
                  .format(", ".join(str(p) for p in bouges_eio)))
        else:
            print("  aucune entree EIO n'a bouge")

        print()
        if bouges_eio:
            p = bouges_eio[0]
            print("  -> Le capteur se lit sur l'ENTREE EIO {}. C'est le bon".format(p))
            print("     cas : un programme peut l'attendre dans la file.")
            print("     Dans votre cellule : capteur_eio={}".format(p))
        elif bouges_gp:
            index = bouges_gp[0]
            print("  -> Le capteur est l'INFRAROUGE sur {}.".format(noms_gp[index]))
            print("     Dans votre cellule : capteur_port=PORTS_GP[\"{}\"]"
                  .format(noms_gp[index]))
            print("     ATTENTION : la file ne sait pas attendre dessus. Un")
            print("     programme televerse ne s'y synchronisera pas ; il")
            print("     faut piloter le cycle depuis le PC, ou recabler le")
            print("     capteur sur une entree EIO.")
        else:
            print("  -> RIEN n'a bouge. Le capteur n'est pas alimente, pas")
            print("     branche la ou on croit, ou l'objet est passe trop")
            print("     loin : l'infrarouge du kit porte quelques")
            print("     centimetres seulement. Verifiez sa diode temoin.")
    finally:
        for index in sorted(noms_gp):
            try:
                li.activer_capteur_ir(index, False)
            except ErreurLiaison:
                pass
        li.arreter_file()
        li.vider_file()
        li.fermer()
        print("\n  capteurs eteints, liaison fermee")


if __name__ == "__main__":
    main()
