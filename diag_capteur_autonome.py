#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le cycle peut-il devenir AUTONOME, PC debranche ?

    python diag_capteur_autonome.py COM4

RIEN NE BOUGE : ni le bras, ni le tapis. On ne fait que lire.

La question, et pourquoi elle decide de la forme du programme
-------------------------------------------------------------
Le capteur du kit se lit de DEUX facons, et elles n'ont pas la meme
valeur :

  * la COMMANDE 138 du kit (`capteur_ir`) est IMMEDIATE. La file du
    controleur ne sait pas s'y synchroniser : le cycle doit etre mene
    depuis le PC, morceau par morceau, et le robot reste attache a
    l'ordinateur du debut a la fin.

  * la LECTURE EN ENTREE NUMERIQUE (`entree(n)`) se laisse ATTENDRE par
    la file. Le cycle redevient alors un programme d'un seul tenant,
    televerse une fois, et le PC peut etre debranche.

Sur le robot 5, le 30/10/2026, `diag_capteur.py` a vu GP5 et l'entree
EIO 5 bouger ENSEMBLE, en sens inverse : c'est la meme broche, vue par
les deux voies. De quoi esperer l'autonomie.

Mais cet essai-la activait la fonction infrarouge sur les quatre ports.
Si c'est ELLE qui pilote la broche, l'EIO ne bougerait plus des que la
fonction est eteinte -- et l'autonomie serait une illusion. C'est
precisement ce qui n'etait pas verifie.

Ce que fait le script
---------------------
Deux temps, et c'est leur COMPARAISON qui conclut :

    temps 1   fonction infrarouge ETEINTE     -> on lit les EIO seules
    temps 2   fonction infrarouge ALLUMEE     -> on lit EIO et GP

Une EIO qui bouge AU TEMPS 1 est lisible pour elle-meme : le cycle peut
etre autonome. Une EIO qui ne bouge qu'au temps 2 est pilotee par la
fonction infrarouge, et il faudra rester attache au PC -- ou recabler
le capteur sur une entree tout-ou-rien.

PASSEZ LA MAIN DEVANT LE CAPTEUR SANS ARRET pendant chacun des deux
temps : quinze secondes chacun. Une valeur figee ne prouve rien.
"""

import sys
import time

from dobot_sim.convoyeur import PORTS_GP
from dobot_sim.liaison import (ErreurLiaison, FonctionES, LiaisonDobot,
                               ports_disponibles)

EIO = range(1, 21)
DUREE = 15.0


def attendre_pret(titre):
    print("")
    print("=" * 64)
    print("  " + titre)
    print("  PASSEZ LA MAIN DEVANT LE CAPTEUR, SANS ARRET,")
    print("  pendant les {:.0f} secondes qui suivent.".format(DUREE))
    print("=" * 64)
    try:
        input("  Entree pour lancer... ")
    except EOFError:
        pass


def scruter(li, eio_lisibles, noms_gp, avec_gp):
    """Lit tout pendant DUREE et rend (EIO qui ont bouge, GP qui ont bouge)."""
    vus_eio = {p: set() for p in eio_lisibles}
    vus_gp = {i: set() for i in noms_gp} if avec_gp else {}
    t0 = time.time()
    while time.time() - t0 < DUREE:
        for p in eio_lisibles:
            try:
                vus_eio[p].add(bool(li.entree(p)))
            except ErreurLiaison:
                pass
        if avec_gp:
            for i in noms_gp:
                try:
                    vus_gp[i].add(bool(li.capteur_ir(i)))
                except ErreurLiaison:
                    pass
    return ([p for p, v in vus_eio.items() if len(v) > 1],
            [i for i, v in vus_gp.items() if len(v) > 1])


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie : robot eteint ou debranche ?")
        port = dispo[0]

    print(__doc__.split("La question,")[0].strip())
    noms_gp = {v: k for k, v in PORTS_GP.items()}
    li = LiaisonDobot(port).ouvrir()
    try:
        # -- temps 1 : fonction infrarouge ETEINTE -----------------------
        for i in noms_gp:
            try:
                li.activer_capteur_ir(i, False)
            except ErreurLiaison:
                pass
        eio_lisibles = []
        for p in EIO:
            try:
                li.configurer_port(p, FonctionES.ENTREE)
                eio_lisibles.append(p)
            except ErreurLiaison:
                pass
        attendre_pret("TEMPS 1 : fonction infrarouge ETEINTE")
        eio1, _ = scruter(li, eio_lisibles, noms_gp, avec_gp=False)
        print("  EIO qui ont bouge : {}".format(
            ", ".join(str(p) for p in eio1) if eio1 else "aucune"))

        # -- temps 2 : fonction infrarouge ALLUMEE -----------------------
        for i in noms_gp:
            try:
                li.activer_capteur_ir(i, True)
            except ErreurLiaison:
                pass
        attendre_pret("TEMPS 2 : fonction infrarouge ALLUMEE")
        eio2, gp2 = scruter(li, eio_lisibles, noms_gp, avec_gp=True)
        print("  EIO qui ont bouge : {}".format(
            ", ".join(str(p) for p in eio2) if eio2 else "aucune"))
        print("  ports GP qui ont bouge : {}".format(
            ", ".join(noms_gp[i] for i in gp2) if gp2 else "aucun"))
    finally:
        for i in noms_gp:
            try:
                li.activer_capteur_ir(i, False)
            except ErreurLiaison:
                pass
        li.fermer()

    print("")
    print("=" * 64)
    print("BILAN")
    print("=" * 64)
    if eio1:
        print("  CYCLE AUTONOME POSSIBLE.")
        print("")
        print("  L'entree EIO {} bouge SANS la fonction infrarouge : la"
              .format(eio1[0]))
        print("  file du controleur peut donc l'attendre elle-meme. Dans")
        print("  la declaration du tapis :")
        print("")
        print("      capteur_eio={},   et plus capteur_port=..."
              .format(eio1[0]))
        print("")
        print("  Le cycle redevient un programme d'un seul tenant, et le")
        print("  PC peut etre debranche apres le televersement.")
    elif eio2 or gp2:
        print("  PAS D'AUTONOMIE : la broche ne parle que lorsque la")
        print("  fonction infrarouge est allumee, et cette fonction ne se")
        print("  lit que par commande immediate. Le cycle reste mene")
        print("  depuis le PC.")
        if gp2:
            print("")
            print("  Le capteur repond sur {} : capteur_port=PORTS_GP[\"{}\"]"
                  .format(noms_gp[gp2[0]], noms_gp[gp2[0]]))
        print("")
        print("  Pour rendre la cellule autonome il faudrait un capteur")
        print("  TOUT-OU-RIEN cable sur une entree EIO, et non")
        print("  l'infrarouge du kit sur un port GP.")
    else:
        print("  RIEN n'a bouge, dans aucun des deux temps.")
        print("")
        print("  Verifiez la diode du capteur et son branchement avant")
        print("  de conclure quoi que ce soit : ici, l'essai n'a rien")
        print("  mesure du tout.")


if __name__ == "__main__":
    main()
