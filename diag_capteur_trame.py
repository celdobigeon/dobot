#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sous quelle FORME DE TRAME ce controleur lit-il son capteur infrarouge ?

    python diag_capteur_trame.py COM4

RIEN NE BOUGE : ni le bras, ni le tapis. On ne fait qu'interroger.

Le defaut qu'on cherche
-----------------------
Sur le robot 5 (firmware 3.3.0), le capteur du kit est bien branche sur
GP4 et sa diode s'allume quand la piece passe. Le controleur, lui,
annonce les quatre ports GP figes sur True -- il ne lit rien.

Or la commande 138 (SetInfraredSensor / GetInfraredSensor) a change de
forme entre les versions du kit. Nous envoyons celle du kit v2 :

    activer : uint8 actif, uint8 port, uint8 VERSION
    lire    : uint8 port, uint8 VERSION      -> uint8 present

L'ancienne forme n'a pas cet octet de version. Et comme ce controleur
ACQUITTE N'IMPORTE QUELLE TRAME -- verifie sur des identifiants de
commande qui n'existent pas -- une activation mal formee ne proteste
pas : elle ne fait rien, et la lecture rend une valeur par defaut.

C'est exactement le genre de panne qu'on ne trouve pas en relisant le
programme : tout y est correct, pour l'autre firmware.

Ce que fait le script
---------------------
Il essaie les combinaisons d'activation et de lecture, l'une apres
l'autre, et retient celles dont la valeur CHANGE pendant que vous
passez la main. Une valeur qui change est une valeur lue ; une valeur
figee ne prouve rien, quelle que soit sa couleur.

PASSEZ LA MAIN DEVANT LE CAPTEUR SANS ARRET pendant tout l'essai --
environ une minute. Passages lents, a deux ou trois centimetres : la
portee de l'infrarouge du kit n'excede guere cela.
"""

import struct
import sys
import time

from dobot_sim.liaison import (CommandeDobot, ErreurLiaison, LiaisonDobot,
                               ports_disponibles)
from dobot_sim.convoyeur import PORTS_GP

PORT_GP = PORTS_GP["GP4"]
DUREE = 6.0            # secondes de scrutation par combinaison

#: (nom, octets d'activation, octets de lecture)
FORMES = [
    ("v2 : activer(actif, port, version=1) / lire(port, 0)",
     lambda p: struct.pack("<3B", 1, p, 1),
     lambda p: struct.pack("<2B", p, 0)),
    ("v1 : activer(actif, port)            / lire(port)",
     lambda p: struct.pack("<2B", 1, p),
     lambda p: struct.pack("<B", p)),
    ("mixte : activer(actif, port, 0)      / lire(port)",
     lambda p: struct.pack("<3B", 1, p, 0),
     lambda p: struct.pack("<B", p)),
    ("mixte : activer(actif, port)         / lire(port, 0)",
     lambda p: struct.pack("<2B", 1, p),
     lambda p: struct.pack("<2B", p, 0)),
]


def scruter(li, lire, duree):
    """Lit sans relache et rend (valeurs vues, nombre de lectures)."""
    vues, n = set(), 0
    fin = time.time() + duree
    while time.time() < fin:
        try:
            p = li.echanger(CommandeDobot.CAPTEUR_IR, lire(PORT_GP))
        except ErreurLiaison as exc:
            return {"erreur : " + str(exc)[:30]}, n
        vues.add(p[0] if p else "reponse vide")
        n += 1
        time.sleep(0.05)
    return vues, n


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie : robot eteint ou debranche ?")
        port = dispo[0]

    print(__doc__.split("Le defaut qu'on cherche")[0].strip())
    print("")
    li = LiaisonDobot(port).ouvrir()
    print("")
    print("=" * 64)
    print("  PASSEZ LA MAIN DEVANT LE CAPTEUR, SANS ARRET,")
    print("  pendant tout l'essai -- environ une minute.")
    print("=" * 64)
    try:
        input("  Entree quand vous etes pret... ")
    except EOFError:
        pass

    resultats = []
    try:
        for nom, activer, lire in FORMES:
            print("")
            print("  " + nom)
            try:
                li.echanger(CommandeDobot.CAPTEUR_IR, activer(PORT_GP),
                            ecriture=True)
            except ErreurLiaison as exc:
                print("    activation refusee : {}".format(str(exc)[:40]))
                continue
            time.sleep(0.2)
            vues, n = scruter(li, lire, DUREE)
            bouge = len(vues) > 1
            print("    {} lectures, valeurs vues : {}   -> {}".format(
                n, sorted(str(v) for v in vues),
                "LE CAPTEUR REPOND" if bouge else "fige"))
            resultats.append((nom, bouge, vues))
            # eteindre avant la forme suivante, pour que chacune parte
            # du meme etat
            try:
                li.echanger(CommandeDobot.CAPTEUR_IR,
                            struct.pack("<3B", 0, PORT_GP, 1), ecriture=True)
                li.echanger(CommandeDobot.CAPTEUR_IR,
                            struct.pack("<2B", 0, PORT_GP), ecriture=True)
            except ErreurLiaison:
                pass
            time.sleep(0.2)
    finally:
        li.fermer()

    print("")
    print("=" * 64)
    print("BILAN")
    print("=" * 64)
    gagnantes = [n for n, bouge, _ in resultats if bouge]
    if gagnantes:
        for n in gagnantes:
            print("  REPOND : " + n)
        print("")
        print("  Portez cette forme dans LiaisonDobot.activer_capteur_ir")
        print("  et capteur_ir, en la choisissant sur la version du")
        print("  firmware -- l'autre machine, en 3.7.0, lit avec la forme")
        print("  v2 et ne doit pas changer.")
    else:
        print("  AUCUNE forme n'a vu bouger quoi que ce soit.")
        print("")
        print("  Le capteur s'allume donc sans que le controleur le lise.")
        print("  Restent : un port GP different de celui qu'on croit, une")
        print("  entree abimee, ou un firmware qui ignore cette commande.")
        print("  Essayez diag_capteur.py sur les quatre ports, capteur")
        print("  deplace physiquement d'un port a l'autre.")


if __name__ == "__main__":
    main()
