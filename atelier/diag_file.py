#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic : que repond le controleur quand sa file est PLEINE ?

AUCUN MOUVEMENT. La file est arretee puis videe, on n'y empile que des
attentes de 0 ms (SetWAITCmd), et on la revide a la fin.

Pour chaque commande empilee, on note la longueur de la reponse et
l'index rendu. Le protocole annonce une reponse vide quand la file est
pleine ; on verifie ce que fait vraiment le firmware.

    python diag_file.py            # premier port Dobot trouve
    python diag_file.py COM6
"""

import struct
import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim.liaison import (CommandeDobot, ErreurLiaison, FilePleine,
                               LiaisonDobot, ports_disponibles)

FILE_PLACE_RESTANTE = 247      # GetQueuedCmdLeftSpace, retire du protocole V1.1.5
N_MAX = 120


def place_restante(li):
    try:
        p = li.echanger(FILE_PLACE_RESTANTE, essais=1)
    except ErreurLiaison as exc:
        return "sans reponse ({})".format(exc)
    if len(p) >= 4:
        return "{} (reponse {} octets)".format(struct.unpack("<I", p[:4])[0], len(p))
    return "reponse de {} octets : {}".format(len(p), p.hex())


def main():
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        ports = ports_disponibles()
        if not ports:
            sys.exit("aucun port serie")
        port = ports[0][0]
    print("port", port)

    li = LiaisonDobot(port, journal=print).ouvrir()
    try:
        print("firmware", li.version())
        li.arreter_file()
        li.vider_file()
        print("index courant apres vidage :", li.index_courant())
        print("place restante (247) avant  :", place_restante(li))

        precedent = None
        anomalies = 0
        for k in range(1, N_MAX + 1):
            try:
                rep = li.echanger(CommandeDobot.ATTENTE, struct.pack("<I", 0),
                                  ecriture=True, en_file=True, essais=1)
                idx = struct.unpack("<Q", rep[:8])[0]
                note = ""
                if precedent is not None and idx != precedent + 1:
                    note = "   <-- index non consecutif"
                    anomalies += 1
                if note or k <= 2 or k % 10 == 0:
                    print("{:4d}  reponse {:2d} octets  index {}{}"
                          .format(k, len(rep), idx, note))
                precedent = idx
            except FilePleine:
                print("{:4d}  REPONSE VIDE (file pleine selon le protocole)".format(k))
                anomalies += 1
            except ErreurLiaison as exc:
                print("{:4d}  PAS DE REPONSE : {}".format(k, exc))
                anomalies += 1
            if anomalies >= 4:
                break

        print("place restante (247) apres  :", place_restante(li))
        print("index courant (file arretee) :", li.index_courant())
    finally:
        try:
            li.vider_file()
            print("file videe")
        finally:
            li.fermer()


if __name__ == "__main__":
    main()
