#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
La file du controleur sait-elle VRAIMENT attendre une entree ?

    python diag_attente_file.py COM4

LE BRAS NE BOUGE PAS. On ne met dans la file que des temporisations.

La question
-----------
Tout le cycle autonome repose sur une seule hypothese : que `TRIG`
(commande 120) suspende la file jusqu'a ce qu'une entree prenne un
niveau. C'est ce que dit le protocole, et c'est ce que notre code
suppose -- mais ce n'avait jamais ete VERIFIE sur la machine.

Et ce controleur acquitte n'importe quelle trame : des identifiants de
commande qui n'existent pas recoivent une reponse polie. Un TRIG
ignore ne proteste donc pas, il laisse simplement passer. Le 30/10/2026
sur le robot 5, le cycle autonome s'est deroule EN ENTIER sans attendre
la piece -- index de file arrive au bout, bras revenu au repli, et
personne pour dire pourquoi.

Comment on tranche
------------------
On ne met dans la file que ceci :

    TRIG(EIO, niveau)   puis   quelques temporisations

Puis on regarde l'INDEX DE LA COMMANDE EN COURS. S'il reste colle sur
le TRIG, la file attend : l'autonomie est possible. S'il file jusqu'au
bout, le TRIG ne bloque rien, et le cycle doit rester mene depuis le PC.

Les deux polarites sont essayees, parce qu'un seuil mal interprete
donnerait le meme symptome qu'une commande ignoree -- et ce ne sont pas
du tout les memes conclusions.
"""

import sys
import time

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim.liaison import FonctionES, LiaisonDobot, ports_disponibles

EIO = 7
TEMPOS = 4           # temporisations placees derriere le TRIG
OBSERVATION = 5.0    # secondes


def essai(li, niveau):
    """Met TRIG(niveau) + temporisations dans la file, et regarde."""
    li.arreter_file()
    li.vider_file()
    li.configurer_port(EIO, FonctionES.ENTREE)
    niveau_lu = bool(li.entree(EIO))
    i_trig = li.attendre_entree(EIO, niveau)
    for _ in range(TEMPOS):
        dernier = li.attendre(0.3)
    li.demarrer_file()

    debut = time.time()
    vus = []
    while time.time() - debut < OBSERVATION:
        vus.append(li.index_courant())
        time.sleep(0.2)
    li.arreter_file()
    li.vider_file()

    atteint = max(vus) if vus else -1
    bloque = atteint <= i_trig
    print("")
    print("  attente de EIO {} = {}   (il vaut {} en ce moment)"
          .format(EIO, niveau, niveau_lu))
    print("    index du TRIG {}, dernier de la file {}, atteint {}"
          .format(i_trig, dernier, atteint))
    print("    -> {}".format(
        "LA FILE ATTEND" if bloque else "la file est passee au travers"))
    return bloque, niveau_lu


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie : robot eteint ou debranche ?")
        port = dispo[0]

    print(__doc__.split("La question")[0].strip())
    print("")
    print("=" * 64)
    print("  NE METTEZ RIEN devant le capteur pendant cet essai :")
    print("  on veut savoir si la file attend une condition FAUSSE.")
    print("=" * 64)
    try:
        input("  Entree pour lancer... ")
    except EOFError:
        pass

    li = LiaisonDobot(port).ouvrir()
    try:
        resultats = {}
        for niveau in (False, True):
            resultats[niveau] = essai(li, niveau)
    finally:
        try:
            li.arreter_file()
            li.vider_file()
        except Exception:
            pass
        li.fermer()

    print("")
    print("=" * 64)
    print("BILAN")
    print("=" * 64)
    bloquants = [n for n, (b, _) in resultats.items() if b]
    niveau_reel = resultats[False][1]
    if not bloquants:
        print("  TRIG NE BLOQUE RIEN, dans aucune polarite.")
        print("")
        print("  La file de ce controleur ne sait pas attendre une")
        print("  entree -- ou pas par cette commande. Le cycle AUTONOME")
        print("  est impossible tel qu'il est ecrit : il faut rester sur")
        print("  le cycle mene depuis le PC.")
        print("")
        print("  Ce resultat est a prendre au serieux : la commande est")
        print("  ACQUITTEE, elle ne dit jamais qu'elle ne fait rien.")
    elif len(bloquants) == 2:
        print("  TRIG bloque dans LES DEUX polarites : l'essai ne conclut")
        print("  pas. La file s'arrete peut-etre pour une autre raison")
        print("  (defaut, file suspendue) -- relancez en regardant les")
        print("  alarmes.")
    else:
        attendue = bloquants[0]
        print("  TRIG BLOQUE sur 'EIO {} = {}', et laisse passer l'autre."
              .format(EIO, attendue))
        print("")
        print("  La file sait donc attendre : le cycle autonome tient.")
        print("  L'entree vaut {} en ce moment, sans piece devant le"
              .format(niveau_reel))
        print("  capteur -- la condition a ecrire dans le programme est")
        print("  donc celle qui est FAUSSE au repos, soit :")
        print("")
        print("      robot.attendre_piece({}, etat={})".format(EIO, attendue))


if __name__ == "__main__":
    main()
