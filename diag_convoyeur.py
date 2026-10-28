#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic : le controleur accepte tout, et le tapis ne tourne pas.

DEJA ECARTE. Un premier diagnostic a essaye sept formes de trame --
Stepper1 et Stepper2, en file et en immediat, six octets compactes et
huit octets alignes. Les sept ont ete ACCEPTEES, aucune n'a fait tourner
la bande. Le controleur ne valide donc pas ce qu'on lui envoie, et la
forme de la trame n'est pas en cause. Le probleme est EN AMONT du
protocole.

Restent trois causes, et ce script les prend dans l'ordre.

  A. LE RAIL CONFISQUE LE PORT. Sur le Magician, l'axe lineaire se
     branche sur Stepper1. Si le controleur croit avoir un rail, il
     reserve ce port et ignore SetEMotor dessus -- sans rien dire. Or
     cette declaration (SetDeviceWithL) RESTE EN MEMOIRE d'une seance a
     l'autre : il suffit qu'elle ait ete posee une fois, par nous ou par
     DobotStudio. On la lit, on la retire, on reessaie.

  B. UN AUTRE CHEMIN DANS LE FIRMWARE. SetEMotorS (136) compte des
     impulsions au lieu de regler une vitesse : c'est un autre code. Et
     les index 2 et 3, au cas ou la numerotation ne soit pas celle qu'on
     croit.

  C. LE MOTEUR N'EST PAS ALIMENTE. C'est le plus probable maintenant, et
     aucune ligne de code ne peut le voir. Mais un pas-a-pas sous tension
     TIENT : son axe resiste a la main, meme immobile. Le dernier essai
     l'excite dix secondes pour que vous le touchiez. Ce test-la tranche.

    python diag_convoyeur.py COM6

LE BRAS NE BOUGE PAS. Rien n'est envoye d'autre que des commandes de
moteur etendu, et tout est arrete a la fin.
"""

import struct
import sys
import time

from dobot_sim.liaison import (CommandeDobot, ErreurLiaison, LiaisonDobot,
                               ports_disponibles)

DUREE = 2.5          # secondes de marche par essai
VITESSE_MM_S = 60.0
IMPULSIONS_PAR_MM = 282.942


def essai(li, titre, identifiant, charge, en_file=True, duree=DUREE):
    """Envoie une trame, dit ce qui en revient, laisse le temps de voir."""
    print("\n--- {} ---".format(titre))
    print("    ID {}  {}  {} octets : {}"
          .format(identifiant, "en file" if en_file else "immediat",
                  len(charge), charge.hex(" ")))
    try:
        reponse = li.echanger(identifiant, charge, ecriture=True,
                              en_file=en_file, essais=1)
        print("    accepte, reponse de {} octet(s)".format(len(reponse)))
    except ErreurLiaison as exc:
        print("    REFUSE : {}".format(exc))
        return False
    if en_file:
        li.demarrer_file()
    print("    >>> LE TAPIS TOURNE-T-IL ? ({:.0f} s) <<<".format(duree))
    time.sleep(duree)
    return True


def arreter(li, *index):
    for i in index:
        for en_file in (False, True):
            try:
                li.echanger(CommandeDobot.MOTEUR_ETENDU,
                            struct.pack("<2Bf", i, 0, 0.0),
                            ecriture=True, en_file=en_file, essais=1)
            except ErreurLiaison:
                pass
    try:
        li.demarrer_file()
    except ErreurLiaison:
        pass


def marche(index, mm_s=VITESSE_MM_S):
    return struct.pack("<2Bf", index, 1, mm_s * IMPULSIONS_PAR_MM)


# ---------------------------------------------------------------------------

def cause_a(li):
    """Le rail confisque-t-il Stepper1 ?"""
    print("\n" + "=" * 68)
    print("A. LE CONTROLEUR CROIT-IL AVOIR UN RAIL ?")
    print("=" * 68)
    try:
        declare = li.rail_present()
    except ErreurLiaison as exc:
        print("  lecture impossible ({}) -- on passe".format(exc))
        return
    print("  GetDeviceWithL repond : {}".format(
        "OUI, un rail est declare" if declare else "non, aucun rail"))
    if not declare:
        print("  Stepper1 n'est donc pas confisque par le rail. Cause A ecartee.")
        return

    print("\n  C'est une piste serieuse : le rail occupe Stepper1 et le")
    print("  controleur y ignore SetEMotor. On retire la declaration.")
    li.declarer_rail(False)
    time.sleep(0.3)
    print("  apres retrait, GetDeviceWithL repond : {}".format(li.rail_present()))
    essai(li, "Stepper1, rail retire", CommandeDobot.MOTEUR_ETENDU, marche(0))
    arreter(li, 0)
    essai(li, "Stepper2, rail retire", CommandeDobot.MOTEUR_ETENDU, marche(1))
    arreter(li, 1)
    print("\n  (le rail reste NON declare en sortie : s'il est monte sur la")
    print("   machine, relancez examples/axe_lineaire.py pour le redeclarer)")


def cause_b(li):
    """Un autre chemin de commande dans le firmware."""
    print("\n" + "=" * 68)
    print("B. UN AUTRE CHEMIN : AUTRE COMMANDE, AUTRES INDEX")
    print("=" * 68)

    # SetEMotorS : on demande un NOMBRE d'impulsions, pas une vitesse.
    # 60 mm/s pendant 2,5 s, soit 150 mm de bande.
    for index in (0, 1):
        essai(li, "Stepper{} - SetEMotorS, 150 mm comptes".format(index + 1),
              CommandeDobot.MOTEUR_ETENDU_DISTANCE,
              struct.pack("<2BfI", index, 1,
                          VITESSE_MM_S * IMPULSIONS_PAR_MM,
                          int(150 * IMPULSIONS_PAR_MM)))
        arreter(li, index)

    for index in (2, 3):
        essai(li, "index {} (au-dela des deux Stepper documentes)".format(index),
              CommandeDobot.MOTEUR_ETENDU, marche(index), duree=1.5)
        arreter(li, index)


def cause_c(li):
    """Le moteur est-il seulement alimente ?"""
    print("\n" + "=" * 68)
    print("C. LE MOTEUR EST-IL ALIMENTE ?")
    print("=" * 68)
    print("""
  Ce test ne regarde plus l'ecran : il se fait avec les doigts.

  Un moteur pas-a-pas sous tension TIENT son axe. Il resiste quand on
  essaie de le tourner a la main, meme quand il n'avance pas. Hors
  tension il tourne librement, avec juste les crans de l'aimant.

  1. AVANT : tournez la poulie du tapis a la main. Notez la resistance.
  2. On excite le moteur tres lentement (2 mm/s) pendant 10 secondes.
  3. PENDANT : retournez la poulie a la main.

     - elle resiste beaucoup plus qu'avant  -> le moteur est alimente,
       il recoit du courant mais pas les pas : c'est le cablage du
       signal, ou le mauvais port ;
     - elle tourne pareil qu'avant          -> aucun courant n'arrive au
       moteur : alimentation ou connecteur. Le logiciel n'y peut rien.
""")
    input("  tournez la poulie, puis Entree pour exciter le moteur... ")
    essai(li, "Stepper1 - excitation lente, 2 mm/s",
          CommandeDobot.MOTEUR_ETENDU, marche(0, 2.0), duree=10.0)
    arreter(li, 0)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        ports = ports_disponibles()
        if not ports:
            sys.exit("aucun port serie")
        port = ports[0][0]
    print("port", port)

    li = LiaisonDobot(port).ouvrir()
    try:
        print("firmware", li.version())
        li.arreter_file()
        li.vider_file()
        li.demarrer_file()

        cause_a(li)
        cause_b(li)
        cause_c(li)

        print("\n" + "=" * 68)
        print("CE QU'IL FAUT ME DIRE")
        print("=" * 68)
        print("  - un essai a-t-il fait tourner la bande, et lequel ?")
        print("  - a l'essai C, la poulie resistait-elle plus qu'a froid ?")
        print("  - sur quel connecteur du controleur le tapis est-il branche,")
        print("    et le kit a-t-il sa propre alimentation, branchee ?")
    finally:
        arreter(li, 0, 1, 2, 3)
        li.arreter_file()
        li.vider_file()
        print("\ntapis arrete, file videe")
        li.fermer()


if __name__ == "__main__":
    main()
