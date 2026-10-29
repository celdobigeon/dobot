#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pourquoi la piece ne s'arrete-t-elle pas au meme endroit ?

    python diag_arret.py COM6

LE BRAS NE BOUGE PAS. Le tapis, lui, tourne : degagez-le.

L'erreur observee est d'environ 100 mm et n'est pas repetable. A
60 mm/s, cent millimetres font 1,7 SECONDE de tapis -- on est trois
ordres de grandeur au-dessus du retard de la liaison serie, qui vaut
une dizaine de millisecondes. La cause est donc grossiere, et il n'y en
a que trois :

  A. LE CAPTEUR SE DECLENCHE N'IMPORTE QUAND. Un infrarouge a reflexion
     ne voit pas un objet a une distance fixe : il voit ce qui lui
     renvoie assez de lumiere. Une piece sombre ou mate est vue tard,
     une piece claire tot -- et la bande elle-meme, ou un reflet, peut
     le declencher sans piece du tout.

  B. LE TAPIS NE S'ARRETE PAS QUAND ON LE LUI DIT. L'ordre part en
     10 ms, mais rien ne garantit que le moteur s'arrete net.

     ATTENTION, la phase B ci-dessous ne le voit PAS. Elle mesure la
     distance totale entre marche et arret, et la rampe de demarrage
     perd exactement ce que la rampe d'arret gagne : le total tombe
     juste meme quand la piece depasse de 145 mm. C'est arrive ici.
     Pour mesurer le depassement SEUL : diag_depassement.py, qui se
     sert du capteur et vous fait mesurer a l'arret.

  C. LA PIECE GLISSE. Posee sur une bande deja lancee, elle patine
     avant d'etre entrainee -- et le patinage n'est jamais deux fois le
     meme.

Les quatre essais ci-dessous les separent. Les deux premiers ne
demandent rien d'autre que de REGARDER ; les deux suivants, une regle.
"""

import statistics
import sys
import time

from dobot_sim.convoyeur import PORTS_GP, IMPULSIONS_PAR_MM
from dobot_sim.liaison import ErreurLiaison, LiaisonDobot, ports_disponibles

VITESSE = 60.0                 # mm/s
GP = PORTS_GP["GP4"]


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


def journal_capteur(li, duree: float, titre: str):
    """Note chaque basculement du capteur pendant `duree` secondes."""
    print("\n  {}".format(titre))
    bascules = []
    t0 = time.perf_counter()
    precedent = None
    while time.perf_counter() - t0 < duree:
        try:
            etat = bool(li.capteur_ir(GP))
        except ErreurLiaison:
            continue
        if precedent is None:
            precedent = etat
            print("    etat initial : {}".format(etat))
            continue
        if etat != precedent:
            t = time.perf_counter() - t0
            bascules.append((t, etat))
            print("    t={:5.2f}s  -> {}".format(t, etat))
            precedent = etat
    print("    {} basculement(s) en {:.0f} s".format(len(bascules), duree))
    return bascules


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

        # ------------------------------------------------- A1
        print("\n" + "=" * 66)
        print("A1. LE CAPTEUR AU REPOS  (tapis arrete, AUCUNE piece)")
        print("=" * 66)
        print("  S'il bascule ici, il ne voit pas une piece mais du bruit :")
        print("  lumiere ambiante, reflet, ou reglage de sensibilite trop haut.")
        demander("  Degagez le capteur, puis Entree... ")
        repos = journal_capteur(li, 12.0, "12 secondes d'observation :")

        # ------------------------------------------------- A2
        print("\n" + "=" * 66)
        print("A2. LE CAPTEUR PENDANT QUE LA BANDE DEFILE  (toujours sans piece)")
        print("=" * 66)
        print("  S'il bascule ici mais pas en A1, c'est la BANDE qu'il voit :")
        print("  une jonction, une trace, une vibration. Le cycle s'arrete")
        print("  alors a un moment sans aucun rapport avec la piece.")
        demander("  Capteur toujours degage, Entree pour lancer le tapis... ")
        li.moteur_etendu(0, True, VITESSE * IMPULSIONS_PAR_MM, en_file=False)
        defile = journal_capteur(li, 12.0, "12 secondes, bande en marche :")
        li.moteur_etendu(0, False, 0.0, en_file=False)

        # ------------------------------------------------- B
        print("\n" + "=" * 66)
        print("B. LE TAPIS S'ARRETE-T-IL NET ?")
        print("=" * 66)
        print("  Collez un repere sur la bande, en face d'un point fixe du")
        print("  bati. Le tapis va tourner 3,00 s a 60 mm/s : il doit")
        print("  avancer de 180 mm exactement. L'ecart mesure, c'est la")
        print("  somme du glissement de la bande et de la distance d'arret.")
        demander("  Repere en place ? Entree... ")
        li.moteur_etendu(0, True, VITESSE * IMPULSIONS_PAR_MM, en_file=False)
        t0 = time.perf_counter()
        time.sleep(3.0)
        li.moteur_etendu(0, False, 0.0, en_file=False)
        reel = time.perf_counter() - t0
        print("  marche effective : {:.3f} s -> {:.0f} mm attendus"
              .format(reel, reel * VITESSE))
        mesure = nombre("  Distance REELLEMENT parcourue par le repere, en mm ? ")
        if mesure is not None:
            attendu = reel * VITESSE
            ecart = mesure - attendu
            print("  ecart : {:+.0f} mm ({:+.1f} %)"
                  .format(ecart, 100 * ecart / max(attendu, 1e-9)))
            print("  pas reel : {:.1f} impulsions/mm (nominal {:.1f})"
                  .format(VITESSE * IMPULSIONS_PAR_MM * reel / max(mesure, 1e-9),
                          IMPULSIONS_PAR_MM))

        # ------------------------------------------------- C
        print("\n" + "=" * 66)
        print("C. OU LA PIECE S'ARRETE-T-ELLE, ET AVEC QUELLE DISPERSION ?")
        print("=" * 66)
        print("  Cinq essais. A chaque fois : le tapis part, vous posez une")
        print("  piece a l'entree, le capteur l'arrete. Vous mesurez ensuite")
        print("  la distance entre le CAPTEUR et le bord avant de la piece.")
        print("  Comptez NEGATIF si la piece s'est arretee AVANT le capteur.")
        ecarts, attentes = [], []
        for essai in range(1, 6):
            demander("\n  essai {}/5 - Entree pour lancer le tapis... ".format(essai))
            li.moteur_etendu(0, True, VITESSE * IMPULSIONS_PAR_MM, en_file=False)
            print("    posez la piece ; j'attends le capteur (30 s max)")
            t0 = time.perf_counter()
            vu = False
            while time.perf_counter() - t0 < 30.0:
                try:
                    if li.capteur_ir(GP):
                        vu = True
                        break
                except ErreurLiaison:
                    continue
            attente = time.perf_counter() - t0
            li.moteur_etendu(0, False, 0.0, en_file=False)
            if not vu:
                print("    aucune detection en 30 s")
                continue
            attentes.append(attente)
            print("    detecte apres {:.2f} s, tapis arrete".format(attente))
            d = nombre("    distance capteur -> bord avant de la piece, en mm ? ")
            if d is not None:
                ecarts.append(d)

        # ------------------------------------------------- bilan
        print("\n" + "=" * 66)
        print("BILAN")
        print("=" * 66)
        if repos:
            print("  A1 : le capteur BASCULE AU REPOS, sans piece.")
            print("       -> c'est du bruit. Reglez sa sensibilite (vis sur")
            print("          le capteur), eloignez-le de la lumiere directe,")
            print("          ou rapprochez-le de la bande. Tant qu'il bascule")
            print("          seul, rien d'autre n'est reparable.")
        elif defile:
            print("  A2 : le capteur ne voit RIEN au repos mais bascule quand")
            print("       la bande defile -> il voit la BANDE, pas la piece.")
            print("       -> meme reglage, et verifiez qu'il ne vise pas une")
            print("          jonction ou une trace de la courroie.")
        else:
            print("  A1 et A2 : le capteur est STABLE sans piece. Bon point,")
            print("       il ne se declenche pas tout seul.")

        if ecarts:
            moy = statistics.mean(ecarts)
            etendue = max(ecarts) - min(ecarts)
            print("\n  C : {} mesures, moyenne {:+.0f} mm, etendue {:.0f} mm"
                  .format(len(ecarts), moy, etendue))
            print("      {}".format(", ".join("{:+.0f}".format(e) for e in ecarts)))
            if etendue > 20:
                print("      -> DISPERSION trop grande pour une prise. Si A1/A2")
                print("         sont propres, la piece GLISSE : posez-la tapis")
                print("         a l'arret, ou baissez la vitesse.")
            else:
                print("      -> dispersion acceptable. Le DECALAGE MOYEN, lui,")
                print("         se rattrape en apprenant le point de prise.")
        if attentes and len(attentes) > 1:
            print("\n  temps d'attente : {}"
                  .format(", ".join("{:.1f}s".format(a) for a in attentes)))
            print("      (ils refletent le moment ou vous avez pose la piece,")
            print("       pas un defaut -- sauf s'ils sont tous identiques et")
            print("       tres courts : le capteur serait alors deja actif.)")
    finally:
        try:
            li.moteur_etendu(0, False, 0.0, en_file=False)
            li.activer_capteur_ir(GP, False)
        except ErreurLiaison:
            pass
        li.arreter_file()
        li.vider_file()
        li.fermer()
        print("\n  tapis arrete, capteur eteint, liaison fermee")


if __name__ == "__main__":
    main()
