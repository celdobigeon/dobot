#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Les capteurs d'angle de ce robot disent-ils la verite ?

    python diag_capteurs_angle.py COM6

RIEN NE BOUGE : on ouvre la liaison, on lit, on ferme, on recommence.

A FAIRE EN PREMIER sur toute machine qu'on decouvre, et avant toute
mise au point de precision. Trente secondes, et cela evite de chercher
pendant des heures une erreur qui n'est pas dans le programme.

Ce que l'essai etablit
----------------------
Le controleur du Magician lit ses capteurs d'angle UNE FOIS a
l'ouverture de la liaison, puis compte ses pas. Un capteur fautif donne
donc une valeur differente a chaque connexion, tout en restant
parfaitement stable a l'interieur de l'une d'elles.

C'est ce qui le rend si couteux a diagnostiquer : dans une seance, le
robot parait tantot juste tantot fou, sans qu'aucune alarme ne
distingue les deux cas. Et quand la valeur fausse reste PLAUSIBLE --
dans les butees du controleur -- rien ne se declenche du tout : le
cycle part sur un repere faux, en silence.

On met donc le bras dans une posture CONNUE, et on relit plusieurs
fois. En posture d'equerre -- bras arriere vertical, avant-bras
horizontal -- les deux angles valent ZERO sur cette convention.

Releve sur la machine au convoyeur, le 30/10/2026 :

    essai 1   J2 = -0,2   J3 =   -2,5     (juste)
    essai 2   J2 = -0,0   J3 = -162,6
    essai 3   J2 = -0,3   J3 = -157,8
    essai 4   J2 = +0,1   J3 = -161,8

La bonne valeur une fois sur quatre, et 160 degres d'erreur le reste du
temps. Son capteur J3 est fautif. Celui de l'autre robot du labo l'est
sur J2 depuis septembre.
"""

import sys
import time

import numpy as np

from dobot_sim import DobotMagician
from dobot_sim.liaison import ErreurLiaison, LiaisonDobot, ports_disponibles

ESSAIS = 5
PAUSE = 0.8            # secondes entre deux lectures

#: --essais N --pause S : suivre l'ETABLISSEMENT de la lecture.
#:
#: Apres un deplacement a la main, le controleur rend d'abord l'ANCIENNE
#: valeur, puis la nouvelle. Pour mesurer en combien de temps : relachez
#: le bras, lancez AUSSITOT avec
#:
#:     python diag_capteurs_angle.py COM9 --tel-quel --essais 20 --pause 1
#:
#: et regardez a quelle lecture la valeur bascule. Chaque lecture rouvre
#: la liaison -- c'est la seule facon de relire le capteur, le controleur
#: ne le consultant qu'a l'ouverture.
if "--essais" in sys.argv:
    ESSAIS = int(sys.argv[sys.argv.index("--essais") + 1])
if "--pause" in sys.argv:
    PAUSE = float(sys.argv[sys.argv.index("--pause") + 1])

#: posture d'equerre : les deux angles valent zero
EQUERRE = (0.0, 0.0)
TOLERANCE = 5.0        # degres

#: --tel-quel : ne rien placer a la main, ne juger que sur l'ETENDUE.
#:
#: L'equerre sert a verifier la JUSTESSE -- les angles valent zero, on
#: le sait. Mais elle n'est pas necessaire pour conclure a un capteur
#: fautif : un bras IMMOBILE qui annonce des valeurs differentes d'une
#: connexion a l'autre a deja tout dit, quelle que soit sa posture.
#:
#: Cette variante demande donc seulement de NE PLUS TOUCHER AU BRAS.
#:
#: ELLE EST FAIBLE, ET IL FAUT LE SAVOIR. Elle ne voit que la
#: REPETABILITE : un controleur qui se fixe sur une valeur fausse et s'y
#: tient passe pour sain. Releve sur le robot 2 le 30/10/2026 -- J1 lu a
#: 0,0 dix-neuf fois de suite, bras physiquement a -21 degres. Le bilan
#: le dit desormais en tete, et non en reserve de bas de page.
#:
#: Et si le bras vient d'etre DEPLACE A LA MAIN, elle ne mesure rien du
#: tout : J1 n'a pas de capteur, le controleur le compte en pas, et
#: moteurs relaches ces pas sont perdus. Prise d'origine d'abord.
TEL_QUEL = "--tel-quel" in sys.argv


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie : robot eteint ou debranche ?")
        port = dispo[0]

    lim = DobotMagician().lim
    print(__doc__.split("Ce que l'essai")[0].strip())
    print("")
    print("=" * 64)
    if TEL_QUEL:
        print("  NE TOUCHEZ PLUS AU BRAS, ou qu'il soit.")
        print("  On ne juge que sur l'ETENDUE des lectures : un bras")
        print("  immobile doit annoncer cinq fois la meme chose.")
    else:
        print("  Mettez le bras EN EQUERRE, a la main :")
        print("    bras arriere VERTICAL, avant-bras HORIZONTAL")
        print("  Dans cette posture, J2 et J3 valent ZERO.")
        print("")
        print("  (--tel-quel : sans equerre, sur l'etendue seule)")
    print("=" * 64)
    try:
        input("  Puis Entree... ")
    except (EOFError, KeyboardInterrupt):
        print("")

    print("")
    print("    t(s) |   J1      J2      J3      J4")
    lectures = []
    depart = time.time()
    for k in range(ESSAIS):
        try:
            li = LiaisonDobot(port).ouvrir()
        except ErreurLiaison as exc:
            print("   {:5.1f}   ouverture impossible : {}"
                  .format(time.time() - depart, str(exc)[:40]))
            continue
        try:
            a = np.asarray(li.pose().angles, dtype=float)
        finally:
            li.fermer()
        lectures.append(a)
        print("   {:5.1f} | {:6.1f} {:6.1f} {:6.1f} {:6.1f}"
              .format(time.time() - depart, *a))
        time.sleep(PAUSE)

    if len(lectures) < 2:
        sys.exit("\n  pas assez de lectures pour conclure")

    L = np.array(lectures)
    noms = ("J1 base", "J2 bras arriere", "J3 avant-bras", "J4 outil")

    # POURQUOI CE SCRIPT N'ACCUSE PLUS PERSONNE SANS L'EQUERRE.
    #
    # Des lectures incoherentes ont DEUX causes, et l'etendue seule ne
    # les separe pas :
    #
    #   - un capteur fautif. Machine du convoyeur, bras en equerre :
    #     J3 = -2,5 (juste) puis -162,6, -157,8, -161,8. Ce sont les
    #     lectures TARDIVES qui mentent.
    #
    #   - un releve commence trop tot apres un deplacement A LA MAIN.
    #     Robot 2, vingt lectures : la PREMIERE donnait J1 = 21,1 et
    #     J3 = 29,7, les dix-neuf suivantes J1 = 0,0 et J3 = 75,x.
    #
    # Les deux motifs sont symetriques -- dans un cas la premiere lecture
    # est la bonne, dans l'autre la mauvaise -- et aucune regle sur leur
    # POSITION ne les distingue. J'avais d'abord ecrit une telle regle :
    # elle classait le capteur reellement fautif en "premiere lecture
    # aberrante", ce qui est exactement l'erreur a ne pas faire.
    #
    # Ce qui tranche, c'est la POSTURE CONNUE : a l'equerre, on sait que
    # la bonne valeur est zero. Sans elle, on constate l'incoherence et
    # on dit comment la lever.
    print("")
    print("=" * 64)
    print("BILAN")
    print("=" * 64)
    fautifs = []
    incoherents = []
    for i in range(4):
        etendue = float(L[:, i].max() - L[:, i].min())
        if etendue > TOLERANCE:
            incoherents.append(noms[i])
        hors = not (lim.q_min[i] <= L[:, i].min()
                    and L[:, i].max() <= lim.q_max[i])
        ecart = ""
        if i in (1, 2) and not TEL_QUEL:
            ecart = "   ecart a l'equerre {:+.1f} deg".format(
                float(np.median(L[:, i]) - EQUERRE[i - 1]))
        print("  {:16s} etendue {:6.1f} deg{}{}"
              .format(noms[i], etendue, ecart,
                      "   HORS BUTEES" if hors else ""))
        if (etendue > TOLERANCE and not TEL_QUEL) or hors:
            fautifs.append(noms[i])
        elif (i in (1, 2) and not TEL_QUEL
              and abs(np.median(L[:, i]) - EQUERRE[i - 1]) > 10.0):
            fautifs.append(noms[i])

    print("")
    if TEL_QUEL:
        print("  AVERTISSEMENT -- ce que cet essai NE PEUT PAS voir.")
        print("")
        print("  Sans posture connue, il ne juge que la REPETABILITE. Or")
        print("  une valeur repetee n'est pas une valeur juste : le")
        print("  controleur peut se fixer sur une lecture fausse et s'y")
        print("  tenir. Releve sur le robot 2 le 30/10/2026 -- J1 lu a")
        print("  0,0 dix-neuf fois de suite alors que le bras etait a")
        print("  -21 degres.")
        print("")
        print("  ET SI VOUS VENEZ DE DEPLACER LE BRAS A LA MAIN, rien ici")
        print("  n'est utilisable : la rotation de base N'A PAS DE")
        print("  CAPTEUR, le controleur la compte en pas, et moteurs")
        print("  relaches ces pas ne sont pas comptes. J1 est perdu, par")
        print("  construction, jusqu'a la prochaine prise d'origine.")
        print("")
        print("  Pour conclure vraiment : prise d'origine, puis cet essai")
        print("  BRAS EN EQUERRE, sans --tel-quel.")
        print("")
    if incoherents and TEL_QUEL:
        print("  LECTURES INCOHERENTES : {}".format(", ".join(incoherents)))
        print("")
        print("  Le bras etant immobile, elles devraient se repeter. DEUX")
        print("  causes possibles, et cet essai NE LES SEPARE PAS :")
        print("")
        print("   1. le releve a commence trop tot apres un deplacement")
        print("      DU BRAS A LA MAIN. Le controleur rend alors une")
        print("      valeur perimee, puis la bonne -- environ six secondes")
        print("      plus tard, le temps d'une reouverture de liaison.")
        print("   2. un capteur d'angle fautif, et la machine ne saura")
        print("      jamais ou est son bras.")
        print("")
        print("  POUR TRANCHER : ne touchez plus au bras, attendez une")
        print("  dizaine de secondes, et relancez cet essai.")
        print("    - stable cette fois  -> c'etait le releve, pas la")
        print("      machine ;")
        print("    - encore incoherent  -> c'est le capteur. Refaites")
        print("      alors l'essai BRAS EN EQUERRE, sans --tel-quel :")
        print("      la posture connue dira quelle valeur ment.")
        return
    if not fautifs:
        if TEL_QUEL:
            print("  LES LECTURES SE REPETENT -- et c'est TOUT ce que cela")
            print("  dit. Voir l'avertissement ci-dessus : repetable n'est")
            print("  pas juste.")
        else:
            print("  CAPTEURS SAINS. Les lectures se repetent et correspondent")
            print("  a la posture. Cette machine sait ou est son bras, et une")
            print("  erreur de cycle sera a chercher ailleurs.")
    else:
        print("  CAPTEUR(S) FAUTIF(S) : {}".format(", ".join(fautifs)))
        print("")
        print("  Le controleur ne sait pas ou est son bras. AUCUNE cible ne")
        print("  sera atteinte de facon fiable, si juste soit-elle, et les")
        print("  essais reussiront par intermittence -- ceux ou la lecture")
        print("  etait bonne.")
        print("")
        print("  CONTOURNEMENT : declarer la posture d'equerre DANS LA")
        print("  LIAISON qui va travailler, et non dans une autre. Une")
        print("  declaration faite a part est perdue des que sa liaison se")
        print("  referme. Voir PROCEDURE_CONVOYEUR.md, section 0 bis, et")
        print("  le drapeau --equerre des exemples.")


if __name__ == "__main__":
    main()
