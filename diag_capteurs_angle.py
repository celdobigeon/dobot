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
#: Elle peut rater un capteur regulierement faux -- stable mais decale
#: --, ce que l'equerre, elle, verrait.
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
    print("  essai |   J1      J2      J3      J4")
    lectures = []
    for k in range(ESSAIS):
        try:
            li = LiaisonDobot(port).ouvrir()
        except ErreurLiaison as exc:
            print("    {}   ouverture impossible : {}".format(k + 1,
                                                              str(exc)[:40]))
            continue
        try:
            a = np.asarray(li.pose().angles, dtype=float)
        finally:
            li.fermer()
        lectures.append(a)
        print("    {}   | {:6.1f} {:6.1f} {:6.1f} {:6.1f}"
              .format(k + 1, *a))
        time.sleep(0.8)

    if len(lectures) < 2:
        sys.exit("\n  pas assez de lectures pour conclure")

    L = np.array(lectures)
    noms = ("J1 base", "J2 bras arriere", "J3 avant-bras", "J4 outil")
    print("")
    print("=" * 64)
    print("BILAN")
    print("=" * 64)
    fautifs = []
    for i in range(4):
        etendue = float(L[:, i].max() - L[:, i].min())
        hors = not (lim.q_min[i] <= L[:, i].min()
                    and L[:, i].max() <= lim.q_max[i])
        ecart = ""
        if i in (1, 2) and not TEL_QUEL:
            ecart = "   ecart a l'equerre {:+.1f} deg".format(
                float(np.median(L[:, i]) - EQUERRE[i - 1]))
        print("  {:16s} etendue {:6.1f} deg{}{}"
              .format(noms[i], etendue, ecart,
                      "   HORS BUTEES" if hors else ""))
        if etendue > TOLERANCE or hors:
            fautifs.append(noms[i])
        elif (i in (1, 2) and not TEL_QUEL
              and abs(np.median(L[:, i]) - EQUERRE[i - 1]) > 10.0):
            fautifs.append(noms[i])

    print("")
    if not fautifs:
        if TEL_QUEL:
            print("  LES LECTURES SE REPETENT. Le bras etant immobile, aucun")
            print("  capteur ne s'est contredit d'une connexion a l'autre.")
            print("")
            print("  Reserve : sans l'equerre, un capteur REGULIEREMENT faux")
            print("  -- stable mais decale -- passerait au travers. Pour")
            print("  l'exclure, refaire l'essai bras en equerre.")
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
