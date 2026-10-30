#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reparer le point de parcage ecrit dans le controleur.

    python reparer_point_origine.py COM4

Ou le bras se gare A LA FIN de la prise d'origine est une valeur
PERSISTANTE, rangee dans la flash du controleur. Elle survit a
l'extinction, et rien dans l'interface ne la montre.

Le defaut qu'on repare
----------------------
Sur le robot 5 (firmware 3.3.0), le 30/10/2026, cette valeur garait le
bras a J1 = -135,2 deg, pour une butee a -135. La prise d'origine
reussissait, la base retrouvait sa reference -- et le bras finissait
HORS COURSE. Au premier mouvement demande, le controleur levait
0x41 (J1 en butee negative) et refusait tout. Une machine saine,
immobilisee par un nombre en flash.

Rien ne le dit : `GetHOMEParams` renvoie ZERO octet sur ce firmware,
le controleur acquitte et ne repond pas. On ne peut donc pas relire la
valeur fautive ; on ne peut que la remplacer, et verifier au geste.

Ce que le script ecrit
----------------------
Pas une valeur calculee : celle que la machine annonce elle-meme quand
elle est AU REPLI. Le bras y va, on lit ce que le controleur dit de sa
propre position, et c'est cela qu'on ecrit. Aucune conversion de
repere ne s'interpose, donc aucune ne peut se tromper.

Le repli -- J = (0, 15, 30, 0), soit x = 222, y = 0, z = 134 -- est la
posture que prennent deja tous les programmes en fin de cycle. Il
surplombe la bande du convoyeur, mais 74 mm au-dessus d'elle.

Apres, verifiez
---------------
    python dobot.py --port COM4 origine
    python dobot.py --port COM4 etat

Le bras doit se garer au repli, et `etat` annoncer J1 proche de zero.
"""

import sys

from dobot_sim import Commande, DobotMagician
from dobot_sim.liaison import alarmes_bloquantes, resume_alarmes
from dobot_sim.pilote import _fermer_liaison_directe, _liaison_directe


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else None

    print(__doc__.split("Le defaut qu'on repare")[0].strip())
    print("")
    print("LE BRAS VA SE REPLIER.")

    c = Commande(simulateur=False, robot=True, port=port, outil="ventouse")
    try:
        c.repli()
        x, y, z, r = c.pose()
        q = c.angles()
    finally:
        c.fermer()

    print("")
    print("  au repli, le controleur annonce :")
    print("    x {:7.1f}  y {:7.1f}  z {:7.1f}  r {:7.1f}".format(x, y, z, r))
    print("    J1 {:+7.2f}  J2 {:+7.2f}  J3 {:+7.2f}  J4 {:+7.2f}".format(*q))

    lim = DobotMagician().lim
    hors = [j for j in range(4)
            if not (lim.q_min[j] <= q[j] <= lim.q_max[j])]
    if hors:
        print("")
        print("  REFUS : le bras annonce une posture hors butees. On")
        print("  n'ecrit pas en flash un point qu'on ne comprend pas.")
        return 2

    # Le point de parcage vit dans les coordonnees du CONTROLEUR. La pose
    # lue ci-dessus en vient, mais `Commande.pose` l'a convertie dans le
    # repere du modele ; on relit donc la brute, sans conversion.
    pont = _liaison_directe(DobotMagician(), port)
    try:
        li = pont.liaison
        p = li.pose()
        print("")
        print("  en coordonnees du controleur :")
        print("    x {:7.1f}  y {:7.1f}  z {:7.1f}  r {:7.1f}"
              .format(p.x, p.y, p.z, p.r))
        li.definir_point_origine(p.x, p.y, p.z, p.r)
        print("")
        print("  point de parcage ecrit en flash.")
        reste = alarmes_bloquantes(li.alarmes())
        for ligne in resume_alarmes(reste):
            print("  reste : " + ligne)
    finally:
        _fermer_liaison_directe()

    print("")
    print("  A VERIFIER MAINTENANT, et c'est la seule verification")
    print("  possible -- le controleur ne relit pas cette valeur :")
    print("    python dobot.py --port {} origine".format(port or "COMx"))
    print("    python dobot.py --port {} etat".format(port or "COMx"))
    print("  le bras doit se garer au repli, J1 proche de zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
