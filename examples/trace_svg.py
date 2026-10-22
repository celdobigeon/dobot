#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracer un dessin SVG au porte-stylo.

    python examples/trace_svg.py                 dessin fourni
    python examples/trace_svg.py mon_logo.svg    le votre

Le dessin est mis a l'echelle pour tenir dans la largeur demandee, centre
sur le repere du papier, puis trace trait par trait : liaison en l'air,
descente, trace, levee.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import programme_depuis_svg, repere_travail      # noqa: E402

ICI = os.path.dirname(os.path.abspath(__file__))

# Le plan du papier. z est l'altitude de la feuille sur la table : mesurez
# la votre, epaisseur comprise. Deplacer la feuille ne demande de changer
# que cette ligne.
FEUILLE = repere_travail(x=235, y=0, z=10)

chemin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ICI, "dessin_exemple.svg")

robot = programme_depuis_svg(
    chemin,
    repere=FEUILLE,
    largeur=80,          # mm -- le dessin tient dans cette largeur
    levee=10,            # mm au-dessus du papier entre deux traits
    profondeur=0,        # negatif pour appuyer le stylo
    vitesse_trace=60,    # mm/s crayon pose
    tolerance=0.2,       # mm -- finesse de l'aplatissement des courbes
    journal=print)

print(robot.resume())

avertissements = robot.controler()
if avertissements:
    print("\nATTENTION :")
    for a in avertissements:
        print("  -", a)
    print("\nDeplacez la feuille ou reduisez la largeur : le bras n'atteint")
    print("pas tout le dessin depuis cette position.")
else:
    print("\nTout le dessin est dans l'espace de travail.")

robot.simuler()
