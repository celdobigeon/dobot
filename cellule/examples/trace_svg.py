#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracer un dessin au porte-stylo, VECTORIEL ou IMAGE.

    python examples/trace_svg.py                 le dessin ci-dessous
    python examples/trace_svg.py Loup.svg        un fichier de chemins
    python examples/trace_svg.py loup.png        une image

L'extension decide : .svg suit les chemins du fichier, une image suit le
CONTOUR de ce qui est encre. Le dessin est mis a l'echelle pour tenir
dans la largeur demandee, centre sur le repere du papier, puis trace
trait par trait : liaison en l'air, descente, trace, levee.

Ce qu'une image donne, et ce qu'elle ne donne pas. On suit une
frontiere : un logo plein, une silhouette, un dessin au trait epais
sortent bien. Un trait y est suivi des DEUX cotes -- c'est son contour,
pas son axe. Une photographie, un degrade, une image bruitee n'ont pas
de frontiere nette et donnent des milliers de contours sans interet ;
seuillez-la d'abord dans un editeur d'images.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import programme_depuis_dessin, repere_travail   # noqa: E402

ICI = os.path.dirname(os.path.abspath(__file__))

# Le plan du papier. z est l'altitude de la feuille sur la table : mesurez
# la votre, epaisseur comprise. Deplacer la feuille ne demande de changer
# que cette ligne.
FEUILLE = repere_travail(x=235, y=0, z=10)

chemin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ICI, "Loup.svg")
# dessin_exemple

# Reglages propres aux IMAGES, ignores pour un SVG. Les trois premiers
# suffisent presque toujours quand le contour ne sort pas comme voulu.
IMAGE = dict(
    seuil=None,          # None = seuil choisi tout seul (Otsu), sinon 0..255
    inverser=False,      # True si le dessin est CLAIR sur fond sombre
    flou=3,              # lissage prealable, pixels impairs ; 5 ou 7 si bruite
    trous=True,          # suivre aussi les contours interieurs (un oeil, un O)
)

robot = programme_depuis_dessin(
    chemin,
    repere=FEUILLE,
    largeur=80,          # mm -- le dessin tient dans cette largeur
    levee=10,            # mm au-dessus du papier entre deux traits
    profondeur=0,        # negatif pour appuyer le stylo
    vitesse_trace=60,    # mm/s crayon pose
    tolerance=0.5,       # mm -- finesse du decoupage des courbes
    journal=print,
    **(IMAGE if not chemin.lower().endswith(".svg") else {}))

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
