# Test.py
import numpy as np

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim import Pilote
from dobot_sim import homogeneous as hg

robot = Pilote(nom="cycle", outil="ventouse",
               vitesse=140, acceleration=600,      # MOVL, MOVC : mm/s
               vitesse_articulaire=50)             # MOVJ      : deg/s

# Une matrice homogene 4x4 EST une cible : on la donne telle quelle a
# movj / movl. Inutile de la passer a cible(), qui sert au contraire a en
# CONSTRUIRE une a partir de coordonnees (x, y, z, rz).
#
#   colonne 0 : axe x de l'outil  (-1, 0, 0)
#   colonne 1 : axe y de l'outil  ( 0, 1, 0)
#   colonne 2 : axe z de l'outil  ( 0, 0,-1)   vers le BAS = direction d'approche
#   colonne 3 : position          ( 0, 220, 0) en mm
#
# y = 220 mm : a 100 mm l'ancienne cible tombait dans la zone morte autour
# de la colonne (il aurait fallu J3 = 127 deg, la butee est a 95 deg).
PRISE = np.array([[-1, 0, 0, 0],
                  [0, 1, 0, 220],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]], dtype=float)

HAUT = hg.translation(0, 0, -85)   # 85 mm au-dessus, exprime dans l'axe outil

robot.repli()
robot.movj(PRISE @ HAUT, lissage=70)
robot.movl(PRISE, vitesse=50, profil="trapeze")
robot.ventouse(True)
robot.attendre(0.3)
robot.movl(PRISE @ HAUT)
robot.repli()

print(robot.resume())     # durées, avertissements, butées
robot.simuler()           # ouvre la fenêtre et lance l'animation
