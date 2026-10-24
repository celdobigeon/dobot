# test_prise.py
import numpy as np

from dobot_sim import Pilote
from dobot_sim import homogeneous as hg

robot = Pilote(nom="cycle", outil="pince",
               vitesse=100, acceleration=200,      # MOVL, MOVC : mm/s
               vitesse_articulaire=50)             # MOVJ      : deg/s

# Longueur MESUREE de la ventouse de cette machine, en millimetres.
#
# Les 62 mm du catalogue sont une valeur nominale ; le montage reel s'en
# ecarte. Mesure : on amene la POINTE au contact du plan de travail, on lit
# le z annonce par le modele, et on l'ajoute a la longueur courante. Ici le
# modele annoncait +17,3 mm alors que la pointe touchait, d'ou 62 + 17,3.
#
# Verifie constant en trois endroits de la table : c'est bien l'outil, et
# non un reste d'erreur d'origine articulaire, qui lui varierait d'un point
# a l'autre.
#
# A appeler APRES le constructeur : monter_outil remet la valeur nominale.
robot.regler_longueur_outil(79.3)
# robot.regler_longueur_outil(62)
# Une matrice homogene 4x4 EST une cible : on la donne telle quelle a
# movj / movl. Inutile de la passer a cible(), qui sert au contraire a en
# CONSTRUIRE une a partir de coordonnees (x, y, z, rz).
#
#   colonne 0 : axe x de l'outil  ( 1, 0, 0)
#   colonne 1 : axe y de l'outil  ( 0,-1, 0)
#   colonne 2 : axe z de l'outil  ( 0, 0,-1)   vers le BAS = direction d'approche
#   colonne 3 : position          ( 0, 220, 0) en mm
#
# y = 220 mm, et PAS 100 : a 100 mm la cible tombe dans la zone morte autour
# de la colonne. Le robot ne peut pas s'y rendre (il faudrait J3 = 127 deg,
# la butee est a 95 deg), il se replie sur lui-meme et le bandeau d'etat
# passe au rouge. A z = 0 le domaine utile va de y = 164 a y = 310 mm.
PRISE = np.array([[1,  0,  0, 220],
                  [0, -1,  0, 115],
                  [0,  0, -1,  88],
                  [0,  0,  0,   1]], dtype=float)

P2 = np.array([[1, 0, 0, 200],
               [0, -1, 0, 0],
               [0, 0, -1, 100],
               [0, 0, 0, 1]], dtype=float)

P3 = np.array([[1, 0, 0, 200],
               [0, -1, 0, -100],
               [0, 0, -1, 25],
               [0, 0, 0, 1]], dtype=float)

HAUT = hg.translation(0, 0, -85)   # 85 mm au-dessus, exprime dans l'axe outil


# robot.repli()
# robot.movj(PRISE @ HAUT, lissage=70)
# robot.movl(PRISE, vitesse=50, profil="trapeze")
# robot.ventouse(True)
# robot.attendre(0.3)
# robot.movl(PRISE @ HAUT)
# robot.movj(P2)
# robot.movj(P3 @ HAUT, lissage=10)
# robot.movl(P3)
# robot.ventouse(False)
# robot.attendre(0.3)
# robot.movj(P3 @ HAUT)
# robot.repli()


robot.repli()

robot.attendre(0.5)

robot.movj(PRISE @ HAUT, lissage=70)
robot.movl(PRISE, vitesse=100, profil="trapeze")
# robot.ventouse(True)
robot.pince(True)
robot.attendre(0.5)
robot.movj(PRISE @ HAUT, vitesse=100, profil="trapeze")
robot.attendre(0.5)
# robot.ventouse(False)
robot.pince(False, couper=True)

robot.repli()


#robot.movj(PRISE @ HAUT, lissage=70)
#robot.movl(PRISE, vitesse=50, profil="trapeze")
#robot.ventouse(True)
#robot.attendre(0.3)
#robot.movl(PRISE @ HAUT)
#robot.repli()

print(robot.resume())     # durées, avertissements, butées

# robot.simuler()           # ouvre la fenêtre et lance l'animation
robot.executer_sur_robot(vitesse=100)