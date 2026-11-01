#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRIGE de l'activite pick and place -- cellule/ACTIVITE_PICK_AND_PLACE.md

    python corrige_pick_and_place.py                  simulation
    python corrige_pick_and_place.py --reel           sur la machine
    python corrige_pick_and_place.py --lissage 0      sans arrondi
    python corrige_pick_and_place.py --angle 30       rangee en biais

NE PAS METTRE DANS cellule/. L'activite consiste a ecrire ce fichier.

Ce que l'activite demande, et ou c'est fait ici
-----------------------------------------------
    1. mise en route                       -- hors du script, dobot.py
    2. mesure et matrice de l'outil        -- section 1
    3. les deux reperes, en matriciel      -- section 2
    4. cibles, approches et retraits par @ -- sections 3 et 4
    5. prises exprimees dans le repere     -- section 3
    6. boucle                              -- section 5
    7. LE LISSAGE, et ce qu'il coute       -- section 5, le coeur

LE LISSAGE, PUISQUE C'EST L'OBJET DE L'ACTIVITE
------------------------------------------------
`lissage` est un POURCENTAGE porte par une instruction, et il concerne
le raccord avec la SUIVANTE. A 0 le robot s'arrete au point ; a 100 il
coupe le coin au maximum admissible, soit 49 % du plus court des deux
segments.

Trois choses a ne pas confondre :

  - un point lisse n'est PAS atteint. Donc on ne lisse jamais le point
    ou l'outil agit : ici la prise et la depose restent a 0 ;
  - le modele refuse de lisser autour d'une ATTENTE, parce qu'une
    temporisation demandee doit etre tenue ;
  - SUR CE CONTROLEUR-CI, le lissage n'existe pas en PTP. Le pont le
    dit a l'envoi, et le robot reel marque un arret a chaque point. Le
    simulateur montre ce qu'on gagnerait avec un controleur qui a des
    zones (ABB) ou du CNT (Fanuc).

Le gain se lit dans `robot.resume()`, qui affiche les secondes
economisees. Mesure sur ce cycle-ci, quatre planches :

    lissage    0 %  ->  95,87 s     ecart au point de degagement  0,0 mm
    lissage   40 %  ->  87,83 s                                   0,1 mm
    lissage   70 %  ->  81,79 s                                   0,7 mm
    lissage  100 %  ->  75,76 s                                   1,8 mm

Et le point de prise, lui, est atteint a 0,00 mm dans les quatre cas :
il n'est pas lisse. Le raccord melange deux lois quintiques, donc de
vitesse nulle aux bords ; les deux segments sont tous deux pres du
point pendant le recouvrement, et l'ecart geometrique reste petit. Ce
qu'on achete avec le lissage, ici, c'est 21 % de temps de cycle pour
1,8 mm -- et non un virage large.
"""

import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

from dobot_sim import Pilote
from dobot_sim import homogeneous as hg

REEL = "--reel" in sys.argv


def option(nom, defaut):
    """Valeur numerique passee en ligne de commande, ou le defaut."""
    if nom in sys.argv:
        return float(sys.argv[sys.argv.index(nom) + 1])
    return defaut


#: pourcentage d'arrondi des raccords de DEGAGEMENT. 0 a 100.
LISSAGE = option("--lissage", 70.0)
#: orientation de la rangee deposee, en degres. 0 = le long de x.
#:
#: 90 PAR DEFAUT, ET CE N'EST PAS UN GOUT. Le chantier est droit devant
#: le robot, cap 90 degres ; or J4 = orientation - J1, et J4 ne va que
#: de -90 a +90. Une rangee a 0 degre demande donc au poignet d'aller
#: exactement en butee -- `controler()` le signale. A 90 degres les
#: planches pointent dans l'axe du bras et le poignet reste a zero.
#: Essayez `--angle 0` pour voir la reserve apparaitre.
ANGLE = option("--angle", 90.0)
#: combien de planches on transporte
PLANCHES = int(option("--planches", 4))

# ---------------------------------------------------------------------------
# 1. L'OUTIL
# ---------------------------------------------------------------------------
#
# De l'axe de rotation du poignet a la pointe de la ventouse, z vers le
# BAS. Voir cellule/assets/reperes_base_outil.svg : les deux reperes
# ont le meme x, et un demi-tour autour de x retourne y et z.
#
#   colonne 0 : ( 1,  0,  0)     x inchange
#   colonne 1 : ( 0, -1,  0)     y retourne
#   colonne 2 : ( 0,  0, -1)     z vers le bas : direction d'approche
#   colonne 3 : ( 0,  0, -L)     la pointe, L millimetres sous la bride
#
# L se MESURE (cellule/mesurer_outil.py) puis se VERIFIE par une
# descente sur un plan de cote connue.
L_OUTIL = 79.3

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot = Pilote(nom="pick-and-place", outil="ventouse",
               vitesse=100, acceleration=300, vitesse_articulaire=50)
robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 2. LES DEUX REPERES
# ---------------------------------------------------------------------------
#
# Donnes PAR RAPPORT A LA BASE DU ROBOT. Un repere de TABLE a son z
# vers le haut -- d'ou le contraste avec OUTIL, dont le z descend.

LONGUEUR, LARGEUR, EPAISSEUR = 125.0, 30.0, 10.0

# le magasin : une planche presentee toujours au meme endroit, couchee
# le long de x. Hors de l'emprise du convoyeur (x = 150 a 270).
MAGASIN = np.array([[1, 0, 0,   60],
                    [0, 1, 0, -220],
                    [0, 0, 1,    0],
                    [0, 0, 0,    1]], dtype=float)

# le chantier : le debut de la rangee, de l'autre cote du robot
CHANTIER = np.array([[1, 0, 0,   0],
                     [0, 1, 0, 240],
                     [0, 0, 1,   0],
                     [0, 0, 0,   1]], dtype=float)

# ---------------------------------------------------------------------------
# 3. LA PRISE
# ---------------------------------------------------------------------------
#
# La ventouse se pose sur le DESSUS de la planche, donc a EPAISSEUR du
# plan du magasin. Pas de rotation : la planche y est deja le long de x.

DANS_LE_MAGASIN = np.array([[1,  0,  0,         0],
                            [0, -1,  0,         0],
                            [0,  0, -1, EPAISSEUR],
                            [0,  0,  0,         1]], dtype=float)

PRISE = MAGASIN @ DANS_LE_MAGASIN

# le degagement, exprime dans le repere de l'OUTIL. Son z pointant vers
# le bas, monter s'ecrit avec un z NEGATIF. On le verifie.
HAUT = hg.translation(0, 0, -80)
assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"

# la planche du simulateur : du decor, aucune instruction. magasin=True
# en fait reapparaitre une des qu'on emporte celle-ci.
robot.poser_piece(x=float(MAGASIN[0, 3]), y=float(MAGASIN[1, 3]),
                  dims=(LONGUEUR, LARGEUR, EPAISSEUR), magasin=True,
                  nom="planche")

# ---------------------------------------------------------------------------
# 4. LA DEPOSE
# ---------------------------------------------------------------------------
#
# Une rangee : les planches cote a cote, ecartees de PAS, toutes dans
# la meme orientation. Le pas depasse la largeur, sinon elles se
# toucheraient.
PAS = LARGEUR + 12.0


def depose(n):
    """
    Cible de depose de la n-ieme planche (des zero).

    LA MATRICE, COLONNE PAR COLONNE -- voir
    cellule/assets/cible_angle.svg, qui montre d'ou viennent c et s :

        colonne 0 : ( cos a,  sin a, 0)   l'axe x de l'outil, donc la
                                          LONGUEUR de la planche
        colonne 1 : ( sin a, -cos a, 0)   son y, retourne par Rx(180)
        colonne 2 : (     0,      0, -1)  son z, vers le BAS
        colonne 3 : l'ecart, perpendiculaire a la planche

    L'ecart se prend PERPENDICULAIREMENT a la planche, le long de
    (-sin a, cos a) : une rangee se range en travers, pas en long.
    """
    a = np.radians(ANGLE)
    c, s = np.cos(a), np.sin(a)
    d = (n - (PLANCHES - 1) / 2.0) * PAS           # centree sur le chantier
    dans_le_chantier = np.array([[c,  s,  0, -s * d],
                                 [s, -c,  0,  c * d],
                                 [0,  0, -1, EPAISSEUR],
                                 [0,  0,  0,      1]], dtype=float)
    return CHANTIER @ dans_le_chantier


# ---------------------------------------------------------------------------
# 5. LE CYCLE, ET LE LISSAGE
# ---------------------------------------------------------------------------
#
# LE LISSAGE NE PORTE QUE SUR LES DEGAGEMENTS. Les deux points ou
# l'outil agit -- la prise et la depose -- restent a 0 : un point lisse
# n'est pas atteint, et une ventouse qui aspire a cote n'emporte rien.

robot.repli()
for n in range(PLANCHES):
    cible = depose(n)

    # prise : approche lissee, descente franche, aspiration, retrait
    robot.movj(PRISE @ HAUT, lissage=LISSAGE)
    robot.movl(PRISE, vitesse=40)                  # lissage 0 par defaut
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(PRISE @ HAUT, lissage=LISSAGE)

    # depose : le meme geste, a l'autre bout
    robot.movj(cible @ HAUT, lissage=LISSAGE)
    robot.movl(cible, vitesse=40)
    robot.ventouse(False)
    robot.attendre(0.3)
    robot.movl(cible @ HAUT, lissage=LISSAGE)
robot.repli()

# ---------------------------------------------------------------------------
# 6. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("planche {:.0f} x {:.0f} x {:.0f} mm, {} planches, pas {:.0f} mm"
          .format(LONGUEUR, LARGEUR, EPAISSEUR, PLANCHES, PAS))
    print("lissage des degagements : {:.0f} %".format(LISSAGE))
    print("orientation de la rangee : {:+.0f} deg".format(ANGLE))
    print("")
    print("prise : x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
    for n in range(PLANCHES):
        T = depose(n)
        print("  depose {} : x={:6.1f}  y={:6.1f}  z={:5.1f}"
              .format(n + 1, T[0, 3], T[1, 3], T[2, 3]))
    print("")
    print(robot.resume().splitlines()[0])

    avertissements = robot.controler()
    if avertissements:
        print("")
        print("reserves :")
        for a in avertissements[:8]:
            print("  - " + a)
    else:
        print("tout le cycle est dans l'espace de travail")

    if REEL:
        robot.executer_sur_robot(vitesse=30)
    else:
        robot.simuler()
