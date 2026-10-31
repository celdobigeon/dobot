#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRIGE de l'activite d'empilement -- cellule/ACTIVITE_EMPILEMENT.md

    python corrige_empilement.py              simulation
    python corrige_empilement.py --reel       sur la machine

NE PAS METTRE DANS cellule/. Ce fichier vit dans l'atelier parce que
l'activite consiste precisement a l'ecrire. Le donner, c'est la
supprimer.

Ce que l'activite demande, et ou c'est fait ici
-----------------------------------------------
    1. mise en route                    -- hors du script, dobot.py
    2. mesure et matrice de l'outil     -- section 1
    3. repere de prise, en matriciel    -- section 2
    4. cibles, approches par @          -- section 3
    5. prises exprimees dans le repere  -- section 3
    6. boucle                           -- section 4
    7. decalage d'empilement            -- section 4

UNE NOTE POUR L'ENSEIGNANT, sur la longueur d'outil
----------------------------------------------------
Il reste un point non tranche dans le code : `definir_repere_outil(T)`
et `regler_longueur_outil(L)` ne commandent PAS la meme hauteur, parce
que seule la seconde ecrit `geo.lt`, qui entre dans la conversion du
pont vers le controleur. Ecart mesure : 27,3 mm.

L'activite contourne la question par la bonne methode d'atelier : on
mesure, on declare, ON VERIFIE PAR UNE DESCENTE, et on corrige la
valeur si l'ecart est constant. Les etudiants apprennent ainsi la
boucle mesure-verification, qui vaut mieux qu'un nombre donne tout
fait -- et le defaut du code ne les atteint pas.
"""

import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

from dobot_sim import Pilote, repere_travail
from dobot_sim import homogeneous as hg

REEL = "--reel" in sys.argv

# ---------------------------------------------------------------------------
# 1. L'OUTIL
# ---------------------------------------------------------------------------
#
# De l'axe de rotation de l'outil a sa pointe, axe z vers le BAS.
#
#   colonne 0 : ( 1,  0,  0)     x inchange
#   colonne 1 : ( 0, -1,  0)     y retourne
#   colonne 2 : ( 0,  0, -1)     z vers le bas : direction d'approche
#   colonne 3 : ( 0,  0, -L)     la pointe, L millimetres sous la bride
#
# L se MESURE (cellule/mesurer_outil.py) puis se VERIFIE par une
# descente sur un plan de cote connue. Une erreur de longueur d'outil
# est CONSTANTE : elle ne depend pas de la posture, et c'est ce qui
# permet de l'imputer a l'outil plutot qu'au repere ou a l'origine.
L_OUTIL = 79.3

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot = Pilote(nom="empilement", outil="ventouse",
               vitesse=100, acceleration=300, vitesse_articulaire=50)
robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 2. LES DEUX REPERES DE PROGRAMMATION
# ---------------------------------------------------------------------------
#
# Donnes PAR RAPPORT A LA BASE DU ROBOT, et ecrits en matrices : c'est
# l'objet de l'activite.
#
# Un repere de travail a son axe z vers le HAUT -- c'est une table, pas
# un outil. D'ou le contraste avec OUTIL ci-dessus, dont le z descend.
#
#   colonne 3 : ou est l'origine du repere, dans la base
#   colonnes 0-2 : comment il est oriente
#
# Ici les deux reperes sont paralleles a la base : partie rotation
# identite. Les incliner d'un quart de tour ne changerait qu'elle.

# le magasin : la ou on prend les pieces
MAGASIN = np.array([[1, 0, 0,  50],
                    [0, 1, 0, 220],
                    [0, 0, 1,   0],
                    [0, 0, 0,   1]], dtype=float)

# le chantier : la ou on empile
CHANTIER = np.array([[1, 0, 0, 150],
                     [0, 1, 0, 150],
                     [0, 0, 1,   0],
                     [0, 0, 0,   1]], dtype=float)

# Verification : repere_travail() construit la meme chose a partir de
# coordonnees. On s'en sert pour CONTROLER ce qu'on a ecrit a la main,
# pas pour le remplacer -- l'activite demande la matrice.
assert np.allclose(MAGASIN, repere_travail(x=50, y=220))
assert np.allclose(CHANTIER, repere_travail(x=150, y=150))

# ---------------------------------------------------------------------------
# 3. LES CIBLES
# ---------------------------------------------------------------------------

PIECE = 25.0            # cube de 25 mm, mesure
PILE = 3                # hauteur de la tour, en pieces
#
# POURQUOI 3 ET PAS 10. Plus la pile monte, plus le bras doit se
# redresser pour l'approcher par le dessus, et J2 se rapproche de sa
# butee. A 4 pieces avec 70 mm de degagement, `controler()` signale
# deja 4,8 deg de marge. L'exercice est de le constater HORS LIGNE,
# avant que la machine ne le decouvre.

# DANS LE MAGASIN : la ventouse sur le dessus de la piece.
#
#   z = PIECE   le dessus du cube, le magasin etant pose sur la table
#   rotation    z vers le bas, sans quart de tour
#
# LE QUART DE TOUR SE CHOISIT, IL NE SE SUBIT PAS. Une ventouse se
# moque de son orientation autour de z, mais le POIGNET, lui, a des
# butees : J4 doit rester dans [-90 ; +90]. Un quart de tour ecrit ici
# donnait J4 = -167 deg, et `controler()` l'a dit hors ligne, avant que
# rien ne bouge. Avec une PINCE, l'orientation serait imposee par la
# piece et il faudrait deplacer le magasin plutot que tordre le poignet.
DANS_LE_MAGASIN = np.array([[1,  0,  0,     0],
                            [0, -1,  0,     0],
                            [0,  0, -1, PIECE],
                            [0,  0,  0,     1]], dtype=float)

# La cible de prise est donc le repere, COMPOSE avec la situation de la
# piece dans ce repere. Deplacer le magasin ne demande de changer que
# MAGASIN : c'est tout l'interet.
PRISE = MAGASIN @ DANS_LE_MAGASIN

# Le degagement, exprime dans le repere de l'OUTIL. L'axe z de l'outil
# pointant vers le bas, un degagement VERS LE HAUT s'ecrit avec un z
# NEGATIF -- c'est le piege classique, et il se verifie : la cote z de
# PRISE @ HAUT doit etre plus GRANDE que celle de PRISE.
HAUT = hg.translation(0, 0, -70)

assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"


def depose(n):
    """
    Cible de depose de la n-ieme piece, n comptee a partir de zero.

    LE DECALAGE D'EMPILEMENT, et c'est le coeur de l'exercice : la
    n-ieme piece se pose sur les n precedentes, donc n epaisseurs plus
    haut. Une seule ligne change, et la tour monte.

    Ecrit comme une TRANSLATION dans le repere du chantier, et non en
    retouchant la cote z d'une matrice recopiee : une structure de
    kapla se decrit par des deplacements relatifs, pas par une liste de
    hauteurs absolues qu'il faudrait toutes reprendre si la piece
    changeait d'epaisseur.
    """
    dans_le_chantier = np.array([[1,  0,  0,                  0],
                                 [0, -1,  0,                  0],
                                 [0,  0, -1, PIECE * (n + 1.0)],
                                 [0,  0,  0,                  1]],
                                dtype=float)
    return CHANTIER @ dans_le_chantier


# ---------------------------------------------------------------------------
# 4. LE CYCLE
# ---------------------------------------------------------------------------

robot.repli()
for n in range(PILE):
    cible_depose = depose(n)

    # prise : approche, descente, aspiration, retrait
    robot.movj(PRISE @ HAUT, lissage=70)
    robot.movl(PRISE, vitesse=40)
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(PRISE @ HAUT)

    # depose : meme geste, a la hauteur de la pile
    robot.movj(cible_depose @ HAUT, lissage=70)
    robot.movl(cible_depose, vitesse=40)
    robot.ventouse(False)
    robot.attendre(0.3)
    robot.movl(cible_depose @ HAUT)
robot.repli()

# ---------------------------------------------------------------------------
# 5. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

print("outil : pointe a {:.1f} mm sous la bride".format(L_OUTIL))
print("prise   : x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
for n in range(PILE):
    print("depose {} : x={:.1f}  y={:.1f}  z={:.1f}"
          .format(n + 1, *depose(n)[:3, 3]))
print("")
print(robot.resume().splitlines()[0])

avertissements = robot.controler()
if avertissements:
    print("")
    print("reserves :")
    for a in avertissements[:6]:
        print("  - " + a)
else:
    print("tout le cycle est dans l'espace de travail")

if REEL:
    robot.executer_sur_robot(vitesse=30)
else:
    robot.simuler()
