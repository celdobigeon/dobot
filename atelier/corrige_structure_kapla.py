#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRIGE de l'activite de la structure de kapla -- cellule/ACTIVITE_STRUCTURE_KAPLA.md

    python corrige_structure_kapla.py                simulation
    python corrige_structure_kapla.py --reel         sur la machine
    python corrige_structure_kapla.py --angle 90     le carre classique

NE PAS METTRE DANS cellule/. Ce fichier vit dans l'atelier parce que
l'activite consiste precisement a l'ecrire. Le donner, c'est la
supprimer.

Ce que l'activite demande, et ou c'est fait ici
-----------------------------------------------
    1. REPERES DE PROGRAMMATION           -- section 3
       (MAGASIN, STRUCTURE), et la prise
       qui s'y exprime
    2. decalage d'empilement              -- section 4
    3. ORIENTATION variable d'une assise  -- section 4, le coeur
       a l'autre
    4. ecart des deux pieces, DEDUIT    -- section 2
       de l'angle
    5. boucle                             -- section 5
    6. butees du poignet                  -- section 4, angle_outil

LES REPERES SONT LE SUJET DE LA SEANCE, PAS UN ACQUIS
------------------------------------------------------
L'activite precedente (`corrige_pick_and_place.py`) ecrit ses deux
cibles DIRECTEMENT par rapport a la base, et c'est delibere : deux
cibles, cela passe. Douze, non. L'etudiant arrive ici en ayant senti
le manque, et la section 3 lui donne la reponse.

Ce qui est en revanche acquis et seulement rappele ci-dessous : la
mise en route, la mesure et la matrice de l'outil, la composition des
degagements par @, et le lissage.

UNE NOTE POUR L'ENSEIGNANT, sur la longueur d'outil
----------------------------------------------------
`definir_repere_outil(T)` et `regler_longueur_outil(L)` ne commandent
PAS la meme hauteur : seule la seconde ecrit `geo.lt`, qui entre dans
la conversion du pont vers le controleur. Ecart mesure : 27,3 mm.

L'activite contourne la question par la bonne methode d'atelier : on
mesure, on declare, ON VERIFIE PAR UNE DESCENTE, et on corrige la
valeur si l'ecart est constant.
"""

import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

from dobot_sim import Pilote
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

robot = Pilote(nom="structure-kapla", outil="ventouse",
               vitesse=100, acceleration=300, vitesse_articulaire=50)
robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 2. LA PIECE, L'ANGLE, ET CE QUE L'ANGLE IMPOSE
# ---------------------------------------------------------------------------

LONGUEUR = 125.0        # mm, mesure
LARGEUR = 30.0
EPAISSEUR = 10.0
ASSISES = 6             # hauteur de la structure, en assises de deux pieces

#: angle dont chaque assise tourne par rapport a la precedente
ANGLE = 75.0
if "--angle" in sys.argv:
    ANGLE = float(sys.argv[sys.argv.index("--angle") + 1])

# L'ORIENTATION DE DEPART, ET POURQUOI ELLE N'EST PAS ZERO.
#
# La structure peut etre batie dans n'importe quelle orientation : la tourner
# en entier ne change rien a sa forme. Mais elle change tout pour le
# POIGNET.
#
# A la structure, la base est a J1 = 90 degres environ. Le poignet doit
# donc fournir J4 = (orientation de la piece) - 90, et il ne tourne
# que dans [-90 ; +90]. Les six assises demandent six orientations qui
# s'etalent sur 150 degres ; il faut donc CENTRER cet etalement sur la
# direction de la base.
#
#   depart =  0 deg  ->  J4 de -90 a +60   marge  0 deg : en butee
#   depart = 15 deg  ->  J4 de -75 a +75   marge 15 deg : confortable
#
# Quinze degres de rotation de la structure entiere, et le poignet respire. Rien
# dans la structure n'a change.
ANGLE_DEPART = 15.0

# LE DECALAGE N'EST PAS LIBRE : IL SE DEDUIT DE L'ANGLE.
#
# Une assise est faite de DEUX pieces paralleles, ecartees de
# +-DECALAGE de part et d'autre du centre. L'assise suivante, tournee
# de ANGLE, repose sur elles -- a condition de les croiser DANS leur
# longueur.
#
# Ou tombe ce croisement ? Une piece du dessus est une droite a la
# distance DECALAGE du centre ; une piece du dessous aussi, tournee
# de ANGLE. Leur intersection est a
#
#       croisement = DECALAGE / tan(ANGLE / 2)
#
# du centre, pour le croisement le plus eloigne des deux. On veut que
# ce point reste a au moins une DEMI-LARGEUR du bout de la piece,
# faute de quoi elle repose sur une arete :
#
#       croisement <= LONGUEUR/2 - LARGEUR/2
#
# d'ou, en prenant l'egalite -- le profil le plus large qui tienne :
#
#       DECALAGE = (LONGUEUR - LARGEUR)/2 * tan(ANGLE/2)
#
# A 90 degres, tan(45) = 1 : on retrouve exactement (L - l)/2 = 47,5,
# c'est-a-dire LE CARRE CLASSIQUE. Le carre n'est donc pas un cas pose
# a la main, c'est cette relation prise en 90 degres.
DECALAGE = (LONGUEUR - LARGEUR) / 2.0 * np.tan(np.radians(ANGLE) / 2.0)

# ET UNE SECONDE CONDITION, QUI BORNE L'ANGLE PAR LE BAS. Les deux
# pieces d'une meme assise ne doivent pas se chevaucher : il leur
# faut au moins une demi-largeur chacune de part et d'autre du centre.
#
#       DECALAGE >= LARGEUR/2    ->   ANGLE >= 35,1 degres
#
# C'est la raison geometrique pour laquelle un quart de tour de 30
# degres est impossible avec ces pieces : le profil degenere avant
# meme qu'on parle de stabilite.
ANGLE_MINIMAL = 2.0 * np.degrees(np.arctan(
    (LARGEUR / 2.0) / ((LONGUEUR - LARGEUR) / 2.0)))

# ---------------------------------------------------------------------------
# 3. LES DEUX REPERES DE PROGRAMMATION, ET LA PRISE
# ---------------------------------------------------------------------------
#
# LE CHANGEMENT DE METHODE EST ICI. La seance precedente ecrivait
# chaque cible directement depuis la base, et chaque cote y melangeait
# deux faits : ou est le magasin, et ou est la piece dedans. Avec
# douze cibles, ce melange devient ingerable.
#
# Un repere de programmation rend a chaque nombre une seule
# signification :
#
#   le magasin est deplace      -> la derniere colonne de MAGASIN
#   les pieces font 8 mm      -> la constante EPAISSEUR
#   la structure est batie ailleurs    -> la derniere colonne de STRUCTURE
#   la structure vrille de 60 deg   -> la constante ANGLE
#
# Donnes PAR RAPPORT A LA BASE DU ROBOT, et ecrits en matrices.
#
# Un repere de travail a son axe z vers le HAUT -- c'est une table, pas
# un outil. D'ou le contraste avec OUTIL ci-dessus, dont le z descend.

# le magasin : une piece presentee toujours au meme endroit, couchee
# le long de x. Hors de l'emprise du convoyeur (x = 150 a 270).
MAGASIN = np.array([[1, 0, 0,   60],
                    [0, 1, 0, -220],
                    [0, 0, 1,    0],
                    [0, 0, 0,    1]], dtype=float)

# la structure : le centre de ce qu'on batit, de l'autre cote du robot
STRUCTURE = np.array([[1, 0, 0,   0],
                     [0, 1, 0, 240],
                     [0, 0, 1,   0],
                     [0, 0, 0,   1]], dtype=float)

# DANS LE MAGASIN : la ventouse sur le dessus de la piece, couchee a
# plat, donc a EPAISSEUR du plan.
#
# Pas de rotation ici : la piece du magasin est deja le long de x.
# C'est l'orientation des DEPOSES qui travaillera.
DANS_LE_MAGASIN = np.array([[1,  0,  0,         0],
                            [0, -1,  0,         0],
                            [0,  0, -1, EPAISSEUR],
                            [0,  0,  0,         1]], dtype=float)

PRISE = MAGASIN @ DANS_LE_MAGASIN

# LA PIECE DANS LE SIMULATEUR. Du decor : aucune instruction, aucune
# trajectoire. La fenetre la dessine et la fait SUIVRE LA VENTOUSE
# pendant la lecture -- on voit la piece quitter le magasin,
# traverser, et se poser sur la structure.
#
# magasin=True : une autre reapparait des qu'on emporte celle-ci,
# comme l'operateur qui repose une piece a chaque cycle. Sans cela le
# magasin se vide au premier passage et les onze cycles suivants
# brasseraient du vide.
robot.poser_piece(x=float(MAGASIN[0, 3]), y=float(MAGASIN[1, 3]),
                  dims=(LONGUEUR, LARGEUR, EPAISSEUR), magasin=True,
                  nom="piece")

# Le degagement, exprime dans le repere de l'OUTIL. L'axe z de l'outil
# pointant vers le bas, un degagement VERS LE HAUT s'ecrit avec un z
# NEGATIF. On le verifie plutot que de le croire.
# 80 mm, et pas 90 : a 90 le bras doit se redresser davantage pour
# approcher le sommet de la structure par le dessus, et `controler()` signale
# J2 a 5 degres de sa butee. Le degagement est un reglage, pas une
# constante -- il se choisit avec la hauteur de la structure.
HAUT = np.array([[1, 0, 0,   0.0],
                 [0, 1, 0,   0.0],
                 [0, 0, 1, -80.0],
                 [0, 0, 0,   1.0]], dtype=float)

assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"

# ---------------------------------------------------------------------------
# 4. LA DEPOSE : POSITION *ET* ORIENTATION
# ---------------------------------------------------------------------------


#: direction de la structure vue de la base, en degres. C'est autour
#: d'elle qu'il faut ramener les orientations -- voir angle_outil.
CAP_STRUCTURE = np.degrees(np.arctan2(STRUCTURE[1, 3], STRUCTURE[0, 3]))


def angle_outil(assise):
    """
    Orientation de l'assise, ramenee dans la course du poignet.

    UNE PIECE EST SYMETRIQUE : la poser a 165 degres ou a -15 revient
    au meme, elle occupe la meme place. On peut donc ajouter ou retirer
    des demi-tours librement, et c'est gratuit pour la structure.

    MAIS LA COURSE DU POIGNET N'EST PAS CENTREE SUR ZERO. J4 est l'ecart
    entre l'orientation de l'outil et celle du BRAS :

        J4 = orientation - J1,   avec J4 dans [-90 ; +90]

    Les orientations atteignables sont donc centrees sur J1, c'est-a-dire
    sur la direction ou pointe le bras -- ici le cap de la structure, 90
    degres. Ramener dans ]-90 ; +90] comme on le ferait par reflexe
    enverrait 165 degres sur -15, et le poignet a -105 : hors course,
    pour exactement la meme piece au meme endroit.

    On ramene donc autour du CAP, et non autour de zero. C'est la
    correction qui a fait tomber quatorze reserves d'un coup.
    """
    a = (ANGLE_DEPART + assise * ANGLE) % 180.0
    while a - CAP_STRUCTURE > 90.0:
        a -= 180.0
    while a - CAP_STRUCTURE < -90.0:
        a += 180.0
    return a


def depose(assise, piece):
    """
    Cible de depose, pour l'assise et la piece donnees (des zero).

    DEUX CHOSES CHANGENT D'UNE CIBLE A L'AUTRE, et c'est tout l'objet
    de l'exercice :

      LA HAUTEUR. L'assise n repose sur les n precedentes, donc
      (n + 1) * EPAISSEUR au-dessus du plan de la structure -- la cote du
      DESSUS de la piece, ou la ventouse se pose.

      L'ORIENTATION. Chaque assise tourne de ANGLE par rapport a la
      precedente. C'est ce qui fait tenir la structure : les pieces se
      croisent au lieu de s'empiler bout a bout, et la structure vrille.

    LA MATRICE, COLONNE PAR COLONNE. Pour un angle a :

        colonne 0 : ( cos a,  sin a, 0)   l'axe x de l'outil, donc la
                                          LONGUEUR de la piece
        colonne 1 : ( sin a, -cos a, 0)   son y, retourne
        colonne 2 : (     0,      0, -1)  son z, vers le BAS
        colonne 3 : l'ecart, perpendiculaire a la piece, et la hauteur

    C'est Rz(a) . Rx(180) ecrit en clair. A a = 0 on retrouve la
    matrice de la prise ; a 90 degres, le quart de tour du carre.
    """
    a = np.radians(angle_outil(assise))
    c, s = np.cos(a), np.sin(a)

    # l'ecart est PERPENDICULAIRE a la piece : on tourne le vecteur
    # (0, 1) de l'angle a, ce qui donne (-sin a, cos a)
    signe = 1.0 if piece == 0 else -1.0
    ex = -signe * DECALAGE * s
    ey = signe * DECALAGE * c
    z = (assise + 1) * EPAISSEUR

    dans_la_structure = np.array([[c,  s,  0, ex],
                                  [s, -c,  0, ey],
                                  [0,  0, -1,  z],
                                  [0,  0,  0,  1]], dtype=float)

    # D'OU VIENNENT CES SEIZE NOMBRES : on va a sa place, on tourne de a
    # autour de la verticale, puis on retourne l'outil pour qu'il
    # approche par le dessus. Rx(180) est ecrit A DROITE, donc il tourne
    # dans le repere que Rz vient de produire -- autour de u, et non
    # autour du x de la base. C'est de la que vient le signe.
    #
    # Chaque facteur s'ecrit en 4x4 comme le reste : une transformation
    # est une matrice, pas un appel de fonction.
    ou_elle_se_pose = np.array([[1, 0, 0, ex],
                                [0, 1, 0, ey],
                                [0, 0, 1,  z],
                                [0, 0, 0,  1]], dtype=float)
    rotation_z = np.array([[c, -s, 0, 0],
                           [s,  c, 0, 0],
                           [0,  0, 1, 0],
                           [0,  0, 0, 1]], dtype=float)
    demi_tour_x = np.array([[1,  0,  0, 0],
                            [0, -1,  0, 0],
                            [0,  0, -1, 0],
                            [0,  0,  0, 1]], dtype=float)
    assert np.allclose(dans_la_structure,
                       ou_elle_se_pose @ rotation_z @ demi_tour_x)
    return STRUCTURE @ dans_la_structure


# ---------------------------------------------------------------------------
# 5. LE CYCLE
# ---------------------------------------------------------------------------

robot.repli()
for assise in range(ASSISES):
    for piece in (0, 1):
        cible = depose(assise, piece)

        # prise : approche, descente, aspiration, retrait
        robot.movj(PRISE @ HAUT, lissage=70)
        robot.movl(PRISE, vitesse=40)
        robot.ventouse(True)
        robot.attendre(0.3)
        robot.movl(PRISE @ HAUT)

        # depose : meme geste, a la hauteur et dans l'orientation voulues
        robot.movj(cible @ HAUT, lissage=70)
        robot.movl(cible, vitesse=40)
        robot.ventouse(False)
        robot.attendre(0.3)
        robot.movl(cible @ HAUT)
robot.repli()

# ---------------------------------------------------------------------------
# 6. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("piece {:.0f} x {:.0f} x {:.0f} mm"
          .format(LONGUEUR, LARGEUR, EPAISSEUR))
    print("angle par assise : {:.0f} deg   (minimum geometrique {:.1f})"
          .format(ANGLE, ANGLE_MINIMAL))
    print("decalage deduit  : {:.1f} mm".format(DECALAGE))
    if DECALAGE < LARGEUR / 2.0:
        print("")
        print("  IMPOSSIBLE : a cet angle, les deux pieces d'une meme")
        print("  assise se chevauchent. Le profil degenere avant meme")
        print("  qu'on parle de stabilite.")
    print("")
    print("prise : x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
    for assise in range(ASSISES):
        for piece in (0, 1):
            T = depose(assise, piece)
            print("  assise {} piece {} : x={:6.1f}  y={:6.1f}  z={:5.1f}"
                  "   a {:+.0f} deg"
                  .format(assise + 1, piece + 1, T[0, 3], T[1, 3], T[2, 3],
                          angle_outil(assise)))
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
