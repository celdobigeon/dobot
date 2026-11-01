#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRIGE de l'activite du convoyeur -- cellule/ACTIVITE_CONVOYEUR.md

    python corrige_convoyeur.py              simulation
    python corrige_convoyeur.py --reel       sur la machine

NE PAS METTRE DANS cellule/. L'activite consiste a ecrire ce fichier.

Ce que l'activite demande, et ou c'est fait ici
-----------------------------------------------
    1. le convoyeur par rapport a la BASE        -- section 2
    2. le capteur par rapport au CONVOYEUR       -- section 3
    3. la prise par rapport au CAPTEUR           -- section 4
    4. le cycle                                  -- section 6

Les trois matrices se composent, et c'est tout l'exercice :

    PRISE = CONVOYEUR @ CAPTEUR @ DANS_LE_CAPTEUR

Bouger le tapis ne touche QUE la premiere. Deplacer le capteur le long
de la bande ne touche QUE la deuxieme, et d'une seule cote. Changer la
taille des pieces ne touche QUE la troisieme.

UNE NOTE POUR L'ENSEIGNANT
--------------------------
`cellule/examples/convoyeur.py` est le programme de reference, plus
complet : il mene le cycle depuis le PC, mesure le retard a chaque
passage, sait tourner en autonome et journalise l'essai. Il donne aussi
la reponse. Faites ecrire celui-ci d'abord, et servez-vous du sien pour
VERIFIER -- la derniere question de l'activite le demande.

Les cotes ci-dessous sont celles de la cellule du labo, relevees le
30/10/2026 photo a l'appui.
"""

import sys

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
import numpy as np

from dobot_sim import Convoyeur, Pilote
from dobot_sim import homogeneous as hg
from dobot_sim.convoyeur import PORTS_GP
from dobot_sim.program import DUREE_LOGIQUE

REEL = "--reel" in sys.argv

# ---------------------------------------------------------------------------
# 1. L'OUTIL
# ---------------------------------------------------------------------------

L_OUTIL = 89.3

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot = Pilote(nom="convoyeur", outil="ventouse",
               vitesse=120, acceleration=600, vitesse_articulaire=50)
robot.definir_repere_outil(OUTIL)

# ---------------------------------------------------------------------------
# 2. LE CONVOYEUR PAR RAPPORT A LA BASE DU ROBOT
# ---------------------------------------------------------------------------
#
# LA REFERENCE EST L'AXE DU MOTEUR, et ce choix n'est pas anodin.
#
# C'est un objet MATERIEL : on le montre du doigt, on y pose un reglet,
# et on y revient demain. Le "debut de la bande" n'est rien -- la bande
# tourne, elle n'a pas de debut. Prendre l'axe moteur, c'est prendre
# ce qu'on peut mesurer.
#
# COTES RELEVEES SUR LA CELLULE le 30/10/2026 :
#
#   x = 210    du centre du robot au CENTRE du convoyeur
#   y = -400   l'axe moteur, a 400 mm sur la DROITE de l'operateur --
#              et la droite de l'operateur est le -y du robot
#   z =  60    hauteur de la bande au-dessus du plan de la base
#
# LA PARTIE ROTATION, COLONNE PAR COLONNE. Le quart de tour (rz = 90)
# couche les axes du tapis sur ceux de la base :
#
#   colonne 0 : ( 0, 1, 0)   l'axe x du tapis -- LE SENS DE MARCHE --
#                            pointe vers le +y du robot
#   colonne 1 : (-1, 0, 0)   son y, en travers de la bande
#   colonne 2 : ( 0, 0, 1)   son z, vers le HAUT : c'est une table
#
# LE SENS DE MARCHE NE SE CHOISIT PAS, IL SE DEDUIT. On pose y sur
# l'axe moteur, z vers le haut, et x = y ^ z pour que le repere reste
# direct. Inverser le tapis, c'est prendre l'axe moteur dans l'autre
# sens -- rien d'autre a reprendre.
CONVOYEUR = np.array([[ 0, -1, 0,  210],
                      [ 1,  0, 0, -400],
                      [ 0,  0, 1,   60],
                      [ 0,  0, 0,    1]], dtype=float)

# ---------------------------------------------------------------------------
# 3. LE CAPTEUR PAR RAPPORT AU CONVOYEUR
# ---------------------------------------------------------------------------
#
# UNE SEULE COTE, et c'est la force de la chose : le capteur est a
# 300 mm de l'axe moteur, LE LONG DE LA BANDE. On la releve au reglet,
# posé sur le tapis, sans rien savoir d'ou est le robot.
#
# Translation pure : le capteur ne tourne pas par rapport au tapis.
#
#   colonne 3 : (300, 0, 0)  -- 300 mm le long de x, donc de la marche
#
# Le deplacer de trois centimetres ne touche que ce 300. Ni la position
# du tapis, ni la taille des pieces, ni le programme.
CAPTEUR_S = 300.0

CAPTEUR = np.array([[1, 0, 0, CAPTEUR_S],
                    [0, 1, 0,         0],
                    [0, 0, 1,         0],
                    [0, 0, 0,         1]], dtype=float)

# ---------------------------------------------------------------------------
# 4. LA PRISE PAR RAPPORT AU CAPTEUR
# ---------------------------------------------------------------------------

COTE = 25.0                 # cube de 25 mm, mesure
VITESSE_TAPIS = 60.0        # mm/s

# LE RETARD DE LA CHAINE, EN SECONDES ET NON EN MILLIMETRES.
#
# Entre l'instant ou le capteur voit la piece et celui ou la bande
# s'arrete, il se passe un temps, pendant lequel la piece avance. Le
# declarer en TEMPS et le convertir avec la vitesse, c'est le corriger
# tout seul quand on change la cadence.
#
# Il depend du CHEMIN d'execution, mesure le 31/10/2026 :
#
#   execution directe depuis le script    9 a  11 ms
#   a travers une fenetre de simulateur  56 a 119 ms, mediane 66
#
# Et il est DISPERSE : une constante n'en corrige que la mediane, pas
# la queue. La ventouse absorbe le reste ; une pince ne le ferait pas.
RETARD_ARRET = 0.067
DERIVE = VITESSE_TAPIS * RETARD_ARRET

# LA VENTOUSE DANS LE REPERE DE LA PIECE ARRETEE AU CAPTEUR.
#
#   +DERIVE le long de la bande : la piece ne s'arrete pas AU capteur,
#   elle le depasse du chemin parcouru pendant le retard.
#
#   PAS de demi-piece, et c'est une correction MESUREE contre un
#   raisonnement. On avait ecrit -COTE/2 en se disant que le capteur
#   voit le BORD AVANT tandis que la ventouse vise le CENTRE. Essai du
#   31/10/2026 : la ventouse se posait une demi-piece trop en amont. Le
#   capteur se declenche donc sur le CENTRE. Le raisonnement etait
#   propre et il etait faux ; le releve l'emporte.
#
#   +COTE en z : le dessus du cube, ou la ventouse se pose.
#
#   La rotation retourne l'outil -- z vers le BAS -- et lui donne le
#   quart de tour qui garde J4 loin de sa butee.
DANS_LE_CAPTEUR = np.array([[ 0, -1,  0, DERIVE],
                            [-1,  0,  0,      0],
                            [ 0,  0, -1,   COTE],
                            [ 0,  0,  0,      1]], dtype=float)

# LES TROIS MATRICES SE COMPOSENT. C'est tout l'exercice.
PRISE = CONVOYEUR @ CAPTEUR @ DANS_LE_CAPTEUR

# ---------------------------------------------------------------------------
# 5. LA DEPOSE, ET LE DEGAGEMENT
# ---------------------------------------------------------------------------
#
# HORS DE L'EMPRISE DU TAPIS, qui occupe x = 150 a 270 sur toute sa
# longueur. Une depose en x=170 reposerait la piece SUR le convoyeur,
# qui la remmenerait -- erreur commise, et le robot ne s'en plaint pas.
DEPOSE = np.array([[0.156434,  0.987688,  0,  15.00],
                   [0.987688, -0.156434,  0, 219.21],
                   [0,         0,        -1,  92.00],
                   [0,         0,         0,      1]], dtype=float)

HAUT = np.array([[1, 0, 0,   0.0],
                 [0, 1, 0,   0.0],
                 [0, 0, 1, -70.0],
                 [0, 0, 0,   1.0]], dtype=float)

assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"

# ---------------------------------------------------------------------------
# 6. LE TAPIS, ET LE CYCLE
# ---------------------------------------------------------------------------
#
# L'objet Convoyeur porte les memes cotes que la matrice ci-dessus : on
# le lui donne pour qu'il se dessine et que le capteur sache ou il est.
TAPIS = Convoyeur(repere=CONVOYEUR, longueur=698, largeur=100,
                  index_moteur=0,
                  # CABLAGE du moteur, pas geometrie. Il valait +1 sur
                  # une machine et -1 sur une autre, tapis inchange.
                  sens=-1,
                  capteur_s=CAPTEUR_S,
                  # la camera lit la COULEUR des pieces. Elle n'est pas
                  # encore utilisee, et elle est pres de l'entree : deux
                  # postes au meme endroit ne se lisent pas.
                  camera_s=120.0,
                  capteur_port=PORTS_GP["GP4"],
                  depassement=0.0)
robot.tapis = TAPIS

# LA PIECE, POSEE SUR LA BANDE -- et non au capteur.
#
# Elle part de l'entree, AVANCE AVEC LE TAPIS, s'arrete quand il
# s'arrete, et passe a la ventouse quand celle-ci l'aspire. C'est ce
# qu'il faut voir : le capteur arrete QUELQUE CHOSE QUI BOUGEAIT. Une
# piece qui apparait deja posee au capteur ne montre pas le cycle, elle
# en montre la fin.
#
# magasin=True : une neuve est lachee a l'entree des que la precedente
# est emportee -- l'operateur qui repose une piece a chaque cycle.
robot.poser_piece_sur_tapis(s=0.0, dims=(COTE, COTE, COTE), magasin=True,
                            nom="cube")

CYCLES = 3

# EN SIMULATION, ON ATTEND UN DELAI ; SUR LA MACHINE, UN CAPTEUR.
#
# La file du controleur ne sait pas attendre sur un port GP (voir
# l'activite), donc le cycle reel est mene depuis le PC et scrute le
# capteur. En simulation il n'y a pas de capteur : on remplace l'attente
# par le temps que la piece met a parcourir la bande.
#
# ET LE DELAI NAIF NE TOMBE PAS JUSTE. `CAPTEUR_S / VITESSE_TAPIS` vaut
# 5,00 s, mais le tapis tourne 5,25 s : l'instruction `convoyeur()`
# occupe elle-meme un quart de seconde pendant lequel la bande avance
# deja. A 60 mm/s, cela fait 15 mm -- la piece depassait le capteur
# d'autant, et la ventouse la prenait decalee.
#
# On retranche donc ce temps propre. Et on ne vise pas le droit du
# capteur mais L'ABSCISSE OU LA PIECE S'IMMOBILISE VRAIMENT : le
# capteur plus la derive du retard, exactement ce que vise PRISE. La
# piece simulee s'arrete alors la ou la vraie s'arrete, et la ventouse
# se pose dessus.
#
# RIEN DE TOUT CELA N'EXISTE SUR LA MACHINE : la, le capteur constate,
# et le programme n'a besoin de connaitre ni la loi de la bande ni le
# temps propre des instructions. C'est tout l'interet d'un capteur, et
# le prix de s'en passer tient dans ces deux lignes.
ARRET_S = CAPTEUR_S + DERIVE
DUREE_ARRIVEE = ARRET_S / VITESSE_TAPIS - DUREE_LOGIQUE

robot.repli()
for i in range(CYCLES):
    robot.convoyeur(VITESSE_TAPIS, commentaire="le tapis amene")
    robot.attendre(DUREE_ARRIVEE, commentaire="la piece arrive au capteur")
    robot.convoyeur(0, commentaire="arret sous le bras")

    robot.movj(PRISE @ HAUT, lissage=70)
    robot.movl(PRISE, vitesse=50)
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(PRISE @ HAUT, lissage=80)

    robot.movj(DEPOSE @ HAUT, lissage=70)
    robot.movl(DEPOSE, vitesse=50)
    robot.ventouse(False)
    robot.attendre(0.3)
    robot.movl(DEPOSE @ HAUT)
robot.repli()
robot.convoyeur(0, commentaire="fin de cycle")

# ---------------------------------------------------------------------------
# 7. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CONVOYEUR (axe moteur dans le repere de la base)")
    print(np.round(CONVOYEUR, 1))
    print("  sens de marche : {}  (= colonne 0)"
          .format(np.round(CONVOYEUR[:3, 0], 1)))
    print("")
    print("capteur dans le repere du robot : {}"
          .format(np.round((CONVOYEUR @ CAPTEUR)[:3, 3], 1)))
    print("PRISE = CONVOYEUR @ CAPTEUR @ DANS_LE_CAPTEUR")
    print("  x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))
    print("  derive du retard : {:.1f} mm".format(DERIVE))
    print("")
    print("Ce qu'il faut changer, et ou :")
    print("  capteur glisse de 30 mm   -> CAPTEUR seul, cote {:.0f} -> {:.0f}"
          .format(CAPTEUR_S, CAPTEUR_S + 30))
    print("  tapis recule de 30 mm     -> CONVOYEUR seul")
    print("  pieces de 40 mm           -> COTE seul")
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
