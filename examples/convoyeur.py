#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Le convoyeur : tapis, capteur de presence, et prise au vol.

    python examples/convoyeur.py              simulation
    python examples/convoyeur.py --reel       sur la machine

Le cycle, dix fois : le tapis avance, le capteur voit arriver une piece,
le tapis s'arrete, le robot la prend et la depose a cote, le tapis
repart.

Le tapis sur la table
---------------------
`repere` place le tapis comme un repere de travail : son origine est son
ENTREE, son axe x suit le sens de marche, son axe y traverse la bande.
Une piece a l'abscisse s se trouve donc en `TAPIS.cible(s, 0, z)`, et
deplacer le tapis ne demande de changer que cette ligne.

Le pas du tapis
---------------
`impulsions_par_mm` vaut par defaut celui du kit Dobot -- 282,9, pour un
moteur de 200 pas, 16 micro-pas, un reducteur 10 et une poulie de 36 mm.
C'est une valeur NOMINALE. Mesurez la votre : marquez la bande, lancez
le tapis une duree connue, mesurez le deplacement, et corrigez ici. Le
nombre voyage ensuite avec le programme jusqu'au robot.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import Convoyeur, Pilote, cible, repere_travail   # noqa: E402
from dobot_sim.convoyeur import PORTS_GP                         # noqa: E402
from dobot_sim import homogeneous as hg                          # noqa: E402

REEL = "--reel" in sys.argv

# ---------------------------------------------------------------------------
# 1. LA CELLULE
# ---------------------------------------------------------------------------

# Le tapis passe DEVANT le robot, en travers, et sa bande va vers les y
# NEGATIFS -- constate sur la machine le 28/10/2026.
#
# Comment on le sait, et pourquoi ce n'est pas une evidence. Un operateur
# place face au robot le voit deplier son bras en s'eloignant de lui :
# le +x du robot pointe donc vers le convoyeur, et le +y tombe a GAUCHE
# de l'operateur (x avant, y gauche, z haut : la convention Dobot). La
# bande, observee allant de gauche a droite depuis cette place, descend
# donc les y. D'ou rz=-90, qui couche l'axe x du tapis -- son sens de
# marche -- sur le -y du robot.
#
# `sens` reste a +1 : il ne decrit pas la geometrie mais le CABLAGE du
# moteur. Une consigne positive a bien fait avancer la bande dans le
# sens declare ici ; si vous rebranchez le moteur a l'envers, c'est
# `sens=-1` qu'il faudra, et rien d'autre a toucher.
#
# A MESURER CHEZ VOUS : x, y et longueur ci-dessous sont plausibles, pas
# releves. Mettez-y vos cotes -- le reste du programme suit.
#
# Le capteur est place la ou le BRAS ATTEINT la bande : plus loin, la
# piece arrive hors de portee. controler() le dit, mais autant poser le
# capteur au bon endroit.
TAPIS = Convoyeur(
    repere=repere_travail(x=230, y=250, z=0, rz=-90),
    longueur=500,          # mm, de l'entree a la sortie
    largeur=60,
    index_moteur=0,        # 0 = Stepper1, 1 = Stepper2
    capteur_s=250,         # abscisse du capteur : pile devant le robot
    camera_s=150,          # abscisse du champ de la camera, en amont
    # Le capteur de presence est l'INFRAROUGE du kit, sur GP4.
    # Ce n'est pas un detail de cablage : voir plus bas, il
    # change la forme meme du cycle sur la machine.
    capteur_port=PORTS_GP["GP4"],
)

# Au droit du capteur, sur la bande -- mais TOURNEE D'UN QUART DE TOUR.
#
# Pourquoi. Une cible prise dans le repere du tapis en herite
# l'orientation : le tapis etant pose a rz=-90, la cible demande -90 au
# poignet, et comme J1 vaut zero a cet endroit c'est J4 qui doit les
# fournir en entier -- pile sur sa butee de 90 deg. controler() le disait
# sans qu'on voie pourquoi.
#
# Une ventouse se moque de son orientation autour de z : elle aspire
# aussi bien dans un sens que dans l'autre. Ce quart de tour ne coute
# donc rien et ramene le poignet au large. Avec une PINCE il serait au
# contraire impose par la piece, et il faudrait deplacer le tapis ou le
# robot plutot que tordre le poignet.
PRISE = TAPIS.cible(250, 0, 85) @ hg.rot_z(math.radians(-90))
# La depose est RAMENEE VERS LE ROBOT : le tapis occupe desormais la
# bande x = 200..260, et l'ancienne depose a (250 ; -60) tombait
# dessus. Posez la votre ou il y a vraiment de la place.
DEPOSE = cible(x=170, y=-140, z=100)        # a cote, sur la table
HAUT = hg.translation(0, 0, -70)           # degagement, repere de l'OUTIL

VITESSE_TAPIS = 60                         # mm/s

# ---------------------------------------------------------------------------
# 2. LE PROGRAMME
# ---------------------------------------------------------------------------

robot = Pilote(nom="convoyeur", outil="ventouse",
               vitesse=140, acceleration=600, vitesse_articulaire=60)
robot.tapis = TAPIS                        # sert de defaut aux commandes
robot.regler_longueur_outil(79.3)          # mesuree sur la machine

CYCLES = 10


def prise_depose(r):
    """
    La prise et la depose, SANS le tapis : le geste du bras, seul.

    Il sert tel quel des deux cotes -- en simulation dans un programme
    d'un seul tenant, sur la machine par morceaux -- et c'est tout
    l'interet de l'isoler : le mouvement ne doit pas changer selon la
    facon dont on attend la piece.
    """
    # prise : ARRET AU POINT, sinon la ventouse aspire avant d'y etre
    r.movj(PRISE @ HAUT, lissage=70)
    r.movl(PRISE, vitesse=50)
    r.ventouse(True)
    r.attendre(0.3)
    r.movl(PRISE @ HAUT, lissage=80)

    # depose a cote
    r.movj(DEPOSE @ HAUT, lissage=70)
    r.movl(DEPOSE, vitesse=50)
    r.ventouse(False)
    r.attendre(0.3)
    r.movl(DEPOSE @ HAUT)


# En SIMULATION, le cycle est un programme d'un seul tenant : le tapis y
# figure comme instruction, et l'attente de la piece est remplacee par le
# temps qu'elle met a parcourir le tapis -- un calcul, pas un capteur.
robot.repli()
duree_arrivee = TAPIS.capteur_s / VITESSE_TAPIS
for i in range(CYCLES):
    robot.convoyeur(VITESSE_TAPIS, commentaire="le tapis amene")
    robot.attendre(duree_arrivee, commentaire="la piece arrive au capteur")
    robot.convoyeur(0, commentaire="arret sous le bras")
    prise_depose(robot)
robot.repli()
robot.convoyeur(0, commentaire="fin de cycle")

# ---------------------------------------------------------------------------
# 3. CONTROLE, PUIS EXECUTION
# ---------------------------------------------------------------------------

print(TAPIS.resume())
print()
print(robot.resume().splitlines()[0])

# Tous les avertissements ne se valent pas : une butee ou une collision
# empechent le cycle, une survitesse ne fait que meriter un coup d'oeil.
# C'est le partage que le pont fait avant de televerser.
from dobot_sim.reel import BLOQUANTS_TELEVERSEMENT       # noqa: E402

avertissements = robot.controler()
bloquants = [a for a in avertissements
             if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
reserves = [a for a in avertissements if a not in bloquants]

if bloquants:
    print("\nCYCLE IMPOSSIBLE :")
    for a in bloquants[:6]:
        print("  -", a)
    print("\nDeplacez le tapis ou revoyez les cibles : le bras n'atteint")
    print("pas tout le cycle depuis cette position.")
    sys.exit(1)
if reserves:
    print("\nreserves (le cycle passe, mais) :")
    for a in reserves[:4]:
        print("  -", a)
else:
    print("Tout le cycle est dans l'espace de travail.")

# Ce que le tapis fait pendant ce temps : on rejoue sa cadence pour
# verifier qu'une piece posee a l'entree arrive bien au capteur.
TAPIS.regler(0, 0.0).regler(VITESSE_TAPIS, 0.0)
piece = TAPIS.poser(s=0, t0=0.0, nom="bloc")
quand = TAPIS.instant_au_capteur(piece)
if quand is not None:
    print("une piece posee a l'entree atteint le capteur en {:.1f} s "
          "({:.0f} mm a {:.0f} mm/s)".format(quand, TAPIS.capteur_s,
                                             VITESSE_TAPIS))

if not REEL:
    robot.simuler()
    sys.exit(0)

# ---------------------------------------------------------------------------
# 4. SUR LA MACHINE : LE CYCLE EST MENE DEPUIS LE PC
# ---------------------------------------------------------------------------
#
# Pourquoi la forme change ici, et pourquoi ce n'est pas un choix.
#
# Le capteur du kit est un INFRAROUGE sur un port GP. Il ne se lit que
# par une commande immediate : la file du controleur ne sait pas
# attendre dessus, et aucune instruction televersee ne peut s'y
# synchroniser. Le cycle se mene donc morceau par morceau --
#
#     tapis en marche  ->  on SCRUTE le capteur  ->  tapis a l'arret
#     ->  on televerse la prise-depose  ->  on recommence
#
# Le robot reste attache au PC pendant tout le cycle : si la liaison
# tombe, le cycle s'arrete.
#
# L'AUTRE CABLAGE, et ce qu'il vaut. Un capteur tout-ou-rien sur une
# entree EIO se laisse attendre par le controleur. On retrouve alors un
# programme d'un seul tenant, `attendre_piece(eio)` a la place de la
# scrutation, et la machine devient autonome -- le PC peut etre
# debranche. Si votre cellule doit tourner seule, c'est ce cablage qu'il
# faut, et il se declare par `capteur_eio=` au lieu de `capteur_port=`.

print("")
print("LE ROBOT ET LE TAPIS VONT BOUGER.")
print("Le capteur est un infrarouge sur un port GP : le cycle est mene")
print("depuis le PC, morceau par morceau. Ctrl+C arrete tout.")
print("")

robot.vider_programme()
robot.repli()
robot.executer_sur_robot(vitesse=30)

try:
    for i in range(CYCLES):
        robot.convoyeur_sur_robot(VITESSE_TAPIS)
        vue = robot.attendre_piece_sur_robot(timeout=30)
        robot.convoyeur_sur_robot(0)
        if not vue:
            print("cycle {} : aucune piece en 30 s -- on s'arrete la."
                  .format(i + 1))
            break
        print("cycle {}/{} : piece au capteur, on la prend"
              .format(i + 1, CYCLES))
        robot.vider_programme()
        prise_depose(robot)
        robot.executer_sur_robot(vitesse=30)
finally:
    # le tapis ne doit pas rester en marche si le script s'interrompt
    robot.convoyeur_sur_robot(0)
    robot.vider_programme()
    robot.repli()
    robot.executer_sur_robot(vitesse=30)
    print("cellule au repos")
