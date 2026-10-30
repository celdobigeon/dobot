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

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import Convoyeur, Pilote, repere_travail          # noqa: E402
from dobot_sim.convoyeur import PORTS_GP                         # noqa: E402
from dobot_sim import homogeneous as hg                          # noqa: E402

REEL = "--reel" in sys.argv

# ---------------------------------------------------------------------------
# 1. LA CELLULE
# ---------------------------------------------------------------------------

# Le tapis passe DEVANT le robot, en travers. REFERENCE : L'AXE DU
# MOTEUR est l'axe y du tapis. C'est lui qu'on montre du doigt, il est
# materiel, et il ne depend pas de la facon dont le moteur est cable.
#
# Le sens de marche n'est alors plus un choix mais un calcul : x = y ^ z,
# et le repere reste direct. Inverser le tapis, c'est prendre l'axe
# moteur dans l'autre sens -- rien d'autre a reprendre.
#
# COTES RELEVEES SUR LA CELLULE le 30/10/2026, photo a l'appui
# (Cellule_Dobot_Convoyeur_reelle.jpg) :
#
#   x = 210    du centre du robot au CENTRE du convoyeur
#   y = -400   l'axe moteur, a 400 mm sur la DROITE de l'operateur --
#              et la droite de l'operateur est le -y du robot
#   z = 60     hauteur de la bande
#
# L'origine du repere est donc l'AXE MOTEUR, a l'extremite droite du
# tapis, et la bande part vers les y POSITIFS : elle entre a y=-400 et
# sort a y=+298.
#
# Le maillage le confirme : sa section fait 218 mm de large a s=0 --
# le moteur en debord -- contre 120 ailleurs, soit exactement le bati
# mesure. L'entree du repere est bien du cote moteur.
#
# Comment on le sait, et pourquoi ce n'est pas une evidence. Un operateur
# place face au robot le voit deplier son bras en s'eloignant de lui :
# le +x du robot pointe donc vers le convoyeur, et le +y tombe a GAUCHE
# de l'operateur (x avant, y gauche, z haut : la convention Dobot). La
# bande, observee allant de gauche a droite depuis cette place, descend
# donc les y. D'ou rz=-90, qui couche l'axe x du tapis -- son sens de
# marche -- sur le -y du robot.
#
# `sens` ne decrit PAS la geometrie mais le CABLAGE du moteur, et c'est
# pourquoi il est separe de tout le reste. Il valait +1 sur le robot au
# convoyeur ; sur le robot 5, branche le 30/10/2026 a sa place, la meme
# consigne positive a fait descendre la bande a contresens. Le tapis
# n'avait pas bouge, ni les cotes : seul le controleur qui le pilote
# avait change. D'ou `sens=-1` -- un nombre, et rien d'autre a reprendre.
#
# C'est tout le benefice d'avoir pris l'AXE MOTEUR comme reference
# geometrique : le sens de marche se deduit (x = y ^ z), et le cablage
# reste une question separee, qui se regle a un endroit.
#
# A MESURER CHEZ VOUS : x, y et longueur ci-dessous sont plausibles, pas
# releves. Mettez-y vos cotes -- le reste du programme suit.
#
# Le capteur est place la ou le BRAS ATTEINT la bande : plus loin, la
# piece arrive hors de portee. controler() le dit, mais autant poser le
# capteur au bon endroit.
TAPIS = Convoyeur(
    repere=repere_travail(x=210, y=-400, z=60, rz=90),
    longueur=698,          # mm, releve sur le STEP du constructeur
    largeur=100,          # mm, bande utile (bati 120)
    index_moteur=0,        # 0 = Stepper1, 1 = Stepper2
    sens=-1,               # CABLAGE du moteur, pas geometrie -- voir plus haut
    capteur_s=300,         # mm depuis l'axe moteur -- releve au reglet,
                           # ce qui met le capteur a y = -100 du robot
    camera_s=200,          # abscisse du champ de la camera, en amont
    # Le capteur de presence est l'INFRAROUGE du kit, sur GP4.
    # Ce n'est pas un detail de cablage : voir plus bas, il
    # change la forme meme du cycle sur la machine.
    capteur_port=PORTS_GP["GP4"],
    # LE TAPIS S'ARRETE NET : mesure sur la machine, 8 ms entre la
    # detection et l'arret effectif, soit 0,2 mm de bande. La piece
    # s'immobilise au droit du capteur, et elle ne quitte jamais son
    # faisceau ensuite -- ce qui le prouve sans regle.
    #
    # J'avais cru le contraire et declare 145 mm de depassement. C'etait
    # faux, et la mesure qui m'avait trompe mesurait la distance TOTALE
    # marche-arret, ou la rampe de demarrage masque celle d'arret.
    # `diag_depassement.py` sait maintenant mesurer le depassement seul,
    # par le capteur ; s'il en trouve un sur votre cellule, mettez-le
    # ici.
    depassement=0.0,
    # Le maillage du constructeur, recale sur le repere du tapis par
    # convertir_cao.py. Absent, le tapis se dessine en boites.
    maillage=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "Modele3D", "convoyeur.obj"),
)

# La MEME vitesse qu'a l'etalonnage du point de prise, sinon
# l'avertissement plus bas vous le rappellera. Le tapis s'arretant net,
# elle ne change pas l'endroit ou la piece s'immobilise -- mais elle
# change le temps d'un cycle, et c'est tout.
VITESSE_TAPIS = 60                         # mm/s

# LE POINT DE PRISE : UNE CHAINE DE REPERES, et non un point appris.
#
# Le point appris -- `etalonner_convoyeur.py` -- reste l'outil de
# DEBOGAGE : on amene la ventouse sur la piece au clavier, et les trois
# nombres obtenus absorbent tout ce qu'on n'a pas su mesurer. Il depanne,
# mais il n'explique rien, et il meurt a la prochaine prise d'origine.
#
# Ici on ecrit ce que l'on sait, chaque cote a sa place :
#
#     TAPIS.repere      ou est le convoyeur par rapport au ROBOT
#     CAPTEUR           ou est le capteur le long de la BANDE
#     DANS_LE_CAPTEUR   ou est la ventouse par rapport a la PIECE
#
# Deplacer le capteur de 30 mm ne touche qu'une matrice, et une seule
# cote dans cette matrice. C'est la difference entre une cellule qu'on
# comprend et trois nombres qu'on recommence.
#
# Rien n'est recopie du repere du tapis : il est DEDUIT de la
# declaration ci-dessus, qui reste l'unique endroit ou vivent x, y, z
# et l'orientation du convoyeur.

# Le capteur, le long de la bande, depuis l'axe moteur. Une translation
# pure : le capteur ne tourne pas par rapport au tapis.
CAPTEUR = np.array([[1, 0, 0, TAPIS.capteur_s],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=float)

COTE = 25.0                                # cube de 25 mm, mesure

# La ventouse dans le repere de la PIECE arretee au capteur.
#
#   -COTE/2 le long de la bande : le capteur voit le BORD AVANT de la
#   piece, la ventouse vise son CENTRE. Une demi-piece, soit la moitie
#   des prises ratees de cette cellule tant qu'elle est restee implicite.
#
#   +COTE en z : le dessus du cube, ou la ventouse se pose.
#
#   La partie rotation retourne l'outil -- z vers le BAS -- et lui donne
#   le quart de tour qui garde J4 loin de sa butee.
DANS_LE_CAPTEUR = np.array([[ 0, -1,  0, -COTE / 2],
                            [-1,  0,  0,         0],
                            [ 0,  0, -1,      COTE],
                            [ 0,  0,  0,         1]], dtype=float)

PRISE = TAPIS.repere @ CAPTEUR @ DANS_LE_CAPTEUR

# LA DEPOSE, HORS DE L'EMPRISE DU TAPIS. Elle etait en x=170, c'est-a-
# dire AU-DESSUS DE LA BANDE (x = 150 a 270) : le bras y reposait la
# piece sur le convoyeur, qui la remmenait. Cotes relevees sur la
# cellule, et verifiees en cycle le 30/10/2026.
DEPOSE = np.array([[0.156434,  0.987688,  0,  15.00],
                   [0.987688, -0.156434,  0, 219.21],
                   [0,         0,        -1,  92.00],
                   [0,         0,         0,      1]], dtype=float)

HAUT = hg.translation(0, 0, -70)           # degagement, repere de l'OUTIL

print("PRISE = TAPIS.repere @ CAPTEUR @ DANS_LE_CAPTEUR")
print("  position x={:.1f}  y={:.1f}  z={:.1f}".format(*PRISE[:3, 3]))


# ---------------------------------------------------------------------------
# 2. LE PROGRAMME
# ---------------------------------------------------------------------------

robot = Pilote(nom="convoyeur", outil="ventouse",
               vitesse=140, acceleration=600, vitesse_articulaire=60)
robot.tapis = TAPIS                        # sert de defaut aux commandes
robot.regler_longueur_outil(89.3)          # ventouse mesuree sur la
                                           # machine : 79,3 faisait
                                           # plonger 10 mm trop bas

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

retards = []
temoins = []
try:
    for i in range(CYCLES):
        robot.convoyeur_sur_robot(VITESSE_TAPIS)
        # front=True : on attend qu'une piece ARRIVE, pas qu'il y en ait
        # une quelque part. Sans cela, la piece restee devant le capteur
        # -- celle qu'on vient de manquer, ou la suivante de la file --
        # rend la main aussitot et le bras plonge sur rien.
        vue = robot.attendre_piece_sur_robot(timeout=30, front=True)
        robot.convoyeur_sur_robot(0)
        # Ce que le PC a coute entre voir la piece et arreter le tapis.
        # Un tapis qui depasse fait toujours accuser le programme : le
        # chiffre le met hors de cause, ou le confond.
        if robot.instant_detection is not None:
            _ms = 1000.0 * (time.perf_counter() - robot.instant_detection)
            retards.append(_ms)
            print("  detection -> arret : {:.0f} ms, soit {:.1f} mm de bande"
                  .format(_ms, VITESSE_TAPIS * _ms / 1000.0))

            # LE CAPTEUR COMME TEMOIN. La piece est dans son faisceau au
            # moment de l'arret. Si elle en sort ensuite, c'est que la
            # bande tourne ENCORE -- et cela se mesure sans regle, sans
            # yeux, et sans rien avoir a recopier.
            _t0 = time.perf_counter()
            _libere = None
            while time.perf_counter() - _t0 < 5.0:
                if not robot.capteur_piece_sur_robot():
                    _libere = time.perf_counter() - _t0
                    break
            if _libere is None:
                _verdict = ("la piece reste devant le capteur : la bande "
                            "s'est bien arretee")
            elif _libere < 0.05:
                # Deja hors faisceau au moment ou l'on coupe : la piece
                # avait donc depasse le capteur AVANT qu'on la voie. Ce
                # n'est plus un probleme d'arret mais de detection.
                _verdict = ("la piece etait DEJA hors du faisceau a "
                            "l'arret : elle avait depasse le capteur "
                            "avant d'etre vue")
            else:
                _verdict = ("la piece a QUITTE le capteur {:.2f} s apres "
                            "l'arret : la bande tournait encore, soit "
                            "~{:.0f} mm"
                            .format(_libere, VITESSE_TAPIS * _libere))
            print("  " + _verdict)
            temoins.append((_ms, _libere))
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

    # Le recapitulatif, en DERNIER : c'est le chiffre qui dit si un
    # depassement du tapis vient du programme ou de la mecanique, et il
    # ne doit pas avoir defile hors de l'ecran quand on le cherche.
    if retards:
        print("")
        print("=" * 60)
        print("RETARD DU PC, DE LA DETECTION A L'ORDRE D'ARRET")
        print("=" * 60)
        print("  {}".format(", ".join("{:.0f} ms".format(r) for r in retards)))
        if robot.tours_scrutation:
            print("  derniere attente : {} lectures du capteur"
                  .format(robot.tours_scrutation))
        _pire = max(retards)
        print("  le pire : {:.0f} ms, soit {:.1f} mm de bande a {:.0f} mm/s"
              .format(_pire, VITESSE_TAPIS * _pire / 1000.0, VITESSE_TAPIS))
        print("")
        try:
            _journal = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "convoyeur_dernier_essai.log")
            with open(_journal, "w", encoding="utf-8") as _f:
                print("vitesse du tapis : {:.0f} mm/s".format(VITESSE_TAPIS),
                      file=_f)
                print("point de prise   : chaine de reperes, "
                      "x={:.1f} y={:.1f} z={:.1f}".format(*PRISE[:3, 3]),
                      file=_f)
                for _k, (_r, _l) in enumerate(temoins, 1):
                    _etat = ("jamais : la bande etait arretee" if _l is None
                             else "{:.2f} s apres l'arret, soit ~{:.0f} mm"
                             .format(_l, VITESSE_TAPIS * _l))
                    print("cycle {} : detection->arret {:.0f} ms ; "
                          "capteur libere {}".format(_k, _r, _etat),
                          file=_f)
            print("  journal ecrit : {}".format(_journal))
        except OSError as _exc:
            print("  journal non ecrit : {}".format(_exc))

        if _pire < 200:
            print("  Le programme n'y est pour rien. Si la piece depasse de")
            print("  plusieurs centimetres, cherchez ailleurs : le capteur")
            print("  qui signale tard, ou le tapis qui n'obeit pas a l'ordre")
            print("  d'arret -- pas le temps de calcul, qui est ici")
            print("  negligeable.")
        else:
            print("  Le PC met trop longtemps. C'est bien le programme.")
