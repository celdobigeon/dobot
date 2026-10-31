#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme a executer dans le simulateur.

Depuis VS Code : ouvrez ce fichier et appuyez sur F5 (ou Ctrl+F5 pour
lancer sans debogueur). La fenetre de simulation s'ouvre et le robot joue
le mouvement.

Toutes les cibles sont des MATRICES HOMOGENES 4x4. Une cible n'est pas un
triplet de nombres : c'est la situation d'un repere, position et
orientation ensemble. On les compose avec l'operateur @.

La section 4 CONTROLE verifie ces cibles avant de bouger : matrices
valides, orientations realisables, compositions correctes, coordonnees
locales retrouvees par inversion. La section 6 verifie ensuite que le
robot les atteint vraiment.

Ce fichier est fait pour etre modifie : c'est votre bac a sable.
"""

import numpy as np

import cellule_sur_le_chemin             # noqa: F401  -- voir ce module
from dobot_sim import (DobotMagician, ErreurCinematique, Pilote, cible,
                       decomposer_cible, repere_travail)
from dobot_sim import homogeneous as hg

MODELE = DobotMagician()

# ===========================================================================
# 1. REPERES DE TRAVAIL
#
# Deux reperes poses sur la table, axe z vers le haut. Tout ce qui suit
# sera exprime par rapport a eux : deplacer physiquement le magasin ne
# demandera de changer qu'une seule ligne.
# ===========================================================================

MAGASIN = repere_travail(x=215, y=-95, z=25)
PALETTE = repere_travail(x=205, y=75, z=25, rz=20)      # palette de biais

PAS_CASE = 24.0                                          # entraxe des cases
N_CASES = 3

# ===========================================================================
# 2. CIBLES
#
# cible(...) construit  T = Trans(x, y, z) . Rz(rz) . Rx(180 deg).
# Le Rx(180) retourne l'axe z : celui de l'outil pointe vers le BAS, c'est
# la direction d'approche. Avec `repere=`, les coordonnees sont locales.
# ===========================================================================

PRISE = cible(repere=MAGASIN)                   # origine du magasin


def case(i):
    """Situation de la i-eme case de la palette, en coordonnees locales."""
    return cible(x=0, y=i * PAS_CASE, z=0, rz=0, repere=PALETTE)


# Degagement de 85 mm vers le haut. Post-multiplier, c'est se deplacer dans
# le repere de l'OUTIL ; comme son axe z pointe vers le bas, monter s'ecrit
# avec un z negatif.
DEGAGEMENT = hg.translation(0, 0, -85)

VIA_TRANSFERT = cible(265, 0, 135)              # point de passage de l'arc

# ===========================================================================
# 3. REGLAGES
#
#   vitesse / acceleration                 MOVL et MOVC, en mm/s et mm/s2
#   vitesse_articulaire / ...              MOVJ, en deg/s et deg/s2
#   lissage   0 = arret exact a chaque point, 100 = raccords les plus arrondis
#   profil    lineaire | trapeze | sinus | polynome5
#
# Les deux jeux de vitesses sont separes parce qu'ils ne mesurent pas la
# meme chose : un MOVJ interpole des angles, il ne borne pas la vitesse du
# TCP, qui depend du bras de levier.
# ===========================================================================

robot = Pilote(robot=MODELE,
               nom="cycle de palettisation",
               outil="ventouse",            # aucun | ventouse | pince | stylo | laser
               vitesse=140, acceleration=600,
               vitesse_articulaire=50, acceleration_articulaire=220,
               lissage=0, profil="polynome5",
               depart=[0, 25, 40, 0])

# L'outil porte une longueur, qui devient le deport lt du modele : a partir
# d'ici, z designe la POINTE de l'outil et non plus la bride.
#
# ATTENTION : les longueurs du catalogue sont NOMINALES. Le montage reel
# s'en ecarte, et l'ecart se retrouve entierement sur z -- le robot descend
# trop bas ou pas assez. Mesurez la votre :
#
#   1. amenez la POINTE au contact du plan de travail ;
#   2. lisez le z annonce par le modele ;
#   3. ajoutez-le a la longueur courante.
#
# Sur la machine du labo, le modele annoncait +17,3 mm alors que la pointe
# touchait : la ventouse fait donc 62 + 17,3 = 79,3 mm. Decommentez et
# ajustez pour votre montage. A appeler APRES le constructeur, car
# monter_outil remet la valeur nominale.
#
# Verifiez que l'ecart est bien CONSTANT en deux ou trois endroits de la
# table : constant, c'est l'outil ; variable, c'est l'origine articulaire
# qu'il faut reprendre.
#
# robot.regler_longueur_outil(79.3)

print("outil monte : {}, longueur {:.0f} mm -> lt = {:.1f} mm"
      .format(robot.outil.libelle, robot.outil.longueur, MODELE.geo.lt))

# --- affectation des entrees / sorties tout-ou-rien ------------------------
CYCLE_EN_COURS = 1          # sortie : voyant de marche
VENTOUSE_OK = 2             # sortie : temoin de prise
PIECE_PRESENTE = 5          # entree : capteur du magasin

# ===========================================================================
# 4. CONTROLE DES CIBLES
#
# On verifie les matrices AVANT de construire le mouvement : une erreur de
# repere se voit ici, pas en regardant le robot partir de travers.
# ===========================================================================


def controler(nom, T):
    """Verifie qu'une matrice est une cible outil valide et atteignable."""
    if not hg.est_homogene(T):
        print("  {:<22} MATRICE NON HOMOGENE".format(nom))
        return False
    try:
        x, y, z, r = decomposer_cible(T)
    except ErreurCinematique as exc:
        print("  {:<22} ORIENTATION REFUSEE : {}".format(nom, str(exc)[:60]))
        return False
    try:
        q = MODELE.mgi_pose(x, y, z, r)
    except ErreurCinematique:
        print("  {:<22} x={:7.1f} y={:7.1f} z={:6.1f} r={:6.1f}   HORS D'ATTEINTE"
              .format(nom, x, y, z, r))
        return False
    butees = MODELE.depassements(q)
    # le robot rejoint-il exactement la situation demandee ?
    ecart = float(np.max(np.abs(MODELE.mgd(q) - T)))
    etat = "BUTEE : " + butees[0] if butees else "ok  (ecart MGD/MGI {:.0e})".format(ecart)
    print("  {:<22} x={:7.1f} y={:7.1f} z={:6.1f} r={:6.1f}   {}"
          .format(nom, x, y, z, r, etat))
    return not butees


print("=" * 78)
print("4. CONTROLE DES CIBLES")
print("=" * 78)

cibles = [("PRISE", PRISE),
          ("PRISE @ DEGAGEMENT", PRISE @ DEGAGEMENT),
          ("VIA_TRANSFERT", VIA_TRANSFERT)]
for i in range(N_CASES):
    cibles.append(("case({})".format(i), case(i)))
    cibles.append(("case({}) @ DEGAGEMENT".format(i), case(i) @ DEGAGEMENT))

toutes_bonnes = all(controler(nom, T) for nom, T in cibles)
print("  -> {}".format("toutes les cibles sont valides et atteignables"
                       if toutes_bonnes else "AU MOINS UNE CIBLE EST INVALIDE"))

# --- le degagement monte-t-il vraiment de 85 mm ? --------------------------
haut = hg.position_de(PRISE @ DEGAGEMENT) - hg.position_de(PRISE)
print()
print("  PRISE @ DEGAGEMENT - PRISE      = {}  (attendu [0 0 85])"
      .format(np.round(haut, 9)))
assert np.allclose(haut, [0, 0, 85]), "le degagement ne monte pas de 85 mm"

# --- l'inverse du repere de travail redonne les coordonnees locales --------
print()
print("  inverse(PALETTE) @ case(i)  ->  coordonnees locales")
for i in range(N_CASES):
    local = hg.position_de(hg.inverse(PALETTE) @ case(i))
    print("    case({}) : {}   (attendu [0 {:.0f} 0])"
          .format(i, np.round(local, 9), i * PAS_CASE))
    assert np.allclose(local, [0, i * PAS_CASE, 0], atol=1e-9)

# --- la palette transmet bien son orientation aux cases --------------------
print()
orientations = [float(decomposer_cible(case(i))[3]) for i in range(N_CASES)]
print("  orientation r des cases : {} deg  (attendu 20 pour toutes)"
      .format([round(v, 6) for v in orientations]))
assert np.allclose(orientations, 20.0), "la palette ne transmet pas sa rotation"

# --- pre- ou post-multiplication : la distinction essentielle --------------
decalage = hg.translation(30, 0, 0)
a = np.radians(20.0)
print()
print("  case(0)                       : {}".format(
    np.round(hg.position_de(case(0)), 1)))
print("  case(0) @ Trans(30,0,0)       : {}   repere OUTIL (tourne de 20 deg)"
      .format(np.round(hg.position_de(case(0) @ decalage), 1)))
print("  Trans(30,0,0) @ case(0)       : {}   repere BASE"
      .format(np.round(hg.position_de(decalage @ case(0)), 1)))
assert np.allclose(hg.position_de(case(0) @ decalage),
                   hg.position_de(case(0)) + [30 * np.cos(a), 30 * np.sin(a), 0],
                   atol=1e-9)
assert np.allclose(hg.position_de(decalage @ case(0)),
                   hg.position_de(case(0)) + [30, 0, 0], atol=1e-9)

# --- une orientation impossible doit etre refusee --------------------------
print()
try:
    MODELE.matrice_vers_pose(PRISE @ hg.rot_y(np.radians(25)))
    raise AssertionError("une cible inclinee aurait du etre refusee")
except ErreurCinematique:
    print("  cible inclinee de 25 deg        : correctement refusee")

# ===========================================================================
# 5. CYCLE
# ===========================================================================

robot.repli()
robot.sortie(CYCLE_EN_COURS, True, "voyant de marche")

for i in range(N_CASES):
    DEPOSE = case(i)

    # --- approche et prise -------------------------------------------------
    robot.movj(PRISE @ DEGAGEMENT, lissage=70,
               commentaire="au-dessus de la piece {}".format(i + 1))
    # dialogue avec la cellule : on n'engage la descente que si le capteur
    # du magasin confirme la presence d'une piece
    robot.attendre_entree(PIECE_PRESENTE, True, timeout=15.0,
                          commentaire="capteur magasin")
    robot.movl(PRISE, vitesse=50, acceleration=200, profil="trapeze",
               commentaire="descente lente")
    robot.ventouse(True)
    robot.sortie(VENTOUSE_OK, True)
    robot.attendre(0.3, "aspiration")

    # --- transfert ---------------------------------------------------------
    robot.movl(PRISE @ DEGAGEMENT, lissage=80, commentaire="remontee")
    robot.movc(via=VIA_TRANSFERT, T=DEPOSE @ DEGAGEMENT,
               vitesse=180, acceleration=700, lissage=80,
               commentaire="transfert en arc")

    # --- depose ------------------------------------------------------------
    robot.movl(DEPOSE, vitesse=50, acceleration=200, profil="trapeze",
               commentaire="descente lente")
    robot.ventouse(False)
    robot.sortie(VENTOUSE_OK, False)
    robot.attendre(0.3, "relachement")
    robot.movl(DEPOSE @ DEGAGEMENT, lissage=70, commentaire="degagement")

robot.sortie(CYCLE_EN_COURS, False)
robot.repli()

# ===========================================================================
# 6. CONTROLE DU MOUVEMENT
#
# Les cibles sont valides ; le robot les atteint-il pour autant ? On
# echantillonne la trajectoire et on mesure la distance minimale du TCP a
# chaque cible. Une cible suivie d'un lissage n'est PAS atteinte
# exactement : c'est le principe meme de l'arrondi de raccord.
# ===========================================================================

print()
print("=" * 78)
print("6. CONTROLE DU MOUVEMENT")
print("=" * 78)
print(robot.resume())

traj = robot.trajectoire()
mesures = traj.echantillonner(0.005)

print()
print("  distance minimale du TCP a chaque cible commandee :")
ecart_max_net = 0.0
for i, ins in enumerate(robot.programme.instructions):
    if ins.type not in ("MOVJ", "MOVL", "MOVC"):
        continue
    p = hg.position_de(ins.matrice(MODELE))
    d = float(np.min(np.linalg.norm(mesures["P"] - p, axis=1)))
    # c'est le recouvrement REELLEMENT applique qui compte, pas la consigne :
    # une attente voisine interdit le lissage, et la Trajectoire le plafonne
    recouvrement = traj.recouvrements[i] if i < len(traj.recouvrements) else 0.0
    if recouvrement > 1e-9:
        note = "coin coupe ({:.0f} %, {:.2f} s)".format(ins.lissage, recouvrement)
    else:
        note = "atteinte"
        if ins.lissage > 0:
            note += "   (lissage neutralise par une attente voisine)"
        ecart_max_net = max(ecart_max_net, d)
    print("    {:02d} {:<5} {:<28} {:7.2f} mm   {}"
          .format(i + 1, ins.type, (ins.commentaire or "")[:28], d, note))

print()
print("  ecart maximal sur les cibles SANS lissage : {:.3f} mm".format(ecart_max_net))
assert ecart_max_net < 0.5, "une cible non lissee n'est pas atteinte"

# --- entrees / sorties -----------------------------------------------------
sorties, entrees = robot.programme.ports_utilises()
print()
print("  ports utilises   sorties EIO{}   entrees EIO{}".format(sorties, entrees))
print("  chronologie des evenements :")
for e in robot.programme.evenements[:8]:
    detail = "EIO{} = {}".format(e.port, 1 if e.etat else 0) \
        if e.type in ("SORTIE", "ENTREE") else robot.outil.etat_texte(e.etat)
    print("    t={:6.2f}s  {:<7} {}".format(e.t, e.type, detail))
print("    ... {} evenements au total".format(len(robot.programme.evenements)))
print()
print("  A la lecture, le robot se FIGE sur chaque attente d'entree.")
print("  Basculez EIO{} dans l'onglet Entrees / Sorties pour le liberer."
      .format(PIECE_PRESENTE))
print("  La duree annoncee plus haut compte les delais d'attente au pire cas ;")
print("  le cycle reel est d'autant plus court que le capteur repond vite.")

# Optionnel : le meme programme pour le VRAI robot (API DobotStudio).
# robot.exporter_python("cycle_robot_reel.py")

# ===========================================================================
# 7. SIMULATION
# ===========================================================================

robot.simuler()
