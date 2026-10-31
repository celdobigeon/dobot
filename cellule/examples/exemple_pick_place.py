#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple 2 - Programmation de trajectoires, en articulaire et en cartesien.

    python examples/exemple_pick_place.py

Construit un cycle de prise et depose, verifie les butees et les vitesses,
puis exporte un script pour le robot reel. Ne necessite ni Qt ni OpenGL.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import DobotMagician                                  # noqa: E402
from dobot_sim import homogeneous as hg                              # noqa: E402
from dobot_sim.program import Instruction, Programme                 # noqa: E402
from dobot_sim.trajectory import (ProfilPolynomial5, ProfilTrapeze,  # noqa: E402
                                  SegmentArticulaire, SegmentCirculaire,
                                  SegmentLineaire, Trajectoire)

robot = DobotMagician()

# ===========================================================================
# A. Construction directe d'une trajectoire, segment par segment
# ===========================================================================
print("=" * 70)
print("A. Trajectoire construite a la main")
print("=" * 70)

q_depart = np.array([0.0, 15.0, 30.0, 0.0])
traj = Trajectoire(robot)

# 1. mouvement ARTICULAIRE : on interpole les angles
q_cible = robot.mgi_pose(220, -90, 60, 0)
traj.ajouter(SegmentArticulaire(robot, q_depart, q_cible,
                                ProfilPolynomial5(1.5), "approche"))

# 2. mouvement CARTESIEN rectiligne : on interpole la matrice homogene
T_bas = robot.pose_vers_matrice(220, -90, 25, 0)
traj.ajouter(SegmentLineaire(robot, traj.segments[-1].q_final, T_bas,
                             ProfilTrapeze(1.0, 0.3), "descente"))

# 3. remontee
T_haut = robot.pose_vers_matrice(220, -90, 110, 0)
traj.ajouter(SegmentLineaire(robot, traj.segments[-1].q_final, T_haut,
                             ProfilTrapeze(1.0, 0.3), "remontee"))

# 4. mouvement CIRCULAIRE par un point de passage
T_via = robot.pose_vers_matrice(265, 0, 130, 0)
T_fin = robot.pose_vers_matrice(220, 90, 110, 45)
traj.ajouter(SegmentCirculaire(robot, traj.segments[-1].q_final, T_via, T_fin,
                               ProfilPolynomial5(2.5), "transfert"))

print("\n{} segments, duree totale {:.2f} s".format(len(traj), traj.duree))
for i, s in enumerate(traj.segments):
    print("  {}. {:<8} {:<12} {:.2f} s".format(i + 1, s.type_mouvement,
                                               s.nom, s.duree))

avertissements = traj.controler(0.02)
print("\nControle : {}".format("aucun probleme" if not avertissements
                               else "{} avertissement(s)".format(len(avertissements))))
for a in avertissements[:10]:
    print("  ! " + a)

e = traj.echantillonner(0.02)
print("\nExtremums sur la trajectoire :")
print("  vitesses articulaires max : "
      + ", ".join("{} {:.0f} deg/s".format(n.split()[0], v)
                  for n, v in zip(robot.NOMS_ARTICULATIONS,
                                  np.max(np.abs(e["dQ"]), axis=0))))
print("  vitesse cartesienne max   : {:.0f} mm/s".format(float(np.max(e["V"]))))
print("  altitude z                : {:.0f} a {:.0f} mm"
      .format(float(np.min(e["P"][:, 2])), float(np.max(e["P"][:, 2]))))

# Verification de la rectitude du segment 2 : l'ecart a la droite doit etre nul
seg = traj.segments[1]
p0, p1 = hg.position_de(seg.T0), hg.position_de(seg.T1)
u = (p1 - p0) / np.linalg.norm(p1 - p0)
ecarts = []
for t in np.linspace(0, seg.duree, 40):
    p = hg.position_de(robot.mgd(seg.q(t)))     # position REELLE via le MGD
    d = p - p0
    ecarts.append(np.linalg.norm(d - np.dot(d, u) * u))
print("\nRectitude du segment MOVL : ecart maximal a la droite = {:.2e} mm"
      .format(max(ecarts)))

# Comparaison : le meme deplacement en articulaire n'est PAS rectiligne
seg_j = SegmentArticulaire(robot, seg.q0, robot.mgi(seg.T1),
                           ProfilPolynomial5(1.0))
ecarts_j = []
for t in np.linspace(0, seg_j.duree, 40):
    p = hg.position_de(robot.mgd(seg_j.q(t)))
    d = p - p0
    ecarts_j.append(np.linalg.norm(d - np.dot(d, u) * u))
print("Meme deplacement en MOVJ  : ecart maximal a la droite = {:.2f} mm"
      .format(max(ecarts_j)))

# ===========================================================================
# B. Le meme cycle sous forme de programme editable
# ===========================================================================
print()
print("=" * 70)
print("B. Programme editable, sauvegardable et exportable")
print("=" * 70)

prog = Programme([
    Instruction(type="MOVJ", mode="articulaire", q=[0, 15, 30, 0], duree=1.5,
                commentaire="repli"),
    Instruction(type="MOVJ", mode="cartesien", pose=[220, -90, 60, 0], duree=1.5,
                commentaire="au-dessus de la piece"),
    Instruction(type="MOVL", mode="cartesien", pose=[220, -90, 25, 0], duree=1.0,
                profil="trapeze", commentaire="descente"),
    Instruction(type="OUTIL", etat=True, commentaire="ventouse ON"),
    Instruction(type="ATTENTE", duree=0.4),
    Instruction(type="MOVL", mode="cartesien", pose=[220, -90, 110, 0], duree=1.0,
                commentaire="remontee"),
    Instruction(type="MOVC", mode="cartesien", via=[265, 0, 130, 0],
                pose=[220, 90, 110, 45], duree=2.5, commentaire="transfert"),
    Instruction(type="MOVL", mode="cartesien", pose=[220, 90, 25, 45], duree=1.0,
                commentaire="descente"),
    Instruction(type="OUTIL", etat=False, commentaire="ventouse OFF"),
    Instruction(type="MOVJ", mode="articulaire", q=[0, 15, 30, 0], duree=2.0,
                commentaire="retour"),
], nom="cycle_pick_place")

for i, ins in enumerate(prog.instructions):
    print("  {:02d}  {}".format(i + 1, ins.resume()))

traj2 = prog.construire(robot, q_depart)
print("\nduree du cycle : {:.2f} s".format(traj2.duree))
print("evenements outil : "
      + ", ".join("t={:.2f}s {}".format(t, "ON" if e else "OFF")
                  for t, e in prog.evenements_outil))

dossier = os.path.dirname(os.path.abspath(__file__))
chemin_json = os.path.join(dossier, "cycle_pick_place.json")
chemin_py = os.path.join(dossier, "cycle_pick_place_robot_reel.py")
chemin_csv = os.path.join(dossier, "cycle_pick_place.csv")

prog.enregistrer(chemin_json)
with open(chemin_py, "w", encoding="utf-8") as f:
    f.write(prog.exporter_python(robot))
with open(chemin_csv, "w", encoding="utf-8") as f:
    f.write(prog.exporter_csv(robot, 0.05, q_depart))

print("\nfichiers ecrits :")
for c in (chemin_json, chemin_py, chemin_csv):
    print("  " + c)

print("\nRelecture du JSON : {} instructions"
      .format(len(Programme.charger(chemin_json))))
