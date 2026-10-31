#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple 1 - Matrices homogenes et modeles geometriques.

    python examples/exemple_matrices.py

Ne necessite ni Qt ni OpenGL : uniquement numpy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dobot_sim import homogeneous as hg          # noqa: E402
from dobot_sim import DobotMagician              # noqa: E402


def titre(t):
    print("\n" + t)
    print("-" * len(t))


# ---------------------------------------------------------------------------
titre("1. Construction et composition")

T1 = hg.translation(100, 0, 50)
T2 = hg.rot_z(np.pi / 4)
T = hg.composer(T1, T2)
print(hg.formater(T, 3, "T = Trans(100,0,50) . Rz(45 deg)"))

p_local = np.array([50.0, 0.0, 0.0])
print("\npoint (50, 0, 0) exprime dans le repere mobile :")
print("  dans le repere fixe ->", np.round(hg.transformer_point(T, p_local), 3))

titre("2. Inversion")
print(hg.formater(hg.inverse(T), 3, "T^-1"))
print("\nT . T^-1 = identite ?", np.allclose(T @ hg.inverse(T), np.eye(4)))

# ---------------------------------------------------------------------------
titre("3. Modele geometrique direct du Dobot")

robot = DobotMagician()
q = [30.0, 20.0, 35.0, -15.0]
print("q =", q, "deg")

for nom, M in zip(["T01", "T12", "T23", "T34", "T45", "T5T"],
                  robot.matrices_liaisons(q)):
    print()
    print(hg.formater(M, 3, nom))

T0T = robot.mgd(q)
print()
print(hg.formater(T0T, 3, "T0_TCP (produit des six matrices)"))
print("\npose Dobot (x, y, z, r) =", np.round(robot.pose(q), 3))

titre("4. Modele geometrique inverse")

q_retrouve = robot.mgi(T0T)
print("q d'origine :", np.round(q, 4))
print("q par MGI   :", np.round(q_retrouve, 4))
print("ecart max   : {:.2e} deg".format(float(np.max(np.abs(np.array(q) - q_retrouve)))))

titre("5. Jacobienne")

J = robot.jacobienne(q)
print("J (lignes vx, vy, vz, wz ; colonnes dq1..dq4, en rad/s) :")
for ligne in J:
    print("  " + "  ".join("{:>9.3f}".format(v) for v in ligne))
print("\ndet(J)        = {:.4g}".format(float(np.linalg.det(J))))
print("manipulabilite = {:.4g}".format(robot.manipulabilite(q)))
print("verification par differences finies : ecart max = {:.2e}"
      .format(float(np.max(np.abs(J - robot.jacobienne_numerique(q))))))

# vitesse cartesienne pour une vitesse articulaire donnee
dq = np.radians([10.0, 5.0, -5.0, 0.0])          # rad/s
print("\npour dq = (10, 5, -5, 0) deg/s :")
print("  vitesse du TCP = {} mm/s, rotation {:.3f} rad/s"
      .format(np.round((J @ dq)[:3], 2), (J @ dq)[3]))

titre("6. Denavit-Hartenberg")

print("Convention classique  Rz(theta).Tz(d).Tx(a).Rx(alpha) :")
print(hg.formater(hg.dh_standard(np.pi / 6, 138.0, 0.0, np.pi / 2), 3))
print("\nConvention modifiee (Khalil-Kleinfinger) "
      "Rx(alpha).Tx(a).Rz(theta).Tz(d) :")
print(hg.formater(hg.dh_modifie(np.pi / 2, 0.0, np.pi / 6, 138.0), 3))

titre("7. Interpolation de pose (base d'un mouvement rectiligne)")

Ta = robot.pose_vers_matrice(200, -100, 40, -30)
Tb = robot.pose_vers_matrice(180, 120, 150, 60)
print("s     x        y        z       r")
for s in np.linspace(0, 1, 6):
    Ts = hg.interpoler(Ta, Tb, s)
    p = hg.position_de(Ts)
    r = np.degrees(np.arctan2(Ts[1, 0], Ts[0, 0]))
    print("{:.2f}  {:7.2f}  {:7.2f}  {:7.2f}  {:7.2f}".format(s, p[0], p[1], p[2], r))

titre("8. Espace de travail")

for z in (0, 50, 100, 150):
    portees = [np.hypot(*robot.pose(robot.mgi_pose(rho, 0, z, 0))[:2])
               for rho in np.arange(60, 380, 2)
               if robot.atteignable(rho, 0, z, 0)]
    if portees:
        print("  z = {:3d} mm  ->  portee radiale de {:.0f} a {:.0f} mm"
              .format(z, min(portees), max(portees)))
    else:
        print("  z = {:3d} mm  ->  aucun point atteignable".format(z))
