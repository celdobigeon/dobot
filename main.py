#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point d'entree du simulateur Dobot Magician.

    python main.py              lance l'interface graphique
    python main.py --verifier   controle l'installation et le modele 3D
"""

from __future__ import annotations

import sys


def verifier() -> int:
    """Diagnostic sans interface : dependances, modele CAO, coherence MGD/MGI."""
    import importlib

    print("Python", sys.version.split()[0])
    ok = True
    for nom in ("numpy", "pyqtgraph"):
        try:
            m = importlib.import_module(nom)
            print("  {:<12} {}".format(nom, getattr(m, "__version__", "?")))
        except ImportError as exc:
            print("  {:<12} ABSENT ({})".format(nom, exc))
            ok = False
    for nom in ("PyQt5", "PySide6", "PySide2", "PyQt6"):
        try:
            importlib.import_module(nom)
            print("  liaison Qt   {}".format(nom))
            break
        except ImportError:
            continue
    else:
        print("  liaison Qt   ABSENTE (installez PyQt5)")
        ok = False
    try:
        importlib.import_module("OpenGL")
        print("  PyOpenGL     present -> rendu 3D disponible")
    except ImportError:
        print("  PyOpenGL     absent  -> repli sur les vues 2D")

    if not ok:
        print("\nInstallation incomplete : pip install -r requirements.txt")
        return 1

    import numpy as np
    from dobot_sim import DobotMagician
    from dobot_sim import meshes as mm

    robot = DobotMagician()
    print("\n{}".format(robot))

    # coherence MGD / MGI sur un echantillon de l'espace articulaire
    err_pos = err_ang = 0.0
    rng = np.random.default_rng(0)
    for _ in range(2000):
        q = np.array([rng.uniform(a, b) for a, b in
                      zip(robot.lim.q_min, robot.lim.q_max)])
        p0 = robot.pose(q)
        p1 = robot.pose(robot.mgi(robot.mgd(q), tolerant=True))
        err_pos = max(err_pos, float(np.max(np.abs(p0[:3] - p1[:3]))))
        # r est defini modulo 360 deg
        da = (p0[3] - p1[3] + 180.0) % 360.0 - 180.0
        err_ang = max(err_ang, abs(float(da)))
    print("MGD/MGI : ecart maximal sur 2000 tirages = {:.3e} mm, {:.3e} deg"
          .format(err_pos, err_ang))

    # jacobienne analytique contre differences finies
    ecart = 0.0
    for _ in range(200):
        q = np.array([rng.uniform(a, b) for a, b in
                      zip(robot.lim.q_min, robot.lim.q_max)])
        ecart = max(ecart, float(np.max(np.abs(
            robot.jacobienne(q) - robot.jacobienne_numerique(q)))))
    print("Jacobienne : ecart analytique / numerique = {:.3e}".format(ecart))

    print("\nModele 3D :")
    pieces = mm.charger_pieces(robot, journal=lambda m: print("  " + m))
    if not pieces:
        print("  aucun maillage trouve : le simulateur utilisera le modele simplifie")
    else:
        total = sum(p.n_faces for p in pieces)
        print("  {} pieces, {} facettes au total".format(len(pieces), total))
    return 0


def main() -> int:
    if "--verifier" in sys.argv or "--check" in sys.argv:
        return verifier()
    try:
        from dobot_sim.ui import lancer
    except ImportError as exc:
        print("Dependance manquante :", exc)
        print("Installez-les avec :  pip install -r requirements.txt")
        print("Diagnostic detaille  :  python main.py --verifier")
        return 1
    return lancer()


if __name__ == "__main__":
    sys.exit(main())
