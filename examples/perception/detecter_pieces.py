#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconnait les pieces d'une capture et les localise dans l'espace.

    .venv/bin/python examples/perception/detecter_pieces.py captures/scene3.npz
    .venv/bin/python examples/perception/detecter_pieces.py captures/scene3.npz --dimensions 16 30 68
    .venv/bin/python examples/perception/detecter_pieces.py --direct
    .venv/bin/python examples/perception/detecter_pieces.py --direct --enregistrer scene5
    .venv/bin/python examples/perception/detecter_pieces.py captures/scene9.npz --methode taches

Affiche, pour chaque piece : la face sur laquelle elle est posee, ses
dimensions mesurees, la position du centre de sa face superieure et son
orientation, dans le repere TABLE (z vers le haut) et dans le repere
CAMERA. Ecrit une image annotee <capture>_detection.png.

Deux methodes : "modele" (defaut, vision_modele.py) cherche des rectangles
de taille connue dans une vue de dessus redressee, et donne un score de
confiance par piece ; "taches" (vision.py) mesure chaque tache qui depasse
de la table. La premiere separe les pieces qui se touchent et tient mieux
en vue rasante (voir evaluer_detection.py) ; la seconde est plus simple et
plus rapide.

Sans sudo : soit sur un fichier ecrit par capture_realsense.py, soit
(--direct) sur une capture fraiche demandee au serveur de camera, qui doit
tourner dans un autre terminal :
    sudo .venv/bin/python examples/perception/serveur_camera.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision  # noqa: E402
import vision_modele  # noqa: E402

import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("capture", nargs="?",
                    help="fichier .npz ecrit par capture_realsense.py")
    ap.add_argument("--direct", action="store_true",
                    help="capture fraiche demandee au serveur de camera")
    ap.add_argument("--enregistrer", metavar="NOM",
                    help="avec --direct : garde la capture dans captures/NOM.npz")
    ap.add_argument("--dimensions", nargs=3, type=float, default=(16, 30, 68),
                    metavar=("A", "B", "C"), help="dimensions de la piece, mm")
    ap.add_argument("--piece", nargs=4, action="append", metavar=("NOM", "A", "B", "C"),
                    help="un type de piece et ses dimensions (mm) ; a repeter "
                         "pour plusieurs types, ex. --piece epaisse 20 20 45 "
                         "--piece fine 10 20 45")
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="ecart maximal accepte, mm (defaut 10)")
    ap.add_argument("--methode", choices=("modele", "taches"), default="modele",
                    help="modele : rectangles de taille connue (defaut) ; "
                         "taches : ce qui depasse de la table")
    args = ap.parse_args()
    if args.direct == bool(args.capture):
        ap.error("donnez un fichier de capture, OU --direct")

    if args.direct:
        import camera
        try:
            brute = camera.capturer()
        except camera.ErreurCamera as exc:
            sys.exit(str(exc))
        dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "captures")
        os.makedirs(dossier, exist_ok=True)
        nom = args.enregistrer or "derniere"
        args.capture = os.path.join(dossier, nom + ".npz")
        np.savez_compressed(args.capture, **brute)
        if args.enregistrer:
            print("capture enregistree : " + args.capture)

    scene = vision.charger(args.capture)
    dims = ({nom: (float(a), float(b), float(c)) for nom, a, b, c in args.piece}
            if args.piece else args.dimensions)
    if args.methode == "modele":
        res = vision_modele.detecter(scene, dims, tolerance=args.tolerance)
    else:
        res = vision.detecter(scene, dims, tolerance=args.tolerance)

    print("table : a {:.0f} mm de la camera, inclinee de {:.0f} deg sur son "
          "axe".format(res.distance_table * 1000,
                       np.degrees(np.arccos(abs(res.normale[2])))))
    print()
    if not res.pieces:
        print("aucune piece reconnue")
    for p in res.pieces:
        x, y, z = p.position_table
        xc, yc, zc = p.T_camera[:3, 3] * 1000
        confiance = (" (confiance {:.2f})".format(p.score)
                     if hasattr(p, "score") else "")
        print("piece {} : {}{}".format(p.numero, " ".join(filter(None, (p.type, p.face))),
                                       confiance))
        print("   dimensions mesurees : {:.0f} x {:.0f} x {:.0f} mm "
              "(ecart {:.1f} mm)".format(*p.dimensions, p.ecart))
        print("   face superieure, repere table  : x={:7.1f}  y={:7.1f}  "
              "z={:5.1f} mm   orientation {:+.0f} deg".format(x, y, z, p.lacet))
        print("   face superieure, repere camera : x={:7.1f}  y={:7.1f}  "
              "z={:5.1f} mm".format(xc, yc, zc))
    for a in res.autres:
        print("ecarte : " + a)
    if res.trous:
        print("{} zone(s) sans mesure de profondeur sur la table : piece "
              "trop brillante (infrarouge sature) ou dans un reflet ?"
              .format(len(res.trous)))

    sortie = os.path.splitext(args.capture)[0] + "_detection.png"
    dessiner = vision_modele.dessiner if args.methode == "modele" else vision.dessiner
    cv2.imwrite(sortie, dessiner(scene, res))
    print("\nimage annotee : " + sortie)


if __name__ == "__main__":
    main()
