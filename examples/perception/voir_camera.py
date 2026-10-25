"""
Controler la perception : la vue de la camera, les pieces detectees en
surimpression (boites 3D, repere de chaque piece), en continu.

    .venv/bin/python examples/perception/voir_camera.py
    .venv/bin/python dobot.py voir [--rapide] [--sortie vue.jpg]

Dans la fenetre : q ou Echap pour quitter, s pour enregistrer l'image,
espace pour figer / reprendre. Le terminal liste les pieces ; avec un
etalonnage camera -> robot, leur position dans le repere du robot, et
celles qui tombent sur le socle du robot (fausses pieces, ignorees par la
saisie) sont signalees.

Le robot n'est pas utilise : seul le serveur de camera doit tourner.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

#: gris des detections ignorees (dans l'emprise du robot)
GRIS_BGR = (150, 150, 150)


def charger_etalonnage(chemin):
    """T_robot_camera (m), ou None sans etalonnage utilisable."""
    try:
        with open(chemin, encoding="utf-8") as f:
            d = json.load(f)
        return np.array(d["T_robot_camera"], float)
    except (OSError, ValueError, KeyError):
        return None


def composer(perception, capture, pieces, etape):
    """Image couleur annotee, carte de profondeur a cote, bandeau en haut."""
    import cv2
    import saisir_piece as sp
    import saisir_piece_graphique as g
    vue = perception.derniere_vue
    gardees = [p for p in pieces if not sp.dans_le_robot(p)]
    img = g.dessiner_boites(vue.image, vue, gardees, perception.types)
    # les fausses pieces sur le socle du robot : en gris, barrees
    for p in pieces:
        if p in gardees:
            continue
        P = np.r_[np.asarray(p.T_camera, float)[:3, 3], 1.0] @ np.asarray(vue.T_cp, float).T
        u, v = (np.asarray(vue.K, float) @ P[:3])[:2] / P[2]
        u, v = int(u), int(v)
        cv2.drawMarker(img, (u, v), GRIS_BGR, cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(img, "robot", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    GRIS_BGR, 1, cv2.LINE_AA)
    z = np.asarray(capture["profondeur"], float)
    valides = z > 0
    carte = np.zeros(z.shape + (3,), np.uint8)
    if valides.any():
        lo, hi = np.percentile(z[valides], (2, 98))
        n = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
        carte = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
        carte[~valides] = 0
    h = img.shape[0]
    carte = cv2.resize(carte, (int(carte.shape[1] * h / carte.shape[0]), h))
    out = np.hstack([img, carte])
    texte = "{} : {} piece(s){}   [q] quitter  [s] enregistrer  [espace] figer".format(
        etape, len(gardees), " + {} sur le robot".format(len(pieces) - len(gardees))
        if len(pieces) > len(gardees) else "")
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, texte, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def lister(pieces) -> None:
    import saisir_piece as sp
    print("{} detection(s) :".format(len(pieces)))
    for k, p in enumerate(pieces):
        ou = ""
        if p.T_robot is not None:
            x, y, z = p.T_robot[:3, 3] * 1000.0
            ou = "  robot ({:6.1f} {:6.1f} {:5.1f}) mm".format(x, y, z)
            if sp.dans_le_robot(p):
                ou += "  SUR LE ROBOT (ignoree)"
        print("  {:2d} {:8s} {:15s} confiance {:.2f}{}".format(
            k, p.type, p.face, float(getattr(p, "confiance", 1.0)), ou))


def main(argv=None) -> int:
    import etalonner_camera_robot as ec
    ap = argparse.ArgumentParser(description="Vue camera, detections en surimpression")
    ap.add_argument("--rapide", action="store_true",
                    help="detection rapide seule (reseau, < 1 s), sans l'affinage")
    ap.add_argument("--images", type=int, default=8,
                    help="images par capture (mediane de la profondeur, 8)")
    ap.add_argument("--sortie", default=None,
                    help="une seule capture, enregistree dans ce fichier image, "
                         "sans fenetre")
    ap.add_argument("--etalonnage", default=ec.FICHIER)
    a = ap.parse_args(argv)

    import cv2
    import camera
    import perception
    if not camera.serveur_present():
        print("serveur de camera absent : lancez examples/perception/serveur_camera.py")
        return 1
    T = charger_etalonnage(a.etalonnage)
    if T is None:
        print("(pas d'etalonnage camera -> robot : positions dans le repere camera seulement)")
    print("chargement de la perception...")
    perc = perception.Perception(T_robot_camera=T)
    fenetre = "dobot : perception"
    fige, image = False, None
    while True:
        if not fige:
            t0 = time.time()
            cap = camera.capturer(a.images)
            pieces = []
            for etape, res in perc.detecter_par_etapes(cap):
                if res is None:
                    continue
                pieces = res
                image = composer(perc, cap, pieces, etape)
                if a.sortie is None:
                    cv2.imshow(fenetre, image)
                    cv2.waitKey(1)
                if a.rapide:
                    break
            lister(pieces)
            print("  ({:.1f} s)".format(time.time() - t0))
            if a.sortie is not None:
                cv2.imwrite(a.sortie, image)
                print("image enregistree : {}".format(a.sortie))
                return 0
        touche = cv2.waitKey(200 if fige else 30) & 0xFF
        if touche in (ord("q"), 27):
            break
        if touche == ord("s") and image is not None:
            nom = time.strftime("vue_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(nom, image)
            print("image enregistree : {}".format(nom))
        if touche == ord(" "):
            fige = not fige
            print("fige" if fige else "reprise")
        try:
            if cv2.getWindowProperty(fenetre, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrompu")
