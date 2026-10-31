"""
Controler la perception : la vue de la camera EN STREAMING (une dizaine
d'images par seconde), avec en surimpression les pieces de la derniere
detection (boites 3D, repere de chaque piece). La detection tourne en
continu a cote, sur des captures moyennees : ses boites sont remplacees a
chaque nouveau resultat.

    .venv/bin/python examples/perception/voir_camera.py
    .venv/bin/python dobot.py voir [--rapide] [--sortie vue.jpg]

Dans la fenetre : q ou Echap pour quitter, s pour enregistrer l'image,
espace pour figer la detection (le flux continue). Le terminal liste les pieces ; avec un
etalonnage camera -> robot, leur position dans le repere du robot, et
celles qui tombent sur le socle du robot (fausses pieces, ignorees par la
saisie) sont signalees.

Le robot n'est pas utilise : seul le serveur de camera doit tourner.
"""

import argparse
import json
import os
import sys
import threading
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


def composer(perception, capture, pieces, etape, vue=None):
    """Image couleur annotee, carte de profondeur a cote, bandeau en haut.
    `vue` : l'image couleur a annoter (par defaut, celle de la detection)."""
    import cv2
    import saisir_piece as sp
    import saisir_piece_graphique as g
    if vue is None:
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
    texte = "{} : {} piece(s){}   [q] quitter  [s] enregistrer  [espace] figer la detection".format(
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


class Detection(threading.Thread):
    """La detection en continu : (etape, pieces) de la derniere passe."""

    def __init__(self, perception, images: int, rapide: bool):
        super().__init__(daemon=True)
        self.perception, self.images, self.rapide = perception, images, rapide
        self.verrou = threading.Lock()
        self.resultat = ("en attente", [])
        self.pause = threading.Event()
        self.fin = threading.Event()

    def run(self):
        import camera
        while not self.fin.is_set():
            if self.pause.is_set():
                time.sleep(0.1)
                continue
            try:
                t0 = time.time()
                cap = camera.capturer(self.images)
                pieces = []
                for etape, res in self.perception.detecter_par_etapes(cap):
                    if res is None:
                        continue
                    pieces = res
                    with self.verrou:
                        self.resultat = (etape, pieces)
                    if self.rapide:
                        break
                lister(pieces)
                print("  ({:.1f} s)".format(time.time() - t0))
            except Exception as exc:                    # noqa: BLE001
                print("  (detection : {})".format(exc))
                time.sleep(1.0)


def une_capture(perc, images, rapide, sortie) -> int:
    """--sortie : une capture, une detection, l'image dans un fichier."""
    import cv2
    import camera
    cap = camera.capturer(images)
    pieces, etape = [], ""
    for etape, res in perc.detecter_par_etapes(cap):
        if res is not None:
            pieces = res
            if rapide:
                break
    lister(pieces)
    cv2.imwrite(sortie, composer(perc, cap, pieces, etape))
    print("image enregistree : {}".format(sortie))
    return 0


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
    ap.add_argument("--cadence", type=float, default=10.0,
                    help="images par seconde du flux (10)")
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
    if a.sortie is not None:
        return une_capture(perc, a.images, a.rapide, a.sortie)
    fenetre = "dobot : perception"
    det = Detection(perc, a.images, a.rapide)
    det.start()
    periode = 1.0 / max(a.cadence, 1.0)
    image = None
    try:
        while True:
            t0 = time.time()
            try:
                cap = camera.capturer(1, delai=5.0)
                vue = perception.vue_couleur(cap)
            except Exception as exc:                    # noqa: BLE001
                print("  (camera : {})".format(exc))
                time.sleep(0.5)
                continue
            with det.verrou:
                etape, pieces = det.resultat
            if vue is not None:
                if det.pause.is_set():
                    etape = "detection figee"
                image = composer(perc, cap, pieces, etape, vue)
                cv2.imshow(fenetre, image)
            attente = max(1, int(1000 * (periode - (time.time() - t0))))
            touche = cv2.waitKey(attente) & 0xFF
            if touche in (ord("q"), 27):
                break
            if touche == ord("s") and image is not None:
                nom = time.strftime("vue_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(nom, image)
                print("image enregistree : {}".format(nom))
            if touche == ord(" "):
                if det.pause.is_set():
                    det.pause.clear()
                    print("detection reprise")
                else:
                    det.pause.set()
                    print("detection figee (le flux continue)")
            try:
                if cv2.getWindowProperty(fenetre, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        det.fin.set()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrompu")
