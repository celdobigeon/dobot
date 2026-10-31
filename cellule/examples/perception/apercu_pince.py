"""
Apercu de l'etalonnage par la pince : la camera en continu, avec en
surimpression

  * en VERT, ce qui differe du decor (ce que l'etalonnage segmente : la
    pince, le bras) ;
  * en JAUNE, des que l'etalonnage a une premiere solution, le modele de la
    pince a la pose commandee, reduit a ses points saillants : le corps en
    fil de fer, le bout des doigts, l'axe de l'outil. Il doit tomber sur la
    pince : l'ecart se voit a l'oeil.

Lance par etalonner_par_pince.py (processus a part : sur macOS, une fenetre
OpenCV doit vivre dans le fil principal). Il lit l'etat que l'etalonnage
publie dans un petit fichier JSON, et s'arrete avec lui.

    .venv/bin/python examples/perception/apercu_pince.py ETAT.json [--pid N]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
sys.path.insert(0, ICI)

MODELE = os.path.join(RACINE, "assets", "pince_dobot.npz")
#: pieces de la CAO : le corps en aluminium, les deux doigts
CORPS, DOIGTS = 5, (3, 4)
#: m : au-dela, la difference avec le decor n'est que du bruit
PORTEE = 1.0
VERT, JAUNE = (80, 220, 80), (0, 220, 255)


def _boite(V):
    """Les 12 aretes de la boite englobante de V (mm, repere outil)."""
    lo, hi = V.min(0), V.max(0)
    c = np.array([[(lo, hi)[i >> 2 & 1][0], (lo, hi)[i >> 1 & 1][1], (lo, hi)[i & 1][2]]
                  for i in range(8)])
    aretes = [(a, b) for a in range(8) for b in range(a + 1, 8)
              if bin(a ^ b).count("1") == 1]
    return [(c[a], c[b]) for a, b in aretes]


def segments_saillants():
    """Segments (mm, repere outil) : corps, bouts des doigts, axe."""
    with np.load(MODELE) as z:
        V, P = z["sommets"].astype(float), z["piece"]
    segs = _boite(V[P == CORPS])
    for d in DOIGTS:
        bas = V[(P == d) & (V[:, 2] < 22.0)]
        if len(bas):
            segs += _boite(bas)
    segs.append((np.zeros(3), np.array([0.0, 0.0, float(V[:, 2].max())])))
    return segs


def lire_etat(chemin, ancien):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return ancien


def vivant(pid) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Apercu de l'etalonnage par la pince")
    ap.add_argument("etat")
    ap.add_argument("--pid", type=int, default=0, help="s'arreter avec ce processus")
    ap.add_argument("--cadence", type=float, default=8.0)
    a = ap.parse_args(argv)

    import cv2
    import camera
    import perception
    segs = segments_saillants()
    fenetre = "etalonnage par la pince"
    etat, fond, fond_de = {}, None, None
    periode = 1.0 / a.cadence
    while vivant(a.pid):
        t0 = time.time()
        etat = lire_etat(a.etat, etat)
        if etat.get("fin"):
            break
        if etat.get("decor") and etat["decor"] != fond_de:
            try:
                with np.load(etat["decor"]) as z:
                    fond = np.asarray(z["profondeur"], float)
                fond_de = etat["decor"]
            except (OSError, KeyError, ValueError):
                fond = None
        try:
            cap = camera.capturer(1, delai=5.0)
        except Exception:                               # noqa: BLE001
            time.sleep(0.5)
            continue
        vue = perception.vue_couleur(cap)
        if vue is None:
            continue
        img = vue.image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        T_cp = np.asarray(vue.T_cp, float)

        # ce qui differe du decor, reprojete dans l'image couleur
        z = np.asarray(cap["profondeur"], float)
        if fond is not None and fond.shape == z.shape:
            K = np.asarray(cap["K"], float)
            # le fond lointain (mur, meubles) est bruite : on se limite au
            # volume de travail, a moins de PORTEE de la camera
            m = (z > 0) & (z < PORTEE) & (fond > 0) & (fond - z > float(etat.get("seuil", 0.025)))
            v, u = np.nonzero(m)
            if len(u):
                zz = z[v, u]
                Q = np.c_[(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz]
                Q = Q @ T_cp[:3, :3].T + T_cp[:3, 3]
                uv = np.round(perception.projeter(Q, vue)).astype(int)
                H, W = img.shape[:2]
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                masque = np.zeros((H, W), np.uint8)
                masque[uv[ok, 1], uv[ok, 0]] = 255
                # la couleur voit plus serre que la profondeur : on bouche
                masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                sel = masque > 0
                img[sel] = (0.55 * img[sel] + 0.45 * np.array(VERT)).astype(np.uint8)
                contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, VERT, 1, cv2.LINE_AA)

        # le modele a la pose commandee, par la solution en cours
        if etat.get("M") and etat.get("T_ct") and etat.get("pose"):
            x, y, zp, r = etat["pose"]
            c, s = np.cos(np.radians(r)), np.sin(np.radians(r))
            A = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, zp], [0, 0, 0, 1.0]])
            T = (np.asarray(etat["T_ct"], float) @ np.linalg.inv(np.asarray(etat["M"], float))
                 @ A)                                     # outil -> camera, mm
            for p, q in segs:
                Q = np.array([p, q]) @ T[:3, :3].T + T[:3, 3]
                Q = Q / 1000.0 @ T_cp[:3, :3].T + T_cp[:3, 3]
                if (Q[:, 2] <= 0.05).any():
                    continue
                uv = np.round(perception.projeter(Q, vue)).astype(int)
                cv2.line(img, tuple(uv[0]), tuple(uv[1]), JAUNE, 1, cv2.LINE_AA)
            bout = (np.zeros(3) @ T[:3, :3].T + T[:3, 3]) / 1000.0 @ T_cp[:3, :3].T + T_cp[:3, 3]
            if bout[2] > 0.05:
                uv = np.round(perception.projeter(bout, vue)[0]).astype(int)
                cv2.circle(img, tuple(uv), 4, JAUNE, 1, cv2.LINE_AA)

        texte = etat.get("texte", "")
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, texte, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        legende = "vert : different du decor" + (
            "   jaune : modele de la pince (solution en cours)" if etat.get("M") else "")
        cv2.putText(img, legende, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(fenetre, img)
        attente = max(1, int(1000 * (periode - (time.time() - t0))))
        if (cv2.waitKey(attente) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
