#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare les deux methodes de detection (taches : vision.detecter ; modele :
vision_modele.detecter) sur des scenes synthetiques a verite connue, puis
sur les captures reelles.

    .venv/bin/python examples/perception/evaluer_detection.py
    .venv/bin/python examples/perception/evaluer_detection.py --scenes 4 --sans-reel
    .venv/bin/python examples/perception/evaluer_detection.py --param seuil_score=0.6

Synthetique : pour chaque configuration de camera, des scenes de 8 pieces
(epaisses 17 x 19 x 68 et fines 8 x 17 x 68, a plat ou sur la tranche,
dont 3 posees CONTRE une autre). On mesure le rappel (pieces trouvees /
posees), les faux positifs, l'erreur de position du centre de la face
superieure dans le plan de la table, l'erreur d'angle, la justesse du type
et de la face, et le rappel des pieces etroites et des pieces en contact.

Reel : pas de verite au millimetre, seulement le nombre de pieces trouvees
(compare au nombre de pieces posees, compte a l'oeil) et l'ecart moyen des
dimensions mesurees aux dimensions du modele.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthese as sy  # noqa: E402
import vision  # noqa: E402
import vision_modele  # noqa: E402

CONFIGS = {
    # comme scene4 / scene7 : camera a 0.5 m, inclinee de 35 deg
    "dessus": dict(cam=sy.Camera(), zone=(-110, 110, -80, 80),
                   table=(-0.4, 0.4, -0.3, 0.3), couleur=False, motif=False),
    # comme scene9 : camera reculee, pieces plus petites dans l'image
    "reculee": dict(cam=sy.Camera(distance=0.65, inclinaison=40),
                    zone=(-130, 130, -90, 90), table=(-0.45, 0.45, -0.35, 0.35),
                    couleur=False, motif=False, reflet=True),
    # comme env2 : vue RASANTE, nappe a fleurs, image couleur
    "rasante": dict(cam=sy.Camera(K=sy.K_848, taille=(848, 480), distance=1.0,
                                  inclinaison=65),
                    zone=(-170, 170, -90, 90), table=(-0.7, 0.7, -1.0, 0.6),
                    couleur=True, motif=True, lissage=3),
}

#: pieces posees sur les captures reelles, comptees sur les images
ATTENDU = {"scene4": 4, "scene7": 8, "scene9": 8, "essai_25": 8, "essai_50": 8,
           "essai_90": 8, "essai_coupe": 8, "env2": 8, "env2_couleur": 8,
           "env2_reglee": 8}

#: centre de la face superieure de chaque piece, en pixels, releve A L'OEIL
#: sur l'image infrarouge ou couleur (a 10 px pres environ)
ANNOTATIONS = {
    "scene4": [(345, 245), (283, 307), (357, 300), (415, 300)],
    "scene7": [(313, 245), (382, 226), (250, 300), (314, 316), (379, 298),
               (437, 281), (400, 353), (300, 370)],
    "scene9": [(403, 160), (338, 186), (283, 222), (340, 250), (405, 230),
               (458, 213), (333, 296), (425, 285)],
    "essai_50": [(365, 248), (440, 230), (297, 295), (357, 322), (435, 320),
                 (495, 300), (450, 365), (348, 360)],
    "essai_90": [(362, 248), (437, 230), (292, 300), (370, 320), (430, 310),
                 (490, 290), (450, 360), (355, 358)],
    "env2_reglee": [(325, 322), (365, 352), (415, 305), (480, 372), (480, 283),
                    (515, 318), (535, 282), (405, 275)],
}


def apparier_image(pieces, centres, tolerance: float = 18.0):
    """(vrais positifs, faux positifs) : centre du contour reprojete a moins
    de `tolerance` pixels d'un centre annote, une piece au plus par centre."""
    if not pieces:
        return 0, 0
    det = np.array([p.contour.mean(axis=0) for p in pieces], float)
    ref = np.array(centres, float)
    d = np.linalg.norm(det[:, None] - ref[None], axis=2)
    paires = sorted((d[i, j], i, j) for i in range(len(det)) for j in range(len(ref)))
    pd, pr = set(), set()
    for dd, i, j in paires:
        if dd <= tolerance and i not in pd and j not in pr:
            pd.add(i)
            pr.add(j)
    return len(pd), len(det) - len(pd)


def scenes(config: str, nombre: int, graine: int = 0):
    c = CONFIGS[config]
    for k in range(nombre):
        rng = np.random.default_rng(1000 * graine + k)
        pcs = sy.placer(rng, 8, sy.TYPES, zone=c["zone"], touchantes=3)
        bruit = sy.Bruit(reflet=c.get("reflet", False), lissage=c.get("lissage", 5))
        yield sy.scene(pcs, c["cam"], bruit, graine=k, couleur=c["couleur"],
                       motif=c["motif"], table=c["table"])


def evaluer_synthetique(nombre: int, params: dict, configs, methodes):
    lignes = []
    for config in configs:
        bilans = {m: sy.Bilan() for m in methodes}
        temps = {m: 0.0 for m in methodes}
        for sc, verite in scenes(config, nombre):
            for m in methodes:
                t = time.time()
                if m == "taches":
                    res = vision.detecter(sc, sy.TYPES)
                else:
                    res = vision_modele.detecter(sc, sy.TYPES, **params)
                temps[m] += time.time() - t
                bilans[m].ajouter(sy.comparer(res.pieces, verite))
        for m in methodes:
            r = bilans[m].resume()
            lignes.append((config, m, r, temps[m] / nombre))
    print("\nSCENES SYNTHETIQUES ({} par configuration, 8 pieces dont 3 en "
          "contact)".format(nombre))
    print("{:8s} {:7s} {:>6s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s} {:>6s} {:>6s} "
          "{:>6s} {:>6s} {:>6s} {:>5s}".format(
              "config", "methode", "rappel", "faux", "fines", "contact",
              "pos_med", "pos_p90", "ang_md", "ang_90", "z_med", "type", "face", "s"))
    for config, m, r, t in lignes:
        print("{:8s} {:7s} {:6.2f} {:5d} {:6.2f} {:7.2f} {:7.1f} {:7.1f} {:6.1f} "
              "{:6.1f} {:6.1f} {:6.2f} {:6.2f} {:5.1f}".format(
                  config, m, r["rappel"], r["faux"], r["rappel_fines"],
                  r["rappel_contact"], r["pos_med"], r["pos_p90"], r["ang_med"],
                  r["ang_p90"], r["z_med"], r["type"], r["face"], t))
    return lignes


def evaluer_reel(params: dict, methodes):
    dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    print("\nCAPTURES REELLES (pieces posees ; trouvees ; justes et fausses d'apres "
          "les centres releves a l'oeil ; ecart moyen des dimensions, mm)")
    for nom, attendu in ATTENDU.items():
        chemin = os.path.join(dossier, nom + ".npz")
        if not os.path.exists(chemin):
            continue
        sc = vision.charger(chemin)
        dims = (16, 30, 68) if nom == "scene4" else sy.TYPES
        cases = []
        for m in methodes:
            if m == "taches":
                res = vision.detecter(sc, dims)
            else:
                res = vision_modele.detecter(sc, dims, **params)
            ecart = np.mean([p.ecart for p in res.pieces]) if res.pieces else np.nan
            if nom in ANNOTATIONS:
                vp, fp = apparier_image(res.pieces, ANNOTATIONS[nom])
                cases.append("{}: {:2d} trouvees, {} justes, {} fausses, ecart {:4.1f}"
                             .format(m, len(res.pieces), vp, fp, ecart))
            else:
                cases.append("{}: {:2d} trouvees{:16s}ecart {:4.1f}"
                             .format(m, len(res.pieces), "", ecart))
        print("  {:12s} {} posees   {}".format(nom, attendu, "   ".join(cases)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scenes", type=int, default=10,
                    help="scenes synthetiques par configuration (defaut 10)")
    ap.add_argument("--config", action="append", choices=sorted(CONFIGS),
                    help="configuration(s) a evaluer (defaut : toutes)")
    ap.add_argument("--methode", action="append", choices=("taches", "modele"),
                    help="methode(s) a evaluer (defaut : les deux)")
    ap.add_argument("--sans-reel", action="store_true")
    ap.add_argument("--sans-synthese", action="store_true")
    ap.add_argument("--param", action="append", default=[], metavar="NOM=VALEUR",
                    help="parametre de vision_modele.detecter, ex. seuil_score=0.6")
    args = ap.parse_args()
    params = {}
    for p in args.param:
        k, v = p.split("=")
        params[k] = float(v)
    methodes = args.methode or ["taches", "modele"]
    if not args.sans_synthese:
        evaluer_synthetique(args.scenes, params, args.config or list(CONFIGS), methodes)
    if not args.sans_reel:
        evaluer_reel(params, methodes)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
