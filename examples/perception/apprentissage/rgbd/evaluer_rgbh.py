#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison RGB / RGB-H / RG-H (et runs/v1) : mAP des masques sur la
validation REELLE et sur la meme validation a FOND CHANGE (exporter_rgbh.py).

    .venv/bin/python examples/perception/apprentissage/rgbd/evaluer_rgbh.py
    .venv/bin/python examples/perception/apprentissage/rgbd/evaluer_rgbh.py \
        --modele v1=runs/v1/weights/best.pt:rgb --modele x=chemin.pt:rgbh

Par defaut : runs/v1 (RGB, copier-coller : reference) et tous les
runs/rgbd_<jeu>/weights/best.pt trouves, chacun evalue sur le jeu de sa
variante d'entree (rgb, rgbh ou rgh). Ecrit rgbd/donnees/comparaison.json
et affiche un tableau (Markdown).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
APPRENTISSAGE = os.path.dirname(ICI)
for _d in (APPRENTISSAGE, ICI):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import commun  # noqa: E402

RUNS = os.path.join(APPRENTISSAGE, "runs")
DONNEES = os.path.join(ICI, "donnees")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def modeles_par_defaut():
    sortie = {}
    v1 = os.path.join(RUNS, "v1", "weights", "best.pt")
    if os.path.exists(v1):
        sortie["v1 (RGB + copier-coller)"] = (v1, "rgb")
    for d in sorted(glob.glob(os.path.join(RUNS, "rgbd_*"))):
        nom = os.path.basename(d)
        if "essai" in nom or nom.endswith(("_val", "_val_fond")):
            continue
        p = os.path.join(d, "weights", "best.pt")
        if os.path.exists(p):
            jeu = nom[len("rgbd_"):]
            sortie[nom] = (p, jeu.split("_")[0])
    return sortie


def evaluer(poids: str, variante: str, imgsz: int = 864, device=None, batch: int = 8,
            donnees: str = DONNEES) -> dict:
    from ultralytics import YOLO
    device = device or commun.device_auto()
    res = {}
    for cle, yaml in (("val", "data.yaml"), ("val_fond", "data_fond.yaml")):
        t = time.time()
        v = YOLO(poids).val(data=os.path.join(donnees, variante, yaml), split="val", imgsz=imgsz,
                            device=device, batch=batch, plots=False, verbose=False,
                            project=os.path.join(RUNS, "rgbd_eval"), name="x", exist_ok=True)
        noms = v.names if isinstance(v.names, dict) else dict(enumerate(v.names))
        res[cle] = {"masque_mAP50": round(float(v.seg.map50), 4),
                    "masque_mAP50_95": round(float(v.seg.map), 4),
                    "boite_mAP50": round(float(v.box.map50), 4),
                    "par_classe_masque_mAP50_95": {
                        noms[int(c)]: round(float(v.seg.maps[int(c)]), 4)
                        for c in getattr(v.seg, "ap_class_index", [])},
                    "ms_par_image": round(float(v.speed.get("inference", float("nan"))), 1),
                    "duree_s": round(time.time() - t, 1)}
    return res


def main():
    ap = argparse.ArgumentParser(description="mAP masques RGB vs RGB-H, fond reel et change")
    ap.add_argument("--modele", action="append", default=[],
                    help="nom=chemin.pt:variante (repetable) ; defaut : v1 et runs/rgbd_*")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sortie", default=os.path.join(DONNEES, "comparaison.json"))
    a = ap.parse_args()
    modeles = {}
    for m in a.modele:
        nom, reste = m.split("=", 1)
        chemin, variante = reste.rsplit(":", 1)
        modeles[nom] = (chemin, variante)
    modeles = modeles or modeles_par_defaut()
    resultats = {}
    for nom, (p, var) in modeles.items():
        print("evaluation", nom, "(", var, ")", flush=True)
        resultats[nom] = {"poids": p, "variante": var, **evaluer(p, var, device=a.device)}
    with open(a.sortie, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=1, ensure_ascii=False)
    print("\n| modele | entree | mAP50-95 masque (val) | mAP50 masque (val) | "
          "mAP50-95 masque (fond change) | mAP50 masque (fond change) |")
    print("|---|---|---|---|---|---|")
    for nom, r in resultats.items():
        print("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            nom, r["variante"], r["val"]["masque_mAP50_95"], r["val"]["masque_mAP50"],
            r["val_fond"]["masque_mAP50_95"], r["val_fond"]["masque_mAP50"]))
    print("\n->", a.sortie)


if __name__ == "__main__":
    main()
