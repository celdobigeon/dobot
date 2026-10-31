#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrainement (fine-tuning) du reseau de segmentation des pieces.

    .venv/bin/python examples/perception/apprentissage/entrainer.py
    .venv/bin/python examples/perception/apprentissage/entrainer.py --modele yolo11n-seg.pt --epochs 100
    .venv/bin/python examples/perception/apprentissage/entrainer.py --demo     # 15 epoques, pour voir si tout tourne
                                                    # (-> modeles/pieces-seg.demo.pt)

Etapes : export du jeu annote (exporter.py, copier-coller compris), puis
fine-tuning de YOLO11-seg pre-entraine sur COCO, validation (mAP des
masques), et copie du MEILLEUR poids (au sens de la validation) dans
examples/perception/modeles/pieces-seg.pt, que perception.Perception
charge tout seul. L'ancien poids est garde (pieces-seg.precedent.pt).

Pourquoi ces reglages (petit jeu, petits objets allonges, camera fixe) :
  - imgsz 864 : l'image (848 x 480) n'est PAS reduite ; une piece fine
    vue de loin ne fait que 10 a 15 pixels de large ;
  - mask_ratio 2 : masques predits a resolution double du defaut ; les
    bords d'une piece de 8 mm comptent ;
  - flipud 0 : la camera regarde en biais, une image retournee de haut en
    bas n'arrive jamais ; fliplr 0.5 oui ;
  - degrees 10, scale 0.3, translate 0.1 : variations moderees (vue fixe) ;
  - mosaic + copy_paste 0.3 (ultralytics) en plus du copier-coller
    plausible d'exporter.py ; close_mosaic 10 : les dernieres epoques sur
    des images entieres, comme a l'usage ;
  - patience 30 : arret si la validation ne progresse plus ; cos_lr.
  - yolo11s-seg par defaut : bon compromis ; yolo11n-seg est 3 fois plus
    rapide, un peu moins precis.

Options pour les images RENDUES (synthese_images.py), sans effet par defaut :
  --synthetiques DOSSIER [--n-synth N] : ajoutees a l'entrainement seulement
    (la validation reste reelle) ;
  --export DOSSIER : ou ecrire l'export YOLO (defaut dataset/yolo) ; pour
    entrainer en parallele ou comparer sans ecraser l'export d'un autre ;
  --test DATA.yaml (repetable) : evaluation supplementaire du meilleur
    poids, par ex. sur synthetique/test_fonds/data.yaml (fonds jamais vus) ;
    les mAP vont dans rapport.json, cle "tests".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun  # noqa: E402
import exporter  # noqa: E402

RUNS = os.path.join(commun.ICI, "runs")
#: certaines operations manquent encore a MPS : repli sur le CPU, pas d'erreur
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def entrainer(modele="yolo11s-seg.pt", epochs=100, imgsz=864, batch=8, device=None,
              patience=30, copies=2, tout=False, nom=None, racine=commun.DATASET,
              workers=2, installer=True, fraction_val=0.2,
              nom_installe="pieces-seg.pt", synthetiques=None, n_synth=0,
              sortie_export=None, tests=()) -> dict:
    from ultralytics import YOLO
    debut = time.time()
    stats = exporter.exporter(racine, sortie=sortie_export, tout=tout, copies=copies,
                              val=fraction_val, synthetiques=synthetiques, n_synth=n_synth)
    print("export : {} images reelles + {} synthetiques{} en entrainement, {} en "
          "validation".format(stats["train"], stats["synthetiques"],
                              " + {} rendues".format(stats["rendues"]) if stats["rendues"] else "",
                              stats["val"]), flush=True)
    poids = modele if os.path.exists(modele) else os.path.join(commun.MODELES, modele)
    device = device or commun.device_auto()
    nom = nom or time.strftime("pieces_%Y%m%d_%H%M%S")
    m = YOLO(poids)
    m.train(data=stats["data"], project=RUNS, name=nom, exist_ok=True,
            epochs=epochs, imgsz=imgsz, batch=batch, device=device, patience=patience,
            workers=workers, seed=0, cos_lr=True, close_mosaic=min(10, max(epochs // 3, 1)),
            degrees=10.0, flipud=0.0, fliplr=0.5, mosaic=1.0, scale=0.3, translate=0.1,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, copy_paste=0.3, copy_paste_mode="flip",
            mask_ratio=2, overlap_mask=True, plots=True, verbose=False)
    dossier = os.path.join(RUNS, nom)
    meilleur = os.path.join(dossier, "weights", "best.pt")
    if not os.path.exists(meilleur):
        meilleur = os.path.join(dossier, "weights", "last.pt")
    # validation finale du meilleur poids, sur les images reelles de validation
    v = YOLO(meilleur).val(data=stats["data"], split="val", imgsz=imgsz, device=device,
                           batch=batch, plots=False, verbose=False,
                           project=RUNS, name=nom + "_val", exist_ok=True)
    rapport = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "modele_depart": os.path.basename(poids),
        "epochs": epochs, "imgsz": imgsz, "device": device,
        "images_train": stats["train"], "images_synthetiques": stats["synthetiques"],
        "images_val": stats["val"], "instances_val": stats["instances_val"],
        "classes": stats["classes"],
        "masque_mAP50": round(float(v.seg.map50), 4),
        "masque_mAP50_95": round(float(v.seg.map), 4),
        "boite_mAP50": round(float(v.box.map50), 4),
        "boite_mAP50_95": round(float(v.box.map), 4),
        "par_classe_masque_mAP50_95": {stats["classes"][int(c)]: round(float(x), 4)
                                       for c, x in zip(v.seg.ap_class_index, v.seg.maps[v.seg.ap_class_index])}
        if len(getattr(v.seg, "ap_class_index", [])) else {},
        "duree_min": round((time.time() - debut) / 60, 1),
        "poids": meilleur,
    }
    if stats.get("rendues"):
        rapport["images_rendues"] = stats["rendues"]
        rapport["synthetiques_dossier"] = synthetiques
    for data_test in tests or ():
        # evaluation supplementaire (ex. fonds jamais vus) : jeu "val" du yaml
        nom_test = os.path.basename(os.path.dirname(os.path.abspath(data_test)))
        t = YOLO(meilleur).val(data=data_test, split="val", imgsz=imgsz, device=device,
                               batch=batch, plots=False, verbose=False,
                               project=RUNS, name=nom + "_test_" + nom_test, exist_ok=True)
        rapport.setdefault("tests", {})[nom_test] = {
            "masque_mAP50": round(float(t.seg.map50), 4),
            "masque_mAP50_95": round(float(t.seg.map), 4),
            "boite_mAP50": round(float(t.box.map50), 4)}
    if installer:
        cible = os.path.join(commun.MODELES, nom_installe)
        if os.path.exists(cible):
            shutil.copy2(cible, cible[:-3] + ".precedent.pt")
        shutil.copy2(meilleur, cible)
        rapport["installe"] = cible
    with open(os.path.join(dossier, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=1, ensure_ascii=False)
    return rapport


def main():
    ap = argparse.ArgumentParser(description="Entrainement YOLO11-seg des pieces")
    ap.add_argument("--modele", default="yolo11s-seg.pt", help="poids de depart (modeles/)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=864)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="mps, cpu, 0 (cuda)... (defaut : auto)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--copies", type=int, default=2, help="copier-coller par image (exporter.py)")
    ap.add_argument("--tout", action="store_true", help="aussi les images non validees")
    ap.add_argument("--nom", default=None, help="nom de l'entrainement (dossier runs/)")
    ap.add_argument("--sans-installer", action="store_true",
                    help="ne pas copier le resultat dans modeles/pieces-seg.pt")
    ap.add_argument("--demo", action="store_true",
                    help="entrainement court de verification (15 epoques, --tout) ; "
                    "installe dans modeles/pieces-seg.demo.pt, PAS pieces-seg.pt")
    ap.add_argument("--synthetiques", default=None,
                    help="dossier d'images rendues (synthese_images.py) ajoutees a "
                    "l'entrainement (defaut : aucune)")
    ap.add_argument("--n-synth", type=int, default=0,
                    help="nombre d'images rendues a prendre (0 : toutes)")
    ap.add_argument("--export", default=None,
                    help="dossier de l'export YOLO (defaut : dataset/yolo)")
    ap.add_argument("--test", action="append", default=[],
                    help="data.yaml d'un jeu de test supplementaire (repetable)")
    a = ap.parse_args()
    if a.demo:
        a.epochs, a.tout, a.patience = 15, True, 15
    r = entrainer(a.modele, a.epochs, a.imgsz, a.batch, a.device, a.patience, a.copies,
                  a.tout, a.nom, installer=not a.sans_installer,
                  nom_installe="pieces-seg.demo.pt" if a.demo else "pieces-seg.pt",
                  synthetiques=a.synthetiques, n_synth=a.n_synth, sortie_export=a.export,
                  tests=a.test)
    print("\nRAPPORT")
    for k, v in r.items():
        print("  {:28s} {}".format(k, v))
    if r["images_val"] < 10:
        print("  ATTENTION : {} image(s) de validation seulement : ces scores ne disent "
              "presque rien. Visez 150 images et plus.".format(r["images_val"]))


if __name__ == "__main__":
    main()
