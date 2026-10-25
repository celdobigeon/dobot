#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export du jeu annote au format YOLO-seg (ultralytics), avec copier-coller.

    .venv/bin/python examples/perception/apprentissage/exporter.py
    .venv/bin/python examples/perception/apprentissage/exporter.py --copies 3 --val 0.2
    .venv/bin/python examples/perception/apprentissage/exporter.py --tout   # aussi les "a revoir"

Ecrit dataset/yolo/ (efface et regenere a chaque fois) :
    images/{train,val}/NNNN.png
    labels/{train,val}/NNNN.txt   une ligne par piece : "classe x1 y1 x2 y2 ..."
                                  (polygone, coordonnees divisees par W et H)
    data.yaml                     chemins et noms des classes (= les types)

DECOUPAGE reproductible et STABLE : une image va en validation selon une
empreinte de son nom (crc32), pas selon un tirage. Ajouter des images ne
deplace donc jamais une image deja placee : les scores de validation restent
comparables d'un entrainement a l'autre, et jamais une image de validation
ne sert a l'entrainement.

COPIER-COLLER (--copies N, voir copier_coller.py) : N images synthetiques
par image d'entrainement, faites de pieces decoupees dans les images
d'ENTRAINEMENT seulement, collees sur ces images et sur les fonds
(dataset/fonds/). La validation reste faite d'images reelles intactes.

IMAGES RENDUES (--synthetiques DOSSIER [--n-synth N], voir
synthese_images.py) : N images d'un jeu genere par randomisation de domaine
sont ajoutees a l'ENTRAINEMENT seulement (prefixe sy_) ; la validation
reste faite d'images reelles. Sans cette option, rien ne change.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import zlib

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun  # noqa: E402
import copier_coller  # noqa: E402


def en_validation(nom: str, fraction: float) -> bool:
    return (zlib.crc32(nom.encode("utf-8")) % 1000) / 1000.0 < fraction


def ligne_yolo(classe: int, polygones, W: int, H: int) -> str:
    """Une instance -> une ligne YOLO-seg. Plusieurs morceaux (piece coupee
    par une autre) sont relies en un seul contour, comme le fait ultralytics
    pour COCO (merge_multi_segment)."""
    segs = [np.asarray(p, float).reshape(-1, 2) for p in polygones if len(p) >= 3]
    if len(segs) > 1:
        from ultralytics.data.converter import merge_multi_segment
        s = np.concatenate(merge_multi_segment([x.reshape(-1).tolist() for x in segs]), axis=0)
    else:
        s = segs[0]
    s = s / np.array([W, H], float)
    s = np.clip(s, 0.0, 1.0)
    return "{} ".format(classe) + " ".join("{:.6f}".format(x) for x in s.reshape(-1))


def masques_de(ann):
    taille = tuple(ann["taille"])
    return ([commun.polygones_vers_masque(i["polygones"], taille) for i in ann["instances"]],
            [i["classe"] for i in ann["instances"]])


def ecrire(dossier_img, dossier_lab, nom, image, polys_classes, noms_classes):
    H, W = image.shape[:2]
    cv2.imwrite(os.path.join(dossier_img, nom + ".png"), image)
    lignes = []
    for polygones, classe in polys_classes:
        if classe not in noms_classes or not polygones:
            continue
        lignes.append(ligne_yolo(noms_classes.index(classe), polygones, W, H))
    with open(os.path.join(dossier_lab, nom + ".txt"), "w") as f:
        f.write("\n".join(lignes) + ("\n" if lignes else ""))
    return len(lignes)


def ajouter_rendues(dossier_synth, n, dirs_train, graine=0) -> tuple:
    """Copie n images rendues (et leurs etiquettes) de dossier_synth/images
    et /labels dans l'entrainement ; n <= 0 : toutes. Tirage reproductible
    (graine). Rend (images, pieces)."""
    images = sorted(glob.glob(os.path.join(dossier_synth, "images", "*.jpg"))
                    + glob.glob(os.path.join(dossier_synth, "images", "*.png")))
    if not images:
        raise SystemExit("aucune image rendue dans {} (synthese_images.py)".format(dossier_synth))
    if 0 < n < len(images):
        rng = np.random.default_rng(graine)
        images = [images[i] for i in sorted(rng.choice(len(images), n, replace=False))]
    pieces = 0
    for f in images:
        base = os.path.splitext(os.path.basename(f))[0]
        lab = os.path.join(dossier_synth, "labels", base + ".txt")
        if not os.path.exists(lab):
            continue
        shutil.copy2(f, os.path.join(dirs_train[0], "sy_" + os.path.basename(f)))
        shutil.copy2(lab, os.path.join(dirs_train[1], "sy_" + base + ".txt"))
        with open(lab) as fl:
            pieces += sum(1 for ligne in fl if ligne.strip())
    return len(images), pieces


def exporter(racine=commun.DATASET, sortie=None, tout=False, val=0.2, copies=0,
             graine=0, collages=(2, 6), synthetiques=None, n_synth=0) -> dict:
    sortie = sortie or os.path.join(racine, "yolo")
    types = commun.charger_types(racine)
    classes = list(types)
    noms = []
    for n in commun.noms_captures(racine):
        ann = commun.lire_annotation(n, racine)
        if ann is not None and (tout or ann.get("statut") == "valide"):
            noms.append(n)
    if not noms:
        raise SystemExit("aucune image annotee{} : annoter.py (ou --tout)".format(
            "" if tout else " ET validee"))
    val_noms = [n for n in noms if en_validation(n, val)]
    train_noms = [n for n in noms if n not in val_noms]
    if not val_noms:                     # tres petit jeu : au moins une
        val_noms = [noms[-1]]
        train_noms = [n for n in noms if n != noms[-1]] or [noms[-1]]
    if not train_noms:
        train_noms = [val_noms.pop()] if len(val_noms) > 1 else list(val_noms)
    for sous in ("images", "labels"):
        shutil.rmtree(os.path.join(sortie, sous), ignore_errors=True)
    dirs = {}
    for part in ("train", "val"):
        dirs[part] = (commun.dossier(sortie, os.path.join("images", part)),
                      commun.dossier(sortie, os.path.join("labels", part)))
    stats = {"train": 0, "val": 0, "instances_train": 0, "instances_val": 0,
             "synthetiques": 0, "classes": classes}
    banque = []
    cibles = []
    for part, liste in (("train", train_noms), ("val", val_noms)):
        for n in liste:
            ann = commun.lire_annotation(n, racine)
            image = commun.charger_image(n, racine)
            k = ecrire(*dirs[part], n, image,
                       [(i["polygones"], i["classe"]) for i in ann["instances"]], classes)
            stats[part] += 1
            stats["instances_" + part] += k
            if part == "train" and copies:
                masques, cls = masques_de(ann)
                sures = [j for j, i in enumerate(ann["instances"]) if not i.get("douteux")]
                banque += copier_coller.extraire(image, [masques[j] for j in sures],
                                                 [cls[j] for j in sures],
                                                 ann.get("plan"), ann.get("K"))
                cibles.append((n, image, masques, cls, ann.get("plan"), ann.get("K")))
    # fonds sans piece : cibles supplementaires
    for chemin in sorted(glob.glob(os.path.join(racine, "fonds", "*.png"))):
        plan = K = None
        npz = chemin[:-4] + ".npz"
        if os.path.exists(npz):
            try:
                from commun import perception
                cap = perception.charger_capture(npz)
                vue = perception.vue_couleur(cap, infrarouge=True)
                tab = perception.analyser_table(cap["profondeur"], np.asarray(cap["K"], float))
                plan, K = tab.plan_dans(vue).tolist(), vue.K.tolist()
            except Exception:
                pass
        cibles.append((os.path.basename(chemin)[:-4], cv2.imread(chemin), [], [], plan, K))
    if copies and banque:
        rng = np.random.default_rng(graine)
        for (n, image, masques, cls, plan, K) in cibles:
            for c in range(copies):
                img2, m2, c2 = copier_coller.coller(image, masques, cls, banque, rng,
                                                    int(rng.integers(collages[0], collages[1] + 1)),
                                                    plan, K)
                polys = [(commun.masque_vers_polygones(m), cl) for m, cl in zip(m2, c2)]
                stats["instances_train"] += ecrire(*dirs["train"], "cc_{}_{}".format(n, c),
                                                   img2, polys, classes)
                stats["synthetiques"] += 1
    stats["rendues"] = 0
    if synthetiques:
        k, p = ajouter_rendues(synthetiques, n_synth, dirs["train"], graine)
        stats["rendues"] = k
        stats["instances_train"] += p
    with open(os.path.join(sortie, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# genere par exporter.py : ne pas modifier a la main\n")
        f.write("path: {}\ntrain: images/train\nval: images/val\n".format(os.path.abspath(sortie)))
        f.write("names:\n")
        for k, c in enumerate(classes):
            f.write("  {}: {}\n".format(k, c))
    stats["data"] = os.path.join(sortie, "data.yaml")
    stats["val_noms"] = val_noms
    return stats


def main():
    ap = argparse.ArgumentParser(description="Export YOLO-seg")
    ap.add_argument("--dataset", default=commun.DATASET)
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--tout", action="store_true", help="exporter aussi les images a revoir")
    ap.add_argument("--val", type=float, default=0.2, help="fraction en validation")
    ap.add_argument("--copies", type=int, default=2,
                    help="images copier-coller par image d'entrainement (0 : aucune)")
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--synthetiques", default=None,
                    help="dossier d'images rendues (synthese_images.py) a ajouter a "
                    "l'entrainement (defaut : aucune)")
    ap.add_argument("--n-synth", type=int, default=0,
                    help="nombre d'images rendues a prendre (0 : toutes)")
    a = ap.parse_args()
    s = exporter(a.dataset, a.sortie, a.tout, a.val, a.copies, a.graine,
                 synthetiques=a.synthetiques, n_synth=a.n_synth)
    print("entrainement : {} images reelles + {} copier-coller + {} rendues, {} pieces ; "
          "validation : {} images, {} pieces ; classes {}\n-> {}".format(
              s["train"], s["synthetiques"], s["rendues"], s["instances_train"], s["val"],
              s["instances_val"], s["classes"], s["data"]))


if __name__ == "__main__":
    main()
