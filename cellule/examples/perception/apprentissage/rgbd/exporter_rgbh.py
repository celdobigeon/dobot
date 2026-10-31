#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export des jeux YOLO-seg a entree RGB-H (couleur + hauteur, rgbh.py).

    .venv/bin/python examples/perception/apprentissage/rgbd/exporter_rgbh.py
    .venv/bin/python examples/perception/apprentissage/rgbd/exporter_rgbh.py --fonds 2

Ecrit, pour chaque variante (rgb, rgbh, rgh), dans rgbd/donnees/<variante>/ :
    images/train/NNNN.png       images REELLES d'entrainement
    images/val/NNNN.png         validation reelle : les MEMES images que
                                runs/v1 (exporter.en_validation, crc32)
    images/val_fond/NNNN_k.png  la validation, FOND CHANGE (rgbh.changer_fond :
                                table repeinte, familles de test bois et
                                carrelage, hauteur intacte), 2 versions
    labels/...                  polygones (exporter.ligne_yolo), memes
                                masques pour toutes les variantes
    data.yaml                   train / val (channels: 4 pour rgbh)
    data_fond.yaml              val = val_fond (pour evaluer)
Avec --fonds N, une variante <variante>_fonds/ de plus : N copies de chaque
image d'entrainement dont la table est repeinte avec les AUTRES familles
(jamais bois ni carrelage) : augmentation de fond rendue possible par la
hauteur (on sait ou est la table), a comparer a la meme chose en RGB.

Pas de copier-coller ici (il faudrait aussi coller la hauteur) : la
comparaison juste est entre variantes de CET export ; runs/v1 (RGB +
copier-coller) est donne en plus, a titre de reference.

Les cartes de hauteur sont mises en cache (donnees/hauteurs/NNNN.npz).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
APPRENTISSAGE = os.path.dirname(ICI)
for _d in (APPRENTISSAGE, ICI):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import commun  # noqa: E402
from commun import perception  # noqa: E402
import rgbh  # noqa: E402

DONNEES = os.path.join(ICI, "donnees")
VARIANTES = ("rgb", "rgbh", "rgh")
FAMILLES_TEST = ("bois", "carrelage")
FAMILLES_ENTRAINEMENT = ("uni", "bruit", "tissu", "motifs", "papier")


def hauteur_cachee(nom: str, racine: str = commun.DATASET,
                   cache: str = os.path.join(DONNEES, "hauteurs")) -> Dict[str, np.ndarray]:
    """{"h": hauteur rebouchee (m, float32), "carte": mesuree (nan sinon),
    "mesuree": bool} pour une capture du jeu ; calculee une fois."""
    chemin = os.path.join(cache, nom + ".npz")
    if os.path.exists(chemin):
        with np.load(chemin) as z:
            carte = z["carte"].astype(np.float32)
            return {"h": z["h"].astype(np.float32), "carte": carte,
                    "mesuree": np.isfinite(carte)}
    cap = perception.charger_capture(os.path.join(racine, "brut", nom + ".npz"))
    vue = perception.vue_couleur(cap)
    carte, mes = rgbh.carte_hauteur(cap, vue)
    h = rgbh.boucher(carte)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(chemin, h=h.astype(np.float16), carte=carte.astype(np.float16))
    return {"h": h, "carte": carte.astype(np.float32), "mesuree": mes}


def _yaml(chemin: str, racine_jeu: str, val: str, classes: List[str], canaux: int) -> None:
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("# genere par rgbd/exporter_rgbh.py : ne pas modifier a la main\n")
        f.write("path: {}\ntrain: images/train\nval: images/{}\n".format(
            os.path.abspath(racine_jeu), val))
        if canaux != 3:
            f.write("channels: {}\n".format(canaux))
        f.write("names:\n")
        for k, c in enumerate(classes):
            f.write("  {}: {}\n".format(k, c))


def exporter(racine: str = commun.DATASET, sortie: str = DONNEES,
             variantes=VARIANTES, versions_fond: int = 2, fonds_train: int = 0,
             val: float = 0.2, graine: int = 0, verbeux: bool = True) -> dict:
    import exporter as exp_seg
    classes = list(commun.charger_types(racine))
    noms = [n for n in commun.noms_captures(racine)
            if (commun.lire_annotation(n, racine) or {}).get("statut") == "valide"]
    jeux = list(variantes) + ([v + "_fonds" for v in variantes] if fonds_train else [])
    for j in jeux:
        for s in ("images", "labels"):
            shutil.rmtree(os.path.join(sortie, j, s), ignore_errors=True)
            for p in ("train", "val", "val_fond"):
                os.makedirs(os.path.join(sortie, j, s, p), exist_ok=True)
    st = {"train": 0, "val": 0, "val_fond": 0, "train_fonds": 0, "classes": classes,
          "familles_test": {}}
    t0 = time.time()

    def ecrire(jeu, part, nom, image, lignes):
        cv2.imwrite(os.path.join(sortie, jeu, "images", part, nom + ".png"), image)
        with open(os.path.join(sortie, jeu, "labels", part, nom + ".txt"), "w") as f:
            f.write("\n".join(lignes) + ("\n" if lignes else ""))

    for k, n in enumerate(noms):
        ann = commun.lire_annotation(n, racine)
        image = commun.charger_image(n, racine)
        H, W = image.shape[:2]
        lignes = [exp_seg.ligne_yolo(classes.index(i["classe"]), i["polygones"], W, H)
                  for i in ann["instances"] if i["classe"] in classes and i["polygones"]]
        hc = hauteur_cachee(n, racine)
        h8 = rgbh.encoder_hauteur(hc["h"])
        part = "val" if exp_seg.en_validation(n, val) else "train"
        for v in variantes:
            ecrire(v, part, n, rgbh.composer(image, h8, v), lignes)
            if fonds_train:
                ecrire(v + "_fonds", part, n, rgbh.composer(image, h8, v), lignes)
        st[part] += 1
        masques = [commun.polygones_vers_masque(i["polygones"], (W, H)) for i in ann["instances"]]
        zone = None
        if part == "val" and versions_fond:
            zone = rgbh.zone_table(hc["carte"], hc["mesuree"], masques)
            for c in range(versions_fond):
                rng = np.random.default_rng([graine, int(n) if n.isdigit() else k, c])
                fam_c = (FAMILLES_TEST[c % len(FAMILLES_TEST)],)
                nv, fam = rgbh.table_texturee(np.asarray(ann["K"], float), W, H, ann["plan"],
                                              rng, fam_c)
                im2 = rgbh.changer_fond(image, zone, nv)
                st["familles_test"][fam] = st["familles_test"].get(fam, 0) + 1
                for v in variantes:
                    ecrire(v, "val_fond", "{}_{}".format(n, c), rgbh.composer(im2, h8, v), lignes)
                    if fonds_train:
                        ecrire(v + "_fonds", "val_fond", "{}_{}".format(n, c),
                               rgbh.composer(im2, h8, v), lignes)
                st["val_fond"] += 1
        if part == "train" and fonds_train:
            zone = rgbh.zone_table(hc["carte"], hc["mesuree"], masques)
            for c in range(fonds_train):
                rng = np.random.default_rng([graine + 1000, int(n) if n.isdigit() else k, c])
                nv, _ = rgbh.table_texturee(np.asarray(ann["K"], float), W, H, ann["plan"], rng,
                                            FAMILLES_ENTRAINEMENT)
                im2 = rgbh.changer_fond(image, zone, nv)
                for v in variantes:
                    ecrire(v + "_fonds", "train", "{}_f{}".format(n, c),
                           rgbh.composer(im2, h8, v), lignes)
                st["train_fonds"] += 1
        if verbeux and (k + 1) % 25 == 0:
            print("  {}/{} images ({:.0f} s)".format(k + 1, len(noms), time.time() - t0), flush=True)
    for j in jeux:
        canaux = 4 if j.startswith("rgbh") else 3
        _yaml(os.path.join(sortie, j, "data.yaml"), os.path.join(sortie, j), "val", classes, canaux)
        _yaml(os.path.join(sortie, j, "data_fond.yaml"), os.path.join(sortie, j), "val_fond",
              classes, canaux)
    st["jeux"] = {j: os.path.join(sortie, j) for j in jeux}
    st["duree_s"] = round(time.time() - t0, 1)
    return st


def main():
    ap = argparse.ArgumentParser(description="Export YOLO-seg RGB / RGB-H / RG-H")
    ap.add_argument("--dataset", default=commun.DATASET)
    ap.add_argument("--sortie", default=DONNEES)
    ap.add_argument("--variantes", default=",".join(VARIANTES))
    ap.add_argument("--versions-fond", type=int, default=2,
                    help="versions a fond change de chaque image de validation")
    ap.add_argument("--fonds", type=int, default=0,
                    help="copies a fond change de chaque image d'entrainement (jeux *_fonds)")
    a = ap.parse_args()
    s = exporter(a.dataset, a.sortie, tuple(a.variantes.split(",")), a.versions_fond, a.fonds)
    print("entrainement {train} (+{train_fonds} a fond change) ; validation {val} ; "
          "validation fond change {val_fond} {familles_test} ; {duree_s} s".format(**s))
    for j, d in s["jeux"].items():
        print("  {:12s} {}".format(j, d))


if __name__ == "__main__":
    main()
