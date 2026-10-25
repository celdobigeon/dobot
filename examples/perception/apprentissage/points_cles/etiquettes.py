#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etiquettes de POINTS CLES (4 coins de la face superieure de chaque piece)
et export au format YOLO-pose (ultralytics).

    .venv/bin/python examples/perception/apprentissage/points_cles/etiquettes.py
    .venv/bin/python examples/perception/apprentissage/points_cles/etiquettes.py \
        --synthetiques synthetique/entrainement --n-synth 2000

REEL : il n'y a pas de coins annotes a la main. On les DERIVE des masques
valides (annotations/*.json) et de la geometrie 3D : pour chaque masque,
perception.ajuster_masque cherche la piece de taille connue, posee sur la
table, dont la silhouette explique le masque (la profondeur donne la table
et la hauteur) ; les coins de sa face superieure, projetes dans l'image
couleur, sont l'etiquette. Les cas douteux sont ecartes : IoU masque /
silhouette < 0.85 (piece inclinee, mal expliquee), type different de la
classe annotee, instance marquee douteuse. Une instance ecartee reste
dans l'image avec sa boite et sa classe mais SANS coins (visibilite 0 :
ultralytics n'en tire aucune perte de coins) : sinon le reseau
apprendrait qu'une piece non etiquetee est du fond.

SYNTHETIQUE (synthese_images.py, dossier meta/) : coins exacts
(coins_dessus_image), classe, pose (a plat, sur la tranche, debout,
inclinee) ; la boite vient du polygone YOLO-seg de la meme instance.

Le calcul reel est mis en cache (donnees/etiquettes_reelles.json) : il sert
aussi de REFERENCE de pose a l'evaluation (evaluer.py).

Sortie (donnees/, ignore par git) :
    donnees/etiquettes_reelles.json
    donnees/yolo/images/{train,val}/   liens symboliques vers les images
    donnees/yolo/labels/{train,val}/   "classe cx cy w h x0 y0 v0 ... x3 y3 v3"
                                       (normalises ; v : 0 absent, 1 cache,
                                       2 visible ; ordre canonique, coins.py)
    donnees/yolo/data.yaml             kpt_shape [4, 3], flip_idx [1, 0, 3, 2]

Validation : les MEMES images reelles que runs/v1 (exporter.en_validation,
empreinte crc32 du nom, fraction 0.2).
"""

from __future__ import annotations

import argparse
import glob
import json
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
import coins as kc  # noqa: E402

DONNEES = os.path.join(ICI, "donnees")
CACHE_REEL = os.path.join(DONNEES, "etiquettes_reelles.json")
IOU_MIN = 0.85


# ---------------------------------------------------------------------------
# Reel : coins derives de l'ajustement geometrique
# ---------------------------------------------------------------------------

def geometrie_capture(nom: str, racine: str = commun.DATASET):
    """(capture, vue couleur brute, table, projection de la profondeur)."""
    cap = perception.charger_capture(os.path.join(racine, "brut", nom + ".npz"))
    vue = perception.vue_couleur(cap)
    K = np.asarray(cap["K"], float)
    table = perception.analyser_table(cap["profondeur"], K)
    proj = perception.projeter_profondeur(cap["profondeur"], K, vue)
    return cap, vue, table, proj


def coins_ajustement(T_table: np.ndarray, dims_mm, table, vue) -> np.ndarray:
    """Coins (ordre du contour, arete 0 -> 1 le long du grand cote) de la
    face superieure d'une piece ajustee, dans l'image de la vue : 4 x 2."""
    L, l, _ = (float(v) / 1000.0 for v in dims_mm)
    T = vue.T_cp @ table.T_ct @ T_table
    q = kc.coins_face(L, l) @ T[:3, :3].T + T[:3, 3]
    return perception.projeter(q, vue)


def visibilites(uv: np.ndarray, masque: np.ndarray, autres: Optional[np.ndarray]) -> List[int]:
    """2 : coin dans l'image et sur sa piece (ou pas sous une autre) ;
    1 : dans l'image mais sous une autre piece ; 0 : hors de l'image."""
    H, W = masque.shape
    m = cv2.dilate(masque.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    v = []
    for x, y in uv:
        if not (0 <= x <= W - 1 and 0 <= y <= H - 1):
            v.append(0)
            continue
        xi, yi = int(round(x)), int(round(y))
        if m[yi, xi] or autres is None or not autres[yi, xi]:
            v.append(2)
        else:
            v.append(1)
    return v


def etiqueter_image(nom: str, racine: str = commun.DATASET, types=None,
                    iou_min: float = IOU_MIN) -> dict:
    """Etiquettes d'une image reelle validee (voir l'en-tete)."""
    types = types or commun.charger_types(racine)
    ann = commun.lire_annotation(nom, racine)
    _, vue, table, proj = geometrie_capture(nom, racine)
    taille = tuple(ann["taille"])
    masques = [commun.polygones_vers_masque(i["polygones"], taille) for i in ann["instances"]]
    sortie = []
    for k, inst in enumerate(ann["instances"]):
        m = masques[k]
        autres = (np.any([x for j, x in enumerate(masques) if j != k], axis=0)
                  if len(masques) > 1 else None)
        e = {"classe": inst["classe"], "boite": commun.boite_de(m) if m.any() else None,
             "garde": False, "raison": ""}
        if not m.any():
            e["raison"] = "masque vide"
            sortie.append(e)
            continue
        if inst.get("douteux"):
            e["raison"] = "douteux"
        aj = perception.ajuster_masque(m, table, vue, proj, types, classe=inst["classe"],
                                       ignorer=autres)
        if aj is None:
            e["raison"] = e["raison"] or "pas d'ajustement"
            sortie.append(e)
            continue
        uv = coins_ajustement(aj.T_table, aj.dims, table, vue)
        e.update({"type": aj.type, "face": aj.face, "dims": list(aj.dims), "base": aj.base,
                  "iou": round(float(aj.iou), 4), "T_table": aj.T_table.round(6).tolist(),
                  "h_mesure": None if not np.isfinite(aj.h_mesure) else round(aj.h_mesure, 2)})
        uv_c = kc.canoniser(uv)
        e["coins"] = uv_c.round(2).tolist()
        e["visibilite"] = visibilites(uv_c, m, autres)
        if not e["raison"]:
            if aj.iou < iou_min:
                e["raison"] = "iou {:.2f}".format(aj.iou)
            elif aj.type != inst["classe"]:
                e["raison"] = "type {} != {}".format(aj.type, inst["classe"])
        e["garde"] = not e["raison"]
        sortie.append(e)
    return {"taille": list(taille), "K": vue.K.tolist(), "instances": sortie}


def etiquettes_reelles(racine: str = commun.DATASET, cache: str = CACHE_REEL,
                       refaire: bool = False, verbeux: bool = True) -> Dict[str, dict]:
    """Toutes les images VALIDEES ; le cache est complete, pas recalcule."""
    donnees = {}
    if os.path.exists(cache) and not refaire:
        with open(cache, encoding="utf-8") as f:
            donnees = json.load(f)
    types = commun.charger_types(racine)
    noms = [n for n in commun.noms_captures(racine)
            if (commun.lire_annotation(n, racine) or {}).get("statut") == "valide"]
    a_faire = [n for n in noms if n not in donnees]
    t0 = time.time()
    for k, n in enumerate(a_faire):
        donnees[n] = etiqueter_image(n, racine, types)
        if verbeux and (k + 1) % 20 == 0:
            print("  {}/{} images ({:.0f} s)".format(k + 1, len(a_faire), time.time() - t0),
                  flush=True)
    if a_faire:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache + ".tmp", "w", encoding="utf-8") as f:
            json.dump(donnees, f)
        os.replace(cache + ".tmp", cache)
    return {n: donnees[n] for n in noms if n in donnees}


# ---------------------------------------------------------------------------
# Format YOLO-pose
# ---------------------------------------------------------------------------

def ligne_pose(classe: int, boite, coins_uv, vis, W: int, H: int) -> str:
    """Une instance -> "classe cx cy w h x0 y0 v0 ... x3 y3 v3" (normalise).
    Un coin de visibilite 0 est ecrit (0, 0, 0), comme le veut ultralytics."""
    x0, y0, x1, y1 = boite
    vals = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
    for (x, y), v in zip(coins_uv, vis):
        if v:
            vals += [min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0), float(v)]
        else:
            vals += [0.0, 0.0, 0.0]
    return "{} ".format(classe) + " ".join("{:.6f}".format(x) for x in vals)


def lire_ligne_pose(ligne: str, W: int, H: int):
    """Inverse de ligne_pose : (classe, boite xyxy px, coins 4 x 2 px, vis)."""
    v = [float(x) for x in ligne.split()]
    c, (cx, cy, w, h) = int(v[0]), v[1:5]
    k = np.array(v[5:], float).reshape(4, 3)
    boite = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
    return c, boite, k[:, :2] * [W, H], k[:, 2].astype(int)


def lignes_reelles(etiq: dict, classes: List[str]) -> List[str]:
    W, H = etiq["taille"]
    lignes = []
    for e in etiq["instances"]:
        if e.get("boite") is None or e["classe"] not in classes:
            continue
        if e["garde"]:
            uv, vis = e["coins"], e["visibilite"]
        else:
            uv, vis = [[0, 0]] * 4, [0, 0, 0, 0]
        lignes.append(ligne_pose(classes.index(e["classe"]), e["boite"], uv, vis, W, H))
    return lignes


def _boites_seg(chemin_label: str, W: int, H: int) -> List[List[float]]:
    boites = []
    with open(chemin_label) as f:
        for ligne in f:
            v = ligne.split()
            if len(v) < 7:
                continue
            p = np.array(v[1:], float).reshape(-1, 2) * [W, H]
            boites.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    return boites


def lignes_synthetiques(meta: dict, chemin_label: str, classes: List[str],
                        W: int, H: int) -> Optional[List[str]]:
    """Etiquettes d'une image rendue (meta de synthese_images.py) ; None si
    les instances du meta et du fichier YOLO-seg ne se correspondent pas."""
    pieces = meta.get("pieces", [])
    boites = _boites_seg(chemin_label, W, H)
    if len(boites) != len(pieces):
        return None
    lignes = []
    for p, b in zip(pieces, boites):
        if p["classe"] not in classes or "coins_dessus_image" not in p:
            continue
        uv = kc.canoniser(np.array(p["coins_dessus_image"], float),
                          np.array(p["coins_dessus_table"], float))
        vis = [2 if (0 <= x <= W - 1 and 0 <= y <= H - 1) else 0 for x, y in uv]
        lignes.append(ligne_pose(classes.index(p["classe"]), b, uv, vis, W, H))
    return lignes


def _lien(source: str, cible: str) -> None:
    if os.path.lexists(cible):
        os.remove(cible)
    try:
        os.symlink(os.path.abspath(source), cible)
    except OSError:
        shutil.copy2(source, cible)


def exporter(racine: str = commun.DATASET, sortie: Optional[str] = None,
             synthetiques: Optional[str] = None, n_synth: int = 0, val: float = 0.2,
             graine: int = 0, refaire: bool = False, verbeux: bool = True) -> dict:
    """Export YOLO-pose (voir l'en-tete). Rend des statistiques."""
    import exporter as exp_seg            # meme decoupage que runs/v1
    sortie = sortie or os.path.join(DONNEES, "yolo")
    classes = list(commun.charger_types(racine))
    etiq = etiquettes_reelles(racine, refaire=refaire, verbeux=verbeux)
    for sous in ("images", "labels"):
        shutil.rmtree(os.path.join(sortie, sous), ignore_errors=True)
    dirs = {p: (commun.dossier(sortie, os.path.join("images", p)),
                commun.dossier(sortie, os.path.join("labels", p))) for p in ("train", "val")}
    st = {"train": 0, "val": 0, "instances_train": 0, "instances_val": 0,
          "coins_train": 0, "coins_val": 0, "ecartees": {}, "rendues": 0, "classes": classes}
    for n, e in sorted(etiq.items()):
        part = "val" if exp_seg.en_validation(n, val) else "train"
        _lien(os.path.join(racine, "images", n + ".png"), os.path.join(dirs[part][0], n + ".png"))
        lignes = lignes_reelles(e, classes)
        with open(os.path.join(dirs[part][1], n + ".txt"), "w") as f:
            f.write("\n".join(lignes) + ("\n" if lignes else ""))
        st[part] += 1
        st["instances_" + part] += len(lignes)
        st["coins_" + part] += sum(1 for i in e["instances"] if i["garde"])
        for i in e["instances"]:
            if not i["garde"]:
                r = i["raison"].split(" ")[0]
                st["ecartees"][r] = st["ecartees"].get(r, 0) + 1
    if synthetiques:
        images = sorted(glob.glob(os.path.join(synthetiques, "images", "*.jpg"))
                        + glob.glob(os.path.join(synthetiques, "images", "*.png")))
        if 0 < n_synth < len(images):
            rng = np.random.default_rng(graine)
            images = [images[i] for i in sorted(rng.choice(len(images), n_synth, replace=False))]
        for f in images:
            base = os.path.splitext(os.path.basename(f))[0]
            meta_f = os.path.join(synthetiques, "meta", base + ".json")
            lab_f = os.path.join(synthetiques, "labels", base + ".txt")
            if not (os.path.exists(meta_f) and os.path.exists(lab_f)):
                continue
            with open(meta_f, encoding="utf-8") as fm:
                meta = json.load(fm)
            im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            H, W = im.shape[:2]
            lignes = lignes_synthetiques(meta, lab_f, classes, W, H)
            if lignes is None:
                continue
            _lien(f, os.path.join(dirs["train"][0], "sy_" + os.path.basename(f)))
            with open(os.path.join(dirs["train"][1], "sy_" + base + ".txt"), "w") as fl:
                fl.write("\n".join(lignes) + ("\n" if lignes else ""))
            st["rendues"] += 1
            st["instances_train"] += len(lignes)
    with open(os.path.join(sortie, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# genere par points_cles/etiquettes.py : ne pas modifier a la main\n")
        f.write("path: {}\ntrain: images/train\nval: images/val\n".format(os.path.abspath(sortie)))
        f.write("kpt_shape: [4, 3]\nflip_idx: [{}]\n".format(", ".join(map(str, kc.FLIP_IDX))))
        f.write("names:\n")
        for k, c in enumerate(classes):
            f.write("  {}: {}\n".format(k, c))
    st["data"] = os.path.join(sortie, "data.yaml")
    return st


def main():
    ap = argparse.ArgumentParser(description="Etiquettes de coins et export YOLO-pose")
    ap.add_argument("--dataset", default=commun.DATASET)
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--synthetiques", default=None,
                    help="dossier synthese_images.py (avec meta/) ajoute a l'entrainement")
    ap.add_argument("--n-synth", type=int, default=0, help="0 : toutes")
    ap.add_argument("--refaire", action="store_true", help="recalculer le cache reel")
    a = ap.parse_args()
    synth = a.synthetiques
    if synth and not os.path.isabs(synth) and not os.path.isdir(synth):
        synth = os.path.join(APPRENTISSAGE, synth)
    s = exporter(a.dataset, a.sortie, synth, a.n_synth, refaire=a.refaire)
    print("entrainement : {train} reelles ({coins_train} pieces avec coins) + {rendues} rendues, "
          "{instances_train} pieces ; validation : {val} images, {instances_val} pieces "
          "({coins_val} avec coins) ; ecartees {ecartees}\n-> {data}".format(**s))


if __name__ == "__main__":
    main()
