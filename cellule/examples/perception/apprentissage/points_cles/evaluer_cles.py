#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation du reseau de points cles sur la validation REELLE (les memes
images que runs/v1) : erreur des coins, puis, apres PnP, erreur de pose par
rapport a l'ajustement geometrique de reference (etiquettes.py), et temps.

    .venv/bin/python examples/perception/apprentissage/points_cles/evaluer_cles.py
    .venv/bin/python examples/perception/apprentissage/points_cles/evaluer_cles.py \
        --modele runs/cles_v1/weights/best.pt --synthetique synthetique/test_fonds

Sans modele (--sans-reseau) : seulement la partie geometrique -- PnP sur
les coins de REFERENCE (doit redonner la pose de reference : verifie la
chaine) et sensibilite au bruit des coins (1, 2, 3 px).

Mesures :
  - detection : rappel / precision des pieces (IoU des boites >= 0.5) ;
  - coins : erreur moyenne par piece (px), au mieux de la symetrie de 180
    deg (la piece est symetrique), sur les pieces dont la reference est
    gardee (etiquettes.py) ; mediane, moyenne, 90e centile ;
  - pose : distance des centres de face superieure (mm) et ecart de lacet
    (deg, modulo 180) entre la pose PnP et la pose de reference ; part du
    bon (type, face) ;
  - temps : reseau (ms / image), table + projection (ms / image), PnP et
    choix du candidat (ms / piece).
--synthetique DOSSIER (meta/ de synthese_images.py, ex. test_fonds : fonds
jamais vus) : erreur des coins et pose sur plan (camera et table exactes du
meta, sans profondeur) par rapport a la pose exacte.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

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
import etiquettes  # noqa: E402
import proposer as prop  # noqa: E402

RUNS = os.path.join(APPRENTISSAGE, "runs")


def iou_boites(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def apparier(dets, gts, seuil: float = 0.5):
    """Appariement glouton par confiance : [(i_det, j_gt)]."""
    paires, pris = [], set()
    for i in np.argsort([-d.conf for d in dets]):
        best, jb = seuil, -1
        for j, g in enumerate(gts):
            if j in pris:
                continue
            s = iou_boites(dets[i].boite, g["boite"])
            if s >= best:
                best, jb = s, j
        if jb >= 0:
            pris.add(jb)
            paires.append((int(i), jb))
    return paires


def stats(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mediane": round(float(np.median(x)), 3),
            "moyenne": round(float(x.mean()), 3), "p90": round(float(np.percentile(x, 90)), 3)}


def pose_de_reference(g) -> np.ndarray:
    return np.asarray(g["T_table"], float)


def bon_candidat(h, g) -> bool:
    return h.type == g["type"] and h.face == g["face"]


def evaluer_geometrie(etiq, noms, types, bruits=(0.0, 1.0, 2.0, 3.0), graine=0) -> dict:
    """PnP sur les coins de reference, avec bruit gaussien (px)."""
    rng = np.random.default_rng(graine)
    res = {}
    for s in bruits:
        dt, da, ok, ms = [], [], [], []
        for n in noms:
            e = etiq[n]
            gardes = [g for g in e["instances"] if g["garde"]]
            if not gardes:
                continue
            _, vue, table, proj = etiquettes.geometrie_capture(n)
            T_vt = vue.T_cp @ table.T_ct
            for g in gardes:
                uv = np.asarray(g["coins"], float) + rng.normal(0, s, (4, 2)) * (s > 0)
                t = time.perf_counter()
                h_mes = prop.hauteur_dans(uv, table, proj)
                hyp = prop.estimer_pose(uv, vue.K, T_vt, types, g["classe"], 1.0, h_mes,
                                        h_bouts=prop.hauteurs_bouts(uv, table, proj))
                ms.append((time.perf_counter() - t) * 1000)
                a, b = kc.ecart_pose(hyp.T_table, pose_de_reference(g))
                dt.append(a)
                da.append(b)
                ok.append(bon_candidat(hyp, g))
        res["bruit_{:.0f}px".format(s)] = {"position_mm": stats(dt), "lacet_deg": stats(da),
                                           "bon_type_face": round(float(np.mean(ok)), 4),
                                           "ms_par_piece": round(float(np.mean(ms)), 2)}
    return res


def evaluer_reseau(modele, etiq, noms, types, device=None, conf=0.25) -> dict:
    P = prop.ProposeurCles(modele, types=types, device=device, conf=conf)
    n_gt = n_det = n_app = 0
    err_coins, dt, da, ok, t_res, t_tab, t_pose = [], [], [], [], [], [], []
    methodes = {}
    par_type = {}
    for n in noms:
        e = etiq[n]
        t = time.perf_counter()
        _, vue, table, proj = etiquettes.geometrie_capture(n)
        t_tab.append(time.perf_counter() - t)
        dets = P.detecter(vue.image)
        t_res.append(P.durees["reseau"])
        gts = [g for g in e["instances"] if g.get("boite") is not None]
        n_gt += len(gts)
        n_det += len(dets)
        paires = apparier(dets, gts)
        n_app += len(paires)
        T_vt = vue.T_cp @ table.T_ct
        for i, j in paires:
            g = gts[j]
            if not g["garde"]:
                continue
            d = dets[i]
            err_coins.append(kc.erreur_coins(kc.canoniser(d.coins), g["coins"]))
            t = time.perf_counter()
            h_mes = prop.hauteur_dans(d.coins, table, proj)
            hyp = prop.estimer_pose(d.coins, vue.K, T_vt, types, d.classe, d.conf, h_mes,
                                    h_bouts=prop.hauteurs_bouts(d.coins, table, proj))
            t_pose.append(time.perf_counter() - t)
            methodes[hyp.methode] = methodes.get(hyp.methode, 0) + 1
            a, b = kc.ecart_pose(hyp.T_table, pose_de_reference(g))
            dt.append(a)
            da.append(b)
            ok.append(bon_candidat(hyp, g))
            par_type.setdefault("{} {}".format(g["type"], g["face"]), []).append(a)
    return {
        "detection": {"pieces_reference": n_gt, "detections": n_det, "appariees": n_app,
                      "rappel": round(n_app / max(n_gt, 1), 4),
                      "precision": round(n_app / max(n_det, 1), 4)},
        "coins_px": stats(err_coins),
        "pose": {"position_mm": stats(dt), "lacet_deg": stats(da),
                 "bon_type_face": round(float(np.mean(ok)) if ok else float("nan"), 4),
                 "methodes": methodes,
                 "position_mm_par_type_face": {k: stats(v) for k, v in sorted(par_type.items())}},
        "temps": {"reseau_ms_image": round(1000 * float(np.mean(t_res)), 1),
                  "table_projection_ms_image": round(1000 * float(np.mean(t_tab)), 1),
                  "pnp_ms_piece": round(1000 * float(np.mean(t_pose)), 2) if t_pose else None,
                  "device": P.device},
    }


def evaluer_synthetique(modele, dossier, types, device=None, n_max=300, conf=0.25) -> dict:
    """Coins et pose sur plan sur un jeu rendu (meta exact, sans profondeur)."""
    P = prop.ProposeurCles(modele, types=types, device=device, conf=conf)
    err, dt, da = [], [], []
    n_gt = n_app = 0
    for f in sorted(glob.glob(os.path.join(dossier, "meta", "*.json")))[:n_max]:
        with open(f, encoding="utf-8") as fm:
            meta = json.load(fm)
        im = cv2.imread(os.path.join(dossier, "images", meta["image"]))
        if im is None:
            continue
        H, W = im.shape[:2]
        K = np.asarray(meta["K"], float)
        T_ct = np.asarray(meta["T_camera_table"], float)
        gts = []
        for p in meta.get("pieces", []):
            uv = kc.canoniser(np.array(p["coins_dessus_image"], float),
                              np.array(p["coins_dessus_table"], float))
            gts.append({"boite": kc.boite_englobante(uv, W, H), "coins": uv, "p": p})
        dets = P.detecter(im)
        n_gt += len(gts)
        paires = apparier(dets, gts, 0.3)
        n_app += len(paires)
        for i, j in paires:
            g, d = gts[j], dets[i]
            err.append(kc.erreur_coins(kc.canoniser(d.coins), g["coins"]))
            if g["p"]["pose"] == "inclinee":
                continue
            hyp = prop.estimer_pose(d.coins, K, T_ct, types, d.classe, d.conf)
            ct = np.asarray(g["p"]["coins_dessus_table"], float)
            T_ref = np.eye(4)
            ax = ct[1] - ct[0] if np.linalg.norm(ct[1] - ct[0]) >= np.linalg.norm(ct[2] - ct[1]) \
                else ct[2] - ct[1]
            ax[2] = 0
            ax /= np.linalg.norm(ax) + 1e-12
            T_ref[:3, 0] = ax
            T_ref[:3, 1] = np.cross([0, 0, 1.0], ax)
            T_ref[:3, 3] = ct.mean(axis=0)
            a, b = kc.ecart_pose(hyp.T_table, T_ref)
            dt.append(a)
            da.append(b)
    return {"dossier": dossier, "pieces": n_gt, "rappel_boites_0.3": round(n_app / max(n_gt, 1), 4),
            "coins_px": stats(err), "pose_plan": {"position_mm": stats(dt), "lacet_deg": stats(da)}}


def main():
    import exporter as exp_seg
    ap = argparse.ArgumentParser(description="Evaluation des points cles (coins, pose PnP)")
    ap.add_argument("--modele", default=prop.POIDS)
    ap.add_argument("--sans-reseau", action="store_true")
    ap.add_argument("--synthetique", default=None, help="jeu rendu avec meta/ (ex. synthetique/test_fonds)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sortie", default=os.path.join(etiquettes.DONNEES, "evaluation.json"))
    a = ap.parse_args()
    types = commun.charger_types()
    etiq = etiquettes.etiquettes_reelles()
    noms = [n for n in sorted(etiq) if exp_seg.en_validation(n, 0.2)]
    rapport = {"images_val": len(noms),
               "pieces_val_avec_coins": sum(1 for n in noms for g in etiq[n]["instances"] if g["garde"])}
    print("geometrie (coins de reference + bruit) ...", flush=True)
    rapport["geometrie"] = evaluer_geometrie(etiq, noms, types)
    if not a.sans_reseau:
        modele = a.modele
        if not os.path.isabs(modele) and not os.path.exists(modele):
            modele = os.path.join(APPRENTISSAGE, modele)
        print("reseau", modele, "...", flush=True)
        rapport["modele"] = modele
        rapport["reseau"] = evaluer_reseau(modele, etiq, noms, types, a.device, a.conf)
        if a.synthetique:
            d = a.synthetique
            if not os.path.isabs(d) and not os.path.isdir(d):
                d = os.path.join(APPRENTISSAGE, d)
            rapport["synthetique"] = evaluer_synthetique(modele, d, types, a.device, conf=a.conf)
    os.makedirs(os.path.dirname(a.sortie), exist_ok=True)
    with open(a.sortie, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=1, ensure_ascii=False)
    print(json.dumps(rapport, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
