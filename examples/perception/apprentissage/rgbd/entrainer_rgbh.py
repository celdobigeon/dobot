#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrainement de YOLO11-seg sur entree RGB-H (couleur + hauteur au-dessus de
la table), et des temoins RGB, dans les memes conditions.

    # apres exporter_rgbh.py ; poids -> runs/rgbd_<jeu>/weights/best.pt
    .venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgbh
    .venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgb
    .venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgh
    # le meilleur jeu a hauteur peut etre installe dans modeles/pieces-seg-rgbd.pt
    .venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgbh --installer
    # essai sur CPU (1 epoque, peu d'images) : verifie la chaine
    .venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgbh --essai

QUATRE CANAUX DANS ULTRALYTICS 8.4 (verifie dans le code installe) :
  - data.yaml "channels: 4" : images lues avec IMREAD_UNCHANGED (PNG BGRA),
    modele construit avec ch=4 ;
  - poids de depart yolo11s-seg.pt : tout est repris SAUF la premiere
    convolution (forme differente). On la reconstruit ici (rappel
    on_pretrain_routine_end, modele et EMA) : poids RGB pre-entraines pour
    les trois premiers canaux, leur MOYENNE pour la hauteur (usage courant
    pour ajouter un canal a un reseau pre-entraine) ;
  - mosaique, perspective, retournement, letterbox : acceptent N canaux
    (le bourrage vaut 114 sur TOUS les canaux, donc 38 mm de "hauteur" dans
    les bandes de bourrage : sans objet, le reseau l'ignore, et c'est le
    meme a la validation) ;
  - RandomHSV ne traite que 3 canaux (sinon rien) : remplace ici, pour
    rgbh, par la meme transformation appliquee aux seuls canaux couleur ;
    pour rgh (H, G, R), par un gain de luminosite et de couleur sur G et R
    seulement (la hauteur n'est jamais modifiee) ;
  - copy_paste (mode flip) : marche sur N canaux (la hauteur suit la piece).
Le remplacement de RandomHSV doit valoir aussi dans les processus du
chargeur de donnees (macOS : spawn, le module __main__ est re-importe) :
il est donc installe a l'IMPORT de ce module, selon la variable
d'environnement RGBD_VARIANTE.

Reglages : ceux d'entrainer.py (imgsz 864, mask_ratio 2, fliplr 0.5,
degrees 10, scale 0.3...), copy_paste 0.3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
POIDS = os.path.join(commun.MODELES, "pieces-seg-rgbd.pt")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------------
# Augmentation de couleur qui respecte le canal de hauteur
# ---------------------------------------------------------------------------

def hsv_couleur(img, hgain: float, sgain: float, vgain: float, variante: str, rng=None):
    """Augmentation de couleur EN PLACE sans toucher la hauteur.
    rgbh (B, G, R, H) : RandomHSV d'ultralytics sur les 3 premiers canaux ;
    rgh (H, G, R) : gain de luminosite commun (vgain) et gains par canal
    (sgain / 2) sur G et R. Rend img."""
    import numpy as np
    import cv2
    rng = rng or np.random
    if variante == "rgbh" and img.ndim == 3 and img.shape[-1] == 4:
        r = rng.uniform(-1, 1, 3) * [hgain, sgain, vgain]
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(np.uint8)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(np.uint8)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(np.uint8)
        lut_sat[0] = 0
        bgr = np.ascontiguousarray(img[..., :3])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lut = np.stack((lut_hue, lut_sat, lut_val), -1).reshape(1, 256, 3)
        img[..., :3] = cv2.cvtColor(cv2.LUT(hsv, lut), cv2.COLOR_HSV2BGR)
    elif variante == "rgh" and img.ndim == 3 and img.shape[-1] == 3:
        v = 1 + rng.uniform(-1, 1) * vgain
        g = 1 + rng.uniform(-1, 1, 2) * sgain / 2
        for k, gk in zip((1, 2), g):
            img[..., k] = np.clip(img[..., k].astype(np.float32) * v * gk, 0, 255).astype(np.uint8)
    return img


def installer_hsv(variante: str = None) -> None:
    """Remplace RandomHSV.apply_image (ce processus) pour la variante
    (variable RGBD_VARIANTE par defaut) ; rgb ou rien : inchange."""
    variante = variante or os.environ.get("RGBD_VARIANTE", "")
    if variante not in ("rgbh", "rgh"):
        return
    try:
        from ultralytics.data import augment
    except Exception:                            # pragma: no cover
        return
    if getattr(augment.RandomHSV, "_variante", None) == variante:
        return

    def apply_image(self, labels, params=None):
        if self.hgain or self.sgain or self.vgain:
            hsv_couleur(labels["img"], self.hgain, self.sgain, self.vgain, variante)
        return labels

    augment.RandomHSV.apply_image = apply_image
    augment.RandomHSV._variante = variante


installer_hsv()          # a l'import : vaut aussi dans les processus du chargeur


# ---------------------------------------------------------------------------
# Premiere convolution a 4 canaux
# ---------------------------------------------------------------------------

def premiere_conv_rgbh(poids_rgb: str):
    """Rappel ultralytics : copie la premiere convolution pre-entrainee (3
    canaux) dans celle du modele a 4 canaux, 4e canal = moyenne des 3."""
    import torch

    def rappel(trainer):
        ck = torch.load(poids_rgb, map_location="cpu", weights_only=False)
        w3 = ck["model"].model[0].conv.weight.detach().float()
        modeles = [trainer.model]
        if getattr(trainer, "ema", None) is not None:
            modeles.append(trainer.ema.ema)
        for m in modeles:
            m = m.module if hasattr(m, "module") else m
            conv = m.model[0].conv
            if conv.weight.shape[1] != 4 or conv.weight.shape[0] != w3.shape[0]:
                continue
            with torch.no_grad():
                w = conv.weight
                w[:, :3] = w3.to(w.device, w.dtype)
                w[:, 3:] = w3.mean(1, keepdim=True).to(w.device, w.dtype)
        print("rgbd : premiere convolution 4 canaux initialisee depuis", os.path.basename(poids_rgb))
    return rappel


def entrainer(jeu: str = "rgbh", nom: str = None, epochs: int = 100, imgsz: int = 864,
              batch: int = 8, device=None, patience: int = 30, workers: int = 2,
              modele: str = "yolo11s-seg.pt", installer: bool = False,
              fraction: float = 1.0, donnees: str = DONNEES) -> dict:
    from ultralytics import YOLO
    variante = jeu.split("_")[0]
    os.environ["RGBD_VARIANTE"] = variante
    installer_hsv(variante)
    debut = time.time()
    device = device or commun.device_auto()
    nom = nom or "rgbd_" + jeu
    data = os.path.join(donnees, jeu, "data.yaml")
    if not os.path.exists(data):
        raise SystemExit("jeu introuvable : {} (exporter_rgbh.py)".format(data))
    poids = modele if os.path.exists(modele) else os.path.join(commun.MODELES, modele)
    m = YOLO(poids)
    if variante == "rgbh":
        m.add_callback("on_pretrain_routine_end", premiere_conv_rgbh(poids))
    m.train(data=data, project=RUNS, name=nom, exist_ok=True, epochs=epochs, imgsz=imgsz,
            batch=batch, device=device, patience=patience, workers=workers, seed=0, cos_lr=True,
            close_mosaic=min(10, max(epochs // 3, 1)), degrees=10.0, flipud=0.0, fliplr=0.5,
            mosaic=1.0, scale=0.3, translate=0.1, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
            copy_paste=0.3, copy_paste_mode="flip", mask_ratio=2, overlap_mask=True,
            plots=True, verbose=False, fraction=fraction)
    dossier = os.path.join(RUNS, nom)
    meilleur = os.path.join(dossier, "weights", "best.pt")
    if not os.path.exists(meilleur):
        meilleur = os.path.join(dossier, "weights", "last.pt")
    rapport = {"date": time.strftime("%Y-%m-%d %H:%M"), "jeu": jeu, "epochs": epochs,
               "imgsz": imgsz, "device": str(device), "poids": meilleur,
               "duree_min": round((time.time() - debut) / 60, 1)}
    for cle, yaml in (("val", "data.yaml"), ("val_fond", "data_fond.yaml")):
        v = YOLO(meilleur).val(data=os.path.join(donnees, jeu, yaml), split="val", imgsz=imgsz,
                               device=device, batch=batch, plots=False, verbose=False,
                               project=RUNS, name=nom + "_" + cle, exist_ok=True)
        rapport[cle] = {"masque_mAP50": round(float(v.seg.map50), 4),
                        "masque_mAP50_95": round(float(v.seg.map), 4),
                        "boite_mAP50": round(float(v.box.map50), 4)}
    if installer:
        if os.path.exists(POIDS):
            shutil.copy2(POIDS, POIDS[:-3] + ".precedent.pt")
        shutil.copy2(meilleur, POIDS)
        rapport["installe"] = POIDS
    with open(os.path.join(dossier, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=1, ensure_ascii=False)
    return rapport


def main():
    ap = argparse.ArgumentParser(description="YOLO11-seg sur RGB-H (et temoins RGB)")
    ap.add_argument("--jeu", default="rgbh",
                    help="rgb, rgbh, rgh, ou *_fonds (dossier de rgbd/donnees/)")
    ap.add_argument("--nom", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=864)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--installer", action="store_true",
                    help="copier le meilleur poids dans modeles/pieces-seg-rgbd.pt")
    ap.add_argument("--essai", action="store_true",
                    help="1 epoque sur CPU, 10 %% des images, imgsz 320")
    a = ap.parse_args()
    if a.essai:
        r = entrainer(a.jeu, "rgbd_essai_" + a.jeu, epochs=1, imgsz=320, batch=4, device="cpu",
                      workers=2, fraction=0.1)
    else:
        r = entrainer(a.jeu, a.nom, a.epochs, a.imgsz, a.batch, a.device, a.patience,
                      installer=a.installer)
    print("\nRAPPORT")
    for k, v in r.items():
        print("  {:12s} {}".format(k, v))


if __name__ == "__main__":
    main()
