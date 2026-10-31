#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrainement du reseau de POINTS CLES : YOLO11-pose, 4 coins de la face
superieure de chaque piece (kpt_shape [4, 3]), classes = types de pieces.

    # export (etiquettes.py) puis entrainement ; poids -> modeles/pieces-cles.pt
    .venv/bin/python examples/perception/apprentissage/points_cles/entrainer_cles.py \
        --synthetiques synthetique/entrainement --nom cles_v1
    # essai rapide sur CPU (verifie que tout tourne ; n'installe rien)
    .venv/bin/python examples/perception/apprentissage/points_cles/entrainer_cles.py \
        --essai

POIDS DE DEPART. Pas de yolo11s-pose.pt pre-entraine dans modeles/ (pas
de telechargement) : le modele est construit depuis yolo11s-pose.yaml et
recoit le tronc et le cou de yolo11s-seg.pt (meme architecture, poids
COCO) ; seule la tete de pose part de zero.

SYMETRIE (voir coins.py). Une face rectangulaire est la meme apres une
rotation de 180 deg ; l'ordre canonique des coins est defini dans l'image
mais saute de 180 deg quand le grand cote passe par la verticale. Pres de
ce saut, deux images presque identiques ont des cibles permutees : une
perte ordinaire pousserait le reseau vers la MOYENNE des deux (le centre).
On remplace donc, dans CE processus seulement (pas de modification
d'ultralytics sur disque, et les poids sauves restent des PoseModel
ordinaires) :
  - la perte des coins (v8PoseLoss.calculate_keypoints_loss) par son
    MINIMUM sur les permutations de symetrie, choisi instance par instance ;
  - la similarite OKS de la validation (kpt_iou) par son MAXIMUM sur ces
    permutations : la mAP de pose et le choix du meilleur poids ne
    penalisent pas une reponse symetrique.

Reglages : ceux de la segmentation (entrainer.py) quand ils ont un sens ;
fliplr 0.5 avec flip_idx [1, 0, 3, 2] (data.yaml) ; pas de copy_paste
(propre a la segmentation) ; pose 12 (gain par defaut des coins).
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
import coins as kc  # noqa: E402

RUNS = os.path.join(APPRENTISSAGE, "runs")
POIDS = os.path.join(commun.MODELES, "pieces-cles.pt")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------------
# Perte et similarite invariantes a la symetrie
# ---------------------------------------------------------------------------

def _permutations(n_kpt: int):
    import torch
    if n_kpt == 4:
        return [torch.tensor(p) for p in kc.SYMETRIES]
    return [torch.arange(n_kpt)]


def meilleure_permutation(pred, gt, sigmas, area):
    """Pour chaque instance, la cible (gt permutee) la plus proche de la
    prediction, au sens de la perte OKS d'ultralytics. pred, gt : (N, K, 2+),
    area : (N, 1). Rend gt reordonne (N, K, 3)."""
    import torch
    pertes, cibles = [], []
    for p in _permutations(gt.shape[1]):
        g = gt[:, p.to(gt.device)]
        m = g[..., 2] != 0
        d = (pred[..., 0] - g[..., 0]).pow(2) + (pred[..., 1] - g[..., 1]).pow(2)
        e = d / ((2 * sigmas).pow(2) * (area + 1e-9) * 2)
        pertes.append(((1 - torch.exp(-e)) * m).sum(1) / (m.sum(1) + 1e-9))
        cibles.append(g)
    k = torch.stack(pertes, 0).argmin(0)                       # (N,)
    C = torch.stack(cibles, 0)                                 # (P, N, K, 3)
    return C[k, torch.arange(gt.shape[0], device=gt.device)]


def installer_symetrie() -> None:
    """Remplace (dans ce processus) la perte des coins et l'OKS de
    validation d'ultralytics par leurs versions invariantes (voir l'en-tete).
    Idempotent."""
    import torch
    from ultralytics.utils import loss as ul
    from ultralytics.utils.metrics import kpt_iou as kpt_iou_orig
    from ultralytics.utils.ops import xyxy2xywh
    import ultralytics.models.yolo.pose.val as pv

    if getattr(ul.v8PoseLoss, "_symetrique", False):
        return

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, stride_tensor,
                                 target_bboxes, pred_kpts):
        kpts_loss = torch.zeros((), device=pred_kpts.device)
        kpts_obj_loss = torch.zeros((), device=pred_kpts.device)
        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = keypoints[masks.nonzero(as_tuple=True)[0], target_gt_idx[masks]]
            gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            with torch.no_grad():
                gt_kpt = meilleure_permutation(pred_kpt.detach(), gt_kpt,
                                               self.keypoint_loss.sigmas, area)
            kpt_mask = gt_kpt[..., 2] != 0
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss

    def kpt_iou_sym(kpt1, kpt2, area, sigma, eps=1e-7):
        sims = [kpt_iou_orig(kpt1[:, p.to(kpt1.device)], kpt2, area, sigma, eps)
                for p in _permutations(kpt1.shape[1])]
        return torch.stack(sims, 0).amax(0)

    ul.v8PoseLoss.calculate_keypoints_loss = calculate_keypoints_loss
    ul.v8PoseLoss._symetrique = True
    pv.kpt_iou = kpt_iou_sym


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------

def modele_depart(base: str = "yolo11s-pose.yaml", tronc: str = "yolo11s-seg.pt"):
    """YOLO-pose construit depuis son yaml, tronc et cou repris des poids de
    segmentation (voir l'en-tete)."""
    from ultralytics import YOLO
    m = YOLO(base)
    chemin = tronc if os.path.exists(tronc) else os.path.join(commun.MODELES, tronc)
    if os.path.exists(chemin):
        m.load(chemin)
    return m


def entrainer(data: str, nom: str = "cles", epochs: int = 80, imgsz: int = 864, batch: int = 8,
              device=None, patience: int = 30, workers: int = 2, base="yolo11s-pose.yaml",
              tronc="yolo11s-seg.pt", installer: bool = True, fraction: float = 1.0) -> dict:
    installer_symetrie()
    debut = time.time()
    device = device or commun.device_auto()
    m = modele_depart(base, tronc)
    m.train(data=data, project=RUNS, name=nom, exist_ok=True, epochs=epochs, imgsz=imgsz,
            batch=batch, device=device, patience=patience, workers=workers, seed=0,
            cos_lr=True, close_mosaic=min(10, max(epochs // 3, 1)), degrees=10.0, flipud=0.0,
            fliplr=0.5, mosaic=1.0, scale=0.3, translate=0.1, hsv_h=0.015, hsv_s=0.5,
            hsv_v=0.4, plots=True, verbose=False, fraction=fraction)
    dossier = os.path.join(RUNS, nom)
    meilleur = os.path.join(dossier, "weights", "best.pt")
    if not os.path.exists(meilleur):
        meilleur = os.path.join(dossier, "weights", "last.pt")
    from ultralytics import YOLO
    v = YOLO(meilleur).val(data=data, split="val", imgsz=imgsz, device=device, batch=batch,
                           plots=False, verbose=False, project=RUNS, name=nom + "_val",
                           exist_ok=True)
    rapport = {"date": time.strftime("%Y-%m-%d %H:%M"), "epochs": epochs, "imgsz": imgsz,
               "device": str(device), "data": data,
               "boite_mAP50": round(float(v.box.map50), 4),
               "boite_mAP50_95": round(float(v.box.map), 4),
               "pose_mAP50": round(float(v.pose.map50), 4),
               "pose_mAP50_95": round(float(v.pose.map), 4),
               "duree_min": round((time.time() - debut) / 60, 1), "poids": meilleur}
    if installer:
        if os.path.exists(POIDS):
            shutil.copy2(POIDS, POIDS[:-3] + ".precedent.pt")
        shutil.copy2(meilleur, POIDS)
        rapport["installe"] = POIDS
    with open(os.path.join(dossier, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=1, ensure_ascii=False)
    return rapport


def main():
    import etiquettes
    ap = argparse.ArgumentParser(description="Entrainement YOLO11-pose (coins des pieces)")
    ap.add_argument("--synthetiques", default=None,
                    help="dossier synthese_images.py (avec meta/), ex. synthetique/entrainement")
    ap.add_argument("--n-synth", type=int, default=0)
    ap.add_argument("--nom", default="cles_v1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=864)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--sans-installer", action="store_true")
    ap.add_argument("--essai", action="store_true",
                    help="1 epoque sur CPU, 5 %% des images, imgsz 320 : verifie la chaine "
                    "(n'installe rien)")
    a = ap.parse_args()
    synth = a.synthetiques
    if synth and not os.path.isabs(synth) and not os.path.isdir(synth):
        synth = os.path.join(APPRENTISSAGE, synth)
    st = etiquettes.exporter(synthetiques=synth, n_synth=a.n_synth)
    print("export : {train} reelles + {rendues} rendues en entrainement, {val} en validation "
          "({coins_val} pieces avec coins)".format(**st), flush=True)
    if a.essai:
        r = entrainer(st["data"], "cles_essai", epochs=1, imgsz=320, batch=4, device="cpu",
                      workers=0, installer=False, fraction=0.05)
    else:
        r = entrainer(st["data"], a.nom, a.epochs, a.imgsz, a.batch, a.device, a.patience,
                      installer=not a.sans_installer)
    print("\nRAPPORT")
    for k, v in r.items():
        print("  {:20s} {}".format(k, v))


if __name__ == "__main__":
    main()
