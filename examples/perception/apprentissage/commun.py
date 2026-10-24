# -*- coding: utf-8 -*-
"""
Ce que partagent les scripts de la chaine d'apprentissage : emplacement du
jeu de donnees, types de pieces, format des annotations, masques et
polygones, et SAM 2.1 (segmentation sur invite).

Le jeu de donnees (dossier ignore par git) :

    dataset/
      config.json        {"types": {"epaisse": [17, 19, 68], ...}} : les classes
      brut/NNNN.npz      la capture entiere (profondeur, infrarouge, couleur...)
      images/NNNN.png    l'image annotee : couleur brute si la capture porte sa
                         geometrie (K_couleur...), sinon couleur recalee
      annotations/NNNN.json   masques (polygones), classes, statut
      apercus/NNNN.png   l'image et ses masques, pour un coup d'oeil
      fonds/*.png        images SANS piece (nappes, table vide) : fonds pour
                         le copier-coller
      yolo/              l'export pour l'entrainement (regenere a chaque fois)

Une annotation (JSON, lisible et corrigeable a la main) :

    {"image": "NNNN.png", "source": "couleur_brute", "taille": [W, H],
     "statut": "a_revoir" | "valide",
     "plan": [nx, ny, nz, d],      plan de la table dans le repere de l'image
     "K": 3x3,                     intrinseques de l'image (pour le plan)
     "instances": [{"classe": "fine", "polygones": [[[x, y], ...]],
                    "invites": {"boite": [x0, y0, x1, y1] | null,
                                "points": [[x, y], ...], "labels": [1, 0, ...]},
                    "origine": "geometrie" | "reseau" | "manuel",
                    "score": 0.9, "face": "a plat"}]}

Les invites sont gardees pour pouvoir AFFINER un masque (ajouter un point
positif ou negatif et relancer SAM) sans repartir de zero.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = os.path.dirname(ICI)
if PERCEPTION not in sys.path:
    sys.path.insert(0, PERCEPTION)
import perception  # noqa: E402

DATASET = os.path.join(ICI, "dataset")
MODELES = os.path.join(PERCEPTION, "modeles")
SAM_DEFAUT = os.path.join(MODELES, "sam2.1_b.pt")
STATUTS = ("a_revoir", "valide")


# ---------------------------------------------------------------------------
# Dossiers et configuration
# ---------------------------------------------------------------------------

def dossier(racine: str, nom: str) -> str:
    d = os.path.join(racine, nom)
    os.makedirs(d, exist_ok=True)
    return d


def charger_types(racine: str = DATASET,
                  defaut: Optional[Dict[str, Sequence[float]]] = None) -> Dict[str, Tuple[float, ...]]:
    """Les types de pieces (= classes, dans cet ordre) du jeu de donnees.
    Ecrits une fois dans config.json : l'ordre des classes ne doit plus
    changer ensuite (les etiquettes YOLO sont des numeros)."""
    chemin = os.path.join(racine, "config.json")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            return {k: tuple(float(x) for x in v) for k, v in json.load(f)["types"].items()}
    types = {k: tuple(float(x) for x in v) for k, v in (defaut or perception.TYPES).items()}
    os.makedirs(racine, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"types": types}, f, indent=1, ensure_ascii=False)
    return types


def lire_types_option(texte: Optional[str]) -> Optional[Dict[str, Tuple[float, ...]]]:
    """'epaisse=17,19,68;fine=8,17,68' -> dictionnaire (option --types)."""
    if not texte:
        return None
    types = {}
    for morceau in texte.split(";"):
        nom, dims = morceau.split("=")
        types[nom.strip()] = tuple(float(x) for x in dims.split(","))
    return types


def noms_captures(racine: str = DATASET) -> List[str]:
    d = os.path.join(racine, "brut")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))


def prochain_nom(racine: str = DATASET) -> str:
    noms = [int(n) for n in noms_captures(racine) if n.isdigit()]
    return "{:04d}".format(max(noms, default=-1) + 1)


def image_de(capture: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """L'image a annoter et sa source (voir perception.vue_couleur)."""
    vue = perception.vue_couleur(capture, infrarouge=True)
    if vue is None:
        raise ValueError("capture sans image couleur ni infrarouge")
    return vue.image, vue.source


def enregistrer_capture(capture: Dict[str, np.ndarray], racine: str = DATASET,
                        nom: Optional[str] = None) -> str:
    """brut/NNNN.npz (tout) + images/NNNN.png ; rend le nom."""
    nom = nom or prochain_nom(racine)
    np.savez_compressed(os.path.join(dossier(racine, "brut"), nom + ".npz"), **capture)
    image, _ = image_de(capture)
    cv2.imwrite(os.path.join(dossier(racine, "images"), nom + ".png"), image)
    return nom


def charger_image(nom: str, racine: str = DATASET) -> np.ndarray:
    chemin = os.path.join(racine, "images", nom + ".png")
    if not os.path.exists(chemin):
        cap = perception.charger_capture(os.path.join(racine, "brut", nom + ".npz"))
        image, _ = image_de(cap)
        cv2.imwrite(os.path.join(dossier(racine, "images"), nom + ".png"), image)
        return image
    return cv2.imread(chemin, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def chemin_annotation(nom: str, racine: str = DATASET) -> str:
    return os.path.join(racine, "annotations", nom + ".json")


def lire_annotation(nom: str, racine: str = DATASET) -> Optional[dict]:
    chemin = chemin_annotation(nom, racine)
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def ecrire_annotation(nom: str, ann: dict, racine: str = DATASET) -> None:
    """Ecriture atomique : un fichier a moitie ecrit (Ctrl+C) ne remplace
    jamais une annotation valide."""
    chemin = chemin_annotation(nom, racine)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=1, ensure_ascii=False)
    os.replace(tmp, chemin)


def _liste(x):
    return np.asarray(x).tolist() if x is not None else None


def instance(classe: str, masque: np.ndarray, invites: Optional[dict] = None,
             origine: str = "manuel", score: float = 1.0, **autres) -> dict:
    inst = {"classe": classe, "polygones": masque_vers_polygones(masque),
            "invites": invites or {"boite": None, "points": [], "labels": []},
            "origine": origine, "score": round(float(score), 4)}
    inst.update(autres)
    return inst


# ---------------------------------------------------------------------------
# Masques et polygones
# ---------------------------------------------------------------------------

def masque_vers_polygones(masque: np.ndarray, aire_min: float = 25.0,
                          epsilon: float = 0.7) -> List[List[List[float]]]:
    """Contours exterieurs d'un masque binaire, simplifies a `epsilon`
    pixel pres (Douglas-Peucker) ; les ilots minuscules sont oublies."""
    m = (np.asarray(masque) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < aire_min:
            continue
        c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(c) >= 3:
            polys.append(c.astype(float).tolist())
    return polys


def polygones_vers_masque(polygones, taille: Tuple[int, int]) -> np.ndarray:
    """taille = (W, H)."""
    m = np.zeros((taille[1], taille[0]), np.uint8)
    for p in polygones:
        cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32)], 1)
    return m > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def boite_de(masque: np.ndarray, marge: int = 0) -> List[float]:
    ys, xs = np.nonzero(masque)
    H, W = masque.shape
    return [float(max(xs.min() - marge, 0)), float(max(ys.min() - marge, 0)),
            float(min(xs.max() + marge, W - 1)), float(min(ys.max() + marge, H - 1))]


# ---------------------------------------------------------------------------
# SAM 2.1
# ---------------------------------------------------------------------------

def device_auto() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class Segmenteur:
    """
    SAM 2.1 (ultralytics) avec les traits de l'image GARDES : l'encodeur
    (lent, ~0.3 s) ne tourne qu'une fois par image ; chaque invite ensuite
    coute quelques millisecondes, ce qui rend l'outil d'annotation fluide.
    """

    def __init__(self, poids: str = SAM_DEFAUT, device: Optional[str] = None):
        from ultralytics import SAM
        if not os.path.exists(poids):
            raise FileNotFoundError(
                "poids SAM introuvables : {} (.venv/bin/python examples/perception/"
                "verifier_installation.py --telecharger)".format(poids))
        self.sam = SAM(poids)
        self.device = device or device_auto()
        self._cle = None

    def _preparer(self, image: np.ndarray) -> None:
        cle = (image.shape, int(image[::17, ::13].sum()), id(image))
        if self.sam.predictor is None:
            # cree le predicteur (une invite factice)
            self.sam(image, bboxes=[[0, 0, 8, 8]], device=self.device, verbose=False)
        if cle != self._cle:
            self.sam.predictor.set_image(image)
            self._cle = cle

    def segmenter(self, image: np.ndarray, boites=None, points=None, labels=None,
                  multimasque: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Masques (N x H x W, bool) et scores. Une invite par objet :
        `boites` (N x 4, xyxy) et/ou `points` (N x P x 2) avec `labels`
        (N x P, 1 = sur l'objet, 0 = hors de l'objet). `multimasque` : trois
        masques par objet (partie, objet, ensemble), a departager ensuite.
        """
        self._preparer(image)
        p = self.sam.predictor
        kw = {}
        if boites is not None and len(boites):
            kw["bboxes"] = np.asarray(boites, float).reshape(-1, 4)
        if points is not None and len(points):
            kw["points"] = np.asarray(points, float)
            kw["labels"] = np.asarray(labels, int)
        if not kw:
            return np.zeros((0,) + image.shape[:2], bool), np.zeros(0)
        masques, boites_sortie = p.inference_features(
            p.features, image.shape[:2], multimask_output=multimasque, **kw)
        if masques is None:
            return np.zeros((0,) + image.shape[:2], bool), np.zeros(0)
        return masques.cpu().numpy().astype(bool), boites_sortie[:, 4].cpu().numpy()

    def segmenter_un(self, image: np.ndarray, boite=None, points=(), labels=(),
                     reference: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Un objet : le meilleur des trois masques proposes -- le plus
        proche de `reference` s'il y en a une, sinon le mieux note."""
        pts = [list(map(float, q)) for q in points]
        lbl = [int(x) for x in labels]
        multi = len(pts) <= 1          # invite ambigue : trois propositions
        m, s = self.segmenter(image, [boite] if boite is not None else None,
                              [pts] if pts else None, [lbl] if lbl else None,
                              multimasque=multi)
        if len(m) == 0:
            return np.zeros(image.shape[:2], bool), 0.0
        if reference is not None:
            k = int(np.argmax([iou(x, reference) + 0.1 * sc for x, sc in zip(m, s)]))
        else:
            k = int(np.argmax(s))
        return m[k], float(s[k])
