#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revue et correction rapides des annotations (masques d'instance + classe).

    .venv/bin/python examples/perception/apprentissage/annoter.py
    .venv/bin/python examples/perception/apprentissage/annoter.py --tout --depuis 0040

La pre-annotation (pre_annoter.py) propose les masques ; ici on les VALIDE
d'une touche, ou on les corrige avec SAM 2.1 : un clic suffit pour ajouter
une piece, un clic de plus pour affiner un masque. Par defaut, seules les
images "a revoir" defilent (--tout : toutes).

SOURIS
  clic gauche sur une piece      la selectionner
  clic gauche ailleurs           AJOUTER une piece (SAM sur ce point)
  glisser (clic gauche tenu)     ajouter une piece par une BOITE (plus sur)
  Maj + clic gauche              point POSITIF pour la piece selectionnee
                                 (agrandir le masque jusque-la)
  Ctrl/Alt + clic gauche         point NEGATIF (retirer cette zone)
  clic droit                     SUPPRIMER la piece sous la souris
CLAVIER
  m          mode du clic gauche : nouvelle piece / point + / point -
             (si Maj ou Ctrl ne passent pas, sous macOS par exemple)
  1..9       classe de la piece selectionnee (et des suivantes)
  Tab        piece suivante ;  x : supprimer la selection ;  z : annuler
  Espace     VALIDER l'image, enregistrer, passer a la suivante
  v          basculer le statut valide / a revoir
  n / p      image suivante / precedente (enregistre si modifiee)
  s          enregistrer ;  h : masquer / montrer les masques
  q, Echap   enregistrer et quitter

Les annotations sont dans dataset/annotations/NNNN.json (voir commun.py).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun  # noqa: E402
from pre_annoter import dessiner_annotation  # noqa: E402

MODES = ("nouvelle", "point +", "point -")
AIDE = ["clic: choisir/ajouter  glisser: boite  Maj+clic: +  Ctrl/Alt+clic: -  "
        "clic droit: supprimer  m: mode",
        "1-9 classe  Tab suivante  x suppr  z annuler  ESPACE valider+suivante  "
        "v statut  n/p image  s sauver  h masques  q fin"]


class Annotateur:
    """L'etat de l'outil, sans affichage : testable sans fenetre."""

    def __init__(self, noms, racine=commun.DATASET, segmenteur=None, echelle=1.5):
        self.noms = list(noms)
        self.racine = racine
        self.types = list(commun.charger_types(racine))
        self.segmenteur = segmenteur
        self.echelle = echelle
        self.classe = self.types[0]
        self.mode = 0
        self.voir_masques = True
        self.i = -1
        self.message, self.t_message = "", 0.0
        self.depart = None
        self.aller(0)

    # -- images -----------------------------------------------------------
    def aller(self, i):
        if not self.noms:
            return
        if self.i >= 0 and self.modifie:
            self.enregistrer()
        self.i = int(np.clip(i, 0, len(self.noms) - 1))
        nom = self.noms[self.i]
        self.image = commun.charger_image(nom, self.racine)
        self.ann = commun.lire_annotation(nom, self.racine) or {
            "image": nom + ".png", "source": "", "statut": "a_revoir",
            "taille": [self.image.shape[1], self.image.shape[0]], "instances": []}
        self.taille = tuple(self.ann["taille"])
        self.masques = [commun.polygones_vers_masque(inst["polygones"], self.taille)
                        for inst in self.ann["instances"]]
        self.selection = None
        self.historique = []
        self.modifie = False

    @property
    def nom(self):
        return self.noms[self.i]

    def enregistrer(self):
        self.ann["modifie"] = time.strftime("%Y-%m-%d %H:%M")
        commun.ecrire_annotation(self.nom, self.ann, self.racine)
        self.modifie = False
        self.dire("enregistre : " + self.nom)

    def dire(self, texte):
        self.message, self.t_message = texte, time.time()

    def _memoriser(self):
        self.historique.append((copy.deepcopy(self.ann["instances"]),
                                [m.copy() for m in self.masques], self.selection))
        self.historique = self.historique[-30:]
        self.modifie = True

    def annuler(self):
        if self.historique:
            self.ann["instances"], self.masques, self.selection = self.historique.pop()
            self.modifie = True

    # -- instances --------------------------------------------------------
    def sous(self, x, y):
        """Indice de la piece sous le pixel (la plus petite si plusieurs)."""
        x = int(np.clip(x, 0, self.taille[0] - 1))
        y = int(np.clip(y, 0, self.taille[1] - 1))
        dedans = [(m.sum(), k) for k, m in enumerate(self.masques) if m[y, x]]
        return min(dedans)[1] if dedans else None

    def _sam(self, invites, reference=None):
        if self.segmenteur is None:
            self.dire("chargement de SAM...")
            self.segmenteur = commun.Segmenteur()
        return self.segmenteur.segmenter_un(self.image, invites.get("boite"),
                                            invites.get("points", []),
                                            invites.get("labels", []), reference)

    def _poser(self, k, masque, invites, origine=None):
        polys = commun.masque_vers_polygones(masque)
        if not polys:
            self.dire("masque vide : rien de change")
            return False
        inst = self.ann["instances"][k]
        inst["polygones"], inst["invites"] = polys, invites
        inst.pop("douteux", None)
        if origine:
            inst["origine"] = origine
        self.masques[k] = commun.polygones_vers_masque(polys, self.taille)
        return True

    def ajouter(self, invites):
        m, s = self._sam(invites)
        if m.sum() < 20:
            self.dire("SAM ne trouve rien ici")
            return
        self._memoriser()
        self.ann["instances"].append(commun.instance(self.classe, m, invites, "manuel", s))
        self.masques.append(commun.polygones_vers_masque(
            self.ann["instances"][-1]["polygones"], self.taille))
        self.selection = len(self.masques) - 1
        self.dire("piece ajoutee ({})".format(self.classe))

    def affiner(self, x, y, positif):
        k = self.selection
        if k is None:
            if positif:
                self.ajouter({"boite": None, "points": [[x, y]], "labels": [1]})
            else:
                self.dire("selectionnez d'abord une piece")
            return
        inv = copy.deepcopy(self.ann["instances"][k].get("invites") or {})
        inv.setdefault("boite", None)
        pts, lbl = list(inv.get("points", [])), list(inv.get("labels", []))
        if not pts and inv["boite"] is None:
            # masque sans invite (geometrie, reseau) : un point positif en son
            # centre pour garder l'objet
            ys, xs = np.nonzero(self.masques[k])
            if len(xs):
                pts.append([float(np.median(xs)), float(np.median(ys))])
                lbl.append(1)
        pts.append([float(x), float(y)])
        lbl.append(1 if positif else 0)
        inv["points"], inv["labels"] = pts, lbl
        m, _ = self._sam(inv)
        self._memoriser()
        if self._poser(k, m, inv):
            self.dire("masque affine ({} points)".format(len(pts)))

    def supprimer(self, k):
        if k is None:
            return
        self._memoriser()
        del self.ann["instances"][k]
        del self.masques[k]
        self.selection = None
        self.dire("piece supprimee")

    def changer_classe(self, c):
        if c >= len(self.types):
            return
        self.classe = self.types[c]
        if self.selection is not None:
            self._memoriser()
            self.ann["instances"][self.selection]["classe"] = self.classe
        self.dire("classe : " + self.classe)

    # -- evenements -------------------------------------------------------
    #: hauteur (pixels d'affichage) du bandeau d'etat au-dessus de l'image
    H_HAUT = 42

    def souris(self, evenement, x, y, drapeaux, *_):
        x, y = x / self.echelle, (y - self.H_HAUT) / self.echelle
        if evenement == cv2.EVENT_LBUTTONDOWN:
            self.depart = (x, y, drapeaux)
        elif evenement == cv2.EVENT_LBUTTONUP and self.depart is not None:
            x0, y0, dr = self.depart
            self.depart = None
            if np.hypot(x - x0, y - y0) * self.echelle > 8:
                self.ajouter({"boite": [min(x0, x), min(y0, y), max(x0, x), max(y0, y)],
                              "points": [], "labels": []})
                return
            maj = dr & cv2.EVENT_FLAG_SHIFTKEY
            neg = dr & (cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_ALTKEY)
            if maj or MODES[self.mode] == "point +":
                self.affiner(x, y, True)
            elif neg or MODES[self.mode] == "point -":
                self.affiner(x, y, False)
            else:
                k = self.sous(x, y)
                if k is not None:
                    self.selection = k
                    self.classe = self.ann["instances"][k]["classe"]
                else:
                    self.ajouter({"boite": None, "points": [[x, y]], "labels": [1]})
        elif evenement == cv2.EVENT_RBUTTONDOWN:
            self.supprimer(self.sous(x, y))

    def touche(self, k) -> bool:
        """Faux pour quitter."""
        if k in (ord("q"), 27):
            if self.modifie:
                self.enregistrer()
            return False
        if ord("1") <= k <= ord("9"):
            self.changer_classe(k - ord("1"))
        elif k == 9:                                   # Tab
            if self.masques:
                self.selection = 0 if self.selection is None else \
                    (self.selection + 1) % len(self.masques)
        elif k in (ord("x"), 8, 127):
            self.supprimer(self.selection)
        elif k == ord("z"):
            self.annuler()
        elif k == ord("m"):
            self.mode = (self.mode + 1) % len(MODES)
            self.dire("mode : " + MODES[self.mode])
        elif k == ord("h"):
            self.voir_masques = not self.voir_masques
        elif k == ord("s"):
            self.enregistrer()
        elif k == ord("v"):
            self.ann["statut"] = "a_revoir" if self.ann.get("statut") == "valide" else "valide"
            self.enregistrer()
        elif k == ord(" "):
            self.ann["statut"] = "valide"
            self.enregistrer()
            if self.i == len(self.noms) - 1:
                self.dire("derniere image validee")
            self.aller(self.i + 1)
        elif k == ord("n"):
            self.aller(self.i + 1)
        elif k == ord("p"):
            self.aller(self.i - 1)
        return True

    # -- rendu ------------------------------------------------------------
    def rendu(self) -> np.ndarray:
        if self.voir_masques:
            img = dessiner_annotation(self.image, self.ann, self.types, self.selection)
        else:
            img = self.image.copy()
        for inst_k, inst in enumerate(self.ann["instances"]):
            if inst_k == self.selection:
                for (px, py), l in zip(inst.get("invites", {}).get("points", []),
                                       inst.get("invites", {}).get("labels", [])):
                    cv2.circle(img, (int(px), int(py)), 4,
                               (0, 255, 0) if l else (0, 0, 255), -1, cv2.LINE_AA)
        img = cv2.resize(img, None, fx=self.echelle, fy=self.echelle,
                         interpolation=cv2.INTER_LINEAR)
        valide = self.ann.get("statut") == "valide"
        haut = "{}  [{}/{}]  {}  |  {} pieces  |  classe: {}  |  mode: {}{}".format(
            self.nom, self.i + 1, len(self.noms), "VALIDE" if valide else "a revoir",
            len(self.masques), self.classe, MODES[self.mode], "  *" if self.modifie else "")
        lignes = [haut, self.message if time.time() - self.t_message < 3 else ""]
        # bandeaux AU-DESSUS et AU-DESSOUS de l'image : rien n'en est cache
        dessus = _bandeau(img.shape[1], self.H_HAUT, lignes,
                          (60, 140, 60) if valide else (50, 50, 50))
        classes = "  ".join("{}={}".format(k + 1, t) for k, t in enumerate(self.types))
        dessous = _bandeau(img.shape[1], 60, [classes] + AIDE, (50, 50, 50))
        return np.vstack([dessus, img, dessous])


def _bandeau(largeur, hauteur, lignes, fond):
    img = np.empty((hauteur, largeur, 3), np.uint8)
    img[:] = fond
    for k, t in enumerate(lignes):
        cv2.putText(img, t, (6, 15 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Revue des annotations")
    ap.add_argument("--dataset", default=commun.DATASET)
    ap.add_argument("--tout", action="store_true", help="aussi les images deja validees")
    ap.add_argument("--depuis", default=None, help="commencer a cette image (NNNN)")
    ap.add_argument("--echelle", type=float, default=1.5, help="agrandissement de l'affichage")
    a = ap.parse_args()
    noms = [n for n in commun.noms_captures(a.dataset)
            if commun.lire_annotation(n, a.dataset) is not None]
    if not a.tout:
        noms = [n for n in noms if commun.lire_annotation(n, a.dataset).get("statut") != "valide"]
    if a.depuis:
        noms = [n for n in noms if n >= a.depuis]
    if not noms:
        sys.exit("rien a revoir (--tout pour tout revoir ; pre_annoter.py d'abord ?)")
    outil = Annotateur(noms, a.dataset, echelle=a.echelle)
    fenetre = "annotation"
    cv2.namedWindow(fenetre, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(fenetre, outil.souris)
    while True:
        cv2.imshow(fenetre, outil.rendu())
        k = cv2.waitKey(40)
        if k == -1:
            continue
        if not outil.touche(k & 0xFF):
            break
    cv2.destroyAllWindows()
    n_val = sum(1 for n in commun.noms_captures(a.dataset)
                if (commun.lire_annotation(n, a.dataset) or {}).get("statut") == "valide")
    print("{} images validees. Ensuite : entrainer.py".format(n_val))


if __name__ == "__main__":
    main()
