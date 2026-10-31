#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-annotation automatique : des masques d'instance PROPOSES pour chaque
capture du jeu de donnees, a revoir ensuite avec annoter.py.

    .venv/bin/python examples/perception/apprentissage/pre_annoter.py
    .venv/bin/python examples/perception/apprentissage/pre_annoter.py --refaire 0003 0007
    .venv/bin/python examples/perception/apprentissage/pre_annoter.py --sans-reseau

Pourquoi : dessiner des masques a la main prend des minutes par image ; les
corriger prend des secondes. Les propositions viennent de deux sources :

  1. la GEOMETRIE (vision_modele si present, sinon vision) : des pieces de
     taille connue trouvees dans la profondeur. Chacune est projetee dans
     l'image (perception.silhouette : les 8 coins du parallelepipede, via
     K_couleur et T_couleur_profondeur), ce qui donne une boite-invite pour
     SAM 2.1 ; des trois masques que SAM propose, on garde celui qui
     ressemble le plus a la silhouette projetee. Si SAM s'en eloigne trop,
     la silhouette elle-meme sert de masque (marque "douteux").
  2. le RESEAU deja entraine (modeles/pieces-seg.pt), s'il existe : ses
     masques s'ajoutent. C'est la boucle classique de l'apprentissage
     "humain dans la boucle" : chaque entrainement rend la pre-annotation
     suivante meilleure, et la revue plus rapide.

Les doublons (meme piece proposee deux fois) sont fusionnes. La CLASSE (le
type) vient de la geometrie : chaque masque final est ajuste par
perception.ajuster_masque, qui dit quel type et quelle face l'expliquent.

Une annotation deja VALIDEE n'est jamais ecrasee (sauf --forcer).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun  # noqa: E402
from commun import perception  # noqa: E402


def propositions_geometriques(capture, vue, table, types):
    """[(silhouette (masque), classe, face, score)] des pieces trouvees par
    la geometrie seule."""
    module = perception._detecteur_geometrique()
    scene = perception.scene_de(capture)
    try:
        res = module.detecter(scene, types)
    except Exception as exc:                      # une capture sans table...
        print("  geometrie en echec : {}".format(exc))
        return []
    sortie = []
    for p in res.pieces:
        if p.type not in types:
            continue
        dims = perception.forme(types[p.type], p.face)
        # la pose de la piece est dans le repere de LA table du detecteur ;
        # on la reporte dans le repere camera puis dans notre table
        T_t = table.T_tc @ p.T_camera
        poly = perception.silhouette(T_t, dims, table, vue)
        m = np.zeros(vue.image.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, poly, 1)
        if m.sum() < 30:
            continue
        score = getattr(p, "score", None)
        if score is None:
            score = max(0.0, 1.0 - p.ecart / 10.0)
        sortie.append((m > 0, p.type, p.face, float(score)))
    return sortie


def propositions_reseau(reseau, vue, types, device):
    res = reseau.predict(vue.image, conf=0.25, device=device, retina_masks=True,
                         verbose=False)[0]
    if res.masks is None:
        return []
    m = res.masks.data.cpu().numpy() > 0.5
    cls = [reseau.names[int(c)] for c in res.boxes.cls.cpu().numpy()]
    conf = res.boxes.conf.cpu().numpy()
    return [(m[i], cls[i], "", float(conf[i])) for i in range(len(m)) if cls[i] in types]


def pre_annoter(nom, racine, types, segmenteur, reseau=None, device="cpu"):
    capture = perception.charger_capture(os.path.join(racine, "brut", nom + ".npz"))
    vue = perception.vue_couleur(capture, infrarouge=True)
    image = vue.image
    cv2.imwrite(os.path.join(commun.dossier(racine, "images"), nom + ".png"), image)
    K = np.asarray(capture["K"], float)
    table = perception.analyser_table(capture["profondeur"], K)
    proj = perception.projeter_profondeur(capture["profondeur"], K, vue)

    candidats = []        # (masque, classe, score, invites, origine, douteux)
    for sil, classe, face, score in propositions_geometriques(capture, vue, table, types):
        boite = commun.boite_de(sil, marge=6)
        m, s_sam = segmenteur.segmenter_un(image, boite=boite, reference=sil)
        accord = commun.iou(m, sil)
        douteux = accord < 0.5
        if douteux:
            m = sil
        candidats.append((m, classe, score * (0.5 + 0.5 * accord), {
            "boite": boite, "points": [], "labels": []}, "geometrie", douteux))
    if reseau is not None:
        for m, classe, face, score in propositions_reseau(reseau, vue, types, device):
            boite = commun.boite_de(m, marge=4)
            candidats.append((m, classe, score, {"boite": boite, "points": [],
                                                 "labels": []}, "reseau", False))

    # fusion des doublons : le mieux note d'abord
    candidats.sort(key=lambda c: -c[2])
    gardes = []
    for c in candidats:
        if all(commun.iou(c[0], g[0]) < 0.5 for g in gardes):
            gardes.append(c)

    instances = []
    tous = [g[0] for g in gardes]
    for i, (m, classe, score, invites, origine, douteux) in enumerate(gardes):
        autres = np.any([t for j, t in enumerate(tous) if j != i], axis=0) if len(tous) > 1 else None
        aj = perception.ajuster_masque(m, table, vue, proj, types, classe=classe,
                                       ignorer=autres)
        face = ""
        if aj is not None and aj.iou > 0.6:
            classe, face = aj.type, aj.face      # la geometrie tranche
        inst = commun.instance(classe, m, invites, origine, score, face=face,
                               iou_geometrie=round(aj.iou, 3) if aj else 0.0)
        if douteux:
            inst["douteux"] = True
        if inst["polygones"]:
            instances.append(inst)

    ann = {"image": nom + ".png", "source": vue.source,
           "taille": [image.shape[1], image.shape[0]], "statut": "a_revoir",
           "plan": np.round(table.plan_dans(vue), 6).tolist(),
           "K": np.asarray(vue.K).tolist(), "instances": instances,
           "pre_annotation": time.strftime("%Y-%m-%d %H:%M")}
    commun.ecrire_annotation(nom, ann, racine)
    cv2.imwrite(os.path.join(commun.dossier(racine, "apercus"), nom + ".png"),
                dessiner_annotation(image, ann, list(types)))
    return ann


def dessiner_annotation(image, ann, noms, selection=None, alpha=0.45):
    """Image + masques colores par classe, numerotes."""
    img = image.copy()
    taille = (image.shape[1], image.shape[0])
    for i, inst in enumerate(ann["instances"]):
        coul = np.array(perception.couleur_type(inst["classe"], noms), float)
        m = commun.polygones_vers_masque(inst["polygones"], taille)
        a = alpha + (0.15 if i == selection else 0.0)
        img[m] = ((1 - a) * img[m] + a * coul).astype(np.uint8)
    for i, inst in enumerate(ann["instances"]):
        coul = perception.couleur_type(inst["classe"], noms)
        epais = 2 if i == selection else 1
        for p in inst["polygones"]:
            pts = np.round(np.asarray(p)).astype(np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 255) if i == selection else coul,
                          epais, cv2.LINE_AA)
        if inst["polygones"]:
            c = np.asarray(inst["polygones"][0]).mean(axis=0).astype(int)
            texte = "{} {}{}".format(i + 1, inst["classe"], " ?" if inst.get("douteux") else "")
            cv2.putText(img, texte, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(img, texte, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("noms", nargs="*", help="captures a traiter (defaut : toutes celles sans annotation)")
    ap.add_argument("--dataset", default=commun.DATASET)
    ap.add_argument("--refaire", action="store_true",
                    help="refaire aussi les annotations a revoir deja presentes")
    ap.add_argument("--forcer", action="store_true", help="ecraser meme les annotations validees")
    ap.add_argument("--sans-reseau", action="store_true",
                    help="ne pas utiliser modeles/pieces-seg.pt meme s'il existe")
    ap.add_argument("--sam", default=commun.SAM_DEFAUT)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    types = commun.charger_types(a.dataset)
    noms = a.noms or commun.noms_captures(a.dataset)
    if not noms:
        sys.exit("aucune capture dans {}/brut : lancez d'abord collecter.py".format(a.dataset))
    device = a.device or commun.device_auto()
    segmenteur = commun.Segmenteur(a.sam, device)
    reseau = None
    if not a.sans_reseau and os.path.exists(perception.MODELE):
        from ultralytics import YOLO
        reseau = YOLO(perception.MODELE)
        print("reseau utilise en plus de la geometrie :", perception.MODELE)
    faits = 0
    for nom in noms:
        ann = commun.lire_annotation(nom, a.dataset)
        if ann is not None:
            if ann.get("statut") == "valide" and not a.forcer:
                continue
            if not (a.refaire or a.noms or a.forcer):
                continue
        t = time.time()
        ann = pre_annoter(nom, a.dataset, types, segmenteur, reseau, device)
        faits += 1
        print("{} : {} pieces proposees ({}), {:.1f} s".format(
            nom, len(ann["instances"]), ann["source"], time.time() - t), flush=True)
    print("{} image(s) pre-annotee(s). Revue : .venv/bin/python examples/perception/"
          "apprentissage/annoter.py".format(faits))


if __name__ == "__main__":
    main()
