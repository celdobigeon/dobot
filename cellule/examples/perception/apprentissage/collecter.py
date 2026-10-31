#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session de collecte d'images pour l'apprentissage, avec le serveur de camera.

    .venv/bin/python examples/perception/apprentissage/collecter.py
    .venv/bin/python examples/perception/apprentissage/collecter.py --auto 3
    .venv/bin/python examples/perception/apprentissage/collecter.py --importer examples/perception/captures/env2_reglee.npz

Le serveur doit tourner (sudo .venv/bin/python examples/perception/
serveur_camera.py, dans un autre terminal). Une fenetre montre ce que voit
la camera ; DANS cette fenetre :

    Entree ou Espace   capture (mediane de 15 images : la meme qualite que
                       la detection)
    a                  mode automatique marche / arret : une capture toutes
                       les N secondes, seulement si la scene a CHANGE depuis
                       la derniere et qu'elle est IMMOBILE (pas de main en
                       mouvement dans le champ)
    f                  capture d'un FOND (aucune piece sur la table) : sert au
                       copier-coller
    c                  conseil suivant
    q ou Echap         fin

Chaque capture est gardee ENTIERE (dataset/brut/NNNN.npz : profondeur,
infrarouge, couleur, geometrie) avec son image (dataset/images/NNNN.png).
La couleur BRUTE est utilisee si le serveur fournit sa geometrie ; sinon,
repli sur la couleur recalee (un message le dit : relancez le serveur a jour).
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

CONSEILS = [
    "Deplacez les pieces : toute la table, pres et loin de la camera, aux bords.",
    "Tournez-les : toutes les orientations, pas seulement face a la camera.",
    "Changez la face d'appui : a plat, sur la tranche, debout.",
    "Mettez des pieces EN CONTACT : cote a cote, bout a bout, en T, en tas.",
    "Melangez les types : epaisses et fines ensemble, et une seule piece aussi.",
    "Variez l'eclairage : lampe allumee/eteinte, rideaux, ombres portees.",
    "Changez de fond : autre nappe, table nue, feuille blanche, tissu sombre.",
    "Ajoutez des INTRUS : regle, crayon, gomme, main... (ce ne sont pas des pieces).",
    "Quelques scenes vides ou presque (touche f pour un fond sans piece).",
    "Peu et beaucoup de pieces : de 1 a 10 par image.",
]


#: m : seule la ZONE DE TRAVAIL compte pour juger si la scene a change ;
#: au-dela, le decor (television allumee, passants, meubles) bougerait sans
#: cesse et la scene ne paraitrait jamais immobile
ZONE_MAX = 1.2
#: un pixel a "change" si son niveau de gris varie de plus de SEUIL_PIXEL
SEUIL_PIXEL = 25
#: proportion de pixels de la zone changes : sous IMMOBILE entre deux vues
#: consecutives, la scene est immobile ; au-dessus de CHANGEE par rapport a
#: la derniere capture, elle a change (une piece deplacee en fait ~1 %)
IMMOBILE = 0.010
CHANGEE = 0.015


def petite(cap) -> tuple:
    """
    Vignette grise floutee de la ZONE DE TRAVAIL, et son masque. On prend
    l'image recalee sur la profondeur (couleur, sinon infrarouge) : chaque
    pixel y a sa distance, ce qui permet d'ignorer le decor lointain.
    """
    img = cap.get("couleur")
    if img is None:
        img = cap["infrarouge"]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    D = cap["profondeur"]
    zone = ((D > 0) & (D < ZONE_MAX)).astype(np.uint8)
    g = cv2.GaussianBlur(cv2.resize(g, (212, 120)), (5, 5), 0).astype(np.float32)
    zone = cv2.resize(zone, (212, 120), interpolation=cv2.INTER_NEAREST) > 0
    return g, zone


def ecart(a, b) -> float:
    """Proportion des pixels de la zone de travail qui ont change."""
    if a is None or b is None:
        return 1.0
    zone = a[1] & b[1]
    if not zone.any():
        return 1.0
    return float(np.mean(np.abs(a[0] - b[0])[zone] > SEUIL_PIXEL))


def importer(chemins, racine):
    for c in chemins:
        cap = perception.charger_capture(c)
        nom = commun.enregistrer_capture(cap, racine)
        _, source = commun.image_de(cap)
        print("{} -> brut/{}.npz ({})".format(c, nom, source))


def capturer_et_sauver(camera, racine, images, fond=False):
    cap = camera.capturer(images)
    if fond:
        image, _ = commun.image_de(cap)
        d = commun.dossier(racine, "fonds")
        nom = "fond_{}".format(time.strftime("%Y%m%d_%H%M%S"))
        cv2.imwrite(os.path.join(d, nom + ".png"), image)
        np.savez_compressed(os.path.join(d, nom + ".npz"), **cap)
        return nom, cap
    return commun.enregistrer_capture(cap, racine), cap


def bandeau(img, lignes, haut=True):
    h = 18 * len(lignes) + 8
    y0 = 0 if haut else img.shape[0] - h
    zone = img[y0:y0 + h]
    zone[:] = (zone * 0.35).astype(np.uint8)
    for k, t in enumerate(lignes):
        cv2.putText(img, t, (8, y0 + 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Collecte d'images pour l'apprentissage")
    ap.add_argument("--dataset", default=commun.DATASET)
    ap.add_argument("--auto", type=float, default=0.0, metavar="SECONDES",
                    help="demarrer en mode automatique, une capture toutes les N s")
    ap.add_argument("--images", type=int, default=15, help="images par capture (mediane)")
    ap.add_argument("--importer", nargs="+", metavar="NPZ",
                    help="ajouter des captures existantes au jeu de donnees, sans camera")
    ap.add_argument("--sans-fenetre", action="store_true",
                    help="sans affichage : --nombre captures, une toutes les --auto s")
    ap.add_argument("--nombre", type=int, default=0, help="arret apres ce nombre de captures")
    ap.add_argument("--types", default=None,
                    help="types de pieces si le jeu de donnees est neuf : 'epaisse=17,19,68;fine=8,17,68'")
    a = ap.parse_args()

    commun.charger_types(a.dataset, commun.lire_types_option(a.types))
    if a.importer:
        importer(a.importer, a.dataset)
        return

    sys.path.insert(0, commun.PERCEPTION)
    import camera
    if not camera.serveur_present():
        sys.exit("pas de serveur de camera. Dans un autre terminal :\n    sudo .venv/bin/"
                 "python examples/perception/serveur_camera.py")
    n0 = len(commun.noms_captures(a.dataset))

    if a.sans_fenetre:
        n = 0
        while n < max(a.nombre, 1):
            nom, cap = capturer_et_sauver(camera, a.dataset, a.images)
            n += 1
            print("capture {} ({})".format(nom, commun.image_de(cap)[1]), flush=True)
            if n < a.nombre:
                time.sleep(max(a.auto, 0.0))
        return

    intervalle = a.auto if a.auto > 0 else 3.0
    auto = a.auto > 0
    fenetre = "collecte (Entree: capture, a: auto, f: fond, q: fin)"
    cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
    precedente = derniere_sauvee = None
    immobile_depuis = time.time()
    t_derniere = 0.0
    i_conseil, t_conseil = 0, time.time()
    collectees, message, t_message = 0, "", 0.0
    avertissement = ""
    while True:
        try:
            apercu = camera.capturer(1, delai=10.0)
        except camera.ErreurCamera as exc:
            print("camera :", exc)
            time.sleep(1.0)
            continue
        image, source = commun.image_de(apercu)
        if source != "couleur_brute":
            avertissement = ("couleur {} : relancez le serveur a jour pour la couleur "
                             "brute".format("recalee" if source == "couleur" else "ABSENTE"))
        vig = petite(apercu)
        if ecart(vig, precedente) > IMMOBILE:
            immobile_depuis = time.time()
        precedente = vig
        immobile = time.time() - immobile_depuis > 1.0
        variation = ecart(vig, derniere_sauvee)
        change = variation > CHANGEE

        if auto and immobile and change and time.time() - t_derniere >= intervalle:
            touche = 13
        else:
            touche = -1
        vue = image.copy()
        etat = "AUTO toutes les {:.0f} s".format(intervalle) if auto else "manuel"
        lignes = ["{} images collectees (dont {} cette session) - mode {}".format(
            n0 + collectees, collectees, etat)]
        if auto:
            lignes.append("scene {} / {} ({:.1f} % de la table modifie)".format(
                "immobile" if immobile else "EN MOUVEMENT",
                "changee : capture imminente" if change and immobile else
                ("changee" if change else "deja capturee : deplacez des pieces"),
                100 * min(variation, 1.0)))
        if avertissement:
            lignes.append(avertissement)
        if time.time() - t_message < 2.5:
            lignes.append(message)
        bandeau(vue, lignes)
        if time.time() - t_conseil > 12:
            i_conseil, t_conseil = (i_conseil + 1) % len(CONSEILS), time.time()
        bandeau(vue, ["Conseil : " + CONSEILS[i_conseil],
                      "Entree/Espace capture   a auto   f fond   c conseil   q fin"], haut=False)
        cv2.imshow(fenetre, vue)
        k = cv2.waitKey(30) & 0xFF
        if k != 255:
            touche = k
        if touche in (ord("q"), 27):
            break
        if touche == ord("a"):
            auto = not auto
        elif touche == ord("c"):
            i_conseil, t_conseil = (i_conseil + 1) % len(CONSEILS), time.time()
        elif touche in (13, 10, ord(" "), ord("f")):
            fond = touche == ord("f")
            try:
                nom, cap = capturer_et_sauver(camera, a.dataset, a.images, fond)
            except camera.ErreurCamera as exc:
                message, t_message = "echec de la capture : {}".format(exc), time.time()
                continue
            t_derniere = time.time()
            derniere_sauvee = petite(cap)
            if not fond:
                collectees += 1
            message = ("fond enregistre : " if fond else "capture enregistree : ") + nom
            t_message = time.time()
            print(message, "({:.0f} % de profondeur mesuree)".format(
                100 * (cap["profondeur"] > 0).mean()), flush=True)
            # flash : on voit que la capture est prise
            cv2.imshow(fenetre, cv2.addWeighted(vue, 0.5, np.full_like(vue, 255), 0.5, 0))
            cv2.waitKey(80)
            if a.nombre and collectees >= a.nombre:
                break
    cv2.destroyAllWindows()
    print("{} images collectees dans {}. Ensuite : pre_annoter.py".format(
        collectees, os.path.join(a.dataset, "brut")))


if __name__ == "__main__":
    main()
