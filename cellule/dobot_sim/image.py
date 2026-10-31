# -*- coding: utf-8 -*-
"""
Le profil d'une image, et le programme qui le trace.

Une image n'a pas de traits : elle a des pixels. On en extrait des
CONTOURS -- la frontiere entre ce qui est encre et ce qui ne l'est pas --
puis on les simplifie en polylignes, que `dobot_sim.trace` transforme en
programme comme s'ils venaient d'un SVG.

    from dobot_sim import programme_depuis_image, repere_travail

    FEUILLE = repere_travail(x=235, y=0, z=10)
    robot = programme_depuis_image("loup.png", repere=FEUILLE, largeur=80)
    robot.simuler()

Ce qui marche, et ce qui ne marche pas
--------------------------------------
Le procede suit une frontiere : il rend donc le CONTOUR des formes, pas
leur interieur. Un logo noir sur fond blanc, une silhouette, un dessin au
trait epais donnent un bon resultat. Une photographie, un degrade, une
image bruitee n'ont pas de frontiere nette : le seuillage y invente des
milliers de petits contours, et le trace n'a plus de sens.

Trois reglages suffisent presque toujours :

    seuil       None = seuil choisi tout seul (methode d'Otsu), sinon
                0..255. Baissez-le si des parties claires manquent.
    inverser    par defaut l'encre est SOMBRE sur fond clair. A mettre
                a True pour un dessin blanc sur fond noir.
    flou        lissage prealable, en pixels impairs. Monte a 5 ou 7 sur
                une image bruitee ou un JPEG tres compresse.

Dependance
----------
La lecture et le suivi de contour s'appuient sur OpenCV, qui n'est PAS
une dependance du simulateur : il n'est demande qu'ici, et seulement a
l'appel. `pip install opencv-python` s'il manque.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pilote import Pilote
from .trace import boite, programme_depuis_polylignes

__all__ = ["lire_image", "programme_depuis_image", "ErreurImage"]

Point = Tuple[float, float]

#: simplification du contour, en millimetres du dessin FINAL
TOLERANCE = 0.5

#: aire minimale d'un contour retenu, en pixels carres : en dessous,
#: c'est une poussiere de seuillage et non un trait
AIRE_MIN = 25.0


class ErreurImage(ValueError):
    """Image illisible, ou vide de tout contour exploitable."""


def _cv2():
    """OpenCV, charge au premier besoin et pas a l'import du simulateur."""
    try:
        import cv2
    except ImportError:
        raise ErreurImage(
            "le trace depuis une image demande OpenCV, qui ne fait pas "
            "partie des dependances du simulateur.\n"
            "    pip install opencv-python")
    return cv2


def lire_image(chemin: str,
               seuil: Optional[float] = None,
               inverser: bool = False,
               flou: int = 3,
               aire_min: float = AIRE_MIN,
               trous: bool = True,
               simplification_px: float = 0.0,
               journal=None) -> List[List[Point]]:
    """
    Extrait les contours d'une image et les rend en polylignes fermees.

    Les coordonnees sont en PIXELS, y vers le bas -- comme l'image. Les
    polylignes sont fermees : leur dernier point repete le premier.

    `trous` garde les contours interieurs (l'oeil d'une silhouette, le
    creux d'un O). A False, seules les frontieres exterieures sortent.
    """
    cv2 = _cv2()
    log = journal or (lambda *_: None)

    if not os.path.exists(chemin):
        raise ErreurImage("fichier introuvable : {}".format(chemin))
    # imread ne gere pas les chemins non ASCII sous Windows : on lit les
    # octets nous-memes et on laisse OpenCV decoder la memoire
    try:
        with open(chemin, "rb") as f:
            donnees = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError as exc:
        raise ErreurImage("lecture impossible : {}".format(exc))
    gris = cv2.imdecode(donnees, cv2.IMREAD_GRAYSCALE)
    if gris is None:
        raise ErreurImage(
            "image illisible : {} -- format non reconnu ?"
            .format(os.path.basename(chemin)))

    if flou and flou >= 3:
        k = int(flou) | 1                      # un noyau de flou est impair
        gris = cv2.GaussianBlur(gris, (k, k), 0)

    # Encre SOMBRE sur fond clair par defaut : la forme est ce qui est
    # sous le seuil, d'ou le BINARY_INV.
    mode = cv2.THRESH_BINARY if inverser else cv2.THRESH_BINARY_INV
    if seuil is None:
        valeur, binaire = cv2.threshold(gris, 0, 255, mode + cv2.THRESH_OTSU)
        log("seuil choisi par Otsu : {:.0f}".format(valeur))
    else:
        _, binaire = cv2.threshold(gris, float(seuil), 255, mode)

    contours, hierarchie = cv2.findContours(
        binaire, cv2.RETR_CCOMP if trous else cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)[-2:]

    polylignes: List[List[Point]] = []
    rejetes = 0
    for k, contour in enumerate(contours):
        if not trous and hierarchie is not None and hierarchie[0][k][3] != -1:
            continue                           # contour interieur ecarte
        if cv2.contourArea(contour) < aire_min:
            rejetes += 1
            continue
        if simplification_px > 0:
            contour = cv2.approxPolyDP(contour, simplification_px, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in contour]
        if len(pts) < 3:
            rejetes += 1
            continue
        if pts[0] != pts[-1]:
            pts.append(pts[0])                 # un contour se referme
        polylignes.append(pts)

    if not polylignes:
        raise ErreurImage(
            "aucun contour dans {} : l'image est-elle uniforme, ou l'encre "
            "est-elle claire sur fond sombre ? Essayez inverser=True, ou un "
            "seuil explicite.".format(os.path.basename(chemin)))
    log("{} contour(s) retenu(s), {} poussiere(s) ecartee(s)"
        .format(len(polylignes), rejetes))
    return polylignes


def programme_depuis_image(chemin: str,
                           repere=None,
                           largeur: float = 100.0,
                           hauteur: Optional[float] = None,
                           levee: float = 10.0,
                           profondeur: float = 0.0,
                           tolerance: float = TOLERANCE,
                           seuil: Optional[float] = None,
                           inverser: bool = False,
                           flou: int = 3,
                           aire_min: float = AIRE_MIN,
                           trous: bool = True,
                           vitesse_trace: float = 60.0,
                           vitesse_liaison: float = 150.0,
                           lissage_trace: float = 60.0,
                           outil: str = "stylo",
                           nom: Optional[str] = None,
                           journal=None) -> Pilote:
    """
    Construit le programme qui trace le profil d'une image.

    `tolerance` est la simplification du contour, exprimee en
    MILLIMETRES DU DESSIN FINAL -- comme pour un SVG, et non en pixels :
    0,5 veut dire la meme chose quelle que soit la definition de
    l'image. Un contour de photographie compte des milliers de points ;
    sans cette simplification, le programme serait injouable.

    Les autres parametres sont ceux de `programme_depuis_svg`.
    """
    log = journal or (lambda *_: None)

    # Deux passes : la premiere donne la taille en pixels, donc l'echelle,
    # donc la valeur en pixels que vaut la tolerance demandee en mm. La
    # seconde simplifie avec la bonne epsilon.
    brut = lire_image(chemin, seuil=seuil, inverser=inverser, flou=flou,
                      aire_min=aire_min, trous=trous, journal=log)
    x0, _, x1, _ = boite(brut)
    echelle = largeur / max(x1 - x0, 1e-9)          # mm par pixel
    epsilon = float(tolerance) / echelle if echelle > 0 else 0.0

    polylignes = lire_image(chemin, seuil=seuil, inverser=inverser, flou=flou,
                            aire_min=aire_min, trous=trous,
                            simplification_px=epsilon)
    log("simplification : {:.2f} mm = {:.1f} pixel(s)".format(tolerance, epsilon))

    return programme_depuis_polylignes(
        polylignes, repere=repere, largeur=largeur, hauteur=hauteur,
        levee=levee, profondeur=profondeur, vitesse_trace=vitesse_trace,
        vitesse_liaison=vitesse_liaison, lissage_trace=lissage_trace,
        outil=outil, y_vers_le_bas=True, journal=log,
        nom=nom or os.path.splitext(os.path.basename(chemin))[0])
