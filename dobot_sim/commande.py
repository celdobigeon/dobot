#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commande "temps reel" : chaque appel s'execute TOUT DE SUITE.

`Pilote` decrit un programme, puis le joue d'un bloc. `Commande` est
l'autre paradigme : chaque ligne du script bouge le robot, et ne rend la
main qu'une fois le mouvement FINI. C'est ce qu'il faut pour un script
interactif -- une cible tapee au clavier, plus tard une camera :

    from dobot_sim import Commande, cible

    with Commande(simulateur=True, robot=False, vitesse=40) as robot:
        robot.movj(cible(220, 0, 80))        # rend la main a l'arrivee
        x, y, z, r = robot.pose()            # lue sur la machine
        robot.movl(z=40)                     # descente verticale
        robot.ventouse(True)
        robot.movl([(cible(220, 60, 80), 1.0),      # plusieurs cibles,
                    (cible(260, 60, 80), 0.5)])     # SANS s'arreter

Memes verbes et memes cibles que `Pilote` : une matrice homogene 4x4 de
preference, ou des coordonnees x, y, z, r ; une composante omise reprend
celle de la cible precedente.

Trois facons de choisir ce qui bouge
------------------------------------
  * `robot=True, simulateur=False` : le script tient lui-meme la liaison
    serie, sans fenetre ;
  * `simulateur=True, robot=False` : chaque commande part a la fenetre du
    simulateur deja ouverte, qui l'anime ; le script attend la fin ;
  * `simulateur=True, robot=True` : la FENETRE tient la liaison (un port
    serie ne s'ouvre qu'une fois) ; le script lui demande de se connecter,
    puis chaque commande passe par elle, qui l'execute sur le robot, miroir
    actif. On attend la fin du ROBOT.

`lancer_simulateur=True` ouvre la fenetre (main.py) dans un PROCESSUS
SEPARE si aucune ne repond : Qt veut le fil principal, que le script garde.
Fermer le script ne ferme pas la fenetre.

Moins sur qu'un programme relu, donc garde-fous explicites
----------------------------------------------------------
Avant CHAQUE mouvement, depuis la posture LUE sur la machine -- et non celle
que le script croit : MGI, butees et collisions sur tout le chemin, zone de
travail (`zone=`), deplacement maximal (`pas_max=`). Un refus affiche une
alerte rouge et leve `ErreurExecution` : rien n'est parti. Un robot en
defaut ouvre l'assistant de mise en route. Ctrl+C declenche un arret
d'urgence.

Mouvements continus
-------------------
Le mode point a point du controleur (PTP) s'arrete a chaque cible. Le mode
CP, lui, enchaine sans arret ; ses regles ont ete MESUREES sur la machine
(firmware 3.7.0, voir `essai_cp.py`) et sont appliquees ici :

  * la vitesse de croisiere est `junctionVel` de CPParams -- le champ
    vitesse de chaque point est ignore -- et CPParams MIS EN FILE change la
    vitesse au bon moment ;
  * l'acceleration decide si cette vitesse est atteinte, sur les droites
    comme dans les courbes : un arc de rayon 60 echantillonne tous les
    5 mm plafonnait a 62 mm/s sous 200 mm/s2 ;
  * pas de r : l'outil ne tourne pas en CP. Un segment qui le fait tourner
    repasse en point a point, et c'est annonce ;
  * un premier point a 1 mm du depart attenue l'a-coup du demarrage.

La trajectoire complete est construite par le moteur du simulateur
(`Programme` / `Trajectoire` : durees, arcs, lissage, MOVJ compris), puis
echantillonnee en points rapproches ; chaque portion recoit la vitesse qui
tient sa duree. Avant le depart, on PREVOIT si chaque duree est tenable, et
sinon on le dit, avec la duree realiste.
"""

from __future__ import annotations

import copy
import math
import numbers
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import homogeneous as hg
from .alertes import (ErreurExecution, _alerter, _echouer, _signaler,
                      _signaler_urgence)
from .outils import outil as catalogue_outil
from .passerelle import demander
from .program import TYPES_MOUVEMENT, Programme
from .robot import DobotMagician, ErreurCinematique
from .trajectory import ProfilTrapeze, Trajectoire

__all__ = ["Commande", "ErreurExecution", "Plan", "planifier",
           "commandes_robot", "vitesse_pour_duree"]


# ---------------------------------------------------------------------------
# Constantes du mode continu, d'apres les mesures sur la machine
# ---------------------------------------------------------------------------

#: mm/s2 (et deg/s2 en articulaire). Mesure : 200 mm/s2 laissait une ligne
#: droite a 107 mm/s pour 120 demandes, 400 l'amenait a 126.
ACCELERATION_DEFAUT = 400.0

#: espacement des points CP le long d'une courbe, en mm. C'est celui de
#: l'arc mesure ; une droite, elle, ne garde que ses extremites.
PAS_CP = 5.0

#: le premier point d'une trajectoire continue, a cette distance du depart,
#: en mm : il absorbe l'a-coup constate au demarrage.
PREMIER_POINT_CP = 1.0

#: s : apres l'ouverture de la pince (ou le relachement de la ventouse),
#: la pompe souffle ; on la coupe au bout de ce delai, le temps que la
#: pince s'ouvre vraiment.
DELAI_COUPURE_POMPE = 1.0

#: vitesse la plus haute MESUREE : 126 mm/s (ligne droite, 120 demandes,
#: acceleration 400). Au-dela, rien n'a ete essaye : on n'y va pas.
VITESSE_CP_MAX = 125.0
VITESSE_CP_MIN = 2.0

#: Vitesse de passage a une jonction entre deux points CP. Modele
#: centripete v^2 = K . a . R, R etant le rayon du cercle qui passe par les
#: points (pour un angle theta entre segments de longueur d : R = d / 2
#: sin(theta / 2)). K est CALE sur la seule mesure disponible : arc de
#: rayon 60 mm, points tous les 5 mm, a = 200 mm/s2 -> 62 mm/s. Que K reste
#: le meme a 400 mm/s2, et dans un angle vif, n'a PAS ete verifie.
K_JONCTION = 62.0 ** 2 / (60.0 * 200.0)

#: en deca, on considere que l'outil ne tourne pas (degres)
TOLERANCE_R = 0.5

#: ecart tolere entre points alignes, en mm, pour alleger une droite
TOLERANCE_ALIGNEMENT = 0.05


def _scalaire(v) -> bool:
    """Un nombre, et non une cible ou une liste."""
    if isinstance(v, bool):
        return False
    return isinstance(v, numbers.Real) or (isinstance(v, np.ndarray)
                                           and v.ndim == 0)


def _ecart_angle(a: float, b: float) -> float:
    """b - a, ramene dans [-180 ; 180[."""
    return (float(b) - float(a) + 180.0) % 360.0 - 180.0


def vitesse_pour_duree(longueur: float, duree: float, acceleration: float,
                       arrets: int = 2) -> Optional[float]:
    """
    Vitesse de croisiere qui fait durer un mouvement `duree` secondes.

    Le controleur accelere et freine a `acceleration` constante (trapeze).
    Partant ou s'arretant `arrets` fois (0, 1 ou 2), chaque rampe coute
    v / 2a de plus qu'a vitesse constante :

        duree = L / v + arrets . v / (2 a)

    d'ou une equation du second degre, dont on prend la plus petite racine
    -- la seule ou le palier existe. None si aucune vitesse ne tient cette
    duree : l'acceleration ne le permet pas.
    """
    L = abs(float(longueur))
    T = float(duree)
    a = max(float(acceleration), 1e-9)
    if L <= 1e-9 or T <= 1e-9:
        return None
    if arrets <= 0:
        return L / T
    disc = T * T - 2.0 * arrets * L / a
    if disc < 0.0:
        return None
    return (T - math.sqrt(disc)) * a / arrets


# ---------------------------------------------------------------------------
# Echantillonnage d'une trajectoire en points CP
# ---------------------------------------------------------------------------

def _echantillons(traj: Trajectoire, t0: float, t1: float,
                  pas_temps: float = 0.004) -> Tuple[np.ndarray, np.ndarray]:
    """Positions du TCP entre t0 et t1, finement (quelques dixiemes de mm)."""
    n = int(min(max(math.ceil((t1 - t0) / pas_temps) + 1, 20), 8000))
    ts = np.linspace(t0, t1, n)
    P = np.array([hg.position_de(traj.pose(t)) for t in ts], dtype=float)
    return ts, P


def _distance_au_segment(p, a, b) -> float:
    ab = b - a
    n = float(ab @ ab)
    if n < 1e-18:
        return float(np.linalg.norm(p - a))
    u = float(np.clip((p - a) @ ab / n, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab)))


def _simplifier(points: List[np.ndarray], depart: np.ndarray,
                proteges: Sequence[int] = ()) -> List[np.ndarray]:
    """
    Retire les points alignes : une droite n'a besoin que de ses bouts.

    Un point n'est retire que si TOUS ceux qu'on a deja retires depuis le
    dernier point garde restent a moins de TOLERANCE_ALIGNEMENT de la
    nouvelle corde -- sans cela l'erreur s'accumulerait le long d'une
    courbe douce. Le dernier point et les points `proteges` restent.
    """
    garde: List[np.ndarray] = []
    ref = np.asarray(depart, dtype=float)
    attente: List[np.ndarray] = []
    for i, p in enumerate(points):
        dernier = i == len(points) - 1
        if dernier or i in proteges:
            garde.append(p)
            ref, attente = p, []
            continue
        suivant = points[i + 1]
        if all(_distance_au_segment(q, ref, suivant) <= TOLERANCE_ALIGNEMENT
               for q in attente + [p]):
            attente.append(p)
        else:
            garde.append(p)
            ref, attente = p, []
    # deux points confondus feraient un segment nul, que le controleur
    # n'aime pas : on les fond
    out: List[np.ndarray] = []
    prec = np.asarray(depart, dtype=float)
    for p in garde:
        if np.linalg.norm(p - prec) > 0.05:
            out.append(p)
            prec = p
    return out


def echantillonner_course(traj: Trajectoire, i0: int, i1: int,
                          pas: float = PAS_CP,
                          premier: float = PREMIER_POINT_CP
                          ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Points CP d'une suite de segments [i0 ; i1] de la trajectoire.

    Rend (depart, portions) : une portion par segment, chacune la liste de
    ses points (x, y, z) du SIMULATEUR, le dernier etant la fin de la
    portion. Avec lissage, deux segments se recouvrent : la frontiere est
    prise au milieu du recouvrement.

    Les points sont espaces de `pas` le long du chemin, puis les points
    alignes sont retires. Le tout premier est place a `premier` mm du
    depart, pour adoucir le demarrage.
    """
    t0 = traj.debuts[i0]
    t1 = traj.debuts[i1] + traj.segments[i1].duree
    bornes = [t0] + [traj.debuts[k + 1] + traj.recouvrements[k] / 2.0
                     for k in range(i0, i1)] + [t1]
    ts, P = _echantillons(traj, t0, t1)
    S = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0),
                                                        axis=1))])
    Sb = np.interp(bornes, ts, S)

    def point(s):
        return np.array([np.interp(s, S, P[:, j]) for j in range(3)])

    portions: List[List[np.ndarray]] = []
    prec = P[0]
    for k in range(len(Sb) - 1):
        sa, sb = float(Sb[k]), float(Sb[k + 1])
        valeurs, proteges = [], []
        debut = sa
        if k == 0 and sb - sa > 3.0 * premier:
            valeurs.append(sa + premier)
            proteges.append(0)
            debut = sa + premier
        s = sa + pas
        while s < sb - 0.3 * pas:
            if s > debut + 0.3 * pas:
                valeurs.append(s)
            s += pas
        # la fin de portion est prise sur la trajectoire elle-meme, et non
        # interpolee : un coin doit tomber exactement sur la cible
        pts = [point(v) for v in valeurs] + [hg.position_de(traj.pose(bornes[k + 1]))]
        pts = _simplifier(pts, prec, proteges)
        portions.append(pts)
        if pts:
            prec = pts[-1]
    return P[0].copy(), portions


def _duree_troncon(d: float, w0: float, w1: float, c: float, a: float) -> float:
    """Duree d'un troncon de longueur d, entree w0, sortie w1, palier c."""
    c = max(c, w0, w1, 1e-6)
    da = (c * c - w0 * w0) / (2.0 * a)
    dd = (c * c - w1 * w1) / (2.0 * a)
    if da + dd <= d:
        return (c - w0) / a + (c - w1) / a + (d - da - dd) / c
    vp = math.sqrt(max((2.0 * a * d + w0 * w0 + w1 * w1) / 2.0, 0.0))
    vp = max(vp, w0, w1)
    return (vp - w0) / a + (vp - w1) / a


def prevoir_cp(depart, portions: List[List[np.ndarray]],
               vitesses: Sequence[float], acceleration: float,
               pas: float = PAS_CP) -> Tuple[List[float], List[bool]]:
    """
    Duree previsible de chaque portion d'une trajectoire CP.

    Modele d'anticipation classique de commande numerique : chaque troncon
    vise la vitesse de sa portion ; chaque jonction est plafonnee par le
    modele centripete (K_JONCTION) ; le robot part et finit a l'arret, et
    ne peut accelerer ni freiner plus fort que `acceleration` -- d'ou une
    passe avant et une passe arriere. Rend (durees, courbure_limitante).
    """
    a = max(float(acceleration), 1e-6)
    pts = [np.asarray(depart, dtype=float)]
    proprio: List[int] = []
    for k, pk in enumerate(portions):
        for p in pk:
            pts.append(np.asarray(p, dtype=float))
            proprio.append(k)
    m = len(portions)
    n = len(pts) - 1
    durees = [0.0] * m
    limite = [False] * m
    if n <= 0:
        return durees, limite
    d = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(n)]
    c = [max(float(vitesses[proprio[i]]), 1e-6) for i in range(n)]
    w = [0.0] * (n + 1)
    for i in range(1, n):
        if d[i - 1] < 1e-9 or d[i] < 1e-9:
            w[i] = min(c[i - 1], c[i])
            continue
        u0 = (pts[i] - pts[i - 1]) / d[i - 1]
        u1 = (pts[i + 1] - pts[i]) / d[i]
        theta = math.acos(float(np.clip(u0 @ u1, -1.0, 1.0)))
        vj = math.inf
        if theta > 1e-4:
            R = min(d[i - 1], d[i], pas) / (2.0 * math.sin(theta / 2.0))
            vj = math.sqrt(K_JONCTION * a * R)
        w[i] = min(c[i - 1], c[i], vj)
        if vj < min(c[i - 1], c[i]) - 1e-6:
            limite[proprio[i]] = True
    for i in range(n):                                   # passe avant
        w[i + 1] = min(w[i + 1], math.sqrt(w[i] ** 2 + 2.0 * a * d[i]))
    for i in range(n - 1, -1, -1):                       # passe arriere
        w[i] = min(w[i], math.sqrt(w[i + 1] ** 2 + 2.0 * a * d[i]))
    for i in range(n):
        durees[proprio[i]] += _duree_troncon(d[i], w[i], w[i + 1], c[i], a)
    return durees, limite


def _longueur_tcp(seg) -> float:
    """Longueur du chemin du TCP, en mm, y compris pour un MOVJ."""
    if seg.type_mouvement in ("MOVL", "MOVC"):
        return float(seg.longueur)
    ts = np.linspace(0.0, seg.duree, 60)
    P = np.array([hg.position_de(seg.pose(t)) for t in ts])
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Plan d'execution
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """
    Ce qu'une commande va faire, decide AVANT de bouger.

    `programme` est ce que joue le simulateur ; `etapes` ce que recoit le
    robot, sous une forme qui passe la passerelle (JSON) :

        {"type": "ptp", "index": i}      l'instruction i du programme, en
                                         point a point (ou action d'outil)
        {"type": "cp", "r": r, "acceleration": a,
         "portions": [{"vitesse": v, "points": [[x, y, z], ...]}, ...]}
    """
    programme: Programme
    etapes: List[Dict[str, Any]]
    annonces: List[str] = field(default_factory=list)
    avertissements: List[str] = field(default_factory=list)
    trajectoire: Optional[Trajectoire] = None

    @property
    def duree(self) -> float:
        return self.trajectoire.duree if self.trajectoire is not None else 0.0

    @property
    def continu(self) -> bool:
        return any(e["type"] == "cp" for e in self.etapes)


def planifier(programme: Programme, robot: DobotMagician, q_depart,
              continu: bool = True,
              acceleration: float = ACCELERATION_DEFAUT,
              acceleration_articulaire: float = ACCELERATION_DEFAUT,
              vitesse: float = 50.0, pas: float = PAS_CP,
              vitesse_max: float = VITESSE_CP_MAX) -> Plan:
    """
    Decide, instruction par instruction, comment le robot va l'executer.

    1. Les mouvements consecutifs forment une COURSE continue (CP), sauf
       si `continu=False`, ou si un segment fait tourner l'outil -- le CP
       n'a pas de r -- ou si une action d'outil s'intercale. Une course
       d'un seul MOVL ou MOVJ part en point a point, mode eprouve qui
       garde r ; un arc seul reste en CP, ou sa vitesse est maitrisee.
    2. En point a point, la duree demandee devient une vitesse (trapeze du
       controleur) : simulateur et robot jouent alors le meme profil.
    3. En continu, chaque segment recoit sa vitesse de croisiere, ajustee
       jusqu'a ce que la duree PREVUE (`prevoir_cp`) colle a la duree
       demandee. Ce qui ne peut pas tenir est signale avec la duree
       realiste, et c'est elle que joue le simulateur.

    Le programme rendu (`plan.programme`) est celui que le simulateur doit
    animer : memes chemins, durees realistes, lois horaires qui
    s'enchainent sans arret la ou le robot ne s'arrete pas.
    """
    q0 = np.asarray(q_depart, dtype=float)
    prog = Programme(copy.deepcopy(programme.instructions), programme.nom,
                     programme.outil, programme.repere_objet,
                     programme.longueur_outil)
    ins = prog.instructions
    traj0 = prog.construire(robot, q0)
    annonces: List[str] = []
    avert: List[str] = []
    a_cart = float(acceleration)
    a_art = float(acceleration_articulaire)
    nb_mouvements = sum(1 for x in ins if x.type in TYPES_MOUVEMENT)

    def nom(i):
        return "{} {}".format(i + 1, ins[i].type)

    def rotation(i) -> float:
        seg = traj0.segments[i]
        return _ecart_angle(robot.pose(seg.q(0.0))[3],
                            robot.pose(seg.q_final)[3])

    # -- 1. decoupage en courses continues et en points a point ------------
    # le lissage du DERNIER mouvement ne raccorde a rien : on ne l'annonce pas
    mouvements = [x for x in ins if x.type in TYPES_MOUVEMENT]
    if not continu and any(x.lissage > 0 for x in mouvements[:-1]):
        annonces.append("lissage ignore : en point a point, le robot "
                        "s'arrete a chaque cible")
    groupes: List[Tuple[str, Any]] = []
    course: List[int] = []

    def clore():
        if course:
            groupes.append(("cp", list(course)))
            course.clear()

    for i, x in enumerate(ins):
        if x.type not in TYPES_MOUVEMENT:
            if continu and course and any(y.type in TYPES_MOUVEMENT
                                          for y in ins[i + 1:]):
                annonces.append("{} : action au milieu du mouvement, le "
                                "robot s'arrete pour l'executer".format(nom(i)))
            clore()
            groupes.append(("ptp", i))
            continue
        dr = rotation(i)
        if abs(dr) > TOLERANCE_R:
            clore()
            groupes.append(("ptp", i))
            if x.type == "MOVC":
                annonces.append("{} : l'outil tourne de {:+.0f} deg sur "
                                "l'arc -> arc du controleur (SetARCCmd), "
                                "duree moins bien tenue".format(nom(i), dr))
            elif continu and nb_mouvements > 1:
                annonces.append("{} : l'outil tourne de {:+.0f} deg -> point "
                                "a point (le mode continu ne tourne pas "
                                "l'outil), arret avant et apres"
                                .format(nom(i), dr))
            continue
        if not continu:
            groupes.append(("cp", [i]))
            continue
        course.append(i)
    clore()
    # une course d'un seul MOVL / MOVJ : le point a point fait aussi bien,
    # et c'est le mode eprouve ; un arc seul reste en CP, sa vitesse y est
    # maitrisee alors que celle de SetARCCmd ne l'est pas
    groupes = [("ptp", g[1][0]) if (g[0] == "cp" and len(g[1]) == 1
                                    and ins[g[1][0]].type != "MOVC") else g
               for g in groupes]

    # -- 2. points a point : la duree devient une vitesse ------------------
    for genre, g in groupes:
        if genre != "ptp" or ins[g].type not in TYPES_MOUVEMENT:
            continue
        x, seg = ins[g], traj0.segments[g]
        a = a_art if x.type == "MOVJ" else a_cart
        unite = "deg" if x.type == "MOVJ" else "mm"
        L = float(seg.longueur)
        # meme prudence qu'en continu : rien au-dela de ce qui a ete mesure
        # (en mm/s), ni des limites articulaires du modele (en deg/s)
        v_lim = (float(min(robot.lim.v_max[:3])) if x.type == "MOVJ"
                 else float(vitesse_max))
        v = x.vitesse_effective()
        if x.mode_temps == "duree":
            v = vitesse_pour_duree(L, x.duree, a, 2)
            if v is None:
                v = math.sqrt(a * L) if L > 1e-6 else x.vitesse_effective()
        if v > v_lim and L > 1e-6:
            v = v_lim
        if x.mode_temps == "duree" and L > 1e-6:
            tenue = ProfilTrapeze.duree_pour(L, v, a)
            if tenue > x.duree * 1.03 + 0.05:
                avert.append("{} : {:.2f} s demandees, {:.2f} s au mieux "
                             "(acceleration {:.0f} {u}/s2, vitesse {:.0f} "
                             "{u}/s au plus)".format(nom(g), x.duree, tenue,
                                                     a, v_lim, u=unite))
        elif v < x.vitesse_effective() - 1e-9:
            avert.append("{} : {:.0f} {u}/s demandes, limites a {:.0f}"
                         .format(nom(g), x.vitesse_effective(), v, u=unite))
        x.mode_temps = "vitesse"
        x.vitesse = float(v)
        x.acceleration = float(a)
        x.profil = "trapeze"
        x.lissage = 0.0

    # -- 3. courses continues : une vitesse par segment ---------------------
    vitesses_courses: Dict[int, List[float]] = {}
    for genre, g in groupes:
        if genre != "cp":
            continue
        n = len(g)
        L = [_longueur_tcp(traj0.segments[i]) for i in g]
        T = [ins[i].duree if ins[i].mode_temps == "duree" else None for i in g]
        # Un MOVJ exprime sa vitesse en deg/s : elle n'a pas de sens pour le
        # TCP. On garde donc la DUREE que lui donne le simulateur a cette
        # vitesse articulaire, et c'est elle que la course doit tenir. Sans
        # cela, vitesse=10 ou vitesse=100 donnaient la meme duree.
        for k, i in enumerate(g):
            if T[k] is None and ins[i].type == "MOVJ":
                T[k] = float(traj0.segments[i].duree)
        arrets = [int(k == 0) + int(k == n - 1) for k in range(n)]
        v: List[float] = []
        for k, i in enumerate(g):
            vk = (ins[i].vitesse_effective() if ins[i].type != "MOVJ"
                  else float(vitesse))
            if T[k] is not None:
                vk = vitesse_pour_duree(L[k], T[k], a_cart, arrets[k])
                if vk is None:
                    vk = (math.sqrt(2.0 * a_cart * L[k] / arrets[k])
                          if arrets[k] and L[k] > 1e-9 else float(vitesse))
            elif vk > vitesse_max:
                avert.append("{} : {:.0f} mm/s demandes, limites a {:.0f} en "
                             "mode continu (vitesse jamais mesuree au-dela)"
                             .format(nom(i), vk, vitesse_max))
            v.append(float(np.clip(vk, VITESSE_CP_MIN, vitesse_max)))
        # Plafond de la correction. Pour rattraper le temps perdu a passer
        # d'une vitesse a l'autre entre portions, la consigne peut monter
        # un peu au-dessus de la vitesse nominale, jamais au-dela de 125 %.
        # Quand c'est la COURBURE qui freine, elle n'en depasse pas la
        # vitesse nominale : le modele de courbure n'est cale que sur une
        # mesure, et s'il etait trop prudent, le robot irait sinon bien
        # plus vite que ce que la duree demandee laissait attendre. Mieux
        # vaut un robot un peu lent, et annonce comme tel.
        nominale = list(v)
        plafond = [min(vitesse_max, 1.25 * vk) for vk in v]
        depart, portions = echantillonner_course(traj0, g[0], g[-1], pas)
        reel, courbe = prevoir_cp(depart, portions, v, a_cart, pas)
        for _ in range(12):
            fini = True
            for k in range(n):
                if T[k] is None or reel[k] <= 1e-6:
                    continue
                if abs(reel[k] - T[k]) > 0.005 * T[k]:
                    haut = nominale[k] if courbe[k] else plafond[k]
                    nouvelle = float(np.clip(v[k] * reel[k] / T[k],
                                             VITESSE_CP_MIN, haut))
                    if abs(nouvelle - v[k]) > 1e-6:
                        fini = False
                    v[k] = nouvelle
            if fini:
                break
            reel, courbe = prevoir_cp(depart, portions, v, a_cart, pas)
        for k, i in enumerate(g):
            if courbe[k]:
                cause = "courbure ou angle : le robot ralentit aux jonctions"
            elif v[k] >= vitesse_max - 1e-6:
                cause = "vitesse plafonnee a {:.0f} mm/s".format(vitesse_max)
            else:
                cause = "acceleration {:.0f} mm/s2".format(a_cart)
            if T[k] is not None and reel[k] > T[k] * 1.03 + 0.05:
                avert.append("{} : {:.2f} s demandees, ~{:.2f} s en realite "
                             "({})".format(nom(i), T[k], reel[k], cause))
            elif T[k] is None and L[k] > 1e-6:
                ideal = L[k] / v[k] + arrets[k] * v[k] / (2.0 * a_cart)
                if reel[k] > ideal * 1.15 + 0.05:
                    avert.append("{} : {:.0f} mm/s non tenus, ~{:.2f} s au "
                                 "lieu de {:.2f} ({})".format(
                                     nom(i), v[k], reel[k], ideal, cause))
        # Durees du simulateur : celles que le robot tiendra vraiment. Avec
        # lissage, le moteur fait se recouvrir deux segments ; on allonge
        # chacun de la moitie de ses recouvrements pour que chaque PORTION
        # (de milieu de raccord a milieu de raccord) dure ce que dure la
        # portion du robot.
        f = [float(np.clip(ins[i].lissage, 0.0, 100.0)) / 100.0
             for i in g[:-1]] + [0.0]
        dur = [max(t, 0.05) for t in reel]
        for _ in range(40):
            rec = [f[k] * Trajectoire.RECOUVREMENT_MAX * min(dur[k], dur[k + 1])
                   for k in range(n - 1)]
            dur = [max(reel[k], 0.05) + (rec[k - 1] / 2.0 if k > 0 else 0.0)
                   + (rec[k] / 2.0 if k < n - 1 else 0.0) for k in range(n)]
        for k, i in enumerate(g):
            x = ins[i]
            x.mode_temps = "duree"
            x.duree = float(dur[k])
            x.vitesse = float(v[k])
            x.acceleration = a_cart
            x.lissage = f[k] * 100.0
            x.profil = ("trapeze" if n == 1 else "demarrage" if k == 0
                        else "arret" if k == n - 1 else "lineaire")
        vitesses_courses[g[0]] = v

    traj = prog.construire(robot, q0)

    # -- 4. etapes du robot, sur le chemin que joue le simulateur ----------
    etapes: List[Dict[str, Any]] = []
    for genre, g in groupes:
        if genre == "ptp":
            etapes.append({"type": "ptp", "index": int(g)})
            continue
        _depart, portions = echantillonner_course(traj, g[0], g[-1], pas)
        r = float(robot.pose(traj.segments[g[0]].q(0.0))[3])
        etapes.append({
            "type": "cp", "r": r, "acceleration": a_cart,
            "portions": [{"vitesse": float(vk),
                          "points": [[float(c) for c in p] for p in pts]}
                         for vk, pts in zip(vitesses_courses[g[0]], portions)]})
    if avert:
        avert.append("duree prevue de l'ensemble : ~{:.2f} s".format(traj.duree))
    return Plan(prog, etapes, annonces, avert, traj)


def commandes_robot(pont, programme: Programme, etapes, outil: str = ""):
    """
    Traduit les etapes d'un plan en commandes de file, SANS les envoyer.

    Point a point et actions d'outil passent par la traduction eprouvee du
    pont ; les courses continues par `commandes_cp`. A donner a
    `PontReel.televerser_commandes`, qui l'appelle apres avoir vide la
    file.
    """
    out = []
    for e in etapes:
        if e["type"] == "ptp":
            out += pont.traduire(programme.instructions[int(e["index"])],
                                 programme, outil or programme.outil)
        elif e["type"] == "cp":
            for p in e["portions"]:
                out += pont.commandes_cp(p["points"], p["vitesse"],
                                         e["acceleration"], e.get("r", 0.0))
    return out


# ---------------------------------------------------------------------------
# La classe Commande
# ---------------------------------------------------------------------------

#: messages de pont sans interet pour l'operateur, une commande apres
#: l'autre : la progression ne s'affiche pas
_JOURNAL_MUET = ("execution reelle lancee", "commandes televersees",
                 "programme televerse", "ratio de vitesse reel",
                 "file du controleur remplie")

_SANS_NOMBRES = re.compile(r"[-+]?\d[\d.]*")


class Commande:
    """
    Pilotage ou chaque appel bouge TOUT DE SUITE, et attend la fin.

        robot = Commande(simulateur=True, robot=False, vitesse=40)
        robot.movj(cible(220, 0, 80))
        robot.movl(cible(220, 50, 80), duree=1.5)
        robot.fermer()

    ou, pour etre sur de fermer :

        with Commande(robot=True, simulateur=False) as robot:
            ...

    Vitesses : `vitesse` en mm/s pour les mouvements cartesiens (MOVL,
    MOVC), `vitesse_articulaire` en deg/s pour les MOVJ -- y compris
    enchaines en continu, ou la duree qu'elle donne au simulateur est
    celle que la course tient. 50 mm/s et 50 deg/s n'ont rien de
    comparable. Une duree (`duree=`) l'emporte sur la vitesse.

    `zone=((xmin, xmax), (ymin, ymax), (zmin, zmax))` refuse toute cible,
    et tout trajet, qui en sortirait. `pas_max=` refuse une cible a plus de
    tant de mm de la precedente (en ligne droite, depart compris).
    """

    def __init__(self, simulateur: bool = True, robot: bool = False,
                 vitesse: float = 50.0, port: Optional[str] = None,
                 lancer_simulateur: bool = False, zone=None,
                 pas_max: Optional[float] = None, outil: str = "ventouse",
                 longueur_outil: Optional[float] = None,
                 vitesse_articulaire: float = 50.0,
                 acceleration: float = ACCELERATION_DEFAUT,
                 acceleration_articulaire: float = ACCELERATION_DEFAUT,
                 lissage: float = 0.0, continu: bool = True):
        if not simulateur and not robot:
            raise ValueError("Commande : choisissez au moins une cible, "
                             "simulateur=True et/ou robot=True")
        self.simulateur = bool(simulateur)
        self.robot_reel = bool(robot)
        self.vitesse = float(vitesse)
        self.vitesse_articulaire = float(vitesse_articulaire)
        self.acceleration = float(acceleration)
        self.acceleration_articulaire = float(acceleration_articulaire)
        self.lissage = float(lissage)
        self.continu = bool(continu)
        self.zone = self._verifier_zone(zone)
        self.pas_max = None if pas_max is None else float(pas_max)

        self.outil = catalogue_outil(outil).nom
        self.modele = DobotMagician()
        self.longueur_outil = float(catalogue_outil(outil).longueur
                                    if longueur_outil is None
                                    else longueur_outil)
        self.modele.geo.lt = self.longueur_outil

        self._pont = None                   # liaison directe (cible 1)
        self._ferme = False
        # coupure de pompe programmee apres une ouverture : (minuterie,
        # instant prevu)
        self._coupure: Optional[Tuple[threading.Timer, float]] = None
        self._vus: set = set()              # reserves deja signalees
        self.processus_simulateur = None    # si lance par nous

        if self.simulateur:
            self._assurer_simulateur(bool(lancer_simulateur))
            if self.robot_reel:
                self._connecter_fenetre(port)
        else:
            self._connecter_direct(port)
        # derniere cible connue : point de depart des composantes omises
        self._pose = list(self.pose())

    # -- cycle de vie ---------------------------------------------------------

    def __enter__(self) -> "Commande":
        return self

    def __exit__(self, *_exc) -> None:
        self.fermer()

    def fermer(self) -> None:
        """
        Libere ce que la commande tient. La fenetre du simulateur, elle,
        reste ouverte -- et garde sa liaison avec le robot s'il y en a une.
        """
        if self._ferme:
            return
        # Une coupure de pompe encore attendue doit partir AVANT que la
        # liaison se ferme : sinon la pompe tournerait jusqu'a l'extinction.
        if self._coupure is not None:
            minuterie, echeance = self._coupure
            minuterie.cancel()
            self._coupure = None
            time.sleep(max(0.0, echeance - time.time()))
            try:
                self.couper_pompe()
            except ErreurExecution:
                pass
        self._ferme = True
        if self._pont is not None:
            from .pilote import _fermer_liaison_directe
            _fermer_liaison_directe()
            self._pont = None

    @property
    def direct(self) -> bool:
        """Vrai quand le script tient lui-meme la liaison serie."""
        return not self.simulateur

    def _exiger_ouverte(self) -> None:
        if self._ferme:
            raise RuntimeError("cette Commande est fermee")

    # -- reglages ---------------------------------------------------------------

    def regler(self, vitesse: Optional[float] = None,
               acceleration: Optional[float] = None,
               vitesse_articulaire: Optional[float] = None,
               acceleration_articulaire: Optional[float] = None,
               lissage: Optional[float] = None,
               continu: Optional[bool] = None) -> "Commande":
        """Change les valeurs par defaut pour la suite du script."""
        if vitesse is not None:
            self.vitesse = float(vitesse)
        if acceleration is not None:
            self.acceleration = float(acceleration)
        if vitesse_articulaire is not None:
            self.vitesse_articulaire = float(vitesse_articulaire)
        if acceleration_articulaire is not None:
            self.acceleration_articulaire = float(acceleration_articulaire)
        if lissage is not None:
            self.lissage = float(lissage)
        if continu is not None:
            self.continu = bool(continu)
        return self

    def regler_longueur_outil(self, millimetres: float) -> "Commande":
        """
        Impose la longueur reelle de l'outil (voir Pilote.regler_longueur_outil).

        Le z designe la POINTE de l'outil : changer de longueur change le
        sens de toutes les cibles. La derniere cible memorisee est donc
        relue sur la machine.
        """
        self.longueur_outil = float(millimetres)
        self.modele.geo.lt = self.longueur_outil
        self._pose = list(self.pose())
        return self

    @staticmethod
    def _verifier_zone(zone):
        if zone is None:
            return None
        try:
            z = [(float(a), float(b)) for a, b in zone]
        except (TypeError, ValueError):
            z = []
        if len(z) != 3 or any(a >= b for a, b in z):
            raise ValueError("zone attendue : ((xmin, xmax), (ymin, ymax), "
                             "(zmin, zmax)), chaque min < max")
        return tuple(z)

    # -- lectures -------------------------------------------------------------

    def angles(self) -> np.ndarray:
        """Les quatre angles articulaires, LUS (robot, ou fenetre), en degres."""
        return self._lire_q()

    def pose(self) -> Tuple[float, float, float, float]:
        """
        (x, y, z, r) de la pointe d'outil, LU, dans le repere du simulateur.

        Sur le robot, c'est la pose du controleur, convertie
        (`PontReel.depuis_controleur`) : son origine et son deport ne sont
        pas ceux du modele.
        """
        self._exiger_ouverte()
        if self.direct:
            try:
                p = self._pont.liaison.pose()
            except Exception as exc:
                _echouer("LECTURE IMPOSSIBLE SUR LE ROBOT", [str(exc)])
            return tuple(float(v) for v in
                         self._pont.depuis_controleur(p.x, p.y, p.z, p.r))
        rep = self._lire()
        if self.robot_reel and rep.get("pose_controleur"):
            from .reel import PontReel
            conv = PontReel(self.modele)
            conv.biais = tuple(float(v) for v in rep.get("biais", (0, 0, 0)))
            return tuple(float(v) for v in
                         conv.depuis_controleur(*rep["pose_controleur"]))
        return tuple(float(v) for v in self.modele.pose(rep["q"]))

    def entree(self, port: int) -> bool:
        """
        Niveau de l'entree EIO`port`. Sur le robot, le port est configure
        en entree au premier usage ; dans le simulateur, c'est le bouton du
        panneau Entrees / Sorties qui repond.
        """
        self._exiger_ouverte()
        if self.direct:
            from .liaison import FonctionES
            try:
                li = self._pont.liaison
                self._pont._assurer_port(li, int(port), FonctionES.ENTREE)
                return bool(li.entree(int(port)))
            except Exception as exc:
                _echouer("LECTURE DE L'ENTREE {} IMPOSSIBLE".format(port),
                         [str(exc)])
        return bool(self._lire(entree=int(port)).get("entree", False))

    def _lire(self, entree: Optional[int] = None) -> dict:
        charge: Dict[str, Any] = {"lire": True, "reel": self.robot_reel}
        if entree is not None:
            charge["entree"] = entree
        rep = demander({"commande": charge})
        if rep is None or "q" not in rep:
            _echouer("LE SIMULATEUR NE REPOND PAS",
                     ["fenetre fermee ? robot deconnecte ?"])
        return rep

    def _lire_q(self) -> np.ndarray:
        self._exiger_ouverte()
        if self.direct:
            try:
                return np.asarray(self._pont.angles_reels(), dtype=float)
            except Exception as exc:
                _echouer("LECTURE IMPOSSIBLE SUR LE ROBOT", [str(exc)])
        return np.asarray(self._lire()["q"], dtype=float)

    # -- mise en service --------------------------------------------------------

    def _connecter_direct(self, port: Optional[str]) -> None:
        """Cible 1 : le script ouvre le port lui-meme, sans fenetre."""
        from .pilote import _liaison_directe
        try:
            self._pont = _liaison_directe(self.modele, port)
        except RuntimeError as exc:
            _echouer("CONNEXION AU ROBOT IMPOSSIBLE", [str(exc)])
        self._pont.robot = self.modele
        self._pont.journal = self._journal_pont
        self._assurer_sans_defaut()
        try:
            # Les vitesses de Commande sont absolues, en mm/s et deg/s : le
            # ratio global ne doit pas les reduire en douce, sinon le robot
            # mettrait plus longtemps que ce qu'on a prevu et annonce.
            self._pont.definir_ratio(100.0)
        except Exception as exc:
            _echouer("REGLAGE DU ROBOT IMPOSSIBLE", [str(exc)])

    @staticmethod
    def _journal_pont(message: str) -> None:
        if not any(m in message for m in _JOURNAL_MUET):
            print("  [robot] " + message)

    def _assurer_sans_defaut(self) -> None:
        """Rien ne part tant que le robot signale un defaut."""
        from .liaison import resume_alarmes
        try:
            alarmes = self._pont.liaison.alarmes()
        except Exception as exc:
            _echouer("LECTURE DES ALARMES IMPOSSIBLE", [str(exc)])
        if not alarmes:
            return
        _signaler("robot en defaut : ouverture de l'assistant de mise en "
                  "route", resume_alarmes(alarmes))
        from .ui import mise_en_route_autonome
        if not mise_en_route_autonome(self._pont, self.modele):
            _echouer("MOUVEMENT REFUSE : le robot est toujours en defaut "
                     "apres la mise en route",
                     resume_alarmes(self._pont.liaison.alarmes()))

    def _assurer_simulateur(self, lancer: bool) -> None:
        """Cibles 2 et 3 : une fenetre doit repondre, sinon on la lance."""
        if demander({"commande": {"lire": True, "reel": False}}) is not None:
            return
        if demander({"ping": True}) is not None:
            _echouer("LE SIMULATEUR OUVERT NE CONNAIT PAS LA COMMANDE TEMPS "
                     "REEL", ["il date d'avant cette version : fermez-le, "
                              "puis relancez-le (python main.py)"])
        if not lancer:
            _echouer("AUCUN SIMULATEUR OUVERT",
                     ["lancez-le d'abord : python main.py",
                      "ou laissez le script l'ouvrir : "
                      "Commande(..., lancer_simulateur=True)"])
        self._lancer_simulateur()

    #: secondes laissees a une fenetre pour s'ouvrir (maillages a charger)
    DELAI_LANCEMENT = 60.0

    def _lancer_simulateur(self) -> None:
        """
        Ouvre main.py dans un AUTRE processus, puis attend sa passerelle.

        Pas de fenetre dans le processus du script : Qt veut le fil
        principal, et le script en a besoin pour attendre ses mouvements.
        Session a part : un Ctrl+C dans le terminal du script ne doit pas
        tuer la fenetre, ni la fin du script la fermer.
        """
        racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        options: Dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0))
        else:
            options["start_new_session"] = True
        print("ouverture du simulateur (main.py)...")
        try:
            proc = subprocess.Popen(
                [sys.executable, os.path.join(racine, "main.py")], cwd=racine,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, **options)
        except OSError as exc:
            _echouer("LE SIMULATEUR N'A PAS PU ETRE LANCE", [str(exc)])
        self.processus_simulateur = proc
        limite = time.time() + self.DELAI_LANCEMENT
        while time.time() < limite:
            if demander({"commande": {"lire": True, "reel": False}},
                        attente=300) is not None:
                return
            if proc.poll() is not None:
                _echouer("LE SIMULATEUR S'EST ARRETE AU DEMARRAGE",
                         ["code de sortie {}".format(proc.returncode),
                          "lancez-le a la main pour voir l'erreur : "
                          "python main.py"])
            time.sleep(0.3)
        _echouer("LE SIMULATEUR NE REPOND PAS",
                 ["lance depuis {:.0f} s, sa passerelle reste muette"
                  .format(self.DELAI_LANCEMENT)])

    def _connecter_fenetre(self, port: Optional[str]) -> None:
        """Cible 3 : la fenetre ouvre le port, et le garde."""
        rep = demander({"commande": {"connecter": {"port": port}}})
        if rep is None:
            _echouer("LE SIMULATEUR REFUSE DE SE CONNECTER AU ROBOT")
        self._suivre_fenetre(rep, "pret", limite=30.0)

    # -- garde-fous -------------------------------------------------------------

    def _verifier_cibles(self, prog: Programme, q0, pas_max: bool = True) -> None:
        """
        Chaque cible, AVANT de construire quoi que ce soit : atteignable,
        dans les butees, dans la zone, pas trop loin de la precedente.

        `pas_max=False` exempte du pas maximal : c'est le cas du repli,
        posture connue, qu'un pas de 30 mm rendrait inaccessible.

        Le MGI est appele SANS tolerance : le moteur de trajectoire, lui,
        ramene une cible hors d'atteinte au point le plus proche, et un MOVJ
        partirait alors ailleurs sans rien dire.
        """
        prec = np.asarray(self.modele.pose(q0)[:3], dtype=float)
        for k, ins in enumerate(prog.instructions):
            if ins.type not in TYPES_MOUVEMENT:
                continue
            nom = "{} {}".format(k + 1, ins.type)
            passages = []
            if ins.type == "MOVC":
                passages.append(("point de passage",
                                 ins.matrice_via(prog.repere_objet)))
            passages.append(("cible", ins.matrice(self.modele,
                                                  prog.repere_objet)))
            for quoi, T in passages:
                p = hg.position_de(T)
                # d'abord les regles du script, les plus lisibles : zone,
                # puis pas maximal ; ensuite seulement la mecanique
                hors = self._hors_zone(p)
                if hors:
                    _echouer("MOUVEMENT REFUSE : {} hors de la zone de "
                             "travail".format(quoi),
                             ["{} ({:.0f}, {:.0f}, {:.0f}) : {}".format(
                                 nom, *p, hors), self._texte_zone(),
                              "rien n'a ete envoye"])
            if self.pas_max is not None and pas_max:
                d = float(np.linalg.norm(p - prec))
                if d > self.pas_max + 1e-9:
                    _echouer("MOUVEMENT REFUSE : deplacement trop grand",
                             ["{} : {:.1f} mm depuis le point precedent, "
                              "pas_max = {:.1f} mm".format(nom, d, self.pas_max),
                              "rien n'a ete envoye"])
            for quoi, T in passages:
                p = hg.position_de(T)
                if ins.mode == "articulaire" and quoi == "cible":
                    q = np.asarray(ins.q, dtype=float)
                else:
                    try:
                        q = self.modele.mgi(T)
                    except ErreurCinematique as exc:
                        _echouer("MOUVEMENT REFUSE : {} hors d'atteinte"
                                 .format(quoi),
                                 [nom + " : " + str(exc),
                                  "rien n'a ete envoye"])
                d = self.modele.depassements(q)
                if d:
                    _echouer("MOUVEMENT REFUSE : {} hors des butees"
                             .format(quoi),
                             ["{} ({:.0f}, {:.0f}, {:.0f}) : {}".format(
                                 nom, *p, d[0])] + d[1:]
                             + ["rien n'a ete envoye"])
            prec = p

    def _hors_zone(self, p) -> str:
        if self.zone is None:
            return ""
        for nom, v, (a, b) in zip("xyz", p, self.zone):
            if not a - 1e-6 <= v <= b + 1e-6:
                return "{} = {:.1f} hors de [{:.0f} ; {:.0f}]".format(nom, v, a, b)
        return ""

    def _texte_zone(self) -> str:
        return "zone : x [{:.0f} ; {:.0f}], y [{:.0f} ; {:.0f}], z [{:.0f} ; " \
               "{:.0f}]".format(*(v for ab in self.zone for v in ab))

    def _controler(self, plan: Plan, q0) -> None:
        """Tout le chemin : butees, collisions, zone ; puis les reserves."""
        from .reel import BLOQUANTS_TELEVERSEMENT
        avertissements = plan.trajectoire.controler(0.05)
        defauts = [a for a in avertissements
                   if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
        if defauts:
            _echouer("MOUVEMENT REFUSE : {} defaut(s) sur le trajet, rien n'a "
                     "ete envoye".format(len(defauts)),
                     defauts[:8] + (["... et {} autre(s)".format(
                         len(defauts) - 8)] if len(defauts) > 8 else []))
        if self.zone is not None and len(plan.trajectoire):
            P = plan.trajectoire.echantillonner(0.02)["P"]
            if not self._hors_zone(P[0]):
                # Un depart HORS de la zone (repli, pose de la seance
                # precedente) n'est pas une faute : on ne juge alors que
                # les cibles. Parti de l'interieur, on n'en sort pas.
                for t, p in zip(plan.trajectoire.echantillonner(0.02)["t"], P):
                    hors = self._hors_zone(p)
                    if hors:
                        _echouer("MOUVEMENT REFUSE : le trajet sort de la "
                                 "zone de travail",
                                 ["a t = {:.2f} s, ({:.0f}, {:.0f}, {:.0f}) : "
                                  "{}".format(t, *p, hors), self._texte_zone(),
                                  "rien n'a ete envoye"])
        reserves = []
        for a in [x for x in avertissements if x not in defauts]:
            # une butee FROLEE se repete a chaque pas d'un deplacement au
            # clavier : on la dit une fois par seance, pas a chaque commande
            if "de sa butee" in a:
                cle = _SANS_NOMBRES.sub("#", a.split(":", 1)[-1])
                if cle in self._vus:
                    continue
                self._vus.add(cle)
            reserves.append(a)
        for a in plan.annonces:
            _signaler(a)
        if plan.avertissements:
            _signaler("duree ou vitesse non tenable : le robot fera au mieux",
                      plan.avertissements)
        if reserves:
            _signaler("le mouvement part, avec {} reserve(s)"
                      .format(len(reserves)), reserves[:5])

    # -- execution --------------------------------------------------------------

    def _nouveau_pilote(self, q0):
        """Un Pilote neuf, pour DECRIRE la commande avec ses verbes a lui."""
        from .pilote import Pilote
        p = Pilote(robot=self.modele, nom="commande", vitesse=self.vitesse,
                   acceleration=self.acceleration,
                   vitesse_articulaire=self.vitesse_articulaire,
                   acceleration_articulaire=self.acceleration_articulaire,
                   lissage=self.lissage, profil="trapeze", depart=q0,
                   outil=self.outil)
        p.regler_longueur_outil(self.longueur_outil)
        p._pose = list(self._pose)
        return p

    def _executer(self, decrire: Callable, continu: Optional[bool],
                  pas_max: bool = True) -> "Commande":
        """
        Le chemin commun : lire, decrire, controler, planifier, executer,
        attendre. La cible memorisee n'avance qu'une fois le mouvement fait :
        un refus laisse le script la ou il en etait.
        """
        self._exiger_ouverte()
        q0 = self._lire_q()
        p = self._nouveau_pilote(q0)
        decrire(p)
        prog = p.programme
        if not prog.instructions:
            return self
        self._verifier_cibles(prog, q0, pas_max)
        plan = planifier(prog, self.modele, q0,
                         continu=self.continu if continu is None else continu,
                         acceleration=self.acceleration,
                         acceleration_articulaire=self.acceleration_articulaire,
                         vitesse=self.vitesse)
        self._controler(plan, q0)
        with self._garde():
            if self.direct:
                self._executer_robot(plan)
            else:
                self._executer_fenetre(plan, q0)
        if self.robot_reel:
            self._verifier_arrivee(plan)
        self._pose = list(p._pose)
        return self

    def _verifier_arrivee(self, plan: Plan) -> None:
        """Le robot s'est arrete : est-ce bien la ou on l'attendait ?"""
        q_fin = np.asarray(plan.trajectoire.q(plan.duree), dtype=float)
        attendu = self.modele.pose(q_fin)[:3]
        arrive = self.modele.pose(self._lire_q())[:3]
        ecart = float(np.linalg.norm(np.asarray(arrive) - np.asarray(attendu)))
        if ecart > 2.0:
            _signaler("le robot s'est arrete a {:.1f} mm de la cible".format(ecart),
                      ["attendu ({:.1f}, {:.1f}, {:.1f}), lu ({:.1f}, {:.1f}, "
                       "{:.1f})".format(*attendu, *arrive)])

    @contextmanager
    def _garde(self):
        """
        Ctrl+C -- et le bouton stop de VS Code, qui envoie SIGTERM -- coupe
        le mouvement AVANT que le script ne s'arrete.
        """
        def _interrompre(*_a):
            raise KeyboardInterrupt
        ancien = None
        if hasattr(signal, "SIGTERM"):
            try:
                ancien = signal.signal(signal.SIGTERM, _interrompre)
            except ValueError:               # pas dans le fil principal
                ancien = None
        try:
            yield
        except KeyboardInterrupt:
            self._arret_urgence()
            raise
        finally:
            if ancien is not None:
                signal.signal(signal.SIGTERM, ancien)

    def _arret_urgence(self) -> None:
        if self.direct:
            try:
                self._pont.arret_urgence()
            except Exception:
                pass
            _signaler_urgence()
            return
        demander({"commande": {"arret_urgence": True,
                                "reel": self.robot_reel}})
        if self.robot_reel:
            _signaler_urgence()
        else:
            _alerter("SIMULATION INTERROMPUE (Ctrl+C ou bouton stop)")

    def _executer_robot(self, plan: Plan) -> None:
        """Cible 1 : file du controleur remplie et suivie par le script."""
        pont = self._pont
        self._assurer_sans_defaut()
        try:
            pont.televerser_commandes(lambda: commandes_robot(
                pont, plan.programme, plan.etapes, self.outil))
            pont.demarrer()
        except Exception as exc:
            try:
                pont.arret_urgence()
            except Exception:
                pass
            _echouer("ENVOI AU ROBOT IMPOSSIBLE", [str(exc)])
        self._attendre_robot(pont, plan.duree)

    #: lectures successives sans mouvement exigees pour conclure a l'arret
    IMMOBILE = 3

    def _attendre_robot(self, pont, duree: float) -> None:
        """
        Attend la fin : derniere commande atteinte ET bras immobile.

        L'index du controleur designe la commande EN COURS : atteindre la
        derniere ne prouve pas qu'elle soit finie. On exige donc en plus
        quelques lectures successives sans mouvement -- sans quoi la
        commande suivante, qui commence par couper la file, tronquerait la
        fin de celle-ci.
        """
        limite = time.time() + 3.0 * duree + 15.0
        q_prec = None
        immobile = 0
        while True:
            try:
                etat = pont.etat()
                fini = pont.termine()
            except Exception as exc:
                _echouer("LIAISON PERDUE PENDANT LE MOUVEMENT", [str(exc)])
            if etat.alarmes:
                _echouer("ROBOT EN DEFAUT : mouvement interrompu", etat.alarmes)
            incident = getattr(pont, "_incident", "")
            if incident:
                _echouer("ALIMENTATION DE LA FILE INTERROMPUE", [incident])
            q = np.asarray(etat.angles, dtype=float)
            if q_prec is not None and float(np.max(np.abs(q - q_prec))) < 0.01:
                immobile += 1
            else:
                immobile = 0
            q_prec = q
            if fini and immobile >= self.IMMOBILE:
                return
            if time.time() > limite:
                try:
                    pont.arret_urgence()
                except Exception:
                    pass
                _echouer("LE MOUVEMENT NE SE TERMINE PAS : arret d'urgence",
                         ["{:.1f} s prevues, rien de fini apres {:.0f} s"
                          .format(duree, 3.0 * duree + 15.0)])
            time.sleep(0.05)

    def _executer_fenetre(self, plan: Plan, q0) -> None:
        """Cibles 2 et 3 : la fenetre anime, ou execute sur le robot."""
        rep = demander({"commande": {"executer": {
            "programme": plan.programme.vers_dict(),
            "q_depart": [float(v) for v in q0],
            "etapes": plan.etapes,
            "reel": self.robot_reel}}})
        if rep is None:
            _echouer("LE SIMULATEUR A REFUSE LA COMMANDE",
                     ["robot deconnecte de la fenetre ?" if self.robot_reel
                      else "fenetre fermee ?"])
        self._suivre_fenetre(rep, "termine", limite=(3.0 * plan.duree + 20.0
                                                    if self.robot_reel else None))

    def _suivre_fenetre(self, etat: dict, attendu: str,
                        limite: Optional[float] = None) -> dict:
        """
        Interroge la fenetre jusqu'a l'etat `attendu`. Aucun affichage de
        progression ; seule la mise en route, qui attend l'operateur, est
        annoncee -- une fois.
        """
        signale = False
        echeance = None if limite is None else time.time() + limite
        while True:
            quoi = (etat or {}).get("etat")
            if quoi == attendu:
                return etat
            if quoi == "refuse":
                _echouer("MOUVEMENT INTERROMPU : {}".format(
                    etat.get("raison", "?")), etat.get("alarmes", []))
            if quoi == "mise_en_route":
                if not signale:
                    print("  robot en defaut : terminez la MISE EN ROUTE dans "
                          "la fenetre du simulateur")
                    signale = True
                if limite is not None:
                    echeance = time.time() + limite   # l'operateur a la main
            elif echeance is not None and time.time() > echeance:
                demander({"commande": {"arret_urgence": True,
                                "reel": self.robot_reel}})
                _echouer("LE SIMULATEUR NE TERMINE PAS : arret d'urgence "
                         "demande", ["etat '{}' depuis plus de {:.0f} s"
                                     .format(quoi, limite)])
            time.sleep(0.05)
            etat = demander({"commande": {"etat": True}})
            if etat is None:
                _echouer("LE SIMULATEUR NE REPOND PLUS",
                         ["fenetre fermee pendant le mouvement ?"])

    # -- cibles ---------------------------------------------------------------

    @staticmethod
    def _est_liste(T) -> bool:
        """Une LISTE de cibles, et non une cible ?"""
        if T is None or isinstance(T, np.ndarray):
            return False
        if not isinstance(T, (list, tuple)) or not T:
            return False
        if all(_scalaire(v) for v in T):
            return False                          # des coordonnees
        try:
            M = np.asarray(T, dtype=float)
            if M.shape == (4, 4) and np.allclose(M[3], (0, 0, 0, 1)):
                return False                      # une matrice en listes
        except (ValueError, TypeError):
            pass
        return True

    @staticmethod
    def _element(e) -> Tuple[Any, Optional[float]]:
        """cible, ou (cible, duree) -> (cible, duree ou None)."""
        if (isinstance(e, (list, tuple)) and len(e) == 2 and _scalaire(e[1])
                and not _scalaire(e[0])):
            return e[0], float(e[1])
        return e, None

    @staticmethod
    def _est_liste_arcs(via) -> bool:
        if not isinstance(via, (list, tuple)) or not via:
            return False
        return all(isinstance(e, (list, tuple)) and len(e) in (2, 3)
                   and not _scalaire(e[0]) and not _scalaire(e[1])
                   for e in via)

    # -- mouvements -------------------------------------------------------------

    def movj(self, T=None, *, x=None, y=None, z=None, r=None, q=None,
             duree=None, vitesse=None, lissage=None, continu=None,
             repere=None, commentaire="") -> "Commande":
        """
        Mouvement articulaire ; rend la main a l'arrivee.

            robot.movj(PRISE)                     une cible (matrice 4x4)
            robot.movj(x=220, y=-50, z=80)        des coordonnees
            robot.movj(q=[0, 15, 30, 0])          des angles
            robot.movj([A, (B, 2.0), C])          plusieurs cibles

        Plusieurs cibles s'enchainent sans arret (`continu=True`) : le
        chemin est celui du MOVJ, suivi en trajectoire continue.
        """
        if self._est_liste(T):
            if any(v is not None for v in (x, y, z, r, q)):
                raise ValueError("movj : x, y, z, r ou q ne vont pas avec une "
                                 "liste de cibles")
            elements = [self._element(e) for e in T]

            def decrire(p):
                for c, d in elements:
                    p.movj(c, duree=d if d is not None else duree,
                           vitesse=vitesse, lissage=lissage, repere=repere,
                           commentaire=commentaire)
            return self._executer(decrire, continu)
        return self._executer(lambda p: p.movj(
            T, x=x, y=y, z=z, r=r, q=q, duree=duree, vitesse=vitesse,
            repere=repere, commentaire=commentaire), continu)

    def movl(self, T=None, *, x=None, y=None, z=None, r=None, duree=None,
             vitesse=None, lissage=None, continu=None, repere=None,
             commentaire="") -> "Commande":
        """
        Mouvement rectiligne ; rend la main a l'arrivee.

            robot.movl(PRISE)
            robot.movl(z=40)                                descente verticale
            robot.movl([(A, 1.0), (B, 0.5), C])             cible, duree en s
            robot.movl([A, B, C], lissage=50)               coins arrondis

        Une duree omise se deduit de la vitesse. `continu=True` (defaut)
        enchaine les cibles SANS arret ; `continu=False` s'arrete a chacune.
        """
        if self._est_liste(T):
            if any(v is not None for v in (x, y, z, r)):
                raise ValueError("movl : x, y, z ou r ne vont pas avec une "
                                 "liste de cibles")
            elements = [self._element(e) for e in T]

            def decrire(p):
                for c, d in elements:
                    p.movl(c, duree=d if d is not None else duree,
                           vitesse=vitesse, lissage=lissage, repere=repere,
                           commentaire=commentaire)
            return self._executer(decrire, continu)
        return self._executer(lambda p: p.movl(
            T, x=x, y=y, z=z, r=r, duree=duree, vitesse=vitesse,
            repere=repere, commentaire=commentaire), continu)

    def movc(self, via, T=None, *, x=None, y=None, z=None, r=None,
             duree=None, vitesse=None, lissage=None, continu=None,
             repere=None, commentaire="") -> "Commande":
        """
        Arc de cercle passant par `via` ; rend la main a l'arrivee.

            robot.movc(cible(260, 0, 80), cible(220, 40, 80))
            robot.movc([(via1, fin1, 1.5), (via2, fin2)])    plusieurs arcs

        L'arc est suivi en trajectoire continue, ce qui en maitrise la
        vitesse ; s'il fait tourner l'outil, il passe par l'arc du
        controleur (annonce).
        """
        if T is None and self._est_liste_arcs(via):
            if any(v is not None for v in (x, y, z, r)):
                raise ValueError("movc : x, y, z ou r ne vont pas avec une "
                                 "liste d'arcs")
            arcs = [(e[0], e[1], float(e[2]) if len(e) == 3 else None)
                    for e in via]

            def decrire(p):
                for v, fin, d in arcs:
                    p.movc(v, fin, duree=d if d is not None else duree,
                           vitesse=vitesse, lissage=lissage, repere=repere,
                           commentaire=commentaire)
            return self._executer(decrire, continu)
        return self._executer(lambda p: p.movc(
            via, T, x=x, y=y, z=z, r=r, duree=duree, vitesse=vitesse,
            repere=repere, commentaire=commentaire), continu)

    def repli(self, duree=None, vitesse=None) -> "Commande":
        """
        Retour a la position de repos J = (0, 15, 30, 0) deg.

        Le repli n'est PAS soumis a `pas_max` : c'est une posture connue,
        et un pas de 30 mm la rendrait inaccessible. La zone, les butees et
        les collisions, elles, s'appliquent comme partout.
        """
        q = list(self.modele.q_home)
        return self._executer(lambda p: p.movj(q=q, duree=duree,
                                               vitesse=vitesse,
                                               commentaire="repli"),
                              None, pas_max=False)

    # -- outil, entrees / sorties, attente -------------------------------------

    def actionner(self, actif: bool = True) -> "Commande":
        """Commande l'outil monte (ventouse, pince, laser) ; attend qu'il ait agi."""
        # une nouvelle action sur l'outil annule la coupure prevue : couper
        # la pompe d'une pince qu'on vient de refermer lui ferait lacher
        # sa piece
        self._annuler_coupure()
        return self._executer(lambda p: p.actionner(bool(actif)), False)

    def ventouse(self, actif: bool = True, couper: bool = False) -> "Commande":
        """
        Aspire (True) ou relache (False). Relacher laisse la pompe tourner
        (elle souffle) : `couper=True` l'arrete DELAI_COUPURE_POMPE (1 s)
        apres, sans bloquer le script.
        """
        self.actionner(actif)
        if couper and not actif:
            self._programmer_coupure()
        return self

    def pince(self, fermee: bool = True, couper: bool = True) -> "Commande":
        """
        Ferme (True) ou ouvre (False) la pince.

        Ouvrir fait SOUFFLER la pompe, qui continuerait de tourner : elle
        est donc coupee DELAI_COUPURE_POMPE (1 s) apres l'ouverture, le
        temps que la pince s'ouvre. Le script, lui, n'attend pas : il peut
        enchainer un mouvement pendant cette seconde. `couper=False` laisse
        la pompe en marche.
        """
        self.actionner(fermee)
        if couper and not fermee:
            self._programmer_coupure()
        return self

    def _programmer_coupure(self) -> None:
        """Coupe la pompe dans DELAI_COUPURE_POMPE s, sans bloquer."""
        self._annuler_coupure()
        if not self.robot_reel:
            return                      # le simulateur n'a pas de pompe
        minuterie = threading.Timer(DELAI_COUPURE_POMPE, self._couper_en_differe)
        minuterie.daemon = True
        self._coupure = (minuterie, time.time() + DELAI_COUPURE_POMPE)
        minuterie.start()

    def _annuler_coupure(self) -> None:
        if self._coupure is not None:
            self._coupure[0].cancel()
            self._coupure = None

    def _couper_en_differe(self) -> None:
        """Appele par la minuterie, hors du fil du script : ne leve jamais."""
        self._coupure = None
        if self._ferme:
            return
        try:
            self.couper_pompe()
        except BaseException as exc:            # pragma: no cover
            _signaler("la pompe n'a pas pu etre coupee apres l'ouverture",
                      [str(exc)])

    def couper_pompe(self) -> "Commande":
        """
        Arrete la pompe a air, tout de suite (ventouse comme pince).

        Une pince fermee ou une ventouse qui aspire LACHE alors sa piece :
        a n'appeler qu'une fois la piece posee. Sans robot reel, ne fait
        rien : le simulateur n'a pas de pompe.
        """
        self._exiger_ouverte()
        if threading.current_thread() is threading.main_thread():
            self._annuler_coupure()
        if not self.robot_reel:
            return self
        try:
            if self.direct:
                self._pont.liaison.couper_pompe()
            elif demander({"commande": {"couper_pompe": True}}) is None:
                raise RuntimeError("le simulateur ne repond pas")
        except Exception as exc:
            _echouer("ARRET DE LA POMPE IMPOSSIBLE", [str(exc)])
        return self

    def sortie(self, port: int, etat: bool = True) -> "Commande":
        """Met la sortie EIO`port` a l'etat demande."""
        return self._executer(lambda p: p.sortie(int(port), bool(etat)), False)

    def attendre(self, secondes: float) -> "Commande":
        """
        Attend sans bouger. Rien ne part vers le robot : le script dort,
        et Ctrl+C l'interrompt comme d'habitude.
        """
        self._exiger_ouverte()
        time.sleep(max(float(secondes), 0.0))
        return self
