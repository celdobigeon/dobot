# -*- coding: utf-8 -*-
"""
Etalonnage camera -> robot par la PINCE seule : aucune piece a poser.

    .venv/bin/python dobot.py etalonner-pince
    .venv/bin/python examples/perception/etalonner_par_pince.py --longueur-outil 107

La camera regarde la pince que le robot promene devant elle, sous
plusieurs angles. Le modele CAO officiel de la pince (assets/pince_dobot.npz)
est recale sur toutes ces vues a la fois : c'est lui la mire.

  1. DECOR : bras en equerre, une capture de profondeur. Tout ce qui sera
     plus pres que ce decor, ensuite, est le bras. Le PLAN DE LA TABLE, vu
     sur cette capture, donne d'emblee l'inclinaison et la hauteur de la
     camera : il ne reste que 4 inconnues (rotation autour de la verticale,
     x, y, et la hauteur de la table dans le repere du robot).
  2. AMORCAGE, prudent : quelques poses juste devant le robot (base a 0),
     pince tournee de -45 a +45 deg. Premier recalage.
  3. COUVERTURE : la pince va la ou la camera la voit -- toute la zone vue,
     deux hauteurs, trois orientations --, loin de la camera, avec de larges
     marges aux butees (l'alarme 0x12 bloque le controleur). Recalage final.
  4. CONTROLE : le recalage refait sur deux moities des vues ; l'ecart entre
     les deux, sur la zone de travail, mesure sa repetabilite.

Le resultat est ecrit dans etalonnage_camera_robot.json (l'ancien est garde
a cote), que les scripts de saisie lisent. Chaque vue est archivee dans
apprentissage/dataset/etalonnage_pince/<date>/ (images d'entrainement).

Pour chaque vue, la pince est isolee par difference avec le decor, et on ne
garde que ce qui tourne AVEC l'outil -- entre le bout des doigts et le haut
du modele, 77 mm au-dessus : l'avant-bras, lui, ne tourne pas avec la pince
et fausserait tout. Le tuyau souple, fixe en haut du corps, est hors de
cette tranche pour l'essentiel.

Personne ne doit passer dans le champ de la camera pendant l'etalonnage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
for _d in (ICI, RACINE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import etalonner_camera_robot as ec     # noqa: E402

MODELE = os.path.join(RACINE, "assets", "pince_dobot.npz")
ARCHIVE = os.path.join(ICI, "apprentissage", "dataset", "etalonnage_pince")

#: images par capture (mediane : moins de bruit de profondeur)
IMAGES = 15
#: m : un pixel appartient au bras s'il est au moins aussi loin devant le decor
SEUIL_DECOR = 0.025
#: deg : marge aux butees de J1, J2, J3, et rotation de l'outil (J4) bornee
MARGE_BUTEE = 8.0
J4_MAX = 60.0
#: deg : marge sur l'angle bras arriere / avant-bras (Limites.couplage)
MARGE_COUPLAGE = 4.0
#: mm : on ne s'approche pas plus de la camera (distance horizontale)
DISTANCE_CAMERA = 200.0
#: deg : equerre de degagement (bras arriere a peine penche)
EQUERRE_J2 = 6.0


# ---------------------------------------------------------------------------
# Geometrie
# ---------------------------------------------------------------------------

def rot_z(deg: float) -> np.ndarray:
    return ec.rot_z(deg)


def pose_outil(x, y, z, r) -> np.ndarray:
    A = np.eye(4)
    A[:3, :3] = rot_z(r)
    A[:3, 3] = [x, y, z]
    return A


def charger_modele(pas: float = 1.5) -> np.ndarray:
    """Points de surface de la pince, repere de l'outil (mm), un par voxel."""
    with np.load(MODELE) as z:
        V = z["sommets"].astype(float)
    return V[np.unique(np.floor(V / pas).astype(np.int64), axis=0, return_index=True)[1]]


def kabsch2(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
    """Rotation (autour de z) et translation 2D amenant a sur b."""
    a0, b0 = a.mean(0), b.mean(0)
    H = (a - a0).T @ (b - b0)
    th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return th, b0 - R @ a0


@dataclass
class Vue:
    nom: str
    points: np.ndarray        # mm, repere de la TABLE (z = hauteur au-dessus)
    A: np.ndarray             # pose de l'outil (robot), 4x4 mm


@dataclass
class Solution:
    theta: float              # rad : rotation table -> robot autour de z
    t: np.ndarray             # (tx, ty, tz) mm ; tz = hauteur de la table (robot)
    distance: float = float("inf")   # mm, mediane scan -> modele

    def M(self) -> np.ndarray:
        """Repere table -> repere robot."""
        M = np.eye(4)
        M[:3, :3] = rot_z(math.degrees(self.theta))
        M[:3, 3] = self.t
        return M


# ---------------------------------------------------------------------------
# Recalage du modele sur les vues
# ---------------------------------------------------------------------------

class Recaleur:
    """Recale le modele de la pince sur des vues (repere table)."""

    def __init__(self, modele: np.ndarray, haut: Optional[float] = None):
        import torch
        self.torch = torch
        self.modele = modele
        self.haut = float(haut if haut is not None else modele[:, 2].max())
        self.mod_t = torch.tensor(modele, dtype=torch.float32)

    def tranche(self, v: Vue, tz: float) -> np.ndarray:
        """Les points de la vue entre le bout des doigts et le haut du modele."""
        h = v.points[:, 2] + tz - v.A[2, 3]
        return v.points[(h > -3.0) & (h < self.haut - 2.0)]

    def ajuster(self, vues: Sequence[Vue], s: Solution, iters: int = 150,
                garder: float = 0.75) -> Solution:
        torch = self.torch
        theta, t = s.theta, np.asarray(s.t, float).copy()
        dist = np.array([np.inf])
        for _ in range(iters):
            Rw = rot_z(math.degrees(theta))
            src, cib, dd = [], [], []
            for v in vues:
                P = self.tranche(v, t[2])
                if len(P) < 30:
                    continue
                W = P @ Rw.T + t
                U = (W - v.A[:3, 3]) @ v.A[:3, :3]
                d, k = torch.cdist(torch.tensor(U, dtype=torch.float32), self.mod_t).min(dim=1)
                m = self.modele[k.numpy()]
                src.append(P)
                cib.append(m @ v.A[:3, :3].T + v.A[:3, 3])
                dd.append(d.numpy())
            if not src:
                break
            P, Wc, dist = np.vstack(src), np.vstack(cib), np.concatenate(dd)
            ok = dist <= np.percentile(dist, 100 * garder)
            th2, txy = kabsch2(P[ok, :2], Wc[ok, :2])
            Rw2 = rot_z(math.degrees(th2))
            tz2 = float(np.median((Wc[ok] - P[ok] @ Rw2.T)[:, 2]))
            t2 = np.array([txy[0], txy[1], tz2])
            pas = abs(th2 - theta) * 300.0 + float(np.linalg.norm(t2 - t))
            theta, t = th2, t2
            if pas < 0.01:
                break
        return Solution(theta, t, float(np.median(dist)) if len(dist) else float("inf"))

    def ajuster_departs(self, vues, s: Solution, departs: int = 8, graine: int = 0,
                        ecart_angle: float = 10.0, ecart_xy: float = 40.0) -> Solution:
        """Plusieurs departs autour de s ; la meilleure solution l'emporte."""
        rng = np.random.default_rng(graine)
        candidats = [s] + [Solution(s.theta + math.radians(rng.uniform(-ecart_angle, ecart_angle)),
                                    np.asarray(s.t, float) + np.r_[rng.uniform(-ecart_xy, ecart_xy, 2),
                                                                   rng.uniform(-10, 10)])
                           for _ in range(departs)]
        meilleur = None
        for c in candidats:
            r = self.ajuster(vues, c, iters=80)
            if meilleur is None or r.distance < meilleur.distance:
                meilleur = r
        return self.ajuster(vues, meilleur, iters=200)


def amorce_par_centres(vues: Sequence[Vue], tz: float = 0.0) -> Solution:
    """
    Premiere estimation, sans rien savoir : le centre de chaque nuage, un
    point inconnu mais fixe de la pince, suit l'outil. Kabsch 2D avec ce
    point (2 inconnues de plus), en alternance.
    """
    C = np.array([np.median(v.points, axis=0) for v in vues])
    c = np.zeros(2)
    th, txy = 0.0, np.zeros(2)
    for _ in range(100):
        Q = np.array([v.A[:2, :2] @ c + v.A[:2, 3] for v in vues])
        th, txy = kabsch2(C[:, :2], Q)
        R = rot_z(math.degrees(th))[:2, :2]
        e = C[:, :2] @ R.T + txy
        c = np.mean([v.A[:2, :2].T @ (e[i] - v.A[:2, 3]) for i, v in enumerate(vues)], axis=0)
    return Solution(th, np.array([txy[0], txy[1], tz]))


def ecart_zone(T1: np.ndarray, T2: np.ndarray, zone=None) -> Tuple[float, float]:
    """Ecart moyen et maximal (mm) entre deux recalages camera -> robot, sur
    les points de la zone de travail."""
    if zone is None:
        zone = [[x, y, z] for x in (180, 240, 300) for y in (-120, 0, 120) for z in (0, 50)]
    e = [float(np.linalg.norm(T1[:3, :3] @ np.linalg.solve(T2[:3, :3], np.array(p) - T2[:3, 3])
                              + T1[:3, 3] - p)) for p in zone]
    return float(np.mean(e)), float(max(e))


# ---------------------------------------------------------------------------
# Les vues : segmentation de la pince
# ---------------------------------------------------------------------------

class Segmenteur:
    """La pince dans une capture, par difference avec le decor."""

    def __init__(self, decor: dict):
        import perception
        self.fond = np.asarray(decor["profondeur"], float)
        self.K = np.asarray(decor["K"], float)
        self.taille = self.fond.shape[::-1]
        table = perception.analyser_table(self.fond, self.K)
        T_ct = np.asarray(table.T_ct, float).copy()
        T_ct[:3, 3] *= 1000.0
        self.T_ct = T_ct                       # table -> camera, mm
        self.T_tc = np.linalg.inv(T_ct)

    @property
    def hauteur_camera(self) -> float:
        return float(self.T_tc[2, 3])

    @property
    def plongee(self) -> float:
        return float(math.degrees(math.asin(max(-1.0, min(1.0, -self.T_tc[2, 2])))))

    def points(self, capture: dict, attendu_table: Optional[np.ndarray] = None) -> np.ndarray:
        """Points du bras (mm, repere TABLE). `attendu_table` : ou la pince
        devrait etre ; seuls les morceaux proches sont gardes."""
        import cv2
        z = np.asarray(capture["profondeur"], float)
        m = ((z > 0) & (self.fond > 0) & (self.fond - z > SEUIL_DECOR)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        K = self.K

        def vers_table(mask):
            v, u = np.nonzero(mask)
            zz = z[v, u]
            X = np.c_[(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz] * 1000.0
            return X @ self.T_tc[:3, :3].T + self.T_tc[:3, 3]

        morceaux = [k for k in range(1, n) if st[k, 4] > 150]
        if not morceaux:
            return np.zeros((0, 3))
        if attendu_table is not None:
            garde = []
            for k in morceaux:
                X = vers_table(lab == k)
                if np.linalg.norm(np.median(X, axis=0) - attendu_table) < 140.0:
                    garde.append(X)
            X = np.vstack(garde) if garde else np.zeros((0, 3))
        else:
            # sans estimation : le morceau le plus proche de la camera, et
            # ceux a la meme distance (la pince et son tuyau)
            meds = [(float(np.median(z[lab == k])), k) for k in morceaux]
            zref = min(meds)[0]
            X = np.vstack([vers_table(lab == k) for zm, k in meds if abs(zm - zref) < 0.08])
        if len(X):
            X = X[np.linalg.norm(X - np.median(X, axis=0), axis=1) < 160.0]
            X = X[np.unique(np.floor(X / 2.0).astype(np.int64), axis=0, return_index=True)[1]]
        return X


# ---------------------------------------------------------------------------
# Le deroule
# ---------------------------------------------------------------------------

class EtalonnagePince:

    def __init__(self, robot, capturer: Callable[[], dict], journal=print,
                 archive: Optional[str] = ARCHIVE, graine: int = 0):
        from dobot_sim import DobotMagician
        self.robot = robot
        self.capturer = capturer
        self.journal = journal
        self.modele_robot = getattr(robot, "modele", None) or DobotMagician()
        self.recaleur = Recaleur(charger_modele())
        self.vues: List[Vue] = []
        self.seg: Optional[Segmenteur] = None
        self.solution: Optional[Solution] = None
        self.rng = np.random.default_rng(graine)
        self.archive = None
        if archive:
            self.archive = os.path.join(archive, time.strftime("%Y%m%d_%H%M%S"))
        self.decor_capture = None

    # -- robot --------------------------------------------------------------
    def permis(self, x, y, z, r) -> bool:
        """Atteignable avec marges, outil peu tourne, loin de la camera."""
        try:
            q = self.modele_robot.mgi_pose(x, y, z, r)
        except Exception:                              # noqa: BLE001
            return False
        if not self.modele_robot.dans_butees(q):
            return False
        if float(np.min(self.modele_robot.marges(q)[:3])) < MARGE_BUTEE or abs(q[3]) > J4_MAX:
            return False
        couplage = self.modele_robot.lim.couplage
        if couplage is not None and (q[2] - q[1]) < couplage[0] + MARGE_COUPLAGE:
            return False
        if self.solution is not None:
            cam = self.T_robot_camera()[:3, 3]
            if math.hypot(x - cam[0], y - cam[1]) < DISTANCE_CAMERA:
                return False
        return True

    def equerre(self):
        q = [float(v) for v in self.robot.angles()]
        self.robot.movj(q=[q[0], EQUERRE_J2, 0.0, q[3]])

    def T_robot_camera(self) -> np.ndarray:
        """Camera -> robot (mm), avec la solution en cours."""
        return self.solution.M() @ self.seg.T_tc

    # -- captures -----------------------------------------------------------
    def _archiver(self, nom, capture, meta):
        if self.archive is None:
            return
        try:
            os.makedirs(self.archive, exist_ok=True)
            np.savez_compressed(os.path.join(self.archive, nom + ".npz"), **capture)
            with open(os.path.join(self.archive, nom + ".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
        except Exception as exc:                       # noqa: BLE001
            self.journal("  (capture non archivee : {})".format(exc))

    def relever_decor(self):
        self.equerre()
        time.sleep(0.5)
        cap = self.capturer()
        self.decor_capture = cap
        self.seg = Segmenteur(cap)
        self._archiver("00_decor", cap, {"pose": [float(v) for v in self.robot.pose()]})
        self.journal("decor releve ; camera a {:.0f} mm au-dessus de la table, plongee {:.1f} "
                     "deg".format(self.seg.hauteur_camera, self.seg.plongee))

    def voir(self, x, y, z, r) -> Optional[Vue]:
        self.robot.movj(x=x, y=y, z=z, r=r)
        time.sleep(0.3)
        cap = self.capturer()
        xp, yp, zp, rp = (float(v) for v in self.robot.pose())
        A = pose_outil(xp, yp, zp, rp)
        attendu = None
        if self.solution is not None:
            # le haut du corps de la pince, ramene dans le repere table
            p = A[:3, :3] @ [0.0, 0.0, 45.0] + A[:3, 3]
            attendu = np.linalg.solve(self.solution.M(), np.r_[p, 1.0])[:3]
        P = self.seg.points(cap, attendu)
        nom = "v{:03d}".format(len(self.vues) + 1)
        self._archiver(nom, cap, {"pose": [xp, yp, zp, rp], "points": int(len(P))})
        if len(P) < 60:
            self.journal("  {}  x {:6.1f} y {:6.1f} z {:5.1f} r {:+5.0f}  : pince NON VUE".format(
                nom, xp, yp, zp, rp))
            return None
        v = Vue(nom, P, A)
        self.vues.append(v)
        self.journal("  {}  x {:6.1f} y {:6.1f} z {:5.1f} r {:+5.0f}  : {} points".format(
            nom, xp, yp, zp, rp, len(P)))
        return v

    # -- etapes -------------------------------------------------------------
    def amorcer(self) -> Solution:
        """Poses juste devant le robot (base a 0), sans tourner la base."""
        self.journal("AMORCAGE : quelques poses devant le robot")
        poses = [(x, 0.0, z, r) for x in (200.0, 250.0) for z in (80.0, 130.0)
                 for r in (-45.0, 0.0, 45.0)]
        for p in poses:
            if self.permis(*p):
                self.voir(*p)
        if len(self.vues) < 5:
            raise ec.Arret("la camera ne voit pas la pince devant le robot ({} vue(s)) : "
                           "orientez-la vers la zone de travail".format(len(self.vues)))
        s = amorce_par_centres(self.vues)
        self.solution = self.recaleur.ajuster_departs(self.vues, s, departs=12, ecart_angle=25,
                                                      ecart_xy=80)
        self.journal("  premier recalage : {:.2f} mm (mediane modele -> mesure)".format(
            self.solution.distance))
        return self.solution

    def visible(self, x, y, z, marge=45) -> bool:
        T = self.T_robot_camera()
        K, (W, H) = self.seg.K, self.seg.taille
        for dz in (5.0, 60.0):
            p = np.linalg.solve(T[:3, :3], np.array([x, y, z + dz]) - T[:3, 3])
            if p[2] < 250.0:
                return False
            u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
            if not (marge < u < W - marge and marge < v < H - marge):
                return False
        return True

    def couvrir(self, n_max: int = 45):
        """La pince sur toute la zone vue, deux hauteurs, trois orientations."""
        tz = float(self.solution.t[2])
        candidats = []
        for x in np.arange(160.0, 321.0, 30.0):
            for y in np.arange(-210.0, 211.0, 35.0):
                for zh in (60.0, 110.0):
                    z = tz + zh
                    if not self.visible(x, y, z):
                        continue
                    for r in (-60.0, 0.0, 60.0):
                        r_abs = math.degrees(math.atan2(y, x)) + r      # outil tourne de r
                        if self.permis(x, y, z, r_abs):
                            candidats.append((float(x), float(y), z, r_abs))
        if not candidats:
            raise ec.Arret("aucune pose a la fois vue et atteinte")
        # les plus etalees possible (plus loin des deja choisies d'abord)
        choisis = [candidats[int(self.rng.integers(len(candidats)))]]
        while len(choisis) < min(n_max, len(candidats)):
            C = np.array(candidats)
            S = np.array(choisis)
            d = np.min(np.linalg.norm((C[:, None, :3] - S[None, :, :3]) * [1, 1, 2], axis=2)
                       + 0.3 * np.abs(C[:, None, 3] - S[None, :, 3]), axis=1)
            choisis.append(candidats[int(np.argmax(d))])
        # trajet : par angle de base, pour des mouvements courts
        choisis.sort(key=lambda p: (round(math.degrees(math.atan2(p[1], p[0])) / 15), p[3]))
        self.journal("COUVERTURE : {} poses dans la zone vue".format(len(choisis)))
        for p in choisis:
            self.voir(*p)
        self.solution = self.recaleur.ajuster_departs(self.vues, self.solution, departs=6)
        self.journal("  recalage : {:.2f} mm (mediane modele -> mesure), sur {} vues".format(
            self.solution.distance, len(self.vues)))

    def stabilite(self) -> Tuple[float, float]:
        """Recalage refait sur deux moitiees des vues : ecart sur la zone."""
        moities = []
        for k in (0, 1):
            s = self.recaleur.ajuster_departs(self.vues[k::2], self.solution, departs=4)
            moities.append(s.M() @ self.seg.T_tc)
        return ecart_zone(moities[0], moities[1])

    def secteur(self) -> Tuple[float, float]:
        """Le secteur de base parcouru, elargi de 3 deg (pour la saisie)."""
        a = [math.degrees(math.atan2(v.A[1, 3], v.A[0, 3])) for v in self.vues]
        return (min(a) - 3.0, max(a) + 3.0)

    def resultat(self, longueur_outil: Optional[float], stabilite) -> dict:
        T = self.T_robot_camera()
        Tm = T.copy()
        Tm[:3, 3] /= 1000.0
        d0 = self.decor_capture
        a = [math.degrees(math.atan2(v.A[1, 3], v.A[0, 3])) for v in self.vues]
        k = int(np.argmin(np.abs(np.array(a) - np.median(a))))
        x, y, _z = self.vues[k].A[:3, 3]
        return {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "methode": "pince",
            "T_robot_camera": Tm.tolist(),
            "unite": "metres (repere de la camera de PROFONDEUR -> repere robot du "
                     "simulateur dobot_sim)",
            "T_robot_camera_mm": T.tolist(),
            "z_table_mm": float(self.solution.t[2]),
            # pas de secteur impose : la camera est hors de portee du bras
            "secteur_base_deg": None,
            "secteur_parcouru_deg": list(self.secteur()),
            # les mors serrent selon y de l'outil : grand cote selon x
            "ecart_r_deg": 0.0,
            "depart": [float(x), float(y), 0.0],
            "longueur_outil_mm": None if longueur_outil is None else float(longueur_outil),
            "camera_K": np.asarray(d0["K"], float).tolist(),
            "camera_taille": [int(v) for v in np.asarray(d0["profondeur"]).shape[::-1]],
            "rms_mm": float(self.solution.distance),
            "paires": len(self.vues),
            "paires_retenues": len(self.vues),
            "stabilite_mm": {"moyenne": stabilite[0], "max": stabilite[1]},
            "hauteur_camera_mm": self.seg.hauteur_camera,
            "plongee_camera_deg": self.seg.plongee,
            "piece": {"etat": "absente"},
            "session": 0,
            "mesures": [],
            "vues": [{"nom": v.nom, "pose_outil": v.A.tolist(), "points": len(v.points)}
                     for v in self.vues],
        }


def enregistrer(resultat: dict, chemin: str = ec.FICHIER) -> Optional[str]:
    """Ecrit le fichier d'etalonnage ; l'ancien est garde a cote."""
    sauvegarde = None
    if os.path.exists(chemin):
        sauvegarde = chemin.replace(".json", ".avant_{}.json".format(time.strftime("%Y%m%d_%H%M%S")))
        shutil.copy(chemin, sauvegarde)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultat, f, indent=2)
    return sauvegarde


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Etalonnage camera -> robot par la pince seule")
    ap.add_argument("--longueur-outil", type=float, default=None,
                    help="longueur mesuree de la pince, mm (defaut : celle du fichier "
                         "d'etalonnage, sinon 107)")
    ap.add_argument("--vitesse", type=float, default=50.0, help="mm/s (50)")
    ap.add_argument("--vues", type=int, default=45, help="poses de couverture (45)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--sortie", default=ec.FICHIER)
    a = ap.parse_args(argv)

    longueur = a.longueur_outil
    if longueur is None and os.path.exists(a.sortie):
        try:
            with open(a.sortie, encoding="utf-8") as f:
                longueur = json.load(f).get("longueur_outil_mm")
        except Exception:                              # noqa: BLE001
            longueur = None
    if longueur is None:
        longueur = 107.0
    print("longueur d'outil : {:.1f} mm".format(longueur))

    import camera
    from dobot_sim import Commande, ErreurExecution
    robot = Commande(simulateur=False, robot=True, port=a.port, outil="pince",
                     vitesse=a.vitesse, vitesse_articulaire=a.vitesse,
                     longueur_outil=longueur)
    et = EtalonnagePince(robot, lambda: camera.capturer(IMAGES))
    try:
        # pompe coupee : la pince n'a rien a tenir (ses doigts, fins, ne
        # comptent presque pas dans le recalage)
        robot.couper_pompe()
        et.relever_decor()
        et.amorcer()
        et.couvrir(a.vues)
        print("CONTROLE : recalages sur deux moities des vues...")
        stab = et.stabilite()
        res = et.resultat(longueur, stab)
        T = np.array(res["T_robot_camera_mm"])
        print("\nRESULTAT : camera en x {:.0f} y {:.0f} z {:.0f} mm ; table a z = {:.1f} mm "
              "(repere robot)".format(*T[:3, 3], res["z_table_mm"]))
        print("  ecart mesure -> modele : {:.2f} mm (mediane) ; {} vues".format(
            res["rms_mm"], len(et.vues)))
        print("  repetabilite (deux moities) : {:.1f} mm en moyenne, {:.1f} au pire".format(*stab))
        avis = ("bonne" if stab[0] < 2.0 else "moyenne" if stab[0] < 4.0
                else "insuffisante pour saisir")
        print("  qualite : {}".format(avis))
        sauvegarde = enregistrer(res, a.sortie)
        print("etalonnage enregistre : {}".format(os.path.relpath(a.sortie)))
        if sauvegarde:
            print("  (l'ancien : {})".format(os.path.relpath(sauvegarde)))
        et.equerre()
    except KeyboardInterrupt:
        print("\ninterrompu (robot arrete) : rien d'enregistre")
    except ec.Arret as exc:
        print("arret : {}".format(exc))
    except ErreurExecution as exc:
        print("ECHEC : {}".format(exc))
    finally:
        try:
            robot.couper_pompe()
        except Exception:                              # noqa: BLE001
            pass
        robot.fermer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
