"""
Lecture de la description URDF officielle du Magician.

Le dossier constructeur fournit un paquet ROS `magician_description` :
un fichier URDF decrivant la chaine cinematique et un maillage STL par
corps. C'est une source bien meilleure que les OBJ d'origine, pour trois
raisons :

  * les maillages sont deja exprimes dans le repere de leur corps, il n'y
    a donc plus aucune pose d'assemblage a identifier ;
  * la decoupe par corps est celle du constructeur, carters de moteur
    compris ;
  * les couleurs sont donnees corps par corps dans le fichier.

Le raccordement a notre modele est AUTOMATIQUE : pour chaque corps URDF on
cherche celui de nos reperes qui rend constante la transformation entre
les deux, sur plusieurs configurations. Si aucun ne convient, le corps est
signale et ignore plutot que place au hasard.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import homogeneous as hg
from .robot import DobotMagician

__all__ = ["lire_stl", "Description", "charger_urdf", "pieces_depuis_urdf"]

#: configurations de controle utilisees pour identifier le repere porteur
POSES_TEST = (
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([25.0, 30.0, 45.0, 0.0]),
    np.array([-40.0, 10.0, 70.0, 0.0]),
)


# ---------------------------------------------------------------------------
# Maillages STL
# ---------------------------------------------------------------------------

def lire_stl(chemin: str, echelle: float = 1000.0
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lit un STL binaire ou ASCII.

    Les STL de ROS sont en metres : `echelle` les ramene en millimetres.
    Les sommets sont dupliques par facette, comme dans le format lui-meme.
    """
    with open(chemin, "rb") as f:
        tete = f.read(84)
        if tete[:5].lower() == b"solid" and b"facet" in tete:
            f.seek(0)
            txt = f.read().decode("ascii", "replace")
            V = np.asarray([[float(x) for x in m.groups()] for m in
                            re.finditer(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)],
                           dtype=np.float64)
        else:
            n = int(np.frombuffer(tete[80:84], dtype="<u4")[0])
            brut = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
            V = np.frombuffer(brut[:, 12:48].tobytes(),
                              dtype="<f4").reshape(n * 3, 3).astype(np.float64)
    V = V * float(echelle)
    F = np.arange(len(V), dtype=np.int32).reshape(-1, 3)
    return V.astype(np.float32), F


# ---------------------------------------------------------------------------
# Lecture du fichier URDF
# ---------------------------------------------------------------------------

def _flottants(texte: str, defaut) -> np.ndarray:
    """Lit une liste de nombres ; renvoie `defaut` si une expression xacro s'y trouve."""
    if not texte:
        return np.asarray(defaut, dtype=float)
    try:
        return np.asarray([float(v) for v in texte.split()], dtype=float)
    except ValueError:
        return np.asarray(defaut, dtype=float)


def _repere(element) -> np.ndarray:
    """Balise <origin xyz rpy> -> matrice homogene, longueurs en metres."""
    if element is None:
        return hg.identite()
    xyz = _flottants(element.get("xyz", ""), [0, 0, 0])
    r, p, y = _flottants(element.get("rpy", ""), [0, 0, 0])
    return hg.composer(hg.translation(*xyz), hg.rot_z(y), hg.rot_y(p), hg.rot_x(r))


class Description:
    """Chaine cinematique et maillages lus dans un URDF."""

    def __init__(self, chemin: str):
        self.chemin = chemin
        self.dossier = os.path.dirname(os.path.dirname(os.path.abspath(chemin)))
        racine = ET.parse(chemin).getroot()

        self.corps: Dict[str, dict] = {}
        for l in racine.findall("link"):
            nom = l.get("name")
            vis = l.find("visual")
            spec = {"maillage": None, "couleur": None,
                    "origine": hg.identite()}
            if vis is not None:
                m = vis.find("geometry/mesh")
                if m is not None and m.get("filename"):
                    spec["maillage"] = m.get("filename").split("/")[-1]
                c = vis.find("material/color")
                if c is not None:
                    spec["couleur"] = tuple(_flottants(c.get("rgba", ""),
                                                       [0.8, 0.8, 0.8, 1.0]))
                spec["origine"] = _repere(vis.find("origin"))
            self.corps[nom] = spec

        self.liaisons: List[dict] = []
        for j in racine.findall("joint"):
            mimic = j.find("mimic")
            self.liaisons.append({
                "nom": j.get("name"),
                "type": j.get("type", "fixed"),
                "parent": j.find("parent").get("link"),
                "enfant": j.find("child").get("link"),
                "origine": _repere(j.find("origin")),
                "axe": _flottants(j.find("axis").get("xyz", "")
                                  if j.find("axis") is not None else "",
                                  [0, 0, 1]),
                "mimique": None if mimic is None else {
                    "source": mimic.get("joint"),
                    "facteur": float(mimic.get("multiplier", 1.0)),
                    "decalage": float(mimic.get("offset", 0.0))},
            })
        self.par_nom = {j["nom"]: j for j in self.liaisons}

    # -- cinematique --------------------------------------------------------

    def _valeur(self, liaison: dict, valeurs: Dict[str, float]) -> float:
        if liaison["type"] in ("fixed", "floating"):
            return 0.0
        m = liaison["mimique"]
        if m is not None:
            return m["facteur"] * self._valeur(self.par_nom[m["source"]],
                                               valeurs) + m["decalage"]
        return float(valeurs.get(liaison["nom"], 0.0))

    def reperes(self, valeurs: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Situations absolues de tous les corps, en MILLIMETRES.

        `valeurs` donne l'angle (radians) des liaisons motrices ; les
        liaisons mimiques sont resolues d'elles-memes.
        """
        enfants: Dict[str, List[dict]] = {}
        for j in self.liaisons:
            enfants.setdefault(j["parent"], []).append(j)
        racines = set(self.corps) - {j["enfant"] for j in self.liaisons}

        out: Dict[str, np.ndarray] = {}
        for r in racines:
            out[r] = hg.identite()
        pile = list(racines)
        while pile:
            parent = pile.pop()
            for j in enfants.get(parent, []):
                M = j["origine"].copy()
                if j["type"] not in ("fixed", "floating"):
                    M = M @ hg.rot_axe(j["axe"], self._valeur(j, valeurs))
                out[j["enfant"]] = out[parent] @ M
                pile.append(j["enfant"])
        # metres -> millimetres, uniquement sur les translations
        for nom, M in out.items():
            N = M.copy()
            N[:3, 3] *= 1000.0
            out[nom] = N
        return out


def charger_urdf(chemin: str) -> Description:
    return Description(chemin)


# ---------------------------------------------------------------------------
# Raccordement a notre modele
# ---------------------------------------------------------------------------

def _valeurs_urdf(q_deg: np.ndarray, noms: Dict[str, str]) -> Dict[str, float]:
    d = np.pi / 180.0
    return {noms["j1"]: q_deg[0] * d,
            noms["j2"]: q_deg[1] * d,
            noms["j3"]: q_deg[2] * d}


def pieces_depuis_urdf(robot: DobotMagician, chemin_urdf: str,
                       couleurs: Optional[Dict[str, list]] = None,
                       journal=None) -> List:
    """
    Construit les pieces affichables a partir d'un URDF.

    Pour chaque corps, on cherche le repere k de NOTRE chaine tel que
    inv(T_k(q)) . T_urdf(q) soit constant : c'est celui qui porte le corps.
    La constance est verifiee sur plusieurs configurations, ce qui valide
    du meme coup que les deux modeles decrivent bien le meme robot.
    """
    from .meshes import Piece                       # import tardif : cycle

    def log(msg):
        if journal:
            journal(msg)

    d = charger_urdf(chemin_urdf)
    noms = {"j1": "magician_joint1", "j2": "magician_joint2",
            "j3": "magician_joint3"}
    manquants = [n for n in noms.values() if n not in d.par_nom]
    if manquants:
        log("URDF : liaisons introuvables {}".format(manquants))
        return []

    # l'origine de la chaine URDF est a l'epaule, sur l'axe J1 ;
    # notre repere 0 est au plan de pose : d1 les separe
    vers_base = hg.translation(0, 0, robot.geo.d1)

    reperes_urdf = [d.reperes(_valeurs_urdf(q, noms)) for q in POSES_TEST]
    reperes_nous = [robot.reperes(q) for q in POSES_TEST]
    n_nous = len(reperes_nous[0])

    pieces = []
    for nom, spec in d.corps.items():
        if not spec["maillage"]:
            continue
        chemin = os.path.join(d.dossier, "meshes", spec["maillage"])
        if not os.path.isfile(chemin):
            log("URDF : maillage absent {}".format(spec["maillage"]))
            continue

        # recherche du repere porteur
        meilleur, ecart_min, C = None, np.inf, None
        for k in range(n_nous):
            Cs = [hg.inverse(reperes_nous[i][k]) @ vers_base @ reperes_urdf[i][nom]
                  for i in range(len(POSES_TEST))]
            ecart = max(float(np.max(np.abs(Cs[0] - c))) for c in Cs[1:])
            if ecart < ecart_min:
                meilleur, ecart_min, C = k, ecart, Cs[0]
        if meilleur is None or ecart_min > 1e-6:
            log("URDF : {} n'est solidaire d'aucun repere (ecart {:.2e})"
                .format(nom, ecart_min))
            continue

        V, F = lire_stl(chemin)          # deja converti en millimetres
        origine = spec["origine"].copy()
        origine[:3, 3] *= 1000.0         # l'URDF est en metres, C est en mm
        Vl = hg.transformer_points(C @ origine,
                                   V.astype(np.float64)).astype(np.float32)

        couleur = (couleurs or {}).get(nom) or spec["couleur"] \
            or (0.85, 0.85, 0.87, 1.0)
        pieces.append(Piece(nom=nom, repere=meilleur, sommets=Vl, faces=F,
                            couleur=tuple(couleur)))
        log("URDF : {:<26} repere {}  {:5d} facettes  (ecart {:.1e})"
            .format(nom, meilleur, len(F), ecart_min))
    return pieces
