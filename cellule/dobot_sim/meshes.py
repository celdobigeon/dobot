"""
Chargement du modele 3D du Dobot et placement des pieces dans les reperes
de la chaine cinematique.

Principe
--------
Les fichiers OBJ fournis decrivent le robot dans UNE pose d'assemblage
unique, exprimee dans un repere CAO. Pour animer le robot il faut exprimer
chaque piece dans le repere de son propre corps. Si q0 est la configuration
articulaire de la pose d'assemblage, la transformation a appliquer aux
sommets de la piece portee par le corps k est :

    M_k = inverse(T0k(q0)) . Trans(t) . Rz(alpha)

ou Rz(alpha) redresse le repere CAO sur le repere robot et Trans(t) amene
l'origine sur l'axe J1, au niveau du plan de pose.

Ces valeurs sont dans `model_config.json` : elles ont ete identifiees sur
la geometrie des maillages et restent ajustables (menu Affichage >
Calibrer le modele 3D).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import homogeneous as hg
from .robot import DobotMagician

__all__ = ["Piece", "charger_obj", "charger_pieces", "modele_procedural",
           "chemin_config_defaut", "RACINE_PROJET"]

RACINE_PROJET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_CACHE = os.path.join(RACINE_PROJET, ".cache_maillages")


# ---------------------------------------------------------------------------

@dataclass
class Piece:
    """Un maillage rigide attache a un repere de la chaine cinematique."""
    nom: str
    repere: int                 # indice dans robot.reperes(q) : 0..6
    sommets: np.ndarray         # (N, 3) float32, DEJA exprimes dans le repere
    faces: np.ndarray           # (M, 3) int32
    couleur: Tuple[float, float, float, float] = (0.72, 0.74, 0.78, 1.0)
    visible: bool = True

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


# ---------------------------------------------------------------------------
# Lecture OBJ
# ---------------------------------------------------------------------------

VERSION_LECTEUR = 3          # a incrementer pour invalider les caches existants


def _cle_cache(chemin: str) -> str:
    st = os.stat(chemin)
    brut = "{}|{}|{}|v{}".format(os.path.abspath(chemin), st.st_size,
                                 int(st.st_mtime), VERSION_LECTEUR)
    return hashlib.md5(brut.encode("utf-8")).hexdigest()[:16]


def _lignes_logiques(fh):
    """
    Itere sur les lignes en recollant les continuations.

    Le format OBJ autorise la coupure d'une declaration longue par une
    barre oblique inverse en fin de ligne ; les fichiers exportes par
    Rhino en contiennent (declarations vn et f).
    """
    tampon = ""
    for brut in fh:
        ligne = brut.rstrip("\r\n")
        if ligne.endswith("\\"):
            tampon += ligne[:-1] + " "
            continue
        if tampon:
            ligne = tampon + ligne
            tampon = ""
        yield ligne
    if tampon:
        yield tampon


def _nettoyer(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Supprime les facettes hors bornes, degenerees ou d'aire nulle."""
    if F.size:
        ok = ((F >= 0).all(axis=1) & (F < len(V)).all(axis=1)
              & (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]))
        F = F[ok]
    if F.size:
        # une facette d'aire nulle n'a pas de normale definie : elle
        # contaminerait le calcul des normales aux sommets (NaN au rendu)
        P = V.astype(np.float64)
        aire = np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]],
                                       P[F[:, 2]] - P[F[:, 0]]), axis=1)
        F = F[aire > 1e-12]
    return V, F


def _lire_obj(chemin: str):
    """Analyse brute : sommets globaux et faces regroupees par 'g'."""
    sommets: List[Tuple[float, float, float]] = []
    groupes: List[Tuple[str, List[Tuple[int, int, int]]]] = []
    courant: List[Tuple[int, int, int]] = []
    nom_courant = "defaut"
    with open(chemin, "r", encoding="utf-8", errors="replace") as fh:
        for ligne in _lignes_logiques(fh):
            if len(ligne) < 2:
                continue
            if ligne.startswith("g ") or ligne.startswith("o "):
                if courant:
                    groupes.append((nom_courant, courant))
                    courant = []
                nom_courant = ligne[2:].strip() or "defaut"
            elif ligne[1] != " ":
                continue
            elif ligne[0] == "v":
                m = ligne.split()
                if len(m) >= 4:
                    sommets.append((float(m[1]), float(m[2]), float(m[3])))
            elif ligne[0] == "f":
                idx = []
                for tok in ligne.split()[1:]:
                    tete = tok.split("/")[0]
                    if not tete:
                        continue
                    k = int(tete)
                    idx.append(k - 1 if k > 0 else len(sommets) + k)
                for j in range(1, len(idx) - 1):
                    courant.append((idx[0], idx[j], idx[j + 1]))
    if courant:
        groupes.append((nom_courant, courant))
    return np.asarray(sommets, dtype=np.float32).reshape(-1, 3), groupes


def charger_obj_groupes(chemin: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Lit un OBJ en conservant la decoupe en groupes ('g').

    Les fichiers livres separent ainsi les corps blancs des carters de
    moteur et des bielles, ce qui permet de les colorer distinctement.
    Chaque groupe recoit ses propres sommets, renumerotes.
    """
    V, groupes = _lire_obj(chemin)
    sortie = []
    for nom, faces in groupes:
        F = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        Vg, Fg = _nettoyer(V, F)
        if not Fg.size:
            continue
        utiles = np.unique(Fg)
        table = np.full(len(Vg), -1, dtype=np.int32)
        table[utiles] = np.arange(len(utiles), dtype=np.int32)
        sortie.append((nom, Vg[utiles].copy(), table[Fg]))
    return sortie


def charger_obj(chemin: str, utiliser_cache: bool = True
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lit un fichier OBJ et renvoie (sommets (N,3) float32, faces (M,3) int32).

    Les polygones a plus de trois cotes sont triangules en eventail.
    Un cache .npz accelere fortement les lancements suivants.
    """
    cache = os.path.join(DOSSIER_CACHE, _cle_cache(chemin) + ".npz")
    if utiliser_cache and os.path.isfile(cache):
        try:
            d = np.load(cache)
            return d["v"], d["f"]
        except Exception:
            pass

    V, groupes = _lire_obj(chemin)
    faces = [t for _, fs in groupes for t in fs]
    F = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    V, F = _nettoyer(V, F)

    if utiliser_cache:
        try:
            os.makedirs(DOSSIER_CACHE, exist_ok=True)
            np.savez_compressed(cache, v=V, f=F)
        except Exception:
            pass
    return V, F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def composante_du_plus_haut(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Les faces du CORPS AUQUEL APPARTIENT LE POINT LE PLUS HAUT.

    Sert a detacher d'un maillage d'un seul tenant la piece qui domine
    la scene -- sur le convoyeur, le bras qui porte le capteur. Rend un
    masque booleen sur `F`.

    POURQUOI « LE PLUS HAUT » ET NON UNE BOITE DE SELECTION. Parce
    qu'une boite se deregle des qu'on change de modele, et sans rien
    dire : elle rendrait des faces arbitraires. Le point culminant,
    lui, designe toujours quelque chose, et sur un tapis c'est le bras
    du capteur -- rien d'autre ne depasse la bande.

    Le decoupage suit les ARETES PARTAGEES : une piece boulonnee mais
    maillee a part reste une composante distincte, ce qui est
    exactement ce qu'on veut.
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64)
    if not len(F):
        return np.zeros(0, dtype=bool)

    parent = np.arange(len(V), dtype=np.int64)

    def trouver(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    for f in F:
        ra, rb, rc = trouver(f[0]), trouver(f[1]), trouver(f[2])
        if ra != rb:
            parent[ra] = rb
            rb = trouver(rb)
        if rb != rc:
            parent[rb] = rc

    racine = trouver(int(np.argmax(V[:, 2])))
    return np.array([trouver(int(f[0])) == racine for f in F], dtype=bool)


def chemin_config_defaut() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "model_config.json")


def _trouver_fichier(motif: str, racine: str) -> Optional[str]:
    """
    Recherche recursive d'un fichier correspondant a un motif glob.

    On evite ainsi de stocker dans le JSON des noms de dossiers contenant
    des caracteres non ASCII (le modele livre en contient).
    """
    if os.path.isabs(motif) and os.path.isfile(motif):
        return motif
    trouves = glob.glob(os.path.join(racine, "**", motif), recursive=True)
    trouves = [t for t in trouves if os.path.isfile(t)]
    trouves.sort(key=len)
    return trouves[0] if trouves else None


def charger_config(chemin: Optional[str] = None) -> Dict:
    with open(chemin or chemin_config_defaut(), "r", encoding="utf-8") as f:
        return json.load(f)


def charger_pieces(robot: DobotMagician,
                   config: Optional[Dict] = None,
                   racine: Optional[str] = None,
                   journal=None) -> List[Piece]:
    """
    Charge et recale toutes les pieces du modele 3D.

    Deux sources possibles, choisies par la cle `source` de la config :

      "urdf"  la description ROS officielle du constructeur. Les maillages
              sont deja dans le repere de leur corps : rien a identifier.
      "obj"   les OBJ d'origine, recales sur une pose d'assemblage.

    En mode "urdf", un echec bascule automatiquement sur "obj" ; si les
    deux echouent, la liste renvoyee est vide et l'appelant utilise
    `modele_procedural`.
    """
    cfg = config or charger_config()

    def log(msg):
        if journal:
            journal(msg)

    if cfg.get("source", "obj") == "urdf":
        pieces = _charger_urdf(robot, cfg, log)
        if pieces:
            return pieces
        log("modele URDF indisponible : repli sur les OBJ d'origine")
    return _charger_obj(robot, cfg.get("obj", cfg), racine, journal)


def _charger_urdf(robot: DobotMagician, cfg: Dict, log) -> List[Piece]:
    """Charge la description ROS officielle."""
    from .urdf import pieces_depuis_urdf           # import tardif : dependance

    spec = cfg.get("urdf", {})
    racine = os.path.join(RACINE_PROJET, spec.get("racine_recherche", "."))
    chemin = _trouver_fichier(spec.get("fichier", "magician.urdf.xacro"), racine)
    if chemin is None:
        log("URDF introuvable : {}".format(spec.get("fichier")))
        return []
    try:
        return pieces_depuis_urdf(robot, chemin, spec.get("couleurs"), log)
    except Exception as exc:                        # pragma: no cover
        log("lecture URDF impossible : {}".format(exc))
        return []


def _charger_obj(robot: DobotMagician,
                 cfg: Dict,
                 racine: Optional[str] = None,
                 journal=None) -> List[Piece]:
    """Charge les OBJ d'origine et les recale sur la pose d'assemblage."""
    racine = racine or os.path.join(RACINE_PROJET, cfg.get("racine_recherche", "Modele3D"))
    if not os.path.isdir(racine):
        racine = RACINE_PROJET

    q0 = np.asarray(cfg.get("pose_assemblage_deg", [0, 0, 0, 0]), dtype=float)
    alpha = float(cfg.get("rotation_z_deg", 0.0)) * np.pi / 180.0
    echelle = float(cfg.get("echelle", 1.0))
    reperes_q0 = robot.reperes(q0)

    def log(msg):
        if journal:
            journal(msg)

    pieces: List[Piece] = []
    for spec in cfg.get("pieces", []):
        motif = spec["fichier"]
        chemin = _trouver_fichier(motif, racine)
        if chemin is None:
            log("maillage introuvable : {}".format(motif))
            continue
        k = int(spec.get("repere", 0))
        k = max(0, min(k, len(reperes_q0) - 1))
        M = hg.composer(hg.inverse(reperes_q0[k]),
                        hg.translation(*spec.get("translation", [0, 0, 0])),
                        hg.rot_z(alpha))
        if echelle != 1.0:
            M[:3, :3] *= echelle

        couleur_base = tuple(spec.get("couleur", [0.72, 0.74, 0.78, 1.0]))
        par_groupe = spec.get("couleurs_groupes") or {}
        nom_piece = spec.get("nom", os.path.basename(chemin))

        try:
            if par_groupe:
                # le fichier est decoupe pour distinguer les carters noirs
                morceaux = [(nom_piece + "/" + g, Vg, Fg)
                            for g, Vg, Fg in charger_obj_groupes(chemin)]
                couleurs = [tuple(par_groupe.get(g.split("/")[-1], couleur_base))
                            for g, _, _ in morceaux]
            else:
                V, F = charger_obj(chemin)
                morceaux = [(nom_piece, V, F)]
                couleurs = [couleur_base]
        except Exception as exc:                      # pragma: no cover
            log("lecture impossible ({}) : {}".format(os.path.basename(chemin), exc))
            continue

        total = 0
        for (nom_m, V, F), couleur in zip(morceaux, couleurs):
            if V.size == 0 or F.size == 0:
                continue
            Vl = hg.transformer_points(M, V.astype(np.float64)).astype(np.float32)
            pieces.append(Piece(nom=nom_m, repere=k, sommets=Vl, faces=F,
                                couleur=couleur))
            total += int(F.shape[0])
        if total == 0:
            log("maillage vide : {}".format(os.path.basename(chemin)))
        else:
            log("charge : {} ({} faces, {} morceau(x), repere {})"
                .format(nom_piece, total, len(morceaux), k))
    return pieces


# ---------------------------------------------------------------------------
# Modele de secours, construit a partir des seuls parametres cinematiques
# ---------------------------------------------------------------------------

def boite(dx: float, dy: float, dz: float, centre=(0.0, 0.0, 0.0)):
    cx, cy, cz = centre
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    V = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
                 dtype=np.float32)
    V += np.array([cx, cy, cz], dtype=np.float32)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)
    return V, F


def cylindre(rayon: float, hauteur: float, axe: int = 2, n: int = 24,
              centre=(0.0, 0.0, 0.0)):
    """Cylindre ferme, axe = 0 (x), 1 (y) ou 2 (z), centre sur `centre`."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = rayon * np.cos(th), rayon * np.sin(th)
    h = hauteur / 2.0
    a, b = (0, 1) if axe == 2 else ((1, 2) if axe == 0 else (2, 0))
    bas = np.zeros((n, 3), dtype=np.float32)
    haut = np.zeros((n, 3), dtype=np.float32)
    bas[:, a], bas[:, b], bas[:, axe] = c, s, -h
    haut[:, a], haut[:, b], haut[:, axe] = c, s, h
    V = np.vstack([bas, haut,
                   np.array([[0, 0, 0]], dtype=np.float32),
                   np.array([[0, 0, 0]], dtype=np.float32)])
    V[2 * n, axe] = -h
    V[2 * n + 1, axe] = h
    V += np.asarray(centre, dtype=np.float32)
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, n + i, n + j], [i, n + j, j]]
        F += [[2 * n, j, i]]
        F += [[2 * n + 1, n + i, n + j]]
    return V, np.asarray(F, dtype=np.int32)


def fusion(*morceaux):
    Vs, Fs, decal = [], [], 0
    for V, F in morceaux:
        Vs.append(V)
        Fs.append(F + decal)
        decal += len(V)
    return np.vstack(Vs).astype(np.float32), np.vstack(Fs).astype(np.int32)


def modele_procedural(robot: DobotMagician) -> List[Piece]:
    """
    Modele simplifie deduit des seuls parametres d1, l1, l2, l3.

    Toujours parfaitement coherent avec la cinematique : sert de secours si
    les fichiers CAO sont absents, et de reference pour verifier le recalage
    du modele detaille.
    """
    g = robot.geo
    gris = (0.62, 0.65, 0.70, 1.0)
    bleu = (0.29, 0.47, 0.72, 1.0)
    orange = (0.85, 0.52, 0.20, 1.0)

    socle = Piece("socle", 0, *boite(158, 158, 55, (-8, 0, 27.5)), couleur=gris)

    colonne = Piece("colonne", 1,
                    *fusion(boite(90, 100, g.d1 - 45, (-10, 0, (g.d1 - 45) / 2 + 45)),
                             cylindre(46, 22, 2, 28, (0, 0, 50))),
                    couleur=bleu)

    bras, faces_bras = fusion(boite(46, 34, g.l1, (0, 0, g.l1 / 2.0)),
                               cylindre(21, 40, 1, 20, (0, 0, 0)),
                               cylindre(19, 36, 1, 20, (0, 0, g.l1)))
    bras_arriere = Piece("bras arriere", 2, bras, faces_bras, couleur=gris)

    av, faces_av = fusion(boite(38, 28, g.l2, (0, 0, g.l2 / 2.0)),
                           cylindre(17, 32, 1, 20, (0, 0, g.l2)))
    avant_bras = Piece("avant-bras", 3, av, faces_av, couleur=bleu)

    tete, faces_tete = fusion(boite(g.l3 + 26, 24, 18, (g.l3 / 2.0 - 6, 0, -6)),
                               cylindre(11, 46, 2, 18, (g.l3, 0, -30)),
                               cylindre(16, 6, 2, 18, (g.l3, 0, -54)))
    effecteur = Piece("effecteur", 4, tete, faces_tete, couleur=orange)

    return [socle, colonne, bras_arriere, avant_bras, effecteur]
