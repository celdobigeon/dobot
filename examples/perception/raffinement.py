# -*- coding: utf-8 -*-
"""
Affinage de la SCENE ENTIERE par "rendre et comparer", sous contraintes
physiques.

POURQUOI. Les detecteurs (YOLO-seg + ajustement par masque, vision_modele)
regardent chaque piece SEULE et la supposent HORIZONTALE (sur la table ou a
plat sur une autre). Dans un vrai tas, beaucoup de pieces sont INCLINEES
(un bout sur la table, l'autre sur une piece), se cachent les unes les
autres, et une piece manquee par le reseau est perdue. Ici on fait
l'inverse : on construit une scene 3D HYPOTHETIQUE (des boites de
dimensions connues, posees de facon physiquement plausible), on la REND
dans les vues de la camera, on la COMPARE aux mesures, et on la corrige
jusqu'a ce qu'elle explique ce que la camera voit. Les detecteurs ne font
plus que proposer.

  1. OBSERVATION (Observation). Plan de la table et correction de courbure
     (perception.analyser_table). Profondeur "corrigee" de chaque pixel :
     celle qui donne la hauteur corrigee, de sorte que la table mesuree
     soit exactement le plan z = 0 du rendu. Bruit de la stereo en k z^2,
     k etalonne sur la table de la capture (dans le domaine de la
     PROFONDEUR : en vue rasante, une erreur de profondeur change peu la
     hauteur). Pixels sans mesure : NEUTRES (poids 0). Pixels VOLANTS (a mi-
     hauteur d'un saut) : poids reduit. Grille d'echantillons espaces
     d'environ 2.4 mm sur la table (2 pixels a 0.5 m, 3 a 0.3 m). Deux
     masques de prudence : le BORD de la table (nappe, chaise : rien n'y
     est propose) et les OBJETS ETRANGERS (surface lisse de plus de 40 mm
     de large et de 6000 mm^2 : livre, planche ; on ne la pave pas de
     pieces).
  2. MODELE DE SCENE (Boite). Chaque piece : type, face d'appui (quelle
     dimension est verticale), et pose SE(3) restreinte : position du
     centre, lacet, tangage (inclinaison le long du grand cote), roulis
     (borne a 40 deg). La plausibilite physique est imposee par des
     penalites : interpenetration entre boites (test des axes separateurs,
     SAT, profondeur de penetration) et avec la table ; APPUI (le dessous
     de la boite doit toucher la table ou une autre boite : rayons
     verticaux vers le bas ; ou un bout contre le flanc d'une boite) ;
     STABILITE (le centre de gravite entre les appuis, le long de la piece
     et en travers). Une piece posee a plat, empilee, ou inclinee (un bout
     sur la table, l'autre sur l'arete ou contre le flanc d'une piece) est
     ainsi decrite par le MEME modele ; une inclinaison coute un peu (a
     priori : la plupart des pieces sont a plat). Etre "en l'air" a un cout
     BORNE : la porteuse peut etre une piece cachee ou pas encore trouvee.
  3. RENDU (lancer). Lancer de rayons sur boites (methode des dalles,
     float32, produits matriciels, par lots de poses candidates) ;
     z-buffer : chaque piece garde sa carte de profondeur, la scene est
     leur minimum (et la table) : les occultations mutuelles sont exactes.
     Dans la vue de PROFONDEUR (residu) et dans la vue COULEUR (brute, avec
     K_couleur et T_couleur_profondeur, ou recalee) pour les masques. Le
     rendu est LISSE comme la camera lisse les sauts de profondeur (rampes
     de pixels volants) : 0 a 3 passes [1 2 1], nombre choisi sur la
     capture (celui qui explique le mieux la mesure ; sans ce lissage, les
     rampes tirent les pieces de 1 a 2 mm et les enfoncent).
  4. COUT (Scene.energie), en "pixels mal expliques" :
       - PROFONDEUR : residu normalise par le bruit, perte robuste de
         Geman-McClure (bornee : un pixel aberrant coute au plus 1) ;
       - MASQUES DU RESEAU (si presents) : desaccord entre la partie visible
         rendue de la piece et le masque qui lui est associe (et l'effet
         de son occultation sur les autres), plus un cout si le type
         contredit la classe du masque. Ils FACONNENT la pose et le type
         mais ne peuvent JAMAIS imposer une piece : une piece sans masque ne
         paie rien, et une piece n'est gardee que si la PROFONDEUR la
         soutient (un masque net abaisse seulement l'exigence) ; une piece
         a travers laquelle la profondeur voit la table est retiree, quel
         que soit son masque ;
       - ARETES (infrarouge, chanfrein oriente) : implantees, NULLES par
         defaut (Reglages.aretes = 0) : aucun gain mesure ; l'image est
         ignoree si la table elle-meme porte trop d'aretes (points du
         projecteur de pres, bois veine). La couleur n'est pas utilisee
         (un fond texture cree des aretes parasites) ;
       - PHYSIQUE : penetration (quadratique, au-dela de 0.3 mm), piece en
         l'air, centre de gravite hors des appuis ; a priori d'inclinaison ;
       - PARCIMONIE (Affinage.garder) : une piece doit expliquer assez de
         pixels (gain de profondeur, plus l'appui qu'elle donne a d'autres :
         35 pixels et 35 % de ses pixels visibles), on ne doit pas voir a
         travers elle, elle ne doit pas etre taillee dans une surface plus
         grande (la mesure continue au-dela de trois de ses cotes, ou de
         ses deux bouts) ; sinon elle est retiree (fantome, doublon).
     Le cout d'une piece est calcule en DIFFERENCE avec son absence, sur
     sa seule fenetre : il ne depend pas de la taille de la fenetre, et
     des variantes (type, face) se comparent directement.
  5. OPTIMISATION. Recherche par motifs : a chaque pas, toutes les
     coordonnees +/- et des PIVOTS (tourner autour d'un bout ou d'un bord,
     pour qu'une piece se leve en gardant un bout sur la table) evalues en
     un seul lot ; GRANDS SAUTS (glisser d'un huitieme a une demi longueur,
     un quart de tour) au depart et apres convergence. Sans gradient :
     robuste aux plateaux d'un cout en pixels. Balayages de toute la scene
     piece par piece (coordonnees par blocs ; les pieces du dessus
     d'abord), avec CONTINUATION : la penetration ne compte d'abord que
     2 %, puis 25 %, puis 100 % (une inclinee lue a plat, en travers de sa
     porteuse, se leve alors comme la mesure le dit au lieu de fuir de
     cote). Choix DISCRETS (type, face) : chaque variante est posee au
     meme dessous (ou au meme dessus), affinee, comparee. Puis, en boucle :
       - REINITIALISATION d'une piece "vue a travers" (12 % de ses pixels) :
         retiree, une piece est reproposee sur ce qu'elle laisse ;
       - RE-PROPOSITION gloutonne : la plus grande zone ou la mesure est
         au-dessus de la scene rendue (ou un masque sans piece) est
         decoupee en segments alignes (RANSAC de droites 3D : une inclinee
         croisant une couchee fait deux segments) ; pour chacun, grand axe
         (rectangle minimal), pente (droite du dessus des points), et des
         positions alignees sur les bouts et les bords ; la meilleure est
         affinee, ses voisines re-ajustees ; gardee si elle (et elles)
         restent justifiees, sinon tout revient en place (Scene.sauver /
         restaurer).
  6. SORTIE : PieceVue (perception.py) : T_camera, T_table (face superieure,
     inclinee s'il le faut), type, face, support (la porteuse, ou la piece
     ou s'appuie une inclinee), appuis (toutes), dessus, inclinaison
     (deg), confiance (part des pixels expliques, marge avec la meilleure
     autre variante, physique, part vue a travers).

MESURES (evaluer_detection.py --raffinement ; 8 scenes de 8 pieces par cas).
Vue de dessus, 3 pieces inclinees : rappel 0.84 -> 0.94, fausses 8 -> 1,
inclinaison des inclinees 11 -> 0.3 deg, leur support juste 0.17 -> 0.68,
position 1.5 -> 0.5 mm, ordre de prise sans faute 0.25 -> 0.62 ; masques
parfaits : 0.92 -> 1.00 ; masques ABIMES (30 % perdus, fusions, 2 faux sur
le fond) : 0.52 -> 0.98. Piles : 0.92 -> 0.92 (fausses 4 -> 1, position
0.9 -> 0.6 mm) ; masques abimes : 0.67 -> 0.98. Pieces a plat en contact :
inchange (1.00, 0.4 mm). Vue reculee, inclinees : 0.75 -> 0.91 (12 -> 3
fausses). Vue rasante (1 m) : inclinees 0.62 -> 0.72, piles 0.50 -> 0.67,
erreurs de position toujours 2.5 a 3.5 mm. Jeu reel annote (40 images de
validation, rendu compare aux masques) : geometrie seule 0.79 / 0.85
(rappel / precision) -> 0.90 / 0.94 ; YOLO 0.98 / 0.98 -> 0.99 / 0.97 ;
YOLO abime 0.66 / 0.93 -> 0.94 / 0.92. Sans chaque terme (--ablation,
dessus et rasante) : sans lissage, position 0.6 -> 1.1 mm ; sans
physique, 3 fausses au lieu de 1 et inclinaison 0.3 -> 0.5 deg ; sans a
priori d'inclinaison, fausses inclinaisons en rasante (0.04 -> 1.5 deg sur
les pieces a plat) ; sans variantes, type et face 0.95 -> 0.88 ; sans
repropositions, rappel des inclinees 0.89 -> 0.56 ; sans masques (avec un
reseau), rappel en rasante 0.90 -> 0.79. Temps de l'affinage : 1 a 3 s
pour 8 pieces (Apple M4 Max, numpy), plus la detection geometrique qui
propose (2 a 4 s).

Propositions : vision_modele (geometrie seule : independante du fond et
de la couleur), masques YOLO (si un modele entraine existe), et les zones
non expliquees. La geometrie suffit seule : YOLO n'est qu'un indice.

Chiffres et comparaisons : evaluer_detection.py --raffinement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:                        # pragma: no cover
    raise ImportError("OpenCV manque : .venv/bin/pip install opencv-python") from exc

import perception as pc
import vision

FACES = ("a plat", "sur la tranche", "debout")


# ---------------------------------------------------------------------------
# Geometrie des boites
# ---------------------------------------------------------------------------

def rotations(P: np.ndarray) -> np.ndarray:
    """Poses (B x 6 : x, y, z, lacet, tangage, roulis) -> rotations B x 3 x 3,
    boite -> table. R = Rz(lacet) Ry(-tangage) Rx(roulis) : le tangage leve
    le bout +x de la boite, le roulis la couche sur le flanc."""
    P = np.atleast_2d(P)
    a, b, c = P[:, 3], P[:, 4], P[:, 5]
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    R = np.empty((len(P), 3, 3))
    # Rz(a) @ [[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]] @ [[1, 0, 0], [0, cc, -sc], [0, sc, cc]]
    R[:, 0, 0] = ca * cb
    R[:, 0, 1] = -sa * cc - ca * sb * sc
    R[:, 0, 2] = sa * sc - ca * sb * cc
    R[:, 1, 0] = sa * cb
    R[:, 1, 1] = ca * cc - sa * sb * sc
    R[:, 1, 2] = -ca * sc - sa * sb * cc
    R[:, 2, 0] = sb
    R[:, 2, 1] = cb * sc
    R[:, 2, 2] = cb * cc
    return R


#: les 8 coins d'une boite unite (demi-dimensions 1)
_SIGNES = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def coins(C: np.ndarray, R: np.ndarray, demi: np.ndarray) -> np.ndarray:
    """Coins (B x 8 x 3) des boites de centres C (B x 3), rotations R, demi-
    dimensions demi (3 ou B x 3)."""
    demi = np.broadcast_to(demi, (len(C), 3))
    loc = _SIGNES[None] * demi[:, None, :]                       # B x 8 x 3
    return np.einsum("bkj,bij->bki", loc, R) + C[:, None, :]


def penetration(C1: np.ndarray, R1: np.ndarray, h1: np.ndarray,
                c2: np.ndarray, R2: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    Profondeur d'interpenetration (m) de B boites (C1 : B x 3, R1 : B x 3 x 3,
    demi-dimensions h1 : 3 ou B x 3) avec UNE boite (c2, R2, h2) : le plus
    petit recouvrement des projections sur les 15 axes separateurs (3 + 3
    faces, 9 produits d'aretes). 0 si un axe les separe : elles sont
    disjointes (theoreme des axes separateurs, exact pour deux boites).
    """
    B = len(C1)
    h1 = np.broadcast_to(np.asarray(h1, float), (B, 3))
    A = np.transpose(R1, (0, 2, 1))                       # B x 3 (axes) x 3
    Bx = np.broadcast_to(R2.T[None], (B, 3, 3))
    croix = np.cross(A[:, :, None, :], Bx[:, None, :, :]).reshape(B, 9, 3)
    axes = np.concatenate([A, Bx, croix], axis=1)         # B x 15 x 3
    n = np.linalg.norm(axes, axis=2, keepdims=True)
    degenere = n[..., 0] < 1e-6
    axes = axes / np.maximum(n, 1e-12)
    rA = np.einsum("bxj,bj->bx", np.abs(np.einsum("bxk,bjk->bxj", axes, A)), h1)
    rB = np.abs(np.einsum("bxk,jk->bxj", axes, R2.T)) @ np.asarray(h2, float)
    d = np.abs(np.einsum("bxk,bk->bx", axes, c2[None, :] - C1))
    rec = rA + rB - d
    rec[degenere] = np.inf
    m = rec.min(axis=1)
    return np.maximum(m, 0.0)


def penetrations(C1, R1, h1, C2, R2, h2) -> np.ndarray:
    """Comme penetration, pour B poses contre M boites : B x M."""
    B, M = len(C1), len(C2)
    h1 = np.broadcast_to(np.asarray(h1, float), (B, 3))
    A = np.transpose(R1, (0, 2, 1))[:, None]                 # B x 1 x 3 x 3
    Bx = np.transpose(R2, (0, 2, 1))[None]                   # 1 x M x 3 x 3
    A = np.broadcast_to(A, (B, M, 3, 3))
    Bx = np.broadcast_to(Bx, (B, M, 3, 3))
    croix = np.cross(A[:, :, :, None, :], Bx[:, :, None, :, :]).reshape(B, M, 9, 3)
    axes = np.concatenate([A, Bx, croix], axis=2)            # B x M x 15 x 3
    n = np.linalg.norm(axes, axis=3, keepdims=True)
    degenere = n[..., 0] < 1e-6
    axes = axes / np.maximum(n, 1e-12)
    rA = np.einsum("bmxj,bj->bmx", np.abs(np.einsum("bmxk,bmjk->bmxj", axes, A)), h1)
    rB = np.einsum("bmxj,mj->bmx", np.abs(np.einsum("bmxk,bmjk->bmxj", axes, Bx)), h2)
    d = np.abs(np.einsum("bmxk,bmk->bmx", axes, C2[None] - C1[:, None]))
    rec = rA + rB - d
    rec[degenere] = np.inf
    return np.maximum(rec.min(axis=2), 0.0)


def lancer(o: np.ndarray, d: np.ndarray, C: np.ndarray, R: np.ndarray,
           demi: np.ndarray) -> np.ndarray:
    """
    Lancer de rayons sur B boites (methode des dalles). Rayons p = o + t d
    (o : 3 ou N x 3 ; d : N x 3), boites de centres C (B x 3), rotations R
    (B x 3 x 3), demi-dimensions demi (3 ou B x 3). Rend t d'entree (B x N),
    inf si le rayon manque la boite ou part de l'interieur. Calcul en
    float32 (le millieme de millimetre suffit), par produits matriciels.
    """
    o = np.asarray(o, float)
    B = len(C)
    demi = np.broadcast_to(np.asarray(demi, float), (B, 3)).astype(np.float32)[:, None, :]
    R32 = R.astype(np.float32)
    if o.ndim == 1:
        ol = np.matmul((o[None, :] - C)[:, None, :].astype(np.float32), R32)   # B x 1 x 3
    else:
        ol = np.matmul((o[None, :, :] - C[:, None, :]).astype(np.float32), R32)  # B x N x 3
    dl = np.matmul(np.asarray(d, np.float32)[None], R32)                      # B x N x 3
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv = 1.0 / dl
        t1 = (-demi - ol) * inv
        t2 = (demi - ol) * inv
    tin = np.fmax(np.fmax(np.fmin(t1[..., 0], t2[..., 0]), np.fmin(t1[..., 1], t2[..., 1])),
                  np.fmin(t1[..., 2], t2[..., 2]))
    tout = np.fmin(np.fmin(np.fmax(t1[..., 0], t2[..., 0]), np.fmax(t1[..., 1], t2[..., 1])),
                   np.fmax(t1[..., 2], t2[..., 2]))
    touche = (tin <= tout) & (tin > 0)
    return np.where(touche, tin, np.float32(np.inf))


# ---------------------------------------------------------------------------
# Pieces hypothetiques
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Boite:
    """Une piece de la scene hypothetique. dims : (L, l, h) en m, h la
    dimension verticale quand la piece est a plat ; p : x, y, z du CENTRE de
    la boite (m, repere table), lacet, tangage, roulis (rad)."""
    type: str
    face: str
    dims: np.ndarray
    p: np.ndarray
    origine: str = ""
    # remplis par l'affinage
    gain: float = 0.0             # pixels de profondeur expliques
    visibles: float = 0.0         # pixels visibles mesures
    contredits: float = 0.0       # part des pixels visibles ou l'on voit A TRAVERS
    physique: float = 0.0
    marge: float = 0.0            # avec la meilleure autre variante (type, face)
    masque: int = -1              # indice du masque YOLO associe
    iou_masque: float = 0.0
    debord: float = 0.0           # part du pourtour ou la surface continue
    bord: bool = False            # proposee au bord de la table
    confiance: float = 0.0

    @property
    def demi(self) -> np.ndarray:
        return np.asarray(self.dims, float) / 2.0

    def R(self) -> np.ndarray:
        cle = self.p.tobytes()
        if getattr(self, "_cle", None) != cle:
            self._R, self._cle = rotations(self.p[None])[0], cle
        return self._R

    def copie(self) -> "Boite":
        return Boite(self.type, self.face, np.array(self.dims, float), np.array(self.p, float),
                     self.origine)


def dims_m(types, nom, face) -> np.ndarray:
    return np.array(pc.forme(types[nom], face), float) / 1000.0


def boite_de_piece(piece, T_tc: np.ndarray, types) -> Optional[Boite]:
    """Une piece detectee (vision.Piece : T_camera = face superieure) ->
    Boite dans le repere table de l'affinage."""
    if piece.type not in types or piece.face not in FACES:
        return None
    dims = dims_m(types, piece.type, piece.face)
    T = T_tc @ piece.T_camera
    R = T[:3, :3]
    c = T[:3, 3] - R[:, 2] * dims[2] / 2
    lacet = np.arctan2(R[1, 0], R[0, 0])
    tangage = np.arcsin(np.clip(R[2, 0], -1, 1))
    roulis = np.arctan2(R[2, 1], R[2, 2])
    if abs(tangage) < np.radians(1.0) and abs(roulis) < np.radians(1.0):
        tangage = roulis = 0.0
    return Boite(piece.type, piece.face, dims,
                 np.array([c[0], c[1], c[2], lacet, tangage, roulis]),
                 getattr(piece, "origine", "") or "detection")


# ---------------------------------------------------------------------------
# Vues et observation
# ---------------------------------------------------------------------------

class Grille:
    """Echantillons reguliers (pas `pas` pixels) dans une fenetre d'une
    image, avec leurs rayons dans le repere TABLE. o : centre optique
    (repere table) ; d : directions (ny x nx x 3), telles que t = profondeur
    dans le repere de la vue (composante z de la direction vue = 1)."""

    def __init__(self, x0, y0, x1, y1, pas, o, rayons_fn):
        self.x0, self.y0, self.pas = int(x0), int(y0), int(pas)
        us = np.arange(self.x0, int(x1), self.pas)
        vs = np.arange(self.y0, int(y1), self.pas)
        self.nx, self.ny = len(us), len(vs)
        self.u, self.v = np.meshgrid(us.astype(float), vs.astype(float))
        self.o = np.asarray(o, float)
        if self.nx and self.ny:
            self.d = rayons_fn(np.stack([self.u.ravel(), self.v.ravel()], 1)).reshape(
                self.ny, self.nx, 3)
        else:
            self.d = np.zeros((self.ny, self.nx, 3))

    def fenetre(self, uv: np.ndarray, marge: float = 2.0) -> Tuple[int, int, int, int]:
        """Indices (r0, r1, c0, c1) de la grille couvrant des pixels uv (N x 2)."""
        if not len(uv) or not np.all(np.isfinite(uv)):
            return 0, self.ny, 0, self.nx
        c0 = int(np.floor((uv[:, 0].min() - self.x0) / self.pas - marge))
        c1 = int(np.ceil((uv[:, 0].max() - self.x0) / self.pas + marge)) + 1
        r0 = int(np.floor((uv[:, 1].min() - self.y0) / self.pas - marge))
        r1 = int(np.ceil((uv[:, 1].max() - self.y0) / self.pas + marge)) + 1
        return (max(r0, 0), min(r1, self.ny), max(c0, 0), min(c1, self.nx))


class Observation:
    """
    Ce que la camera a mesure, pret a etre compare a une scene rendue (voir
    l'en-tete, etape 1). `capture` : dictionnaire (perception.charger_capture).
    `masques` : liste de (masque booleen dans l'image de la vue couleur,
    classe, confiance) ; `vue` : perception.VueCouleur de ces masques.
    """

    def __init__(self, capture: Dict[str, np.ndarray], types, masques=None,
                 vue: Optional[pc.VueCouleur] = None, pas: Optional[int] = None,
                 pas_couleur: Optional[int] = None, espacement: float = 0.0024,
                 aretes: bool = True, aretes_densite_max: float = 0.06,
                 aire_etranger: float = 6000.0,
                 table: Optional[pc.Table] = None):
        self.types = types
        D = np.asarray(capture["profondeur"], float).copy()
        D[D > vision.PROFONDEUR_MAX] = 0.0
        K = np.asarray(capture["K"], float)
        self.K = K
        self.H, self.W = D.shape
        self.table = table if table is not None else pc.analyser_table(D, K)
        tab = self.table
        self.T_ct = tab.T_ct
        self.T_tc = np.linalg.inv(tab.T_ct)
        hauteur = np.where(tab.valides, tab.hauteur, np.nan)
        self.sigma_table = max(float(tab.bruit), 0.0005)
        # zone de la table : la plus grande composante du plan, bouchee
        plan = (tab.valides & (np.abs(tab.hauteur) < max(0.003, 3 * self.sigma_table))).astype(np.uint8)
        nb, lab, st, _ = cv2.connectedComponentsWithStats(plan)
        self.zone = np.zeros(D.shape, bool)
        if nb > 1:
            grand = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
            cont, _ = cv2.findContours(grand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            z = np.zeros_like(grand)
            cv2.drawContours(z, [cv2.convexHull(max(cont, key=cv2.contourArea))], -1, 1, -1)
            self.zone = z > 0
        self.h_min = float(np.clip(4.0 * self.sigma_table, 0.003, 0.006))
        self.hauteur = hauteur
        haut = (tab.valides & self.zone & (tab.hauteur > self.h_min)
                & (tab.hauteur < 0.12)).astype(np.uint8)
        haut = cv2.morphologyEx(haut, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.saillant = haut > 0

        # vue couleur et masques
        self.vue = vue
        self.masques_img = []
        self.classes = []
        self.conf_masques = []
        if masques is not None and vue is not None:
            for m, c, s in masques:
                if m is not None and m.sum() >= 30:
                    self.masques_img.append(np.asarray(m, bool))
                    self.classes.append(c)
                    self.conf_masques.append(float(s))

        # region utile : ce qui depasse, et ce que les masques couvrent
        utile = self.saillant.copy()
        if self.masques_img:
            u, v, ok = pc.projeter_profondeur(D, K, vue)
            union = np.any(self.masques_img, axis=0)
            dans = np.zeros_like(ok)
            dans[ok] = union[np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
            utile |= dans & self.zone
        utile = cv2.dilate(utile.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
        utile &= self.zone
        ys, xs = np.nonzero(utile)
        self.vide = len(xs) == 0
        if self.vide:
            x0 = y0 = x1 = y1 = 0
        else:
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        o = self.T_tc[:3, 3]

        def rayons_profondeur(uv):
            r = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1],
                          np.ones(len(uv))], 1)
            return r @ self.T_tc[:3, :3].T

        # PAS DE LA GRILLE : environ `espacement` (2.4 mm) sur la table entre
        # deux echantillons, quelle que soit la distance de la camera (2
        # pixels a 0.5 m, 3 a 0.3 m) : meme precision, cout borne
        z_tab = float(-o[2] / rayons_profondeur(np.array([[K[0, 2], (y0 + y1) / 2.0]]))[0, 2]) \
            if not self.vide else 0.5
        if pas is None:
            pas = int(np.clip(round(espacement * K[0, 0] / max(z_tab, 0.1)), 2, 4))
        if pas_couleur is None and vue is not None:
            pas_couleur = int(np.clip(round(espacement * vue.K[0, 0] / max(z_tab, 0.1)), 2, 6))
        self.g = Grille(x0, y0, x1, y1, pas, o, rayons_profondeur)
        g = self.g
        iy = g.v.astype(int)
        ix = g.u.astype(int)
        h = hauteur[iy, ix] if g.nx and g.ny else np.zeros((0, 0))
        dz = g.d[..., 2] if g.nx and g.ny else np.zeros((0, 0))
        # profondeur corrigee : celle qui donne la hauteur corrigee (la table
        # mesuree devient le plan z = 0 du rendu)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.t_table = np.where(dz < -1e-9, -o[2] / dz, np.inf)
            self.z_obs = (h - o[2]) / dz
        self.connu = np.isfinite(self.z_obs) & (self.z_obs > 0) & (utile[iy, ix] if g.nx else False)
        self.z_obs = np.where(self.connu, self.z_obs, 0.0)
        # BRUIT DE PROFONDEUR sigma = k z^2 (stereo), k etalonne sur la table
        # de cette capture : l'ecart de hauteur h d'un point de la table vient
        # d'une erreur de profondeur h / |dz| le long du rayon (dz : pente du
        # rayon ; en vue rasante, une erreur de profondeur change peu la
        # hauteur, il ne faut pas confondre les deux)
        vv, uu = np.nonzero(tab.valides & self.zone & (np.abs(tab.hauteur) < 0.01)
                            & ~(cv2.dilate(haut, np.ones((9, 9), np.uint8)) > 0))
        if len(uu) > 200:
            sel = np.linspace(0, len(uu) - 1, min(len(uu), 20000)).astype(int)
            uu, vv = uu[sel], vv[sel]
            dzt = rayons_profondeur(np.stack([uu, vv], 1).astype(float))[:, 2]
            zt = (tab.hauteur[vv, uu] - o[2]) / dzt
            e = tab.hauteur[vv, uu] / np.abs(dzt) / zt ** 2
            self.k_z2 = max(1.4826 * float(np.median(np.abs(e - np.median(e)))), 5e-4)
        else:
            self.k_z2 = 0.003
        zz = np.where(self.connu, self.z_obs, np.where(np.isfinite(self.t_table), self.t_table, 0.5))
        self.sigma = np.maximum(self.k_z2 * zz ** 2, 0.0005)
        # pixels VOLANTS (lissage stereo d'un saut de profondeur) : a mi-
        # hauteur, nettement sous le plus haut ET au-dessus du plus bas de
        # leur voisinage ; poids reduit. Le dessus d'une piece etroite, lui,
        # est au niveau du plus haut : il garde tout son poids.
        Hn = np.nan_to_num(hauteur, nan=0.0).astype(np.float32)
        noyau = np.ones((3, 3), np.uint8)
        haut_loc, bas_loc = cv2.dilate(Hn, noyau), cv2.erode(Hn, noyau)
        volant = ((Hn - bas_loc) > 0.0035) & ((haut_loc - Hn) > 0.0035)
        self.poids = np.where(self.connu, 1.0, 0.0)
        if g.nx:
            self.poids[volant[iy, ix] & self.connu] = 0.35
        self.points = o + self.z_obs[..., None] * g.d if g.nx else np.zeros((0, 0, 3))
        # OBJETS ETRANGERS candidats : surfaces lisses (sans saut de plus de
        # 2.5 mm entre voisins) et PLANES (a 2 mm pres), de plus de 40 mm de
        # large et de 6000 mm^2 (quatre pieces a plat cote a cote en font
        # 4600) : livre, planche, boite. Affinage.etrangers ne garde que
        # celles que les pieces proposees n'expliquent pas (un groupe de
        # pieces en contact a la meme hauteur fait aussi une grande surface
        # plane) ; on n'y propose pas de pieces.
        self.aire_etranger = aire_etranger
        self.etranger_lab = np.zeros(D.shape, np.int32)
        if self.saillant.any():
            Hs = cv2.medianBlur(np.nan_to_num(hauteur, nan=0.0).astype(np.float32), 5)
            k3 = np.ones((3, 3), np.uint8)
            lisse = self.saillant & ((cv2.dilate(Hs, k3) - cv2.erode(Hs, k3)) < 0.0025)
            nb_, lab_, st_, _ = cv2.connectedComponentsWithStats(lisse.astype(np.uint8))
            P_t = tab.P @ self.T_tc[:3, :3].T + self.T_tc[:3, 3]
            n_e = 0
            for k_ in range(1, nb_):
                if st_[k_, cv2.CC_STAT_AREA] >= 300 and _etranger(P_t[lab_ == k_], aire_etranger):
                    n_e += 1
                    m_ = cv2.dilate((lab_ == k_).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                    self.etranger_lab[m_ & (self.etranger_lab == 0)] = n_e
        self.etranger_lab = self.etranger_lab[iy, ix] if g.nx else np.zeros((0, 0), np.int32)
        self.etranger = np.zeros(self.etranger_lab.shape, bool)
        # loin du BORD de la table : ce qui depasse au bord (nappe qui tombe,
        # dossier de chaise, autre meuble) n'y est pas propose comme piece
        interieur = cv2.erode(self.zone.astype(np.uint8), np.ones((15, 15), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0
        self.interieur = interieur[iy, ix] if g.nx else np.zeros((0, 0), bool)
        self.echelle = float(pas * pas)

        # ARETES de l'infrarouge (meme repere que la profondeur) : les points
        # du projecteur effaces (ouverture), Canny, distance a l'arete la plus
        # proche dans 4 orientations. Une image trop texturee (bois veine,
        # points du projecteur trop gros de pres) n'est pas fiable : si la
        # table elle-meme porte trop d'aretes, on n'en tient pas compte.
        self.aretes = None
        self.d_max = 6.0
        self.densite_aretes = float("nan")
        if aretes and "infrarouge" in capture and g.nx and not self.vide:
            ir = np.asarray(capture["infrarouge"])
            if ir.ndim == 3:
                ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
            ir = ir.astype(np.uint8)
            o_ = cv2.morphologyEx(ir, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            o_ = cv2.GaussianBlur(o_, (0, 0), 1.5)
            med = float(np.median(o_[self.zone])) if self.zone.any() else float(np.median(o_))
            E = cv2.Canny(o_, 0.3 * med + 5, 0.8 * med + 10) > 0
            loin = self.zone & ~(cv2.dilate(haut, np.ones((21, 21), np.uint8)) > 0)
            self.densite_aretes = float(E[loin].mean()) if loin.any() else 1.0
            if self.densite_aretes <= aretes_densite_max:
                gx = cv2.Sobel(o_.astype(np.float32), cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(o_.astype(np.float32), cv2.CV_32F, 0, 1)
                ang = np.arctan2(gy, gx) % np.pi
                dts = []
                for kk in range(4):
                    a0 = kk * np.pi / 4
                    da = np.abs((ang - a0 + np.pi / 2) % np.pi - np.pi / 2)
                    Ek = (E & (da < np.radians(35))).astype(np.uint8)
                    dt = cv2.distanceTransform(1 - Ek, cv2.DIST_L2, 3)
                    dts.append(dt[iy, ix])
                self.aretes = np.stack(dts).astype(np.float32)

        # grille couleur (masques)
        self.gc = None
        if self.masques_img and not self.vide:
            T_tv = np.linalg.inv(vue.T_cp @ self.T_ct)
            Wc, Hc = vue.taille
            q = self.points[self.connu].reshape(-1, 3)
            uv = self.projeter_couleur(q) if len(q) else np.zeros((0, 2))
            boites = [np.array([[xs_.min(), ys_.min()], [xs_.max(), ys_.max()]])
                      for xs_, ys_ in ((np.nonzero(m)[1], np.nonzero(m)[0]) for m in self.masques_img)]
            uv = np.concatenate([uv] + boites) if len(uv) else np.concatenate(boites)
            cx0 = max(int(uv[:, 0].min()) - 20, 0)
            cx1 = min(int(uv[:, 0].max()) + 20, Wc)
            cy0 = max(int(uv[:, 1].min()) - 20, 0)
            cy1 = min(int(uv[:, 1].max()) + 20, Hc)

            def rayons_couleur(uv_):
                return pc.rayons(uv_, vue) @ T_tv[:3, :3].T

            self.gc = Grille(cx0, cy0, cx1, cy1, pas_couleur, T_tv[:3, 3], rayons_couleur)
            gc = self.gc
            self.M = np.stack([m[gc.v.astype(int), gc.u.astype(int)] for m in self.masques_img])
            # un pixel couleur vaut (f_profondeur / f_couleur)^2 pixel de profondeur
            self.echelle_c = float(pas_couleur ** 2) * (K[0, 0] / vue.K[0, 0]) ** 2

    # -- projections ------------------------------------------------------
    def projeter_profondeur(self, Q: np.ndarray) -> np.ndarray:
        """Points (N x 3, repere table) -> pixels de l'image de profondeur."""
        q = Q @ self.T_ct[:3, :3].T + self.T_ct[:3, 3]
        z = np.maximum(q[:, 2], 1e-6)
        K = self.K
        return np.stack([K[0, 0] * q[:, 0] / z + K[0, 2], K[1, 1] * q[:, 1] / z + K[1, 2]], 1)

    def projeter_couleur(self, Q: np.ndarray) -> np.ndarray:
        T = self.vue.T_cp @ self.T_ct
        q = Q @ T[:3, :3].T + T[:3, 3]
        return pc.projeter(q, self.vue)

    @property
    def avec_masques(self) -> bool:
        return self.gc is not None


# ---------------------------------------------------------------------------
# Parametres du cout
# ---------------------------------------------------------------------------

@dataclass
class Reglages:
    """Poids et seuils de l'affinage (unites : pixels de profondeur mal
    expliques ; voir l'en-tete)."""
    tau: float = 2.5              # residu (en ecarts-types) ou la perte vaut 1/2
    flou: int = 1                 # lisser le rendu comme la camera lisse les sauts :
                                  # nombre de passes [1 2 1] (0 : rendu net)
    flou_auto: bool = True        # choisir ce nombre (0 a 3) sur la capture
    masques: float = 0.5          # par pixel en desaccord avec le masque associe
    classe: float = 40.0          # type contraire a la classe du masque (x confiance)
    aretes: float = 0.0           # par pixel de contour (chanfrein, infrarouge) ; 0 : non
    aretes_densite_max: float = 0.06  # au-dela (texture, points du projecteur) : ignorees
    k_penetration: float = 60.0   # par mm^2 de penetration (au-dela de 0.3 mm)
    k_air: float = 15.0           # par mm^2 d'ecart a l'appui le plus proche
    k_stabilite: float = 8.0      # par mm du centre de gravite hors des appuis
    air_max: float = 4.0          # mm^2 : au-dela, etre en l'air ne coute pas plus
    stabilite_max: float = 60.0
    k_inclinaison: float = 0.6    # par degre de tangage + roulis
    roulis_max: float = np.radians(40.0)
    tangage_max: float = np.radians(60.0)
    gain_min: float = 35.0        # pixels expliques au minimum par une piece
    part_min: float = 0.35        # part de ses pixels visibles qu'elle doit expliquer
    part_min_masque: float = 0.15  # ... si un masque du reseau la recouvre bien
    part_min_bord: float = 0.6    # ... si elle est proposee au bord de la table
    confiance_min: float = 0.15   # en dessous, la piece n'est pas rendue
    contredit_max: float = 0.35   # part maximale de pixels "vus a travers"
    debord: bool = True           # rejeter une piece taillee dans une surface plus grande
    appui_tol: float = 0.0025     # m : un point du dessous a moins est en appui
    balayages: int = 2
    repropositions: int = 3
    variantes: bool = True
    debout: bool = True


def rho(u: np.ndarray, tau: float) -> np.ndarray:
    """Perte de Geman-McClure normalisee : ~ (u/tau)^2 pres de 0, bornee par 1."""
    u2 = u * u
    return u2 / (u2 + tau * tau)


# ---------------------------------------------------------------------------
# La scene hypothetique
# ---------------------------------------------------------------------------

class Scene:
    """
    Scene hypothetique : des Boites, leurs cartes de profondeur rendues
    (z-buffer par piece) dans les grilles de profondeur et de couleur, et le
    cout de chaque piece par rapport a son absence (voir l'en-tete).
    """

    def __init__(self, obs: Observation, reglages: Optional[Reglages] = None):
        self.obs = obs
        self.r = replace(reglages) if reglages is not None else Reglages()
        self.boites: List[Boite] = []
        g = obs.g
        self.Td = np.zeros((0, g.ny, g.nx), np.float32)
        self.Tc = (np.zeros((0, obs.gc.ny, obs.gc.nx), np.float32)
                   if obs.avec_masques else None)
        self.evaluations = 0
        self.nouvelle_libre = False

    # -- rendu --------------------------------------------------------------
    def _rendre(self, grille: Grille, P: np.ndarray, dims: np.ndarray, projeter,
                fen=None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """t (B x n) des poses P dans la fenetre de la grille qui les couvre."""
        R = rotations(P)
        demi = np.asarray(dims, float) / 2.0
        if fen is None:
            Q = coins(P[:, :3], R, demi).reshape(-1, 3)
            fen = grille.fenetre(projeter(Q))
        r0, r1, c0, c1 = fen
        if r1 <= r0 or c1 <= c0:
            return np.zeros((len(P), 0)), fen
        d = grille.d[r0:r1, c0:c1].reshape(-1, 3)
        return lancer(grille.o, d, P[:, :3], R, demi), fen

    def _carte(self, grille, b: Boite, projeter) -> np.ndarray:
        T = np.full((grille.ny, grille.nx), np.inf, np.float32)
        t, (r0, r1, c0, c1) = self._rendre(grille, b.p[None], b.dims, projeter)
        if t.size:
            T[r0:r1, c0:c1] = t[0].reshape(r1 - r0, c1 - c0)
        return T

    def maj(self, i: int) -> None:
        """Re-rend la piece i apres un changement de pose."""
        b = self.boites[i]
        self.Td[i] = self._carte(self.obs.g, b, self.obs.projeter_profondeur)
        if self.Tc is not None:
            self.Tc[i] = self._carte(self.obs.gc, b, self.obs.projeter_couleur)

    def ajouter(self, b: Boite) -> int:
        self.boites.append(b)
        g = self.obs.g
        self.Td = np.concatenate([self.Td, np.full((1, g.ny, g.nx), np.inf, np.float32)])
        if self.Tc is not None:
            gc = self.obs.gc
            self.Tc = np.concatenate([self.Tc, np.full((1, gc.ny, gc.nx), np.inf, np.float32)])
        self.maj(len(self.boites) - 1)
        return len(self.boites) - 1

    def retirer(self, i: int) -> None:
        del self.boites[i]
        self.Td = np.delete(self.Td, i, axis=0)
        if self.Tc is not None:
            self.Tc = np.delete(self.Tc, i, axis=0)

    def sauver(self):
        """Etat complet (pour essayer un changement et revenir en arriere)."""
        return ([replace(b, dims=np.array(b.dims), p=np.array(b.p)) for b in self.boites],
                self.Td.copy(), None if self.Tc is None else self.Tc.copy())

    def restaurer(self, etat) -> None:
        boites, Td, Tc = etat
        self.boites = [replace(b, dims=np.array(b.dims), p=np.array(b.p)) for b in boites]
        self.Td = Td.copy()
        self.Tc = None if Tc is None else Tc.copy()

    def autres(self, i: Optional[int]) -> List[int]:
        return [j for j in range(len(self.boites)) if j != i]

    def energie_profondeur(self, flou: int) -> float:
        """Cout de profondeur de toute la scene (pixels), rendu lisse de
        `flou` passes."""
        obs = self.obs
        zp = self.profondeur_rendue()
        if flou:
            zp = _lisser(zp[None], flou)[0]
        u = (obs.z_obs - zp) / obs.sigma
        return float((rho(u, self.r.tau) * obs.poids).sum() * obs.echelle)

    def profondeur_rendue(self, sauf: Optional[int] = None) -> np.ndarray:
        """Profondeur de la scene (grille de profondeur), sans la piece `sauf`."""
        idx = self.autres(sauf)
        t = self.obs.t_table
        if idx:
            t = np.minimum(t, self.Td[idx].min(axis=0))
        return t

    # -- cout ---------------------------------------------------------------
    def energie(self, i: Optional[int], P: np.ndarray, dims: np.ndarray,
                detail: bool = False, type_: Optional[str] = None):
        """
        Cout de B poses (P : B x 6) d'une piece de dimensions `dims` (m), les
        autres pieces fixes, PAR RAPPORT A SON ABSENCE. i : indice de la
        piece dans la scene (None : piece nouvelle). Rend (B,) ou, avec
        detail, (total, dict des termes).
        """
        self.evaluations += len(P)
        obs, r = self.obs, self.r
        self._type_courant = type_ if type_ is not None else (
            self.boites[i].type if i is not None else None)
        P = np.atleast_2d(np.asarray(P, float))
        idx = self.autres(i)
        # profondeur : seulement la fenetre de la piece
        t, (r0, r1, c0, c1) = self._rendre(obs.g, P, dims, obs.projeter_profondeur)
        e_prof = np.zeros(len(P))
        e_aret = np.zeros(len(P))
        vis = np.zeros(len(P))
        contre = np.zeros(len(P))
        if t.size:
            sl = (slice(r0, r1), slice(c0, c1))
            t_aut = obs.t_table[sl]
            if idx:
                t_aut = np.minimum(t_aut, self.Td[idx][:, r0:r1, c0:c1].min(axis=0))
            t_aut = t_aut.reshape(-1)
            z = obs.z_obs[sl].reshape(-1)
            s = obs.sigma[sl].reshape(-1)
            w = obs.poids[sl].reshape(-1)
            zp = np.minimum(t, t_aut[None])
            t_net = t_aut
            if r.flou:
                # la camera LISSE les sauts de profondeur (la stereo agrege une
                # fenetre de pixels) : on lisse le rendu de la meme facon
                # avant de le comparer, sinon les rampes des bords tirent les
                # pieces et les enfoncent
                hh, ww = r1 - r0, c1 - c0
                zp = _lisser(zp.reshape(len(P), hh, ww), r.flou).reshape(len(P), -1)
                t_aut = _lisser(t_aut.reshape(1, hh, ww), r.flou).reshape(-1)
            sans = rho((z - t_aut) / s, r.tau)
            avec = rho((z[None] - zp) / s[None], r.tau)
            e_prof = ((avec - sans[None]) * w[None]).sum(axis=1) * obs.echelle
            devant = (t < t_net[None]) & (w[None] > 0)
            if obs.aretes is not None and r.aretes > 0:
                e_aret = self._energie_aretes(t < t_net[None], r0, r1, c0, c1, len(P))
            vis = (devant * w[None]).sum(axis=1) * obs.echelle
            # vu A TRAVERS la piece : la mesure est nettement plus loin
            loin = (z[None] - t) > (3 * s[None] + 0.003)
            contre = ((devant & loin) * w[None]).sum(axis=1) * obs.echelle
        # masques
        e_masq = np.zeros(len(P))
        if obs.avec_masques and r.masques > 0:
            e_masq = self._energie_masques(i, P, dims)
            # la CLASSE du masque associe (le reseau voit la couleur de la
            # piece) : un type contraire coute, sans rien imposer
            if i is not None and self.boites[i].masque >= 0 and r.classe > 0:
                k = self.boites[i].masque
                if obs.classes[k] != self._type_courant:
                    e_masq = e_masq + r.classe * obs.conf_masques[k]
        e_phys = self.physique(i, P, dims)
        incl = np.degrees(np.abs(P[:, 4]) + np.abs(P[:, 5])) * r.k_inclinaison
        total = e_prof + e_masq + e_phys + incl + e_aret
        if detail:
            return total, {"profondeur": e_prof, "masques": e_masq, "physique": e_phys,
                           "inclinaison": incl, "aretes": e_aret, "visibles": vis,
                           "contredits": contre}
        return total

    def _energie_aretes(self, V, r0, r1, c0, c1, B) -> np.ndarray:
        """
        ARETES (chanfrein oriente) : le contour de la partie visible rendue
        de la piece doit tomber sur des aretes de l'image infrarouge de meme
        orientation. Chaque echantillon du contour coute
        min(d, d_max) / d_max - 1/2 (d : distance, en pixels, a l'arete la
        plus proche de meme orientation) : un contour bien place rapporte,
        un contour dans le vide coute.
        """
        obs = self.obs
        hh, ww = r1 - r0, c1 - c0
        if hh < 3 or ww < 3:
            return np.zeros(B)
        V = V.reshape(B, hh, ww).astype(np.float32)
        P = np.pad(V, ((0, 0), (1, 1), (1, 1)), mode="edge")
        gx = P[:, 1:-1, 2:] - P[:, 1:-1, :-2]
        gy = P[:, 2:, 1:-1] - P[:, :-2, 1:-1]
        bord = (V > 0) & ((np.abs(gx) + np.abs(gy)) > 0)
        # pas les bords de la fenetre
        bord[:, 0, :] = bord[:, -1, :] = False
        bord[:, :, 0] = bord[:, :, -1] = False
        ang = np.arctan2(gy, gx) % np.pi
        k = np.round(ang / (np.pi / 4)).astype(int) % 4
        dt = obs.aretes[:, r0:r1, c0:c1]                               # 4 x h x w
        d = np.take_along_axis(np.broadcast_to(dt[None], (B,) + dt.shape),
                               k[:, None], axis=1)[:, 0]
        c = np.where(bord, np.minimum(d, obs.d_max) / obs.d_max - 0.5, 0.0)
        return self.r.aretes * obs.g.pas * c.sum(axis=(1, 2))

    def _energie_masques(self, i, P, dims) -> np.ndarray:
        """Desaccord de la partie visible avec le masque associe a la piece,
        en difference avec son absence : sum vis * (1 - 2 M), plus l'effet
        de l'occultation sur la piece de devant (si elle a un masque)."""
        obs = self.obs
        k = self.boites[i].masque if i is not None else -1
        t, (r0, r1, c0, c1) = self._rendre(obs.gc, P, dims, obs.projeter_couleur)
        if not t.size:
            return np.zeros(len(P))
        idx = self.autres(i)
        n = (r1 - r0) * (c1 - c0)
        e_moi = np.zeros(n) if k < 0 else (1.0 - 2.0 * obs.M[k, r0:r1, c0:c1].reshape(-1))
        e_aut = np.zeros(n)
        t_aut = np.full(n, np.inf)
        if idx:
            Ta = self.Tc[idx][:, r0:r1, c0:c1].reshape(len(idx), -1)
            devant = Ta.argmin(axis=0)
            t_aut = Ta.min(axis=0)
            for a, j in enumerate(idx):
                kj = self.boites[j].masque
                if kj < 0:
                    continue
                sel = (devant == a) & np.isfinite(t_aut)
                e_aut[sel] = 1.0 - 2.0 * obs.M[kj, r0:r1, c0:c1].reshape(-1)[sel]
        vis = t < t_aut[None]
        return self.r.masques * obs.echelle_c * (vis * (e_moi - e_aut)[None]).sum(axis=1)

    def physique(self, i: Optional[int], P: np.ndarray, dims: np.ndarray,
                 idx: Optional[List[int]] = None) -> np.ndarray:
        """Penalites physiques (voir l'en-tete) de B poses, les autres fixes."""
        r = self.r
        idx = self.autres(i) if idx is None else idx
        R = rotations(P)
        demi = np.asarray(dims, float) / 2.0
        C = P[:, :3]
        e = np.zeros(len(P))
        cs = coins(C, R, demi)
        # table
        sous = np.maximum(-cs[:, :, 2].min(axis=1) * 1000.0 - 0.3, 0.0)
        e += r.k_penetration * sous ** 2
        # autres boites (sauf pendant la proposition d'une piece nouvelle :
        # elle se pose d'abord ou la MESURE la place, puis ses voisines
        # se re-ajustent)
        proches = self._proches(C, demi, idx)
        if proches and not (i is None and self.nouvelle_libre):
            Cj, Rj, hj = self._tableaux(proches)
            pen = penetrations(C, R, demi, Cj, Rj, hj) * 1000.0           # B x M
            e += r.k_penetration * (np.maximum(pen - 0.3, 0.0) ** 2).sum(axis=1)
        # appui : points du dessous, rayons verticaux vers le bas
        e += self._appui(C, R, demi, proches)[0]
        return e

    def _tableaux(self, idx):
        bs = [self.boites[j] for j in idx]
        return (np.array([b.p[:3] for b in bs]), np.array([b.R() for b in bs]),
                np.array([b.demi for b in bs]))

    def _proches(self, C, demi, idx) -> List[int]:
        """Les boites dont la sphere englobante touche celle d'une des poses
        (test grossier avant les tests exacts)."""
        if not idx:
            return []
        rayon = float(np.linalg.norm(demi))
        c = C.mean(axis=0)
        etendue = float(np.linalg.norm(C - c, axis=1).max())
        return [j for j in idx
                if np.linalg.norm(self.boites[j].p[:3] - c)
                < rayon + etendue + np.linalg.norm(self.boites[j].demi) + 0.005]

    #: points du dessous (coordonnees reduites) : grille 13 x 3 ; le roulis
    #: etant borne a 40 deg, la face du dessous reste la plus basse
    _DESSOUS = np.array([[sx, sy, -1.0] for sx in np.linspace(-1, 1, 13)
                         for sy in (-1.0, 0.0, 1.0)])
    #: points des deux bouts (appuis lateraux)
    _BOUTS = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 0.0, 1.0)
                       for sz in (-1.0, 0.0, 1.0)])

    def _appui(self, C, R, demi, idx):
        """(penalite (B,), appuis : liste par pose des indices touches -1 :
        table). Le point le plus bas de chaque verticale sous le dessous de
        la boite : la table (z = 0) ou le dessus d'une autre boite."""
        r = self.r
        B = len(C)
        loc = self._DESSOUS[None] * demi[None, None, :] if np.ndim(demi) == 1 else \
            self._DESSOUS[None] * demi[:, None, :]
        pts = np.einsum("bkj,bij->bki", loc, R) + C[:, None, :]      # B x 39 x 3
        n = pts.shape[1]
        sol = np.zeros((B, n))
        qui = np.full((B, n), -1)
        tol = r.appui_tol
        flat = pts.reshape(-1, 3)
        o = flat + np.array([0, 0, tol])
        if idx:
            Cj, Rj, hj = self._tableaux(idx)
            bas = np.broadcast_to(np.array([0.0, 0.0, -1.0]), o.shape)
            t = lancer(o, bas, Cj, Rj, hj)                                  # M x N
            z = o[None, :, 2] - t
            z = np.where(np.isfinite(z), z, -np.inf)
            k = z.argmax(axis=0)
            zb = z[k, np.arange(z.shape[1])].reshape(B, n)
            mieux = zb > sol
            sol = np.where(mieux, zb, sol)
            qui = np.where(mieux, np.asarray(idx)[k].reshape(B, n), qui)
        ecart = (pts[:, :, 2] - sol) * 1000.0                           # mm
        touche = ecart < tol * 1000.0                                    # B x n
        # appuis LATERAUX : un bout de la piece contre le flanc d'une autre
        # (piece inclinee "contre" une piece) ; points des deux bouts a
        # moins de tol d'une boite
        nb_ = len(self._BOUTS)
        cote = np.zeros((B, nb_), bool)
        qui_c = np.full((B, nb_), -1)
        if idx:
            locb = self._BOUTS[None] * np.broadcast_to(demi, (B, 3))[:, None, :]
            pb = np.einsum("bkj,bij->bki", locb, R) + C[:, None, :]       # B x 18 x 3
            Cj, Rj, hj = self._tableaux(idx)
            ql = np.einsum("bkmi,mij->bkmj", pb[:, :, None, :] - Cj[None, None], Rj)
            dedans = np.all(np.abs(ql) <= hj[None, None] + tol, axis=3)   # B x 18 x M
            cote = dedans.any(axis=2)
            qui_c = np.where(cote, np.asarray(idx)[dedans.argmax(axis=2)], -1)
        aucun = ~touche.any(axis=1)
        # en l'air : penalite BORNEE (la porteuse peut etre une piece encore
        # non trouvee, ou cachee : la physique guide, la mesure decide)
        e = np.where(aucun, r.k_air * np.minimum((ecart.min(axis=1) - tol * 1000.0) ** 2,
                                                 r.air_max), 0.0)
        # STABILITE, en coordonnees de la boite : le centre de gravite (0, 0)
        # doit etre entre les appuis le long du grand cote ET en travers
        # (rectangle englobant des appuis ; mm hors de ce rectangle)
        touche = np.concatenate([touche, cote], axis=1)
        loc_mm = np.concatenate([self._DESSOUS, self._BOUTS])[None, :, :2] * \
            (np.broadcast_to(demi, (B, 3))[:, None, :2] * 1000.0)
        big = 1e6
        sx = np.where(touche, loc_mm[..., 0], np.nan)
        sy = np.where(touche, loc_mm[..., 1], np.nan)
        with np.errstate(all="ignore"):
            hors = (np.maximum(np.nanmin(np.nan_to_num(sx, nan=big), axis=1), 0)
                    + np.maximum(-np.nanmax(np.nan_to_num(sx, nan=-big), axis=1), 0)
                    + np.maximum(np.nanmin(np.nan_to_num(sy, nan=big), axis=1), 0)
                    + np.maximum(-np.nanmax(np.nan_to_num(sy, nan=-big), axis=1), 0))
        e += np.where(aucun, 0.0, np.minimum(r.k_stabilite * np.maximum(hors - 1.0, 0.0),
                                             r.stabilite_max))
        appuis = None
        if B == 1:
            bas_ = set(int(q) for q in qui[0, touche[0, :n]])
            lat = set(int(q) for q in qui_c[0, cote[0]]) - bas_
            appuis = [(sorted(bas_), sorted(lat))]
        return e, appuis

    def appuis(self, i: int) -> Tuple[List[int], List[int]]:
        """(appuis par dessous, appuis lateraux) de la piece i : indices des
        boites (-1 : la table)."""
        b = self.boites[i]
        return self._appui(b.p[None, :3], b.R()[None], b.demi, self.autres(i))[1][0]

    # -- optimisation -----------------------------------------------------
    def optimiser(self, i: int, pas=(0.004, 0.004, 0.003, np.radians(8), np.radians(6), np.radians(6)),
                  fin=(0.0003, 0.0003, 0.0003, np.radians(0.5), np.radians(0.5), np.radians(0.5)),
                  libres=(0, 1, 2, 3, 4, 5), iterations: int = 60, sauts: bool = False) -> float:
        """Recherche par motifs sur la pose de la piece i (voir l'en-tete)."""
        b = self.boites[i]
        p, e = self._motifs(i, b.p, b.dims, pas, fin, libres, iterations, sauts)
        b.p = p
        self.maj(i)
        return e

    def _motifs(self, i, p, dims, pas, fin, libres, iterations, sauts: bool = False,
                type_: Optional[str] = None):
        r = self.r
        p = np.array(p, float)
        pas = np.array(pas, float)
        fin = np.array(fin, float)
        libres = [k for k in libres if pas[k] > 0]
        e = float(self.energie(i, p[None], dims, type_=type_)[0])
        pas0 = pas.copy()
        relances = 2 if sauts else 0
        if sauts:
            p, e = self._sauter(i, p, dims, e, type_)
        for _ in range(iterations * (1 + relances)):
            cands = []
            for k in libres:
                for sg in (1.0, -1.0):
                    q = p.copy()
                    q[k] += sg * pas[k]
                    cands.append(q)
            cands += _pivots(p, dims, pas, libres)
            Q = np.array(cands)
            Q[:, 4] = np.clip(Q[:, 4], -r.tangage_max, r.tangage_max)
            Q[:, 5] = np.clip(Q[:, 5], -r.roulis_max, r.roulis_max)
            E = self.energie(i, Q, dims, type_=type_)
            k = int(np.argmin(E))
            if E[k] < e - 1e-3:
                # on continue dans la meme direction tant que ca descend
                dq = Q[k] - p
                p, e = Q[k], float(E[k])
                # plus loin dans la meme direction (un seul lot)
                Q2 = p[None] + np.arange(1, 5)[:, None] * dq[None]
                Q2[:, 4] = np.clip(Q2[:, 4], -r.tangage_max, r.tangage_max)
                Q2[:, 5] = np.clip(Q2[:, 5], -r.roulis_max, r.roulis_max)
                E2 = self.energie(i, Q2, dims, type_=type_)
                k2 = int(np.argmin(E2))
                if E2[k2] < e - 1e-3:
                    p, e = Q2[k2], float(E2[k2])
            else:
                pas[libres] /= 2.0
                if np.all(pas[libres] < fin[libres]):
                    # converge : un grand saut (glisser d'un quart de
                    # longueur...) fait-il mieux ? alors on repart de la
                    if relances > 0:
                        relances -= 1
                        p2, e2 = self._sauter(i, p, dims, e, type_)
                        if e2 < e - 1e-3:
                            p, e = p2, e2
                            pas = pas0.copy()
                            continue
                    break
        return p, e

    def _sauter(self, i, p, dims, e, type_=None):
        """Grands deplacements (une demi ou un quart de longueur le long du
        grand cote, une largeur de cote, un quart de tour) : une piece posee
        A CHEVAL sur deux pieces bout a bout, ou sur la mauvaise de deux
        voisines, sort ainsi de son optimum local."""
        L, l, _ = dims
        a = p[3]
        u = np.array([np.cos(a), np.sin(a)])
        nrm = np.array([-u[1], u[0]])
        cands = []
        for d in (L / 2, -L / 2, L / 4, -L / 4, L / 8, -L / 8):
            q = p.copy()
            q[:2] += d * u
            cands.append(q)
        for d in (l, -l):
            q = p.copy()
            q[:2] += d * nrm
            cands.append(q)
        q = p.copy()
        q[3] += np.pi / 2
        cands.append(q)
        E = self.energie(i, np.array(cands), dims, type_=type_)
        k = int(np.argmin(E))
        if E[k] < e - 1e-3:
            return cands[k], float(E[k])
        return p, e

    # -- bilan par piece ----------------------------------------------------
    def bilan(self, i: int) -> Dict[str, float]:
        b = self.boites[i]
        _, d = self.energie(i, b.p[None], b.dims, detail=True)
        return {k: float(v[0]) for k, v in d.items()}

    def debord(self, i: int) -> Tuple[float, int, bool]:
        """
        La surface mesuree CONTINUE-t-elle a la meme hauteur au-dela des
        bords de la face superieure de la piece i (a 4, 8 et 12 mm, hors des
        autres pieces, des pixels caches, et a plus de 3 pixels de sa
        silhouette) ? Une piece taillee dans une surface plus grande
        (planche, livre, rebord) n'est pas une piece : une vraie piece a des
        bords, au-dela la mesure descend. Rend (part du pourtour qui
        continue, nombre de cotes sur 4 qui continuent, les DEUX bouts
        continuent). Une piece contre une voisine non encore trouvee
        continue d'un cote ; un morceau de planche, de trois, ou des deux
        bouts.
        """
        obs = self.obs
        b = self.boites[i]
        L, l, h = b.dims
        pts, cote = [], []
        for m in (0.004, 0.008, 0.012):
            a, c = L / 2 + m, l / 2 + m
            n_l = max(int(2 * a / 0.004), 2)
            n_c = max(int(2 * c / 0.004), 2)
            for x in np.linspace(-a, a, n_l)[1:-1]:
                pts += [(x, -c), (x, c)]
                cote += [2, 3]
            for y in np.linspace(-c, c, n_c)[1:-1]:
                pts += [(-a, y), (a, y)]
                cote += [0, 1]
        cote = np.array(cote)
        loc = np.c_[np.array(pts), np.full(len(pts), h / 2)]
        Q = loc @ b.R().T + b.p[:3]
        garde = np.ones(len(Q), bool)
        # pas dans une autre piece (contact)
        autres = self.autres(i)
        if autres:
            Cj, Rj, hj = self._tableaux(autres)
            ql = np.einsum("nmk,mkj->nmj", Q[:, None, :] - Cj[None], Rj)
            garde &= ~np.all(np.abs(ql) <= hj[None] + 0.002, axis=2).any(axis=1)
        uvf = obs.projeter_profondeur(Q)
        uv = np.round(uvf).astype(int)
        garde &= (uv[:, 0] >= 0) & (uv[:, 0] < obs.W) & (uv[:, 1] >= 0) & (uv[:, 1] < obs.H)
        # a au moins 3 pixels de la silhouette de la piece : plus pres, le
        # lissage de la camera prolonge la piece (surtout en vue rasante,
        # ou quelques millimetres font moins d'un pixel)
        sil = cv2.convexHull(obs.projeter_profondeur(
            coins(b.p[None, :3], b.R()[None], b.demi)[0]).astype(np.float32))
        garde &= np.array([cv2.pointPolygonTest(sil, (float(x), float(y)), True) < -3.0
                           for x, y in uvf])
        uv = np.clip(uv, 0, [obs.W - 1, obs.H - 1])
        h_obs = obs.hauteur[uv[:, 1], uv[:, 0]]
        garde &= np.isfinite(h_obs)
        # cache par une piece de la scene (plus proche de la camera) : on
        # ne sait pas
        if self.boites and garde.any():
            q = Q @ obs.T_ct[:3, :3].T + obs.T_ct[:3, 3]
            r = np.stack([(uv[:, 0] - obs.K[0, 2]) / obs.K[0, 0],
                          (uv[:, 1] - obs.K[1, 2]) / obs.K[1, 1], np.ones(len(uv))],
                         1) @ obs.T_tc[:3, :3].T
            t = np.stack([lancer(obs.T_tc[:3, 3], r, c.p[None, :3], c.R()[None], c.demi)[0]
                          for c in self.boites]).min(axis=0)
            garde &= ~(t < q[:, 2] - 0.003)
        if garde.sum() < 10:
            return 0.0, 0, False
        meme = np.abs(h_obs - Q[:, 2]) < max(0.003, 2.5 * self.obs.sigma_table)
        continuent = []
        for k in range(4):
            sel = garde & (cote == k)
            continuent.append(sel.sum() >= 3 and meme[sel].mean() > 0.5)
        return (float(meme[garde].mean()), int(sum(continuent)),
                bool(continuent[0] and continuent[1]))

    def soutien(self, i: int) -> float:
        """Ce que perdraient les pieces appuyees sur i si elle disparaissait
        (penalite d'appui sans elle, moins avec elle)."""
        tot = 0.0
        for j in self.autres(i):
            bj = self.boites[j]
            avec = self.physique(j, bj.p[None], bj.dims)[0]
            sans = self.physique(j, bj.p[None], bj.dims,
                                 idx=[k for k in self.autres(j) if k != i])[0]
            tot += max(sans - avec, 0.0)
        return tot


def _lisser(Z: np.ndarray, passes: int = 1) -> np.ndarray:
    """Lissage [1 2 1] / 4 dans chaque direction (B x h x w), `passes` fois,
    bords recopies. Sur la grille au pas de 2 pixels, une passe vaut a peu
    pres une fenetre de 5 pixels, deux passes 7 a 9 pixels (le lissage
    d'une D435 reelle, mesure sur nos captures)."""
    Z = np.asarray(Z, np.float32)
    for _ in range(int(passes)):
        if Z.shape[1] >= 3:
            P = np.concatenate([Z[:, :1], Z, Z[:, -1:]], axis=1)
            Z = 0.25 * P[:, :-2] + 0.5 * P[:, 1:-1] + 0.25 * P[:, 2:]
        if Z.shape[2] >= 3:
            P = np.concatenate([Z[:, :, :1], Z, Z[:, :, -1:]], axis=2)
            Z = 0.25 * P[:, :, :-2] + 0.5 * P[:, :, 1:-1] + 0.25 * P[:, :, 2:]
    return Z


def _pivots(p, dims, pas, libres) -> List[np.ndarray]:
    """Mouvements COUPLES : tourner (tangage, roulis, lacet) autour d'un BOUT
    ou d'un BORD de la piece plutot que de son centre. Une piece qui
    s'incline en gardant un bout sur la table, ou qui pivote autour du bout
    qui est bien place, le fait en un seul pas (en coordonnees separees, il
    faudrait passer par des poses bien pires)."""
    L, l, h = dims
    R = rotations(p[None])[0]
    sortie = []
    for k, bras in ((4, L / 2), (5, l / 2), (3, L / 2)):
        if k not in libres or pas[k] <= 0:
            continue
        axe_bras = R[:, 0] if k in (3, 4) else R[:, 1]
        for cote in (1.0, -1.0):
            # point fixe : bout (ou bord) du dessous, du cote `cote`
            fixe = p[:3] + cote * bras * axe_bras - (R[:, 2] * h / 2 if k != 3 else 0.0)
            for sg in (1.0, -1.0):
                q = p.copy()
                q[k] += sg * pas[k]
                R2 = rotations(q[None])[0]
                axe2 = R2[:, 0] if k in (3, 4) else R2[:, 1]
                fixe2 = q[:3] + cote * bras * axe2 - (R2[:, 2] * h / 2 if k != 3 else 0.0)
                q[:3] += fixe - fixe2
                sortie.append(q)
    return sortie


# ---------------------------------------------------------------------------
# Propositions
# ---------------------------------------------------------------------------

def variantes_dims(types, debout: bool) -> List[Tuple[str, str, np.ndarray]]:
    return [(nom, f, dims_m(types, nom, f)) for nom in types for f in FACES
            if debout or f != "debout"]


def _etranger(pts: np.ndarray, aire_min: float) -> bool:
    """
    Une zone INEXPLIQUEE qui est une surface PLANE (a 2 mm pres) de plus de
    40 mm de large et de `aire_min` mm^2 (6000 : quatre pieces a plat cote
    a cote en font 4600) est un objet etranger (livre, planche, boite) : on
    ne cherche pas a la paver de pieces. (Un groupe de pieces DEJA trouvees
    n'est pas concerne : seule la partie inexpliquee de la mesure compte.)
    """
    if len(pts) < 50:
        return False
    xy = (pts[:, :2] * 1000.0).astype(np.float32)
    (_, _), (wa, wb), _ = cv2.minAreaRect(xy)
    if wa * wb <= aire_min or min(wa, wb) <= 40.0:
        return False
    p = pts[::max(1, len(pts) // 3000)]
    _, sv, _ = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
    return sv[2] / np.sqrt(len(p)) < 0.002


def segments_3d(pts: np.ndarray, rayon: float = 0.006, essais: int = 150,
                 n_max: int = 3, graine: int = 0) -> List[np.ndarray]:
    """
    Decoupe un nuage (N x 3, m) en groupes ALIGNES : RANSAC de droites 3D
    (deux points a 20-75 mm l'un de l'autre ; inliers a moins de `rayon` de
    la droite). Une zone non expliquee contient souvent plusieurs pieces
    qui se touchent (une inclinee sur une couchee) : chacune est un segment.
    Rend les groupes de points, du plus grand au plus petit.
    """
    rng = np.random.default_rng(graine)
    restants = np.arange(len(pts))
    groupes = []
    for _ in range(n_max):
        if len(restants) < 15:
            break
        P = pts[restants]
        meilleur, sel_m = 0, None
        for _ in range(essais):
            i, j = rng.integers(len(P), size=2)
            d = P[j] - P[i]
            n = np.linalg.norm(d[:2])
            if n < 0.02 or n > 0.075:
                continue
            u = d / np.linalg.norm(d)
            q = P - P[i]
            dist = np.linalg.norm(q - np.outer(q @ u, u), axis=1)
            sel = dist < rayon
            c = int(sel.sum())
            if c > meilleur:
                meilleur, sel_m = c, sel
        if sel_m is None or meilleur < 15:
            break
        groupes.append(P[sel_m])
        restants = restants[~sel_m]
    if not groupes:
        groupes = [pts]
    return groupes


def proposer_region(scene: Scene, pts: np.ndarray, types, n_max: int = 3,
                    origine: str = "residu") -> List[Boite]:
    """
    Pieces candidates pour un nuage de points (N x 3, repere table) qu'aucune
    piece n'explique. Le nuage est decoupe en segments alignes
    (segments_3d) ; pour chacun : grand axe par le rectangle minimal, pente
    par la droite du DESSUS des points le long de cet axe (une piece
    inclinee a son dessus en pente) ; positions au milieu et aux deux bouts
    (deux pieces bout a bout, ou une piece en partie cachee), au milieu et
    sur les deux bords (cote a cote). Chaque (type, face) est note par le
    cout de la scene ; rend la meilleure de chaque segment (n_max au plus),
    deja affinee.
    """
    if len(pts) < 15:
        return []
    sortie = []
    for groupe in segments_3d(pts, n_max=n_max):
        cands, meta = _candidats(scene, groupe, types)
        if not cands:
            continue
        scene.nouvelle_libre = True
        try:
            sortie += _proposer(scene, cands, meta, 1, origine)
        finally:
            scene.nouvelle_libre = False
    return sortie


def _candidats(scene: Scene, pts: np.ndarray, types):
    if len(pts) < 10:
        return [], []
    xy = (pts[:, :2] * 1000.0).astype(np.float32)
    (cx, cy), (a, b_), ang = cv2.minAreaRect(xy)
    th = np.radians(ang if a >= b_ else ang + 90.0)
    ex, ey = max(a, b_) / 1000.0, min(a, b_) / 1000.0
    u = np.array([np.cos(th), np.sin(th)])
    nrm = np.array([-u[1], u[0]])
    s = (pts[:, :2] - [cx / 1000.0, cy / 1000.0]) @ u
    # dessus le long de l'axe : 90e centile par tranche de 4 mm, droite
    tr = np.round(s / 0.004).astype(int)
    xs, zs = [], []
    for k in np.unique(tr):
        sel = tr == k
        if sel.sum() >= 3:
            xs.append(s[sel].mean())
            zs.append(np.percentile(pts[sel, 2], 90))
    xs, zs = np.array(xs), np.array(zs)
    pente, z0 = 0.0, float(np.percentile(pts[:, 2], 90))
    if len(xs) >= 4 and np.ptp(xs) > 0.02:
        A = np.stack([xs, np.ones_like(xs)], 1)
        pente, z0 = np.linalg.lstsq(A, zs, rcond=None)[0]
        # robuste : on retire les points a plus de 3 mm et on recommence
        res = np.abs(A @ [pente, z0] - zs)
        garde = res < max(0.003, 2 * np.median(res))
        if garde.sum() >= 3:
            pente, z0 = np.linalg.lstsq(A[garde], zs[garde], rcond=None)[0]
    tangage = float(np.arctan(pente))
    if abs(tangage) < np.radians(4.0):
        tangage, pente = 0.0, 0.0
    cands, meta = [], []
    for nom, face, dims in variantes_dims(types, scene.r.debout):
        L, l, h = dims
        if face == "debout" and z0 < 0.04:
            continue
        Lh = L * np.cos(tangage)
        # un bout de la piece aligne sur un bout de la zone (zone plus longue :
        # deux pieces bout a bout ; plus courte : piece en partie cachee)
        s_pos = {0.0}
        if abs(ex - Lh) > 0.006:
            s_pos |= {(ex - Lh) / 2, -(ex - Lh) / 2}
        w_pos = {0.0}
        if abs(ey - l) > 0.004:
            w_pos |= {(ey - l) / 2, -(ey - l) / 2}
        for sp in s_pos:
            for wp in w_pos:
                c = np.array([cx, cy]) / 1000.0 + sp * u + wp * nrm
                z_top = z0 + pente * sp
                zc = z_top - h / 2 * np.cos(tangage)
                # la table porte ce que la profondeur ne montre pas
                zc = max(zc, h / 2 * np.cos(tangage) + L / 2 * abs(np.sin(tangage)) - 0.001)
                cands.append(np.array([c[0], c[1], zc, th, tangage, 0.0]))
                meta.append((nom, face, dims))
    return cands, meta


def _proposer(scene, cands, meta, n_max, origine):
    # notation groupee par variante (memes dimensions : un seul lot)
    E = np.empty(len(cands))
    cles = [m[:2] for m in meta]
    for cle in set(cles):
        sel = [k for k, c in enumerate(cles) if c == cle]
        E[sel] = scene.energie(None, np.array([cands[k] for k in sel]), meta[sel[0]][2])
    ordre = np.argsort(E)[: max(n_max, 2)]
    sortie = []
    for k in ordre:
        nom, face, dims = meta[k]
        b = Boite(nom, face, np.array(dims), cands[k], origine)
        p, e = scene._motifs(None, b.p, b.dims,
                             (0.003, 0.003, 0.002, np.radians(6), np.radians(4), np.radians(4)),
                             (0.001, 0.001, 0.001, np.radians(2), np.radians(2), np.radians(2)),
                             (0, 1, 2, 3, 4, 5), 25)
        b.p = p
        sortie.append((e, b))
    sortie.sort(key=lambda t: t[0])
    return [b for _, b in sortie[:n_max]]


def residu(scene: Scene) -> np.ndarray:
    """Echantillons (grille de profondeur) ou la mesure est nettement AU-DESSUS
    de la scene rendue : quelque chose que la scene n'explique pas."""
    obs = scene.obs
    zp = scene.profondeur_rendue()
    ecart = zp - obs.z_obs
    h = obs.points[..., 2]
    return (obs.connu & ~obs.etranger & (obs.poids > 0.5)
            & (ecart > np.maximum(3 * obs.sigma, 0.004))
            & (h > obs.h_min))


# ---------------------------------------------------------------------------
# L'affinage
# ---------------------------------------------------------------------------

def _doublon(a: Boite, b: Boite) -> bool:
    """Deux propositions pour la meme piece : centres proches (moins d'un
    tiers de longueur), meme direction (25 deg), et elles se penetrent
    largement. Une piece proposee a plat EN TRAVERS de sa porteuse (une
    inclinee mal lue) n'est pas un doublon : c'est l'affinage qui tranche."""
    if np.linalg.norm(a.p[:3] - b.p[:3]) < 0.008:
        return True
    if np.linalg.norm(a.p[:3] - b.p[:3]) > max(a.dims[0], b.dims[0]) / 3:
        return False
    da = abs((a.p[3] - b.p[3] + np.pi / 2) % np.pi - np.pi / 2)
    if da > np.radians(25):
        return False
    pen = penetration(a.p[None, :3], a.R()[None], a.demi, b.p[:3], b.R(), b.demi)[0]
    return pen > 0.5 * min(min(a.dims), min(b.dims))


class Affinage:
    """
    L'affinage complet (voir l'en-tete). Usage :

        aff = Affinage(obs, propositions)       # Boites
        boites = aff.executer()
    """

    def __init__(self, obs: Observation, propositions: Sequence[Boite],
                 reglages: Optional[Reglages] = None, trace: bool = False):
        self.obs = obs
        self.scene = Scene(obs, reglages)
        self.r = self.scene.r
        self.propositions = list(propositions)
        self.journal: List[str] = []
        self.refus: List[Boite] = []
        self.trace = trace
        self.durees: Dict[str, float] = {}

    def _log(self, m):
        if self.trace:
            print("  [raffinement]", m)
        self.journal.append(m)

    # -- etapes ---------------------------------------------------------------
    def initialiser(self):
        sc = self.scene
        for b in self.propositions:
            if any(_doublon(b, c) for c in sc.boites):
                continue
            sc.ajouter(b.copie())

    def associer_masques(self):
        """Chaque piece prend le masque YOLO qui recouvre le mieux sa partie
        visible rendue (IoU >= 0.25), un masque au plus par piece."""
        sc, obs = self.scene, self.obs
        if not obs.avec_masques:
            return
        n = len(sc.boites)
        for b in sc.boites:
            b.masque, b.iou_masque = -1, 0.0
        if not n:
            return
        devant = sc.Tc.argmin(axis=0)
        fini = np.isfinite(sc.Tc.min(axis=0))
        paires = []
        for i in range(n):
            V = (devant == i) & fini
            if not V.any():
                continue
            inter = (obs.M & V[None]).sum(axis=(1, 2))
            union = (obs.M | V[None]).sum(axis=(1, 2))
            iou = inter / np.maximum(union, 1)
            for k in np.nonzero(iou >= 0.25)[0]:
                paires.append((iou[k], i, k))
        pris_i, pris_k = set(), set()
        for iou, i, k in sorted(paires, reverse=True):
            if i in pris_i or k in pris_k:
                continue
            pris_i.add(i)
            pris_k.add(k)
            sc.boites[i].masque, sc.boites[i].iou_masque = int(k), float(iou)

    def balayer(self, fin=False):
        sc = self.scene
        # du plus haut au plus bas : les pieces du dessus d'abord
        ordre = sorted(range(len(sc.boites)), key=lambda i: -sc.boites[i].p[2])
        for i in ordre:
            if fin:
                sc.optimiser(i, pas=(0.0015, 0.0015, 0.001, np.radians(3), np.radians(2), np.radians(2)),
                             fin=(0.0002, 0.0002, 0.0002, np.radians(0.4), np.radians(0.4),
                                  np.radians(0.4)))
            else:
                sc.optimiser(i, sauts=True, fin=(0.0008, 0.0008, 0.0006, np.radians(1.5),
                                                 np.radians(1), np.radians(1)))

    def variantes(self):
        """Choix discret du type et de la face : chaque variante est posee au
        meme endroit (meme dessous, ou meme dessus), affinee, et comparee."""
        sc = self.scene
        for i in range(len(sc.boites)):
            b = sc.boites[i]
            e0 = float(sc.energie(i, b.p[None], b.dims)[0])
            essais = []
            for nom, face, dims in variantes_dims(self.obs.types, self.r.debout):
                if nom == b.type and face == b.face:
                    continue
                if face == "debout" and b.face != "debout" and b.p[2] + b.dims[2] / 2 < 0.04:
                    continue
                if b.face == "debout" and face != "debout" and b.p[2] + b.dims[2] / 2 > 0.05:
                    continue
                Q = []
                for sgn in (1, -1):              # meme dessous / meme dessus
                    q = b.p.copy()
                    q[2] += sgn * (dims[2] - b.dims[2]) / 2
                    Q.append(q)
                E = sc.energie(i, np.array(Q), dims, type_=nom)
                essais.append((float(E.min()), nom, face, dims, Q[int(np.argmin(E))]))
            essais.sort(key=lambda t: t[0])
            meilleurs = []
            for e, nom, face, dims, q in essais[:3]:
                p, e2 = sc._motifs(i, q, dims, (0.002, 0.002, 0.002, np.radians(4), np.radians(3),
                                                 np.radians(3)),
                                   (0.0005, 0.0005, 0.0005, np.radians(1), np.radians(1),
                                    np.radians(1)), (0, 1, 2, 3, 4, 5), 30, type_=nom)
                meilleurs.append((e2, nom, face, dims, p))
            meilleurs.sort(key=lambda t: t[0])
            if meilleurs and meilleurs[0][0] < e0 - max(5.0, 0.02 * abs(e0)):
                e2, nom, face, dims, p = meilleurs[0]
                self._log("piece {} : {} {} -> {} {} ({:.0f} -> {:.0f})".format(
                    i, b.type, b.face, nom, face, e0, e2))
                b.type, b.face, b.dims, b.p = nom, face, np.array(dims), p
                sc.maj(i)
                b.marge = e0 - e2 if len(meilleurs) < 2 else meilleurs[1][0] - e2
            else:
                b.marge = (meilleurs[0][0] - e0) if meilleurs else 100.0

    def garder(self, i: int) -> Tuple[bool, Dict[str, float]]:
        """La piece i est-elle justifiee (voir l'en-tete, parcimonie) ?"""
        sc, r = self.scene, self.r
        b = sc.boites[i]
        d = sc.bilan(i)
        soutien = sc.soutien(i)
        gain = -d["profondeur"] + soutien
        vis = d["visibles"]
        b.gain, b.visibles = gain, vis
        b.contredits = d["contredits"] / max(vis, 1.0)
        b.physique = d["physique"]
        if b.contredits > r.contredit_max and vis >= 25:
            return False, d
        if r.debord:
            b.debord, n_cotes, deux_bouts = sc.debord(i)
            if n_cotes >= 3 or deux_bouts:
                return False, d
        if vis >= 25 or gain >= r.gain_min:
            # la profondeur doit soutenir la piece ; un masque du reseau qui
            # la recouvre bien abaisse l'exigence (la profondeur, lissee ou
            # trouee, n'explique alors qu'une partie de ses pixels), sans
            # jamais suffire seul
            part = r.part_min if b.iou_masque < 0.6 else r.part_min_masque
            if b.bord:
                part = max(part, r.part_min_bord)
            ok = gain >= r.gain_min and gain >= part * vis
            return ok, d
        # presque rien de mesure sous la piece (bois sature, ou cachee) : seul
        # un masque net, ou l'appui d'une autre piece, la justifie
        return (b.iou_masque >= 0.5 or soutien > 20.0), d

    def elaguer(self):
        sc = self.scene
        while sc.boites:
            pires = []
            for i in range(len(sc.boites)):
                ok, _ = self.garder(i)
                if not ok:
                    pires.append((sc.boites[i].gain, i))
            if not pires:
                break
            _, i = min(pires)
            self._log("retrait de la piece {} ({} {}, gain {:.0f}, visibles {:.0f}, contredits "
                      "{:.2f}, debord {:.2f})".format(
                          i, sc.boites[i].type, sc.boites[i].face, sc.boites[i].gain,
                          sc.boites[i].visibles, sc.boites[i].contredits, sc.boites[i].debord))
            sc.retirer(i)

    def etrangers(self) -> None:
        """Les objets etrangers candidats (Observation) que la scene actuelle
        n'explique pas a moitie deviennent des zones interdites aux
        propositions."""
        obs = self.obs
        obs.etranger = np.zeros(obs.etranger_lab.shape, bool)
        if not obs.etranger_lab.any():
            return
        zp = self.scene.profondeur_rendue()
        explique = obs.connu & (np.abs(obs.z_obs - zp) < np.maximum(3 * obs.sigma, 0.003)) & \
            (zp < obs.t_table - 0.002)
        for k in range(1, int(obs.etranger_lab.max()) + 1):
            m = (obs.etranger_lab == k) & obs.connu
            if m.any() and explique[m].mean() < 0.5:
                obs.etranger |= obs.etranger_lab == k
                self._log("objet etranger : {} echantillons".format(int(m.sum())))

    def _regions(self, echecs) -> List[Tuple[float, np.ndarray]]:
        """Zones a expliquer : composantes du residu (voir residu), et masques
        YOLO sans piece (leurs points de profondeur au-dessus de la table) ;
        (taille, points), les plus grandes d'abord, sauf celles deja essayees
        en vain (`echecs` : centres)."""
        sc, obs = self.scene, self.obs
        regions = []
        res = residu(sc).astype(np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        nb, lab, st, _ = cv2.connectedComponentsWithStats(res)
        aire_min = max(40.0 / obs.echelle, 8)
        for k in range(1, nb):
            if st[k, cv2.CC_STAT_AREA] < aire_min:
                continue
            if _etranger(obs.points[lab == k], obs.aire_etranger):
                continue
            # surtout le long du BORD de la table (nappe, chaise...) : une
            # piece n'y sera gardee que si elle explique tres bien la mesure
            bord = obs.interieur[lab == k].mean() < 0.5
            regions.append((float(st[k, cv2.CC_STAT_AREA]), obs.points[lab == k], bord))
        if obs.avec_masques:
            pris = {b.masque for b in sc.boites}
            sel = obs.connu & (obs.points[..., 2] > obs.h_min)
            q = obs.points[sel]
            if len(q):
                uv = np.round(obs.projeter_couleur(q)).astype(int)
                Wc, Hc = obs.vue.taille
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < Wc) & (uv[:, 1] >= 0) & (uv[:, 1] < Hc)
                for k in range(len(obs.M)):
                    if k in pris:
                        continue
                    dans = np.zeros(len(q), bool)
                    dans[ok] = obs.masques_img[k][uv[ok, 1], uv[ok, 0]]
                    if dans.sum() >= 15:
                        regions.append((float(dans.sum()), q[dans], False))
        regions = [r for r in regions
                   if not any(np.linalg.norm(r[1].mean(axis=0) - e) < 0.006 for e in echecs)]
        regions.sort(key=lambda r: -r[0])
        return regions

    def _essayer(self, b: Boite, fin_, e_max: Optional[float] = None) -> bool:
        """Ajoute la piece b, re-ajuste ses voisines ; la garde si elle (et
        ses voisines) restent justifiees - une voisine qui n'explique plus
        rien est retiree si la scene y gagne -, et, avec e_max, si le cout
        de profondeur de la scene finit sous e_max. Sinon tout revient en
        place."""
        sc = self.scene
        etat = sc.sauver()
        e_avant = sc.energie_profondeur(self.r.flou)
        i = sc.ajouter(b)
        # les voisines qu'elle chevauche se re-ajustent (elles etaient
        # peut-etre a cheval sur elle et sur une autre)
        voisines = [j for j in range(len(sc.boites) - 1)
                    if penetration(b.p[None, :3], b.R()[None], b.demi, sc.boites[j].p[:3],
                                   sc.boites[j].R(), sc.boites[j].demi)[0] > 0.0005]
        for j in voisines:
            sc.optimiser(j, sauts=True, fin=fin_)
        sc.optimiser(i, fin=fin_)
        for j in voisines:
            sc.optimiser(j, fin=fin_)
        self.associer_masques()
        ok, _ = self.garder(i)
        retirees = 0
        if ok:
            # une voisine qui n'explique plus rien etait fausse (a cheval,
            # doublon) : elle part, si la scene y gagne au total
            mortes = [j for j in voisines if not self.garder(j)[0]]
            for j in sorted(mortes, reverse=True):
                sc.retirer(j)
            retirees = len(mortes)
            e_apres = sc.energie_profondeur(self.r.flou)
            if mortes and e_apres >= e_avant - self.r.gain_min:
                ok = False
            if e_max is not None and e_apres >= e_max:
                ok = False
        if ok:
            self._log("ajout {} {} ({}, gain {:.0f}){}".format(
                b.type, b.face, b.origine, b.gain,
                ", {} voisine(s) retiree(s)".format(retirees) if retirees else ""))
            self.associer_masques()
            return True
        self._log("refus {} {} ({}, gain {:.0f}, visibles {:.0f}, contredits {:.2f})".format(
            b.type, b.face, b.origine, b.gain, b.visibles, b.contredits))
        self.refus.append(b.copie())
        sc.restaurer(etat)
        self.associer_masques()
        return False

    def reinitialiser(self, seuil: float = 0.12) -> int:
        """
        Une piece a travers laquelle on VOIT (une part notable de ses pixels
        mesures bien plus loin qu'elle) est mal placee, souvent coincee dans
        un optimum local (deux pieces bout a bout decalees ensemble, une
        piece a cheval). On l'enleve, on repropose une piece sur la zone
        qu'elle laisse inexpliquee (proposer_region), voisines re-ajustees ;
        le remplacement est garde si la scene explique mieux la mesure.
        """
        sc = self.scene
        changees = 0
        fin_ = (0.001, 0.001, 0.001, np.radians(2), np.radians(2), np.radians(2))
        faites = []
        for _ in range(len(sc.boites) + 2):
            cands = []
            for i in range(len(sc.boites)):
                self.garder(i)
                b = sc.boites[i]
                if b.contredits >= seuil and b.visibles >= 25 and \
                        not any(np.linalg.norm(b.p[:3] - f) < 0.003 for f in faites):
                    cands.append((b.contredits, i))
            if not cands:
                break
            _, i = max(cands)
            b = sc.boites[i]
            faites.append(b.p[:3].copy())
            etat = sc.sauver()
            e_avant = sc.energie_profondeur(self.r.flou)
            sc.retirer(i)
            res = residu(sc)
            # la zone que la piece laisse : composantes du residu sous elle
            Q = coins(b.p[None, :3], b.R()[None], b.demi)[0]
            r0, r1, c0, c1 = self.obs.g.fenetre(self.obs.projeter_profondeur(Q), marge=1)
            fen = np.zeros_like(res)
            fen[r0:r1, c0:c1] = True
            nb, lab, st, _ = cv2.connectedComponentsWithStats(res.astype(np.uint8))
            ks = [k for k in range(1, nb) if (fen & (lab == k)).any()]
            remplacee = False
            if ks:
                k = max(ks, key=lambda k: st[k, cv2.CC_STAT_AREA])
                for nv in proposer_region(sc, self.obs.points[lab == k], self.obs.types, n_max=3):
                    if self._essayer(nv, fin_, e_max=e_avant - 10.0):
                        remplacee = True
                        break
            if remplacee:
                self._log("piece {} {} replacee (contredite a {:.0f} %)".format(
                    b.type, b.face, 100 * b.contredits))
                changees += 1
            else:
                sc.restaurer(etat)
        self.associer_masques()
        return changees

    def reproposer(self, max_ajouts: int = 8) -> int:
        """
        Nouvelles pieces la ou la mesure n'est pas expliquee, et la ou un
        masque YOLO n'a pas de piece. GLOUTON : une zone a la fois (la plus
        grande), la meilleure proposition (proposer_region), ses voisines
        re-ajustees ; gardee si elle explique assez (garder), sinon tout est
        remis en place. Le residu est recalcule apres chaque ajout : deux
        propositions ne se disputent jamais les memes pixels.
        """
        sc = self.scene
        ajout = 0
        echecs: List[np.ndarray] = []
        fin_ = (0.001, 0.001, 0.001, np.radians(2), np.radians(2), np.radians(2))
        for _ in range(max_ajouts + 6):
            if ajout >= max_ajouts:
                break
            regions = self._regions(echecs)
            if not regions:
                break
            _, pts, bord = regions[0]
            props = proposer_region(sc, pts, self.obs.types, n_max=3)
            for b in props:
                b.bord = bord
            essais = []
            for b in props:
                if any(np.linalg.norm(b.p[:3] - c.p[:3]) < 0.008 for c in sc.boites):
                    continue
                essais.append((float(sc.energie(None, b.p[None], b.dims)[0]), b))
            essais.sort(key=lambda t: t[0])
            reussi = False
            for _, b in essais:
                if self._essayer(b, fin_):
                    ajout += 1
                    reussi = True
                    break
            if not reussi:
                echecs.append(pts.mean(axis=0))
        return ajout

    def poser(self):
        """Une piece presque horizontale est posee exactement a plat sur ce
        qui la porte (si cela ne coute presque rien)."""
        sc = self.scene
        for i, b in enumerate(sc.boites):
            if abs(b.p[4]) > np.radians(2.5) or abs(b.p[5]) > np.radians(2.5):
                continue
            q = b.p.copy()
            q[4] = q[5] = 0.0
            e0 = float(sc.energie(i, b.p[None], b.dims)[0])
            # hauteur de ce qui la porte : le plus haut des appuis sous elle
            C = q[None, :3].copy()
            C[0, 2] += 0.004
            R = rotations(q[None])
            _, app = sc._appui(C, R, b.demi, sc.autres(i))
            sol = 0.0
            for j in app[0][0]:
                if j >= 0:
                    bj = sc.boites[j]
                    cj = coins(bj.p[None, :3], bj.R()[None], bj.demi)[0]
                    sol = max(sol, cj[:, 2].max())
            q[2] = sol + b.dims[2] / 2
            if abs(q[2] - b.p[2]) > 0.004:
                continue
            e1 = float(sc.energie(i, q[None], b.dims)[0])
            if e1 <= e0 + 10.0:
                b.p = q
                sc.maj(i)

    def confiances(self):
        sc = self.scene
        for i, b in enumerate(sc.boites):
            self.garder(i)
            expl = np.clip(b.gain / max(0.85 * b.visibles, 1.0), 0.0, 1.0) if b.visibles >= 25 else 0.5
            marge = 1.0 - np.exp(-max(b.marge, 0.0) / max(20.0, 0.1 * b.visibles))
            phys = np.exp(-b.physique / 60.0)
            b.confiance = float(np.clip(expl * (0.6 + 0.4 * marge) * phys
                                        * (1.0 - 0.5 * b.contredits), 0.0, 1.0))

    def executer(self) -> List[Boite]:
        t0 = time.perf_counter()
        if self.obs.vide:
            return []
        self.initialiser()
        self.associer_masques()
        self.durees["init"] = time.perf_counter() - t0
        ev = self.scene.evaluations
        # CONTINUATION : la penetration entre pieces ne compte d'abord que
        # peu. Deux propositions qui se croisent (une piece inclinee lue a
        # plat, en travers de sa porteuse) se separent alors comme la MESURE
        # le dit (la piece se leve sur sa porteuse), au lieu de fuir de cote
        # pour echapper a une penalite enorme ; puis la penalite reprend
        # tout son poids.
        k_pen = self.r.k_penetration
        self.r.k_penetration = 0.02 * k_pen
        self.balayer()
        self.associer_masques()
        if self.r.flou_auto and self.scene.boites:
            # LISSAGE DE LA CAMERA, estime sur la capture : la scene (deja
            # a peu pres en place) est rendue avec 0 a 3 passes de lissage,
            # et celle qui explique le mieux la mesure l'emporte (une camera
            # simulee nette, une D435 qui lisse beaucoup les bords...)
            couts = [self.scene.energie_profondeur(k) for k in range(4)]
            meilleur = int(np.argmin(couts))
            self._log("lissage : couts {} -> {} passe(s)".format(
                " ".join("{:.0f}".format(c) for c in couts), meilleur))
            if meilleur != self.r.flou:
                self.r.flou = meilleur
                self.balayer()
        for f in (0.25, 1.0)[:max(self.r.balayages, 1)]:
            self.r.k_penetration = f * k_pen
            self.balayer()
        self.r.k_penetration = k_pen
        self.associer_masques()
        if self.r.variantes:
            self.variantes()
        self.elaguer()
        self.etrangers()
        self.durees["balayages"] = time.perf_counter() - t0
        self.durees["ev_balayages"] = self.scene.evaluations - ev
        ev = self.scene.evaluations
        for _ in range(self.r.repropositions):
            if not (self.reinitialiser() + self.reproposer()):
                break
            self.balayer()
            self.associer_masques()
            if self.r.variantes:
                self.variantes()
            self.elaguer()
        self.durees["repropositions"] = time.perf_counter() - t0
        self.durees["ev_repropositions"] = self.scene.evaluations - ev
        self.balayer(fin=True)
        self.poser()
        self.elaguer()
        self.confiances()
        # une piece a peine justifiee (confiance tres basse : physique
        # impossible, a peine expliquee) n'est pas rendue
        for i in reversed(range(len(self.scene.boites))):
            if self.scene.boites[i].confiance < self.r.confiance_min:
                self._log("confiance trop basse : piece {} retiree ({:.2f})".format(
                    i, self.scene.boites[i].confiance))
                self.scene.retirer(i)
        self.durees["total"] = time.perf_counter() - t0
        return self.scene.boites


# ---------------------------------------------------------------------------
# Sortie
# ---------------------------------------------------------------------------

def repere_dessus(b: Boite) -> np.ndarray:
    """Repere de la face superieure (repere table) : x le long du grand
    cote, z vers le haut, lacet ramene dans ]-90, 90] deg."""
    R = b.R()
    if R[2, 2] < 0:
        R = R @ np.diag([1.0, -1.0, -1.0])
    if R[0, 0] < 0 or (abs(R[0, 0]) < 1e-9 and R[1, 0] < 0):
        R = R @ np.diag([-1.0, -1.0, 1.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = b.p[:3] + R[:, 2] * b.dims[2] / 2
    return T


def masque_visible(boites: Sequence[Boite], obs_or_T_ct, vue: pc.VueCouleur,
                   pas: int = 1) -> np.ndarray:
    """Etiquettes (image de la vue, -1 : aucune piece) de la piece VISIBLE en
    chaque pixel : rendu z-buffer de la scene a pleine resolution."""
    T_ct = obs_or_T_ct.T_ct if hasattr(obs_or_T_ct, "T_ct") else obs_or_T_ct
    Wc, Hc = vue.taille
    lab = np.full((Hc, Wc), -1, np.int32)
    if not boites:
        return lab
    T_tv = np.linalg.inv(vue.T_cp @ T_ct)
    T_vt = vue.T_cp @ T_ct
    # fenetre : projection des coins de toutes les boites
    Q = np.concatenate([coins(b.p[None, :3], b.R()[None], b.demi)[0] for b in boites])
    q = Q @ T_vt[:3, :3].T + T_vt[:3, 3]
    uv = pc.projeter(q, vue)
    x0, y0 = max(int(uv[:, 0].min()) - 2, 0), max(int(uv[:, 1].min()) - 2, 0)
    x1, y1 = min(int(uv[:, 0].max()) + 3, Wc), min(int(uv[:, 1].max()) + 3, Hc)
    if x1 <= x0 or y1 <= y0:
        return lab
    g = Grille(x0, y0, x1, y1, pas, T_tv[:3, 3],
               lambda uv_: pc.rayons(uv_, vue) @ T_tv[:3, :3].T)
    d = g.d.reshape(-1, 3)
    t = np.stack([lancer(g.o, d, b.p[None, :3], b.R()[None], b.demi)[0] for b in boites])
    k = t.argmin(axis=0)
    k[~np.isfinite(t.min(axis=0))] = -1
    sous = k.reshape(g.ny, g.nx)
    if pas == 1:
        lab[y0:y0 + g.ny, x0:x0 + g.nx] = sous
    else:
        lab[y0:y1, x0:x1] = cv2.resize(sous.astype(np.float32), (x1 - x0, y1 - y0),
                                       interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return lab


def vers_pieces(boites: Sequence[Boite], obs: Observation, vue: Optional[pc.VueCouleur],
                T_robot_camera: Optional[np.ndarray] = None) -> List["pc.PieceVue"]:
    """Boites affinees -> PieceVue (perception.py), avec support, appuis,
    dessus, inclinaison, confiance."""
    sc = Scene(obs, Reglages())
    for b in boites:
        sc.boites.append(b)
    lab = masque_visible(boites, obs, vue) if vue is not None else None
    pieces = []
    appuis = [sc.appuis(i) for i in range(len(boites))]
    for i, b in enumerate(boites):
        T_t = repere_dessus(b)
        T_c = obs.T_ct @ T_t
        dims_mm = tuple(float(x) * 1000.0 for x in b.dims)
        Q = coins(b.p[None, :3], b.R()[None], b.demi)[0]
        if vue is not None:
            T = vue.T_cp @ obs.T_ct
            uv = pc.projeter(Q @ T[:3, :3].T + T[:3, 3], vue)
        else:
            uv = obs.projeter_profondeur(Q)
        contour = cv2.convexHull(np.round(uv).astype(np.int32)).reshape(-1, 2)
        masque = (lab == i) if lab is not None else None
        bas_, lat = appuis[i]
        incl = float(np.degrees(np.arcsin(np.clip(abs(T_t[2, 0]), 0, 1))))
        # une piece inclinee tient aussi par un bout contre le flanc d'une
        # autre ; une piece a plat qui en touche une autre de cote, non
        sup_boites = [j for j in bas_ if j >= 0] + ([j for j in lat if j >= 0] if incl > 3.0 else [])
        support = None
        if sup_boites:
            # la porteuse principale : la plus haute
            support = max(sup_boites, key=lambda j: boites[j].p[2]) + 1
        base = float(max(Q[:, 2].min(), 0.0) * 1000.0)
        if incl < 2.5 and abs(np.degrees(b.p[5])) < 2.5:
            base = round(base, 1)
        pieces.append(pc.PieceVue(
            i + 1, b.face, dims_mm, 0.0, T_c, T_t, contour,
            int(masque.sum()) if masque is not None else 0, b.type,
            support=support, hauteur_base=base if (support is not None or base > 3.0) else 0.0,
            confiance=b.confiance, masque=masque,
            T_robot=None if T_robot_camera is None else T_robot_camera @ T_c,
            iou=b.iou_masque if b.masque >= 0 else float("nan"),
            origine="raffinement:" + b.origine, inclinaison=incl,
            appuis=[j + 1 for j in sup_boites]))
    for i, p in enumerate(pieces):
        p.dessus = [q.numero for q in pieces if (i + 1) in q.appuis]
    return pieces


def raffiner(capture, propositions: Sequence, types, masques=None,
             vue: Optional[pc.VueCouleur] = None, reglages: Optional[Reglages] = None,
             T_robot_camera=None, table: Optional[pc.Table] = None,
             trace: bool = False, rendre_affinage: bool = False):
    """
    Affine une detection. `capture` : chemin, dictionnaire ou vision.Scene ;
    `propositions` : pieces detectees (vision.Piece, T_camera dans le repere
    de la profondeur) ; `masques` : [(masque booleen de la vue, classe,
    confiance)] du reseau, ou None ; `vue` : la vue couleur des masques (et
    des contours rendus). Rend une liste de PieceVue (et l'Affinage si
    rendre_affinage).
    """
    capture = pc.charger_capture(capture)
    types = {k: tuple(float(x) for x in v) for k, v in types.items()}
    if vue is None:
        vue = pc.vue_couleur(capture, infrarouge=True)
    r_ = reglages or Reglages()
    obs = Observation(capture, types, masques, vue, table=table, aretes=r_.aretes > 0,
                      aretes_densite_max=r_.aretes_densite_max)
    boites = []
    for p in propositions:
        b = boite_de_piece(p, obs.T_tc, types)
        if b is not None:
            boites.append(b)
    aff = Affinage(obs, boites, reglages, trace)
    res = aff.executer()
    pieces = vers_pieces(res, obs, vue, T_robot_camera)
    return (pieces, aff) if rendre_affinage else pieces
