# -*- coding: utf-8 -*-
"""
Detection des pieces PAR MODELE, dans une vue de dessus redressee.

vision.detecter cherche des TACHES (ce qui depasse de la table) puis mesure
chaque tache. Deux pieces qui se touchent font une seule tache, et la tache
d'une piece etroite est rognee par le lissage de la camera. Ici on part de
l'autre bout : on SAIT quelles pieces peuvent etre posees (types x faces
d'appui, donc des rectangles de taille connue a une hauteur connue), et on
cherche ou les placer pour EXPLIQUER au mieux les mesures.

  1. TABLE : plan par RANSAC puis correction de courbure (fonctions de
     vision.py). Les pixels "volants" (a mi-hauteur d'un saut, flottant dans
     un vide entre une piece et la table qu'elle cache) sont MARQUES : ils
     servent aux contours mais pas au remplissage (retirer_volants).
  2. VUE REDRESSEE (orthophoto) : chaque point mesure est porte dans le
     repere de la table et range dans une grille vue de dessus a 1 mm par
     case. Un rectangle y a sa vraie taille quel que soit l'angle de vue.
     Les pixels sont sur-echantillonnes entre voisins d'une meme surface,
     pour que les cases loin de la camera ne restent pas vides ; ce qui
     reste vide est INCONNU (ombre d'une piece, reflet) : ni table ni piece.
  3. INDICES, fusionnes en log-rapport de vraisemblance : hauteur au-dessus
     d'un seuil adapte au bruit de la table ; couleur eloignee du modele de
     couleur de la table (melange de gaussiennes en Lab appris sur la table)
     et brillance infrarouge, possibles mais desactivees par defaut (aucun
     gain mesure, voir indices). Contours : sauts de hauteur (Canny sur le gradient de hauteur) et
     contours de l'infrarouge et de la couleur REDRESSES AU PLAN DE CHAQUE
     FACE possible (balayage de plans, textures_redressees) ; segments
     droits (LSD) et orientation locale (tenseur de structure).
  4. RECTANGLES DE TAILLE CONNUE : des hypotheses (centre, angle, modele)
     sont NOTEES (Noteur) par l'appui des aretes (chanfrein oriente),
     le remplissage, l'accord et la planeite de la hauteur de la face
     superieure, le debordement de la meme surface, le chevauchement des
     pieces deja retenues. Selection GLOUTONNE avec EXPLICATION : on retient
     la meilleure, on marque ses cases comme expliquees, on renote et
     re-affine les autres, tant que le score depasse un seuil ; deux pieces
     qui se touchent donnent ainsi deux rectangles. Chaque pose est AFFINEE
     par recherche locale sur le score (lu en bilineaire, donc continu), a
     une fraction de millimetre. Enfin un test de PARCIMONIE (fusionner)
     remplace deux pieces etroites accolees par une large si elle explique
     aussi bien.
  5. SORTIE : la meme que vision.detecter (Piece avec T_camera, T_table,
     type, face, dimensions mesurees, ecart), plus un score de confiance
     entre 0 et 1.
  6. PIECES EMPILEES. Une piece peut reposer sur une autre : sa face
     superieure est alors a (base + epaisseur), ou la base est le sommet
     de sa porteuse. Les bases possibles sont DISCRETES (bases_de_pile :
     0, puis 8, 17, 19, puis 16, 25, 27, 34, 36, 38 mm pour les types par
     defaut) : chaque modele couche existe aussi "pose sur" chacune, et
     c'est l'EPAISSEUR (sommet - base) qui dit le type et la face. Le
     noteur tient compte de la hauteur : une piece retenue n'est en
     CONFLIT qu'avec une hypothese de la meme tranche de hauteur ; posee
     au-dessus, elle CACHE la partie qu'elle recouvre (ni pour ni contre :
     comme l'inconnu) ; au-dessous, a la hauteur de la base, c'est la
     PORTEUSE. La selection gloutonne prend donc d'ordinaire la piece du
     dessus (entiere) puis celle du dessous (cachee en partie, renotee),
     ou l'inverse, la porteuse servant alors de preuve d'appui. Une piece
     empilee doit montrer cette preuve (porteuse retenue sous elle, ou
     porteuse qui depasse autour d'elle, a la hauteur de la base, hors des
     rampes du lissage) et paie un petit prix par piece cachee dessous
     (rasoir d'Occam). Pour la piece du dessus, les cases de RAMPE (grand
     saut de hauteur dans 5 x 5 mm : le lissage y met des points a
     mi-hauteur, jusqu'a l'interieur de la face d'une piece haute) ne
     comptent pas. Apres la selection, PAS DE FANTOME (coherence_piles) :
     une piece empilee sans porteuse retenue, ou dont la porteuse ne
     depasse pas vraiment (aire_degagee), est re-examinee : on cherche sa
     porteuse sous elle (porteuse_visible : en grande partie cachee, elle
     etait sous le seuil), sinon on la repose sur la table. Seule une face
     a une hauteur qu'aucune piece seule n'atteint (34 mm) garde sa pile
     sans porteuse vue : la porteuse, entierement cachee, est alors
     DEDUITE sous elle, avec une confiance faible (cachee = 1). Chaque
     piece porte `support` (numero de la porteuse, None : la table),
     `hauteur_base`, `dessus` et `cachee` ; ordre_de_prise (vision.py) rend
     les pieces libres d'abord, du haut vers le bas.

Mesures (evaluer_detection.py, 10 scenes synthetiques de 8 pieces par
configuration, dont 3 en contact) : en vue de dessus, rappel 0.97-0.99,
aucun faux positif, erreur de position mediane 0.4-0.6 mm, contre un rappel
de 0.35-0.44 pour vision.detecter qui ne separe pas les pieces en contact ;
en vue rasante, rappel 0.80 (contre 0.07). Sur les captures reelles
annotees a l'oeil : 38 pieces justes sur 44 et 1 fausse, contre 32 et 0.
Plus lent : 2.5 a 5 s par image contre 0.3 s (4 a 6 s avec la recherche
d'empilements).

Empilements (evaluer_detection.py --empilement : 10 scenes de 8 pieces
dont 3 posees sur une autre, par configuration). Vue de dessus : rappel
0.91 (0.51 sans l'etape 6), pieces empilees trouvees 0.90 (0.20), type et
face justes 0.96 pour elles, support juste 0.90, ordre de prise sans
faute dans 7 scenes sur 10 (3) ; vue reculee : 0.85 (0.45), 0.90 (0.10),
ordre 8 sur 10 (0). Les scenes sans pile ne changent pas. Vue rasante :
rappel 0.53 (0.40), piles 0.37 (0.17) : la piece du dessus y cache sa
porteuse et projette une grande ombre. Non modelise : une piece INCLINEE
(un bout sur la table, l'autre sur une piece), cas frequent dans un tas
reel.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:                        # pragma: no cover
    raise ImportError("OpenCV manque : .venv/bin/pip install opencv-python") from exc

import vision
from vision import (PROFONDEUR_MAX, Resultat, Scene, corriger_courbure,  # noqa: F401
                    nuage, ordre_de_prise, plan_table, repere_table)

#: nombre de classes d'orientation des contours (sur 180 deg)
N_ORIENT = 8


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

@dataclass
class Modele:
    """Un type de piece pose sur une face : rectangle L x l, epaisseur h (mm)
    sous la face superieure, posee a la hauteur `base` (0 : sur la table ;
    sinon sur la face superieure d'une autre piece). Sa face superieure est
    a `haut` = base + h."""
    type: str
    face: str
    L: float
    l: float
    h: float
    base: float = 0.0

    @property
    def haut(self) -> float:
        return self.base + self.h

    def sur(self, base: float) -> "Modele":
        """Le meme, pose a la hauteur `base`."""
        return Modele(self.type, self.face, self.L, self.l, self.h, float(base))


def modeles(types: Dict[str, Sequence[float]], debout: bool = True) -> List[Modele]:
    """Toutes les facons de poser chaque type : a plat, sur la tranche,
    debout (la hauteur est la petite, la moyenne, la grande dimension)."""
    sortie = []
    for nom, dims in types.items():
        a, b, c = sorted(float(x) for x in dims)
        sortie.append(Modele(nom, "a plat", c, b, a))
        sortie.append(Modele(nom, "sur la tranche", c, a, b))
        if debout:
            sortie.append(Modele(nom, "debout", b, a, c))
    return sortie


def bases_de_pile(mods: Sequence[Modele], etages: int = 2) -> List[float]:
    """
    Hauteurs ou une piece peut etre POSEE : la table (0), et le sommet d'une
    pile de 1 a `etages` pieces couchees (a plat ou sur la tranche ; une
    piece debout ne porte rien). Ces hauteurs sont discretes : 8, 17, 19,
    puis 16, 25, 27, 34, 36, 38 mm pour les types par defaut. C'est ce qui
    rend l'empilement mesurable : la hauteur de la face superieure d'une
    piece posee sur une autre vaut l'une de ces bases PLUS l'epaisseur d'une
    piece, et la base se verifie (la porteuse depasse, ou elle est vue).
    """
    hs = sorted({round(m.h, 1) for m in mods if m.face != "debout" and m.base == 0})
    niveaux = {0.0}
    courant = {0.0}
    for _ in range(etages):
        courant = {round(b + h, 1) for b in courant for h in hs}
        niveaux |= courant
    return sorted(niveaux)


def tolerance_hauteur(h, base=0.0):
    """
    Ecart tolere (mm) entre la hauteur mesuree et la face superieure
    attendue d'une piece d'epaisseur h. Sur la table : 5 mm ou 35 % (la
    camera lisse les faces etroites). Empilee : 3.5 mm ou 25 %, car les
    hauteurs possibles sont serrees (une fine a plat sur 16 mm donne 24 mm,
    une epaisse sur la tranche 19 mm) et une pile fantome au-dessus d'une
    piece serait sinon acceptee.
    """
    h = np.asarray(h, float)
    if np.ndim(base) == 0 and base <= 0:
        return np.maximum(5.0, 0.35 * h)
    return np.where(np.asarray(base) > 0, np.maximum(3.5, 0.25 * h),
                    np.maximum(5.0, 0.35 * h))


def etages_sous(mods: Sequence[Modele], etages: int = 2) -> Dict[float, int]:
    """Pour chaque base de pile (bases_de_pile), le plus petit nombre de
    pieces couchees qui l'atteint (16 mm : deux fines a plat)."""
    hs = sorted({round(m.h, 1) for m in mods if m.face != "debout" and m.base == 0})
    sortie = {0.0: 0}
    courant = {0.0}
    for n in range(1, etages + 1):
        courant = {round(b + h, 1) for b in courant for h in hs}
        for b in courant:
            sortie.setdefault(b, n)
    return sortie


def modeles_empiles(mods: Sequence[Modele], bases: Sequence[float],
                    hauteur_max: float = 100.0) -> List[Modele]:
    """Les modeles couches (pas debout) poses sur chaque base non nulle."""
    return [m.sur(b) for b in bases if b > 0 for m in mods
            if m.base == 0 and m.face != "debout" and b + m.h < hauteur_max]


@dataclass
class PieceModele(vision.Piece):
    """Une piece, avec le score de confiance de la detection (0 a 1) et la
    part de sa face superieure cachee par les pieces posees dessus (0 a 1 ;
    1 : piece entierement cachee, deduite de la hauteur de celle du dessus)."""
    score: float = 0.0
    cachee: float = 0.0


@dataclass
class ResultatModele(Resultat):
    vue: Optional["VueDessus"] = None
    rectangles: List[Tuple[float, float, float, float, float]] = field(default_factory=list)


@dataclass
class Table:
    """Ce que l'on sait de la table apres l'etape 1."""
    D: np.ndarray                 # profondeur nettoyee (m)
    valides: np.ndarray           # pixels mesures
    P: np.ndarray                 # H x W x 3, repere camera (m)
    n: np.ndarray
    d: float
    T_ct: np.ndarray              # repere table vu de la camera
    T_tc: np.ndarray
    hauteur: np.ndarray           # m, au-dessus de la table (corrigee)
    zone: np.ndarray              # uint8 : enveloppe de la table dans l'image
    table: np.ndarray             # uint8 : pixels du plan
    bruit: float                  # m, ecart-type robuste de la table


@dataclass
class VueDessus:
    """
    Grille vue de dessus, dans le repere TABLE. La case (ligne r, colonne c)
    a son centre en x = x0 + (c + 0.5) res, y = y0 + (r + 0.5) res (mm).
    Les canaux valent NaN ou il n'y a aucune mesure.
    """
    res: float
    x0: float
    y0: float
    hauteur: np.ndarray           # mm
    nombre: np.ndarray            # points tombes dans la case
    infrarouge: Optional[np.ndarray] = None
    couleur: Optional[np.ndarray] = None     # Lab, H x W x 3
    douteux: Optional[np.ndarray] = None     # part de points volants, 0..1

    def vers_case(self, x, y):
        """mm (repere table) -> coordonnees continues (colonne, ligne)."""
        return (np.asarray(x) - self.x0) / self.res - 0.5, \
            (np.asarray(y) - self.y0) / self.res - 0.5

    @property
    def connue(self) -> np.ndarray:
        return ~np.isnan(self.hauteur)


# ---------------------------------------------------------------------------
# 1. Table
# ---------------------------------------------------------------------------

def retirer_volants(tab: "Table", saut_min: float, fenetre: int = 5,
                    part: float = 0.3, etirement: float = 3.0) -> np.ndarray:
    """
    Pixels "volants" : la camera lisse la profondeur sur une petite fenetre,
    si bien qu'au bord d'une piece apparaissent des points A MI-HAUTEUR
    entre la face superieure et la table, qui n'existent pas. Ils tombent
    dans l'ombre de la piece (du cote oppose a la camera) et y dessinent une
    rampe, ou l'on croirait voir une piece fine.

    Deux conditions :
      - le pixel est dans la partie MEDIANE d'un vrai saut de hauteur de sa
        fenetre (amplitude > `saut_min` m, a plus de `part` de l'amplitude
        du bas comme du haut). Un seuil absolu ne convient pas : sur une
        face bruitee, le plus haut de 25 voisins depasse la moyenne de deux
        ecarts-types ;
      - il est sur une discontinuite d'OCCULTATION : vu de dessus, il est a
        plus de `etirement` fois l'empreinte normale d'un pixel de ses DEUX
        voisins d'une meme ligne (ou colonne). La face verticale d'une piece, cote camera, est elle
        aussi a mi-hauteur, mais ses points sont empiles (voisins proches
        en x, y) : ce sont de vrais points, qui marquent le bord exact de
        la piece. Les retirer degradait la position (mesure sur les scenes
        synthetiques).
    On compare des HAUTEURS au-dessus de la table, pas des profondeurs : en
    vue rasante, la profondeur de la table elle-meme varie de plusieurs mm
    d'un pixel au suivant. Rend le masque des pixels a garder.
    """
    valides, hauteur = tab.valides, tab.hauteur
    noyau = np.ones((fenetre, fenetre), np.uint8)
    h_haut = np.where(valides, hauteur, -1.0).astype(np.float32)
    h_bas = np.where(valides, hauteur, 1.0).astype(np.float32)
    hmax = cv2.dilate(h_haut, noyau)
    hmin = cv2.erode(h_bas, noyau)
    ampl = hmax - hmin
    milieu = (valides & (ampl > saut_min) & (hauteur - hmin > part * ampl)
              & (hmax - hauteur > part * ampl))
    # saut en x, y (repere table) vers les voisins de droite et du dessous
    T = tab.T_tc
    xy = tab.P @ T[:3, :2]
    xy = xy + T[:2, 3]
    du = np.full(hauteur.shape, np.nan)
    dv = np.full(hauteur.shape, np.nan)
    ok_u = valides[:, 1:] & valides[:, :-1]
    ok_v = valides[1:, :] & valides[:-1, :]
    du[:, :-1] = np.where(ok_u, np.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=2), np.nan)
    dv[:-1, :] = np.where(ok_v, np.linalg.norm(xy[1:, :] - xy[:-1, :], axis=2), np.nan)
    table = (tab.table > 0)
    ref_u = np.nanmedian(du[table]) if np.isfinite(du[table]).any() else 0.002
    ref_v = np.nanmedian(dv[table]) if np.isfinite(dv[table]).any() else 0.002
    # un point volant flotte dans un VIDE : grand saut vers ses deux voisins
    # d'une meme direction. Le bord d'une face n'a de saut que d'un cote --
    # une face etroite (6 pixels) serait sinon jetee tout entiere
    saut = np.zeros(hauteur.shape, bool)
    with np.errstate(invalid="ignore"):
        su = du > etirement * ref_u          # saut entre (v, u) et (v, u+1)
        sv = dv > etirement * ref_v
    saut[:, 1:-1] = su[:, :-2] & su[:, 1:-1]
    saut[1:-1, :] |= sv[:-2, :] & sv[1:-1, :]
    volant = milieu & saut
    return valides & ~volant


def analyser_table(scene: Scene, hauteur_max: float = 0.10) -> Table:
    """Etape 1, comme vision.detecter : plan, zone de la table, courbure."""
    D = scene.profondeur.astype(np.float64).copy()
    D[D > PROFONDEUR_MAX] = 0.0
    valides = D > 0
    P = nuage(D, scene.K)
    n, d, sur_plan = plan_table(P[valides])
    hauteur = np.zeros(D.shape)
    hauteur[valides] = P[valides] @ n + d
    masque_plan = np.zeros(D.shape, np.uint8)
    masque_plan[valides] = sur_plan.astype(np.uint8)
    _, lab, st, _ = cv2.connectedComponentsWithStats(masque_plan)
    plus_grand = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    table = (lab == plus_grand).astype(np.uint8)
    contours, _ = cv2.findContours(table, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    zone = np.zeros_like(table)
    cv2.drawContours(zone, [cv2.convexHull(max(contours, key=cv2.contourArea))],
                     -1, 1, -1)
    T_ct = repere_table(n, d)
    T_tc = np.linalg.inv(T_ct)
    hauteur = corriger_courbure(hauteur, P, T_tc, valides & (zone > 0))
    h_table = hauteur[(table > 0) & valides]
    bruit = 1.4826 * float(np.median(np.abs(h_table - np.median(h_table))))
    return Table(D, valides, P, n, d, T_ct, T_tc, hauteur, zone, table, bruit)


# ---------------------------------------------------------------------------
# 2. Vue redressee
# ---------------------------------------------------------------------------

def _completer_satures(scene: Scene, tab: Table, garde: np.ndarray,
                       seuil_h: float, seuil_brillant: int = 200,
                       aire_max: int = 3000):
    """
    Pixels SANS profondeur mais brillants dans l'infrarouge et colles a un
    objet : la partie saturee de sa face superieure (cf. vision.detecter).
    On les place a la hauteur de l'objet voisin, par le rayon du pixel
    (rayons_sur_plan). Une grande tache (reflet de lampe) est ignoree.
    Rend (u, v, h en m) des pixels completes.
    """
    if scene.infrarouge is None:
        return None
    trous = ((~tab.valides) & (tab.zone > 0)
             & (scene.infrarouge >= seuil_brillant)).astype(np.uint8)
    trous = cv2.morphologyEx(trous, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nb, lab, st, _ = cv2.connectedComponentsWithStats(trous)
    objet = garde & (tab.hauteur > seuil_h)
    us, vs, hs = [], [], []
    for i in range(1, nb):
        if st[i, cv2.CC_STAT_AREA] > aire_max:
            continue
        x, y, w, hh = st[i, :4]
        x0, y0 = max(x - 4, 0), max(y - 4, 0)
        sous = (lab[y0:y + hh + 4, x0:x + w + 4] == i).astype(np.uint8)
        anneau = cv2.dilate(sous, np.ones((7, 7), np.uint8)) > 0
        voisins = objet[y0:y + hh + 4, x0:x + w + 4] & anneau
        if voisins.sum() < 5:
            continue
        h = float(np.percentile(tab.hauteur[y0:y + hh + 4, x0:x + w + 4][voisins], 80))
        v_, u_ = np.nonzero(sous)
        us.append(u_ + x0)
        vs.append(v_ + y0)
        hs.append(np.full(len(u_), h))
    if not us:
        return None
    return np.concatenate(us), np.concatenate(vs), np.concatenate(hs)


def vue_dessus(scene: Scene, tab: Table, garde: np.ndarray, res: float = 1.0,
               seuil_h: float = 0.004, hauteur_max: float = 0.10,
               etendue_max: float = 1200.0, sur_ech_max: int = 8,
               continuite: float = 4.0,
               douteux: Optional[np.ndarray] = None) -> VueDessus:
    """
    Etape 2 : orthophoto de la table.

    Une case de 1 mm est plus petite que l'empreinte d'un pixel loin de la
    camera (3 mm sur 8 en vue rasante a 1 m) : projeter seulement le centre
    des pixels laisserait des cases vides entre eux. On SUR-ECHANTILLONNE
    donc chaque pixel (k_u x k_v sous-points, k choisi d'apres l'empreinte
    mesuree) en interpolant entre lui et ses voisins, mais seulement si les
    quatre sont sur une meme surface (hauteurs proches) : on ne fabrique pas
    de points entre une piece et la table.

    `douteux` : pixels (image) a projeter mais a marquer, les pixels volants ;
    la vue garde pour chaque case la part de ses points qui sont douteux.
    """
    Hh, Ww = tab.D.shape
    T = tab.T_tc
    Q = (tab.P @ T[:3, :3].T + T[:3, 3]) * 1000.0          # mm, repere table
    Q[..., 2] = tab.hauteur * 1000.0
    ok = garde & (tab.zone > 0) & (tab.hauteur > -0.02) & (tab.hauteur < hauteur_max)
    Q[~ok] = np.nan

    # empreinte d'un pixel sur la table, le long de u et de v
    du = np.linalg.norm(Q[:, 1:, :2] - Q[:, :-1, :2], axis=2)
    dv = np.linalg.norm(Q[1:, :, :2] - Q[:-1, :, :2], axis=2)
    du = du[np.isfinite(du) & (du < 30)]
    dv = dv[np.isfinite(dv) & (dv < 30)]
    k_u = int(np.clip(np.ceil(np.percentile(du, 90) / res), 1, sur_ech_max)) if du.size else 1
    k_v = int(np.clip(np.ceil(np.percentile(dv, 90) / res), 1, sur_ech_max)) if dv.size else 1

    ir = None
    if scene.infrarouge is not None:
        # le projecteur seme des points brillants : on les efface
        ir = cv2.medianBlur(scene.infrarouge, 7).astype(np.float32)
    lab_img = None
    if scene.couleur is not None:
        lab_img = cv2.cvtColor(scene.couleur.astype(np.float32) / 255.0,
                               cv2.COLOR_BGR2Lab)
        lab_img[scene.couleur.max(axis=2) == 0] = np.nan

    # quatre coins de chaque maille (v, u), (v, u+1), (v+1, u), (v+1, u+1)
    Q00, Q01, Q10, Q11 = Q[:-1, :-1], Q[:-1, 1:], Q[1:, :-1], Q[1:, 1:]
    hs = np.stack([Q00[..., 2], Q01[..., 2], Q10[..., 2], Q11[..., 2]])
    with np.errstate(invalid="ignore"):
        # (max et min rendent NaN des qu'un coin manque : la maille n'est
        # alors pas continue)
        continu = hs.max(axis=0) - hs.min(axis=0) < continuite
    porteur = np.isfinite(Q00[..., 2])
    vv, uu = np.nonzero(porteur)
    c = continu[vv, uu]
    morceaux = []
    for a in (np.arange(k_u) / k_u):
        for b in (np.arange(k_v) / k_v):
            q = Q00[vv, uu]
            if a or b:
                qi = ((1 - a) * (1 - b) * Q00[vv, uu] + a * (1 - b) * Q01[vv, uu]
                      + (1 - a) * b * Q10[vv, uu] + a * b * Q11[vv, uu])
                q = np.where(c[:, None], qi, q)
            morceaux.append(q)
    pts = np.concatenate(morceaux)
    src = np.tile(np.stack([vv, uu], axis=1), (k_u * k_v, 1))

    # pixels satures, completes a la hauteur de la piece voisine
    sat = _completer_satures(scene, tab, garde, seuil_h)
    if sat is not None:
        u_s, v_s, h_s = sat
        # le rayon du pixel coupe le plan parallele a la table a la hauteur
        # h (cf. vision.rayons_sur_plan, ici avec un h par pixel)
        fx, fy, cx, cy = scene.K[0, 0], scene.K[1, 1], scene.K[0, 2], scene.K[1, 2]
        r = np.stack([(u_s - cx) / fx, (v_s - cy) / fy, np.ones(len(u_s))], axis=1)
        t = (h_s - tab.d) / (r @ tab.n)
        p3 = r * t[:, None]
        q3 = (p3 @ T[:3, :3].T + T[:3, 3]) * 1000.0
        q3[:, 2] = h_s * 1000.0
        pts = np.concatenate([pts, q3])
        src = np.concatenate([src, np.stack([v_s, u_s], axis=1)])

    # etendue de la grille : autour de la table, bornee
    xy = pts[:, :2]
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    centre = (lo + hi) / 2
    demi = np.minimum((hi - lo) / 2 + 10, etendue_max / 2)
    x0, y0 = centre - demi
    nx, ny = (np.ceil(2 * demi / res).astype(int))
    col = np.floor((xy[:, 0] - x0) / res).astype(np.int64)
    lig = np.floor((xy[:, 1] - y0) / res).astype(np.int64)
    dedans = (col >= 0) & (col < nx) & (lig >= 0) & (lig < ny)
    idx = lig[dedans] * nx + col[dedans]
    h = pts[dedans, 2]
    src = src[dedans]
    nombre = np.bincount(idx, minlength=nx * ny)
    if douteux is not None:
        n_d = np.bincount(idx, weights=douteux[src[:, 0], src[:, 1]].astype(float),
                          minlength=nx * ny)
    somme = np.bincount(idx, weights=h, minlength=nx * ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        hauteur = (somme / nombre).reshape(ny, nx).astype(np.float32)
    vue = VueDessus(res, float(x0), float(y0), hauteur,
                    nombre.reshape(ny, nx).astype(np.int32))
    if douteux is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            vue.douteux = (n_d / nombre).reshape(ny, nx).astype(np.float32)
    if ir is not None:
        s = np.bincount(idx, weights=ir[src[:, 0], src[:, 1]], minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            vue.infrarouge = (s / nombre).reshape(ny, nx).astype(np.float32)
    if lab_img is not None:
        lab_pts = lab_img[src[:, 0], src[:, 1]]
        bon = np.isfinite(lab_pts).all(axis=1)
        nb_c = np.bincount(idx[bon], minlength=nx * ny)
        canaux = []
        for k in range(3):
            s = np.bincount(idx[bon], weights=lab_pts[bon, k], minlength=nx * ny)
            with np.errstate(invalid="ignore", divide="ignore"):
                canaux.append((s / nb_c).reshape(ny, nx))
        vue.couleur = np.dstack(canaux).astype(np.float32)
    _boucher(vue)
    return vue


def _boucher(vue: VueDessus, passes: int = 2) -> None:
    """Petits trous (case vide entouree de cases mesurees) : moyenne des
    voisines. Les grands vides (ombres) restent INCONNUS."""
    for _ in range(passes):
        canaux = [vue.hauteur] + [c for c in (vue.infrarouge, vue.douteux) if c is not None]
        connu = np.isfinite(vue.hauteur)
        n_vois = cv2.boxFilter(connu.astype(np.float32), -1, (3, 3), normalize=False)
        a_remplir = (~connu) & (n_vois >= 5)
        if not a_remplir.any():
            break
        for img in canaux:
            z = np.where(np.isfinite(img), img, 0).astype(np.float32)
            s = cv2.boxFilter(z, -1, (3, 3), normalize=False)
            img[a_remplir] = (s / np.maximum(n_vois, 1))[a_remplir]


# ---------------------------------------------------------------------------
# 3. Indices
# ---------------------------------------------------------------------------

def _melange(x: np.ndarray, k: int = 4, graine: int = 0):
    """Melange de gaussiennes simplifie (k-moyennes puis covariance de
    chaque groupe) : la nappe a fleurs a plusieurs couleurs, une seule
    gaussienne la decrirait mal."""
    x = x.astype(np.float32)
    k = min(k, max(1, len(x) // 50))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    cv2.setRNGSeed(graine)
    _, lab, centres = cv2.kmeans(x, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    groupes = []
    for i in range(k):
        xi = x[lab.ravel() == i]
        if len(xi) < 10:
            continue
        cov = np.cov(xi.T) + np.eye(x.shape[1]) * 4.0
        groupes.append((xi.mean(axis=0), np.linalg.inv(cov)))
    return groupes


def _mahalanobis(x: np.ndarray, groupes) -> np.ndarray:
    d2 = np.full(len(x), np.inf)
    for mu, icov in groupes:
        e = x - mu
        d2 = np.minimum(d2, np.einsum("ij,jk,ik->i", e, icov, e))
    return np.sqrt(d2)


@dataclass
class Indices:
    p: np.ndarray                 # probabilite de premier plan, 0..1
    H: np.ndarray                 # hauteur (mm), 0 la ou inconnue
    connu: np.ndarray             # float 0/1
    dt: np.ndarray                # N_ORIENT x ny x nx : distance (mm) aux contours
    aretes: np.ndarray            # uint8, pour l'affichage
    segments: np.ndarray          # M x 4 (cases) : segments LSD
    orient: np.ndarray            # angle du grand axe local (rad), tenseur de structure
    seuil_h: float                # mm
    sigma_h: float                # mm, bruit de hauteur de la table
    d_inconnu: np.ndarray         # mm, distance a la case inconnue la plus proche
    couleur_utile: bool = False
    textures: Optional[Textures] = None
    hors_table: Optional[np.ndarray] = None   # 1 : case hors de la table
    rampe: Optional[np.ndarray] = None        # 1 : case sur un saut de hauteur


def indices(vue: VueDessus, seuil_h: float, sigma_h: float,
            poids_couleur: float = 0.0, poids_ir: float = 0.0) -> Indices:
    """
    Etape 3 : premier plan et contours, en fusionnant hauteur, couleur et
    infrarouge. Chaque indice donne un log-rapport de vraisemblance borne
    (+ : piece, - : table, 0 : ne sait pas), et on les ADDITIONNE, ce qui
    revient a les supposer independants (Bayes naif). La hauteur domine.

    La couleur et l'infrarouge sont DESACTIVES par defaut (poids 0) dans le
    premier plan : mesure faite (evaluer_detection.py), ils n'ajoutaient
    aucune piece juste, ni sur les scenes synthetiques ni sur les captures
    reelles, et la couleur ajoutait des faux positifs (taches de la nappe
    mal decrites par le modele de la table) et degradait la position. Ils
    servent en revanche comme CONTOURS, redresses au plan de la face
    (textures_redressees).
    """
    res = vue.res
    # deux lectures de la hauteur : TOUS les points pour les contours (les
    # pixels volants y marquent le bord eloigne de la piece), et les points
    # FIABLES pour le reste (les volants dessinent dans l'ombre une rampe ou
    # l'on croirait voir une piece fine : ces cases sont tenues pour
    # inconnues)
    connu_tout = np.isfinite(vue.hauteur)
    H_tout = np.where(connu_tout, vue.hauteur, 0.0).astype(np.float32)
    connu = connu_tout.copy()
    if vue.douteux is not None:
        connu &= ~(np.nan_to_num(vue.douteux, nan=0.0) > 0.5)
    H = np.where(connu, vue.hauteur, 0.0).astype(np.float32)
    s = max(sigma_h, 0.8)
    L_h = np.clip((H - seuil_h) / s, -3.0, 3.0)
    L_h[~connu] = 0.0
    L = L_h.copy()

    # table "sure" : basse, et loin de tout ce qui depasse
    haut = (connu & (H > seuil_h)).astype(np.uint8)
    loin = cv2.distanceTransform(1 - haut, cv2.DIST_L2, 3) * res > 6.0
    sure = connu & (np.abs(H) < max(2.5 * sigma_h, 1.5)) & loin

    dist_c = None
    couleur_utile = False
    if vue.couleur is not None and poids_couleur > 0:
        c_ok = np.isfinite(vue.couleur).all(axis=2)
        x_tab = vue.couleur[sure & c_ok]
        if len(x_tab) > 500:
            rng = np.random.default_rng(0)
            if len(x_tab) > 20000:
                x_tab = x_tab[rng.choice(len(x_tab), 20000, replace=False)]
            groupes = _melange(x_tab)
            dist_c = np.zeros(H.shape, np.float32)
            dist_c[c_ok] = _mahalanobis(vue.couleur[c_ok], groupes)
            # seuil : ce que depasse 2 % de la table elle-meme
            ref = np.percentile(_mahalanobis(x_tab, groupes), 98)
            L_c = np.clip((dist_c - ref) / max(0.25 * ref, 0.5), -1.5, 1.5)
            L_c[~c_ok] = 0.0
            L += poids_couleur * L_c
            couleur_utile = True

    if vue.infrarouge is not None and poids_ir > 0:
        i_ok = np.isfinite(vue.infrarouge)
        ir_tab = vue.infrarouge[sure & i_ok]
        if len(ir_tab) > 500:
            med = np.median(ir_tab)
            mad = 1.4826 * np.median(np.abs(ir_tab - med)) + 1.0
            L_i = np.clip((vue.infrarouge - med) / (3 * mad) - 1.0, -1.0, 1.0)
            L_i[~i_ok] = 0.0
            L += poids_ir * L_i

    p = (1.0 / (1.0 + np.exp(-L))).astype(np.float32)

    # --- contours de hauteur : Canny sur le gradient de la hauteur (de
    # TOUS les points, volants compris)
    Hl = cv2.GaussianBlur(H_tout, (0, 0), 0.7)
    gx = cv2.Sobel(Hl, cv2.CV_32F, 1, 0, ksize=3) / (8 * res)
    gy = cv2.Sobel(Hl, cv2.CV_32F, 0, 1, ksize=3) / (8 * res)
    g_tab = np.hypot(gx, gy)[sure]
    bruit_g = float(np.percentile(g_tab, 99)) if g_tab.size else 0.5
    # seuils de Canny : le seuil bas au niveau du bruit de la table (99e
    # centile du gradient), le seuil haut 1.6 fois plus. Une piece fine a
    # plat (8 mm) dont le bord est lisse sur 6 mm ne donne qu'un gradient de
    # 1.2 mm/mm : des seuils plus hauts la perdaient (scene9)
    bas = max(0.25, bruit_g)
    echelle = 100.0
    ar_h = cv2.Canny((gx * echelle).astype(np.int16), (gy * echelle).astype(np.int16),
                     bas * echelle, 1.6 * bas * echelle, L2gradient=True) > 0
    # un contour colle a une case INCONNUE n'est pas garde : les cases
    # inconnues valant 0, le saut est artificiel. Du cote oppose a la camera,
    # la face superieure se termine dans une bande inconnue (ombre, pixels
    # volants retires) : le vrai bord est QUELQUE PART dans cette bande, pas
    # au bout du plateau mesure. Le noteur traite ces bords comme neutres.
    ny, nx = H.shape
    inconnu = (~connu_tout).astype(np.uint8)
    pres_inconnu = cv2.dilate(inconnu, np.ones((5, 5), np.uint8)) > 0
    ar_h &= ~pres_inconnu
    ang_h = np.arctan2(gy, gx)
    d_inconnu = cv2.distanceTransform(connu.astype(np.uint8), cv2.DIST_L2, 3) * res

    aretes = np.zeros(H.shape, np.uint8)
    aretes[ar_h] = 1
    angle = ang_h
    if dist_c is not None:
        dc = cv2.GaussianBlur(np.clip(dist_c, 0, 3 * ref).astype(np.float32), (0, 0), 1.0)
        cgx = cv2.Sobel(dc, cv2.CV_32F, 1, 0, ksize=3)
        cgy = cv2.Sobel(dc, cv2.CV_32F, 0, 1, ksize=3)
        e = max(ref, 1.0)
        ar_c = cv2.Canny((cgx / e * 100).astype(np.int16), (cgy / e * 100).astype(np.int16),
                         60, 120, L2gradient=True) > 0
        ar_c &= aretes == 0
        aretes[ar_c] = 3
        angle = np.where(ar_c, np.arctan2(cgy, cgx), angle)

    # distance (mm) aux contours de chaque classe d'orientation : le
    # chanfrein ORIENTE ne compte un contour que s'il a la direction du
    # cote du rectangle, ce qui evite qu'un cote "colle" a une arete
    # perpendiculaire (une piece voisine en T, une fleur)
    dt = _dt_orientes(aretes > 0, angle, res)

    # segments droits (LSD) sur l'image de hauteur, et sur la couleur
    lsd = cv2.createLineSegmentDetector()
    img_h = np.clip(H * 8, 0, 255).astype(np.uint8)
    segs = [lsd.detect(img_h)[0]]
    if dist_c is not None:
        segs.append(lsd.detect(np.clip(dist_c / max(3 * ref, 1) * 255, 0, 255).astype(np.uint8))[0])
    segs = [s.reshape(-1, 4) for s in segs if s is not None]
    segments = np.concatenate(segs) if segs else np.zeros((0, 4), np.float32)
    longueur = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]) * res
    segments = segments[longueur >= 8.0]

    # orientation locale (tenseur de structure) : direction du grand axe
    sig = 6.0 / res
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sig)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sig)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sig)
    orient = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2
    # RAMPES : cases dont le voisinage (5 x 5 mm) couvre un grand saut de
    # hauteur. Le lissage de la camera y place des points a mi-hauteur,
    # jusqu'a l'INTERIEUR de l'empreinte d'une piece haute (cote camera) :
    # pour une piece empilee (34 mm), cette rampe mord la moitie d'une face
    # de 19 mm et la ferait prendre pour une piece etroite. Seules les
    # pieces empilees les ignorent (voir Noteur.noter)
    H_haut = np.where(connu, H, -1e3).astype(np.float32)
    H_bas = np.where(connu, H, 1e3).astype(np.float32)
    noyau = np.ones((5, 5), np.uint8)
    etendue = cv2.dilate(H_haut, noyau) - cv2.erode(H_bas, noyau)
    rampe = (connu & (etendue > max(6.0, 4.0 * sigma_h)) & (etendue < 500)).astype(np.float32)
    return Indices(p, H, connu.astype(np.float32), dt, aretes, segments,
                   orient.astype(np.float32), seuil_h, sigma_h, d_inconnu,
                   couleur_utile, rampe=rampe)


def _vers_image(tab: Table, vue: VueDessus, K: np.ndarray, h: float):
    """Pixel (u, v) ou se projette chaque case de la vue, a la hauteur h (mm)."""
    ny, nx = vue.hauteur.shape
    lig, col = np.mgrid[0:ny, 0:nx].astype(np.float32)
    X = (vue.x0 + (col + 0.5) * vue.res) / 1000.0
    Y = (vue.y0 + (lig + 0.5) * vue.res) / 1000.0
    R, t = tab.T_ct[:3, :3], tab.T_ct[:3, 3]
    Pc = [R[i, 0] * X + R[i, 1] * Y + R[i, 2] * h / 1000.0 + t[i] for i in range(3)]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = (K[0, 0] * Pc[0] / Pc[2] + K[0, 2]).astype(np.float32)
        mv = (K[1, 1] * Pc[1] / Pc[2] + K[1, 2]).astype(np.float32)
    H_img, W_img = tab.D.shape
    dedans = (Pc[2] > 0) & (mu >= 1) & (mu < W_img - 2) & (mv >= 1) & (mv < H_img - 2)
    return mu, mv, dedans


def hors_table(tab: Table, vue: VueDessus, K: np.ndarray) -> np.ndarray:
    """Cases de la vue hors de la zone de la table (ou hors de l'image).
    L'inconnu y n'est pas neutre : une piece n'y est pas posee."""
    mu, mv, dedans = _vers_image(tab, vue, K, 0.0)
    zone = cv2.remap(tab.zone, mu, mv, cv2.INTER_NEAREST, borderValue=0)
    return ((zone == 0) | ~dedans).astype(np.float32)


@dataclass
class Textures:
    """Contours de l'infrarouge et de la couleur, redresses sur des plans
    paralleles a la table, un par hauteur de face superieure possible."""
    niveaux: np.ndarray           # mm
    dt: np.ndarray                # niveaux x N_ORIENT x ny x nx, uint8 : distance (mm) x 4
    segments: np.ndarray          # M x 4 (cases), au niveau le plus frequent


def _dt_orientes(aretes: np.ndarray, angle: np.ndarray, res: float,
                 code: bool = False) -> np.ndarray:
    """Distance (mm) aux contours de chaque classe d'orientation."""
    phi = np.mod(angle, np.pi)
    classe = np.floor(phi / (np.pi / N_ORIENT)).astype(int) % N_ORIENT
    sortie = np.empty((N_ORIENT,) + aretes.shape, np.uint8 if code else np.float32)
    for b in range(N_ORIENT):
        m = aretes & (classe == b)
        d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) * res
        sortie[b] = np.clip(d * 4, 0, 255).astype(np.uint8) if code else d
    return sortie


def textures_redressees(scene: Scene, tab: Table, vue: VueDessus,
                        niveaux: Sequence[float]) -> Optional[Textures]:
    """
    La vue redressee range chaque point MESURE a sa place. Mais un pixel sans
    profondeur (face saturee, emetteur eteint, bord d'occultation) n'y
    apparait pas, alors que l'infrarouge et la couleur le voient tres bien.
    On redresse donc AUSSI les images elles-memes, par BALAYAGE DE PLANS :
    pour chaque hauteur h de face superieure possible (8, 17, 19 mm...), la
    case (x, y) de la grille est reprojetee dans l'image au point 3D
    (x, y, h) du repere table, et l'on y lit l'infrarouge (cv2.remap). Une
    face a la hauteur h est alors a sa vraie place et a sa vraie taille dans
    l'image du niveau h -- et decalee dans les autres. Le bord ELOIGNE de la
    face, perdu par la profondeur (ombre, lissage), y est net.

    Contours : Canny sur l'infrarouge redresse, et sur la distance de
    couleur a la table (modele de melange appris sur les pixels de la
    table), seuils adaptes (centiles du gradient).
    """
    images = []
    if scene.infrarouge is not None:
        ir = cv2.medianBlur(scene.infrarouge, 5).astype(np.float32)
        images.append((ir, None))
    if scene.couleur is not None:
        noir = scene.couleur.max(axis=2) == 0
        lab = cv2.cvtColor(scene.couleur.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        table = (tab.valides & (np.abs(tab.hauteur) < 0.002) & (tab.zone > 0) & ~noir)
        x = lab[table]
        if len(x) > 500:
            rng = np.random.default_rng(0)
            if len(x) > 20000:
                x = x[rng.choice(len(x), 20000, replace=False)]
            groupes = _melange(x)
            dist = _mahalanobis(lab.reshape(-1, 3), groupes).reshape(noir.shape)
            ref = float(np.percentile(_mahalanobis(x, groupes), 98))
            dist = np.clip(dist / max(ref, 1e-6), 0, 3).astype(np.float32)
            images.append((dist, noir))
    if not images:
        return None
    ny, nx = vue.hauteur.shape
    res = vue.res
    dts = np.empty((len(niveaux), N_ORIENT, ny, nx), np.uint8)
    segments = np.zeros((0, 4), np.float32)
    for k, h in enumerate(niveaux):
        mu, mv, dans_image = _vers_image(tab, vue, scene.K, h)
        aretes = np.zeros((ny, nx), bool)
        angle = np.zeros((ny, nx), np.float32)
        for img, invalide in images:
            o = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            o = cv2.GaussianBlur(o, (0, 0), 1.0)
            ok = dans_image.copy()
            if invalide is not None:
                inv = cv2.remap(invalide.astype(np.uint8), mu, mv, cv2.INTER_NEAREST)
                ok &= cv2.dilate(inv, np.ones((5, 5), np.uint8)) == 0
            gx = cv2.Sobel(o, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(o, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)[ok]
            if mag.size < 100:
                continue
            haut = max(float(np.percentile(mag, 95)), 1e-3)
            echelle = 1000.0 / haut
            a = cv2.Canny((gx * echelle).astype(np.int16), (gy * echelle).astype(np.int16),
                          500, 1000, L2gradient=True) > 0
            a &= ok
            angle = np.where(a & ~aretes, np.arctan2(gy, gx), angle)
            aretes |= a
            if k == len(niveaux) // 2:
                lsd = cv2.createLineSegmentDetector()
                g8 = np.clip(o / max(float(np.percentile(o[ok], 99)), 1e-6) * 255,
                             0, 255).astype(np.uint8)
                sg = lsd.detect(g8)[0]
                if sg is not None:
                    segments = np.concatenate([segments, sg.reshape(-1, 4)])
        dts[k] = _dt_orientes(aretes, angle, res, code=True)
    return Textures(np.asarray(niveaux, float), dts, segments)


# ---------------------------------------------------------------------------
# 4. Rectangles de taille connue
# ---------------------------------------------------------------------------

def _echantillonner(img: np.ndarray, c: np.ndarray, l: np.ndarray,
                    bilineaire: bool = True, defaut: float = 0.0) -> np.ndarray:
    """
    Valeurs d'une image (1 a 4 canaux, float32) aux positions continues
    (colonne c, ligne l), de forme quelconque, par cv2.remap : en C, et tous
    les canaux d'un coup. Rend un tableau de forme c.shape (+ canaux).
    """
    forme = c.shape
    n = c.size
    larg = 1024
    nl = -(-n // larg)
    mx = np.zeros(nl * larg, np.float32)
    my = np.zeros(nl * larg, np.float32)
    mx[:n] = c.ravel()
    my[:n] = l.ravel()
    out = cv2.remap(img, mx.reshape(nl, larg), my.reshape(nl, larg),
                    cv2.INTER_LINEAR if bilineaire else cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=defaut)
    nc = 1 if img.ndim == 2 else img.shape[2]
    return out.reshape(-1, nc)[:n].reshape(forme + ((nc,) if nc > 1 else ()))


@dataclass
class Gabarit:
    """Points d'echantillonnage d'un rectangle unite (cotes 1 x 1), mis a
    l'echelle de chaque hypothese : interieur, bords (avec l'angle de leur
    normale), anneau exterieur."""
    interieur: np.ndarray        # n x 2, dans [-0.5, 0.5]
    bord: np.ndarray             # m x 2
    normale: np.ndarray          # m : 0 (bout) ou pi/2 (flanc), repere piece
    bord_long: np.ndarray        # m : vrai sur les grands cotes
    anneau: np.ndarray           # k x 2 : points sur les cotes, a decaler
    anneau_dir: np.ndarray       # k x 2 : direction du decalage


def gabarit(L: float, l: float, pas: float) -> Gabarit:
    nL = max(3, int(round(L / pas)))
    nl = max(2, int(round(l / pas)))
    s = (np.arange(nL) + 0.5) / nL - 0.5
    t = (np.arange(nl) + 0.5) / nl - 0.5
    S, Tt = np.meshgrid(s, t)
    interieur = np.stack([S.ravel(), Tt.ravel()], axis=1)
    # bords : on evite les coins, arrondis par le lissage
    s_b = s[np.abs(s) < 0.45] if (np.abs(s) < 0.45).sum() >= 2 else s
    t_b = t[np.abs(t) < 0.40] if (np.abs(t) < 0.40).sum() >= 1 else t
    bord, normale, long_, anneau, adir = [], [], [], [], []
    for sg in (-0.5, 0.5):
        bord += [[x, sg] for x in s_b]
        normale += [np.pi / 2] * len(s_b)
        long_ += [True] * len(s_b)
        anneau += [[x, sg] for x in s]
        adir += [[0, np.sign(sg)]] * len(s)
        bord += [[sg, y] for y in t_b]
        normale += [0.0] * len(t_b)
        long_ += [False] * len(t_b)
        anneau += [[sg, y] for y in t]
        adir += [[np.sign(sg), 0]] * len(t)
    return Gabarit(interieur, np.array(bord), np.array(normale), np.array(long_),
                   np.array(anneau), np.array(adir, float))


class Noteur:
    """
    Note des hypotheses de rectangles, par paquets (tout est vectorise : des
    milliers d'hypotheses a la fois). Le score, entre 0 et 1, combine :

      E : appui des ARETES. Pour chaque point du bord, la distance d au
          contour le plus proche de meme orientation (chanfrein oriente),
          convertie en exp(-d^2 / 2 s^2) ; contours de hauteur, ou de
          texture redressee au plan de la face (ponderes par
          `poids_texture`). Un bord qui tombe dans l'INCONNU vaut 0.5 : ni
          appui, ni absence d'appui. C'est l'indice qui place le rectangle
          au millimetre et qui separe deux pieces qui se touchent (leurs
          autres cotes sont francs).
      F : REMPLISSAGE. Probabilite moyenne de premier plan a l'interieur,
          en ne comptant pas ce qui est deja explique par une autre piece.
      Hf: HAUTEUR. Accord de la hauteur mesuree a l'interieur avec la
          hauteur attendue de la face superieure : c'est ce qui dit le type
          et la face d'appui.
      O : DEBORDEMENT. Part de l'anneau exterieur (3 mm autour), cote par
          cote, a une hauteur proche de celle de la face : la meme surface
          continue au-dela du rectangle, il est trop petit ou mal place.
    Pour une hypothese EMPILEE (base > 0), les cases recouvertes par une
    piece retenue posee au-dessus sont cachees (neutres), celles de sa
    porteuse ne sont pas un chevauchement, et le score est multiplie par
    la preuve d'appui (voir l'en-tete, etape 6).
    S = 0.35 E + 0.20 F + 0.25 Hf + 0.20 (1 - O), puis des penalites :
      X : CHEVAUCHEMENT avec les pieces deja retenues (au-dela de 20 %,
          rejet) ; hors de la table ; PENTE de la hauteur a travers la
          largeur (une rampe n'est pas une face superieure) ; debordement
          de plus de la moitie des cotes ; et un facteur qui exige qu'au
          moins le quart de la face soit vu a la bonne hauteur.
    Les poids ont ete regles a la main sur les scenes synthetiques, puis
    verifies sur les captures reelles (evaluer_detection.py).
    """

    def __init__(self, vue: VueDessus, ind: Indices, poids=(0.35, 0.20, 0.25, 0.20),
                 sigma_e: float = 1.5, decal_anneau: float = 3.0,
                 poids_texture: float = 0.8, poids_pente: float = 0.2,
                 tol_pile: float = 0.6):
        self.poids_texture = poids_texture
        self.poids_pente = poids_pente
        self.vue = vue
        self.ind = ind
        self.poids = poids
        self.sigma_e = sigma_e
        self.decal = decal_anneau
        self.tol_pile = tol_pile
        self._explique = np.zeros(ind.H.shape, np.float32)
        # pieces retenues : (x, y, th, L, l, base, haut, trace) ; `trace`
        # n'est fourni que pour une explication donnee en bloc (setter)
        self._couches: List[tuple] = []
        self._piles: Dict[tuple, np.ndarray] = {}
        self.tol_h = max(2.5, 2.0 * ind.sigma_h)
        self._gab: Dict[Tuple[float, float, float], Gabarit] = {}
        #: nombre de pieces sous chaque base (etages_sous) ; rempli par detecter
        self.etages: Dict[float, int] = {}

    # -- explication : quelles cases sont deja expliquees, et par quoi ------
    @property
    def explique(self) -> np.ndarray:
        """Union des empreintes des pieces retenues (0 a 1)."""
        return self._explique

    @explique.setter
    def explique(self, valeur: np.ndarray) -> None:
        # explication donnee en bloc : on ne sait pas a quelle hauteur sont
        # les pieces, elle entre en conflit avec toute hypothese
        self._explique = valeur
        self._couches = ([(0, 0, 0, 0, 0, -np.inf, np.inf, valeur.astype(np.float32))]
                         if np.any(valeur) else [])
        self._piles = {}

    def sauver(self):
        return self._explique.copy(), list(self._couches)

    def restaurer(self, etat) -> None:
        self._explique, self._couches = etat[0].copy(), list(etat[1])
        self._piles = {}

    def classe(self, couche, base: float, haut: float) -> int:
        """
        Relation entre une piece retenue (`couche`) et une hypothese posee a
        `base`, de face superieure a `haut` (mm), la ou leurs empreintes se
        recouvrent :
          0 CONFLIT : leurs tranches de hauteur se coupent, deux pieces ne
            peuvent pas occuper le meme volume ;
          1 DESSUS : la piece retenue est posee au-dessus de l'hypothese ; la
            ou elle la recouvre, la face de l'hypothese est CACHEE (ni pour,
            ni contre) ;
          2 SUPPORT : la piece retenue est sous l'hypothese, son sommet a la
            hauteur de la base : c'est la porteuse ;
          3 DESSOUS : plus bas, sans contact : sans effet.
        Les hauteurs comparees sont NOMINALES (celles des modeles), d'ou la
        petite tolerance `tol_pile` : une piece de 24 mm de haut ne repose
        pas sur une piece de 19 mm. Avec des pieces toutes sur la table,
        tout recouvrement est un conflit, comme avant l'empilement.
        """
        t = self.tol_pile
        b_k, h_k = couche[5], couche[6]
        if b_k >= haut - t:
            return 1
        if h_k <= base + t:
            return 2 if (base > 0 and abs(h_k - base) <= t) else 3
        return 0

    def _trace(self, couche) -> np.ndarray:
        if couche[7] is not None:
            return couche[7]
        m = np.zeros_like(self._explique)
        self._dessiner(m, *couche[:5])
        return m

    def pile(self, base: float = 0.0, haut: Optional[float] = None) -> np.ndarray:
        """
        Les cartes lues a chaque point, empilees pour cv2.remap : premier
        plan, mesure connue (et non cachee), hauteur, conflit, cache, support.
        Elles dependent de la hauteur de l'hypothese (ce qui est conflit pour
        l'une est support pour l'autre) : une pile par configuration.
        """
        classes = tuple(self.classe(c, base, np.inf if haut is None else haut)
                        for c in self._couches)
        cle = classes
        if cle not in self._piles:
            ind = self.ind
            zero = np.zeros_like(self._explique)
            if all(k == 0 for k in classes):
                conflit, cache, support = self._explique, zero, zero
            else:
                conflit, cache, support = zero.copy(), zero.copy(), zero.copy()
                for c, k in zip(self._couches, classes):
                    if k == 0:
                        if c[7] is None:
                            self._dessiner(conflit, *c[:5])
                        else:
                            conflit = np.maximum(conflit, c[7])
                    elif k == 1:
                        cache = np.maximum(cache, self._trace(c))
                    elif k == 2:
                        support = np.maximum(support, self._trace(c))
            connu = ind.connu * (1.0 - cache)
            rampe = ind.rampe if ind.rampe is not None else zero
            self._piles[cle] = np.dstack([ind.p, connu, ind.H, conflit, cache,
                                          support, rampe]).astype(np.float32)
        return self._piles[cle]

    def gab(self, L, l, pas) -> Gabarit:
        cle = (round(L, 1), round(l, 1), pas)
        if cle not in self._gab:
            self._gab[cle] = gabarit(L, l, pas)
        return self._gab[cle]

    def noter_m(self, x, y, th, m: Modele, **kw):
        """noter pour un modele (dimensions, hauteur et base)."""
        return self.noter(x, y, th, m.L, m.l, m.haut, base=m.base, h_piece=m.h, **kw)

    def noter(self, x, y, th, L, l, h, pas: float = 1.5, fin: bool = True,
              detail: bool = False, base: float = 0.0, h_piece: Optional[float] = None,
              appui_min: float = 0.0):
        """
        x, y (mm, repere table), th (rad), L, l (mm) : tableaux (N,) ; h :
        hauteur de la face superieure (mm), la meme pour tout le paquet.
        `base` : hauteur de la face d'appui (0 : la table), `h_piece` :
        epaisseur de la piece (h - base par defaut). `appui_min` : preuve
        d'appui accordee d'office a une piece empilee (voir detecter).
        """
        x, y, th = (np.asarray(v, np.float64) for v in (x, y, th))
        N = len(x)
        L = np.broadcast_to(np.asarray(L, np.float64), (N,))
        l = np.broadcast_to(np.asarray(l, np.float64), (N,))
        h = np.broadcast_to(np.asarray(h, np.float64), (N,))
        hp = h - base if h_piece is None else np.broadcast_to(
            np.asarray(h_piece, np.float64), (N,))
        g = self.gab(float(L.max()), float(l.max()), pas)
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        ind = self.ind
        pile = self.pile(base, float(h[0]) if N else 0.0)

        def placer(loc):                       # loc : (N, n, 2) repere piece
            X = x[:, None] + c * loc[..., 0] - s * loc[..., 1]
            Y = y[:, None] + s * loc[..., 0] + c * loc[..., 1]
            return self.vue.vers_case(X, Y)

        LL = np.stack([L, l], axis=1)[:, None, :]
        # interieur, retreci de 1.5 mm : les bords sont lisses
        marge = np.minimum(1.5, 0.2 * np.stack([L, l], axis=1))[:, None, :]
        ci, li = placer(g.interieur[None] * (LL - 2 * marge))
        v_in = _echantillonner(pile, ci, li, fin)
        p_in, k_in, h_in, x_in, occ_in, sup_in, r_in = (v_in[..., k] for k in range(7))
        F = (p_in * (1 - x_in)).mean(axis=1)
        X = x_in.mean(axis=1)
        if base > 0:
            # piece empilee : les rampes du lissage ne disent rien de sa
            # face superieure (voir indices)
            k_in = k_in * (1.0 - r_in)
        accord = np.exp(-0.5 * ((h_in - h[:, None]) / self.tol_h) ** 2) * k_in
        # PLANEITE : une face superieure est plate ; la face laterale lissee
        # d'une piece voisine forme une rampe. Pente a travers la largeur
        # par moindres carres ponderes : variation de hauteur d'un bord a
        # l'autre, en mm
        t = g.interieur[None, :, 1]
        sk = np.maximum(k_in.sum(axis=1, keepdims=True), 1e-6)
        tm = (k_in * t).sum(axis=1, keepdims=True) / sk
        hm = (k_in * h_in).sum(axis=1, keepdims=True) / sk
        var_t = (k_in * (t - tm) ** 2).sum(axis=1)
        pente = np.abs((k_in * (t - tm) * (h_in - hm)).sum(axis=1)
                       / np.maximum(var_t, 1e-6))
        pente = np.where(var_t > 0.02 * k_in.shape[1] * 0.08, pente, 0.0)
        nk = k_in.sum(axis=1)
        Hf = np.where(nk > 0.2 * k_in.shape[1], accord.sum(axis=1) / np.maximum(nk, 1e-6), 0.3)

        # bords : chanfrein oriente
        cb, lb = placer(g.bord[None] * LL)
        phi = np.mod(g.normale[None] + th[:, None], np.pi)
        classe = np.floor(phi / (np.pi / N_ORIENT)).astype(int) % N_ORIENT
        frac = phi / (np.pi / N_ORIENT) - classe
        voisine = np.where(frac < 0.5, classe - 1, classe + 1) % N_ORIENT
        ny, nx = ind.H.shape
        cbi = np.clip(np.round(cb).astype(np.int64), 0, nx - 1)
        lbi = np.clip(np.round(lb).astype(np.int64), 0, ny - 1)
        d = np.minimum(ind.dt[classe, lbi, cbi], ind.dt[voisine, lbi, cbi])
        hors = (cb < 0) | (cb > nx - 1) | (lb < 0) | (lb > ny - 1)
        d = np.where(hors, 99.0, d)
        e = np.exp(-0.5 * (d / self.sigma_e) ** 2)
        tx = ind.textures
        if tx is not None and self.poids_texture > 0 and N:
            # contours de texture redresses au plan de la face (la hauteur
            # du modele) : un peu moins surs que les sauts de hauteur. Pas
            # de niveau assez proche (piece empilee, niveau pas encore
            # calcule) : pas de texture
            k = int(np.argmin(np.abs(tx.niveaux - h[0])))
            if abs(tx.niveaux[k] - h[0]) <= 1.5:
                d_t = np.minimum(tx.dt[k][classe, lbi, cbi],
                                 tx.dt[k][voisine, lbi, cbi]).astype(np.float32) / 4.0
                d_t = np.where(hors, 99.0, d_t)
                e = np.maximum(e, self.poids_texture * np.exp(-0.5 * (d_t / self.sigma_e) ** 2))
        # bord dans l'inconnu, ou cache par une piece posee dessus : ni
        # appui, ni absence d'appui
        d_inc = ind.d_inconnu[lbi, cbi]
        e = np.where(d_inc < 2.0, np.maximum(e, 0.5), e)
        if self._couches:
            occ_b = pile[lbi, cbi, 4]
            e = np.where(occ_b > 0.5, np.maximum(e, 0.5), e)
        # grands cotes et bouts comptent pour moitie chacun : les grands
        # cotes, plus longs, ne doivent pas masquer des bouts sans appui
        lg = g.bord_long
        E = 0.5 * e[:, lg].mean(axis=1) + 0.5 * e[:, ~lg].mean(axis=1)

        # anneau : la meme surface continue-t-elle au-dela ? (au-dessus de
        # la mi-hauteur de la piece elle-meme, comptee depuis sa base)
        ca_, la_ = placer(g.anneau[None] * LL + g.anneau_dir[None] * self.decal)
        v_a = _echantillonner(pile, ca_, la_, fin)
        k_a, h_a, x_a = v_a[..., 1], v_a[..., 2], v_a[..., 3]
        meme = 1.0 / (1.0 + np.exp(-(h_a - (h - 0.4 * hp)[:, None]) / 1.0))
        o = meme * k_a * (1 - x_a)
        # moyenne par COTE : une crete plus longue que la piece deborde par
        # ses deux bouts, peu de points de l'anneau mais deux cotes sur quatre
        O = np.mean([o[:, (g.anneau_dir == d_).all(axis=1)].mean(axis=1)
                     for d_ in ((0, -1), (0, 1), (-1, 0), (1, 0))], axis=0)

        wE, wF, wH, wO = self.poids
        S = wE * E + wF * F + wH * Hf + wO * (1 - O) - 1.0 * X
        # une surface qui deborde de plus de la moitie des cotes n'est pas
        # une piece (le dessus d'un ordinateur portable, une boite) ; deux
        # pieces ne se chevauchent pas
        S = S - 0.5 * np.maximum(O - 0.5, 0.0) - (X > 0.2)
        S = S - self.poids_pente * np.clip((pente - 3.0) / 5.0, 0.0, 1.0)
        if ind.hors_table is not None:
            # la partie du rectangle hors de la table est une faute franche
            S = S - _echantillonner(ind.hors_table, ci, li, False, 1.0).mean(axis=1)
        # l'inconnu est neutre, mais une piece doit montrer au moins le
        # quart de sa face superieure, mesuree a peu pres a la bonne
        # hauteur : sinon un rectangle pose dans un trou de mesure, ou a cote
        # d'une piece dont la face laterale est bien visible en infrarouge,
        # obtiendrait un score moyen sans aucune preuve. Une piece CACHEE en
        # partie par une piece posee dessus montre le quart de sa partie
        # visible (et au moins 6 % de sa face)
        bonne_h = np.abs(h_in - h[:, None]) < tolerance_hauteur(hp, base)[:, None]
        vu = (k_in * (p_in > 0.5) * bonne_h).mean(axis=1)
        cachee = occ_in.mean(axis=1)
        vu = vu / np.maximum(1.0 - cachee, 0.25)
        S = S * np.clip(vu / 0.25, 0.0, 1.0)
        appui = np.ones(N)
        if base > 0:
            # APPUI : une piece empilee doit reposer sur quelque chose de
            # visible a la hauteur de sa base : une piece deja retenue sous
            # elle (support), ou la porteuse qui depasse autour d'elle (un
            # second anneau, a 6 mm, a la hauteur de la base). Sans cela,
            # une piece couchee sur la table, un peu plus haute ou plus
            # basse que prevu, s'expliquerait aussi par une pile fantome.
            # Et un leger prix (rasoir d'Occam) : a egalite, la table.
            ca2, la2 = placer(g.anneau[None] * LL + g.anneau_dir[None] * 6.0)
            v2 = _echantillonner(pile, ca2, la2, fin)
            k2, h2, x2, r2 = v2[..., 1], v2[..., 2], v2[..., 3], v2[..., 6]
            # (pas sur une rampe : le bord lisse de toute piece passe par
            # la hauteur de la base)
            pres = (np.abs(h2 - base) < max(2.5, 0.15 * base)) * k2 * (1 - x2) * (1 - r2)
            dessous = np.abs(sup_in).mean(axis=1)
            appui = np.clip(np.maximum(pres.mean(axis=1) / 0.08, dessous / 0.10), 0.0, 1.0)
            appui = np.maximum(appui, appui_min)
            # et 0.02 par piece cachee dessous : a preuve egale, la pile la
            # plus simple (une epaisse a plat sur une epaisse, 17 + 17,
            # plutot qu'une epaisse sur la tranche sur deux fines, 8 + 8 + 19)
            S = S * (0.4 + 0.6 * appui) - 0.02 * self.etages.get(round(base, 1), 1)
        if detail:
            h_med = np.array([np.median(hi[ki > 0.5]) if (ki > 0.5).any() else np.nan
                              for hi, ki in zip(h_in, k_in)])
            # surface (mm^2) vue PLATE a la bonne hauteur (hors rampes)
            aire = (L - 2 * marge[:, 0, 0]) * (l - 2 * marge[:, 0, 1])
            plat = (k_in * (1 - r_in) * (p_in > 0.5) * bonne_h).mean(axis=1) * aire
            return S, dict(E=E, F=F, H=Hf, O=O, X=X, pente=pente, h_mes=h_med,
                           cachee=cachee, appui=appui, plat=plat)
        return S

    def affiner_m(self, x, y, th, m: Modele, **kw):
        """affiner pour un modele (dimensions, hauteur et base)."""
        return self.affiner(x, y, th, m.L, m.l, m.haut, base=m.base, h_piece=m.h, **kw)

    def affiner(self, x, y, th, L, l, h, iterations: int = 14,
                pas=(1.0, 1.0, np.radians(3.0)), taille: bool = False,
                **kw):
        """
        Recherche locale (motifs de Hooke et Jeeves) sur le score : on
        essaie +/- un pas sur chaque parametre, on garde le meilleur, on
        divise les pas par deux quand plus rien ne progresse. Le score est
        lu en bilineaire : il varie continument, la pose trouvee est a une
        fraction de case pres. Avec `taille`, L et l sont libres aussi (la
        MESURE des dimensions, pour l'ecart affiche). `kw` : passe a noter
        (base, h_piece, appui_min).
        """
        etat = np.stack([x, y, th, np.broadcast_to(L, len(x)).astype(float),
                         np.broadcast_to(l, len(x)).astype(float)], axis=1).astype(float)
        h = np.broadcast_to(np.asarray(h, float), (len(x),))
        pas_ = np.tile(np.array(list(pas) + [1.0, 1.0]), (len(x), 1))
        nparam = 5 if taille else 3
        score = self.noter(*etat.T[:5], h, **kw)
        for _ in range(iterations):
            essais = [etat]
            for k in range(nparam):
                for sg in (-1, 1):
                    e = etat.copy()
                    e[:, k] += sg * pas_[:, k]
                    essais.append(e)
            tout = np.concatenate(essais)
            sc = self.noter(*tout.T[:5], np.tile(h, len(essais)), **kw).reshape(len(essais), -1)
            meilleur = sc.argmax(axis=0)
            ameliore = sc[meilleur, np.arange(len(x))] > score + 1e-6
            idx = np.arange(len(x))
            etat = np.where(ameliore[:, None],
                            np.stack(essais)[meilleur, idx], etat)
            score = np.maximum(score, sc[meilleur, idx])
            pas_[~ameliore] *= 0.5
        return etat, score

    def reexpliquer(self, choisis) -> None:
        """L'explication refaite a partir d'une liste (score, x, y, th, modele)."""
        self._explique = np.zeros_like(self._explique)
        self._couches = []
        self._piles = {}
        for _, x, y, th, m in choisis:
            self.expliquer(x, y, th, m.L, m.l, m.base, m.haut)

    @staticmethod
    def _coins(x, y, th, L, l):
        coins = np.array([[-L, -l], [L, -l], [L, l], [-L, l]]) / 2.0
        c, s = np.cos(th), np.sin(th)
        return x + coins[:, 0] * c - coins[:, 1] * s, y + coins[:, 0] * s + coins[:, 1] * c

    def _dessiner(self, carte, x, y, th, L, l) -> None:
        X, Y = self._coins(x, y, th, L, l)
        cc, ll = self.vue.vers_case(X, Y)
        poly = np.round(np.stack([cc, ll], axis=1) * 16).astype(np.int32)
        cv2.fillPoly(carte, [poly], 1.0, lineType=cv2.LINE_AA, shift=4)

    def expliquer(self, x, y, th, L, l, base: float = 0.0,
                  haut: Optional[float] = None) -> None:
        """Marque les cases du rectangle comme expliquees, par une piece
        posee a `base`, de face superieure a `haut` (mm ; None : conflit
        avec toute autre piece)."""
        self._dessiner(self._explique, x, y, th, L, l)
        self._couches.append((x, y, th, L, l, base, np.inf if haut is None else haut, None))
        self._piles = {}

    def en_conflit(self, m: Modele, couche_m: Modele) -> bool:
        """Une piece retenue de modele `couche_m` est-elle en conflit (meme
        tranche de hauteur) avec une hypothese de modele `m` ?"""
        return self.classe((0, 0, 0, 0, 0, couche_m.base, couche_m.haut, None),
                           m.base, m.haut) == 0


def hypotheses(vue: VueDessus, ind: Indices, mods: List[Modele],
               pas_centre: float = 2.0, rayon_segments: float = 45.0,
               angles_max: int = 8):
    """
    Hypotheses (x, y, angle, indice du modele). Centres : les cases de
    premier plan, sur une grille de `pas_centre` mm. Angles : ceux des
    segments droits proches (un segment peut etre un grand cote ou un bout,
    donc angle et angle + 90 deg), et l'orientation locale du tenseur de
    structure. Modeles : ceux dont la hauteur de face superieure (base
    comprise, pour une piece empilee) est compatible avec la hauteur mesuree
    au centre ; la ou rien n'est mesure, seulement les pieces sur la table.
    """
    res = vue.res
    step = max(1, int(round(pas_centre / res)))
    fg = (ind.p > 0.5)
    ny, nx = fg.shape
    lig, col = np.mgrid[step // 2:ny:step, step // 2:nx:step]
    garde = fg[lig, col]
    lig, col = lig[garde], col[garde]
    if len(lig) == 0:
        return np.zeros((0, 4))
    X = vue.x0 + (col + 0.5) * res
    Y = vue.y0 + (lig + 0.5) * res
    # hauteur au centre : mediane 5 x 5 des cases connues
    Hm = np.where(ind.connu > 0, ind.H, np.nan)
    pad = np.pad(Hm, 2, constant_values=np.nan)
    blocs = np.stack([pad[lig + i, col + j] for i in range(5) for j in range(5)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # bloc tout inconnu
        h_c = np.nanmedian(blocs, axis=0)

    seg = ind.segments
    if len(seg):
        mx = vue.x0 + ((seg[:, 0] + seg[:, 2]) / 2 + 0.5) * res
        my = vue.y0 + ((seg[:, 1] + seg[:, 3]) / 2 + 0.5) * res
        a_seg = np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])
        lg_seg = np.hypot(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0]) * res
    sortie = []
    o_loc = ind.orient[lig, col]
    for k in range(len(X)):
        angles = [o_loc[k], o_loc[k] + np.pi / 2]
        if len(seg):
            dd = np.hypot(mx - X[k], my - Y[k])
            proche = np.nonzero(dd < rayon_segments)[0]
            if len(proche):
                ordre = proche[np.argsort(-lg_seg[proche] / (1 + dd[proche] / 20))][:angles_max]
                for a in a_seg[ordre]:
                    angles += [a, a + np.pi / 2]
        a = np.mod(np.array(angles), np.pi)
        # dedoublonnage a 4 deg pres
        a = np.sort(a)
        garde_a = np.r_[True, np.diff(a) > np.radians(4)]
        a = a[garde_a]
        if len(a) > 1 and (a[-1] - a[0]) > np.pi - np.radians(4):
            a = a[:-1]
        for i_m, m in enumerate(mods):
            if np.isfinite(h_c[k]):
                if abs(h_c[k] - m.haut) > tolerance_hauteur(m.h, m.base):
                    continue
            elif m.base > 0:
                continue
            for aa in a:
                sortie.append((X[k], Y[k], aa, i_m))
    return np.array(sortie, float).reshape(-1, 4)


def fusionner(choisis: list, noteur: Noteur, mods: List[Modele],
              marge: float = 0.05) -> list:
    """
    PARCIMONIE. Une piece large peut etre "expliquee" par deux pieces
    etroites cote a cote (une epaisse de 19 mm par deux fines de 8 mm sur la
    tranche, a la meme hauteur de 17 mm) : la selection gloutonne prend la
    premiere fine, et la seconde explique le reste. Pour chaque paire de
    pieces paralleles et accolees, on essaie donc UN modele plus large qui
    les couvre toutes deux ; s'il obtient un score au moins egal a celui de
    la meilleure des deux, a `marge` pres, il les remplace : c'est le rasoir
    d'Occam, une seule piece est l'explication la plus simple. Deux fines
    sur la tranche accolees (16 mm de large, 17 de haut) et une epaisse sur
    la tranche (17 et 19) ne se distinguent presque pas avec ce capteur ;
    sans indice franc pour deux pieces, on en annonce une.
    """
    change = True
    while change:
        change = False
        for i in range(len(choisis)):
            for j in range(i + 1, len(choisis)):
                si, xi, yi, ti, mi = choisis[i]
                sj, xj, yj, tj, mj = choisis[j]
                if abs(mi.base - mj.base) > 1.0:
                    continue
                dth = abs((ti - tj + np.pi / 2) % np.pi - np.pi / 2)
                if dth > np.radians(10):
                    continue
                dx, dy = xj - xi, yj - yi
                u = dx * np.cos(ti) + dy * np.sin(ti)
                v = -dx * np.sin(ti) + dy * np.cos(ti)
                if abs(u) > 12 or abs(v) > (mi.l + mj.l) / 2 + 3:
                    continue
                autres = [c for k, c in enumerate(choisis) if k not in (i, j)]
                noteur.reexpliquer(autres)
                xm, ym = (xi + xj) / 2, (yi + yj) / 2
                meilleur = None
                for m in mods:
                    if m.base != 0 or m.l < max(mi.l, mj.l) + 2:
                        continue
                    m = m.sur(mi.base)
                    e, sc = noteur.affiner_m([xm], [ym], [ti], m)
                    if meilleur is None or sc[0] > meilleur[0]:
                        meilleur = (float(sc[0]), e[0, 0], e[0, 1], e[0, 2], m)
                if meilleur is not None and meilleur[0] > max(si, sj) - marge:
                    choisis = autres + [meilleur]
                    change = True
                    break
                noteur.reexpliquer(choisis)
            if change:
                break
    noteur.reexpliquer(choisis)
    return choisis


def ajouter_niveaux(scene: Scene, tab: Table, vue: VueDessus, ind: Indices,
                    hauteurs: Sequence[float], tol: float = 1.5) -> None:
    """Complete les contours de texture redresses (textures_redressees) aux
    hauteurs demandees qui n'ont pas encore de niveau a `tol` mm pres : ceux
    des faces superieures de pieces EMPILEES, calcules seulement s'il y a
    des candidates (chaque niveau coute une reprojection des images)."""
    tx = ind.textures
    if tx is None or not len(hauteurs):
        return
    manquants = sorted({round(float(h), 1) for h in hauteurs
                        if np.min(np.abs(tx.niveaux - h)) > tol})
    niveaux: List[float] = []
    for h in manquants:
        if not niveaux or h - niveaux[-1] > tol:
            niveaux.append(h)
    if not niveaux:
        return
    nouv = textures_redressees(scene, tab, vue, niveaux)
    if nouv is None:
        return
    tx.niveaux = np.concatenate([tx.niveaux, nouv.niveaux])
    tx.dt = np.concatenate([tx.dt, nouv.dt])


def _rectangle_mm(x, y, th, m: Modele) -> np.ndarray:
    coins = np.array([[-m.L, -m.l], [m.L, -m.l], [m.L, m.l], [-m.L, m.l]]) / 2.0
    c, s = np.cos(th), np.sin(th)
    return (coins @ np.array([[c, s], [-s, c]]) + [x, y]).astype(np.float32)


def aire_commune(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Aire de l'intersection de deux polygones CONVEXES (sommets dans le sens
    trigonometrique), par decoupage de Sutherland-Hodgman : P est coupe
    successivement par le demi-plan interieur de chaque cote de Q.
    (cv2.intersectConvexConvex rend 0 pour deux rectangles dont des cotes
    sont confondus, cas courant ici : une porteuse deduite a la pose de la
    piece du dessus.)
    """
    poly = [tuple(p) for p in np.asarray(P, float)]
    Q = np.asarray(Q, float)
    for k in range(len(Q)):
        a, b = Q[k], Q[(k + 1) % len(Q)]
        if not poly:
            break

        def cote(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        sortie = []
        for i in range(len(poly)):
            p, q = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
            cp, cq = cote(p), cote(q)
            if cp >= 0:
                sortie.append(tuple(p))
            if (cp >= 0) != (cq >= 0):
                t = cp / (cp - cq)
                sortie.append(tuple(p + t * (q - p)))
        poly = sortie
    if len(poly) < 3:
        return 0.0
    x, y = np.array(poly).T
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def recouvrement(a, b) -> float:
    """Aire commune (mm^2) des empreintes de deux pieces (x, y, th, modele)."""
    return aire_commune(_rectangle_mm(*a), _rectangle_mm(*b))


def relier_piles(pieces: Sequence[tuple], tol: float = 0.6, part: float = 0.08) -> list:
    """
    La porteuse de chaque piece (x, y, th, modele) : None si elle est sur
    la table ; sinon l'indice de la piece dont le SOMMET est a la hauteur
    de sa base (hauteurs nominales, a `tol` mm pres) et qui recouvre le plus son empreinte (au
    moins `part` de sa surface : une piece en travers sur une autre la
    recouvre d'environ 28 %), a defaut en elargissant les deux empreintes
    de 5 mm ; -1 si aucune piece vue ne la porte.
    """
    sortie = []
    for i, a in enumerate(pieces):
        m = a[3]
        if m.base <= 0:
            sortie.append(None)
            continue
        meilleur = -1
        # puis, faute de mieux, avec les empreintes elargies de `marge` mm :
        # la pose d'une porteuse en grande partie cachee est moins sure
        for marge in (0.0, 5.0):
            aire_m = part * m.L * m.l
            a2 = (a[0], a[1], a[2], Modele(m.type, m.face, m.L + 2 * marge,
                                           m.l + 2 * marge, m.h, m.base))
            for j, b in enumerate(pieces):
                if j == i or abs(b[3].haut - m.base) > tol:
                    continue
                mb = b[3]
                b2 = (b[0], b[1], b[2], Modele(mb.type, mb.face, mb.L + 2 * marge,
                                               mb.l + 2 * marge, mb.h, mb.base))
                aire = recouvrement(a2, b2)
                if aire >= aire_m:
                    meilleur, aire_m = j, aire
            if meilleur != -1:
                break
        sortie.append(meilleur)
    return sortie


def coherence_piles(choisis: list, noteur: Noteur, mods: List[Modele],
                    seuil: float, departs: Sequence[tuple] = (),
                    tol: float = 0.6) -> list:
    """
    PAS DE PIECE FANTOME. Une piece retenue comme empilee doit avoir une
    porteuse parmi les pieces retenues (relier_piles). La preuve d'appui du
    noteur est locale : une piece couchee sur la table et touchant une
    piece plus basse (une fine a plat de 8 mm) peut l'avoir obtenue. On
    cherche d'abord la porteuse sous elle (porteuse_visible) : retenue si
    elle atteint la moitie du seuil (elle est en partie cachee). Une
    porteuse deja retenue doit DEPASSER (aire_degagee : 80 mm^2 et 15 % de
    sa face) quand la hauteur de la piece s'explique aussi sans pile (16 mm : fine sur fine, ou une
    epaisse de 17 mm mesuree un peu bas). Sinon,
    l'hypothese empilee est remplacee par la meilleure piece
    au meme endroit, sur la table ou sur le sommet d'une piece retenue qui
    la recouvre ; ou retiree si aucune n'atteint le seuil. Exception : une
    face superieure qu'aucune piece seule n'atteint (34 mm) ne s'explique
    que par une pile, dont la porteuse est alors cachee (voir
    chercher_porteuses).
    """
    h_table = [m.h for m in mods if m.base == 0]
    for _ in range(len(choisis) + 2):
        sup = relier_piles([c[1:] for c in choisis], tol)
        ambigu = [min(abs(c[4].haut - h) for h in h_table) < 4.0 for c in choisis]
        orph = [i for i, sp in enumerate(sup) if sp == -1 and ambigu[i]]
        # porteuse retenue, mais qui ne depasse pas vraiment (aire_degagee) :
        # la pile n'est pas prouvee, quand la hauteur s'explique aussi par
        # une seule piece sur la table
        faibles = [i for i, sp in enumerate(sup) if sp is not None and sp != -1
                   and ambigu[i] and aire_degagee(noteur, choisis[sp][1:], choisis[i][1:])
                   < max(80.0, 0.15 * choisis[sp][4].L * choisis[sp][4].l)]
        if not orph and not faibles:
            break
        faible = None
        if orph:
            i = min(orph, key=lambda k: choisis[k][0])
        else:
            i = faibles[0]
            faible = choisis[sup[i]]
        s, x, y, th, m = choisis[i]
        noteur.reexpliquer(choisis)
        if faible is None:
            porteuse = porteuse_visible(choisis[i], noteur, mods, departs)
            if porteuse is not None and porteuse[0] >= 0.5 * seuil:
                choisis = choisis + [porteuse]
                continue
        autres = choisis[:i] + choisis[i + 1:]
        noteur.reexpliquer(autres)
        bases = {0.0} | {round(c[4].haut, 1) for c in autres if c is not faible
                         and recouvrement((x, y, th, m), c[1:]) > 0.08 * m.L * m.l}
        meilleur = None
        for b in sorted(bases):
            for m0 in mods:
                if m0.base != 0 or (b > 0 and m0.face == "debout"):
                    continue
                mm = m0.sur(b) if b > 0 else m0
                if abs(mm.haut - m.haut) > max(5.0, 0.35 * mm.h):
                    continue
                e, sc = noteur.affiner_m([x], [y], [th], mm)
                if meilleur is None or sc[0] > meilleur[0]:
                    meilleur = (float(sc[0]), e[0, 0], e[0, 1], e[0, 2], mm)
        choisis = autres
        if meilleur is not None and meilleur[0] >= seuil:
            choisis = autres + [meilleur]
        if faible is not None:
            # l'ancienne porteuse, sous la piece qui ne repose plus sur elle :
            # gardee seulement si elle tient encore (moitie du seuil)
            reste = [c for c in choisis if c is not faible]
            noteur.reexpliquer(reste)
            if noteur.noter_m([faible[1]], [faible[2]], [faible[3]], faible[4])[0] < 0.5 * seuil:
                choisis = reste
        noteur.reexpliquer(choisis)
    return choisis


def _candidats_porteuse(mods: List[Modele], m: Modele) -> List[Modele]:
    """Les modeles couches dont le sommet est a la base de `m`, poses sur une
    base possible (la table, ou le sommet d'une pile)."""
    bases = bases_de_pile(mods)
    return [m0.sur(m.base - m0.h) if m.base - m0.h > 0.5 else m0 for m0 in mods
            if m0.base == 0 and m0.face != "debout"
            and min(abs(m.base - m0.h - b) for b in bases) < 1.0]


def aire_degagee(noteur: Noteur, porteuse: tuple, dessus: tuple,
                 marge: float = 4.0, pas: float = 1.0) -> float:
    """
    Surface (mm^2) de la face superieure de la `porteuse` (x, y, th, modele)
    vue PLATE a sa hauteur, a plus de `marge` mm de l'empreinte de la piece
    du `dessus` : la partie qui depasse vraiment. Un simple lisere le long
    du flanc de la piece du dessus est ce que donne le lissage du bord
    (une rampe qui passe par la hauteur de la porteuse) ; une epaisse de
    17 mm se laisserait sinon decrire comme une fine posee sur une fine.
    """
    x, y, th, m = porteuse
    xd, yd, thd, md = dessus
    s = np.arange(-m.L / 2 + pas / 2, m.L / 2, pas)
    t = np.arange(-m.l / 2 + pas / 2, m.l / 2, pas)
    S, T = np.meshgrid(s, t)
    c, sn = np.cos(th), np.sin(th)
    X = x + c * S - sn * T
    Y = y + sn * S + c * T
    # coordonnees dans le repere de la piece du dessus
    cd, sd = np.cos(thd), np.sin(thd)
    u = (X - xd) * cd + (Y - yd) * sd
    v = -(X - xd) * sd + (Y - yd) * cd
    dehors = (np.abs(u) > md.L / 2 + marge) | (np.abs(v) > md.l / 2 + marge)
    ind = noteur.ind
    cc, ll = noteur.vue.vers_case(X, Y)
    pile = np.dstack([ind.connu, ind.H, ind.rampe if ind.rampe is not None
                      else np.zeros_like(ind.H)]).astype(np.float32)
    v_ = _echantillonner(pile, cc.astype(np.float32), ll.astype(np.float32), False)
    k, h, r = v_[..., 0], v_[..., 1], v_[..., 2]
    # a 2.5 mm pres : la porteuse est une face plate a une hauteur connue,
    # pas la pente douce qui borde une piece du cote de son ombre
    bon = dehors & (k > 0.5) & (r < 0.5) & (np.abs(h - m.haut) < 2.5)
    return float(bon.sum() * pas * pas)


def porteuse_visible(dessus: tuple, noteur: Noteur, mods: List[Modele],
                     departs: Sequence[tuple]) -> Optional[tuple]:
    """
    La meilleure porteuse (score, x, y, th, modele) sous la piece `dessus`
    (score, x, y, th, modele), le reste de la scene etant explique tel
    quel, dont au moins 80 mm^2 de face superieure sont vus PLATS a la
    bonne hauteur et DEGAGES de la piece du dessus (aire_degagee) : une
    "porteuse" faite des seules rampes du lissage autour de la piece du
    dessus n'en est pas une. Departs : les hypotheses affinees (x, y, th, modele) de chaque
    modele candidat qui recouvrent la piece du dessus, et une grille de
    poses autour d'elle (dans son axe ou en travers, glissees jusqu'a
    30 mm : posee sous la piece du dessus, une porteuse cachee n'offre
    aucune pente au score, il faut partir pres de sa partie visible).
    """
    s, x, y, th, m = dessus
    grille = []
    for a in (th, th + np.pi / 2):
        ax = np.array([np.cos(a), np.sin(a)])
        no = np.array([-ax[1], ax[0]])
        for u in (-30, -20, -10, 0, 10, 20, 30):
            for v in (-8, 0, 8):
                q = np.array([x, y]) + u * ax + v * no
                grille.append((q[0], q[1], a))
    meilleur = None
    plat_min = 80.0                 # mm^2 de face vue plate : 8 x 10 mm
    for mc in _candidats_porteuse(mods, m):
        essais = grille + [d[:3] for d in departs if d[3] == mc
                           and recouvrement((x, y, th, m), d) > 0.08 * m.L * m.l][:40]
        e0 = np.array(essais, float)
        e, sc = noteur.affiner_m(e0[:, 0], e0[:, 1], e0[:, 2], mc)
        for k in np.argsort(-sc):
            if recouvrement((x, y, th, m), (e[k, 0], e[k, 1], e[k, 2], mc)) \
                    < 0.08 * m.L * m.l:
                continue
            if meilleur is None or sc[k] > meilleur[0]:
                if aire_degagee(noteur, (e[k, 0], e[k, 1], e[k, 2], mc),
                                (x, y, th, m)) >= plat_min:
                    meilleur = (float(sc[k]), e[k, 0], e[k, 1], e[k, 2], mc)
            break
    return meilleur


def chercher_porteuses(choisis: list, noteur: Noteur, mods: List[Modele],
                       departs: Sequence[tuple], seuil: float,
                       tol: float = 0.6) -> Tuple[list, set]:
    """
    Une piece empilee sans porteuse retenue (relier_piles) : sa porteuse
    existe forcement, mais elle est en grande partie CACHEE (par la piece du
    dessus, et par l'ombre que celle-ci projette), donc souvent sous le
    seuil de la selection gloutonne. On la cherche la ou elle doit etre
    (porteuse_visible), la partie cachee etant neutre. Si la meilleure
    atteint la moitie du seuil, on la garde (son score dit qu'elle est peu
    sure) ; sinon la porteuse est entierement cachee (meme sens,
    exactement dessous) : on la DEDUIT, a la pose de la piece du dessus,
    avec une confiance faible. Rend la liste completee et l'ensemble des
    indices des pieces deduites.
    """
    deduites: set = set()
    renonce: set = set()
    for _ in range(len(choisis)):
        sup = relier_piles([c[1:] for c in choisis], tol)
        # (pas de deduction en chaine : une porteuse deduite posee plus haut
        # que la table reste sur un objet non identifie)
        orph = [i for i, sp in enumerate(sup) if sp == -1 and i not in renonce
                and i not in deduites]
        if not orph:
            break
        i = orph[0]
        s, x, y, th, m = choisis[i]
        noteur.reexpliquer(choisis)
        meilleur = porteuse_visible(choisis[i], noteur, mods, departs)
        if meilleur is not None and meilleur[0] >= 0.5 * seuil:
            choisis = choisis + [meilleur]
            continue
        # entierement cachee : deduite, sous la piece du dessus ; la plus
        # basse possible, et la plus large qui reste cachee sous elle
        cands = sorted(_candidats_porteuse(mods, m), key=lambda c: (
            c.base, c.l > m.l + 2.0, -c.l if c.l <= m.l + 2.0 else c.l))
        if not cands:
            renonce.add(i)                     # rien ne convient : on renonce
            continue
        choisis = choisis + [(0.3 * s, x, y, th, cands[0])]
        deduites.add(len(choisis) - 1)
    noteur.reexpliquer(choisis)
    return choisis, deduites


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class Preparation:
    """Etapes 1 a 3, avant la recherche des rectangles."""
    tab: Table
    vue: VueDessus
    ind: Indices
    noteur: "Noteur"
    mods: List[Modele]
    hauteur_min: float            # m


def preparer(scene: Scene, dimensions=(16, 30, 68), res: float = 1.0,
             hauteur_min: Optional[float] = None, hauteur_max: float = 0.10,
             poids_couleur: float = 0.0, poids_ir: float = 0.0,
             debout: bool = True, volants: bool = True,
             continuite: float = 4.0, etirement: float = 1.5,
             poids_texture: float = 0.8, poids_pente: float = 0.2) -> Preparation:
    """Table, vue redressee, indices et noteur (voir detecter)."""
    types = (dict(dimensions) if isinstance(dimensions, dict)
             else {"": tuple(dimensions)})
    mods = modeles(types, debout)
    mods = [m for m in mods if m.h / 1000.0 < hauteur_max]
    tab = analyser_table(scene, hauteur_max)

    # seuil de hauteur : comme vision.detecter
    plus_petite = min(min(d) for d in types.values())
    if hauteur_min is None:
        hauteur_min = float(np.clip(4.0 * tab.bruit, 0.003, 0.006))
        hauteur_min = min(hauteur_min, 0.5 * plus_petite / 1000.0)
    garde = (retirer_volants(tab, 1.5 * hauteur_min, etirement=etirement)
             if volants else tab.valides)
    vue = vue_dessus(scene, tab, tab.valides, res, hauteur_min, hauteur_max,
                     continuite=continuite, douteux=tab.valides & ~garde)
    # bruit de hauteur DANS la vue (moyennes par case) : pour les tolerances
    h_tab = vue.hauteur[np.isfinite(vue.hauteur) & (np.abs(vue.hauteur) < 5)]
    sigma_h = (1.4826 * float(np.median(np.abs(h_tab - np.median(h_tab))))
               if h_tab.size else 1.0)
    ind = indices(vue, hauteur_min * 1000.0, sigma_h, poids_couleur, poids_ir)
    ind.hors_table = hors_table(tab, vue, scene.K)
    if poids_texture > 0:
        niveaux = sorted({round(m.h, 1) for m in mods})
        ind.textures = textures_redressees(scene, tab, vue, niveaux)
        if ind.textures is not None and len(ind.textures.segments):
            ind.segments = np.concatenate([ind.segments, ind.textures.segments])
    return Preparation(tab, vue, ind,
                       Noteur(vue, ind, poids_texture=poids_texture,
                              poids_pente=poids_pente),
                       mods, hauteur_min)


def detecter(scene: Scene, dimensions=(16, 30, 68), res: float = 1.0,
             seuil_score: float = 0.6, hauteur_min: Optional[float] = None,
             hauteur_max: float = 0.10, poids_couleur: float = 0.0,
             poids_ir: float = 0.0, debout: bool = True,
             tolerance: float = 10.0, max_pieces: int = 40,
             volants: bool = True, continuite: float = 4.0,
             etirement: float = 1.5, fusion: bool = True,
             poids_texture: float = 0.8, poids_pente: float = 0.2,
             empilement: bool = True, etages: int = 2,
             max_hypotheses: int = 250,
             trace: Optional[dict] = None) -> ResultatModele:
    """
    Pieces de dimensions connues (mm), par modele. Memes arguments
    principaux que vision.detecter (`dimensions` : un type (a, b, c) ou un
    dictionnaire de types), et meme resultat, plus `score` par piece et la
    vue redressee (`vue`, `rectangles`, pour dessiner_vue).

    `seuil_score` : score minimal d'une piece retenue (0 a 1).
    `res` : taille d'une case de la vue redressee, mm.
    `poids_couleur`, `poids_ir` : poids de la couleur et de la brillance
    infrarouge dans le premier plan (la hauteur a le poids 1).
    `poids_texture` : poids des contours infrarouge et couleur redresses.
    `volants` : marquer les pixels volants (sinon tous comptent).
    `fusion` : test de parcimonie sur les paires accolees.
    `tolerance` : comme vision.detecter ; ici un rectangle n'est ecarte
    qu'au-dela du double, car ses dimensions sont celles du modele.
    `empilement` : chercher aussi des pieces posees sur d'autres, en piles
    de `etages` pieces au plus sous la piece cherchee (voir l'en-tete).
    `max_hypotheses` : hypotheses gardees apres la notation grossiere.

    Chaque piece porte `support` (numero de sa porteuse, None : la table),
    `hauteur_base` (mm), `dessus` (numeros des pieces posees sur elle) et
    `cachee` (part de sa face superieure cachee) ; ordre_de_prise(pieces)
    dit dans quel ordre les saisir.
    """
    prep = preparer(scene, dimensions, res, hauteur_min, hauteur_max,
                    poids_couleur, poids_ir, debout, volants, continuite,
                    etirement, poids_texture, poids_pente)
    tab, vue, ind, noteur, mods = prep.tab, prep.vue, prep.ind, prep.noteur, prep.mods

    # --- modeles : chaque type sur chaque face, sur la table, et (pieces
    # couchees) sur le sommet d'une pile de 1 a `etages` pieces
    mods_tous = list(mods)
    if empilement:
        mods_tous += modeles_empiles(mods, bases_de_pile(mods, etages),
                                     hauteur_max * 1000.0)
        noteur.etages = etages_sous(mods, etages)
    # une face superieure a une hauteur qu'aucune piece couchee sur la
    # table n'atteint (34 mm : deux epaisses a plat) ne s'explique QUE par
    # une pile : la porteuse peut alors etre entierement cachee, on lui
    # accorde d'office une bonne part de la preuve d'appui
    h_table = [m.h for m in mods]
    appui_min = [0.0 if m.base == 0 or min(abs(m.haut - h) for h in h_table) < 4.0
                 else 0.75 for m in mods_tous]

    # --- hypotheses, notees grossierement (plus proche voisin, pas de 3 mm)
    hyp = hypotheses(vue, ind, mods_tous)
    cands = []
    for i_m, m in enumerate(mods_tous):
        sel = hyp[hyp[:, 3] == i_m]
        if not len(sel):
            continue
        sc = noteur.noter_m(sel[:, 0], sel[:, 1], sel[:, 2], m, pas=3.0, fin=False,
                            appui_min=appui_min[i_m])
        bon = sc > 0.6 * seuil_score
        for (x, y, a, _), s in zip(sel[bon], sc[bon]):
            cands.append((s, x, y, a, i_m))
    cands.sort(key=lambda t: -t[0])
    # suppression des non-maxima : a moins de 4 mm et 12 deg d'une
    # meilleure hypothese du meme modele, une hypothese est redondante
    # Deux quotas : pieces sur la table (`max_hypotheses`, comme sans
    # empilement) et pieces empilees (la moitie en plus) ; sinon les
    # variantes empilees, nombreuses, chassent les porteuses
    retenus: List[Tuple] = []
    quota = {False: max_hypotheses, True: max_hypotheses // 2}
    for cnd in cands:
        s, x, y, a, i_m = cnd
        empilee = mods_tous[i_m].base > 0
        if quota[empilee] <= 0:
            if quota[not empilee] <= 0:
                break
            continue
        doublon = False
        for r in retenus:
            if r[4] == i_m and np.hypot(r[1] - x, r[2] - y) < 4.0:
                da = abs((r[3] - a + np.pi / 2) % np.pi - np.pi / 2)
                if da < np.radians(12):
                    doublon = True
                    break
        if not doublon:
            retenus.append(cnd)
            quota[empilee] -= 1
    # contours de texture aux hauteurs des pieces empilees candidates
    if ind.textures is not None and poids_texture > 0:
        ajouter_niveaux(scene, tab, vue, ind,
                        [mods_tous[r[4]].haut for r in retenus if mods_tous[r[4]].base > 0])

    # --- affinage de chaque hypothese retenue. On garde la pose de DEPART :
    # une hypothese nee sur une piece peut glisser, en s'affinant, vers sa
    # voisine plus nette ; si cette voisine est retenue, l'hypothese est
    # re-affinee depuis son depart, avec la voisine deja expliquee.
    depart = np.array([[r[1], r[2], r[3], r[4]] for r in retenus]).reshape(-1, 4)
    etat = depart[:, :3].copy()

    def noter(ix, i_m):
        return noteur.noter_m(etat[ix, 0], etat[ix, 1], etat[ix, 2], mods_tous[i_m],
                              appui_min=appui_min[i_m])

    def affiner_tout(idx, **kw):
        for i_m in np.unique(depart[idx, 3]).astype(int):
            sel = idx[depart[idx, 3] == i_m]
            e, _ = noteur.affiner_m(depart[sel, 0], depart[sel, 1], depart[sel, 2],
                                    mods_tous[i_m], appui_min=appui_min[i_m], **kw)
            etat[sel] = e[:, :3]

    actifs = np.arange(len(depart))
    affiner_tout(actifs)

    # --- selection gloutonne avec explication. Une piece retenue n'exclut
    # que les hypotheses de MEME tranche de hauteur : celle qu'elle recouvre
    # (posee plus bas, en partie cachee) ou qui la recouvre (posee dessus)
    # restent candidates, et sont renotees avec la partie cachee neutre.
    # L'ordre se regle seul : la piece du dessus, entiere, est en general
    # retenue la premiere, puis celle du dessous ; si celle du dessous
    # l'est d'abord, elle devient la preuve d'appui de celle du dessus.
    choisis = []
    while len(actifs) and len(choisis) < max_pieces:
        scores = np.empty(len(actifs))
        for i_m in np.unique(depart[actifs, 3]).astype(int):
            sel = depart[actifs, 3] == i_m
            scores[sel] = noter(actifs[sel], i_m)
        k = int(np.argmax(scores))
        if scores[k] < seuil_score:
            break
        j = actifs[k]
        m = mods_tous[int(depart[j, 3])]
        x, y, th = etat[j]
        choisis.append((float(scores[k]), x, y, th, m))
        avant = noteur.explique.copy()
        noteur.expliquer(x, y, th, m.L, m.l, m.base, m.haut)
        # les hypotheses NEES dans la piece retenue, et de la meme tranche
        # de hauteur, sont epuisees ; celles qui y ont seulement glisse
        # repartent de leur depart
        c, s_ = np.cos(th), np.sin(th)
        dx, dy = depart[actifs, 0] - x, depart[actifs, 1] - y
        u = np.abs(dx * c + dy * s_)
        v = np.abs(-dx * s_ + dy * c)
        conflit = np.array([noteur.en_conflit(mods_tous[int(depart[a, 3])], m)
                            for a in actifs])
        nee = (u < m.L / 2 - 2) & (v < m.l / 2 - 1) & conflit
        actifs = actifs[~nee & (actifs != j)]
        if not len(actifs):
            break
        cc, ll = noteur.vue.vers_case(etat[actifs, 0], etat[actifs, 1])
        nouveau = noteur.explique - avant
        touche = _echantillonner(cv2.dilate(nouveau, np.ones((41, 41), np.uint8)),
                                 cc, ll, False) > 0
        if touche.any():
            affiner_tout(actifs[touche])

    if trace is not None:               # pour le diagnostic
        trace.update(depart=depart, etat=etat, mods=mods_tous, glouton=list(choisis),
                     noteur=noteur, prep=prep)
    if fusion:
        choisis = fusionner(choisis, noteur, mods)
        choisis.sort(key=lambda c: -c[0])
    if empilement:
        departs = [(etat[k, 0], etat[k, 1], etat[k, 2], mods_tous[int(depart[k, 3])])
                   for k in range(len(depart))]
        choisis = coherence_piles(choisis, noteur, mods, seuil_score, departs)
        choisis.sort(key=lambda c: -c[0])
        choisis, deduites = chercher_porteuses(choisis, noteur, mods, departs,
                                               seuil_score)
    else:
        deduites = set()

    # --- sortie
    gardes = []                     # (score, x, y, th, m, mesures, ecart, cachee)
    autres = []
    for i_c, (s, x, y, th, m) in enumerate(choisis):
        if i_c in deduites:
            # porteuse deduite, entierement cachee : rien a mesurer
            gardes.append((s, x, y, th, m, (m.L, m.l, m.h), 0.0, 1.0))
            continue
        # mesures : les cases de la piece elle-meme ne doivent pas compter
        # comme deja expliquees ; celles des AUTRES pieces si (conflit,
        # piece posee dessus : la part cachee)
        sauve = noteur.sauver()
        noteur.reexpliquer([c for c in choisis if (c[1], c[2]) != (x, y)])
        _, det = noteur.noter_m([x], [y], [th], m, detail=True)
        h_mes = float(det["h_mes"][0])
        cachee = float(det["cachee"][0])
        # mesure des dimensions : L et l libres
        e, _ = noteur.affiner_m([x], [y], [th], m, iterations=10,
                                pas=(0.5, 0.5, np.radians(1.0)), taille=True)
        noteur.restaurer(sauve)
        L_mes, l_mes = float(e[0, 3]), float(e[0, 4])
        if not np.isfinite(h_mes):
            h_mes = m.haut
        h_mes -= m.base             # epaisseur mesuree de la piece
        ecart = float(np.sqrt((L_mes - m.L) ** 2 + (l_mes - m.l) ** 2 + (h_mes - m.h) ** 2))
        if ecart > 2 * tolerance:
            autres.append("rectangle {:.0f} x {:.0f} mm, {:.0f} mm de haut ecarte "
                          "(score {:.2f}, ecart {:.0f} mm)".format(L_mes, l_mes, h_mes, s, ecart))
            continue
        # une piece en partie cachee est moins sure : on le dit
        gardes.append((s * (1.0 - 0.5 * cachee), x, y, th, m,
                       (L_mes, l_mes, h_mes), ecart, cachee))

    # piles : porteuse de chaque piece (-1 : posee sur un objet non vu)
    supports = relier_piles([(g[1], g[2], g[3], g[4]) for g in gardes])
    supports = [None if sp == -1 else sp for sp in supports]

    pieces: List[PieceModele] = []
    rects = []
    for i, (s, x, y, th, m, dims_mes, ecart, cachee) in enumerate(gardes):
        th_n = (th + np.pi / 2) % np.pi - np.pi / 2
        T_t = np.eye(4)
        c, s_ = np.cos(th_n), np.sin(th_n)
        T_t[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
        # la hauteur du MODELE : la mesure a servi a le choisir, et elle est
        # biaisee par le lissage sur une piece etroite
        T_t[:3, 3] = [x / 1000.0, y / 1000.0, m.haut / 1000.0]
        T_c = tab.T_ct @ T_t
        coins = np.array([[-m.L, -m.l], [m.L, -m.l], [m.L, m.l], [-m.L, m.l]]) / 2.0
        R2 = np.array([[c, -s_], [s_, c]])
        xy = (coins @ R2.T + [x, y]) / 1000.0
        coins3 = np.c_[xy, np.full(4, m.haut / 1000.0), np.ones(4)]
        cc = (tab.T_ct @ coins3.T)[:3]
        uv = (scene.K @ (cc / cc[2]))[:2].T
        sup = supports[i]
        pieces.append(PieceModele(i + 1, m.face, dims_mes, ecart, T_c, T_t,
                                  uv.astype(np.int32), 0, m.type,
                                  support=None if sup is None else sup + 1,
                                  hauteur_base=float(m.base),
                                  dessus=[k + 1 for k, sk in enumerate(supports) if sk == i],
                                  score=float(np.clip(s, 0, 1)), cachee=cachee))
        rects.append((x, y, th_n, m.L, m.l))

    # zones sans mesure, comme vision.detecter
    trous = []
    vide = ((~tab.valides) & (tab.zone > 0)).astype(np.uint8)
    vide = cv2.morphologyEx(vide, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nb, lab, st, _ = cv2.connectedComponentsWithStats(vide)
    for i in range(1, nb):
        x, y, w, hh, aire = st[i]
        if 300 <= aire <= 20000 and x > 0 and y > 0 \
                and x + w < tab.D.shape[1] and y + hh < tab.D.shape[0]:
            trous.append((int(x), int(y), int(w), int(hh)))
    hauteur_img = tab.hauteur
    return ResultatModele(pieces, tab.T_ct, tab.n, tab.d, hauteur_img, trous,
                          autres, vue=vue, rectangles=rects)


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def dessiner(scene: Scene, res: Resultat) -> np.ndarray:
    """Image annotee (comme vision.dessiner), avec le score de chaque piece."""
    img = vision.dessiner(scene, res)
    for p in res.pieces:
        if hasattr(p, "score"):
            u, v = p.contour.mean(axis=0).astype(int)
            texte = "score {:.2f}".format(p.score)
            if getattr(p, "support", None) is not None:
                texte += " sur {}".format(p.support)
            elif getattr(p, "hauteur_base", 0.0) > 0:
                texte += " sur ?"
            cv2.putText(img, texte, (u - 40, v + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 255), 1,
                        cv2.LINE_AA)
    return img


def dessiner_vue(res: ResultatModele, h_max: float = 25.0) -> np.ndarray:
    """La vue redressee (hauteur en fausses couleurs, inconnu en noir), avec
    les rectangles retenus."""
    vue = res.vue
    H = np.nan_to_num(vue.hauteur, nan=0.0)
    img = cv2.applyColorMap(np.clip(H / h_max * 255, 0, 255).astype(np.uint8),
                            cv2.COLORMAP_VIRIDIS)
    img[~vue.connue] = 0
    for p, (x, y, th, L, l) in zip(res.pieces, res.rectangles):
        coins = np.array([[-L, -l], [L, -l], [L, l], [-L, l]]) / 2.0
        c, s = np.cos(th), np.sin(th)
        X = x + coins[:, 0] * c - coins[:, 1] * s
        Y = y + coins[:, 0] * s + coins[:, 1] * c
        cc, ll = vue.vers_case(X, Y)
        pts = np.round(np.stack([cc, ll], axis=1) + 0.5).astype(np.int32)
        cv2.polylines(img, [pts], True, (60, 60, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "{} {:.2f}".format(p.numero, getattr(p, "score", 0)),
                    (int(cc.mean()) - 10, int(ll.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return img
