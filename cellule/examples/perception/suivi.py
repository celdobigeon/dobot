# -*- coding: utf-8 -*-
"""
Suivi temporel des pieces : d'une image a l'autre, la meme piece garde le
meme identifiant, sa pose est lissee et son type stabilise.

    from perception import Perception
    from suivi import Suivi
    perception, suivi = Perception(), Suivi()
    while True:
        pieces = suivi.mettre_a_jour(perception.detecter())
        for p in pieces:
            print(p.id_piste, p.type, p.face, p.position_table, p.lacet, p.confiance)
        for e in suivi.evenements:      # apparue, disparue, deplacee, ...
            print(e)

POURQUOI. Une detection isolee est bruitee (quelques mm, quelques degres),
se trompe parfois de type ou de face, manque parfois une piece (bras du
robot devant, reflet) et invente parfois un objet. Quand le robot travaille,
la scene change PEU d'une image a l'autre : en rapprochant chaque detection
de ce qu'on savait deja, on moyenne le bruit, on vote sur le type, on garde
une piece cachee quelques images et on ignore un faux positif isole.

PRINCIPE (suivi multi-objets classique : association + filtrage + cycle de
vie, comme SORT, mais dans le repere TABLE et pour des objets immobiles qui
bougent par SAUTS, quand le robot les deplace).

  1. ETAT d'une piste : x, y, z (mm) et lacet (deg) de la face superieure
     dans le repere table, chacun filtre par un petit filtre de Kalman
     scalaire a modele CONSTANT (une piece posee ne derive pas : pas de
     vitesse). Le lacet est periodique de 180 deg (un parallelepipede vu de
     dessus est le meme tourne d'un demi-tour) : les ecarts d'angle sont
     ramenes dans [-90, 90[. Si la face superieure est presque carree
     (piece debout : 19 x 17 mm), le lacet est mal defini : son bruit de
     mesure est fortement augmente et il ne sert ni a l'association ni a
     detecter un deplacement.
     Le type et la face sont VOTES : chaque detection ajoute sa confiance a
     une distribution cumulee (avec oubli) ; le type rendu ne change que si
     un autre le depasse nettement (hysteresis).
  2. ASSOCIATION detections <-> pistes : cout = distance 3D / portee + ecart
     d'angle (modulo 180) + incompatibilite de type (1 - probabilite du type
     detecte selon la piste) ; au-dela de la portee, association interdite.
     Affectation optimale par l'algorithme HONGROIS (scipy s'il est la,
     sinon implementation propre ci-dessous). Ce critere GLOBAL evite deja
     l'echange de deux pieces voisines (il faudrait que les deux mesures
     soient plus proches de la mauvaise piste) ; en plus, une association
     qui ferait SAUTER une piste (voir 3) n'est jamais prise sur une seule
     image : pieces qui se touchent, deux pieces fusionnees en une
     detection, faux positif a cote d'une piece cachee... dans le doute, on
     ne bouge rien.
  3. CYCLE DE VIE : une detection non associee cree une piste PROVISOIRE,
     confirmee apres `confirmations` detections (un faux positif isole
     n'est donc jamais rendu) ; une piste confirmee sans detection est
     maintenue `absences_max` mises a jour (le bras cache souvent la piece
     qu'il va prendre) avec une confiance decroissante, puis supprimee.
     Un saut de position (ou d'angle) au-dela du bruit : la piece a ete
     DEPLACEE, la piste repart de la nouvelle mesure, sans lissage. Par
     defaut (confirmer_sauts) le saut doit etre CONFIRME : la mesure
     eloignee ouvre une piste provisoire ; quand celle-ci est confirmee
     (image suivante) alors qu'une piste du meme type a disparu de pres de
     la (moins de portee_saut : la piece a pu etre emportee loin),
     l'identite lui est transmise : evenement DEPLACEE, une image plus
     tard, mais un faux positif isole ne fait jamais sauter une piece
     cachee par le bras, puisqu'il n'est jamais confirme. Avec
     confirmer_sauts=False, un saut sans autre piece a portee est pris tout
     de suite.
  4. EVENEMENTS (suivi.evenements, ceux de la derniere mise a jour) :
     apparue, disparue, deplacee, changee (le type ou la face vote a
     change : piece retournee ?), saisie_probable (une piece LIBRE -- rien
     dessus -- a disparu : le robot l'a prise ? ; immediat si le programme
     a annonce la saisie par suivi.signaler_saisie(id_piste)).

SORTIE : les memes objets que l'entree (copies : PieceVue de perception.py,
vision.Piece, ou tout objet portant T_table), avec la pose lissee (T_table,
et T_camera / T_robot recalcules de facon coherente), le type et la face
votes, et en plus :
    id_piste            identifiant stable (ne change pas d'une image a
                        l'autre, jamais reutilise) ; `numero` lui est egal,
                        et `support` / `dessus` sont traduits en
                        identifiants de piste : ordre_de_prise(pieces)
                        marche directement
    confiance           confiance de la PISTE (moyenne des detections,
                        confirmations, decroissance pendant une absence)
    confiance_detection celle de la derniere detection
    T_table_brute       la pose de cette detection, non lissee
    vue                 False : piece non detectee cette fois-ci, rendue de
                        memoire (absences : depuis combien de mises a jour)
    probabilites_type, probabilites_face : les votes normalises
`contour` et `masque` restent ceux de la derniere detection (image).

AUCUNE CADENCE n'est supposee : Perception peut prendre plusieurs secondes
par image. Le temps ne sert que si on le veut (bruit de processus par
seconde, duree d'absence maximale) ; sinon tout se compte en mises a jour.

Demonstration :
    .venv/bin/python examples/perception/suivi.py --direct      # serveur de camera
    .venv/bin/python examples/perception/suivi.py a.npz b.npz   # captures enregistrees
"""

from __future__ import annotations

import copy
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))

#: cout d'une association interdite (hors portee)
INTERDIT = 1e6


# ---------------------------------------------------------------------------
# Angles et affectation
# ---------------------------------------------------------------------------

def angle_180(a):
    """Angle (deg) ramene dans [-90, 90[ : l'orientation d'un rectangle est
    definie a 180 deg pres, donc 89 deg et -89 deg ne different que de 2."""
    return (np.asarray(a, float) + 90.0) % 180.0 - 90.0


def _hongrois_propre(C: np.ndarray) -> List[Tuple[int, int]]:
    """Affectation de cout minimal (algorithme hongrois avec potentiels,
    O(n^2 m)), matrice n x m avec n <= m. Rend [(ligne, colonne)]."""
    n, m = C.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, int)          # p[j] : ligne (1..n) affectee a la colonne j
    voie = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        mini = np.full(m + 1, np.inf)
        utilise = np.zeros(m + 1, bool)
        while True:
            utilise[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if not utilise[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < mini[j]:
                        mini[j], voie[j] = cur, j0
                    if mini[j] < delta:
                        delta, j1 = mini[j], j
            for j in range(m + 1):
                if utilise[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    mini[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = voie[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def hongrois(C: np.ndarray) -> List[Tuple[int, int]]:
    """Affectation de cout minimal d'une matrice rectangulaire quelconque :
    scipy.optimize.linear_sum_assignment s'il est installe, sinon
    l'implementation ci-dessus (transposee si plus de lignes que de
    colonnes)."""
    C = np.asarray(C, float)
    if C.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        lignes, colonnes = linear_sum_assignment(C)
        return [(int(i), int(j)) for i, j in zip(lignes, colonnes)]
    except ImportError:
        pass
    if C.shape[0] <= C.shape[1]:
        return _hongrois_propre(C)
    return [(i, j) for j, i in _hongrois_propre(C.T)]


def associer(C: np.ndarray, limite: float = INTERDIT / 2) -> List[Tuple[int, int]]:
    """Paires (ligne, colonne) de l'affectation optimale, sans celles dont
    le cout depasse `limite` (associations interdites)."""
    return [(i, j) for i, j in hongrois(C) if C[i, j] < limite]


# ---------------------------------------------------------------------------
# Lecture des detections (sans dependre des details de perception.py)
# ---------------------------------------------------------------------------

@dataclass
class Mesure:
    """Ce que le suivi lit d'une detection."""
    piece: object
    position: np.ndarray            # mm, repere table (centre de la face superieure)
    lacet: float                    # deg, dans [-90, 90[
    inclinaison: float              # deg, axe z de la piece / verticale
    type: str
    face: str
    confiance: float
    lacet_defini: bool              # face superieure assez allongee
    numero: Optional[int]


def mesure_de(piece, rapport_min: float = 1.25) -> Optional[Mesure]:
    """Lit une detection ; None si elle n'a pas de T_table exploitable."""
    T = getattr(piece, "T_table", None)
    if T is None:
        return None
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return None
    conf = getattr(piece, "confiance", None)
    if conf is None:
        conf = getattr(piece, "score", 1.0)
    conf = float(np.clip(conf if conf is not None and np.isfinite(conf) else 1.0, 0.0, 1.0))
    dims = getattr(piece, "dimensions", None)
    defini = True
    try:
        L, l = float(dims[0]), float(dims[1])
        defini = l <= 0 or L / l >= rapport_min
    except (TypeError, IndexError, ValueError):
        pass
    return Mesure(piece=piece, position=T[:3, 3] * 1000.0,
                  lacet=float(angle_180(math.degrees(math.atan2(T[1, 0], T[0, 0])))),
                  inclinaison=float(math.degrees(math.acos(float(np.clip(T[2, 2], -1, 1))))),
                  type=str(getattr(piece, "type", "") or ""),
                  face=str(getattr(piece, "face", "") or ""),
                  confiance=conf, lacet_defini=defini,
                  numero=getattr(piece, "numero", None))


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------

class Filtre1D:
    """
    Filtre de Kalman scalaire a modele constant : x(t+1) = x(t) + bruit de
    processus q. Assez pour une piece posee (elle ne bouge que par sauts,
    traites a part). `periode` : grandeur periodique (le lacet, 180 deg),
    l'innovation est alors ramenee dans [-periode/2, periode/2[.

    Il detecte aussi une DERIVE lente (piece poussee de quelques mm) que le
    test de saut ne voit pas et qu'un filtre tres lisseur suivrait avec
    retard : si les `n_biais` dernieres innovations sont toutes du meme
    signe et que leur moyenne s'ecarte de zero de plus de k_biais
    ecarts-types (de la moyenne), l'etat est recale sur les dernieres
    mesures.
    """

    def __init__(self, x: float, r: float, periode: Optional[float] = None,
                 n_biais: int = 4, k_biais: float = 3.0):
        self.periode = periode
        self.x = self._ramener(x)
        self.P = r
        self.n_biais, self.k_biais = n_biais, k_biais
        self.innovations: List[Tuple[float, float]] = []   # (innovation, variance)

    def _ramener(self, a: float) -> float:
        if self.periode is None:
            return float(a)
        h = self.periode / 2.0
        return float((a + h) % self.periode - h)

    def ecart(self, z: float) -> float:
        """Innovation z - x (ramenee si periodique)."""
        d = z - self.x
        return self._ramener(d) if self.periode is not None else float(d)

    def predire(self, q: float) -> None:
        self.P += q

    def corriger(self, z: float, r: float) -> float:
        """Integre la mesure z (variance r) ; rend l'innovation normalisee
        nu^2 / S (vaut 1 en moyenne si le bruit r est le bon)."""
        nu = self.ecart(z)
        S = self.P + r
        K = self.P / S
        self.x = self._ramener(self.x + K * nu)
        self.P = (1.0 - K) * self.P
        self.innovations = (self.innovations + [(nu, S)])[-self.n_biais:]
        if len(self.innovations) == self.n_biais:
            moy = sum(e for e, _ in self.innovations) / self.n_biais
            sigma = math.sqrt(sum(ss for _, ss in self.innovations)) / self.n_biais
            meme_signe = (all(e > 0 for e, _ in self.innovations)
                          or all(e < 0 for e, _ in self.innovations))
            if meme_signe and abs(moy) > self.k_biais * sigma:
                # derive : les dernieres mesures sont d'accord entre elles et
                # pas avec l'etat ; on les croit
                self.x = self._ramener(self.x + (1.0 - K) * moy)
                self.P = max(self.P, sigma ** 2)
                self.innovations = []
        return nu * nu / S

    def reinitialiser(self, z: float, r: float) -> None:
        self.x, self.P, self.innovations = self._ramener(z), r, []


class Vote:
    """
    Distribution cumulee sur des etiquettes (type, ou face), ponderee par la
    confiance des detections, avec oubli (les vieilles detections comptent
    moins : une piece retournee finit par changer de face). L'etiquette
    rendue ne change que si une autre la depasse de `marge` en probabilite
    (hysteresis : pas de clignotement entre deux types proches).
    """

    def __init__(self, oubli: float = 0.85, marge: float = 0.15):
        self.scores: Dict[str, float] = {}
        self.oubli, self.marge = oubli, marge
        self.choix: Optional[str] = None

    def ajouter(self, etiquette: str, poids: float) -> bool:
        """Ajoute une voix ; True si l'etiquette choisie a change."""
        for k in self.scores:
            self.scores[k] *= self.oubli
        self.scores[etiquette] = self.scores.get(etiquette, 0.0) + max(poids, 1e-3)
        ancien = self.choix
        p = self.probabilites()
        meilleur = max(p, key=p.get)
        if ancien is None or ancien not in p:
            self.choix = meilleur
        elif meilleur != ancien and p[meilleur] > p[ancien] + self.marge:
            self.choix = meilleur
        return ancien is not None and self.choix != ancien

    def probabilites(self) -> Dict[str, float]:
        total = sum(self.scores.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in self.scores.items()}

    def proba(self, etiquette: str) -> float:
        return self.probabilites().get(etiquette, 0.0)

    def recalculer(self) -> None:
        """Choix = etiquette la plus probable (sans hysteresis)."""
        p = self.probabilites()
        if p:
            self.choix = max(p, key=p.get)

    def affaiblir(self, facteur: float) -> None:
        for k in self.scores:
            self.scores[k] *= facteur


# ---------------------------------------------------------------------------
# Reglages, evenements, pistes
# ---------------------------------------------------------------------------

@dataclass
class Reglages:
    """Parametres du suivi (mm, degres, nombres de mises a jour)."""
    # bruit des detections (ecart-type, pour une confiance de 1)
    sigma_position: float = 2.0       # mm, x et y
    sigma_hauteur: float = 1.5        # mm, z
    sigma_lacet: float = 3.0          # deg
    sigma_lacet_indefini: float = 40.0  # deg, face superieure presque carree
    rapport_lacet: float = 1.25       # long / large en dessous : lacet indefini
    # bruit de processus (variance ajoutee par mise a jour, et par seconde)
    q_position: float = 0.02          # mm^2
    q_lacet: float = 0.05             # deg^2
    q_par_seconde: float = 0.0        # mm^2/s (deg^2/s pour le lacet)
    bruit_adaptatif: bool = True      # sigma_* corriges par les innovations observees
    # association
    portee: float = 30.0              # mm, distance 3D maximale detection-piste
    poids_angle: float = 0.5          # cout d'un ecart de 90 deg
    poids_type: float = 0.6           # cout d'un type jamais vu sur la piste
    poids_face: float = 0.2
    doublon: float = 8.0              # mm : detection en double d'une piece suivie
    # sauts : la piece a ete deplacee
    saut_position: float = 8.0        # mm (et au moins 4 sigma)
    saut_hauteur: float = 6.0         # mm
    saut_angle: float = 15.0          # deg
    k_saut: float = 4.0
    portee_saut: float = 300.0        # mm : piste provisoire -> piste disparue
    confirmer_sauts: bool = True      # un saut doit etre vu deux fois
    # cycle de vie
    confirmations: int = 2            # detections pour confirmer une piste
    absences_max: int = 4             # mises a jour sans detection, piste confirmee
    absences_provisoire: int = 0      # idem, piste provisoire
    duree_absence_max: Optional[float] = None   # s ; None : pas de limite de temps
    decroissance: float = 0.7         # confiance x decroissance^absences
    amorcage: bool = True             # 1re mise a jour : pistes confirmees d'emblee
    # votes
    oubli_vote: float = 0.85
    marge_vote: float = 0.15
    rendre_provisoires: bool = False
    memoire: int = 10                 # mises a jour ou une piste supprimee
                                      # peut renaitre a la meme place


@dataclass
class Evenement:
    """Ce qui a change. genre : apparue, disparue, deplacee, changee,
    saisie_probable."""
    genre: str
    id_piste: int
    type: str
    position: Tuple[float, float, float]     # mm, repere table
    details: str = ""
    instant: float = 0.0

    def __str__(self):
        x, y, z = self.position
        return "{:16s} piste {:3d} {:8s} x {:6.1f} y {:6.1f} z {:5.1f}{}".format(
            self.genre, self.id_piste, self.type, x, y, z,
            "  " + self.details if self.details else "")


class Piste:
    """Une piece suivie."""

    def __init__(self, identifiant: int, m: Mesure, reg: Reglages, instant: float,
                 echelles: Optional[Dict[str, float]] = None):
        self.id = identifiant
        self.reg = reg
        # facteurs de variance de mesure appris par le Suivi (partages)
        self.echelles = echelles if echelles is not None else {"xy": 1.0, "z": 1.0, "lacet": 1.0}
        self.nis: Dict[str, List[float]] = {}   # innovations normalisees, derniere correction
        r_xy, r_z, r_a = self._bruits(m)
        self.f = [Filtre1D(m.position[0], r_xy), Filtre1D(m.position[1], r_xy),
                  Filtre1D(m.position[2], r_z),
                  Filtre1D(m.lacet, r_a, periode=180.0)]
        self.vote_type = Vote(reg.oubli_vote, reg.marge_vote)
        self.vote_face = Vote(reg.oubli_vote, reg.marge_vote)
        self.vote_type.ajouter(m.type, m.confiance)
        self.vote_face.ajouter(m.face, m.confiance)
        self.confiance_moyenne = m.confiance
        self.inclinaison = m.inclinaison
        self.lacet_defini = m.lacet_defini
        self.age = 1                         # mises a jour depuis la creation
        self.detections = 1
        self.absences = 0
        self.confirmee = False
        self.creation = instant
        self.vue_le = instant
        self.derniere = m.piece              # derniere detection brute
        self.par_cle: Dict[Tuple[str, str], object] = {(m.type, m.face): m.piece}
        self.support: Optional[int] = None   # en identifiants de piste
        self.dessus: List[int] = []
        self.saisie_annoncee = False

    # -- etat -------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return np.array([self.f[0].x, self.f[1].x, self.f[2].x])

    @property
    def lacet(self) -> float:
        return self.f[3].x

    @property
    def type(self) -> str:
        return self.vote_type.choix or ""

    @property
    def face(self) -> str:
        return self.vote_face.choix or ""

    @property
    def libre(self) -> bool:
        return not self.dessus

    @property
    def confiance(self) -> float:
        """Moyenne des confiances des detections, moins sure tant que la
        piste est jeune, decroissante pendant une absence."""
        jeunesse = 1.0 - 0.5 ** self.detections
        return float(self.confiance_moyenne * jeunesse
                     * self.reg.decroissance ** self.absences)

    def _bruits(self, m: Mesure) -> Tuple[float, float, float]:
        """Variances de mesure : une detection peu sure pese moins."""
        c = max(m.confiance, 0.2)
        e = self.echelles
        sa2 = (self.reg.sigma_lacet ** 2 * e["lacet"] if m.lacet_defini
               else self.reg.sigma_lacet_indefini ** 2)
        return (self.reg.sigma_position ** 2 * e["xy"] / c,
                self.reg.sigma_hauteur ** 2 * e["z"] / c, sa2 / c)

    # -- association --------------------------------------------------------
    def distance(self, m: Mesure) -> float:
        return float(np.linalg.norm(m.position - self.position))

    def cout(self, m: Mesure) -> float:
        d = self.distance(m)
        if d > self.reg.portee:
            return INTERDIT
        c = d / self.reg.portee
        if m.lacet_defini and self.lacet_defini:
            c += self.reg.poids_angle * abs(self.f[3].ecart(m.lacet)) / 90.0
        c += self.reg.poids_type * (1.0 - self.vote_type.proba(m.type))
        c += self.reg.poids_face * (1.0 - self.vote_face.proba(m.face))
        return c

    def saut(self, m: Mesure) -> Optional[str]:
        """Description du saut si la mesure est trop loin de l'etat pour
        etre du bruit (piece deplacee), sinon None."""
        reg = self.reg
        r_xy, r_z, r_a = self._bruits(m)
        dxy = float(np.hypot(m.position[0] - self.f[0].x, m.position[1] - self.f[1].x))
        dz = abs(m.position[2] - self.f[2].x)
        sxy = math.sqrt(max(self.f[0].P, self.f[1].P) + r_xy)
        raisons = []
        if dxy > max(reg.saut_position, reg.k_saut * sxy):
            raisons.append("{:.0f} mm".format(dxy))
        if dz > max(reg.saut_hauteur, reg.k_saut * math.sqrt(self.f[2].P + r_z)):
            raisons.append("dz {:+.0f} mm".format(m.position[2] - self.f[2].x))
        if m.lacet_defini and self.lacet_defini:
            da = self.f[3].ecart(m.lacet)
            if abs(da) > max(reg.saut_angle, reg.k_saut * math.sqrt(self.f[3].P + r_a)):
                raisons.append("{:+.0f} deg".format(da))
        return ", ".join(raisons) if raisons else None

    def saut_vers(self, autre: "Piste") -> bool:
        """L'estimation d'une autre piste (moyenne de plusieurs mesures)
        est-elle trop loin de celle-ci pour etre du bruit ? Meme test que
        saut(), avec les incertitudes des deux estimations."""
        reg = self.reg
        dxy = float(np.hypot(*(autre.position[:2] - self.position[:2])))
        sxy = math.sqrt(max(self.f[0].P, self.f[1].P) + max(autre.f[0].P, autre.f[1].P))
        if dxy > max(reg.saut_position, reg.k_saut * sxy):
            return True
        if abs(autre.f[2].x - self.f[2].x) > max(
                reg.saut_hauteur, reg.k_saut * math.sqrt(self.f[2].P + autre.f[2].P)):
            return True
        if self.lacet_defini and autre.lacet_defini:
            da = self.f[3].ecart(autre.f[3].x)
            return abs(da) > max(reg.saut_angle,
                                 reg.k_saut * math.sqrt(self.f[3].P + autre.f[3].P))
        return False

    def fusionner(self, autre: "Piste") -> None:
        """Integre l'estimation d'une autre piste comme une mesure (de
        variance son incertitude) : la piste provisoire ouverte par une
        mesure qu'on croyait un saut n'etait que du bruit."""
        for k in range(4):
            if k == 3 and not autre.lacet_defini:
                continue
            self.f[k].corriger(autre.f[k].x, autre.f[k].P)
        for k, v in autre.vote_type.scores.items():
            self.vote_type.ajouter(k, v)
        for k, v in autre.vote_face.scores.items():
            self.vote_face.ajouter(k, v)
        self.derniere = autre.derniere
        self.par_cle.update(autre.par_cle)
        self.absences = 0
        self.vue_le = autre.vue_le
        self.detections += autre.detections

    # -- mise a jour --------------------------------------------------------
    def predire(self, dt: float) -> None:
        q = self.reg.q_par_seconde * max(dt, 0.0)
        for f in self.f[:3]:
            f.predire(self.reg.q_position + q)
        self.f[3].predire(self.reg.q_lacet + q)
        self.age += 1

    def corriger(self, m: Mesure, instant: float, sauter: bool) -> List[str]:
        """Integre une detection ; rend les changements de vote."""
        r_xy, r_z, r_a = self._bruits(m)
        r = (r_xy, r_xy, r_z, r_a)
        # innovations normalisees, pour apprendre le bruit (piste installee)
        self.nis = {}
        installee = self.detections >= 3 and not sauter
        for k in range(3):
            if sauter:
                self.f[k].reinitialiser(m.position[k], r[k])
            else:
                nis = self.f[k].corriger(m.position[k], r[k])
                if installee:
                    self.nis.setdefault("xy" if k < 2 else "z", []).append(nis)
        if sauter or (m.lacet_defini and not self.lacet_defini):
            # piece reposee, ou premier lacet bien defini : on le prend tel quel
            self.f[3].reinitialiser(m.lacet, r_a)
            self.lacet_defini = m.lacet_defini
        elif m.lacet_defini or not self.lacet_defini:
            # un lacet mal defini (face carree) n'informe que si la piste
            # n'a rien de mieux
            nis = self.f[3].corriger(m.lacet, r_a)
            if installee and m.lacet_defini:
                self.nis["lacet"] = [nis]
        if sauter:
            self.vote_face.affaiblir(0.3)   # reposee : peut-etre sur une autre face
        changes = []
        if self.vote_type.ajouter(m.type, m.confiance):
            changes.append("type -> {}".format(self.type))
        if self.vote_face.ajouter(m.face, m.confiance):
            changes.append("face -> {}".format(self.face))
        self.confiance_moyenne += 0.3 * (m.confiance - self.confiance_moyenne)
        self.inclinaison += 0.3 * (m.inclinaison - self.inclinaison)
        self.detections += 1
        self.absences = 0
        self.vue_le = instant
        self.derniere = m.piece
        self.par_cle[(m.type, m.face)] = m.piece
        return changes

    def adopter(self, autre: "Piste") -> None:
        """Reprend l'etat d'une autre piste (piece deplacee loin : la piste
        provisoire confirmee a la nouvelle place transmet sa mesure a la
        piste disparue, qui garde son identifiant)."""
        self.f = autre.f
        self.lacet_defini = autre.lacet_defini
        for k, v in autre.vote_type.scores.items():
            self.vote_type.scores[k] = self.vote_type.scores.get(k, 0.0) * 0.5 + v
        self.vote_type.recalculer()
        self.vote_face = autre.vote_face
        self.derniere = autre.derniere
        self.par_cle.update(autre.par_cle)
        self.absences = 0
        self.vue_le = autre.vue_le
        self.detections += autre.detections
        self.inclinaison = autre.inclinaison


# ---------------------------------------------------------------------------
# Le suivi
# ---------------------------------------------------------------------------

def _rz(degres: float) -> np.ndarray:
    c, s = math.cos(math.radians(degres)), math.sin(math.radians(degres))
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    return R


class Suivi:
    """
    Suivi multi-objets des pieces detectees.

        suivi = Suivi()
        pieces = suivi.mettre_a_jour(perception.detecter())
        suivi.evenements          # ce qui a change a cette mise a jour

    Les reglages (Reglages) se passent en mots-cles : Suivi(portee=40,
    absences_max=6). Voir l'en-tete du module pour le principe.
    """

    def __init__(self, reglages: Optional[Reglages] = None, **kw):
        self.reg = reglages or Reglages()
        for k, v in kw.items():
            if not hasattr(self.reg, k):
                raise TypeError("reglage inconnu : {}".format(k))
            setattr(self.reg, k, v)
        self.pistes: List[Piste] = []
        self.evenements: List[Evenement] = []
        #: facteurs des variances de mesure (bruit_adaptatif), partages par
        #: les pistes : 1 si les sigma_* des reglages sont justes
        self.echelles: Dict[str, float] = {"xy": 1.0, "z": 1.0, "lacet": 1.0}
        self.memoire: List[Tuple[Piste, int]] = []   # pistes supprimees recemment
        self._ids = itertools.count(1)
        self._instant: Optional[float] = None
        self._mises_a_jour = 0

    def reinitialiser(self) -> None:
        """Oublie tout (scene changee du tout au tout, camera deplacee)."""
        self.__init__(self.reg)

    def piste(self, id_piste: int) -> Optional[Piste]:
        for p in self.pistes:
            if p.id == id_piste:
                return p
        return None

    def signaler_saisie(self, id_piste: int) -> None:
        """Le programme annonce qu'il prend cette piece : a sa prochaine
        absence, elle est supprimee tout de suite (pas de maintien pour
        occultation) avec l'evenement saisie_probable."""
        p = self.piste(id_piste)
        if p is not None:
            p.saisie_annoncee = True

    # -- la mise a jour -------------------------------------------------------
    def mettre_a_jour(self, pieces: Sequence, instant: Optional[float] = None) -> List:
        """
        Integre les detections d'une image (sortie de Perception.detecter())
        et rend les pieces suivies (copies, pose lissee, id_piste). `instant`
        (s) : l'heure de la capture, time.monotonic() par defaut ; elle ne
        sert qu'aux reglages par seconde.
        """
        reg = self.reg
        instant = time.monotonic() if instant is None else float(instant)
        dt = 0.0 if self._instant is None else instant - self._instant
        self._instant = instant
        self._mises_a_jour += 1
        self.evenements = []
        amorce = reg.amorcage and self._mises_a_jour == 1

        mesures = [m for m in (mesure_de(p, reg.rapport_lacet) for p in pieces) if m is not None]
        for t in self.pistes:
            t.predire(dt)

        # 1. association optimale, avec portee
        C = np.array([[t.cout(m) for m in mesures] for t in self.pistes]).reshape(
            len(self.pistes), len(mesures))
        paires = associer(C)
        piste_de: Dict[int, Piste] = {}          # indice de mesure -> piste
        sauts: Dict[int, str] = {}
        for i, j in paires:
            t, m = self.pistes[i], mesures[j]
            raison = t.saut(m)
            if raison is not None and (self.reg.confirmer_sauts or self._ambigu(t, m)):
                # saut a confirmer, ou pres d'une autre piece : la mesure
                # ouvre une piste provisoire, qui heritera de l'identite si
                # elle se confirme (etape 5)
                continue
            piste_de[j] = t
            if raison is not None:
                sauts[j] = raison

        # 2. corrections
        for j, t in piste_de.items():
            m = mesures[j]
            ancienne = t.position
            changes = t.corriger(m, instant, sauter=j in sauts)
            if t.confirmee:
                if j in sauts:
                    self._evenement("deplacee", t, "de ({:.0f}, {:.0f}, {:.0f}) : {}".format(
                        *ancienne, sauts[j]))
                for c in changes:
                    self._evenement("changee", t, c)
        if reg.bruit_adaptatif:
            self._apprendre_bruit(piste_de.values())

        # 3. detections restantes : doublons ignores, sinon pistes provisoires
        suivies = list(piste_de.values())
        for j, m in enumerate(mesures):
            if j in piste_de:
                continue
            if any(t.distance(m) < reg.doublon for t in suivies):
                continue
            t = Piste(next(self._ids), m, reg, instant, self.echelles)
            self.pistes.append(t)
            piste_de[j] = t

        # 4. absences
        vues = set(id(t) for t in piste_de.values())
        for t in list(self.pistes):
            if id(t) in vues:
                continue
            t.absences += 1
            if not t.confirmee:
                if t.absences > reg.absences_provisoire:
                    self.pistes.remove(t)
                continue
            trop = t.absences > reg.absences_max or (
                reg.duree_absence_max is not None
                and instant - t.vue_le > reg.duree_absence_max)
            if t.saisie_annoncee or trop:
                self._supprimer(t)

        # 5. confirmations (et identite transmise d'une piece deplacee loin)
        for t in list(self.pistes):
            if t.confirmee or id(t) not in vues:
                continue
            if not (amorce or t.detections >= reg.confirmations):
                continue
            ancienne, revenue = (None, False) if amorce else self._disparue_proche(t)
            if ancienne is not None and not revenue and not ancienne.saut_vers(t):
                # pas de saut, tout compte fait : du bruit, pris pour un
                # saut sur une seule mesure ; la piste absente l'integre
                ancienne.fusionner(t)
                self.pistes.remove(t)
                for j, tt in piste_de.items():
                    if tt is t:
                        piste_de[j] = ancienne
            elif ancienne is not None:
                depart = ancienne.position
                ancienne.adopter(t)
                self.pistes.remove(t)
                if revenue:
                    self.memoire = [(u, k) for u, k in self.memoire if u is not ancienne]
                    self.pistes.append(ancienne)
                for j, tt in piste_de.items():
                    if tt is t:
                        piste_de[j] = ancienne
                saut = float(np.linalg.norm(ancienne.position - depart))
                if revenue:
                    self._evenement("apparue", ancienne, "de retour ({:.0f} mm de sa "
                                    "derniere place)".format(saut))
                else:
                    self._evenement("deplacee", ancienne,
                                    "de ({:.0f}, {:.0f}, {:.0f}) : {:.0f} mm".format(*depart, saut))
            else:
                t.confirmee = True
                self._evenement("apparue", t)

        # 6. empilement, traduit en identifiants de piste
        id_de_numero = {mesures[j].numero: t.id for j, t in piste_de.items()
                        if mesures[j].numero is not None}
        for j, t in piste_de.items():
            p = mesures[j].piece
            s = getattr(p, "support", None)
            t.support = id_de_numero.get(s) if s is not None else None
            t.dessus = [id_de_numero[n] for n in (getattr(p, "dessus", None) or [])
                        if n in id_de_numero]
        return self.pieces()

    def _ambigu(self, t: Piste, m: Mesure) -> bool:
        """Une autre piste est a portee de la mesure : un saut de t vers
        cette mesure pourrait etre une confusion de pieces voisines."""
        return any(u is not t and u.confirmee and u.distance(m) < self.reg.portee
                   for u in self.pistes)

    def _disparue_proche(self, t: Piste) -> Tuple[Optional[Piste], bool]:
        """La piece que la nouvelle piste t confirmee pourrait etre : la
        plus proche, du meme type, parmi les pistes confirmees absentes
        (dans portee_saut : deplacee, peut-etre loin) et les pistes
        supprimees depuis moins de `memoire` mises a jour (dans portee
        seulement : revenue a sa place apres une longue occultation).
        Rend (piste, revenue) ; (None, False) si aucune."""
        reg = self.reg

        def dxy(u):
            return float(np.hypot(*(u.position[:2] - t.position[:2])))

        candidates = [(dxy(u), False, u) for u in self.pistes
                      if u is not t and u.confirmee and u.absences > 0
                      and not u.saisie_annoncee and u.type == t.type
                      and dxy(u) < reg.portee_saut]
        candidates += [(dxy(u), True, u) for u, k in self.memoire
                       if self._mises_a_jour - k <= reg.memoire
                       and u.type == t.type and dxy(u) < reg.portee]
        if not candidates:
            return None, False
        _, revenue, u = min(candidates, key=lambda c: c[0])
        return u, revenue

    def _apprendre_bruit(self, pistes) -> None:
        """Bruit de mesure ADAPTATIF : si les sigma_* des reglages sont
        justes, l'innovation normalisee nu^2/S vaut 1 en moyenne. Plus grande,
        les detections sont plus bruitees que prevu (le filtre leur ferait
        trop confiance, et prendrait du bruit pour des sauts) ; plus petite,
        elles sont meilleures (on lisse trop). Chaque facteur d'echelle est
        corrige doucement (moyenne geometrique glissante), borne a [1/4, 16]."""
        for cle in self.echelles:
            valeurs = [v for t in pistes if t.confirmee for v in t.nis.get(cle, [])]
            if not valeurs:
                continue
            moyenne = float(np.clip(np.mean(valeurs), 0.2, 5.0))
            alpha = 0.2 * min(1.0, len(valeurs) / 4.0)
            self.echelles[cle] = float(np.clip(self.echelles[cle] * moyenne ** alpha,
                                               0.25, 16.0))

    def _supprimer(self, t: Piste) -> None:
        self.pistes.remove(t)
        if not t.saisie_annoncee:
            self.memoire = [(u, k) for u, k in self.memoire
                            if self._mises_a_jour - k <= self.reg.memoire] + [(t, self._mises_a_jour)]
        self._evenement("disparue", t, "apres {} absence(s)".format(t.absences))
        if t.libre:
            self._evenement("saisie_probable", t,
                            "annoncee" if t.saisie_annoncee else "piece libre disparue")

    def _evenement(self, genre: str, t: Piste, details: str = "") -> None:
        self.evenements.append(Evenement(genre, t.id, t.type, tuple(float(v) for v in t.position),
                                         details, self._instant or 0.0))

    # -- sortie ---------------------------------------------------------------
    def pieces(self) -> List:
        """Les pieces suivies (confirmees, et provisoires si
        rendre_provisoires), par identifiant croissant."""
        rendues = [t for t in self.pistes if t.confirmee or self.reg.rendre_provisoires]
        ids = {t.id for t in rendues}
        return [self._piece(t, ids) for t in sorted(rendues, key=lambda t: t.id)]

    def _piece(self, t: Piste, ids: set):
        """Copie de la detection la plus representative, pose lissee."""
        base = (t.par_cle.get((t.type, t.face))
                or next((p for (ty, _), p in t.par_cle.items() if ty == t.type), None)
                or t.derniere)
        p = copy.copy(base)
        T_brut = np.asarray(base.T_table, float)
        lacet_brut = math.degrees(math.atan2(T_brut[1, 0], T_brut[0, 0]))
        T = _rz(float(angle_180(t.lacet - lacet_brut))) @ T_brut
        T[:3, 3] = t.position / 1000.0
        _poser(p, "T_table", T)
        for nom in ("T_camera", "T_robot"):
            M = getattr(base, nom, None)
            if M is not None:
                # M = A @ T_brut (A : repere table -> camera / robot) ; on
                # garde A et on remplace T_brut par T
                _poser(p, nom, np.asarray(M, float) @ np.linalg.inv(T_brut) @ T)
        _poser(p, "type", t.type)
        _poser(p, "face", t.face)
        _poser(p, "numero", t.id)
        _poser(p, "support", t.support if t.support in ids else None)
        _poser(p, "dessus", [n for n in t.dessus if n in ids])
        _poser(p, "confiance_detection", getattr(base, "confiance", None))
        _poser(p, "confiance", t.confiance)
        for nom, val in (("id_piste", t.id), ("vue", t.absences == 0),
                         ("T_table_brute", T_brut),
                         ("absences", t.absences), ("confirmations", t.detections),
                         ("age", t.age), ("inclinaison", t.inclinaison),
                         ("probabilites_type", t.vote_type.probabilites()),
                         ("probabilites_face", t.vote_face.probabilites())):
            _poser(p, nom, val)
        return p


def _poser(objet, nom: str, valeur) -> None:
    """setattr qui tolere un attribut en lecture seule (propriete d'une
    classe d'entree inconnue)."""
    try:
        setattr(objet, nom, valeur)
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def afficher(pieces: Sequence, evenements: Sequence[Evenement], titre: str = "") -> None:
    if titre:
        print(titre)
    for p in pieces:
        x, y, z = np.asarray(p.T_table, float)[:3, 3] * 1000.0
        lacet = math.degrees(math.atan2(p.T_table[1, 0], p.T_table[0, 0]))
        etat = "vue" if getattr(p, "vue", True) else "cachee ({})".format(getattr(p, "absences", "?"))
        sup = getattr(p, "support", None)
        print("  piste {:3d} {:8s} {:15s} x {:6.1f} y {:6.1f} z {:5.1f} mm {:5.0f} deg "
              "conf {:.2f}  {}{}".format(p.id_piste, p.type, p.face, x, y, z, lacet,
                                         p.confiance, etat,
                                         "" if sup is None else "  sur {}".format(sup)))
    for e in evenements:
        print("  >>", e)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Suivi des pieces d'une image a l'autre")
    ap.add_argument("captures", nargs="*", help="fichiers .npz, dans l'ordre")
    ap.add_argument("--direct", action="store_true",
                    help="captures successives par le serveur de camera")
    ap.add_argument("--nombre", type=int, default=0,
                    help="nombre de captures en direct (0 : jusqu'a Ctrl-C)")
    ap.add_argument("--images", type=int, default=None,
                    help="images par capture (mediane), defaut de camera.py")
    ap.add_argument("--sans-reseau", action="store_true")
    ap.add_argument("--sortie", default=None,
                    help="image annotee de la derniere mise a jour (identifiants)")
    a = ap.parse_args(argv)
    if not a.direct and not a.captures:
        ap.error("donner des captures .npz, ou --direct")
    if ICI not in sys.path:
        sys.path.insert(0, ICI)
    from perception import Perception
    perception = Perception(modele=False if a.sans_reseau else None)
    suivi = Suivi()
    print("perception : {}".format(perception.mode))
    sources = a.captures if not a.direct else (
        itertools.count() if a.nombre <= 0 else range(a.nombre))
    pieces = []
    try:
        for k, source in enumerate(sources):
            t0 = time.monotonic()
            if a.direct:
                detections = perception.detecter(None, images=a.images)
                nom = "capture {}".format(k + 1)
            else:
                detections = perception.detecter(source)
                nom = os.path.basename(source)
            pieces = suivi.mettre_a_jour(detections, instant=t0)
            afficher(pieces, suivi.evenements, "{} : {} detection(s), {} piece(s) suivie(s), "
                     "{:.1f} s".format(nom, len(detections), len(pieces), time.monotonic() - t0))
    except KeyboardInterrupt:
        print("arret")
    if a.sortie and perception.derniere_vue is not None:
        import cv2
        cv2.imwrite(a.sortie, perception.dessiner(pieces))
        print("image :", a.sortie)
    return 0


if __name__ == "__main__":
    sys.exit(main())
