"""
Modele cinematique du Dobot Magician.

Structure : 4 axes, architecture serie a parallelogramme.

    J1  rotation de la colonne autour de la verticale
    J2  bras arriere      (0 deg = bras vertical, positif vers l'avant)
    J3  avant-bras        (0 deg = avant-bras horizontal, positif vers le bas)
    J4  rotation de l'outil autour de la verticale

Le parallelogramme du Magician maintient l'axe de l'outil VERTICAL quelles
que soient les valeurs de J2 et J3. Le robot possede donc 4 degres de
liberte utiles : x, y, z et l'orientation r autour de la verticale.

Chaine de transformations homogenes retenue :

    T01 = Rz(q1)                                base -> colonne
    T12 = Tz(d1) . Ry(q2)                       colonne -> bras arriere
    T23 = Tz(l1) . Ry(q3 - q2 + pi/2)           bras arriere -> avant-bras
    T34 = Tz(l2) . Ry(-pi/2 - q3)               avant-bras -> poignet
    T45 = Tx(l3) . Rz(q4)                       poignet -> axe outil
    T5T = Tz(-lt) . Rx(pi)                      axe outil -> TCP (z vers le bas)

On verifie que la composition des rotations autour de Y se simplifie :
q2 + (q3 - q2 + pi/2) + (-pi/2 - q3) = 0, ce qui traduit exactement l'effet
du parallelogramme.

API publique : les angles articulaires sont en DEGRES (comme l'API Dobot),
les longueurs en MILLIMETRES.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import homogeneous as hg

__all__ = ["Geometrie", "Limites", "Encombrement",
           "DobotMagician", "ErreurCinematique",
           "cible", "repere_travail", "decomposer_cible",
           "vers_xyzrpy", "depuis_xyzrpy",
           "verifier_repere_outil", "verifier_repere_objet"]

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class ErreurCinematique(Exception):
    """Levee quand une pose est hors de l'espace atteignable."""


# ---------------------------------------------------------------------------
# Construction des cibles sous forme de matrices homogenes
# ---------------------------------------------------------------------------

def repere_travail(x: float = 0.0, y: float = 0.0, z: float = 0.0,
                   rz: float = 0.0) -> np.ndarray:
    """
    Repere de travail pose sur la table : axe z vers le HAUT.

        R = Trans(x, y, z) . Rz(rz)

    Sert a exprimer des cibles en coordonnees locales : un magasin, une
    palette, un gabarit. Les angles sont en degres.
    """
    return hg.composer(hg.translation(x, y, z), hg.rot_z(rz * D2R))


def cible(x: float = 0.0, y: float = 0.0, z: float = 0.0, rz: float = 0.0,
          repere: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Situation d'une cible OUTIL, sous forme de matrice homogene 4x4.

        T = Trans(x, y, z) . Rz(rz) . Rx(180 deg)

    Le Rx(180) retourne l'axe z : celui de l'outil pointe vers le BAS,
    c'est la direction d'approche. Consequence pratique, un degagement
    vers le haut s'ecrit `T @ translation(0, 0, -h)`, le deplacement etant
    exprime dans le repere de l'outil.

    `repere` permet de donner x, y, z, rz dans un repere de travail plutot
    que dans le repere de base :

        PALETTE = repere_travail(215, 75, 25, rz=20)
        case_3  = cible(0, 3 * 24, 0, repere=PALETTE)

    Piege a connaitre : comme l'axe z de l'outil pointe vers le bas,
    post-multiplier par Rz(a) tourne de -a vu de la base. Pour fixer
    l'orientation, utilisez l'argument rz plutot que `T @ rot_z(...)`.

    Les angles sont en degres, les longueurs en millimetres.
    """
    if np.ndim(x) != 0:
        # Erreur frequente : passer une matrice 4x4 a une fonction qui en
        # CONSTRUIT une. Une matrice est deja une cible, elle se donne
        # directement a movj / movl / movc.
        raise TypeError(
            "cible() construit une matrice 4x4 a partir de coordonnees "
            "(x, y, z, rz) ; elle n'en recoit pas une.\n"
            "  Une matrice homogene est DEJA une cible :\n"
            "      PRISE = np.array([[...], [...], [...], [...]])\n"
            "      robot.movl(PRISE)\n"
            "  Pour l'exprimer dans un repere de travail, passez-la en "
            "pre-multiplication : REPERE @ PRISE, ou utilisez "
            "cible(x, y, z, rz, repere=REPERE).")
    T = hg.composer(hg.translation(x, y, z), hg.rot_z(rz * D2R), hg.rot_x(np.pi))
    return T if repere is None else np.asarray(repere, dtype=float) @ T


def vers_xyzrpy(T) -> np.ndarray:
    """
    Matrice homogene -> (X, Y, Z, Rx, Ry, Rz), en millimetres et degres.

    C'est la forme d'affichage usuelle en robotique industrielle. Les
    angles suivent la convention roulis-tangage-lacet autour des axes
    fixes : R = Rz(Rz) . Ry(Ry) . Rx(Rx).

    Sur le Magician, Rx et Ry ne sont pas commandables : le parallelogramme
    impose Rx = 180 deg et Ry = 0 pour une cible ou un repere outil, et
    Rx = Ry = 0 pour un repere objet. Les afficher rend cette contrainte
    visible plutot que tacite.
    """
    T = np.asarray(T, dtype=float)
    roulis, tangage, lacet = hg.rotation_vers_rpy(T)
    p = T[:3, 3]
    v = np.array([p[0], p[1], p[2],
                  np.degrees(roulis), np.degrees(tangage),
                  np.degrees(lacet)], dtype=float)
    return v + 0.0          # supprime les zeros negatifs a l'affichage


def depuis_xyzrpy(x: float = 0.0, y: float = 0.0, z: float = 0.0,
                  rx: float = 0.0, ry: float = 0.0, rz: float = 0.0
                  ) -> np.ndarray:
    """(X, Y, Z, Rx, Ry, Rz) en mm et degres -> matrice homogene."""
    return hg.composer(hg.translation(x, y, z),
                       hg.rpy_vers_rotation(rx * D2R, ry * D2R, rz * D2R))


def verifier_repere_outil(T, tolerance_deg: float = 1.0) -> np.ndarray:
    """
    Valide un repere OUTIL : axe 5 (axe de rotation de l'outil) -> TCP.

    Il decrit ou se trouve la pointe par rapport a l'axe J4, et comment
    elle est orientee. Un deport lateral et une rotation propre sont
    parfaitement realisables ; une INCLINAISON ne l'est pas, puisque le
    parallelogramme maintient l'axe outil vertical. Le controle est donc
    le meme que pour une cible : l'axe z du repere doit pointer vers le bas.

    Renvoie (dx, dy, dz, angle) : le deport dans le repere de l'axe outil,
    en millimetres, et la rotation propre en degres.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not hg.est_homogene(T, tol=1e-6):
        raise ErreurCinematique(
            "un repere outil doit etre une matrice homogene 4x4 valide")
    ecart = np.degrees(np.arccos(float(np.clip(
        np.dot(T[:3, 2], [0.0, 0.0, -1.0]), -1.0, 1.0))))
    if ecart > tolerance_deg:
        raise ErreurCinematique(
            "repere outil inclinable de {:.1f} deg par rapport a la verticale "
            "descendante : le parallelogramme du Magician maintient l'axe de "
            "l'outil vertical. Un deport lateral et une rotation autour de z "
            "sont possibles, pas une inclinaison.".format(ecart))
    p = T[:3, 3]
    return np.array([p[0], p[1], p[2],
                     np.degrees(np.arctan2(T[1, 0], T[0, 0]))], dtype=float)


def verifier_repere_objet(T, tolerance_deg: float = 1.0) -> np.ndarray:
    """
    Valide un repere OBJET : base -> plan de travail. Axe z vers le HAUT.

    Renvoie (x, y, z, rz) en millimetres et degres.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not hg.est_homogene(T, tol=1e-6):
        raise ErreurCinematique(
            "un repere objet doit etre une matrice homogene 4x4 valide")
    ecart = np.degrees(np.arccos(float(np.clip(
        np.dot(T[:3, 2], [0.0, 0.0, 1.0]), -1.0, 1.0))))
    if ecart > tolerance_deg:
        raise ErreurCinematique(
            "repere objet incline de {:.1f} deg : le robot ne peut orienter "
            "son outil que verticalement, un plan de travail incline rendrait "
            "toutes ses cibles inatteignables. Seuls une translation et une "
            "rotation autour de z sont admis.".format(ecart))
    p = T[:3, 3]
    return np.array([p[0], p[1], p[2],
                     np.degrees(np.arctan2(T[1, 0], T[0, 0]))], dtype=float)


def decomposer_cible(T, tolerance_deg: float = 1.0) -> np.ndarray:
    """
    Matrice homogene -> (x, y, z, r), avec verification de faisabilite.

    Le parallelogramme du Magician impose un axe d'outil vertical dirige
    vers le bas. Toute cible dont ce n'est pas le cas est rejetee ici,
    plutot que silencieusement projetee : c'est une contrainte mecanique,
    pas une approximation.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ErreurCinematique(
            "une cible doit etre une matrice homogene 4x4, recu {}"
            .format(T.shape))
    if not hg.est_homogene(T, tol=1e-6):
        raise ErreurCinematique(
            "la matrice n'est pas homogene valide : sa partie rotation "
            "n'est pas orthonormale directe")

    z_outil = T[:3, 2]
    cos = float(np.clip(np.dot(z_outil, [0.0, 0.0, -1.0]), -1.0, 1.0))
    ecart = np.degrees(np.arccos(cos))
    if ecart > tolerance_deg:
        raise ErreurCinematique(
            "orientation inatteignable : l'axe z de l'outil fait {:.1f} deg "
            "avec la verticale descendante. Le parallelogramme du Magician "
            "maintient l'outil vertical ; seule la rotation r autour de la "
            "verticale est commandable. Construisez la cible avec cible() "
            "et ne composez qu'avec des translations ou des Rz."
            .format(ecart))

    p = T[:3, 3]
    r = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    return np.array([p[0], p[1], p[2], r], dtype=float)


# ---------------------------------------------------------------------------

#: Table de Denavit-Hartenberg standard publiee par Dobot, sous la forme
#: (theta variable, d, a, alpha). Les longueurs sont en millimetres.
#: Elle decrit le BRAS seul : la hauteur du socle n'y figure pas.
DH_CONSTRUCTEUR = (
    # theta,  d,    a,      alpha
    ("theta1", 8.0, 0.0, -np.pi / 2),
    ("theta2", 0.0, 135.0, 0.0),
    ("theta3", 0.0, 147.0, 0.0),
    ("theta4", 0.0, 59.7, -np.pi / 2),
)


@dataclass
class Geometrie:
    """
    Parametres geometriques du Magician, en millimetres.

    Valeurs relevees sur les documents constructeur du dossier
    `Magician & Accessories-2D&3D files` :

      * d1 vient de l'URDF officiel (`magician_description`), qui place
        l'axe d'epaule a 56,9 + 82,1 = 139,0 mm au-dessus du socle ;
      * l1 et l2 sont confirmes par l'URDF et par la table DH ;
      * l3 vient de la table DH (a4 = 59,7 mm).

    Elles restent modifiables : `lt` en particulier depend de l'outil
    monte, et la notice constructeur precise que la distance entre la
    pointe d'un montage maison et le point de reference doit etre
    configuree a la main.
    """
    d1: float = 139.0   # hauteur de l'axe d'epaule au-dessus du plan de base
    l1: float = 135.0   # bras arriere : epaule -> coude
    l2: float = 147.0   # avant-bras   : coude  -> poignet
    l3: float = 59.7    # deport horizontal poignet -> axe de l'outil
    lt: float = 0.0     # deport vertical de la pointe d'outil (vers le bas)


@dataclass
class Encombrement:
    """
    Volumes fixes du robot, pour la detection de collision.

    Les valeurs sont MESUREES sur les maillages du modele CAO officiel
    (boites englobantes des pieces de `magician_description`) :

      * l'embase est une boite de 158 x 158 mm haute de 57,9 mm ;
      * la colonne (link1) est un cylindre de rayon 69,5 mm qui monte
        jusqu'a 176,2 mm, donc au-dessus de l'axe d'epaule (139 mm) ;
      * le plan de pose est le plan z = 0, celui sur lequel l'embase
        repose.

    Les valeurs retenues sont legerement arrondies vers le bas, de sorte
    qu'un effleurement ne declenche pas d'alerte.

    Ce que ce modele NE couvre PAS : la forme exacte des maillages, les
    servos qui depassent lateralement a l'epaule, et tout obstacle exterieur
    au robot. C'est une enveloppe simple, pas un moteur de collision.
    """
    plan_z: float = 0.0          # plan de pose
    embase_demi: float = 79.0    # demi-cote de la boite de l'embase
    embase_haut: float = 58.0
    colonne_rayon: float = 69.5
    colonne_haut: float = 176.0
    pas: float = 6.0             # pas d'echantillonnage le long du squelette

    #: Rayon de chaque maillon mobile, pour l'auto-collision : bras
    #: arriere, avant-bras, deport du poignet, outil. Releves sur la
    #: demi-largeur en y des maillages CAO (link2 +/-20, link3 +/-15,
    #: link4 +/-10), arrondis vers le haut.
    rayons_maillons: Tuple[float, float, float, float] = (22.0, 15.0, 12.0, 15.0)


@dataclass
class Limites:
    """
    Butees articulaires (degres) et limites de vitesse.

    `couplage` : bornes de (q3 - q2), la contrainte liant J2 et J3 (angle
    entre le bras arriere et l'avant-bras). Mesuree le 25/10/2026 sur la
    machine du labo (firmware 3.7.0) : le controleur REFUSE une cible ou
    q3 - q2 vaut -58,2 ou -66 deg (alarme 0x12, qui bloque ensuite sa file
    jusqu'a l'extinction), et accepte -54,3 (le pire de 96 poses reussies).
    D'ou -55. L'autre borne n'a pas ete atteinte (+41 deg sans souci).
    """
    q_min: Tuple[float, float, float, float] = (-135.0, 0.0, -10.0, -90.0)
    q_max: Tuple[float, float, float, float] = (135.0, 85.0, 95.0, 90.0)
    couplage: Optional[Tuple[float, float]] = (-55.0, 100.0)    # bornes de (q3 - q2)

    #: en deca de cette marge, la butee est FROLEE : la pose reste admissible
    #: pour le modele, mais une machine reelle la refuse souvent. Ses butees
    #: ne sont jamais exactement celles de la table, et un zero mal repris
    #: suffit a faire basculer les quelques degres restants.
    marge_alerte: float = 5.0

    # Limites cinematiques utilisees pour dimensionner les trajectoires.
    v_max: Tuple[float, float, float, float] = (200.0, 200.0, 200.0, 300.0)  # deg/s
    a_max: Tuple[float, float, float, float] = (500.0, 500.0, 500.0, 800.0)  # deg/s^2
    v_cart_max: float = 250.0     # mm/s
    a_cart_max: float = 800.0     # mm/s^2


# ---------------------------------------------------------------------------

class DobotMagician:
    """Modele geometrique direct, inverse et differentiel du Dobot Magician."""

    NOMS_ARTICULATIONS = ("J1 base", "J2 bras arriere", "J3 avant-bras", "J4 outil")
    NOMS_REPERES = ("R0 base", "R1 colonne", "R2 epaule", "R3 coude",
                    "R4 poignet", "R5 axe outil", "R6 TCP")

    def __init__(self,
                 geometrie: Optional[Geometrie] = None,
                 limites: Optional[Limites] = None,
                 encombrement: Optional[Encombrement] = None):
        self.geo = geometrie or Geometrie()
        self.lim = limites or Limites()
        self.enc = encombrement or Encombrement()
        self.q_home = np.array([0.0, 15.0, 30.0, 0.0])
        #: repere outil explicite (axe 5 -> TCP). Laisse a None, il est
        #: deduit de geo.lt : Tz(-lt).Rx(180). Le renseigner permet un
        #: deport lateral et une rotation propre de l'outil.
        self._repere_outil: Optional[np.ndarray] = None

    # -- repere outil --------------------------------------------------------

    @property
    def repere_outil(self) -> np.ndarray:
        """Transformation axe outil -> TCP, en matrice homogene."""
        if self._repere_outil is not None:
            return self._repere_outil.copy()
        return hg.composer(hg.translation(0, 0, -self.geo.lt), hg.rot_x(np.pi))

    def definir_repere_outil(self, T=None, tolerance_deg: float = 1.0) -> None:
        """
        Fixe le repere outil a partir d'une matrice homogene 4x4.

        `T = None` revient au comportement par defaut, deduit de `geo.lt`.
        La matrice est validee : une inclinaison est refusee.
        """
        if T is None:
            self._repere_outil = None
            return
        verifier_repere_outil(T, tolerance_deg)
        self._repere_outil = np.asarray(T, dtype=float).copy()


    @property
    def outil_est_par_defaut(self) -> bool:
        return self._repere_outil is None

    # -- Modele geometrique direct -----------------------------------------

    def matrices_liaisons(self, q_deg: Sequence[float]) -> List[np.ndarray]:
        """
        Matrices de passage ELEMENTAIRES T01, T12, T23, T34, T45, T5T.

        C'est le detail pedagogique du MGD : le produit de ces six matrices
        donne la situation de l'outil dans le repere de base.
        """
        q1, q2, q3, q4 = (float(v) * D2R for v in q_deg)
        g = self.geo
        return [
            hg.rot_z(q1),
            hg.composer(hg.translation(0, 0, g.d1), hg.rot_y(q2)),
            hg.composer(hg.translation(0, 0, g.l1), hg.rot_y(q3 - q2 + np.pi / 2)),
            hg.composer(hg.translation(0, 0, g.l2), hg.rot_y(-np.pi / 2 - q3)),
            hg.composer(hg.translation(g.l3, 0, 0), hg.rot_z(q4)),
            self.repere_outil,
        ]

    def reperes(self, q_deg: Sequence[float]) -> List[np.ndarray]:
        """
        Situations ABSOLUES de tous les reperes : [T00, T01, T02, ..., T0TCP].

        Utilise pour l'affichage 3D (placement des maillages et des trieds).
        """
        out = [hg.identite()]
        T = hg.identite()
        for M in self.matrices_liaisons(q_deg):
            T = T @ M
            out.append(T)
        return out

    def mgd(self, q_deg: Sequence[float]) -> np.ndarray:
        """Modele geometrique direct : q (degres) -> matrice homogene base->TCP."""
        T = hg.identite()
        for M in self.matrices_liaisons(q_deg):
            T = T @ M
        return T

    # alias anglais
    fk = mgd

    def pose(self, q_deg: Sequence[float]) -> np.ndarray:
        """
        Situation de l'outil au format Dobot : (x, y, z, r).

        x, y, z en mm dans le repere de base ; r en degres = J1 + J4,
        rotation absolue de l'outil autour de la verticale.
        """
        q1, q2, q3, q4 = (float(v) for v in q_deg)
        g = self.geo
        if self.outil_est_par_defaut:
            # forme fermee, la plus rapide : l'outil est sur l'axe J4
            r_h = g.l1 * np.sin(q2 * D2R) + g.l2 * np.cos(q3 * D2R) + g.l3
            z = g.d1 + g.l1 * np.cos(q2 * D2R) - g.l2 * np.sin(q3 * D2R) - g.lt
            return np.array([r_h * np.cos(q1 * D2R), r_h * np.sin(q1 * D2R),
                             z, q1 + q4], dtype=float)
        # repere outil quelconque : on passe par la chaine complete
        T = self.mgd(q_deg)
        p = hg.position_de(T)
        return np.array([p[0], p[1], p[2],
                         np.degrees(np.arctan2(T[1, 0], T[0, 0]))], dtype=float)

    def positions_articulations(self, q_deg: Sequence[float]) -> np.ndarray:
        """Positions 3D des centres articulaires, forme (7, 3). Utile pour le squelette."""
        return np.array([hg.position_de(T) for T in self.reperes(q_deg)])

    # -- Modele geometrique inverse ----------------------------------------

    @staticmethod
    def _recentrer(angle: float) -> float:
        """Ramene un angle en radians dans ]-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def mgi_pose(self, x: float, y: float, z: float, r: float = 0.0,
                 tolerant: bool = False, coude: str = "auto") -> np.ndarray:
        """
        Modele geometrique inverse a partir de (x, y, z, r).

        Le triangle epaule - coude - poignet admet DEUX solutions, selon le
        cote de la droite epaule-poignet ou se place le coude. En notant
        gamma = 90 + q3 - q2 la flexion du coude, la solution physique est
        celle de signe sin(gamma) : "coude haut" pour gamma dans ]0, 180[,
        "coude bas" au-dela.

        Les butees du Magician autorisent gamma de -5 a 185 deg : la solution
        coude haut couvre donc la quasi-totalite de l'espace de travail, mais
        deux minces bandes d'hyperextension et d'hyperflexion ne sont
        atteignables qu'avec l'autre. Le mode "auto" (defaut) prend la
        solution coude haut et bascule sur l'autre uniquement si la premiere
        viole les butees. Aux deux singularites (bras tendu, gamma = 0, et
        bras replie, gamma = 180) les deux solutions se confondent : le
        raccordement est continu.

        `coude="haut"` ou `"bas"` force une branche.
        `tolerant=True` renvoie la solution la plus proche au lieu de lever
        une exception quand la cible est legerement hors d'atteinte.
        """
        g = self.geo

        # On retire d'abord le repere outil : le triangle se resout sur
        # l'AXE OUTIL (repere 5), pas sur la pointe. Avec le repere par
        # defaut cela revient exactement a remonter de lt.
        T5 = cible(x, y, z, r) @ hg.inverse(self.repere_outil)
        p5 = T5[:3, 3]
        psi = np.arctan2(T5[1, 0], T5[0, 0])     # orientation de l'axe outil

        q1 = np.arctan2(p5[1], p5[0])
        rho = float(np.hypot(p5[0], p5[1]))
        rp = rho - g.l3                 # projection radiale poignet
        zp = p5[2] - g.d1               # hauteur du poignet / epaule

        D = float(np.hypot(rp, zp))
        d_min, d_max = abs(g.l1 - g.l2), g.l1 + g.l2
        # tolerance sub-micronique : bras exactement tendu ou exactement
        # replie, l'arrondi flottant fait sortir D de l'intervalle
        tol = 1e-6
        if D > d_max + tol or D < d_min - tol:
            if not tolerant:
                raise ErreurCinematique(
                    "point ({:.1f}, {:.1f}, {:.1f}) hors d'atteinte : "
                    "distance epaule-poignet = {:.1f} mm, admissible [{:.1f} ; {:.1f}]"
                    .format(x, y, z, D, d_min, d_max))
        D = float(np.clip(D, d_min, d_max))

        # Loi des cosinus dans le triangle epaule - coude - poignet
        cos_beta = (g.l1 ** 2 + D ** 2 - g.l2 ** 2) / (2.0 * g.l1 * D)
        beta = np.arccos(float(np.clip(cos_beta, -1.0, 1.0)))
        phi = np.arctan2(rp, zp)        # angle epaule->poignet depuis la verticale
        q4 = self._recentrer(psi - q1)

        def solution(signe: float) -> np.ndarray:
            # Le recentrage est indispensable : bras completement replie,
            # beta vaut pi et phi - beta sort de ]-pi, pi].
            q2 = self._recentrer(phi + signe * beta)
            ex = g.l1 * np.sin(q2)
            ez = g.l1 * np.cos(q2)
            q3 = self._recentrer(np.arctan2(ez - zp, rp - ex))
            return np.array([q1, q2, q3, q4]) * R2D

        if coude == "haut":
            return solution(-1.0)
        if coude == "bas":
            return solution(+1.0)

        haut = solution(-1.0)
        if self.dans_butees(haut):
            return haut
        bas = solution(+1.0)
        return bas if self.dans_butees(bas) else haut

    def mgi(self, T, tolerant: bool = False, coude: str = "auto") -> np.ndarray:
        """
        Modele geometrique inverse a partir d'une matrice homogene.

        Seules la position et la rotation autour de la verticale sont
        exploitables : l'orientation de l'outil est imposee par la mecanique.
        """
        T = np.asarray(T, dtype=float)
        p = hg.position_de(T)
        R = hg.rotation_de(T)
        # R attendue = Rz(r) . Rx(pi) : la premiere colonne donne directement r
        r = np.arctan2(R[1, 0], R[0, 0]) * R2D
        return self.mgi_pose(p[0], p[1], p[2], r, tolerant=tolerant, coude=coude)

    ik = mgi

    def pose_vers_matrice(self, x: float, y: float, z: float, r: float = 0.0,
                          repere: Optional[np.ndarray] = None) -> np.ndarray:
        """(x, y, z, r) -> matrice homogene de l'outil (axe z dirige vers le bas)."""
        return cible(x, y, z, r, repere)

    @staticmethod
    def matrice_vers_pose(T, tolerance_deg: float = 1.0) -> np.ndarray:
        """Matrice homogene -> (x, y, z, r), avec verification de faisabilite."""
        return decomposer_cible(T, tolerance_deg)

    # -- Modele cinematique (jacobienne) -----------------------------------

    def jacobienne(self, q_deg: Sequence[float]) -> np.ndarray:
        """
        Jacobienne analytique 4x4 :  [vx, vy, vz, wz]^T = J . [dq1..dq4]^T

        Entree en degres, mais J relie des vitesses articulaires en RAD/s a
        des vitesses en mm/s et rad/s.
        """
        q1, q2, q3, _q4 = (float(v) * D2R for v in q_deg)
        g = self.geo
        c1, s1 = np.cos(q1), np.sin(q1)
        r_h = g.l1 * np.sin(q2) + g.l2 * np.cos(q3) + g.l3
        return np.array([
            [-r_h * s1, g.l1 * np.cos(q2) * c1, -g.l2 * np.sin(q3) * c1, 0.0],
            [r_h * c1, g.l1 * np.cos(q2) * s1, -g.l2 * np.sin(q3) * s1, 0.0],
            [0.0, -g.l1 * np.sin(q2), -g.l2 * np.cos(q3), 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def jacobienne_numerique(self, q_deg: Sequence[float], h: float = 1e-6) -> np.ndarray:
        """Jacobienne par differences finies centrees. Sert a valider l'analytique."""
        q = np.asarray(q_deg, dtype=float)
        J = np.zeros((4, 4))
        for j in range(4):
            dq = np.zeros(4)
            dq[j] = h * R2D          # perturbation de h radians, exprimee en degres
            p_plus = self.pose(q + dq)
            p_moins = self.pose(q - dq)
            d = (p_plus - p_moins) / (2.0 * h)
            d[3] *= D2R              # la 4e composante est un angle en degres
            J[:, j] = d
        return J

    def manipulabilite(self, q_deg: Sequence[float]) -> float:
        """Mesure de Yoshikawa sqrt(det(J J^T)). Tend vers 0 pres des singularites."""
        J = self.jacobienne(q_deg)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def est_singulier(self, q_deg: Sequence[float], seuil: float = 1e3) -> bool:
        """Vrai au voisinage d'une singularite (bras tendu / replie, TCP sur l'axe)."""
        return self.manipulabilite(q_deg) < seuil

    # -- Butees et espace de travail ---------------------------------------

    def depassements(self, q_deg: Sequence[float]) -> List[str]:
        """Liste des butees violees. Vide si la configuration est admissible."""
        msgs = []
        for i, v in enumerate(q_deg):
            lo, hi = self.lim.q_min[i], self.lim.q_max[i]
            if v < lo - 1e-6 or v > hi + 1e-6:
                msgs.append("{} = {:.2f} deg hors de [{:.0f} ; {:.0f}]"
                            .format(self.NOMS_ARTICULATIONS[i], v, lo, hi))
        if self.lim.couplage is not None:
            lo, hi = self.lim.couplage
            d = q_deg[2] - q_deg[1]
            if d < lo - 1e-6 or d > hi + 1e-6:
                msgs.append("couplage J3-J2 = {:.2f} deg hors de [{:.0f} ; {:.0f}]"
                            .format(d, lo, hi))
        return msgs

    def dans_butees(self, q_deg: Sequence[float]) -> bool:
        return not self.depassements(q_deg)

    def marges(self, q_deg: Sequence[float]) -> np.ndarray:
        """Distance de chaque articulation a sa butee la plus proche, en degres."""
        q = np.asarray(q_deg, dtype=float)
        return np.minimum(q - np.asarray(self.lim.q_min, dtype=float),
                          np.asarray(self.lim.q_max, dtype=float) - q)

    def butees_frolees(self, q_deg: Sequence[float],
                       marge: Optional[float] = None) -> List[str]:
        """
        Articulations qui passent tout pres d'une butee sans la franchir.

        Le modele les accepte -- elles sont dans les bornes -- mais une
        machine reelle les refuse souvent : ses butees ne sont jamais
        exactement celles de la table, et une prise d'origine approximative
        suffit a consommer les quelques degres restants.

        Retour vide si tout est au large. Ce n'est PAS un defaut bloquant,
        seulement un avertissement.
        """
        seuil = float(self.lim.marge_alerte if marge is None else marge)
        if seuil <= 0.0:
            return []
        msgs = []
        m = self.marges(q_deg)
        for i, v in enumerate(m):
            if 0.0 <= v < seuil:
                msgs.append("{} a {:.1f} deg de sa butee [{:.0f} ; {:.0f}] "
                            "(marge d'alerte {:.0f} deg)"
                            .format(self.NOMS_ARTICULATIONS[i], v,
                                    self.lim.q_min[i], self.lim.q_max[i], seuil))
        return msgs

    # -- Collisions ---------------------------------------------------------

    def points_mobiles(self, q_deg: Sequence[float],
                       pas: Optional[float] = None) -> np.ndarray:
        """
        Echantillonne le squelette MOBILE : coude, poignet, axe outil, TCP.

        Le bras arriere (epaule -> coude) est volontairement exclu : il est
        solidaire de la colonne et la butee J2 l'empeche d'atteindre aussi
        bien la table que l'embase. L'inclure ferait signaler l'epaule
        elle-meme, qui vit legitimement a l'interieur de la colonne.
        """
        pas = float(pas if pas is not None else self.enc.pas)
        coins = [M[:3, 3] for M in self.reperes(q_deg)[3:]]
        pts = [np.asarray(coins[0], dtype=float)]
        for a, b in zip(coins[:-1], coins[1:]):
            d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
            n = max(int(np.linalg.norm(d) / pas), 1)
            t = np.linspace(0.0, 1.0, n + 1)[1:]
            pts.extend(np.asarray(a, dtype=float) + np.outer(t, d))
        return np.asarray(pts, dtype=float)

    #: Maillons mobiles : (nom, repere de debut, repere de fin). Les
    #: paires ADJACENTES se touchent par construction a leur articulation
    #: commune ; seules les autres peuvent reellement interferer.
    MAILLONS = (("bras arriere", 2, 3), ("avant-bras", 3, 4),
                ("deport du poignet", 4, 5), ("outil", 5, 6))
    PAIRES_MAILLONS = ((0, 2), (0, 3), (1, 3))

    @staticmethod
    def _distance_segments(p1, q1, p2, q2) -> float:
        """Distance minimale entre deux segments de droite, en 3D."""
        d1, d2, r = q1 - p1, q2 - p2, p1 - p2
        a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
        c, b = float(d1 @ r), float(d1 @ d2)
        den = a * e - b * b
        s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
        t = (b * s + f) / e if e > 1e-12 else 0.0
        tc = float(np.clip(t, 0.0, 1.0))
        if tc != t and a > 1e-12:
            s = float(np.clip((b * tc - c) / a, 0.0, 1.0))
        return float(np.linalg.norm((p1 + s * d1) - (p2 + tc * d2)))

    def auto_collisions(self, q_deg: Sequence[float],
                        marge: float = 0.0) -> List[str]:
        """
        Le bras se heurte-t-il LUI-MEME ?

        Chaque maillon est traite comme une capsule -- un segment entoure
        d'un rayon -- et l'on mesure la distance entre les paires non
        adjacentes.

        Le cas qui compte sur ce robot : quand l'avant-bras se replie, le
        DEPORT DU POIGNET revient vers le bras arriere. Le contact se
        produit a `J3 - J2 = 77 deg`, et le jeu ne depend que de cette
        difference -- consequence directe du parallelogramme. C'est la
        raison d'etre de `Limites.couplage`, que ce controle geometrique
        rend inutile a regler a la main.
        """
        rayons = self.enc.rayons_maillons
        P = [M[:3, 3] for M in self.reperes(q_deg)]
        msgs = []
        for i, j in self.PAIRES_MAILLONS:
            nom_i, a, b = self.MAILLONS[i]
            nom_j, c, d = self.MAILLONS[j]
            if (np.linalg.norm(P[b] - P[a]) < 1e-9
                    or np.linalg.norm(P[d] - P[c]) < 1e-9):
                continue                 # maillon de longueur nulle
            jeu = (self._distance_segments(P[a], P[b], P[c], P[d])
                   - rayons[i] - rayons[j])
            if jeu < marge - 1e-9:
                msgs.append("auto-collision : {} contre {}, {}"
                            .format(nom_i, nom_j,
                                    "penetration de {:.1f} mm".format(-jeu)
                                    if jeu < 0 else
                                    "jeu de {:.1f} mm sous la marge de {:.1f}"
                                    .format(jeu, marge)))
        return msgs

    def collisions(self, q_deg: Sequence[float],
                   marge: float = 0.0) -> List[str]:
        """
        Interferences : plan de pose, embase, colonne, et le bras avec
        LUI-MEME.

        Retourne une liste de messages, vide si la pose est saine -- meme
        forme que `depassements`, pour pouvoir les enchainer.

        `marge` en millimetres declenche l'alerte AVANT le contact ; utile
        pour se garder une distance de securite.

        Les trois interferences sont reellement atteignables a l'interieur
        des butees : la table des que l'on monte un outil, l'embase et la
        colonne par la geometrie du bras replie. Voir `Encombrement` pour
        ce que ce modele ne couvre pas.
        """
        e = self.enc
        P = self.points_mobiles(q_deg)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        rho = np.hypot(x, y)
        msgs = []

        def signaler(nom, contact, profondeur, forme):
            """Distingue le contact reel de la simple entree dans la marge."""
            if contact:
                msgs.append("{} : penetration de {:.1f} mm ({})"
                            .format(nom, profondeur, forme))
            else:
                msgs.append("{} : approche a {:.1f} mm, sous la marge de "
                            "securite de {:.1f} mm ({})"
                            .format(nom, profondeur, marge, forme))

        # plan de pose
        k = int(np.argmin(z))
        if z[k] < e.plan_z + marge - 1e-6:
            if z[k] < e.plan_z - 1e-6:
                msgs.append("plan de pose traverse de {:.1f} mm "
                            "(point le plus bas a z = {:.1f} mm)"
                            .format(e.plan_z - z[k], z[k]))
            else:
                msgs.append("plan de pose approche a {:.1f} mm, sous la marge "
                            "de securite de {:.1f} mm"
                            .format(z[k] - e.plan_z, marge))

        # embase : boite carree
        cote = np.maximum(np.abs(x), np.abs(y))
        dans = (cote < e.embase_demi + marge) & (z < e.embase_haut + marge)
        if dans.any():
            contact = bool(((cote < e.embase_demi) & (z < e.embase_haut)).any())
            d = float(np.min(np.maximum(cote[dans] - e.embase_demi,
                                        z[dans] - e.embase_haut)))
            signaler("embase", contact, abs(d),
                     "boite {:.0f} x {:.0f} mm, haute de {:.0f} mm"
                     .format(2 * e.embase_demi, 2 * e.embase_demi,
                             e.embase_haut))

        # colonne : cylindre
        dans = (rho < e.colonne_rayon + marge) & (z < e.colonne_haut + marge)
        if dans.any():
            contact = bool(((rho < e.colonne_rayon)
                            & (z < e.colonne_haut)).any())
            d = float(np.min(np.maximum(rho[dans] - e.colonne_rayon,
                                        z[dans] - e.colonne_haut)))
            signaler("colonne", contact, abs(d),
                     "rayon {:.1f} mm, hauteur {:.0f} mm"
                     .format(e.colonne_rayon, e.colonne_haut))

        # le bras contre lui-meme : ce n'est pas un obstacle fixe, mais
        # c'est bien une interference, et elle est atteignable dans les
        # butees -- le deport du poignet revient sur le bras arriere
        msgs.extend(self.auto_collisions(q_deg, marge))
        return msgs

    def sans_collision(self, q_deg: Sequence[float]) -> bool:
        return not self.collisions(q_deg)

    def defauts(self, q_deg: Sequence[float]) -> List[str]:
        """Butees violees ET collisions, dans cet ordre."""
        return self.depassements(q_deg) + self.collisions(q_deg)

    def saturer(self, q_deg: Sequence[float]) -> np.ndarray:
        """Ramene chaque articulation dans ses butees."""
        return np.clip(np.asarray(q_deg, dtype=float), self.lim.q_min, self.lim.q_max)

    def atteignable(self, x: float, y: float, z: float, r: float = 0.0) -> bool:
        """Vrai si (x, y, z, r) est atteignable butees comprises."""
        try:
            q = self.mgi_pose(x, y, z, r)
        except ErreurCinematique:
            return False
        return self.dans_butees(q)

    def enveloppe(self, n_q2: int = 40, n_q3: int = 40) -> np.ndarray:
        """
        Contour de l'espace de travail dans le plan meridien (rho, z).

        Renvoie un nuage (N, 2) : utile pour tracer l'enveloppe accessible.
        """
        q2 = np.linspace(self.lim.q_min[1], self.lim.q_max[1], n_q2)
        q3 = np.linspace(self.lim.q_min[2], self.lim.q_max[2], n_q3)
        Q2, Q3 = np.meshgrid(q2, q3)
        g = self.geo
        rho = g.l1 * np.sin(Q2 * D2R) + g.l2 * np.cos(Q3 * D2R) + g.l3
        z = g.d1 + g.l1 * np.cos(Q2 * D2R) - g.l2 * np.sin(Q3 * D2R) - g.lt
        return np.column_stack([rho.ravel(), z.ravel()])

    # -- Utilitaires --------------------------------------------------------

    def duree_minimale_articulaire(self, q0, q1) -> float:
        """Duree minimale d'un mouvement articulaire compte tenu de v_max."""
        dq = np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))
        return float(np.max(dq / np.asarray(self.lim.v_max, dtype=float)))

    def duree_minimale_cartesienne(self, T0, T1) -> float:
        """Duree minimale d'un mouvement rectiligne compte tenu de v_cart_max."""
        d = float(np.linalg.norm(hg.position_de(T1) - hg.position_de(T0)))
        return d / self.lim.v_cart_max

    def avec_outil(self, deport_vertical: float) -> "DobotMagician":
        """Copie du robot avec une pointe d'outil decalee verticalement (mm)."""
        return DobotMagician(replace(self.geo, lt=float(deport_vertical)),
                             self.lim, self.enc)

    def __repr__(self) -> str:
        g = self.geo
        return ("DobotMagician(d1={:.1f}, l1={:.1f}, l2={:.1f}, l3={:.1f}, lt={:.1f})"
                .format(g.d1, g.l1, g.l2, g.l3, g.lt))
