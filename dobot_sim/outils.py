"""
Catalogue des outils montables au bout du bras.

Un outil apporte trois choses :

  * une LONGUEUR, le deport vertical entre l'axe J4 et la pointe active.
    C'est elle qui alimente `Geometrie.lt` : changer d'outil change la
    signification de la coordonnee z, et donc le modele geometrique.
  * un MAILLAGE, attache au repere 5 (l'axe outil, qui tourne avec J4).
  * un ETAT binaire : ventouse aspirante ou non, pince ouverte ou fermee.

Les maillages sont PARAMETRIQUES et non importes de la CAO : les fichiers
STEP des outils fournis (`ventouse.stp`, `Support_outil.stp`) sont des
B-Rep exacts, sans facettes, et les tesseller demanderait un noyau
geometrique. Les dimensions retenues sont celles des outils Dobot, a
quelques millimetres pres ; elles restent modifiables ici meme, et la
longueur est de toute facon la seule grandeur qui entre dans le calcul.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .meshes import Piece, boite, cylindre, fusion

__all__ = ["Outil", "CATALOGUE", "outil", "noms_outils", "tronc_cone"]


def tronc_cone(rayon_bas: float, rayon_haut: float, hauteur: float,
               n: int = 24, centre=(0.0, 0.0, 0.0)):
    """Tronc de cone d'axe z, ferme aux deux bouts."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    h = hauteur / 2.0
    bas = np.zeros((n, 3), dtype=np.float32)
    haut = np.zeros((n, 3), dtype=np.float32)
    bas[:, 0], bas[:, 1], bas[:, 2] = rayon_bas * np.cos(th), rayon_bas * np.sin(th), -h
    haut[:, 0], haut[:, 1], haut[:, 2] = rayon_haut * np.cos(th), rayon_haut * np.sin(th), h
    V = np.vstack([bas, haut,
                   np.array([[0.0, 0.0, -h]], dtype=np.float32),
                   np.array([[0.0, 0.0, h]], dtype=np.float32)])
    V += np.asarray(centre, dtype=np.float32)
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, n + i, n + j], [i, n + j, j], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V.astype(np.float32), np.asarray(F, dtype=np.int32)


# ---------------------------------------------------------------------------

NOIR = (0.13, 0.14, 0.16, 1.0)
GRIS = (0.45, 0.47, 0.50, 1.0)
ROUGE = (0.80, 0.20, 0.18, 1.0)
ALU = (0.66, 0.68, 0.72, 1.0)


@dataclass
class Outil:
    """
    Un outil monte sur l'axe J4.

    `longueur` est le deport vertical de la pointe sous l'axe J4, en mm :
    c'est la valeur transmise a `Geometrie.lt`. Avec lt renseigne, la
    coordonnee z commandee designe la POINTE de l'outil, et non plus la
    bride : c'est ainsi que l'on vient effleurer la table.
    """
    nom: str
    libelle: str
    longueur: float
    fabrique: Callable[["Outil", bool], List[Tuple[np.ndarray, np.ndarray, tuple]]]
    commandable: bool = True          # possede un etat ON/OFF
    libelle_actif: str = "actif"
    libelle_inactif: str = "inactif"
    ouverture: float = 0.0            # ecartement des doigts, pince seulement
    description: str = ""

    def pieces(self, actif: bool = False) -> List[Piece]:
        """Maillages de l'outil, exprimes dans le repere 5 (axe outil)."""
        out = []
        for i, (V, F, couleur) in enumerate(self.fabrique(self, actif)):
            out.append(Piece("outil.{}.{}".format(self.nom, i), 5, V, F,
                             couleur=couleur))
        return out

    def etat_texte(self, actif: bool) -> str:
        return self.libelle_actif if actif else self.libelle_inactif


# ---------------------------------------------------------------------------
# Fabriques de maillages. Origine au niveau de l'axe J4, z vers le HAUT :
# l'outil descend donc vers les z negatifs.
# ---------------------------------------------------------------------------

def _aucun(o: Outil, actif: bool):
    # simple bride de fixation, pour que le bout du bras ne soit pas nu
    return [(*boite(30, 26, 10, (0, 0, -5)), ALU)]


def _ventouse(o: Outil, actif: bool):
    """Electrovanne, tige, puis la ventouse proprement dite."""
    corps = fusion(boite(28, 22, 26, (0, 0, -13)),          # bloc electrovanne
                   cylindre(5.0, 8, 2, 16, (0, 0, -30)))    # raccord
    tige = cylindre(3.5, 24, 2, 16, (0, 0, -42))
    # la ventouse s'ecrase legerement quand elle aspire
    h = 10.0 if not actif else 7.0
    cuvette = tronc_cone(11.0, 5.0, h, 24, (0, 0, -o.longueur + h / 2.0))
    return [(*corps, NOIR), (*tige, ALU),
            (*cuvette, ROUGE if actif else NOIR)]


def _pince(o: Outil, actif: bool):
    """Pince pneumatique a deux doigts. `actif` = pince fermee."""
    corps = fusion(boite(34, 26, 26, (0, 0, -13)),
                   boite(36, 20, 12, (0, 0, -32)))
    demi = (2.0 if actif else o.ouverture / 2.0 + 2.0)
    doigts = []
    for signe in (-1.0, 1.0):
        x = signe * (demi + 3.0)
        doigts.append(fusion(boite(6, 14, 34, (x, 0, -55)),
                             boite(6, 14, 8, (x - signe * 3.0, 0, -74))))
    return [(*corps, NOIR),
            (*doigts[0], ALU), (*doigts[1], ALU)]


def _stylo(o: Outil, actif: bool):
    """Porte-stylo : bague de serrage et corps cylindrique."""
    bague = fusion(boite(30, 26, 16, (0, 0, -8)),
                   cylindre(9.0, 34, 2, 20, (0, 0, -30)))
    corps = cylindre(5.0, 34, 2, 20, (0, 0, -56))
    pointe = tronc_cone(5.0, 0.8, 8.0, 16, (0, 0, -o.longueur + 4.0))
    return [(*bague, NOIR), (*corps, GRIS), (*pointe, NOIR)]


def _laser(o: Outil, actif: bool):
    corps = fusion(boite(26, 26, 40, (0, 0, -20)),
                   cylindre(7.0, 16, 2, 16, (0, 0, -48)))
    faisceau = cylindre(0.8, 30, 2, 10, (0, 0, -o.longueur - 13))
    out = [(*corps, NOIR)]
    if actif:
        out.append((*faisceau, ROUGE))
    return out


# ---------------------------------------------------------------------------

CATALOGUE: Dict[str, Outil] = {
    "aucun": Outil(
        nom="aucun", libelle="aucun (bride nue)", longueur=0.0,
        fabrique=_aucun, commandable=False,
        description="z designe l'axe du poignet, a 61 mm en avant du coude."),
    "ventouse": Outil(
        nom="ventouse", libelle="ventouse", longueur=62.0,
        fabrique=_ventouse,
        libelle_actif="aspiration", libelle_inactif="relachee",
        description="kit ventouse : electrovanne, tige et cuvette. "
                    "Commandee par SetEndEffectorSuctionCup."),
    "pince": Outil(
        nom="pince", libelle="pince", longueur=78.0,
        fabrique=_pince, ouverture=26.0,
        libelle_actif="fermee", libelle_inactif="ouverte",
        description="pince pneumatique a deux doigts. "
                    "Commandee par SetEndEffectorGripper."),
    "stylo": Outil(
        nom="stylo", libelle="porte-stylo", longueur=70.0,
        fabrique=_stylo, commandable=False,
        description="porte-stylo pour le trace. Aucun actionneur."),
    "laser": Outil(
        nom="laser", libelle="laser", longueur=56.0,
        fabrique=_laser,
        libelle_actif="allume", libelle_inactif="eteint",
        description="module de gravure. Commande par SetEndEffectorLaser. "
                    "ATTENTION au rayonnement sur le robot reel."),
}


def noms_outils() -> List[str]:
    return list(CATALOGUE.keys())


def outil(nom: str) -> Outil:
    """
    Recupere un outil du catalogue par son nom.

    Les longueurs du catalogue sont NOMINALES. Le montage reel s'en ecarte
    souvent, et c'est au programme de porter la valeur mesuree -- voir
    `Pilote.regler_longueur_outil`. On a essaye de la ranger dans un
    fichier de configuration lu ici : mauvaise idee. Cette fonction sert
    aussi au modele et aux tests, qui doivent rester reproductibles ; une
    mesure propre a une machine n'a rien a faire dans le catalogue.
    """
    cle = str(nom).lower()
    if cle not in CATALOGUE:
        raise ValueError("outil inconnu : {} (choix : {})"
                         .format(nom, ", ".join(CATALOGUE)))
    return CATALOGUE[cle]
