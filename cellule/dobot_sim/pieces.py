# -*- coding: utf-8 -*-
"""
Les pieces que le robot prend et depose, et leur animation.

    from dobot_sim import PieceLibre, piece_posee

    robot.pieces = [piece_posee(x=60, y=-220, dims=(125, 30, 10),
                                magasin=True)]

La fenetre les dessine, et les fait SUIVRE LA VENTOUSE pendant la
lecture : on voit la planche quitter le magasin, traverser, et se
poser.

Pourquoi ce n'est pas une simulation physique
----------------------------------------------
Parce que rien, dans cette fenetre, n'est accumule. La configuration
articulaire, les sorties, l'etat de l'outil, la bande du convoyeur :
tout est FONCTION DU TEMPS, recalcule a chaque image. C'est ce qui
permet de rembobiner, d'aller au milieu du programme, de faire du pas a
pas -- et c'est ce qu'une simulation pas a pas interdirait, puisqu'elle
derive des qu'on recule.

Une piece portee doit donc obeir a la meme regle. On calcule UNE FOIS,
au chargement du programme, l'histoire de chaque piece :

    posee ici           de t=0      a t=3,2 s
    portee par l'outil  de t=3,2 s  a t=7,8 s
    posee la            de t=7,8 s  a la fin

Ensuite, a l'instant t, sa pose SE LIT. Rembobiner la remet exactement
ou elle etait.

Le magasin qui se recharge
--------------------------
`magasin=True` : des que le robot emporte cette piece, une autre
reapparait au meme endroit. C'est ce que fait l'operateur dans la vraie
cellule -- il repose une planche a chaque cycle -- et cela evite de
dessiner douze planches empilees que le programme ne prendrait jamais
qu'une par une, au meme endroit.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import homogeneous as hg

__all__ = ["PieceLibre", "piece_posee", "Manutention"]


@dataclass
class PieceLibre:
    """
    Une piece posee sur la table, que le robot peut prendre et deposer.

    A ne pas confondre avec `convoyeur.Piece`, qui est portee par la
    bande et n'a qu'une abscisse : celle-ci a une pose complete, et
    peut aller n'importe ou.

    `pose` est le repere de la piece : son origine est son CENTRE
    geometrique, ses axes ceux de la boite (x = longueur, y = largeur,
    z = epaisseur, vers le haut).
    """

    pose: np.ndarray = field(default_factory=hg.identite)
    dims: Tuple[float, float, float] = (125.0, 30.0, 10.0)
    nom: str = ""
    #: remplacee des qu'on l'emporte -- voir l'en-tete du module
    magasin: bool = False
    #: sert a la couleur dans la vue 3D
    type: str = "piece"


def piece_posee(x: float = 0.0, y: float = 0.0, rz: float = 0.0,
                dims: Sequence[float] = (125.0, 30.0, 10.0),
                z_plan: float = 0.0, **kw) -> PieceLibre:
    """
    Une piece couchee a plat sur un plan, a la position donnee.

    Raccourci pour le cas courant. `z_plan` est la cote du plan qui la
    porte ; la piece repose dessus, donc son centre est une demi-
    epaisseur plus haut -- c'est le genre de demi-cote qu'on oublie, et
    qui se paie en voyant la planche a moitie enfoncee dans la table.
    """
    e = float(dims[2])
    pose = hg.composer(hg.translation(float(x), float(y), z_plan + e / 2.0),
                       hg.rot_z(np.radians(rz)))
    return PieceLibre(pose=pose, dims=tuple(float(d) for d in dims), **kw)


class Manutention:
    """
    L'histoire de toutes les pieces d'un programme, en fonction du temps.

    `preparer()` la calcule une fois ; `boites(t)` la relit.
    """

    #: a quelle distance de la ventouse une piece est consideree prise,
    #: en millimetres. Large : la cible de prise vise le DESSUS de la
    #: piece, dont le centre est une demi-epaisseur plus bas, et le
    #: programme peut viser a quelques millimetres pres.
    TOLERANCE = 45.0

    def __init__(self, pieces: Optional[Sequence[PieceLibre]] = None):
        self.pieces: List[PieceLibre] = list(pieces or [])
        self._instances: List[dict] = []
        self._trajectoire = None
        self._robot = None

    # -- calcul -------------------------------------------------------------

    def preparer(self, robot, trajectoire, programme) -> "Manutention":
        """
        Rejoue le programme et note quand chaque piece est prise, portee,
        reposee. A appeler quand le programme change, pas a chaque image.
        """
        self._robot = robot
        self._trajectoire = trajectoire
        self._instances = []
        if not self.pieces or trajectoire is None:
            return self

        for p in self.pieces:
            self._instances.append(self._neuve(p, 0.0))

        for t, prise in self._transitions(programme):
            T = self._pose_outil(t)
            if T is None:
                continue
            if prise:
                self._prendre(t, T)
            else:
                self._poser(t, T)
        return self

    @staticmethod
    def _neuve(source: PieceLibre, t: float) -> dict:
        return {"source": source, "dims": source.dims, "type": source.type,
                "nom": source.nom,
                "etapes": [(t, "posee", np.asarray(source.pose, dtype=float))]}

    @staticmethod
    def _transitions(programme) -> List[Tuple[float, bool]]:
        """Les instants ou l'outil CHANGE d'etat, dans l'ordre."""
        etat = bool(getattr(programme, "outil_initial", False))
        out = []
        for te, e in getattr(programme, "evenements_outil", []):
            if bool(e) != etat:
                etat = bool(e)
                out.append((float(te), etat))
        return out

    def _pose_outil(self, t: float) -> Optional[np.ndarray]:
        try:
            return self._robot.mgd(self._trajectoire.q(t))
        except Exception:
            return None

    def _prendre(self, t: float, T: np.ndarray) -> None:
        """La ventouse aspire : elle emporte la piece la plus proche."""
        centre_outil = T[:3, 3]
        candidate, meilleure = None, self.TOLERANCE
        for inst in self._instances:
            mode, data = inst["etapes"][-1][1], inst["etapes"][-1][2]
            if mode != "posee":
                continue
            d = float(np.linalg.norm(data[:3, 3] - centre_outil))
            if d < meilleure:
                candidate, meilleure = inst, d
        if candidate is None:
            return
        pose = candidate["etapes"][-1][2]
        # offset : la piece vue depuis l'outil. Fige tant qu'elle est
        # portee -- c'est ce qui la fait suivre le poignet, orientation
        # comprise.
        candidate["etapes"].append((t, "portee", np.linalg.inv(T) @ pose))
        source = candidate["source"]
        if source.magasin:
            self._instances.append(self._neuve(source, t))

    def _poser(self, t: float, T: np.ndarray) -> None:
        """La ventouse relache : la piece reste ou elle est."""
        for inst in self._instances:
            if inst["etapes"][-1][1] == "portee":
                offset = inst["etapes"][-1][2]
                inst["etapes"].append((t, "posee", T @ offset))

    # -- lecture ------------------------------------------------------------

    def pose(self, inst: dict, t: float) -> Optional[np.ndarray]:
        """Ou est cette piece a l'instant t, ou None si elle n'existe pas."""
        courant = None
        for t0, mode, data in inst["etapes"]:
            if t0 <= t + 1e-9:
                courant = (mode, data)
            else:
                break
        if courant is None:
            return None                       # pas encore apparue
        mode, data = courant
        if mode == "posee":
            return data
        T = self._pose_outil(t)
        return None if T is None else T @ data

    def boites(self, t: float = 0.0) -> List[dict]:
        """
        Les pieces a l'instant t, au format attendu par la vue 3D :
        {"centre", "axes", "dims", "type"}.
        """
        out = []
        for inst in self._instances:
            P = self.pose(inst, t)
            if P is None:
                continue
            out.append({"centre": tuple(float(v) for v in P[:3, 3]),
                        "axes": [[float(v) for v in ligne]
                                 for ligne in P[:3, :3]],
                        "dims": tuple(float(d) for d in inst["dims"]),
                        "type": inst["type"]})
        return out

    # -- transport ----------------------------------------------------------

    def vers_liste(self) -> List[dict]:
        """Les pieces DECLAREES, pour voyager avec le programme."""
        return [{"pose": [[float(v) for v in ligne] for ligne in p.pose],
                 "dims": list(p.dims), "nom": p.nom,
                 "magasin": bool(p.magasin), "type": p.type}
                for p in self.pieces]

    @classmethod
    def depuis_liste(cls, data: Optional[Sequence[dict]]) -> "Manutention":
        pieces = []
        for d in data or []:
            pieces.append(PieceLibre(
                pose=np.asarray(d.get("pose", np.eye(4)), dtype=float),
                dims=tuple(float(v) for v in d.get("dims", (125, 30, 10))),
                nom=d.get("nom", ""), magasin=bool(d.get("magasin", False)),
                type=d.get("type", "piece")))
        return cls(pieces)
