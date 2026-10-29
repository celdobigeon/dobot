# -*- coding: utf-8 -*-
"""
Le convoyeur : sa geometrie, sa cadence, ses pieces et son capteur.

Le tapis est entraine par un moteur pas-a-pas branche sur `Stepper1` ou
`Stepper2` du Magician. Le controleur ne connait que des IMPULSIONS PAR
SECONDE ; tout le reste du simulateur raisonne en millimetres. La
conversion vit ici, et nulle part ailleurs.

    from dobot_sim import Convoyeur, repere_travail

    TAPIS = Convoyeur(repere=repere_travail(x=180, y=-180, z=0, rz=90),
                      longueur=600, largeur=60)
    TAPIS.impulsions_par_seconde(50)      # 50 mm/s -> impulsions/s

Le pas du tapis
---------------
Les valeurs par defaut sont celles de la demonstration officielle Dobot :
moteur 200 pas par tour (1,8 deg), 16 micro-pas, reducteur 10, poulie de
36 mm de diametre. Soit 32 000 impulsions par tour pour 113,10 mm de
tapis, c'est-a-dire **282,9 impulsions par millimetre**.

Ce sont des valeurs NOMINALES, au meme titre que la longueur d'un outil :
une poulie differente, un autre reducteur, et le compte change. Mesurez
le votre -- marquez le tapis, commandez 10 000 impulsions, mesurez le
deplacement -- et passez `impulsions_par_mm` au constructeur. Le reste du
programme suit.

Le repere du tapis
------------------
`repere` place le tapis sur la table, comme un repere de travail : son
origine est l'ENTREE du tapis, son axe x suit le sens de marche, son axe
y traverse la bande, son axe z monte. Une piece posee a l'abscisse s se
trouve donc en `repere @ translation(s, decalage, 0)`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import homogeneous as hg

__all__ = ["Convoyeur", "Piece", "IMPULSIONS_PAR_MM", "PORTS_GP"]

#: Ports GP du Magician, ou se branchent les capteurs du kit convoyeur.
#: Les valeurs viennent du SDK Dobot ; le protocole V1.1.5 nomme le type
#: `Port` sans jamais l'enumerer, et DobotDll.cpp ne porte pas son
#: en-tete. A confirmer sur la machine si un capteur reste muet.
PORTS_GP = {"GP1": 0, "GP2": 1, "GP4": 2, "GP5": 3}

#: Pas du tapis, en impulsions par millimetre (voir l'en-tete du module).
PAS_MOTEUR = 200.0          # pas par tour, moteur 1,8 deg
MICRO_PAS = 16.0            # micro-pas du pilote
REDUCTION = 10.0            # reducteur du kit convoyeur
DIAMETRE_POULIE = 36.0      # mm
IMPULSIONS_PAR_MM = (PAS_MOTEUR * MICRO_PAS * REDUCTION
                     / (np.pi * DIAMETRE_POULIE))


@dataclass
class Piece:
    """
    Une piece posee sur le tapis.

    `s` est son abscisse le long du tapis au moment `t0`, en millimetres
    depuis l'entree ; `y` son decalage lateral par rapport a l'axe.
    """

    s: float = 0.0
    y: float = 0.0
    t0: float = 0.0
    nom: str = ""
    cote: float = 25.0                    # taille apparente, pour le dessin
    emportee: bool = False                # prise par le robot : elle quitte


@dataclass
class Convoyeur:
    """
    Un tapis, sa cadence et ce qu'il transporte.

    La vitesse est pilotee par une suite de changements dates --
    `(instant, vitesse)` -- exactement comme le programme les produit. Le
    deplacement du tapis a un instant donne s'en deduit par integration,
    ce qui permet de rejouer, reculer et faire du pas a pas sans
    accumuler d'erreur.
    """

    repere: np.ndarray = field(
        default_factory=lambda: hg.translation(180.0, -180.0, 0.0))
    longueur: float = 600.0               # mm, de l'entree a la sortie
    largeur: float = 60.0                 # mm, largeur utile de la bande
    index_moteur: int = 0                 # 0 = Stepper1, 1 = Stepper2
    sens: int = 1                         # +1 ou -1 : sens de marche
    impulsions_par_mm: float = IMPULSIONS_PAR_MM

    #: abscisse du capteur de presence le long du tapis, en mm
    capteur_s: float = 450.0
    #: demi-largeur de la fenetre vue par le capteur, en mm
    capteur_fenetre: float = 12.0
    #: port GP du capteur infrarouge ; None si le capteur est cable sur
    #: une entree EIO ordinaire (il se lit alors comme n'importe quelle
    #: entree, par `attendre_entree`)
    capteur_port: Optional[int] = PORTS_GP["GP2"]
    #: port EIO, quand le capteur est cable en tout-ou-rien
    capteur_eio: Optional[int] = None

    #: abscisse du champ de la camera, et son diametre apparent
    camera_s: float = 300.0
    camera_champ: float = 80.0

    #: Distance parcourue APRES l'ordre d'arret, en mm, et la vitesse a
    #: laquelle elle a ete mesuree. Le tapis ne s'arrete pas net : le
    #: moteur decelere, et la piece continue.
    #:
    #: ATTENTION, c'est invisible a la mesure naive. Chronometrer la
    #: distance totale entre « marche » et « arret » ne le montre PAS :
    #: la rampe de demarrage perd exactement ce que la rampe d'arret
    #: gagne, et le total tombe juste. Il faut mesurer le depassement
    #: SEUL, apres l'ordre d'arret.
    depassement: float = 0.0
    vitesse_depassement: float = 60.0
    #: "retard" : la piece avance d'un temps fixe, depassement ~ vitesse.
    #: "rampe"  : deceleration constante, depassement ~ vitesse au carre.
    #: Les deux coincident a la vitesse de mesure et divergent ailleurs --
    #: d'un facteur trois entre 60 et 20 mm/s. `diag_depassement.py`
    #: mesure a plusieurs vitesses et tranche.
    loi_depassement: str = "retard"

    #: maillage OBJ du tapis, DEJA exprime dans le repere du Convoyeur --
    #: origine a l'entree, x dans le sens de marche, zero a la surface de
    #: la bande. `convertir_cao.py --tapis` produit exactement cela a
    #: partir du STEP du constructeur. None : le tapis est dessine en
    #: boites schematiques, ce qui suffit pour raisonner.
    maillage: Optional[str] = None

    pieces: List[Piece] = field(default_factory=list)

    #: changements de vitesse dates : (instant, vitesse en mm/s)
    cadence: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0)])

    # -- cadence ------------------------------------------------------------

    def impulsions_par_seconde(self, vitesse_mm_s: float) -> float:
        """Vitesse du tapis en mm/s -> consigne du controleur, signee."""
        return float(vitesse_mm_s) * self.impulsions_par_mm * self.sens

    def millimetres_par_seconde(self, impulsions_s: float) -> float:
        """L'inverse : ce que vaut une consigne du controleur, en mm/s."""
        return float(impulsions_s) / self.impulsions_par_mm * self.sens

    def regler(self, vitesse_mm_s: float, instant: float = 0.0) -> "Convoyeur":
        """Inscrit un changement de vitesse a l'instant donne."""
        instant = float(instant)
        # on remplace un reglage deja pose au meme instant
        self.cadence = [(t, v) for t, v in self.cadence if t < instant - 1e-9]
        self.cadence.append((instant, float(vitesse_mm_s)))
        self.cadence.sort(key=lambda c: c[0])
        return self

    def vitesse(self, t: float) -> float:
        """Vitesse du tapis a l'instant t, en mm/s."""
        instants = [c[0] for c in self.cadence]
        i = bisect.bisect_right(instants, float(t)) - 1
        return self.cadence[i][1] if i >= 0 else 0.0

    def deplacement(self, t: float) -> float:
        """
        Distance parcourue par le tapis entre 0 et t, en millimetres.

        Integrale de la vitesse : une suite de paliers, donc une somme de
        rectangles. Exacte, et independante du pas d'echantillonnage --
        c'est ce qui permet de reculer dans l'animation sans deriver.
        """
        t = float(t)
        total = 0.0
        for (t0, v), (t1, _) in zip(self.cadence, self.cadence[1:]):
            if t <= t0:
                break
            total += v * (min(t, t1) - t0)
        dernier_t, derniere_v = self.cadence[-1]
        if t > dernier_t:
            total += derniere_v * (t - dernier_t)
        return total

    # -- les pieces ---------------------------------------------------------

    def poser(self, s: float = 0.0, y: float = 0.0, t0: float = 0.0,
              nom: str = "", cote: float = 25.0) -> Piece:
        """Pose une piece sur le tapis, a l'abscisse s et a l'instant t0."""
        piece = Piece(s=float(s), y=float(y), t0=float(t0), nom=nom,
                      cote=float(cote))
        self.pieces.append(piece)
        return piece

    def abscisse(self, piece: Piece, t: float) -> float:
        """Ou se trouve une piece a l'instant t, en mm depuis l'entree."""
        return piece.s + self.deplacement(t) - self.deplacement(piece.t0)

    def situation(self, piece: Piece, t: float) -> np.ndarray:
        """La matrice 4x4 de la piece a l'instant t, dans le repere BASE."""
        return self.repere @ hg.translation(self.abscisse(piece, t),
                                            piece.y, 0.0)

    def sur_le_tapis(self, piece: Piece, t: float) -> bool:
        """La piece est-elle encore entre l'entree et la sortie ?"""
        if piece.emportee:
            return False
        return 0.0 <= self.abscisse(piece, t) <= self.longueur

    # -- le capteur et la camera --------------------------------------------

    def devant_capteur(self, t: float) -> Optional[Piece]:
        """La piece devant le capteur a l'instant t, s'il y en a une."""
        for piece in self.pieces:
            if not self.sur_le_tapis(piece, t):
                continue
            if abs(self.abscisse(piece, t) - self.capteur_s) <= self.capteur_fenetre:
                return piece
        return None

    def capteur(self, t: float) -> bool:
        """L'etat du capteur de presence a l'instant t."""
        return self.devant_capteur(t) is not None

    def devant_camera(self, t: float) -> List[Piece]:
        """Les pieces dans le champ de la camera a l'instant t."""
        demi = self.camera_champ / 2.0
        return [p for p in self.pieces
                if self.sur_le_tapis(p, t)
                and abs(self.abscisse(p, t) - self.camera_s) <= demi]

    def instant_au_capteur(self, piece: Piece, depart: float = 0.0,
                           limite: float = 600.0, pas: float = 0.01
                           ) -> Optional[float]:
        """
        Quand cette piece passera-t-elle devant le capteur ?

        Cherche le premier instant apres `depart` ou le capteur la voit.
        Rend None si elle n'y arrive pas dans le delai -- tapis a l'arret,
        ou piece deja passee.
        """
        t = float(depart)
        while t <= depart + limite:
            if (self.sur_le_tapis(piece, t)
                    and abs(self.abscisse(piece, t) - self.capteur_s)
                    <= self.capteur_fenetre):
                return t
            t += pas
        return None

    # -- geometrie, pour le dessin et les cibles ----------------------------

    def depassement_a(self, vitesse_mm_s: float) -> float:
        """
        De combien la piece depasse le capteur, a cette vitesse-la.

        Nul tant que le depassement n'a pas ete mesure : mieux vaut un
        modele qui avoue ignorer qu'un modele qui invente.
        """
        if not self.depassement:
            return 0.0
        v = abs(float(vitesse_mm_s))
        v0 = abs(float(self.vitesse_depassement)) or 1.0
        if self.loi_depassement == "rampe":
            return float(self.depassement) * (v / v0) ** 2
        return float(self.depassement) * (v / v0)

    def abscisse_arret(self, vitesse_mm_s: Optional[float] = None) -> float:
        """
        Ou la piece s'immobilise vraiment, le long du tapis.

        C'est l'abscisse du capteur PLUS le depassement. C'est la que le
        bras doit aller chercher la piece -- pas au droit du capteur,
        qui n'est que l'endroit ou on l'a vue passer.
        """
        if self.capteur_s is None:
            raise ValueError("aucun capteur declare sur ce tapis")
        v = self.vitesse_depassement if vitesse_mm_s is None else vitesse_mm_s
        return float(self.capteur_s) + self.depassement_a(v)

    def boites(self, t: float = 0.0) -> List[dict]:
        """
        De quoi DESSINER le tapis : une liste de boites orientees.

        Chaque boite donne `centre` (repere de la base), `axes` (colonnes
        = axes de la boite), `dims` et un `role` -- bande, bati, capteur,
        fenetre, champ, piece. La vue n'a plus qu'a les peindre.

        La forme vit ici plutot que dans le widget pour deux raisons :
        un objet sait decrire sa propre geometrie, et surtout cela se
        VERIFIE sans ouvrir de fenetre OpenGL.
        """
        R = np.asarray(self.repere, dtype=float)
        axes = R[:3, :3]
        L, l = float(self.longueur), float(self.largeur)

        def boite(centre, dims, role):
            c = (R @ np.append(np.asarray(centre, dtype=float), 1.0))[:3]
            return {"centre": c, "axes": axes, "dims": tuple(dims),
                    "role": role}

        epaisseur = 8.0
        formes = [boite((L / 2, 0.0, -epaisseur / 2), (L, l, epaisseur), "bande")]
        for cote in (-1, 1):
            formes.append(boite((L / 2, cote * (l / 2 + 6), 4.0),
                                (L, 12.0, 24.0), "bati"))

        if self.capteur_s is not None:
            s = float(self.capteur_s)
            formes.append(boite((s, l / 2 + 26, 26.0), (16.0, 16.0, 16.0),
                                "capteur"))
            # la fenetre surveillee, en travers de la bande : la voir
            # explique d'un coup qu'une piece trop rapide la traverse
            # entre deux lectures
            formes.append(boite((s, 0.0, 1.0),
                                (2 * float(self.capteur_fenetre), l, 2.0),
                                "fenetre"))

        if self.camera_s is not None:
            champ = float(self.camera_champ)
            formes.append(boite((float(self.camera_s), 0.0, 1.0),
                                (champ, min(champ, l), 2.0), "champ"))

        for piece in self.pieces:
            if not self.sur_le_tapis(piece, t):
                continue
            cote = float(piece.cote)
            formes.append(boite((self.abscisse(piece, t), piece.y, cote / 2),
                                (cote, cote, cote), "piece"))
        return formes

    def point(self, s: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        """Un point du tapis, en coordonnees de la BASE du robot."""
        return (self.repere @ hg.translation(float(s), float(y),
                                             float(z)))[:3, 3]

    def cible(self, s: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        """
        Une cible OUTIL au-dessus du tapis : axe z vers le bas.

        De quoi aller prendre une piece sans quitter les matrices
        homogenes : `robot.movl(TAPIS.cible(450, 0, 5))`.
        """
        from .robot import cible as _cible
        return _cible(x=float(s), y=float(y), z=float(z), repere=self.repere)

    def resume(self) -> str:
        """Ce que le tapis est, en quelques lignes."""
        p = self.point(0), self.point(self.longueur)
        return ("convoyeur : {:.0f} x {:.0f} mm, de ({:.0f}, {:.0f}) a "
                "({:.0f}, {:.0f})\n"
                "  moteur Stepper{}, sens {:+d}, {:.1f} impulsions/mm\n"
                "  capteur a {:.0f} mm{}, camera a {:.0f} mm\n"
                "  {} piece(s) posee(s)"
                .format(self.longueur, self.largeur,
                        p[0][0], p[0][1], p[1][0], p[1][1],
                        self.index_moteur + 1, self.sens,
                        self.impulsions_par_mm, self.capteur_s,
                        "" if self.capteur_eio is None
                        else " (entree EIO {})".format(self.capteur_eio),
                        self.camera_s, len(self.pieces)))
