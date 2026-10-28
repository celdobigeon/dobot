"""
Programme robot : suite d'instructions editables, sauvegardables en JSON,
compilables en trajectoire simulee et exportables vers le robot reel.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .robot import (DobotMagician, cible, decomposer_cible,
                    verifier_repere_objet)
from .trajectory import (ProfilTroncons, SegmentAction, SegmentArticulaire,
                         SegmentAttente, SegmentCirculaire, SegmentLineaire,
                         Trajectoire, creer_profil, creer_profil_vitesse)

__all__ = ["Instruction", "Programme", "Evenement", "TYPES_INSTRUCTION",
           "TYPES_MOUVEMENT"]

TYPES_MOUVEMENT = ("MOVJ", "MOVL", "MOVC")
TYPES_INSTRUCTION = ("MOVJ", "MOVL", "MOVC", "ATTENTE", "OUTIL",
                     "SORTIE", "ENTREE", "CONVOYEUR")

#: duree fictive attribuee aux instructions instantanees, pour qu'elles
#: restent visibles sur la frise temporelle
DUREE_LOGIQUE = 0.25


@dataclass
class Evenement:
    """Changement d'etat date, produit par la compilation du programme."""
    t: float
    type: str            # OUTIL | SORTIE | ENTREE | CONVOYEUR
    etat: bool = False
    #: grandeur portee par l'evenement, quand il en porte une : la
    #: vitesse du tapis pour un CONVOYEUR, en mm/s
    valeur: float = 0.0
    port: int = 0
    segment: int = 0
    timeout: float = 0.0


#: valeurs par defaut de vitesse et acceleration, par espace d'interpolation
VITESSE_DEFAUT = {"deg": 90.0, "mm": 120.0}         # deg/s   et  mm/s
ACCELERATION_DEFAUT = {"deg": 300.0, "mm": 500.0}   # deg/s^2 et  mm/s^2


@dataclass
class Instruction:
    """
    Une ligne de programme.

    type        MOVJ    deplacement articulaire vers une cible
                MOVL    deplacement cartesien rectiligne
                MOVC    deplacement cartesien circulaire (via + cible)
                ATTENTE temporisation
                OUTIL   commande de l'outil monte (ventouse, pince, laser)
                CONVOYEUR vitesse du tapis, en mm/s
                SORTIE  met une sortie tout-ou-rien a l'etat demande
                ENTREE  attend qu'une entree passe a l'etat demande,
                        au plus pendant `duree` secondes

    mode        "articulaire" -> la cible est lue dans q
                "cartesien"   -> la cible est lue dans pose = (x, y, z, r)

    mode_temps  "duree"   -> la duree du mouvement est imposee telle quelle
                "vitesse" -> la duree est CALCULEE a partir de la longueur du
                             chemin, de `vitesse` et de `acceleration`

    vitesse     vitesse maximale : deg/s pour un MOVJ (articulation la plus
                sollicitee), mm/s pour un MOVL ou un MOVC (TCP)
    acceleration    idem, par seconde carree

    lissage     arrondi du raccord AVEC L'INSTRUCTION SUIVANTE, en pourcentage.
                0 = point d'arret exact, 100 = recouvrement maximal. Le point
                de passage n'est alors plus atteint exactement.

    continu     MOVL seulement : le robot enchaine avec le MOVL suivant sans
                s'arreter, en passant EXACTEMENT par la cible (trajectoire
                affine par morceaux, mode CP du controleur). Il ralentit
                dans les coins, d'autant plus qu'ils sont vifs. La vitesse
                de croisiere est `vitesse` (mm/s, 125 au plus) ; la duree
                imposee et la loi horaire sont ignorees. L'outil ne tourne
                pas : r doit rester le meme tout du long.
    """
    type: str = "MOVJ"
    mode: str = "cartesien"
    q: List[float] = field(default_factory=lambda: [0.0, 15.0, 30.0, 0.0])
    pose: List[float] = field(default_factory=lambda: [200.0, 0.0, 50.0, 0.0])
    via: List[float] = field(default_factory=lambda: [200.0, 60.0, 80.0, 0.0])
    duree: float = 2.0
    profil: str = "polynome5"
    etat: bool = False
    commentaire: str = ""
    # OUTIL : couper la pompe apres une ouverture / un relachement (robot
    # reel). Faux : la pompe continue de souffler.
    couper: bool = True
    # champs ajoutes : les programmes enregistres avant leur introduction
    # se rechargent en mode "duree", donc a l'identique
    mode_temps: str = "duree"
    vitesse: float = 0.0            # 0 => valeur par defaut selon l'espace
    acceleration: float = 0.0       # 0 => valeur par defaut selon l'espace
    lissage: float = 0.0            # pourcentage
    port: int = 1                   # SORTIE / ENTREE : numero de port EIO
    repere: str = "base"            # "base" ou "objet" : repere de la cible
    # MOVL : enchainer SANS ARRET avec le MOVL suivant (trajectoire
    # continue, mode CP du robot ; voir Pilote.trajet)
    continu: bool = False
    # CONVOYEUR : le pas du tapis voyage AVEC le programme, comme la
    # longueur d'outil. Le controleur ne connait que des impulsions par
    # seconde ; sans ce nombre, un programme ecrit pour un tapis mesure
    # repartirait sur la valeur nominale des qu'il change de main, et le
    # tapis n'avancerait plus de la meme distance. `vitesse` porte alors
    # les mm/s, `port` l'index du moteur (0 = Stepper1, 1 = Stepper2).
    impulsions_par_mm: float = 0.0
    # Axe lineaire : position du chariot pour ce mouvement, en mm. `rail`
    # n'a de sens que si `avec_rail` est vrai -- sinon zero voudrait dire
    # "chariot a l'origine" au lieu de "pas de rail", et tout programme
    # ecrit sans rail ramenerait le chariot a zero.
    rail: float = 0.0
    avec_rail: bool = False

    # -- unites et valeurs effectives --------------------------------------

    @property
    def unite(self) -> str:
        """
        Unite de l'espace ou s'effectue l'interpolation.

        Elle depend du TYPE de mouvement, pas de la facon dont la cible est
        saisie : un MOVJ interpole les angles meme si sa cible est donnee en
        coordonnees cartesiennes, sa vitesse est donc en degres par seconde.
        """
        return "deg" if self.type == "MOVJ" else "mm"

    def matrice(self, robot=None, repere_objet=None) -> Optional[np.ndarray]:
        """
        Situation ABSOLUE de la cible, en matrice homogene 4x4.

        Les coordonnees stockees sont exprimees dans le repere indique par
        `self.repere` : dans la base, ou dans le repere objet. La matrice
        renvoyee, elle, est toujours dans le repere de base.

        Sur ce robot l'orientation de l'outil est imposee par la mecanique :
        une cible est entierement decrite par (x, y, z, r) plus son repere.
        Le programme la stocke donc sous cette forme, plus lisible dans un
        JSON, et la matrice est reconstruite a la demande.

        Une cible donnee en articulaire exige le robot pour appliquer le MGD.
        """
        if self.type not in ("MOVJ", "MOVL", "MOVC"):
            return None
        if self.mode == "articulaire" and self.type != "MOVC":
            if robot is None:
                raise ValueError("cible articulaire : le robot est requis "
                                 "pour appliquer le modele geometrique direct")
            return robot.mgd(self.q)
        return self._absolue(self.pose, repere_objet)

    def matrice_via(self, repere_objet=None) -> Optional[np.ndarray]:
        """Situation absolue du point de passage d'un MOVC."""
        if self.type != "MOVC":
            return None
        return self._absolue(self.via, repere_objet)

    def _absolue(self, coords, repere_objet) -> np.ndarray:
        T = cible(*coords)
        if self.repere == "objet" and repere_objet is not None:
            return np.asarray(repere_objet, dtype=float) @ T
        return T

    def vitesse_effective(self) -> float:
        return self.vitesse if self.vitesse > 0 else VITESSE_DEFAUT[self.unite]

    def acceleration_effective(self) -> float:
        return self.acceleration if self.acceleration > 0 \
            else ACCELERATION_DEFAUT[self.unite]

    def resume(self, outil=None) -> str:
        """Texte court affiche dans la liste du programme."""
        if self.type == "ATTENTE":
            return self._suffixe("ATTENTE  {:.2f} s".format(self.duree))
        if self.type == "OUTIL":
            if outil is not None:
                etat = "{} {}".format(outil.libelle, outil.etat_texte(self.etat))
            else:
                etat = "actif" if self.etat else "inactif"
            return self._suffixe("OUTIL    {}".format(etat))
        if self.type == "CONVOYEUR":
            return self._suffixe("TAPIS    {}".format(
                "{:+.0f} mm/s".format(self.vitesse) if self.etat else "arret"))
        if self.type == "SORTIE":
            return self._suffixe("SORTIE   EIO{} = {}".format(
                self.port, 1 if self.etat else 0))
        if self.type == "ENTREE":
            return self._suffixe("ENTREE   attendre EIO{} = {}   (max {:.1f} s)"
                                 .format(self.port, 1 if self.etat else 0,
                                         self.duree))

        if self.type == "MOVC":
            tete = ("MOVC     via ({:.0f}, {:.0f}, {:.0f})  ->  "
                    "({:.0f}, {:.0f}, {:.0f})  r={:.0f}"
                    .format(self.via[0], self.via[1], self.via[2],
                            self.pose[0], self.pose[1], self.pose[2],
                            self.pose[3]))
        elif self.mode == "articulaire" and self.type != "MOVC":
            tete = ("{:<8} J = ({:.1f}, {:.1f}, {:.1f}, {:.1f}) deg"
                    .format(self.type, *self.q))
        else:
            tete = ("{:<8} P = ({:.0f}, {:.0f}, {:.0f}) mm  r={:.0f} deg"
                    .format(self.type, self.pose[0], self.pose[1],
                            self.pose[2], self.pose[3]))

        if self.mode_temps == "vitesse":
            u = self.unite
            tete += "  v={:.0f} {}/s  a={:.0f} {}/s2".format(
                self.vitesse_effective(), u, self.acceleration_effective(), u)
        else:
            tete += "  {:.1f}s".format(self.duree)
        if self.continu and self.type == "MOVL":
            tete += "  continu ->"
        elif self.lissage > 0:
            tete += "  lissage {:.0f}%".format(self.lissage)
        return self._suffixe(tete)

    def _suffixe(self, txt: str) -> str:
        return txt + ("   ; " + self.commentaire if self.commentaire else "")


class Programme:
    """Liste ordonnee d'instructions + compilation en trajectoire."""

    def __init__(self, instructions: Optional[List[Instruction]] = None,
                 nom: str = "programme", outil: str = "ventouse",
                 repere_objet=None, longueur_outil: Optional[float] = None):
        self.instructions: List[Instruction] = list(instructions or [])
        self.nom = nom
        self.outil = outil                       # nom dans outils.CATALOGUE
        #: Longueur REELLE de l'outil monte, en millimetres.
        #:
        #: Le nom seul ne suffit pas : les longueurs du catalogue sont
        #: nominales, et un montage particulier s'en ecarte. Sans cette
        #: valeur, un programme ecrit avec un outil mesure repasserait a la
        #: longueur du catalogue des qu'il change de main -- simulateur,
        #: fichier JSON, robot reel -- et le z ne designerait plus le meme
        #: point. `None` signifie "prendre celle du catalogue".
        self.longueur_outil = (None if longueur_outil is None
                               else float(longueur_outil))
        #: repere objet : base -> plan de travail, matrice homogene 4x4.
        #: Les instructions marquees repere="objet" y sont exprimees.
        self.repere_objet = (np.eye(4) if repere_objet is None
                             else np.asarray(repere_objet, dtype=float).copy())
        self.evenements: List[Evenement] = []
        #: courses du chariot : (debut, duree, position visee)
        self.courses_rail: List[Tuple[float, float, float]] = []
        #: Etat de l'outil et des sorties AVANT la premiere instruction.
        #: Un programme se joue d'habitude depuis un etat de repos, d'ou
        #: False partout. La commande temps reel (dobot_sim.Commande), elle,
        #: enchaine de petits programmes : une ventouse activee par l'un est
        #: encore active au debut du suivant, et la fenetre doit le montrer.
        #: Propre a l'animation : ni enregistre, ni envoye au robot.
        self.outil_initial = False
        self.sorties_initiales: Optional[List[bool]] = None
        #: trajectoires continues trouvees par construire() : listes des
        #: indices d'instructions (MOVL) parcourues sans arret
        self.chaines: List[List[int]] = []

    def definir_repere_objet(self, T=None) -> None:
        """Fixe le repere objet a partir d'une matrice homogene 4x4."""
        if T is None:
            self.repere_objet = np.eye(4)
            return
        verifier_repere_objet(T)
        self.repere_objet = np.asarray(T, dtype=float).copy()

    # -- edition ------------------------------------------------------------

    def ajouter(self, instr: Instruction, index: Optional[int] = None) -> None:
        if index is None:
            self.instructions.append(instr)
        else:
            self.instructions.insert(index, instr)

    def supprimer(self, index: int) -> None:
        if 0 <= index < len(self.instructions):
            del self.instructions[index]

    def deplacer(self, index: int, delta: int) -> int:
        j = index + delta
        if 0 <= index < len(self.instructions) and 0 <= j < len(self.instructions):
            self.instructions[index], self.instructions[j] = \
                self.instructions[j], self.instructions[index]
            return j
        return index

    def __len__(self):
        return len(self.instructions)

    # -- compilation --------------------------------------------------------

    def _cible_q(self, robot: DobotMagician, instr: Instruction,
                 q_courant: np.ndarray) -> np.ndarray:
        if instr.mode == "articulaire":
            return np.asarray(instr.q, dtype=float)
        return robot.mgi(instr.matrice(robot, self.repere_objet), tolerant=True)

    @staticmethod
    def _cadencer(instr: Instruction, seg) -> None:
        """
        Fixe la loi horaire du segment.

        En mode "vitesse", la duree n'est pas donnee : elle se deduit de la
        longueur du chemin (que seul le segment construit connait, notamment
        pour un arc) et des consignes de vitesse et d'acceleration.
        """
        if instr.mode_temps == "vitesse":
            seg.profil = creer_profil_vitesse(instr.profil, seg.longueur,
                                              instr.vitesse_effective(),
                                              instr.acceleration_effective())
        else:
            seg.profil = creer_profil(instr.profil, max(instr.duree, 1e-3))

    def construire(self, robot: DobotMagician,
                   q_depart: Optional[np.ndarray] = None) -> Trajectoire:
        """
        Transforme le programme en Trajectoire simulable.

        Les instructions OUTIL sont converties en breves temporisations et
        memorisees dans `evenements_outil` pour l'animation. Les dates de ces
        evenements tiennent compte du lissage, qui raccourcit la trajectoire.
        """
        q = np.asarray(q_depart if q_depart is not None else robot.q_home, dtype=float)
        self.evenements = []

        segments, lissages, logiques = [], [], []
        rails: List[Tuple[int, Instruction]] = []
        indices: List[int] = []             # instruction de chaque segment
        provisoire = creer_profil("lineaire", 1.0)

        for i, instr in enumerate(self.instructions):
            etiquette = "{}. {}".format(i + 1, instr.commentaire or instr.type)
            if getattr(instr, "avec_rail", False):
                rails.append((len(segments), instr))

            if instr.type == "ATTENTE":
                seg = SegmentAttente(robot, q, max(instr.duree, 1e-3), etiquette)

            elif instr.type in ("OUTIL", "SORTIE", "CONVOYEUR"):
                # SegmentAction et non SegmentAttente : une action ne
                # demande pas de temps, elle ne neutralise donc pas le
                # lissage du mouvement qui la precede. Lisser un point
                # ou l'on agit, c'est agir avant d'y etre.
                seg = SegmentAction(robot, q, DUREE_LOGIQUE, etiquette)
                logiques.append((len(segments), instr))

            elif instr.type == "ENTREE":
                # la duree compilee est le delai d'attente maximal ; a la
                # lecture, le temps se fige tant que l'entree n'est pas au
                # bon etat (voir l'animation de l'IHM)
                seg = SegmentAttente(robot, q, max(instr.duree, DUREE_LOGIQUE),
                                     etiquette)
                logiques.append((len(segments), instr))

            elif instr.type == "MOVJ":
                seg = SegmentArticulaire(robot, q, self._cible_q(robot, instr, q),
                                         provisoire, etiquette)
                self._cadencer(instr, seg)

            elif instr.type == "MOVL":
                T1 = instr.matrice(robot, self.repere_objet)
                seg = SegmentLineaire(robot, q, T1, provisoire, etiquette)
                self._cadencer(instr, seg)

            elif instr.type == "MOVC":
                seg = SegmentCirculaire(robot, q,
                                        instr.matrice_via(self.repere_objet),
                                        instr.matrice(robot, self.repere_objet),
                                        provisoire, etiquette)
                self._cadencer(instr, seg)
            else:
                continue

            seg.instruction = instr          # lien segment -> ligne de programme
            indices.append(i)
            segments.append(seg)
            # 0 a 100 % du recouvrement maximal que les deux segments
            # voisins peuvent absorber ; la Trajectoire fait la conversion
            lissages.append(max(0.0, min(instr.lissage, 100.0)) / 100.0)
            q = seg.q_final

        self.chaines = self._chaines(robot, segments, indices)
        for chaine in self.chaines:
            for k in chaine[:-1]:
                lissages[indices.index(k)] = 0.0
        traj = Trajectoire(robot, segments, lissages)
        if self.chaines:
            self._cadencer_chaines(traj, indices)
            traj = Trajectoire(robot, segments, lissages)
        for k, instr in logiques:
            self.evenements.append(Evenement(
                t=traj.debuts[k], type=instr.type, etat=bool(instr.etat),
                port=int(instr.port), segment=k,
                valeur=float(instr.vitesse) if instr.type == "CONVOYEUR"
                else 0.0,
                timeout=float(instr.duree) if instr.type == "ENTREE" else 0.0))
        # le chariot : debut, duree et position visee de chaque
        # mouvement qui l'emmene, pour pouvoir l'interpoler
        self.courses_rail = [(traj.debuts[k], segments[k].duree,
                              float(instr.rail))
                             for k, instr in rails]
        return traj

    def _chaines(self, robot: DobotMagician, segments,
                 indices: List[int]) -> List[List[int]]:
        """
        Suites de MOVL consecutifs marques `continu` (le dernier de la
        suite ne l'est pas forcement). Une suite ou l'outil tourne n'est
        pas continue : le mode CP du controleur ne tourne pas l'outil.
        """
        from .commande import TOLERANCE_R, _ecart_angle
        par_instruction = dict(zip(indices, segments))
        chaines: List[List[int]] = []

        def clore(suite: List[int]) -> None:
            if len(suite) < 2:
                return
            r0 = robot.pose(par_instruction[suite[0]].q(0.0))[3]
            if all(abs(_ecart_angle(r0, robot.pose(
                    par_instruction[k].q_final)[3])) <= TOLERANCE_R
                   for k in suite):
                chaines.append(suite)

        courante: List[int] = []
        for i, instr in enumerate(self.instructions):
            mouvl = instr.type == "MOVL" and i in par_instruction
            if mouvl and (courante or instr.continu):
                courante.append(i)
            if courante and not (mouvl and instr.continu):
                clore(courante)
                courante = []
        clore(courante)
        return chaines

    def _cadencer_chaines(self, traj: Trajectoire, indices: List[int]) -> None:
        """
        Lois horaires des trajectoires continues : celles du controleur en
        mode CP (vitesse de croisiere, ralentissement aux coins, depart et
        arrivee a l'arret), et non les profils de chaque MOVL.
        """
        from .commande import echantillonner_course, troncons_cp
        for chaine in self.chaines:
            s0, s1 = indices.index(chaine[0]), indices.index(chaine[-1])
            depart, portions = echantillonner_course(traj, s0, s1)
            v, a = self.parametres_chaine(chaine)
            troncons, _ = troncons_cp(depart, portions, v, a)
            for j, tr in enumerate(troncons):
                traj.segments[s0 + j].profil = ProfilTroncons(tr, a)

    def parametres_chaine(self, chaine: List[int]) -> Tuple[List[float], float]:
        """Vitesses (une par MOVL) et acceleration envoyees au robot."""
        from .commande import VITESSE_CP_MAX, VITESSE_CP_MIN
        v = [float(np.clip(self.instructions[k].vitesse_effective(),
                           VITESSE_CP_MIN, VITESSE_CP_MAX)) for k in chaine]
        return v, self.instructions[chaine[0]].acceleration_effective()

    @property
    def evenements_outil(self) -> List[Tuple[float, bool]]:
        """Dates de commande de l'outil, au format (instant, etat)."""
        return [(e.t, e.etat) for e in self.evenements if e.type == "OUTIL"]

    def etat_outil(self, t: float) -> bool:
        """Etat de l'outil a l'instant t (apres construire())."""
        etat = bool(self.outil_initial)
        for te, e in self.evenements_outil:
            if te <= t:
                etat = e
        return etat

    def etat_sorties(self, t: float, nb_ports: int = 20) -> List[bool]:
        """Etat des sorties tout-ou-rien a l'instant t (apres construire())."""
        sorties = [False] * nb_ports
        for i, v in enumerate((self.sorties_initiales or [])[:nb_ports]):
            sorties[i] = bool(v)
        for e in self.evenements:
            if e.type == "SORTIE" and e.t <= t and 1 <= e.port <= nb_ports:
                sorties[e.port - 1] = e.etat
        return sorties

    def cadence_convoyeur(self) -> List[Tuple[float, float]]:
        """
        Les changements de vitesse du tapis, dates (apres construire()).

        Exactement la forme qu'attend `Convoyeur.cadence` : il suffit de
        la lui donner pour que la bande et les pieces se placent a
        n'importe quel instant. L'integration etant exacte et par
        morceaux, on peut avancer, reculer et faire du pas a pas sans
        accumuler d'erreur -- ce qui est le point : une animation qui
        derive au rembobinage ne montre plus le programme.
        """
        cadence = [(0.0, 0.0)]
        for e in self.evenements:
            if e.type == "CONVOYEUR":
                cadence.append((float(e.t), float(e.valeur) if e.etat else 0.0))
        return cadence

    def position_rail(self, t: float, depart: float = 0.0) -> float:
        """
        Ou se trouve le chariot a l'instant t (apres construire()).

        Le rail avance AVEC le bras : le controleur traite la cinquieme
        coordonnee comme les quatre autres. On interpole donc sur la
        duree du mouvement, au lieu de faire sauter le chariot a
        l'arrivee -- sinon l'animation montrerait un deplacement qui
        n'existe pas.
        """
        l = float(depart)
        for t0, duree, vise in getattr(self, "courses_rail", []):
            if t >= t0 + duree:
                l = vise
            elif t > t0:
                avancement = (t - t0) / max(duree, 1e-9)
                return l + (vise - l) * avancement
            else:
                break
        return l

    def ports_utilises(self) -> Tuple[List[int], List[int]]:
        """(ports de sortie, ports d'entree) references par le programme."""
        s = sorted({i.port for i in self.instructions if i.type == "SORTIE"})
        e = sorted({i.port for i in self.instructions if i.type == "ENTREE"})
        return s, e

    # -- persistance --------------------------------------------------------

    def vers_dict(self) -> Dict:
        return {"nom": self.nom,
                "version": 4,
                "outil": self.outil,
                "longueur_outil": self.longueur_outil,
                # matrice 4x4, telle qu'on peut la saisir ou la relire
                "repere_objet": [[float(v) for v in ligne]
                                 for ligne in self.repere_objet],
                "instructions": [asdict(i) for i in self.instructions]}

    @classmethod
    def depuis_dict(cls, data: Dict) -> "Programme":
        instrs = []
        for d in data.get("instructions", []):
            champs = {k: v for k, v in d.items()
                      if k in Instruction.__dataclass_fields__}
            instrs.append(Instruction(**champs))
        return cls(instrs, data.get("nom", "programme"),
                   data.get("outil", "ventouse"),
                   data.get("repere_objet"),
                   data.get("longueur_outil"))

    def enregistrer(self, chemin: str) -> None:
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self.vers_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def charger(cls, chemin: str) -> "Programme":
        with open(chemin, "r", encoding="utf-8") as f:
            return cls.depuis_dict(json.load(f))

    # -- export vers le robot reel -----------------------------------------

    def _commande_outil(self, ins: Instruction) -> str:
        """Commande DobotDllType correspondant a l'outil monte."""
        actif = 1 if ins.etat else 0
        if self.outil == "pince":
            # enableCtrl, grip : True = serre
            return ("dernier = dType.SetEndEffectorGripper(api, 1, {}, "
                    "isQueued=1)[0]".format(actif))
        if self.outil == "laser":
            return ("dernier = dType.SetEndEffectorLaser(api, 1, {}, "
                    "isQueued=1)[0]".format(actif))
        if self.outil in ("aucun", "stylo"):
            return "# outil '{}' : aucun actionneur a commander".format(self.outil)
        return ("dernier = dType.SetEndEffectorSuctionCup(api, 1, {}, "
                "isQueued=1)[0]".format(actif))

    def exporter_python(self, robot: DobotMagician) -> str:
        """
        Genere un script Python pilotant le VRAI Dobot via l'API officielle
        DobotDllType (la meme que celle fournie avec DobotStudio).
        """
        L = [
            '"""',
            "Script genere automatiquement par le simulateur Dobot.",
            "Programme : {}".format(self.nom),
            "",
            "Prerequis : DobotDllType.py et les DLL Dobot dans le meme dossier.",
            "ATTENTION : degagez la zone de travail avant de lancer ce script.",
            "Outil monte : {}".format(self.outil),
            '"""',
            "",
        ]
        # `time` ne sert qu'a la scrutation des entrees
        if any(i.type == "ENTREE" for i in self.instructions):
            L += ["import time", ""]
        L += [
            "import DobotDllType as dType",
            "",
            "api = dType.load()",
            'etat = dType.ConnectDobot(api, "", 115200)[0]',
            "if etat != dType.DobotConnect.DobotConnect_NoError:",
            '    raise SystemExit("connexion au Dobot impossible : %s" % etat)',
            "",
            "dType.SetQueuedCmdClear(api)",
            "dType.SetHOMEParams(api, 200, 0, 0, 0, isQueued=1)",
            "dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)",
            "dType.SetPTPCoordinateParams(api, 200, 200, 200, 200, isQueued=1)",
            "dType.SetPTPCommonParams(api, 100, 100, isQueued=1)",
            "",
        ]

        # configuration des ports EIO utilises : 2 = sortie, 3 = entree
        p_sorties, p_entrees = self.ports_utilises()
        if p_sorties or p_entrees:
            L.append("# multiplexage des ports EIO (2 = sortie, 3 = entree)")
            for p in p_sorties:
                L.append("dType.SetIOMultiplexing(api, {}, 2, isQueued=1)".format(p))
            for p in p_entrees:
                L.append("dType.SetIOMultiplexing(api, {}, 3, isQueued=1)".format(p))
            L.append("")
        L += ["dernier = 0", ""]
        dernier_reglage = None
        lissage_demande = False
        self.construire(robot)                   # trouve les chaines continues
        en_continu = {}
        for chaine in self.chaines:
            vitesses, a = self.parametres_chaine(chaine)
            for k, v in zip(chaine, vitesses):
                en_continu[k] = (v, a)
        for i, ins in enumerate(self.instructions):
            if ins.commentaire:
                L.append("# {}".format(ins.commentaire))

            if i in en_continu:
                # trajectoire continue (CP) : la vitesse de croisiere est
                # junctionVel ; le champ velocity de SetCPCmd est ignore
                v, a = en_continu[i]
                reglage = ("cp", round(v, 2), round(a, 2))
                if reglage != dernier_reglage:
                    L.append("dType.SetCPParams(api, {a:.2f}, {v:.2f}, {a:.2f}, "
                             "0, isQueued=1)".format(a=a, v=v))
                    dernier_reglage = reglage
                x, y, z, _r = decomposer_cible(ins.matrice(robot,
                                                           self.repere_objet))
                L.append("dernier = dType.SetCPCmd(api, 1, {:.2f}, {:.2f}, "
                         "{:.2f}, {:.2f}, isQueued=1)[0]".format(x, y, z, v))
                continue

            # vitesse et acceleration : on ne reemet la commande que si elles
            # changent, pour ne pas saturer la file du controleur
            if ins.type in ("MOVJ", "MOVL", "MOVC"):
                v, a = ins.vitesse_effective(), ins.acceleration_effective()
                reglage = (ins.unite, round(v, 2), round(a, 2))
                if reglage != dernier_reglage:
                    if ins.unite == "deg":
                        L.append("dType.SetPTPJointParams(api, {v}, {v}, {v}, {v}, "
                                 "{a}, {a}, {a}, {a}, isQueued=1)"
                                 .format(v="{:.2f}".format(v), a="{:.2f}".format(a)))
                    else:
                        L.append("dType.SetPTPCoordinateParams(api, {v:.2f}, {v:.2f}, "
                                 "{a:.2f}, {a:.2f}, isQueued=1)".format(v=v, a=a))
                    dernier_reglage = reglage
                if ins.lissage > 0:
                    lissage_demande = True

            if ins.type == "ATTENTE":
                L.append("dernier = dType.SetWAITCmd(api, {:d}, isQueued=1)[0]"
                         .format(int(ins.duree * 1000)))
            elif ins.type == "OUTIL":
                L.append(self._commande_outil(ins))
            elif ins.type == "SORTIE":
                L.append("dernier = dType.SetIODO(api, {}, {}, isQueued=1)[0]"
                         .format(ins.port, 1 if ins.etat else 0))
            elif ins.type == "ENTREE":
                # une attente sur entree ne peut pas vivre dans la file de
                # commandes : on vide la file, puis on scrute depuis le PC
                L += [
                    "dType.SetQueuedCmdStartExec(api)",
                    "while dernier > dType.GetQueuedCmdCurrentIndex(api)[0]:",
                    "    dType.dSleep(50)",
                    "_t0 = time.time()",
                    "while dType.GetIODI(api, {})[0] != {}:".format(
                        ins.port, 1 if ins.etat else 0),
                    "    if time.time() - _t0 > {:.2f}:".format(max(ins.duree, 0.1)),
                    '        print("ATTENTE EIO{} : delai depasse")'.format(ins.port),
                    "        break",
                    "    dType.dSleep(20)",
                    "dType.SetQueuedCmdStopExec(api)",
                ]
            elif ins.type == "MOVC":
                L.append("dernier = dType.SetARCCmd(api, [{:.2f}, {:.2f}, {:.2f}, {:.2f}], "
                         "[{:.2f}, {:.2f}, {:.2f}, {:.2f}], isQueued=1)[0]"
                         .format(*decomposer_cible(ins.matrice_via(self.repere_objet)),
                                 *decomposer_cible(ins.matrice(robot, self.repere_objet))))
            else:
                if ins.mode == "articulaire":
                    mode = "PTPMOVJANGLEMode" if ins.type == "MOVJ" else "PTPMOVLANGLEMode"
                    vals = ins.q
                else:
                    mode = "PTPMOVJXYZMode" if ins.type == "MOVJ" else "PTPMOVLXYZMode"
                    # le controleur ne connait pas de repere objet : on
                    # exporte des coordonnees absolues
                    vals = decomposer_cible(ins.matrice(robot, self.repere_objet))
                L.append("dernier = dType.SetPTPCmd(api, dType.PTPMode.{}, "
                         "{:.2f}, {:.2f}, {:.2f}, {:.2f}, isQueued=1)[0]"
                         .format(mode, *vals))
        if not np.allclose(self.repere_objet, np.eye(4)):
            L.insert(0, "# Les cibles exprimees dans le repere objet ont ete "
                        "converties en")
            L.insert(1, "# coordonnees ABSOLUES : le controleur Dobot ne "
                        "connait pas cette notion.")
            L.insert(2, "")
        if not robot.outil_est_par_defaut:
            L.insert(0, "# ATTENTION : un repere outil deporte est en place "
                        "dans le simulateur.")
            L.insert(1, "# Les coordonnees ci-dessous designent SA pointe. "
                        "Verifiez que le montage")
            L.insert(2, "# reel correspond, le controleur ne connait que le "
                        "deport vertical.")
            L.insert(3, "")
        if lissage_demande:
            L.insert(0, "# ATTENTION : le programme utilise le lissage des "
                        "raccords, qui est une")
            L.insert(1, "# fonction du SIMULATEUR. Les commandes PTP du Dobot "
                        "marquent un arret a")
            L.insert(2, "# chaque point : le robot reel passera donc exactement "
                        "par les points de")
            L.insert(3, "# passage, un peu plus lentement, sans couper les angles.")
            L.insert(4, "")
        L += [
            "",
            "dType.SetQueuedCmdStartExec(api)",
            "while dernier > dType.GetQueuedCmdCurrentIndex(api)[0]:",
            "    dType.dSleep(100)",
            "dType.SetQueuedCmdStopExec(api)",
            "dType.DisconnectDobot(api)",
            "",
        ]
        return "\n".join(L)

    def exporter_csv(self, robot: DobotMagician, dt: float = 0.05,
                     q_depart: Optional[np.ndarray] = None) -> str:
        """Echantillonnage de la trajectoire au format CSV (consignes)."""
        traj = self.construire(robot, q_depart)
        e = traj.echantillonner(dt)
        lignes = ["t;J1;J2;J3;J4;x;y;z;r"]
        for k in range(len(e["t"])):
            lignes.append("{:.3f};{:.4f};{:.4f};{:.4f};{:.4f};"
                          "{:.3f};{:.3f};{:.3f};{:.3f}"
                          .format(e["t"][k], *e["Q"][k], *e["P"][k], e["R"][k]))
        return "\n".join(lignes)


# ---------------------------------------------------------------------------

def programme_demo() -> Programme:
    """
    Petit programme de prise et depose, charge au demarrage.

    Il est ecrit en mode "vitesse" : les durees sont calculees a partir des
    consignes de vitesse et d'acceleration. Les deux mouvements d'approche
    sont lisses entre eux pour montrer l'effet de l'arrondi de raccord.
    """
    def m(**kw):
        base = dict(mode_temps="vitesse", profil="polynome5")
        base.update(kw)
        return Instruction(**base)

    return Programme([
        m(type="MOVJ", mode="articulaire", q=[0, 15, 30, 0],
          vitesse=90, acceleration=300, commentaire="position de repli"),
        m(type="MOVJ", mode="cartesien", pose=[200, -80, 60, 0],
          vitesse=110, acceleration=400, lissage=40,
          commentaire="approche de la piece"),
        m(type="MOVL", mode="cartesien", pose=[200, -80, 10, 0],
          vitesse=60, acceleration=250, profil="trapeze",
          commentaire="descente lente"),
        m(type="OUTIL", etat=True, commentaire="prise"),
        m(type="MOVL", mode="cartesien", pose=[200, -80, 90, 0],
          vitesse=120, acceleration=500, lissage=60, commentaire="remontee"),
        m(type="MOVC", mode="cartesien", via=[230, 0, 120, 0],
          pose=[200, 80, 90, 0], vitesse=150, acceleration=500, lissage=60,
          commentaire="transfert en arc de cercle"),
        m(type="MOVL", mode="cartesien", pose=[200, 80, 10, 0],
          vitesse=60, acceleration=250, profil="trapeze",
          commentaire="descente lente"),
        m(type="OUTIL", etat=False, commentaire="depose"),
        m(type="MOVL", mode="cartesien", pose=[200, 80, 90, 0],
          vitesse=120, acceleration=500, commentaire="degagement"),
        m(type="MOVJ", mode="articulaire", q=[0, 15, 30, 0],
          vitesse=90, acceleration=300, commentaire="retour au repli"),
    ], nom="prise et depose")
