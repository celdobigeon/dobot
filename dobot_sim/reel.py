#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pont entre le simulateur et le Dobot Magician reel.

Traduit un `Programme` en commandes de file pour le controleur, suit leur
execution, et renvoie la pose reelle pour animer le modele 3D.

Le repere du controleur n'est PAS celui du modele
-------------------------------------------------
Mesure sur une machine reelle (firmware 3.7.0), la pose annoncee par
`GetPose` vaut, en fonction des angles :

    rho_ctrl = l1.sin(J2) + l2.cos(J3) + xBias
    z_ctrl   = l1.cos(J2) - l2.sin(J3) + zBias

Deux differences avec le modele du simulateur :

  * le controleur place son origine a HAUTEUR D'EPAULE : son z ne contient
    pas `d1`. L'ecart mesure valait exactement -139,000 mm ;
  * son deport horizontal est `xBias`, lu dans `EndEffectorParams`, et non
    le `l3 = 59,7 mm` de la table DH. Sur la machine d'essai xBias valait
    61,0 mm, d'ou 1,3 mm d'ecart radial.

D'ou la conversion exacte, dans les deux sens :

    rho_ctrl = rho_tcp - l3 + xBias        z_ctrl = z_tcp + lt - d1 + zBias
    rho_tcp  = rho_ctrl + l3 - xBias       z_tcp  = z_ctrl - lt + d1 - zBias

`lt` apparait parce que le z du simulateur designe la POINTE de l'outil,
alors que celui du controleur se rapporte au plan du poignet tant que
zBias vaut 0.

Les ANGLES, eux, n'ont besoin d'aucune conversion : les quatre valeurs
renvoyees par `GetPose` sont exactement celles du modele. C'est pourquoi le
miroir temps reel les utilise plutot que les coordonnees.

Securite
--------
Rien ne part vers la machine sans appel explicite. `televerser()` refuse un
programme dont le controle signale une butee ou une collision, sauf
`forcer=True`. `arret_urgence()` coupe le mouvement en cours et vide la file.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .liaison import (CommandeDobot, ErreurLiaison, FilePleine, FonctionES,
                      LiaisonDobot, PTP, resume_alarmes)
from .robot import decomposer_cible

__all__ = ["PontReel", "ErreurPont", "EtatReel"]


#: avertissements de controler() qui interdisent le televersement : le
#: robot irait en butee, hors d'atteinte ou dans son propre volume
BLOQUANTS_TELEVERSEMENT = ("hors de [", "hors d'atteinte", "plan de pose",
                           "embase :", "colonne :", "auto-collision")


class ErreurPont(RuntimeError):
    """Le programme ne peut pas etre transmis en l'etat."""


@dataclass
class EtatReel:
    """Instantane de la machine, pour l'affichage."""
    connecte: bool = False
    version: str = ""
    port: str = ""
    angles: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pose_controleur: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    index_courant: int = 0
    index_final: int = 0
    alarmes: List[str] = field(default_factory=list)

    @property
    def progression(self) -> float:
        """0 a 1, avance dans la file televersee."""
        if self.index_final <= self._depart:
            return 1.0
        return float(np.clip((self.index_courant - self._depart)
                             / (self.index_final - self._depart), 0.0, 1.0))

    _depart: int = 0


class PontReel:
    """
    Envoie un programme au robot reel et suit son execution.

        pont = PontReel(robot_modele, journal=print)
        pont.connecter("COM4")
        pont.televerser(programme)        # remplit la file, ne bouge pas
        pont.demarrer()                   # LA, le robot bouge
    """

    #: Profondeur de la file du controleur, en commandes. MESUREE sur la
    #: machine du labo (firmware 4.0.3) : GetQueuedCmdLeftSpace repond 32
    #: file vide et 0 file pleine, et la 33e commande est jetee. C'est
    #: aussi le rang ou s'arretait le programme de 150 commandes qui a
    #: revele la panne -- les 21 % de la barre de progression.
    CAPACITE_FILE = 32

    #: Places qu'on laisse toujours libres quand le controleur annonce sa
    #: place restante. Entre la reponse et l'envoi suivant, la file a pu
    #: se remplir d'une commande partie d'ailleurs (pilotage manuel,
    #: effecteur force a la main) : cette garde l'absorbe.
    GARDE_FILE = 2

    #: Ce qu'on s'autorise a laisser en vol. En dessous de la capacite,
    #: parce qu'un index lu est toujours un peu en retard sur la realite :
    #: la marge absorbe ce retard.
    MARGE_FILE = 24

    #: Plancher du repli : en dessous, le dialogue couterait plus que le
    #: mouvement. Quatre commandes en vol suffisent a ne pas affamer un
    #: robot dont chaque mouvement dure une seconde.
    MARGE_MINIMALE = 4

    #: Echecs de liaison consecutifs toleres avant d'abandonner le reste
    #: du programme. Une trame perdue n'est pas une liaison perdue.
    ECHECS_AVANT_ABANDON = 20

    def __init__(self, robot, journal: Optional[Callable[[str], None]] = None):
        self.robot = robot                    # DobotMagician du simulateur
        self.journal = journal or (lambda _m: None)
        self.liaison: Optional[LiaisonDobot] = None
        self.biais: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.index_final = 0
        self._index_depart = 0
        self._ports_configures: set = set()
        # commandes qui n'ont pas tenu dans la file au televersement, et la
        # generation de file a laquelle elles appartiennent
        self._reste: List[Callable[[], int]] = []
        self._generation = -1
        self._alimentation: Optional[threading.Thread] = None
        self._cadence: dict = {}           # dernier reglage, par famille
        # ce que le fil d'alimentation a rencontre : l'IHM le relit, le fil
        # n'ecrivant pas lui-meme dans le journal
        self._incident = ""
        # marge de travail, ramenee a la baisse si le controleur refuse :
        # la profondeur de sa file n'est pas publiee par le protocole
        self.marge = self.MARGE_FILE

    # -- connexion ----------------------------------------------------------

    @property
    def connecte(self) -> bool:
        return self.liaison is not None and self.liaison.ouverte

    def connecter(self, port: str) -> str:
        """Ouvre la liaison et releve la configuration. Ne fait RIEN bouger."""
        self.deconnecter()
        self.liaison = LiaisonDobot(port, journal=self.journal).ouvrir()
        try:
            version = self.liaison.version()
            self.biais = self.liaison.biais_effecteur()
        except Exception:
            # Sans cela le port resterait ouvert -- et verrouille -- derriere
            # un etat "connexion impossible" : plus rien ne pourrait l'ouvrir,
            # pas meme un nouvel essai depuis cette fenetre.
            self.deconnecter()
            raise
        self.journal("robot reel connecte sur {} - firmware {}".format(port, version))
        self.journal("  EndEffectorParams du controleur : xBias={:.2f} "
                     "yBias={:.2f} zBias={:.2f} mm"
                     .format(*self.biais))
        ecart = self.biais[0] - self.robot.geo.l3
        if abs(ecart) > 0.05:
            self.journal("  NOTE : le controleur utilise xBias = {:.2f} mm la ou "
                         "le modele a l3 = {:.2f} mm ({:+.2f} mm). La conversion "
                         "en tient compte ; la calibration machine n'est PAS "
                         "modifiee.".format(self.biais[0], self.robot.geo.l3, ecart))
        al = self.liaison.alarmes()
        for m in resume_alarmes(al):
            self.journal("  " + m)
        self._ports_configures = set()
        return version

    def deconnecter(self) -> None:
        if self.liaison is not None:
            try:
                self.liaison.fermer()
            finally:
                self.liaison = None

    def _exiger(self) -> LiaisonDobot:
        if not self.connecte:
            raise ErreurPont("aucun robot reel connecte")
        return self.liaison

    # -- conversion de repere ----------------------------------------------

    def vers_controleur(self, x: float, y: float, z: float,
                        r: float) -> Tuple[float, float, float, float]:
        """Pose TCP du simulateur -> coordonnees du controleur."""
        xb, _yb, zb = self.biais
        g = self.robot.geo
        rho = float(np.hypot(x, y))
        rho2 = rho - g.l3 + xb
        if rho > 1e-9:
            k = rho2 / rho
            x2, y2 = x * k, y * k
        else:                                  # sur l'axe : la direction vient de r
            a = np.radians(r)
            x2, y2 = rho2 * np.cos(a), rho2 * np.sin(a)
        return x2, y2, z + g.lt - g.d1 + zb, r

    def depuis_controleur(self, x: float, y: float, z: float,
                          r: float) -> Tuple[float, float, float, float]:
        """Coordonnees du controleur -> pose TCP du simulateur."""
        xb, _yb, zb = self.biais
        g = self.robot.geo
        rho = float(np.hypot(x, y))
        rho2 = rho + g.l3 - xb
        if rho > 1e-9:
            k = rho2 / rho
            x2, y2 = x * k, y * k
        else:
            a = np.radians(r)
            x2, y2 = rho2 * np.cos(a), rho2 * np.sin(a)
        return x2, y2, z - g.lt + g.d1 - zb, r

    # -- lecture ------------------------------------------------------------

    def etat(self) -> EtatReel:
        """Instantane complet ; une poignee de millisecondes par appel."""
        if not self.connecte:
            return EtatReel()
        li = self._exiger()
        p = li.pose()
        e = EtatReel(connecte=True, port=li.port,
                     angles=p.angles,
                     pose_controleur=(p.x, p.y, p.z, p.r),
                     index_courant=li.index_courant(),
                     index_final=self.index_final,
                     alarmes=resume_alarmes(li.alarmes()))
        e._depart = self._index_depart
        return e

    def angles_reels(self) -> np.ndarray:
        """
        Les quatre angles lus sur la machine, en degres.

        Aucune conversion : le controleur et le modele emploient la meme
        convention articulaire. C'est la voie sure pour le miroir 3D.
        """
        return np.asarray(self._exiger().pose().angles, dtype=float)

    # -- ecriture -----------------------------------------------------------

    def prise_origine(self) -> int:
        """
        Prise d'origine : LE ROBOT PARCOURT TOUTE SA COURSE.

        Meme precaution qu'au televersement : on COUPE la file avant de la
        vider, sinon une file restee en mode "demarre" ferait partir la
        commande a son arrivee, avant meme qu'on ait repris la main. Les
        alarmes sont effacees au passage : une alarme active peut faire
        refuser le mouvement.
        """
        li = self._exiger()
        li.arreter_file()
        li.vider_file()
        self._reste = []
        li.effacer_alarmes()
        idx = li.prise_origine()
        self._index_depart = idx - 1
        self.index_final = idx
        li.demarrer_file()
        return idx

    def coherence(self) -> Tuple[float, Tuple[float, float, float, float]]:
        """
        Confronte la pose annoncee par le controleur a celle que MON modele
        deduit des angles qu'il annonce en meme temps.

        `GetPose` renvoie les deux : si les deux descriptions concordent, le
        modele et la machine parlent bien de la meme GEOMETRIE. Retourne
        l'ecart maximal et la pose recalculee.

        Ce controle ne dit RIEN de la calibration. Pose et angles sont
        calcules a partir des memes compteurs : ils restent d'accord entre
        eux meme quand le zero a derive. Pour savoir si une prise d'origine
        manque, comparez la posture du miroir 3D a celle du vrai bras.
        """
        li = self._exiger()
        p = li.pose()
        q = np.asarray(p.angles, dtype=float)
        xb, _yb, zb = self.biais
        g = self.robot.geo
        a = np.radians(q[0])
        rho = (g.l1 * np.sin(np.radians(q[1]))
               + g.l2 * np.cos(np.radians(q[2])) + xb)
        attendu = (float(rho * np.cos(a)), float(rho * np.sin(a)),
                   float(g.l1 * np.cos(np.radians(q[1]))
                         - g.l2 * np.sin(np.radians(q[2])) + zb),
                   float(q[0] + q[3]))
        ecart = float(np.abs(np.array([p.x, p.y, p.z, p.r])
                             - np.array(attendu)).max())
        return ecart, attendu

    def arret_urgence(self) -> None:
        self._reste = []
        if self.connecte:
            self.liaison.arret_urgence()

    def demarrer(self) -> None:
        self._exiger().demarrer_file()
        self.journal("execution reelle lancee")
        self._lancer_alimentation()

    def suspendre(self) -> None:
        self._exiger().arreter_file()
        self.journal("execution reelle suspendue")

    def definir_ratio(self, vitesse_pct: float) -> None:
        """Bride globalement les vitesses, en pourcentage."""
        self._exiger().definir_ratio(float(vitesse_pct), float(vitesse_pct))
        self.journal("ratio de vitesse reel : {:.0f} %".format(vitesse_pct))

    # -- televersement ------------------------------------------------------

    def televerser(self, programme, q_depart=None, forcer: bool = False,
                   progression: Optional[Callable[[int, int], None]] = None,
                   outil: str = "") -> int:
        """
        Vide la file et y depose le programme. NE LANCE RIEN.

        Retourne l'index de la derniere commande. Un programme dont le
        controle signale une butee ou une collision est REFUSE, sauf
        `forcer=True`.

        La file du controleur ne tient que quelques dizaines de commandes.
        Ce qui n'y entre pas est garde ici et envoye par `demarrer()` au fur
        et a mesure que l'execution libere des places : sans cela, un cycle
        repete dix fois ne s'executerait que deux fois, le reste etant
        refuse par le controleur.
        """
        li = self._exiger()

        traj = programme.construire(self.robot, q_depart)
        defauts = [a for a in traj.controler(0.05)
                   if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
        if defauts and not forcer:
            raise ErreurPont(
                "programme refuse : {} defaut(s) sur la trajectoire.\n  {}"
                .format(len(defauts), "\n  ".join(defauts[:6])))

        if any(getattr(i, "lissage", 0) for i in programme.instructions):
            self.journal("NOTE : le lissage n'existe pas en PTP sur le "
                         "controleur ; le robot reel marquera un arret a "
                         "chaque point.")

        self._preparer_file(li)
        commandes: List[Callable[[], int]] = []
        couper = False
        for ins in programme.instructions:
            commandes.extend(self._traduire(li, ins, programme, outil))
            # Ouvrir la pince ou relacher la ventouse fait SOUFFLER la
            # pompe, qui ne s'arrete plus. On la coupe dans la file, APRES
            # le mouvement suivant : elle souffle pendant ce mouvement, le
            # temps que la pince s'ouvre, sans pause ajoutee au cycle.
            if ins.type == "OUTIL" and outil in ("pince", "ventouse"):
                couper = (not bool(ins.etat)) and bool(getattr(ins, "couper", True))
            elif couper and ins.type in ("MOVJ", "MOVL", "MOVC"):
                commandes.extend(self._couper_pompe_en_file(li))
                couper = False
        if couper:
            # ouverture en fin de programme : une seconde, puis coupure
            commandes.append(lambda: li.attendre(1.0))
            commandes.extend(self._couper_pompe_en_file(li))
        return self._deposer(li, commandes, progression,
                             "programme televerse : {} instructions"
                             .format(len(programme.instructions)))

    def televerser_commandes(self, fabriquer: Callable[[], List[Callable[[], int]]],
                             progression: Optional[Callable[[int, int], None]]
                             = None, quoi: str = "commandes televersees") -> int:
        """
        Vide la file et y depose des commandes deja traduites. NE LANCE RIEN.

        `fabriquer` est appele APRES le vidage et rend la liste des
        commandes, chacune une fonction qui envoie UNE commande en file et
        rend son index -- la forme de `_traduire`, `traduire` et
        `commandes_cp`. On la fabrique apres le vidage, et non avant,
        parce que la traduction peut envoyer des commandes IMMEDIATES (le
        multiplexage d'un port) et qu'elle se souvient de ce qu'elle a deja
        regle : ces souvenirs ne valent que pour la file neuve.

        C'est la porte d'entree de `dobot_sim.Commande`, qui melange
        mouvements point a point et trajectoire continue. Le surplus suit
        le meme chemin que pour `televerser` : garde ici, verse pendant
        l'execution au rythme ou la file se vide.

        Aucun controle de trajectoire ici : l'appelant les a faits, sur un
        chemin que ce pont ne connait pas (points CP echantillonnes).
        """
        li = self._exiger()
        self._preparer_file(li)
        return self._deposer(li, list(fabriquer()), progression, quoi)

    def _preparer_file(self, li: LiaisonDobot) -> None:
        """
        Arrete puis vide la file, et oublie ce qui la concernait.

        ORDRE CRITIQUE : arreter AVANT de vider. Si la file etait restee en
        mode "demarre", chaque commande televersee partirait des son arrivee
        et le robot se mettrait a bouger pendant le remplissage.
        """
        li.arreter_file()
        li.vider_file()
        self._reste = []
        self._generation = li.generation_file
        self._index_depart = li.index_courant()
        self._cadence = {}

    def _deposer(self, li: LiaisonDobot, commandes: List[Callable[[], int]],
                 progression: Optional[Callable[[int, int], None]],
                 quoi: str) -> int:
        """Remplit la file jusqu'a la marge et garde le reste pour plus tard."""
        n = len(commandes)
        premier = None
        envoyees = 0
        for envoyer in commandes:
            # La file est ARRETEE pendant le televersement : rien ne s'y
            # consomme, et on ne peut donc pas en mettre plus que la marge.
            # Ce comptage remplace la detection d'un refus : sur la machine,
            # une file pleine repond un index valide et JETTE la commande --
            # on croyait tout avoir envoye, le robot s'arretait au bout de
            # ce que la file contenait vraiment.
            if envoyees >= self.marge:
                break
            try:
                idx = envoyer()
            except FilePleine:
                self._reduire_marge()       # second filet, si le firmware refuse
                break
            if premier is None:
                premier = idx
            envoyees += 1
            if progression is not None:
                progression(envoyees, n)
        if progression is not None:
            progression(n, n)

        if n and premier is None:
            raise ErreurPont("le controleur refuse toute commande : file "
                             "pleine alors qu'elle vient d'etre videe")
        self._reste = commandes[envoyees:]
        # Le controleur numerote ses commandes en file une a une : l'index
        # de la derniere se deduit de celui de la premiere, meme pour celles
        # qui ne partiront qu'en cours d'execution.
        self.index_final = (premier + n - 1) if n else self._index_depart
        self.journal("{}, {} commandes, index {} -> {}"
                     .format(quoi, n, self._index_depart, self.index_final))
        if self._reste:
            self.journal("file du controleur remplie a {} commandes (marge) : "
                         "les {} suivantes partiront pendant l'execution, au "
                         "rythme ou la file se vide"
                         .format(envoyees, len(self._reste)))
        return self.index_final

    def traduire(self, ins, programme, outil: str = "") -> List[Callable[[], int]]:
        """Version publique de `_traduire`, sur la liaison ouverte."""
        return self._traduire(self._exiger(), ins, programme, outil)

    def commandes_cp(self, points, vitesse: float, acceleration: float,
                     r: float = 0.0) -> List[Callable[[], int]]:
        """
        Points de trajectoire CONTINUE, SANS les envoyer.

        `points` sont des (x, y, z) du SIMULATEUR : la conversion vers le
        repere du controleur se fait ici, comme pour les PTP. `r` ne sert
        qu'a orienter un point pose exactement sur l'axe de la base ; le
        mode CP ne fait pas tourner l'outil.

        Le reglage de vitesse precede les points, MIS EN FILE : c'est ainsi
        qu'une trajectoire change de vitesse en route sans s'arreter. Un
        reglage identique au precedent n'est pas renvoye, comme pour les
        PTP.
        """
        li = self._exiger()
        out: List[Callable[[], int]] = []
        reglage = (round(float(acceleration), 3), round(float(vitesse), 3))
        if self._cadence.get(CommandeDobot.PARAMS_CP) != reglage:
            self._cadence[CommandeDobot.PARAMS_CP] = reglage
            a, v = reglage
            out.append(lambda: li.parametres_cp(a, v))
        for p in points:
            x, y, z, _r = self.vers_controleur(float(p[0]), float(p[1]),
                                               float(p[2]), float(r))
            out.append(lambda x=x, y=y, z=z, v=reglage[1]: li.cp(x, y, z, v))
        return out

    @staticmethod
    def _couper_pompe_en_file(li: LiaisonDobot) -> List[Callable[[], int]]:
        """Pompe arretee (ventouse et pince), comme commandes DE FILE."""
        return [lambda: li.ventouse(False, active=False),
                lambda: li.pince(False, active=False)]

    def _traduire(self, li: LiaisonDobot, ins, programme,
                  outil: str) -> List[Callable[[], int]]:
        """
        Traduit UNE instruction en commande(s) de file, SANS les envoyer.

        Chaque element est une fonction qui envoie une seule commande et
        retourne son index : une commande refusee faute de place peut ainsi
        etre renvoyee telle quelle. Les commandes immediates (multiplexage
        des ports) partent, elles, tout de suite.
        """
        t = ins.type

        if t in ("MOVJ", "MOVL"):
            cadence = self._regler_cadence(li, ins)
            if getattr(ins, "mode", "cartesien") == "articulaire":
                q = [float(v) for v in ins.q]
                mode = PTP.MOVJ_ANGLE if t == "MOVJ" else PTP.MOVL_ANGLE
                return cadence + [lambda: li.ptp(mode, *q)]
            x, y, z, r = self._cible(ins, programme)
            mode = PTP.MOVJ_XYZ if t == "MOVJ" else PTP.MOVL_XYZ
            return cadence + [lambda: li.ptp(mode, x, y, z, r)]

        if t == "MOVC":
            cadence = self._regler_cadence(li, ins)
            via = self.vers_controleur(*self._pose_de(ins.matrice_via(
                programme.repere_objet)))
            cible = self._cible(ins, programme)
            return cadence + [lambda: li.arc(via, cible)]

        if t == "ATTENTE":
            duree = float(ins.duree)
            return [lambda: li.attendre(duree)]

        if t == "SORTIE":
            self._assurer_port(li, ins.port, FonctionES.SORTIE)
            port, etat = int(ins.port), bool(ins.etat)
            return [lambda: li.sortie(port, etat)]

        if t == "ENTREE":
            self._assurer_port(li, ins.port, FonctionES.ENTREE)
            port, etat = int(ins.port), bool(ins.etat)
            return [lambda: li.attendre_entree(port, etat)]

        if t == "OUTIL":
            actif = bool(ins.etat)
            if outil == "pince":
                return [lambda: li.pince(actif)]
            if outil == "laser":
                return [lambda: li.laser(actif)]
            return [lambda: li.ventouse(actif)]

        self.journal("instruction {} ignoree : sans equivalent sur le "
                     "controleur".format(t))
        return []

    def _cible(self, ins, programme) -> Tuple[float, float, float, float]:
        T = ins.matrice(self.robot, programme.repere_objet)
        return self.vers_controleur(*self._pose_de(T))

    def _pose_de(self, T) -> Tuple[float, float, float, float]:
        """Matrice homogene -> (x, y, z, r) du simulateur."""
        return tuple(float(v) for v in decomposer_cible(T))

    def _regler_cadence(self, li: LiaisonDobot, ins) -> List[Callable[[], int]]:
        """
        Pose vitesse et acceleration juste avant le mouvement.

        Les commandes de parametres sont elles-memes MISES EN FILE : le
        reglage prend donc effet au bon moment de la sequence, et non a
        l'instant du televersement. Un reglage identique au precedent n'est
        pas renvoye : il occuperait une place de file pour rien.

        L'unite decide de la famille : un MOVJ s'exprime en deg/s et vise
        les parametres articulaires, un MOVL en mm/s et vise les
        parametres cartesiens.
        """
        v = float(getattr(ins, "vitesse", 0.0) or 0.0)
        a = float(getattr(ins, "acceleration", 0.0) or 0.0)
        if v <= 0.0 or a <= 0.0:
            return []
        if getattr(ins, "unite", "mm") == "deg":
            ident = CommandeDobot.PARAMS_PTP_ARTICULAIRE
            params = struct.pack("<8f", v, v, v, v, a, a, a, a)
        else:
            ident = CommandeDobot.PARAMS_PTP_CARTESIEN
            params = struct.pack("<4f", v, 100.0, a, 100.0)
        # les deux familles sont independantes dans le controleur
        if self._cadence.get(ident) == params:
            return []
        self._cadence[ident] = params

        def envoyer() -> int:
            li.echanger(ident, params, ecriture=True, en_file=True)
            return li.dernier_index
        return [envoyer]

    def _assurer_port(self, li: LiaisonDobot, port: int, fonction: int) -> None:
        """Configure le multiplexage d'un port EIO, une seule fois."""
        cle = (int(port), int(fonction))
        if cle in self._ports_configures:
            return
        li.configurer_port(int(port), int(fonction))
        self._ports_configures.add(cle)

    # -- alimentation de la file pendant l'execution -------------------------

    def _lancer_alimentation(self) -> None:
        """Envoie en arriere-plan ce qui n'a pas tenu dans la file."""
        if not self._reste:
            return
        if self._alimentation is not None and self._alimentation.is_alive():
            return                          # reprise apres suspension
        self._incident = ""
        self._alimentation = threading.Thread(
            target=self._alimenter,
            args=(self.liaison, self._reste, self._generation, self),
            name="alimentation-file-dobot", daemon=True)
        self._alimentation.start()

    @staticmethod
    def _alimenter(li: LiaisonDobot, reste: List[Callable[[], int]],
                   generation: int, pont: "PontReel") -> None:
        """
        Ecoule les commandes restantes au rythme ou la file se vide.

        On ne demande JAMAIS au controleur s'il a de la place : on compte.
        Tant que `dernier_index - index_courant` atteint la marge, on
        attend ; sinon on envoie. C'est ce comptage, et non la detection
        d'un refus, qui garantit qu'aucune commande ne se perd.

        S'arrete des que la file a ete videe par ailleurs (arret d'urgence,
        nouveau televersement, pilotage manuel) : ces commandes
        appartiennent alors a un programme abandonne. Pas de journal depuis
        ce fil, qui n'est pas celui de l'interface : les incidents sont
        deposes dans `pont._incident`, que l'IHM relit.
        """
        echecs = 0
        while reste:
            envoye = 0
            # verrou tenu entre le controle et l'envoi : un vidage ne peut
            # pas se glisser entre les deux
            with li._verrou:
                if li.generation_file != generation or not li.ouverte:
                    reste.clear()
                    return
                try:
                    libre = pont._places_libres(li)
                    while reste and libre > 0:
                        reste[0]()
                        reste.pop(0)
                        libre -= 1
                        envoye += 1
                    echecs = 0
                except FilePleine:
                    echecs = 0
                    pont._reduire_marge()   # la marge etait trop large
                except ErreurLiaison as exc:
                    # Une trame perdue n'est pas une liaison perdue : on
                    # reessaie. Abandonner ici laisserait le programme
                    # s'arreter en silence au milieu du cycle.
                    echecs += 1
                    if echecs >= PontReel.ECHECS_AVANT_ABANDON:
                        pont._incident = (
                            "alimentation de la file interrompue apres {} "
                            "echecs ({}) : {} commande(s) non envoyee(s)"
                            .format(echecs, exc, len(reste)))
                        reste.clear()
                        return
            time.sleep(0.02 if envoye else 0.08)

    def _places_libres(self, li: LiaisonDobot) -> int:
        """
        Combien de commandes la file peut encore recevoir sans deborder.

        Deux sources, dans cet ordre :

        1. le controleur lui-meme. GetQueuedCmdLeftSpace a ete RETIRE du
           protocole V1.1.5 (son historique, page 3), mais les firmwares
           le servent encore -- releve sur 4.0.3, qui repond 32 file vide
           et 0 file pleine. Quand il repond, on le croit : c'est la seule
           grandeur qui ne soit pas une deduction.
        2. sinon, l'ecart des index, borne par la marge.

        On garde la marge comme plafond dans les deux cas : elle protege
        du firmware qui repondrait une place optimiste.
        """
        place = li.place_restante()
        if place is not None:
            return max(0, min(int(place) - self.GARDE_FILE, self.marge))
        en_vol = li.dernier_index - li.index_courant()
        return max(0, self.marge - int(en_vol))

    def _reduire_marge(self) -> None:
        """
        Un refus prouve que la marge etait trop large : on la reduit.

        La profondeur de file n'est pas publiee par le protocole, et rien
        ne dit qu'elle soit la meme d'un firmware a l'autre. Quand le
        controleur refuse franchement, il nous renseigne : on en profite,
        plutot que de retomber indefiniment sur le meme mur.
        """
        nouvelle = max(self.MARGE_MINIMALE, self.marge // 2)
        if nouvelle < self.marge:
            self.marge = nouvelle
            self.journal("file du controleur plus courte que prevu : marge "
                         "ramenee a {} commandes".format(self.marge))

    # -- attente de fin -----------------------------------------------------

    def _reste_valide(self) -> bool:
        """Reste-t-il des commandes a envoyer pour le programme en file ?"""
        return bool(self._reste) and self.connecte \
            and self.liaison.generation_file == self._generation

    def programme_en_file(self) -> bool:
        """
        Le programme televerse reste-t-il (en partie) a executer ?

        Faux apres un vidage de la file, quel qu'en soit l'auteur : il
        faut alors televerser a nouveau pour rejouer le programme.
        """
        if not self.connecte or self.index_final <= 0:
            return False
        if self.liaison.generation_file != self._generation:
            return False
        return self._reste_valide() \
            or self.liaison.index_courant() < self.index_final

    def termine(self) -> bool:
        """Vrai quand le controleur a depasse la derniere commande envoyee."""
        if not self.connecte:
            return True
        if self._reste_valide():
            return False
        return self.liaison.index_courant() >= self.index_final

    def attendre_fin(self, periode: float = 0.2,
                     limite: float = 600.0) -> bool:
        """Bloque jusqu'a la fin du programme. Reserve aux scripts."""
        debut = time.time()
        while time.time() - debut < limite:
            if self.termine():
                return True
            time.sleep(periode)
        return False
