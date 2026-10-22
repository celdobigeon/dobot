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
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .liaison import (CommandeDobot, FonctionES, LiaisonDobot, PTP,
                      resume_alarmes)
from .robot import decomposer_cible

__all__ = ["PontReel", "ErreurPont", "EtatReel"]


class ErreurPont(RuntimeError):
    """Le programme ne peut pas etre transmis en l'etat."""


#: capacite de la file du controleur, MESUREE (firmware 3.7.0) : au-dela,
#: la commande est acceptee en apparence mais jetee, avec le meme index
CAPACITE_FILE = 32
#: places laissees libres, par prudence
MARGE_FILE = 2


class _Enregistreur(LiaisonDobot):
    """
    Fausse liaison qui ENREGISTRE les trames au lieu de les envoyer.

    Le pont traduit le programme comme d'habitude (memes methodes ptp,
    arc, ventouse...), mais vers cette liaison : on obtient la liste
    exacte des commandes, dans l'ordre, prete a etre versee au controleur.
    """

    def __init__(self):
        super().__init__(port="(enregistrement)")
        self.commandes: List[Tuple[int, bytes, bool, bool]] = []

    @property
    def ouverte(self) -> bool:
        return True

    def echanger(self, identifiant: int, params: bytes = b"",
                 ecriture: bool = False, en_file: bool = False,
                 essais: int = 3) -> bytes:
        self.commandes.append((int(identifiant), bytes(params),
                               bool(ecriture), bool(en_file)))
        if en_file:
            self._dernier_index += 1
        return struct.pack("<Q", self._dernier_index)


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

    def __init__(self, robot, journal: Optional[Callable[[str], None]] = None):
        self.robot = robot                    # DobotMagician du simulateur
        self.journal = journal or (lambda _m: None)
        self.liaison: Optional[LiaisonDobot] = None
        self.biais: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.index_final = 0
        self._index_depart = 0
        self._ports_configures: set = set()
        # commandes preparees mais pas encore dans la file du controleur
        self._envois: List[Tuple[int, bytes, bool, bool]] = []
        self._dernier_envoye = 0

    # -- connexion ----------------------------------------------------------

    @property
    def connecte(self) -> bool:
        return self.liaison is not None and self.liaison.ouverte

    def connecter(self, port: str) -> str:
        """Ouvre la liaison et releve la configuration. Ne fait RIEN bouger."""
        self.deconnecter()
        self.liaison = LiaisonDobot(port, journal=self.journal).ouvrir()
        # tout vidage de la file, d'ou qu'il vienne -- arret d'urgence,
        # pilotage, repli, prise d'origine --, abandonne aussi la reserve :
        # la suite d'un ancien programme ne doit jamais repartir
        self.liaison.sur_vidage = self._abandonner_envois
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
        self._abandonner_envois()
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
        if self.connecte:
            self.liaison.arret_urgence()

    def demarrer(self) -> None:
        self._exiger().demarrer_file()
        self.journal("execution reelle lancee")

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
        Vide la file et y depose tout le programme. NE LANCE RIEN.

        Retourne l'index de la derniere commande. Un programme dont le
        controle signale une butee ou une collision est REFUSE, sauf
        `forcer=True`.
        """
        li = self._exiger()

        traj = programme.construire(self.robot, q_depart)
        defauts = [a for a in traj.controler(0.05)
                   if any(m in a for m in ("hors de [", "hors d'atteinte",
                                           "plan de pose", "embase :",
                                           "colonne :"))]
        if defauts and not forcer:
            raise ErreurPont(
                "programme refuse : {} defaut(s) sur la trajectoire.\n  {}"
                .format(len(defauts), "\n  ".join(defauts[:6])))

        if any(getattr(i, "lissage", 0) for i in programme.instructions):
            self.journal("NOTE : le lissage n'existe pas en PTP sur le "
                         "controleur ; le robot reel marquera un arret a "
                         "chaque point.")

        # ORDRE CRITIQUE : arreter AVANT de vider. Si la file etait restee en
        # mode "demarre", chaque commande televersee partirait des son arrivee
        # et le robot se mettrait a bouger pendant le remplissage.
        li.arreter_file()
        li.vider_file()
        self._index_depart = li.index_courant()
        n = len(programme.instructions)

        # La file du controleur ne tient que CAPACITE_FILE commandes, et au-
        # dela il JETTE sans rien dire (il renvoie le meme index). On
        # prepare donc tout le programme, puis on le verse au fil de l'eau :
        # alimenter() recomplete la file a mesure qu'elle se vide.
        enreg = _Enregistreur()
        for k, ins in enumerate(programme.instructions):
            self._emettre(enreg, ins, programme, outil)
            if progression is not None:
                progression(k + 1, n)
        self._envois = list(enreg.commandes)
        self._dernier_envoye = self._index_depart
        en_file = sum(1 for c in self._envois if c[3])
        self.index_final = self._index_depart + en_file

        self.alimenter()
        reste = len(self._envois)
        self.journal("programme televerse : {} instructions, {} commandes, "
                     "index {} -> {}{}".format(
                         n, en_file, self._index_depart, self.index_final,
                         "" if not reste else
                         " ; {} en reserve, versees pendant l'execution"
                         .format(reste)))
        return self.index_final

    # -- versement au fil de l'eau -----------------------------------------

    @property
    def en_reserve(self) -> int:
        """Commandes preparees qui n'ont pas encore pu entrer dans la file."""
        return len(self._envois)

    def _abandonner_envois(self) -> None:
        if self._envois:
            self.journal("{} commande(s) en reserve abandonnee(s)"
                         .format(len(self._envois)))
        self._envois = []

    def alimenter(self) -> int:
        """
        Recomplete la file du controleur avec les commandes en reserve.

        A appeler regulierement pendant l'execution ; sans reserve, ne fait
        rien. Retourne le nombre de commandes envoyees.

        Deux garde-fous : on ne depasse pas CAPACITE_FILE - MARGE_FILE
        commandes en attente, et si le controleur renvoie un index qui
        n'avance pas, c'est qu'il a jete la commande -- elle reste alors en
        reserve pour le prochain passage.
        """
        if not self._envois or not self.connecte:
            return 0
        li = self.liaison
        courant = li.index_courant()
        envoyees = 0
        while self._envois:
            ident, params, ecriture, en_file = self._envois[0]
            if en_file and (self._dernier_envoye - courant
                            >= CAPACITE_FILE - MARGE_FILE):
                break
            reponse = li.echanger(ident, params, ecriture=ecriture,
                                  en_file=en_file)
            if en_file:
                idx = (struct.unpack("<Q", reponse[:8])[0]
                       if len(reponse) >= 8 else None)
                if idx is None or idx <= self._dernier_envoye:
                    break                    # jetee : file pleine
                self._dernier_envoye = idx
            self._envois.pop(0)
            envoyees += 1
        return envoyees

    def _emettre(self, li: LiaisonDobot, ins, programme, outil: str) -> int:
        """Traduit UNE instruction en commande(s) de file."""
        t = ins.type

        if t in ("MOVJ", "MOVL"):
            self._regler_cadence(li, ins)
            if getattr(ins, "mode", "cartesien") == "articulaire":
                q = [float(v) for v in ins.q]
                mode = PTP.MOVJ_ANGLE if t == "MOVJ" else PTP.MOVL_ANGLE
                return li.ptp(mode, *q)
            x, y, z, r = self._cible(ins, programme)
            mode = PTP.MOVJ_XYZ if t == "MOVJ" else PTP.MOVL_XYZ
            return li.ptp(mode, x, y, z, r)

        if t == "MOVC":
            self._regler_cadence(li, ins)
            via = self.vers_controleur(*self._pose_de(ins.matrice_via(
                programme.repere_objet)))
            cible = self._cible(ins, programme)
            return li.arc(via, cible)

        if t == "ATTENTE":
            return li.attendre(float(ins.duree))

        if t == "SORTIE":
            self._assurer_port(li, ins.port, FonctionES.SORTIE)
            return li.sortie(int(ins.port), bool(ins.etat))

        if t == "ENTREE":
            self._assurer_port(li, ins.port, FonctionES.ENTREE)
            return li.attendre_entree(int(ins.port), bool(ins.etat))

        if t == "OUTIL":
            actif = bool(ins.etat)
            if outil == "pince":
                return li.pince(actif)
            if outil == "laser":
                return li.laser(actif)
            return li.ventouse(actif)

        self.journal("instruction {} ignoree : sans equivalent sur le "
                     "controleur".format(t))
        return li.dernier_index

    def _cible(self, ins, programme) -> Tuple[float, float, float, float]:
        T = ins.matrice(self.robot, programme.repere_objet)
        return self.vers_controleur(*self._pose_de(T))

    def _pose_de(self, T) -> Tuple[float, float, float, float]:
        """Matrice homogene -> (x, y, z, r) du simulateur."""
        return tuple(float(v) for v in decomposer_cible(T))

    def _regler_cadence(self, li: LiaisonDobot, ins) -> None:
        """
        Pose vitesse et acceleration juste avant le mouvement.

        Les commandes de parametres sont elles-memes MISES EN FILE : le
        reglage prend donc effet au bon moment de la sequence, et non a
        l'instant du televersement.

        L'unite decide de la famille : un MOVJ s'exprime en deg/s et vise
        les parametres articulaires, un MOVL en mm/s et vise les
        parametres cartesiens.
        """
        v = float(getattr(ins, "vitesse", 0.0) or 0.0)
        a = float(getattr(ins, "acceleration", 0.0) or 0.0)
        if v <= 0.0 or a <= 0.0:
            return
        if getattr(ins, "unite", "mm") == "deg":
            li.echanger(CommandeDobot.PARAMS_PTP_ARTICULAIRE,
                        struct.pack("<8f", v, v, v, v, a, a, a, a),
                        ecriture=True, en_file=True)
        else:
            li.echanger(CommandeDobot.PARAMS_PTP_CARTESIEN,
                        struct.pack("<4f", v, 100.0, a, 100.0),
                        ecriture=True, en_file=True)

    def _assurer_port(self, li: LiaisonDobot, port: int, fonction: int) -> None:
        """Configure le multiplexage d'un port EIO, une seule fois."""
        cle = (int(port), int(fonction))
        if cle in self._ports_configures:
            return
        li.configurer_port(int(port), int(fonction))
        self._ports_configures.add(cle)

    # -- attente de fin -----------------------------------------------------

    def termine(self) -> bool:
        """Vrai quand le controleur a depasse la derniere commande envoyee."""
        if not self.connecte:
            return True
        return self.liaison.index_courant() >= self.index_final

    def attendre_fin(self, periode: float = 0.2,
                     limite: float = 600.0) -> bool:
        """Bloque jusqu'a la fin du programme. Reserve aux scripts."""
        debut = time.time()
        while time.time() - debut < limite:
            self.alimenter()
            if self.termine():
                return True
            time.sleep(periode)
        return False
