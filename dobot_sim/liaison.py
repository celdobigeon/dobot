#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocole de communication du Dobot Magician, en Python pur.

Implemente le protocole serie officiel decrit dans
`Dobot-Communication-Protocol-V1.1.5.pdf`, present a la racine du projet.
Aucune DLL, aucun DobotStudio : seul `pyserial` est necessaire, et le
module fonctionne donc sur Windows, Linux et macOS.

Trame
-----
    0xAA 0xAA | Len | ID | Ctrl | Params... | Checksum

  * `Len` vaut 2 + n, ou n est la taille des parametres : la charge utile
    compte l'ID et le Ctrl ;
  * `Ctrl` : bit 0 = rw (0 lire, 1 ecrire), bit 1 = isQueued ;
  * `Checksum` est le complement a deux de la somme de la charge utile,
    ID et Ctrl compris : `(-somme) & 0xFF` ;
  * les parametres sont en LITTLE ENDIAN (section 1.2 du protocole).

Deux familles de commandes
--------------------------
Les commandes IMMEDIATES s'executent tout de suite et ne renvoient rien de
particulier. Les commandes EN FILE (`isQueued = 1`) sont empilees dans le
controleur et renvoient un index sur 64 bits. C'est ce mecanisme qui permet
de televerser un programme entier puis de le lancer : on compare l'index
courant du controleur a l'index de la derniere commande envoyee pour savoir
ou en est l'execution.

Securite
--------
Ce module PILOTE UNE MACHINE REELLE. Rien ne bouge sans un appel explicite :
la connexion seule ne declenche aucun mouvement, et `LiaisonDobot` n'envoie
jamais de commande de son propre chef. L'arret d'urgence
(`arret_urgence()`) vide la file et coupe l'execution.
"""

from __future__ import annotations

import os
import struct
import sys
import threading
import time

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ErreurLiaison", "ErreurTrame", "FilePleine", "CommandeDobot", "PTP", "FonctionES",
    "construire_trame", "decoder_trame", "LiaisonDobot", "ports_disponibles",
    "BAUD", "EN_TETE",
]

#: Parametres de liaison, section 1.1 du protocole (USB - port serie).
BAUD = 115200
EN_TETE = b"\xAA\xAA"


class ErreurLiaison(RuntimeError):
    """Communication impossible : port absent, delai depasse, robot muet."""


class ErreurTrame(ErreurLiaison):
    """Trame recue invalide : en-tete, longueur ou checksum incoherents."""


class FilePleine(ErreurLiaison):
    """
    Commande en file REFUSEE : la file du controleur est pleine.

    Le controleur repond quand meme, mais avec des parametres vides au lieu
    de l'index (protocole V1.1.5, tableau 198 : "queuedCmdIndex or Empty").
    La commande n'est PAS stockee : il faut la renvoyer quand une place se
    libere, c'est-a-dire une fois la file demarree.
    """


class CommandeDobot:
    """
    Identifiants de commande, releves dans le protocole V1.1.5.

    Le numero entre parentheses renvoie a la section du document.
    """
    # -- informations appareil
    NOM_APPAREIL = 1            # 1.3.2
    VERSION = 2                 # 1.3.3
    # -- pose temps reel
    POSE = 10                   # 1.4.1  -> 8 float : x y z r J1 J2 J3 J4
    REINIT_POSE = 11            # 1.4.2  uint8 manuel + 2 float (bras, avant-bras)
    # -- calibration des capteurs d'angle
    ERREUR_CAPTEURS = 140       # 1.16.1  2 float : erreur bras, erreur avant-bras
    # -- alarmes
    ALARMES = 20                # 1.5
    EFFACER_ALARMES = 21        # 1.5
    # -- prise d'origine
    PARAMS_HOME = 30            # 1.6.1
    HOME = 31                   # 1.6.2  (file)
    # -- effecteur
    PARAMS_EFFECTEUR = 60       # 1.8.1
    LASER = 61                  # 1.8.2
    VENTOUSE = 62               # 1.8.3  (isCtrlEnable, isSucked)
    PINCE = 63                  # 1.8.4  (isCtrlEnable, isGripped)
    # -- JOG : mode manuel, IMMEDIAT (hors file) ; le mouvement dure tant
    # qu'on ne renvoie pas la commande 0
    PARAMS_JOG_ARTICULAIRE = 70   # 1.9.1  8 float : v[4] a[4]
    JOG = 73                      # 1.9.5  uint8 isJoint + uint8 cmd
    # -- PTP
    PARAMS_PTP_ARTICULAIRE = 80   # 1.10.1  8 float : v[4] a[4]
    PARAMS_PTP_CARTESIEN = 81     # 1.10.2  4 float : vxyz vr axyz ar
    PARAMS_PTP_SAUT = 82          # 1.10.3
    PARAMS_PTP_COMMUNS = 83       # 1.10.4  2 float : ratio v, ratio a
    PTP = 84                      # 1.10.5  (file) uint8 mode + 4 float
    # -- CP : trajectoire continue, les points s'enchainent SANS arret
    PARAMS_CP = 90              # 1.11.1  3 float + uint8 : planAcc,
    #                                     junctionVel, acc, realTimeTrack
    CP = 91                     # 1.11.2  (file) uint8 cpMode + 4 float :
    #                                     x, y, z, velocity
    # -- ARC
    PARAMS_ARC = 100            # 1.12.1
    ARC = 101                   # 1.12.2  (file) 8 float : via xyzr + cible xyzr
    # -- attente
    ATTENTE = 110               # 1.13.1  (file) uint32 ms
    # -- declenchement sur entree : c'est LUI qui realise "attendre EIOn",
    # la file n'ayant pas de commande d'attente conditionnelle
    TRIG = 120                  # 1.14.1  (file)
    # -- entrees / sorties
    ES_MULTIPLEXAGE = 130       # 1.15.1  (file) uint8 adresse, uint8 fonction
    ES_SORTIE = 131             # 1.15.2  (file) uint8 adresse, uint8 niveau
    ES_ENTREE = 133             # 1.15.4  -> uint8 adresse, uint8 niveau
    # -- controle de la file
    FILE_DEMARRER = 240         # 1.19.1
    FILE_ARRETER = 241          # 1.19.2
    FILE_ARRET_FORCE = 242      # 1.19.3
    FILE_VIDER = 245            # 1.19.6
    FILE_INDEX_COURANT = 246    # 1.19.7  -> uint64
    # GetQueuedCmdLeftSpace : RETIRE du protocole V1.1.5 (voir son
    # historique, page 3), mais toujours implemente par le firmware --
    # releve sur 4.0.3, qui repond 32 file vide et 0 file pleine. On s'en
    # sert quand il repond, et on s'en passe sinon.
    FILE_PLACE_RESTANTE = 247


class PTP:
    """Modes de `SetPTPCmd`, enumeration de la section 1.10.5."""
    SAUT_XYZ = 0
    MOVJ_XYZ = 1
    MOVL_XYZ = 2
    SAUT_ANGLE = 3
    MOVJ_ANGLE = 4
    MOVL_ANGLE = 5
    MOVJ_INC = 6
    MOVL_INC = 7
    MOVJ_XYZ_INC = 8
    SAUT_MOVL_XYZ = 9


class FonctionES:
    """Multiplexage d'un port EIO, enumeration `IOFunction` (Program 31)."""
    AUCUNE = 0
    PWM = 1
    SORTIE = 2
    ENTREE = 3
    ADC = 4


# ---------------------------------------------------------------------------
# Cadrage : des fonctions pures, testables sans materiel
# ---------------------------------------------------------------------------

def _somme_de_controle(charge: bytes) -> int:
    """Complement a deux de la somme de la charge utile (section 1.2.2)."""
    return (-sum(charge)) & 0xFF


def construire_trame(identifiant: int, params: bytes = b"",
                     ecriture: bool = False, en_file: bool = False) -> bytes:
    """
    Assemble une trame prete a emettre.

    GetPose : ID 10, aucun parametre, lecture immediate. La somme de la
    charge utile vaut 0x0A, son complement a deux 0xF6 -- c'est l'exemple
    meme de la section 1.2.2 du protocole.

    >>> construire_trame(10).hex().upper()
    'AAAA020A00F6'
    """
    if not 0 <= identifiant <= 255:
        raise ValueError("identifiant de commande hors [0 ; 255]")
    ctrl = (1 if ecriture else 0) | (2 if en_file else 0)
    charge = bytes((identifiant, ctrl)) + params
    if len(charge) > 255:
        raise ValueError("charge utile de {} octets : la longueur tient sur "
                         "un seul octet".format(len(charge)))
    return (EN_TETE + bytes((len(charge),)) + charge
            + bytes((_somme_de_controle(charge),)))


def decoder_trame(trame: bytes) -> Tuple[int, int, bytes]:
    """
    Verifie et decoupe une trame recue -> (identifiant, ctrl, parametres).

    Leve `ErreurTrame` sur un en-tete, une longueur ou un checksum faux.
    """
    if len(trame) < 6:
        raise ErreurTrame("trame de {} octets : il en faut au moins 6"
                          .format(len(trame)))
    if trame[:2] != EN_TETE:
        raise ErreurTrame("en-tete {} au lieu de AAAA"
                          .format(trame[:2].hex().upper()))
    longueur = trame[2]
    attendu = 3 + longueur + 1
    if len(trame) != attendu:
        raise ErreurTrame("longueur annoncee {} : {} octets attendus, {} recus"
                          .format(longueur, attendu, len(trame)))
    charge = trame[3:3 + longueur]
    if trame[-1] != _somme_de_controle(charge):
        raise ErreurTrame("checksum {:02X} au lieu de {:02X}"
                          .format(trame[-1], _somme_de_controle(charge)))
    return charge[0], charge[1], charge[2:]


def ports_disponibles() -> List[Tuple[str, str]]:
    """
    Ports serie presents -> [(peripherique, description), ...].

    Le Magician se presente par un pont USB-serie CH340
    (VID:PID = 1A86:7523) ; on remonte ces ports en tete de liste.
    """
    try:
        from serial.tools import list_ports
    except ImportError:                      # pragma: no cover
        return []

    trouves = []
    for p in list_ports.comports():
        hwid = (p.hwid or "").upper()
        probable = ("1A86:7523" in hwid or "CH340" in (p.description or "").upper()
                    or "SILABS" in hwid or "10C4:EA60" in hwid)
        trouves.append((0 if probable else 1, p.device, p.description or ""))
    trouves.sort()
    return [(d, desc) for _, d, desc in trouves]


def brltty_actif() -> bool:
    """
    Vrai si brltty tourne (Linux). Ce service pour afficheurs braille
    s'empare des ponts CH340 : le port du Dobot apparait puis disparait.
    """
    try:
        for pid in os.listdir("/proc"):
            if pid.isdigit():
                try:
                    with open("/proc/{}/comm".format(pid)) as f:
                        if f.read().strip().startswith("brltty"):
                            return True
                except OSError:
                    continue
    except OSError:
        pass
    return False


def _conseil_ouverture(exc: Exception) -> str:
    """Explication, selon le systeme, d'un echec d'ouverture du port."""
    detail = str(exc)
    bas = detail.lower()
    # Linux : le port existe mais l'utilisateur n'a pas le droit de l'ouvrir.
    # Meme texte que sous Windows ("Access is denied"), mais cause toute
    # autre : sous Windows, c'est le port deja pris.
    if sys.platform.startswith("linux") and (
            "errno 13" in bas or "permission denied" in bas):
        return ("\n  Votre compte n'a pas le droit d'utiliser ce port. Cela ne"
                "\n  se regle pas sans droits administrateur : signalez-le a"
                "\n  l'enseignant. Le poste doit etre configure une fois par"
                "\n  admin/configurer_poste_linux.sh.")
    if "errno 2" in bas or "no such file" in bas or "filenotfound" in bas:
        conseil = "\n  Le port n'existe plus : robot debranche ou eteint ?"
        if sys.platform.startswith("linux") and brltty_actif():
            conseil += ("\n  brltty tourne sur ce poste : il capture le port du"
                        "\n  Dobot. Signalez-le a l'enseignant (le poste doit"
                        "\n  etre configure par admin/configurer_poste_linux.sh).")
        return conseil
    # un port serie ne s'ouvre qu'une fois : l'erreur la plus frequente
    # n'est pas un robot absent, c'est un autre programme qui tient deja
    # la liaison (verrou exclusif sous Linux/macOS, refus sous Windows)
    if ("access" in bas or "acc" in bas or "denied" in bas
            or "lock" in bas or "busy" in bas):
        return ("\n  Le port est deja pris. Fermez ce qui l'occupe :"
                "\n    - une autre fenetre du simulateur, ou son "
                "panneau Robot reel (bouton Deconnecter)"
                "\n    - un 'calibrer_robot.py --suivre' en cours "
                "(Ctrl+C)"
                "\n    - DobotStudio")
    return ""


# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Situation temps reel renvoyee par `GetPose` (section 1.4.1)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    angles: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def depuis_octets(cls, params: bytes) -> "Pose":
        if len(params) < 32:
            raise ErreurTrame("GetPose : {} octets au lieu de 32"
                              .format(len(params)))
        v = struct.unpack("<8f", params[:32])
        return cls(v[0], v[1], v[2], v[3], (v[4], v[5], v[6], v[7]))


@dataclass
class LiaisonDobot:
    """
    Liaison serie avec un Dobot Magician reel.

    Usage :

        with LiaisonDobot("COM4") as robot:
            print(robot.version())
            print(robot.pose())

    Rien ne bouge tant qu'aucune commande de mouvement n'est appelee :
    ouvrir la liaison est une operation en LECTURE.
    """

    port: str
    baud: int = BAUD
    delai: float = 1.0                 # secondes, delai de reponse
    journal: Any = None                # callable(str), optionnel

    _serie: Any = field(default=None, init=False, repr=False)
    _verrou: Any = field(default_factory=threading.RLock, init=False, repr=False)
    _dernier_index: int = field(default=0, init=False)
    # le firmware repond-il encore a GetQueuedCmdLeftSpace ? passe a True
    # au premier refus, pour ne pas le redemander a chaque commande
    _place_restante_absente: bool = field(default=False, init=False)
    # incremente a chaque vidage : permet a qui alimente la file de savoir
    # qu'elle a ete videe sous ses pieds (arret d'urgence, pilotage...)
    generation_file: int = field(default=0, init=False)

    # -- cycle de vie -------------------------------------------------------

    def ouvrir(self) -> "LiaisonDobot":
        try:
            import serial
        except ImportError as exc:       # pragma: no cover
            raise ErreurLiaison(
                "le module pyserial est absent : pip install pyserial") from exc
        try:
            # exclusive : sous Linux et macOS, rien n'empeche par defaut deux
            # programmes d'ouvrir le meme port, et leurs trames se melent.
            # Le verrou fait echouer le second, comme Windows le fait de
            # lui-meme.
            self._serie = serial.Serial(self.port, self.baud, timeout=self.delai,
                                        write_timeout=self.delai, exclusive=True)
        except Exception as exc:
            raise ErreurLiaison("ouverture de {} impossible : {}{}".format(
                self.port, exc, _conseil_ouverture(exc))) from exc
        # Le pont USB-serie a besoin d'un instant, et le tampon peut contenir
        # les restes d'une session precedente.
        #
        # 0,15 s suffisait au CH340 ; un pont CP210x, lui, laisse la carte
        # muette un moment apres l'ouverture -- et une ouverture repetee
        # rapidement la rend muette tout court. D'ou l'attente plus large et
        # le reveil ci-dessous : on jette une premiere commande a la
        # poubelle, celle qui sert a la reveiller.
        time.sleep(0.6)
        self._serie.reset_input_buffer()
        self._serie.reset_output_buffer()
        self._reveiller()
        self._noter("liaison ouverte sur {} a {} bauds".format(self.port, self.baud))
        return self

    def _reveiller(self, essais: int = 4) -> None:
        """
        Interroge la carte jusqu'a ce qu'elle reponde, sans lever d'erreur.

        La premiere commande apres ouverture se perd souvent. Plutot que de
        la faire echouer chez l'appelant, on l'absorbe ici.
        """
        for k in range(essais):
            try:
                self.echanger(CommandeDobot.VERSION, essais=1)
                if k:
                    self._noter("carte reveillee apres {} tentative(s)"
                                .format(k + 1))
                return
            except ErreurLiaison:
                time.sleep(0.35)
        self._noter("la carte n'a pas repondu au reveil ; on tente quand meme")

    def fermer(self) -> None:
        with self._verrou:
            if self._serie is not None:
                try:
                    self._serie.close()
                finally:
                    self._serie = None
                self._noter("liaison fermee")

    @property
    def ouverte(self) -> bool:
        return self._serie is not None and getattr(self._serie, "is_open", False)

    def __enter__(self):
        return self.ouvrir()

    def __exit__(self, *_):
        self.fermer()

    def _noter(self, msg: str) -> None:
        if self.journal is not None:
            self.journal(msg)

    # -- couche transport ---------------------------------------------------

    def echanger(self, identifiant: int, params: bytes = b"",
                 ecriture: bool = False, en_file: bool = False,
                 essais: int = 3) -> bytes:
        """
        Envoie une commande et retourne les parametres de la reponse.

        Le protocole garantit une reponse a TOUTE commande, lecture comme
        ecriture : on peut donc se synchroniser sur elle.
        """
        if not self.ouverte:
            raise ErreurLiaison("liaison fermee")
        trame = construire_trame(identifiant, params, ecriture, en_file)
        derniere = None
        with self._verrou:
            for tentative in range(essais):
                try:
                    self._serie.reset_input_buffer()
                    self._serie.write(trame)
                    self._serie.flush()
                    rid, _ctrl, reponse = self._lire_trame()
                    if rid != identifiant:
                        raise ErreurTrame(
                            "reponse de la commande {} alors que {} etait "
                            "demandee".format(rid, identifiant))
                    if en_file:
                        if len(reponse) < 8:
                            # reponse valide, mais commande non stockee
                            raise FilePleine(
                                "commande {} refusee : file du controleur "
                                "pleine".format(identifiant))
                        index = struct.unpack("<Q", reponse[:8])[0]
                        # Mesure sur firmware 4.0.3 : la file pleine ne
                        # repond PAS vide. Elle repond huit octets d'allure
                        # normale, mais l'index N'AVANCE PLUS -- et la
                        # commande est jetee. Un index qui ne progresse pas
                        # est donc le vrai signal de refus ; sans lui, on
                        # croit avoir televerse un programme dont les trois
                        # quarts n'existent nulle part.
                        if index <= self._dernier_index:
                            raise FilePleine(
                                "commande {} jetee : file pleine (l'index "
                                "reste a {})".format(identifiant, index))
                        self._dernier_index = index
                    return reponse
                except FilePleine:
                    raise                   # renvoyer tout de suite ne sert a rien
                except ErreurLiaison as exc:
                    derniere = exc
                    time.sleep(0.05)
        raise ErreurLiaison("commande {} sans reponse valide apres {} essais : {}"
                            .format(identifiant, essais, derniere))

    def _lire_trame(self) -> Tuple[int, int, bytes]:
        """Resynchronise sur 0xAA 0xAA puis lit exactement la trame annoncee."""
        limite = time.time() + self.delai
        fenetre = b""
        while time.time() < limite:
            octet = self._serie.read(1)
            if not octet:
                continue
            fenetre = (fenetre + octet)[-2:]
            if fenetre == EN_TETE:
                entete = self._serie.read(1)
                if not entete:
                    break
                longueur = entete[0]
                reste = self._serie.read(longueur + 1)
                if len(reste) != longueur + 1:
                    raise ErreurTrame(
                        "trame tronquee : {} octets sur {} annonces"
                        .format(len(reste), longueur + 1))
                return decoder_trame(EN_TETE + entete + reste)
        raise ErreurLiaison("aucune reponse en {:.1f} s".format(self.delai))

    # -- lectures (aucun mouvement) ----------------------------------------

    def version(self) -> str:
        p = self.echanger(CommandeDobot.VERSION)
        if len(p) < 3:
            raise ErreurTrame("GetDeviceVersion : reponse de {} octets".format(len(p)))
        return "{}.{}.{}".format(p[0], p[1], p[2])

    def nom(self) -> str:
        return self.echanger(CommandeDobot.NOM_APPAREIL).decode("ascii", "replace") \
            .rstrip("\x00")

    def pose(self) -> Pose:
        """(x, y, z, r) en mm/deg et les quatre angles articulaires."""
        return Pose.depuis_octets(self.echanger(CommandeDobot.POSE))

    def entree(self, port: int) -> bool:
        """Niveau logique d'un port EIO configure en entree."""
        p = self.echanger(CommandeDobot.ES_ENTREE, struct.pack("<B", int(port)))
        if len(p) < 2:
            raise ErreurTrame("GetIODI : reponse de {} octets".format(len(p)))
        return bool(p[1])

    def alarmes(self) -> List[int]:
        """Indices des bits d'alarme leves. Liste vide = rien a signaler."""
        p = self.echanger(CommandeDobot.ALARMES)
        leves = []
        for octet, valeur in enumerate(p):
            for bit in range(8):
                if valeur & (1 << bit):
                    leves.append(octet * 8 + bit)
        return leves

    def index_courant(self) -> int:
        """Index de la commande en cours d'execution dans la file."""
        p = self.echanger(CommandeDobot.FILE_INDEX_COURANT)
        if len(p) < 8:
            raise ErreurTrame("GetQueuedCmdCurrentIndex : {} octets".format(len(p)))
        return struct.unpack("<Q", p[:8])[0]

    @property
    def dernier_index(self) -> int:
        """Index renvoye par la derniere commande mise en file."""
        return self._dernier_index

    def place_restante(self) -> Optional[int]:
        """
        Places libres dans la file, ou None si le firmware l'ignore.

        Le protocole V1.1.5 a retire cette commande, mais les firmwares
        la servent encore : 4.0.3 repond 32 file vide, 0 file pleine.
        Quand elle repond, c'est la grandeur la plus sure qui soit --
        elle vient du controleur lui-meme. Sinon on se rabat sur l'ecart
        des index, qui ne demande rien a personne.

        Le resultat est mis en cache : inutile de redemander a chaque
        commande si le firmware ne sait pas repondre.
        """
        if self._place_restante_absente:
            return None
        try:
            p = self.echanger(CommandeDobot.FILE_PLACE_RESTANTE, essais=1)
        except ErreurLiaison:
            self._place_restante_absente = True
            return None
        if len(p) < 4:
            self._place_restante_absente = True
            return None
        return int(struct.unpack("<I", p[:4])[0])

    # -- ecritures immediates ----------------------------------------------

    def effacer_alarmes(self) -> None:
        self.echanger(CommandeDobot.EFFACER_ALARMES, ecriture=True)

    # -- mode manuel (JOG) ---------------------------------------------------
    #
    # Immediat, hors file : il marche meme quand la file est bloquee par un
    # defaut de planification (alarme 0x12, firmware 3.7.0), ou aucun
    # mouvement programme ne part plus. cmd (articulaire) : 1/2 = J1+/-,
    # 3/4 = J2+/-, 5/6 = J3+/-, 7/8 = J4+/- ; 0 = arret.

    def jog(self, cmd: int, articulaire: bool = True) -> None:
        self.echanger(CommandeDobot.JOG, struct.pack("<BB", int(bool(articulaire)),
                                                     int(cmd)), ecriture=True)

    def parametres_jog_articulaire(self, vitesse: float, acceleration: float) -> None:
        """deg/s et deg/s2, les memes pour les quatre articulations."""
        self.echanger(CommandeDobot.PARAMS_JOG_ARTICULAIRE,
                      struct.pack("<8f", *([float(vitesse)] * 4 + [float(acceleration)] * 4)),
                      ecriture=True)

    # -- calibration des capteurs d'angle ----------------------------------
    #
    # Le protocole (section 1.16) : "Angle sensors of forearm and rear arm
    # may have a static offset due to angle sensor welding, machine status,
    # and so on." C'est exactement le defaut qui fait qu'un bras reel n'est
    # pas la ou le controleur le croit. Ces deux commandes ne provoquent
    # AUCUN mouvement : elles ne changent que l'interpretation des capteurs.

    def erreur_capteurs(self) -> Tuple[float, float]:
        """(erreur bras arriere, erreur avant-bras) en degres, persistante."""
        p = self.echanger(CommandeDobot.ERREUR_CAPTEURS)
        if len(p) < 8:
            raise ErreurTrame("GetAngleSensorStaticError : {} octets"
                              .format(len(p)))
        return struct.unpack("<2f", p[:8])

    def definir_erreur_capteurs(self, bras: float, avant_bras: float) -> None:
        """
        Ecrit l'erreur statique des capteurs d'angle. Ne fait rien bouger.

        Le reglage est PERSISTANT : il survit a la mise hors tension. Notez
        la valeur precedente avant d'ecrire, pour pouvoir revenir en arriere.
        """
        self.echanger(CommandeDobot.ERREUR_CAPTEURS,
                      struct.pack("<2f", float(bras), float(avant_bras)),
                      ecriture=True)

    def redefinir_pose(self, bras: float, avant_bras: float) -> None:
        """
        Declare les angles VRAIS du bras a sa posture actuelle (ResetPose).

        A utiliser apres avoir place le bras a une posture connue et
        verifiable. Effet immediat, aucun mouvement, mais NON persistant :
        la mise hors tension l'oublie. Pour un reglage durable, passez par
        `definir_erreur_capteurs`.
        """
        self.echanger(CommandeDobot.REINIT_POSE,
                      struct.pack("<B2f", 1, float(bras), float(avant_bras)),
                      ecriture=True)

    # -- point d'origine ----------------------------------------------------

    def point_origine(self) -> Tuple[float, float, float, float]:
        """
        Point ou le bras se parque a la fin de la prise d'origine.

        En coordonnees du CONTROLEUR. Attention, ces quatre flottants
        vivent en flash : ils peuvent etre corrompus. Un `nan` suffit a
        faire echouer toute la prise d'origine.
        """
        p = self.echanger(CommandeDobot.PARAMS_HOME)
        if len(p) < 16:
            raise ErreurTrame("GetHOMEParams : {} octets".format(len(p)))
        return struct.unpack("<4f", p[:16])

    def definir_point_origine(self, x: float, y: float, z: float,
                              r: float) -> None:
        """
        Fixe le point de parcage. PERSISTANT, mais ne fait rien bouger.

        Refuse une valeur non finie : c'est precisement ce genre de valeur
        qui fait echouer la prise d'origine, autant ne pas en ecrire une.
        """
        v = [float(x), float(y), float(z), float(r)]
        for nom, val in zip("xyzr", v):
            if not np.isfinite(val):
                raise ValueError(
                    "{} = {} : le point d'origine doit etre fini".format(nom, val))
        self.echanger(CommandeDobot.PARAMS_HOME, struct.pack("<4f", *v),
                      ecriture=True)

    def redefinir_pose_auto(self) -> None:
        """ResetPose automatique : le controleur recale seul (manual = 0)."""
        self.echanger(CommandeDobot.REINIT_POSE,
                      struct.pack("<B2f", 0, 0.0, 0.0), ecriture=True)

    def vider_file(self) -> None:
        with self._verrou:
            self.echanger(CommandeDobot.FILE_VIDER, ecriture=True)
            self._dernier_index = 0
            self.generation_file += 1

    def demarrer_file(self) -> None:
        self.echanger(CommandeDobot.FILE_DEMARRER, ecriture=True)

    def arreter_file(self) -> None:
        self.echanger(CommandeDobot.FILE_ARRETER, ecriture=True)

    def arret_urgence(self) -> None:
        """Coupe l'execution SANS finir le mouvement en cours, puis vide la file."""
        self.echanger(CommandeDobot.FILE_ARRET_FORCE, ecriture=True)
        self.vider_file()
        self._noter("ARRET D'URGENCE : file coupee et videe")

    def definir_vitesses(self, v_articulaire: Sequence[float],
                         a_articulaire: Sequence[float],
                         v_cartesienne: float, a_cartesienne: float,
                         v_rotation: float = 100.0,
                         a_rotation: float = 100.0) -> None:
        """Vitesses et accelerations maximales, articulaires et cartesiennes."""
        self.echanger(CommandeDobot.PARAMS_PTP_ARTICULAIRE,
                      struct.pack("<8f", *v_articulaire, *a_articulaire),
                      ecriture=True)
        self.echanger(CommandeDobot.PARAMS_PTP_CARTESIEN,
                      struct.pack("<4f", float(v_cartesienne), float(v_rotation),
                                  float(a_cartesienne), float(a_rotation)),
                      ecriture=True)

    def definir_ratio(self, vitesse_pct: float, acceleration_pct: float) -> None:
        """
        Ratio global applique a toutes les vitesses PTP, en pourcentage.

        C'est le reglage de securite le plus utile : 20 ou 30 % pour une
        premiere execution, 100 % une fois le programme valide.
        """
        self.echanger(CommandeDobot.PARAMS_PTP_COMMUNS,
                      struct.pack("<2f", float(vitesse_pct),
                                  float(acceleration_pct)), ecriture=True)

    def configurer_port(self, port: int, fonction: int) -> None:
        """Multiplexage d'un port EIO : entree, sortie, PWM..."""
        self.echanger(CommandeDobot.ES_MULTIPLEXAGE,
                      struct.pack("<2B", int(port), int(fonction)),
                      ecriture=True)

    # -- commandes de mouvement (mises en file) -----------------------------

    def prise_origine(self) -> int:
        """
        Prise d'origine. LE ROBOT PARCOURT TOUTE SA COURSE : degagez la zone.
        """
        self._noter("prise d'origine demandee")
        self.echanger(CommandeDobot.HOME, struct.pack("<I", 0),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def ptp(self, mode: int, x: float, y: float, z: float, r: float) -> int:
        self.echanger(CommandeDobot.PTP,
                      struct.pack("<B4f", int(mode), float(x), float(y),
                                  float(z), float(r)),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def arc(self, via: Sequence[float], cible: Sequence[float]) -> int:
        """Arc de cercle : point de passage puis point d'arrivee, (x, y, z, r)."""
        self.echanger(CommandeDobot.ARC,
                      struct.pack("<8f", *(float(v) for v in via),
                                  *(float(v) for v in cible)),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def parametres_cp(self, acceleration: float, vitesse: float) -> int:
        """
        Reglage du mode CP, MIS EN FILE : il prend effet entre deux points.

        Mesure sur la machine (firmware 3.7.0), ce qui contredit la lecture
        naive du protocole :

          * la vitesse de croisiere est `junctionVel` (ici `vitesse`), pas
            le champ `velocity` de SetCPCmd, que le controleur IGNORE ;
          * mis en file, ce reglage change la vitesse au bon moment de la
            trajectoire : aller demande a 30 mm/s, 4,03 s pour 4,00 ;
          * l'acceleration decide si la vitesse est atteinte, sur les
            droites comme dans les courbes.

        `realTimeTrack` reste a 0 : le mode "temps reel" (1) a ete essaye,
        il est lent et fait greziller les moteurs.
        """
        a = float(acceleration)
        self.echanger(CommandeDobot.PARAMS_CP,
                      struct.pack("<3fB", a, float(vitesse), a, 0),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def cp(self, x: float, y: float, z: float, vitesse: float = 0.0) -> int:
        """
        Point de trajectoire continue, en coordonnees ABSOLUES du controleur.

        Le robot y passe sans s'arreter si le point suivant est deja en
        file. Pas de r : l'outil ne tourne pas en CP. `vitesse` est
        transmise par fidelite au protocole, mais le controleur l'ignore
        (voir `parametres_cp`).
        """
        self.echanger(CommandeDobot.CP,
                      struct.pack("<B4f", 1, float(x), float(y), float(z),
                                  float(vitesse)),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def attendre(self, secondes: float) -> int:
        self.echanger(CommandeDobot.ATTENTE,
                      struct.pack("<I", max(int(round(secondes * 1000)), 0)),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def attendre_entree(self, port: int, niveau: bool) -> int:
        """
        Suspend la file jusqu'a ce que le port EIO prenne le niveau voulu.

        Realise par TRIG (section 1.14.1) en mode niveau : la file n'a pas
        de commande d'attente conditionnelle, TRIG en tient lieu.
        `condition = 0` signifie "egal au seuil".
        """
        self.echanger(CommandeDobot.TRIG,
                      struct.pack("<3BH", int(port), 0, 0, 1 if niveau else 0),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def biais_effecteur(self) -> Tuple[float, float, float]:
        """
        (xBias, yBias, zBias) configures dans le CONTROLEUR, en mm.

        Le controleur calcule sa propre pose avec ces valeurs : xBias y
        joue le role du deport l3 du modele. Les lire est indispensable
        pour convertir correctement les coordonnees.
        """
        p = self.echanger(CommandeDobot.PARAMS_EFFECTEUR)
        if len(p) >= 12:
            return struct.unpack("<3f", p[:12])
        # Certains firmwares (3.3.0 constate) ne renvoient que xBias, sur
        # 4 octets. Verifie sur machine : la pose annoncee par le controleur
        # s'explique exactement avec ce xBias et yBias = zBias = 0.
        if len(p) >= 4:
            return struct.unpack("<f", p[:4])[0], 0.0, 0.0
        raise ErreurTrame("GetEndEffectorParams : {} octets".format(len(p)))

    def sortie(self, port: int, niveau: bool) -> int:
        self.echanger(CommandeDobot.ES_SORTIE,
                      struct.pack("<2B", int(port), 1 if niveau else 0),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def ventouse(self, aspire: bool, active: bool = True) -> int:
        self.echanger(CommandeDobot.VENTOUSE,
                      struct.pack("<2B", 1 if active else 0, 1 if aspire else 0),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def pince(self, fermee: bool, active: bool = True) -> int:
        self.echanger(CommandeDobot.PINCE,
                      struct.pack("<2B", 1 if active else 0, 1 if fermee else 0),
                      ecriture=True, en_file=True)
        return self._dernier_index

    def couper_pompe(self) -> None:
        """
        Arrete la pompe a air de la ventouse ET de la pince, tout de suite.

        Ouvrir la pince ou relacher la ventouse laisse la pompe EN MARCHE
        (isCtrlEnable = 1 : elle souffle). Seul isCtrlEnable = 0 l'arrete.
        Commandes IMMEDIATES, hors file : elles agissent meme pendant un
        mouvement, et sans attendre la fin du programme.
        """
        self.echanger(CommandeDobot.VENTOUSE, struct.pack("<2B", 0, 0),
                      ecriture=True)
        self.echanger(CommandeDobot.PINCE, struct.pack("<2B", 0, 0),
                      ecriture=True)

    def laser(self, allume: bool, active: bool = True) -> int:
        self.echanger(CommandeDobot.LASER,
                      struct.pack("<2B", 1 if active else 0, 1 if allume else 0),
                      ecriture=True, en_file=True)
        return self._dernier_index


def resume_alarmes(indices: Sequence[int]) -> List[str]:
    """Traduit les bits d'alarme connus en phrases lisibles."""
    connus: Dict[int, str] = {
        0x00: "reinitialisation du controleur",
        0x01: "commande indefinie",
        0x02: "erreur du systeme de fichiers",
        0x03: "communication MCU-FPGA en echec a l'initialisation",
        0x04: "capteur d'angle : valeur aberrante",
        # planification : refus AVANT le mouvement
        0x10: "planif. : point de depart ou d'arrivee singulier",
        0x11: "planif. : cible hors de l'espace de travail",
        0x12: "planif. : la cible sort des butees articulaires",
        0x13: "planif. : points repetes (ARC ou JUMP)",
        0x14: "planif. : parametres d'arc incorrects",
        0x15: "planif. : parametres JUMP incorrects",
        0x16: "planif. : changement de bras sur une droite",
        0x17: "planif. : la droite sort de l'espace de travail",
        0x18: "planif. : l'arc sort de l'espace de travail",
        0x19: "planif. : type de mouvement invalide",
        0x1A: "planif. : parametres de vitesse incorrects",
        0x1B: "planif. : calcul CP en echec",
        # mouvement : defaut PENDANT le deplacement
        0x20: "mouvement : singularite rencontree",
        0x21: "mouvement : sortie de l'espace de travail",
        0x22: "mouvement : butee articulaire atteinte",
    }
    for k, j in enumerate(("J1", "J2", "J3", "J4")):
        connus[0x30 + k] = "{} en survitesse".format(j)
        connus[0x40 + 2 * k] = "{} en butee positive".format(j)
        connus[0x41 + 2 * k] = "{} en butee negative".format(j)
        connus[0x50 + k] = "{} a perdu des pas".format(j)
        for d, quoi in enumerate(("variateur en defaut", "depassement",
                                  "erreur de poursuite")):
            connus[0x60 + 3 * k + d] = "{} : {}".format(j, quoi)
    connus[0x48] = "parallelogramme (J2+J3) en butee haute"
    connus[0x49] = "parallelogramme (J2+J3) en butee basse"
    for base, moteur in ((0x70, "moteur arriere (J2)"),
                         (0x80, "moteur avant (J3)"),
                         (0x90, "moteur Z"), (0xA0, "moteur R")):
        for i in range(0x0F):
            connus[base + i] = "{} : defaut materiel".format(moteur)
    for i in range(3):
        connus[0xB0 + i] = "E/S de bout de bras : defaut materiel"
    # Le protocole V1.1.5 ne contient PAS cette table : sa section 1.5
    # renvoie au "Dobot Magician ALARM Description", absent du projet. Les
    # codes ci-dessus suivent ce document tel que le reprend la bibliotheque
    # pydobot ; ils sont confirmes sur machine pour 0x43 / 0x44 (J2 a -9 deg).
    return ["alarme 0x{:02X} : {}".format(
        i, connus.get(i, "code inconnu ; voir le document Dobot Magician "
                         "ALARM Description"))
            for i in indices]
