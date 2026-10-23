#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Passerelle entre un script Python et un simulateur DEJA ouvert.

Le probleme qu'elle resout : lancer un script depuis VS Code demarre un
NOUVEAU processus Python. Il ne voit pas la fenetre deja ouverte, en ouvre
donc une seconde -- et le port serie, tenu par la premiere, lui est
refuse. Il fallait fermer le simulateur avant chaque execution.

Desormais `Pilote.simuler()` frappe d'abord a la porte : si un simulateur
tourne, il lui envoie le programme et s'arrete la. Sinon il ouvre une
fenetre, comme avant.

Le transport est un socket local -- un tube nomme sous Windows, un socket
UNIX ailleurs. Pas de port TCP, donc rien a ouvrir dans un pare-feu et
aucun risque de collision avec un autre logiciel.

Le message est du JSON prefixe de sa longueur sur 4 octets :

    <taille big-endian sur 4 octets><JSON en UTF-8>

et contient le programme serialise, la configuration de depart, et un
drapeau disant s'il faut lancer la lecture.
"""

from __future__ import annotations

import getpass
import importlib
import json
import os
import struct
from typing import Any, Callable, Dict, Optional

__all__ = ["nom_service", "envoyer", "Serveur", "ATTENTE", "reseau",
           "VARIABLE_SERVICE"]

#: delai d'attente, en millisecondes, pour joindre un simulateur
ATTENTE = 800


def reseau():
    """
    Module QtNetwork du binding Qt reellement utilise.

    `pyqtgraph.Qt` ne le reexporte pas, mais il dit quel binding est en
    place : on le charge par la, ce qui marche avec PyQt5 comme avec
    PySide. Retourne None si le module manque -- la passerelle est alors
    simplement indisponible, et chacun ouvre sa fenetre comme avant.
    """
    try:
        from pyqtgraph.Qt import QT_LIB

        return importlib.import_module(QT_LIB + ".QtNetwork")
    except Exception:                        # pragma: no cover
        return None


#: variable d'environnement qui isole une passerelle des autres
VARIABLE_SERVICE = "DOBOT_SIM_SERVICE"


def nom_service() -> str:
    """
    Nom du tube, propre a l'utilisateur.

    Deux comptes ouvrant chacun un simulateur sur la meme machine ne
    doivent pas se marcher dessus.

    `DOBOT_SIM_SERVICE` permet de choisir un autre nom : deux simulateurs
    independants cohabitent alors sur le meme compte -- deux postes de TP
    sur une machine partagee, ou une recette qui ne doit pas parler a la
    fenetre ouverte a cote.
    """
    choisi = os.environ.get(VARIABLE_SERVICE, "").strip()
    if choisi:
        return "dobot_sim_{}".format(
            "".join(c for c in choisi if c.isalnum() or c in "_-") or "x")
    try:
        qui = getpass.getuser()
    except Exception:                        # pragma: no cover
        qui = "defaut"
    return "dobot_sim_{}".format("".join(c for c in qui if c.isalnum()) or "x")


def _cadrer(charge: Dict[str, Any]) -> bytes:
    donnees = json.dumps(charge, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(donnees)) + donnees


def demander(charge: Dict[str, Any], attente: int = ATTENTE) -> Optional[Any]:
    """
    Transmet une demande au simulateur en cours et rend sa reponse.

    None s'il n'y a pas de simulateur, ou s'il a refuse ; sinon ce que la
    fenetre a repondu ({} pour un simple accuse). Ne leve jamais :
    l'absence de simulateur est le cas NORMAL, celui du premier lancement.
    """
    QtNetwork = reseau()
    if QtNetwork is None:                    # pragma: no cover
        return None

    socket = QtNetwork.QLocalSocket()
    socket.connectToServer(nom_service())
    if not socket.waitForConnected(attente):
        return None
    try:
        socket.write(_cadrer(charge))
        if not socket.waitForBytesWritten(attente):
            return None
        # on attend l'accuse : sans lui on ne saurait pas si la fenetre a
        # vraiment pris le programme, et on ouvrirait peut-etre un doublon
        socket.waitForReadyRead(attente)
        reponse = bytes(socket.readAll())
        while socket.waitForReadyRead(50):   # une reponse longue arrive
            reponse += bytes(socket.readAll())   # parfois en plusieurs fois
        if not reponse.startswith(b"ok"):
            return None
        corps = reponse[2:].strip()
        try:
            return json.loads(corps.decode("utf-8")) if corps else {}
        except ValueError:
            return {}
    finally:
        socket.disconnectFromServer()


def envoyer(charge: Dict[str, Any], attente: int = ATTENTE) -> bool:
    """Transmet un programme au simulateur en cours. Faux s'il n'y en a pas."""
    return demander(charge, attente) is not None


class Serveur:
    """
    Ecoute les programmes envoyes par les scripts, pour la fenetre.

        self._serveur = Serveur(self._recevoir_programme, journal=self.journal)

    `rappel` recoit le dictionnaire transmis. S'il leve, l'expediteur en
    est informe et ouvrira sa propre fenetre plutot que de rester muet.
    """

    def __init__(self, rappel: Callable[[Dict[str, Any]], None],
                 journal: Optional[Callable[[str], None]] = None):
        self.rappel = rappel
        self.journal = journal or (lambda _m: None)
        self.serveur = None
        QtNetwork = reseau()
        if QtNetwork is None:                # pragma: no cover
            self.journal("passerelle indisponible : QtNetwork est absent")
            return
        self.serveur = QtNetwork.QLocalServer()
        self._tampons: Dict[Any, bytes] = {}

        nom = nom_service()
        if not self.serveur.listen(nom):
            # Un tube orphelin traine, laisse par un simulateur ferme
            # brutalement. On ne l'enleve QUE si personne ne repond
            # derriere : sinon on volerait sa place a une fenetre vivante.
            if not envoyer({"ping": True}, attente=200):
                QtNetwork.QLocalServer.removeServer(nom)
                self.serveur.listen(nom)
        if self.serveur.isListening():
            self.serveur.newConnection.connect(self._nouvelle_connexion)
            self.journal("passerelle a l'ecoute : les scripts lances depuis "
                         "l'exterieur enverront leur programme ici")
        else:                                # pragma: no cover
            self.journal("passerelle indisponible : un script ouvrira sa "
                         "propre fenetre")

    @property
    def actif(self) -> bool:
        return self.serveur is not None and self.serveur.isListening()

    def fermer(self) -> None:
        if self.actif:
            self.serveur.close()

    # -- reception ----------------------------------------------------------

    def _nouvelle_connexion(self) -> None:
        socket = self.serveur.nextPendingConnection()
        if socket is None:
            return
        self._tampons[socket] = b""
        socket.readyRead.connect(lambda s=socket: self._lire(s))
        socket.disconnected.connect(lambda s=socket: self._tampons.pop(s, None))

    def _lire(self, socket) -> None:
        self._tampons[socket] = self._tampons.get(socket, b"") \
            + bytes(socket.readAll())
        tampon = self._tampons[socket]
        if len(tampon) < 4:
            return
        taille = struct.unpack(">I", tampon[:4])[0]
        if len(tampon) < 4 + taille:
            return                           # message encore incomplet

        self._tampons[socket] = tampon[4 + taille:]
        try:
            charge = json.loads(tampon[4:4 + taille].decode("utf-8"))
        except Exception as exc:
            socket.write(b"erreur: " + str(exc).encode("utf-8", "replace"))
            socket.flush()
            return

        if charge.get("ping"):               # simple test de presence
            socket.write(b"ok")
            socket.flush()
            return
        try:
            reponse = self.rappel(charge)
            # un rappel peut rendre une valeur (etat d'une execution...),
            # transmise en JSON derriere l'accuse
            socket.write(b"ok" if reponse is None else
                         b"ok " + json.dumps(reponse).encode("utf-8"))
        except Exception as exc:             # pragma: no cover
            self.journal("programme recu mais refuse : {}".format(exc))
            socket.write(b"erreur: " + str(exc).encode("utf-8", "replace"))
        socket.flush()
