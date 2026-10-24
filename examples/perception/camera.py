# -*- coding: utf-8 -*-
"""
La camera RealSense, ouverte UNE fois et partagee.

Sous macOS, ouvrir la camera demande les droits administrateur (sudo), et
chaque arret-redemarrage du flux risque de la bloquer. D'ou un SERVEUR :
serveur_camera.py ouvre la camera, la garde ouverte, et sert des captures
aux autres scripts -- detection, etalonnage, boucle de prise --, qui n'ont
alors besoin ni de sudo ni de rouvrir quoi que ce soit.

Ce module contient :
  - les reglages communs de la camera (densite de mesure, exposition,
    puissance du projecteur) et la mediane de plusieurs images ;
  - le format d'echange : une capture voyage comme un fichier .npz en
    memoire, precede de sa longueur ;
  - le CLIENT : `capturer()` demande une capture fraiche au serveur.

Une capture est un dictionnaire : profondeur (m, float32, 0 = pas de
mesure), infrarouge (uint8, aligne sur la profondeur), K (intrinseques),
coeffs (distorsion), et, si le serveur a la couleur : couleur (recalee sur
la profondeur, pixel pour pixel) et couleur_brute (entiere, non recalee).

Avec couleur_brute, le serveur joint sa GEOMETRIE (sans elle, l'image
entiere ne peut pas etre reliee a la profondeur) :
  K_couleur             3 x 3, intrinseques de la camera couleur ;
  coeffs_couleur        (5,), sa distorsion (souvent nulle) ;
  modele_couleur        texte, le modele de distorsion RealSense de ces
                        coefficients (ex. "distortion.inverse_brown_conrady") ;
  T_couleur_profondeur  4 x 4, metres : un point p du repere de la
                        profondeur est en T_couleur_profondeur @ [p, 1] dans
                        celui de la couleur, puis se projette par K_couleur.
Les captures d'un serveur plus ancien n'ont pas ces cles : la couleur
recalee, deja dans le repere de la profondeur, sert alors (K, identite).
"""

from __future__ import annotations

import io
import json
import socket
import struct
from typing import Dict, Optional

import numpy as np

#: le serveur n'ecoute QUE la machine elle-meme
HOTE = "127.0.0.1"
PORT = 47321

LARGEUR, HAUTEUR, CADENCE = 640, 480, 15
#: images combinees par capture : la mediane efface l'essentiel du bruit
N_IMAGES = 15
#: mW. Defaut de la camera : 150. Moins fort, le projecteur sature moins le
#: bois clair, et il reste assez de motif sur la table.
PUISSANCE_LASER = 90


class ErreurCamera(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Reglages et traitement (cote camera)
# ---------------------------------------------------------------------------

def regler_capteur(rs, profil) -> float:
    """Reglages communs ; rend l'echelle de profondeur (m par unite)."""
    capteur = profil.get_device().first_depth_sensor()
    for option, valeur in (
            # "haute densite" : le plus de pixels mesures possible ; "haute
            # precision" en laisse trop vides sur une table sombre
            (rs.option.visual_preset, 4),
            # bois clair + lampe + projecteur : l'infrarouge sature et la
            # profondeur laisse un trou. Exposition automatique, et
            # projecteur moins fort.
            (rs.option.enable_auto_exposure, 1),
            (rs.option.laser_power, PUISSANCE_LASER)):
        if capteur.supports(option):
            try:
                capteur.set_option(option, valeur)
            except Exception:
                pass
    return float(capteur.get_depth_scale())


def mediane(profondeurs) -> np.ndarray:
    """Mediane des seules mesures valides ; un pixel jamais mesure reste a 0."""
    pile = np.stack([np.asarray(p, dtype=np.float32) for p in profondeurs])
    pile[pile <= 0] = np.nan
    jamais = np.all(np.isnan(pile), axis=0)
    pile[:, jamais] = 0.0
    return np.nan_to_num(np.nanmedian(pile, axis=0), nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Format d'echange
# ---------------------------------------------------------------------------

def empaqueter(capture: Dict[str, np.ndarray]) -> bytes:
    tampon = io.BytesIO()
    np.savez_compressed(tampon, **capture)
    return tampon.getvalue()


def depaqueter(donnees: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(donnees)) as z:
        return {k: z[k] for k in z.files}


def envoyer_message(sock: socket.socket, donnees: bytes) -> None:
    sock.sendall(struct.pack(">Q", len(donnees)) + donnees)


def recevoir_message(sock: socket.socket) -> bytes:
    def lire(n):
        morceaux, reste = [], n
        while reste:
            m = sock.recv(min(reste, 1 << 20))
            if not m:
                raise ErreurCamera("connexion coupee par le serveur")
            morceaux.append(m)
            reste -= len(m)
        return b"".join(morceaux)
    taille = struct.unpack(">Q", lire(8))[0]
    return lire(taille)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def serveur_present(hote: str = HOTE, port: int = PORT) -> bool:
    try:
        with socket.create_connection((hote, port), timeout=0.5):
            return True
    except OSError:
        return False


def capturer(images: int = N_IMAGES, hote: str = HOTE, port: int = PORT,
             delai: float = 30.0) -> Dict[str, np.ndarray]:
    """
    Demande au serveur une capture FRAICHE : la mediane des `images`
    prochaines images (aucune prise avant la demande). Sans sudo.
    """
    try:
        sock = socket.create_connection((hote, port), timeout=2.0)
    except OSError as exc:
        raise ErreurCamera(
            "pas de serveur de camera sur {}:{}. Lancez-le, dans un terminal "
            "a part :\n    sudo .venv/bin/python examples/perception/"
            "serveur_camera.py".format(hote, port)) from exc
    with sock:
        sock.settimeout(delai)
        envoyer_message(sock, json.dumps(
            {"demande": "capture", "images": int(images)}).encode("utf-8"))
        reponse = recevoir_message(sock)
    if reponse.startswith(b"ERREUR"):
        raise ErreurCamera(reponse.decode("utf-8", "replace"))
    return depaqueter(reponse)


def _demander(charge: dict, hote: str = HOTE, port: int = PORT,
              delai: float = 30.0) -> dict:
    try:
        sock = socket.create_connection((hote, port), timeout=2.0)
    except OSError as exc:
        raise ErreurCamera("pas de serveur de camera sur {}:{}"
                           .format(hote, port)) from exc
    with sock:
        sock.settimeout(delai)
        envoyer_message(sock, json.dumps(charge).encode("utf-8"))
        reponse = recevoir_message(sock)
    if reponse.startswith(b"ERREUR"):
        raise ErreurCamera(reponse.decode("utf-8", "replace"))
    return json.loads(reponse.decode("utf-8"))


def etat(**kw) -> dict:
    """Resolution, reglages et nombre d'images lues par le serveur."""
    return _demander({"demande": "etat"}, **kw)


def regler(**reglages) -> dict:
    """
    Reglages a chaud, sans sudo ni redemarrage :
    laser (mW, 0 a 360), preset (4 = haute densite), exposition_auto (0/1),
    exposition (us, si exposition_auto=0), projecteur (0/1).
    """
    return _demander({"demande": "regler", "reglages": reglages})


def redemarrer(resolution: Optional[str] = None,
               couleur: Optional[bool] = None) -> dict:
    """
    Change la resolution ("848x480", "1280x720"...) ou active la couleur :
    le serveur arrete et relance le flux. En cas d'echec, il revient a
    l'ancienne configuration.
    """
    charge: Dict[str, object] = {"demande": "redemarrer"}
    if resolution is not None:
        charge["resolution"] = resolution
    if couleur is not None:
        charge["couleur"] = bool(couleur)
    return _demander(charge, delai=60.0)


def capturer_scene(images: int = N_IMAGES, **kw):
    """Comme capturer(), mais rend directement une vision.Scene."""
    import vision
    c = capturer(images, **kw)
    return vision.Scene(profondeur=c["profondeur"].astype(np.float64),
                        K=c["K"], infrarouge=c.get("infrarouge"),
                        couleur=c.get("couleur"))
