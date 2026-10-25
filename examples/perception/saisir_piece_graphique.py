# -*- coding: utf-8 -*-
"""
Saisie guidee par la camera, version graphique.

    .venv/bin/python examples/perception/saisir_piece_graphique.py --longueur-outil 78

Ouvre le simulateur (ou se sert de celui qui est ouvert), relie au robot :
la fenetre montre le VRAI bras en miroir. La detection tourne en
permanence, dans un fil a part : les pieces vues s'affichent dans la vue
3D, a leur place dans le repere du robot, la CIBLE en rouge -- celle que le
robot prendrait. Deplacez les pieces sur la table : l'affichage suit
(une detection prend 2 a 4 s).

Le panneau Programme, sans objet ici, est remplace par un panneau Saisie :
"Saisir la cible" -- le robot la saisit, revient en equerre, la repose
ailleurs, revient en equerre, comme saisir_piece.py -- et "Forcer la
saisie", actif quand la cible vient d'etre jugee hors d'atteinte. Entree
(ou f + Entree) dans le terminal fait la meme chose. La detection est
suspendue pendant les mouvements (le bras est dans le champ). Ctrl+C dans
le terminal pour arreter ; le panneau Programme revient.

Il faut l'etalonnage camera -> robot (etalonner_camera_robot.py) et le
serveur de camera en marche.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
for _d in (ICI, RACINE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import etalonner_camera_robot as ec      # noqa: E402
import saisir_piece as sp                # noqa: E402


def boite(piece, types) -> dict:
    """Une piece vue -> boite pour la vue 3D (repere robot, mm)."""
    import perception
    T = np.asarray(piece.T_robot, float)
    if piece.type in types and piece.face in perception.FACES:
        dims = perception.forme(types[piece.type], piece.face)
    else:
        dims = tuple(float(v) for v in piece.dimensions)
    centre = T[:3, 3] * 1000.0 - T[:3, 2] * dims[2] / 2.0
    return {"centre": [float(v) for v in centre], "axes": T[:3, :3].tolist(),
            "dims": [float(v) for v in dims], "type": piece.type}


def afficher(boites, cible=None) -> None:
    from dobot_sim.passerelle import demander
    demander({"scene": {"boites": boites, "cible": cible}})


def panneau(**etat) -> None:
    """Met a jour le panneau Saisie de la fenetre (cible, texte, boutons)."""
    from dobot_sim.passerelle import demander
    demander({"saisie": {"etat": etat}})


def lire_bouton():
    """'saisir', 'forcer' ou None : le dernier clic dans le panneau Saisie."""
    from dobot_sim.passerelle import demander
    rep = demander({"saisie": {"lire": True}})
    return (rep or {}).get("demande")


def lire_clavier():
    """Entree ('saisir') ou f + Entree ('forcer') tapes dans le terminal,
    sans bloquer."""
    import select
    try:
        pret, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not pret:
        return None
    ligne = sys.stdin.readline()
    return "forcer" if ligne.strip().lower().startswith("f") else "saisir"


class Detecteur(threading.Thread):
    """
    La detection en continu. Une image prise pendant un mouvement du bras
    (ou commencee avant) est jetee : `epoque` avance a chaque mouvement, et
    seule une detection de l'epoque courante compte.
    """

    def __init__(self, perception, essai):
        super().__init__(daemon=True)
        self.perception = perception
        self.essai = essai
        self.pause = threading.Event()
        self.fin = threading.Event()
        self.epoque = 0
        self.verrou = threading.Lock()
        self.derniere = None           # (epoque, pieces, cible)
        self._erreur = ""

    def run(self):
        while not self.fin.is_set():
            if self.pause.is_set():
                time.sleep(0.1)
                continue
            ep = self.epoque
            try:
                pieces = [p for p in self.perception.detecter() if p.T_robot is not None]
            except Exception as exc:                    # noqa: BLE001
                if str(exc) != self._erreur:
                    self._erreur = str(exc)
                    print("\n  detection impossible : {}".format(exc))
                time.sleep(1.0)
                continue
            if self.pause.is_set() or ep != self.epoque:
                continue                    # le bras a bouge pendant la prise
            cible = self.essai.choisir(pieces) if pieces else None
            boites = [boite(p, self.perception.types) for p in pieces]
            try:
                afficher(boites, pieces.index(cible) if cible is not None else None)
            except Exception:                           # noqa: BLE001
                pass
            with self.verrou:
                self.derniere = (ep, pieces, cible)
            if cible is not None:
                x, y, z = cible.T_robot[:3, 3] * 1000.0
                texte = "{} piece(s) vue(s) ; cible : {} {} en x {:.0f} y {:.0f}".format(
                    len(pieces), cible.type, cible.face, x, y)
            else:
                texte = "{} piece(s) vue(s) ; aucune cible".format(len(pieces))
            try:
                panneau(cible=texte, saisir=cible is not None)
            except Exception:                           # noqa: BLE001
                pass

    def cible_fraiche(self, delai: float = 20.0):
        """La cible d'une detection faite APRES le dernier mouvement."""
        fin = time.time() + delai
        while time.time() < fin:
            with self.verrou:
                d = self.derniere
            if d is not None and d[0] == self.epoque:
                return d[2]
            time.sleep(0.1)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saisie guidee par la camera, dans le "
                                             "simulateur en miroir")
    ap.add_argument("--longueur-outil", type=float, default=None)
    ap.add_argument("--vitesse", type=float, default=20.0, help="mm/s (20)")
    ap.add_argument("--port", default=None, help="port serie du robot")
    ap.add_argument("--etalonnage", default=ec.FICHIER)
    a = ap.parse_args(argv)

    with open(a.etalonnage, encoding="utf-8") as f:
        donnees = json.load(f)
    if "T_robot_camera" not in donnees:
        print("etalonnage inacheve : lancez d'abord etalonner_camera_robot.py")
        return 1
    T = np.array(donnees["T_robot_camera"], float)

    from dobot_sim import Commande, ErreurExecution
    import perception
    print("chargement de la perception...")
    perc_robot = perception.Perception(T_robot_camera=T)     # saisies, controles
    perc_continue = perception.Perception(T_robot_camera=T)  # le fil de detection
    print("ouverture du simulateur, relie au robot...")
    robot = Commande(simulateur=True, robot=True, lancer_simulateur=True, port=a.port,
                     outil="pince", vitesse=a.vitesse, longueur_outil=a.longueur_outil)
    essai = sp.Essai(robot, perc_robot, donnees, demander=ec._demander)
    det = Detecteur(perc_continue, essai)
    from dobot_sim.passerelle import demander
    try:
        demander({"saisie": {"activer": True}})
        robot.pince(False)
        essai.equerre()
        det.start()
        print("La detection tourne : pieces dans la vue 3D, la cible en rouge.\n"
              "Bouton 'Saisir la cible' dans la fenetre (ou Entree ici) ; Ctrl+C pour "
              "arreter.")
        while True:
            quoi = lire_bouton() or lire_clavier()
            if quoi is None:
                time.sleep(0.2)
                continue
            panneau(occupe=True, texte="attente d'une detection a jour...")
            cible = det.cible_fraiche()
            if cible is None:
                panneau(occupe=False, forcer=False,
                        texte="aucune cible : posez une piece dans le champ de la camera")
                continue
            det.pause.set()
            forcer = quoi == "forcer"
            try:
                t0 = time.time()
                panneau(occupe=True, texte="saisie en cours{}...".format(
                    " (forcee, sans marges)" if forcer else ""))
                essai.un_tour(cible, quand_meme=forcer, interactif=False)
                panneau(occupe=False, forcer=False,
                        texte="tour fait en {:.0f} s".format(time.time() - t0))
            except sp.HorsAtteinte as exc:
                print("  piece HORS D'ATTEINTE ({})".format(exc))
                panneau(occupe=False, forcer=True,
                        texte="HORS D'ATTEINTE : {}.\nDeplacez la piece, ou 'Forcer la "
                              "saisie'.".format(exc))
            finally:
                det.epoque += 1
                det.pause.clear()
    except KeyboardInterrupt:
        print("\ninterrompu (robot arrete)")
    except ec.Arret as exc:
        print("arret : {}".format(exc))
    except ErreurExecution as exc:
        print("ECHEC : {}".format(exc))
    finally:
        det.fin.set()
        try:
            demander({"saisie": {"desactiver": True}})
        except Exception:                               # noqa: BLE001
            pass
        try:
            robot.pince(False)
        except Exception:                               # noqa: BLE001
            pass
        robot.fermer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
