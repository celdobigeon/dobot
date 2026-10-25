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

Entree dans le terminal : le robot saisit la cible, revient en equerre,
la repose ailleurs, revient en equerre -- comme saisir_piece.py. La
detection est suspendue pendant les mouvements (le bras est dans le
champ). Ctrl+C pour arreter.

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
            sys.stdout.write("\r  [camera] {:<70s}".format(texte))
            sys.stdout.flush()

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
    try:
        robot.pince(False)
        essai.equerre()
        det.start()
        print("La detection tourne : les pieces vues s'affichent dans la vue 3D, la "
              "cible en rouge.")
        while True:
            ec._demander("\nEntree : saisir la cible (rouge) -- Ctrl+C pour arreter ")
            print("  attente d'une detection a jour...")
            cible = det.cible_fraiche()
            if cible is None:
                print("  aucune cible : posez une piece dans le champ de la camera")
                continue
            det.pause.set()
            try:
                t0 = time.time()
                essai.un_tour(cible)
                print("  tour fait en {:.0f} s".format(time.time() - t0))
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
            afficher([])
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
