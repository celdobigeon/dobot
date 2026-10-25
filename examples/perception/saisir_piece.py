# -*- coding: utf-8 -*-
"""
Essai de saisie guidee par la camera.

    .venv/bin/python examples/perception/saisir_piece.py --longueur-outil 78
    .venv/bin/python examples/perception/saisir_piece.py --longueur-outil 78 --boucle

Le robot se met en EQUERRE (hors du champ de la camera). On pose une piece
sur la table, Entree : la camera la voit, le robot va la saisir -- descente
et remontee verticales --, revient en equerre, la repose AILLEURS sur la
table, revient en equerre, et dit ou la camera la voit alors (ecart a
l'endroit vise). Avec --boucle, il recommence avec la piece ainsi
deplacee, jusqu'a Ctrl+C.

Il faut l'etalonnage camera -> robot (etalonner_camera_robot.py) : il
fournit T_robot_camera, la hauteur de la table, le secteur de la base
qu'on ne quitte pas, et l'angle de la pince par rapport a la piece.

Avant de lancer : serveur de camera en marche, robot mis en route,
fenetre du simulateur fermee.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
for _d in (ICI, RACINE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import etalonner_camera_robot as ec      # noqa: E402

#: mm : on ne saisit qu'une piece dont la largeur passe entre les mors
#: (ouverture ~26 mm)
LARGEUR_MAX = 22.0
#: mm : les mors prennent au plus cette hauteur de piece, en partant du
#: dessus ; jamais plus bas que PIED au-dessus de la table
PRISE_HAUTEUR, PIED = 12.0, 2.0
#: mm : la nouvelle place est au moins a cette distance de l'ancienne
ECART_DEPOSE = 60.0


class Essai:

    def __init__(self, robot, perception, donnees, journal=print, demander=input,
                 graine=None):
        self.robot = robot
        self.perception = perception
        self.journal = journal
        self.demander = demander
        self.z_table = float(donnees["z_table_mm"])
        self.ecart_r = float(donnees.get("ecart_r_deg", 0.0))
        self.secteur = tuple(donnees["secteur_base_deg"]) if donnees.get(
            "secteur_base_deg") else None
        # les outils de l'etalonnage : atteignable avec marges, dans le
        # secteur, dans le champ de la camera
        cam = type("Champ", (), {})()
        cam.K = np.array(donnees["camera_K"], float) if donnees.get("camera_K") else None
        cam.taille = tuple(donnees["camera_taille"]) if donnees.get("camera_taille") else None
        self.et = ec.Etalonnage(robot, cam, z_table=self.z_table, secteur=self.secteur,
                                journal=journal, graine=graine)
        self.et.restaurer({k: donnees.get(k) for k in
                           ("z_table_mm", "secteur_base_deg", "depart", "mesures",
                            "camera_K", "camera_taille")} | {"piece": None})
        self.rng = np.random.default_rng(graine)

    # -- mouvements ---------------------------------------------------------
    def equerre(self):
        """Posture d'equerre, base au milieu du secteur : hors du champ."""
        q = [float(v) for v in self.robot.angles()]
        j1 = 0.5 * (self.secteur[0] + self.secteur[1]) if self.secteur else q[0]
        xa, ya, za, ra = self.robot.pose()
        m = []
        if za < self.z_table + ec.TRANSIT - 1.0:
            m.append(("movl", dict(x=xa, y=ya, z=self.z_table + ec.TRANSIT, r=ra)))
        m.append(("movj", dict(q=[j1, ec.EQUERRE_J2, ec.EQUERRE_J3, q[3]])))
        self.et._enchainer(m)

    def _descendre(self, x, y, z, r):
        """Au-dessus de (x, y) puis descente VERTICALE jusqu'a z."""
        m = self.et._approche(x, y, r)[:-1]
        m.append(("movl", dict(x=x, y=y, z=z, r=r)))
        self.et._enchainer(m)

    def _remonter_en_equerre(self):
        self.equerre()

    # -- la camera ----------------------------------------------------------
    def voir(self):
        pieces = self.perception.detecter()
        return [p for p in pieces if p.T_robot is not None]

    def choisir(self, pieces):
        """La piece a prendre : libre (rien dessus), la plus haute, et qui
        passe entre les mors."""
        from perception import ordre_de_prise
        for p in ordre_de_prise(pieces):
            long_, large, haut = p.dimensions
            if large <= LARGEUR_MAX:
                return p
            self.journal("  piece {} ({} {}) trop large pour la pince ({:.0f} mm)"
                         .format(p.numero, p.type, p.face, large))
        return None

    def prise_de(self, piece):
        """(x, y, z, r) de la pince pour saisir la piece."""
        T = piece.T_robot
        cx, cy, cz = T[:3, 3] * 1000.0
        lacet = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        haut = float(piece.dimensions[2])
        z = max(cz - min(PRISE_HAUTEUR, haut - PIED), self.z_table + PIED)
        # la pince est symetrique : r a 180 deg pres, le plus pres de 0
        r = ec._angle(lacet + self.ecart_r, 180.0)
        # hauteur de la pince au-dessus du DESSOUS de la piece : pour la
        # reposer, dessous sur la table
        self.au_dessus_du_dessous = z - (cz - haut)
        return cx, cy, z, r

    # -- le deroule ---------------------------------------------------------
    def saisir(self, piece):
        x, y, z, r = self.prise_de(piece)
        if not self.et._atteignable(x, y, r):
            r2 = ec._angle(r + 180.0)
            if self.et._atteignable(x, y, r2):
                r = r2
            else:
                raise ec.Arret("piece hors d'atteinte (ou hors du secteur) : x {:.0f} "
                               "y {:.0f}".format(x, y))
        self.journal("  saisie en x {:.1f} y {:.1f} z {:.1f} r {:.1f}  ({} {}, dessus a "
                     "z {:.1f})".format(x, y, z, r, piece.type, piece.face,
                                        piece.T_robot[2, 3] * 1000.0))
        self.robot.pince(False)
        self._descendre(x, y, z, r)
        self.robot.pince(True)
        self.robot.attendre(ec.DELAI_PINCE)
        self._remonter_en_equerre()
        return x, y, z, r

    def deposer(self, depuis):
        """Ailleurs : un point vu, atteint, dans le secteur, loin du depart.
        Meme hauteur de pince qu'a la saisie : la piece touche la table."""
        x0, y0, _z, r0 = depuis
        z = self.z_table + self.au_dessus_du_dessous
        for _ in range(500):
            x = self.rng.uniform(150.0, 320.0)
            y = self.rng.uniform(-220.0, 220.0)
            r = ec._angle(r0 + self.rng.uniform(-45.0, 45.0))
            if math.hypot(x - x0, y - y0) < ECART_DEPOSE:
                continue
            if self.et._atteignable(x, y, r) and self.et._visible(x, y):
                break
        else:
            raise ec.Arret("aucune place libre, vue et atteinte, pour reposer la piece")
        self.journal("  depose en x {:.1f} y {:.1f} r {:.1f}".format(x, y, r))
        self._descendre(x, y, z, r)
        self.robot.pince(False)
        self.robot.attendre(ec.DELAI_PINCE)
        self._remonter_en_equerre()
        return x, y, z, r

    def controler(self, vise):
        """Ou la camera voit la piece reposee, par rapport au point vise."""
        x, y, _z, _r = vise
        pieces = self.voir()
        if not pieces:
            self.journal("  controle : la camera ne voit pas la piece")
            return None
        p = min(pieces, key=lambda p: math.hypot(p.T_robot[0, 3] * 1000 - x,
                                                 p.T_robot[1, 3] * 1000 - y))
        e = math.hypot(p.T_robot[0, 3] * 1000 - x, p.T_robot[1, 3] * 1000 - y)
        self.journal("  controle : piece vue a {:.1f} mm du point vise".format(e))
        return p

    def un_tour(self, piece=None):
        if piece is None:
            pieces = self.voir()
            piece = self.choisir(pieces)
            if piece is None:
                self.journal("  la camera ne voit aucune piece saisissable")
                return None
        prise = self.saisir(piece)
        vise = self.deposer(prise)
        return self.controler(vise)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saisie d'une piece guidee par la camera")
    ap.add_argument("--longueur-outil", type=float, default=None)
    ap.add_argument("--vitesse", type=float, default=20.0, help="mm/s (20)")
    ap.add_argument("--boucle", action="store_true",
                    help="recommencer avec la piece deplacee, jusqu'a Ctrl+C")
    ap.add_argument("--etalonnage", default=ec.FICHIER)
    a = ap.parse_args(argv)

    with open(a.etalonnage, encoding="utf-8") as f:
        donnees = json.load(f)
    if "T_robot_camera" not in donnees:
        print("etalonnage inacheve : lancez d'abord etalonner_camera_robot.py")
        return 1

    from dobot_sim import Commande, ErreurExecution
    import perception
    print("chargement de la perception...")
    perc = perception.Perception(T_robot_camera=np.array(donnees["T_robot_camera"], float))
    robot = Commande(simulateur=False, robot=True, outil="pince", vitesse=a.vitesse,
                     longueur_outil=a.longueur_outil)
    essai = Essai(robot, perc, donnees, demander=ec._demander)
    try:
        robot.pince(False)
        essai.equerre()
        essai.demander("Posez une piece sur la table (dans le champ de la camera), "
                       "puis Entree ")
        piece = None
        while True:
            t0 = time.time()
            piece = essai.un_tour(piece)
            print("  tour fait en {:.0f} s".format(time.time() - t0))
            if not a.boucle:
                break
            if piece is None:
                essai.demander("Reposez une piece sur la table, puis Entree ")
    except KeyboardInterrupt:
        print("\ninterrompu (robot arrete)")
    except ec.Arret as exc:
        print("arret : {}".format(exc))
    except ErreurExecution as exc:
        print("ECHEC : {}".format(exc))
    finally:
        try:
            robot.pince(False)
        except Exception:                             # noqa: BLE001
            pass
        robot.fermer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
