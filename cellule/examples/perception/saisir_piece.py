# -*- coding: utf-8 -*-
"""
Essai de saisie guidee par la camera.

    .venv/bin/python examples/perception/saisir_piece.py --longueur-outil 78
    .venv/bin/python examples/perception/saisir_piece.py --longueur-outil 78 --boucle

Le robot se met en EQUERRE (hors du champ de la camera). On pose une piece
sur la table, Entree : la camera la voit, le robot va la saisir -- descente
et remontee verticales --, revient en equerre, la repose AILLEURS sur la
table, revient en equerre, et dit ou la camera la voit alors (ecart a
l'endroit vise). Puis il recommence : on pose une autre piece (ou on
laisse celle-ci), Entree... jusqu'a Ctrl+C. Avec --boucle, il enchaine
sans attendre Entree, avec la piece qu'il vient de deplacer.

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
#: mm : hauteur du bout des doigts au-dessus de la surface sur laquelle
#: repose la piece (la table, ou la piece du dessous). Juste si la longueur
#: d'outil et la hauteur de table sont justes : le 25/10, la pince mesurait
#: 107 mm et non 78 (29 mm d'erreur, verifies a la regle a l'equerre).
PIED = 3.0
#: mm : la nouvelle place est au moins a cette distance de l'ancienne
ECART_DEPOSE = 60.0


class HorsAtteinte(Exception):
    """La piece vue ne peut pas etre saisie la ou elle est."""


class Essai:

    def __init__(self, robot, perception, donnees, journal=print, demander=input,
                 graine=None, forcer: bool = False, pied: float = PIED):
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
        if self.et.recalage is None and donnees.get("T_robot_camera_mm"):
            # etalonnage par la pince : pas de mesures, la transformation seule
            T = np.array(donnees["T_robot_camera_mm"], float)
            self.et.recalage = ec.Recalage(T[:3, :3], T[:3, 3], {}, self.ecart_r,
                                           np.zeros(1), np.ones(1, bool))
        self.rng = np.random.default_rng(graine)
        #: sans marges : butees et secteur au plus juste, transit abaisse
        #: si besoin (au moins APPROCHE mm de descente verticale)
        self.forcer = bool(forcer)
        self.pied = float(pied)
        #: rappels pour un affichage : quand_saisie(piece, x, y, z, r) juste
        #: avant de fermer la pince ; quand_lache() juste apres l'ouverture
        self.quand_saisie = None
        self.quand_lache = None
        self.modele = self.et.modele

    # -- atteignabilite -----------------------------------------------------
    def _pose_ok(self, x, y, z, r) -> str:
        """'' si (x, y, z, r) est permis, sinon la raison."""
        try:
            q = self.modele.mgi_pose(x, y, z, r)
        except Exception:                               # noqa: BLE001
            return "hors de portee du bras"
        if not self.modele.dans_butees(q):
            return "une articulation passerait sa butee"
        if not self.forcer and float(np.min(self.modele.marges(q))) < ec.MARGE_BUTEE:
            return "trop pres d'une butee (marge {:.0f} deg ; --forcer pour l'ignorer)"\
                .format(ec.MARGE_BUTEE)
        if not self.et.dans_secteur(x, y, z, r, 0.0 if self.forcer else ec.MARGE_SECTEUR):
            return "hors du secteur de la base"
        return ""

    def verifier(self, x, y, z, r):
        """(raison, hauteur de transit) : raison vide si la prise en z est
        permise ET qu'un transit au-dessus l'est aussi."""
        raison = self._pose_ok(x, y, z, r)
        if raison:
            return raison, None
        hauts = [self.z_table + ec.TRANSIT]
        if self.forcer:
            hauts += list(np.arange(self.z_table + ec.TRANSIT - 5.0, z + ec.APPROCHE - 1e-6,
                                    -5.0))
        for h in hauts:
            if h >= z + ec.APPROCHE and not self._pose_ok(x, y, h, r):
                return "", float(h)
        return ("pas de place au-dessus pour une descente verticale" +
                ("" if self.forcer else " (--forcer : transit plus bas)")), None

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

    def _descendre(self, x, y, z, r, h=None):
        """Au-dessus de (x, y), a la hauteur h, puis descente VERTICALE
        jusqu'a z."""
        h = self.z_table + ec.TRANSIT if h is None else h
        xa, ya, za, ra = self.robot.pose()
        m = []
        if za < h - 1.0:
            m.append(("movl", dict(x=xa, y=ya, z=h, r=ra)))
        tourne = abs(ec._angle(r - ra)) > 0.5
        m.append(("movj" if tourne else "movl", dict(x=x, y=y, z=h, r=r)))
        m.append(("movl", dict(x=x, y=y, z=z, r=r)))
        self.et._enchainer(m)

    def _remonter_en_equerre(self):
        self.equerre()

    # -- la camera ----------------------------------------------------------
    def voir(self):
        pieces = self.perception.detecter()
        return [p for p in pieces if p.T_robot is not None and not dans_le_robot(p)]

    def choisir(self, pieces):
        """
        La piece a prendre : libre (rien dessus), la plus haute, qui passe
        entre les mors -- et ATTEIGNABLE d'abord. Une piece hors d'atteinte
        n'est proposee que s'il n'y a rien d'autre (pour 'Forcer'). Les
        detections dans l'emprise du robot (son socle vu par la camera) sont
        ignorees.
        """
        from perception import ordre_de_prise
        possibles = []
        for p in ordre_de_prise([p for p in pieces if not dans_le_robot(p)]):
            long_, large, haut = p.dimensions
            if large <= LARGEUR_MAX:
                possibles.append(p)
        for p in possibles:
            try:
                if not self.statut(p):
                    return p
            except Exception:                           # noqa: BLE001
                continue
        return possibles[0] if possibles else None

    def prise_de(self, piece):
        """(x, y, z, r) de la pince pour saisir la piece."""
        T = piece.T_robot
        cx, cy, cz = T[:3, 3] * 1000.0
        lacet = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        haut = float(piece.dimensions[2])
        dessous = cz - haut
        # une piece posee sur la table : on prend la hauteur de table
        # etalonnee plutot que son estimation par la camera
        if abs(dessous - self.z_table) < 5.0:
            dessous = self.z_table
        z = dessous + self.pied
        # la pince est symetrique : r a 180 deg pres, le plus pres de 0
        r = ec._angle(lacet + self.ecart_r, 180.0)
        # hauteur de la pince au-dessus du DESSOUS de la piece -- le meme
        # dessous que pour la saisie : pour la reposer, dessous sur la table
        self.au_dessus_du_dessous = z - dessous
        return cx, cy, z, r

    def statut(self, piece) -> str:
        """'' si la piece est saisissable telle quelle (avec les marges),
        sinon la raison -- sans rien bouger."""
        x, y, z, r = self.prise_de(piece)
        raison, _h = self.verifier(x, y, z, r)
        if raison and not self.verifier(x, y, z, ec._angle(r + 180.0))[0]:
            return ""
        return raison

    # -- le deroule ---------------------------------------------------------
    def saisir(self, piece, quand_meme: bool = False):
        """
        Saisit la piece. `quand_meme` : l'utilisateur a demande d'essayer
        malgre un refus -- sans marges, transit abaisse, et si le modele
        refuse encore, on laisse le robot en juger.
        """
        x, y, z, r = self.prise_de(piece)
        forcer = self.forcer
        if quand_meme:
            self.forcer = True
        try:
            raison, h = self.verifier(x, y, z, r)
            if raison:
                r2 = ec._angle(r + 180.0)
                raison2, h2 = self.verifier(x, y, z, r2)
                if not raison2:
                    r, h = r2, h2
                elif quand_meme:
                    h = max(z + ec.APPROCHE, min(self.z_table + ec.TRANSIT, z + 40.0))
                    self.journal("  le modele refuse encore ({}) : on essaie quand meme"
                                 .format(raison))
                else:
                    raise HorsAtteinte("x {:.0f} y {:.0f} : {}".format(x, y, raison))
        finally:
            self.forcer = forcer
        self.journal("  saisie en x {:.1f} y {:.1f} z {:.1f} r {:.1f}  ({} {}, dessus a "
                     "z {:.1f})".format(x, y, z, r, piece.type, piece.face,
                                        piece.T_robot[2, 3] * 1000.0))
        self.robot.pince(False)
        self._descendre(x, y, z, r, h)
        if self.quand_saisie is not None:
            self.quand_saisie(piece, x, y, z, r)
        self.robot.pince(True)
        self.robot.attendre(ec.DELAI_PINCE)
        self._remonter_en_equerre()
        return x, y, z, r

    def deposer(self, depuis, apres_lacher=None):
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
            if not self.et._visible(x, y):
                continue
            raison, h = self.verifier(x, y, z, r)
            if not raison:
                break
        else:
            raise ec.Arret("aucune place libre, vue et atteinte, pour reposer la piece")
        self.journal("  depose en x {:.1f} y {:.1f} r {:.1f}".format(x, y, r))
        self._descendre(x, y, z, r, h)
        self.robot.pince(False)
        if self.quand_lache is not None:
            self.quand_lache()
        if apres_lacher is not None:
            apres_lacher()
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

    def un_tour(self, piece=None, quand_meme: bool = False, interactif: bool = True,
                apres_lacher=None, controle: bool = True):
        """
        Saisir, reposer ailleurs, controler. Piece jugee hors d'atteinte :
        en mode `interactif`, on demande (Entree : je la deplace / f :
        essayer quand meme) ; sinon HorsAtteinte remonte a l'appelant, qui
        relancera avec `quand_meme=True` s'il le faut.
        """
        from dobot_sim import ErreurExecution
        if piece is None:
            pieces = self.voir()
            piece = self.choisir(pieces)
            if piece is None:
                self.journal("  la camera ne voit aucune piece saisissable")
                return None
        try:
            prise = self.saisir(piece, quand_meme=quand_meme)
        except HorsAtteinte as exc:
            if not interactif:
                raise
            rep = self.demander("  piece HORS D'ATTEINTE ({}).\n  [Entree : je la deplace "
                                "vers le milieu de la zone / f : essayer quand meme] "
                                .format(exc))
            if not rep.strip().lower().startswith("f"):
                return None
            quand_meme = True
            try:
                prise = self.saisir(piece, quand_meme=True)
            except ErreurExecution as exc2:
                self.journal("  le robot a refuse : {}".format(exc2))
                self.equerre()
                return None
        except ErreurExecution as exc2:
            if not quand_meme or not interactif:
                raise                       # l'appelant l'affiche
            self.journal("  le robot a refuse : {}".format(exc2))
            self.equerre()
            return None
        vise = self.deposer(prise, apres_lacher)
        if not controle:
            return vise
        return self.controler(vise)


#: mm : une detection a moins de cette distance de l'axe de la base est
#: dans l'emprise du robot (socle 158 x 158 mm, colonne) : ce n'est pas une
#: piece
EMPRISE_ROBOT = 130.0


def dans_le_robot(piece) -> bool:
    """La piece detectee est-elle dans l'emprise du robot (son socle ou sa
    colonne, pris pour une piece) ?"""
    T = getattr(piece, "T_robot", None)
    if T is None:
        return False
    x, y = T[0, 3] * 1000.0, T[1, 3] * 1000.0
    return math.hypot(x, y) < EMPRISE_ROBOT


def probleme_etalonnage(donnees) -> str:
    """'' si l'etalonnage est utilisable ; sinon pourquoi."""
    T = np.array(donnees["T_robot_camera_mm"], float)
    rc = ec.Recalage(T[:3, :3], T[:3, 3], {}, 0.0, np.zeros(1), np.ones(1, bool))
    pb = rc.invraisemblable(float(donnees.get("z_table_mm", 0.0)))
    if pb:
        return pb
    rms = float(donnees.get("rms_mm", 0.0))
    if rms > 5.0:
        return "ecart moyen de {:.0f} mm (fausses pieces ?)".format(rms)
    if int(donnees.get("paires_retenues", donnees.get("paires", 0))) < 8:
        return "trop peu de poses ({})".format(donnees.get("paires_retenues"))
    return ""


def longueur_outil(demandee, donnees):
    """La longueur d'outil : celle demandee, sinon celle de l'etalonnage
    (les hauteurs du fichier en dependent). Previent d'un desaccord."""
    du_fichier = donnees.get("longueur_outil_mm")
    if demandee is None:
        if du_fichier is not None:
            print("longueur d'outil de l'etalonnage : {:.1f} mm".format(du_fichier))
        return du_fichier
    if du_fichier is not None and abs(float(demandee) - float(du_fichier)) > 0.5:
        print("ATTENTION : longueur d'outil {:.1f} mm, mais l'etalonnage a ete fait avec "
              "{:.1f} mm :\n  les hauteurs seront fausses de {:+.1f} mm.".format(
                  demandee, du_fichier, float(du_fichier) - float(demandee)))
    return float(demandee)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saisie d'une piece guidee par la camera")
    ap.add_argument("--longueur-outil", type=float, default=None)
    ap.add_argument("--vitesse", type=float, default=20.0, help="mm/s (20)")
    ap.add_argument("--hauteur-prise", type=float, default=PIED,
                    help="mm visees entre le bout des doigts et la table ({:.0f})"
                    .format(PIED))
    ap.add_argument("--forcer", action="store_true",
                    help="sans marges : butees et secteur au plus juste, transit "
                         "abaisse si besoin (plus de positions saisissables)")
    ap.add_argument("--boucle", action="store_true",
                    help="enchainer sans attendre Entree, avec la piece deplacee")
    ap.add_argument("--etalonnage", default=ec.FICHIER)
    a = ap.parse_args(argv)

    with open(a.etalonnage, encoding="utf-8") as f:
        donnees = json.load(f)
    if "T_robot_camera" not in donnees:
        print("etalonnage inacheve : lancez d'abord etalonner_camera_robot.py")
        return 1
    probleme = probleme_etalonnage(donnees) if "probleme_etalonnage" in globals() \
        else sp.probleme_etalonnage(donnees)
    if probleme:
        print("ETALONNAGE INUTILISABLE : {}.\n  Refaites-le : dobot.py etalonner-avec-pieces "
              "--recommencer".format(probleme))
        return 1

    from dobot_sim import Commande, ErreurExecution
    import perception
    longueur = longueur_outil(a.longueur_outil, donnees)
    print("chargement de la perception...")
    perc = perception.Perception(T_robot_camera=np.array(donnees["T_robot_camera"], float))
    robot = Commande(simulateur=False, robot=True, outil="pince", vitesse=a.vitesse,
                     longueur_outil=longueur)
    essai = Essai(robot, perc, donnees, demander=ec._demander, forcer=a.forcer,
                  pied=a.hauteur_prise)
    try:
        robot.pince(False)
        essai.equerre()
        piece = None
        while True:
            if not a.boucle or piece is None:
                essai.demander("\nPosez une piece sur la table (ou laissez celle qui y "
                               "est), puis Entree -- Ctrl+C pour arreter ")
                piece = None
            t0 = time.time()
            piece = essai.un_tour(piece)
            print("  tour fait en {:.0f} s".format(time.time() - t0))
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
