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

En bas, le panneau Camera montre le flux video (une image a la fois,
quelques fois par seconde) avec les boites de la derniere detection, et la
carte de profondeur.

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


def _jpeg(image, largeur: int = 640, qualite: int = 80) -> str:
    import base64
    import cv2
    h, w = image.shape[:2]
    if w > largeur:
        image = cv2.resize(image, (largeur, int(round(h * largeur / w))),
                           interpolation=cv2.INTER_AREA)
    ok, tampon = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, qualite])
    return base64.b64encode(tampon.tobytes()).decode("ascii") if ok else ""


#: couleurs BGR des boites dans l'image : par type, et la cible
COULEURS_BGR = {"epaisse": (40, 170, 255), "fine": (255, 200, 40)}
ROUGE_BGR = (40, 40, 230)


ORANGE_BGR = (0, 140, 255)


def dessiner_boites(image, vue, pieces, types, cible=None, hors_atteinte: str = ""):
    """
    Chaque piece en BOITE 3D dans l'image couleur : sa pose estimee,
    reprojetee (repere de la profondeur -> camera couleur), face superieure
    en trait epais, et son repere (x rouge, y vert, z bleu). La cible en
    rouge, faces teintees.
    """
    import cv2
    import perception as pc
    img = image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    voile = img.copy()

    def proj(P):
        P = np.c_[P, np.ones(len(P))] @ np.asarray(vue.T_cp, float).T
        uv = P[:, :3] @ np.asarray(vue.K, float).T
        return (uv[:, :2] / uv[:, 2:3]).astype(np.int32)

    for p in pieces:
        if p.type in types and p.face in pc.FACES:
            L, l, h = (v / 1000.0 for v in pc.forme(types[p.type], p.face))
        else:
            L, l, h = (v / 1000.0 for v in p.dimensions)
        coins = np.array([[sx * L / 2, sy * l / 2, z] for z in (0.0, -h)
                          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        M = np.asarray(p.T_camera, float)
        uv = proj(coins @ M[:3, :3].T + M[:3, 3])
        est_cible = p is cible
        col = ((ORANGE_BGR if hors_atteinte else ROUGE_BGR) if est_cible
               else COULEURS_BGR.get(p.type, (200, 200, 200)))
        if est_cible:
            for face in ((0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
                cv2.fillConvexPoly(voile, uv[list(face)], col)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), col, 3 if est_cible else 2, cv2.LINE_AA)
        for a, b in ((4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), col, 2 if est_cible else 1, cv2.LINE_AA)
        o = M[:3, 3]
        ax = proj(np.array([o] + [o + 0.02 * M[:3, k] for k in range(3)]))
        for k, c in enumerate(((0, 0, 255), (0, 200, 0), (255, 80, 0))):
            cv2.arrowedLine(img, tuple(ax[0]), tuple(ax[k + 1]), c, 2, cv2.LINE_AA,
                            tipLength=0.25)
        texte = "{}{} {}".format(("CIBLE HORS D'ATTEINTE : " if hors_atteinte else "CIBLE : ")
                                 if est_cible else "", p.type, p.face)
        org = (int(uv[:4, 0].mean()) - 40, int(uv[:4, 1].min()) - 10)
        cv2.putText(img, texte, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, texte, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return cv2.addWeighted(voile, 0.3, img, 0.7, 0)


def images_camera(perception, capture, pieces, cible) -> dict:
    """Image couleur, pieces en boites 3D (cible en rouge), et carte de
    profondeur en fausses couleurs, en JPEG base64 pour la fenetre."""
    import cv2
    out = {}
    try:
        vue = perception.derniere_vue
        out["couleur"] = _jpeg(dessiner_boites(vue.image, vue, pieces, perception.types,
                                               cible))
    except Exception:                                   # noqa: BLE001
        pass
    z = np.asarray(capture.get("profondeur"), float)
    if z.size:
        valides = z > 0
        if valides.any():
            lo, hi = np.percentile(z[valides], (2, 98))
            n = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
            carte = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
            carte[~valides] = 0
            out["profondeur"] = _jpeg(carte)
    return out


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


#: images par capture de detection : la mediane de 8 images (0,5 s a
#: 15 images/s) suffit ; l'etalonnage en prend 15
IMAGES_DETECTION = 8


class Detecteur(threading.Thread):
    """
    La detection en continu, PAR ETAPES (Perception.detecter_par_etapes) :
    le resultat rapide (reseau seul, < 1 s) s'affiche des qu'il n'y a rien
    de mieux pour la position actuelle du bras, puis la scene consolidee
    (affinage) le remplace. Une image prise pendant un mouvement du bras
    (ou commencee avant) est jetee : `epoque` avance a chaque mouvement, et
    seule une detection de l'epoque courante compte. La saisie n'utilise
    qu'une detection CONSOLIDEE.
    """

    def __init__(self, perception, essai, capturer=None):
        super().__init__(daemon=True)
        if capturer is None:
            import camera
            capturer = lambda: camera.capturer(IMAGES_DETECTION)  # noqa: E731
        self.capturer = capturer
        self.perception = perception
        self.essai = essai
        self.pause = threading.Event()
        self.fin = threading.Event()
        self.epoque = 0
        self.verrou = threading.Lock()
        self.derniere = None           # (epoque, pieces, cible, niveau)
        self.raison_cible = ""         # cible hors d'atteinte : pourquoi
        #: point ou la piece vient d'etre reposee : la premiere detection
        #: consolidee dit a combien de lui on la voit
        self.controle = None
        self._erreur = ""
        # avancement de la detection en cours, pour la barre de progression
        self.etape = ""                # "", capture, rapide, propositions, consolidee
        self.affinage = 0.0            # avancement de l'affinage (0 a 1)
        perception.suivi_affinage = self._suivre_affinage
        self.debut = 0.0
        self.duree_moyenne = 5.0       # s, moyenne glissante d'un cycle complet

    def progression(self) -> float:
        """Part (0 a 1) de la detection en cours : estimee d'apres la duree
        habituelle, jamais en dessous de l'etape atteinte, 95 % au plus
        tant qu'elle n'est pas finie."""
        if not self.etape:
            return 0.0
        if self.etape == "consolidee":
            return 1.0
        atteint = self.perception.PROGRESSION.get(self.etape, 0.0)
        if self.etape == "propositions":
            # l'affinage dit lui-meme ou il en est
            debut = atteint
            return float(min(0.99, debut + (1.0 - debut) * self.affinage))
        estime = (time.time() - self.debut) / max(self.duree_moyenne, 0.5)
        return float(min(self.perception.PROGRESSION["propositions"], max(atteint, estime)))

    #: mm : au-dela, une piece a bouge
    SEUIL_CHANGEMENT = 15.0

    @classmethod
    def changement(cls, avant, apres) -> bool:
        """La scene a-t-elle change ? Pieces ajoutees, retirees, deplacees
        (plus de SEUIL_CHANGEMENT mm), ou d'un autre type."""
        if len(avant) != len(apres):
            return True
        restants = list(avant)
        for p in apres:
            c = p.T_robot[:3, 3] * 1000.0
            proches = [q for q in restants if q.type == p.type and
                       np.linalg.norm(q.T_robot[:3, 3] * 1000.0 - c) < cls.SEUIL_CHANGEMENT]
            if not proches:
                return True
            restants.remove(proches[0])
        return False

    def _suivre_affinage(self, fraction: float) -> None:
        self.affinage = float(fraction)

    def _publier(self, ep, pieces, niveau):
        pieces = [p for p in pieces if p.T_robot is not None]
        with self.verrou:
            d = self.derniere
            # un resultat rapide ne remplace une scene consolidee de la meme
            # position du bras que si la scene a CHANGE : sinon on garde le
            # meilleur ; s'il a change, l'ancienne detection est perimee
            if niveau == "rapide" and d is not None and d[0] == ep \
                    and d[3] == "consolidee" and not self.changement(d[1], pieces):
                return
        cible = self.essai.choisir(pieces) if pieces else None
        raison = ""
        if cible is not None and hasattr(self.essai, "statut"):
            try:
                raison = self.essai.statut(cible)
            except Exception as exc:                    # noqa: BLE001
                raison = str(exc)
        self.raison_cible = raison
        boites = [boite(p, self.perception.types) for p in pieces]
        try:
            afficher(boites, pieces.index(cible) if cible is not None else None)
        except Exception:                               # noqa: BLE001
            pass
        with self.verrou:
            self.derniere = (ep, pieces, cible, niveau)
        if niveau == "consolidee" and self.controle is not None and pieces:
            x, y = self.controle
            e = min(np.hypot(p.T_robot[0, 3] * 1000.0 - x, p.T_robot[1, 3] * 1000.0 - y)
                    for p in pieces)
            self.controle = None
            try:
                panneau(texte="controle : piece reposee vue a {:.1f} mm du point vise"
                        .format(e))
            except Exception:                           # noqa: BLE001
                pass
        quoi = "detection rapide" if niveau == "rapide" else "detection consolidee"
        if cible is not None:
            x, y, z = cible.T_robot[:3, 3] * 1000.0
            texte = "{} : {} piece(s) ; cible : {} {} en x {:.0f} y {:.0f}".format(
                quoi, len(pieces), cible.type, cible.face, x, y)
            if raison:
                texte += ("<br><b style='color:#e06c5a'>CIBLE HORS D'ATTEINTE</b> : {}."
                          "<br>Deplacez-la, ou 'Forcer la saisie'.".format(raison))
        else:
            texte = "{} : {} piece(s) ; aucune cible".format(quoi, len(pieces))
        try:
            panneau(cible=texte, saisir=cible is not None and not raison,
                    forcer=cible is not None and bool(raison))
        except Exception:                               # noqa: BLE001
            pass

    def run(self):
        while not self.fin.is_set():
            if self.pause.is_set():
                self.etape = ""
                time.sleep(0.1)
                continue
            ep = self.epoque
            self.debut, self.etape, self.affinage = time.time(), "capture", 0.0
            try:
                capture = self.capturer()
                for etape, pieces in self.perception.detecter_par_etapes(capture):
                    if self.pause.is_set() or ep != self.epoque:
                        break               # le bras bouge : resultat perime
                    self.etape = etape
                    if pieces is not None:
                        self._publier(ep, pieces, "consolidee" if etape == "consolidee"
                                      else "rapide")
                else:
                    duree = time.time() - self.debut
                    self.duree_moyenne = 0.7 * self.duree_moyenne + 0.3 * duree
            except Exception as exc:                    # noqa: BLE001
                self.etape = ""
                if str(exc) != self._erreur:
                    self._erreur = str(exc)
                    print("\n  detection impossible : {}".format(exc))
                time.sleep(1.0)

    def cible_fraiche(self, delai: float = 30.0):
        """La cible d'une detection CONSOLIDEE faite APRES le dernier
        mouvement."""
        fin = time.time() + delai
        while time.time() < fin:
            with self.verrou:
                d = self.derniere
            if d is not None and d[0] == self.epoque and d[3] == "consolidee":
                return d[2]
            time.sleep(0.1)
        return None


class Flux(threading.Thread):
    """
    Le flux video du panneau Camera : une image a la fois (pas de
    mediane), quelques fois par seconde, avec les boites de la DERNIERE
    detection. Pendant un mouvement du bras, les boites ne sont plus a
    jour : on montre l'image seule.
    """

    def __init__(self, detecteur, types, capturer=None, cadence: float = 8.0):
        super().__init__(daemon=True)
        if capturer is None:
            import camera
            capturer = lambda: camera.capturer(1, delai=5.0)  # noqa: E731
        self.capturer = capturer
        self.detecteur = detecteur
        self.types = types
        self.periode = 1.0 / float(cadence)
        self.fin = threading.Event()
        self.images = 0

    def run(self):
        import perception
        from dobot_sim.passerelle import demander
        while not self.fin.is_set():
            t0 = time.time()
            try:
                capture = self.capturer()
                vue = perception.vue_couleur(capture)
                d = self.detecteur
                with d.verrou:
                    derniere = d.derniere
                pieces, cible, niveau = [], None, ""
                if derniere is not None and derniere[0] == d.epoque \
                        and not d.pause.is_set():
                    pieces, cible, niveau = derniere[1], derniere[2], derniere[3]
                images = {"profondeur": carte_profondeur(capture)}
                if vue is not None:
                    img = dessiner_boites(vue.image, vue, pieces, self.types, cible,
                                          d.raison_cible if cible is not None else "")
                    images["couleur"] = _jpeg(bandeau(img, niveau, d))
                prog = int(round(100 * d.progression()))
                demander({"saisie": {"images": images,
                                     "progression": prog,
                                     "etape": libelle_etape(d)}}, attente=2000)
                self.images += 1
            except Exception:                           # noqa: BLE001
                time.sleep(0.5)
            time.sleep(max(0.0, self.periode - (time.time() - t0)))


LIBELLES = {"capture": "capture", "rapide": "detection rapide faite, consolidation",
            "propositions": "consolidation : affinage de la scene",
            "consolidee": "detection consolidee"}


def libelle_etape(d) -> str:
    if d.pause.is_set():
        return "robot en mouvement : detection suspendue"
    return LIBELLES.get(d.etape, "en attente")


def bandeau(img, niveau, d):
    """Ecrit en haut de l'image ce qu'on voit : boites rapides ou
    consolidees, et l'avancement de la detection en cours."""
    import cv2
    if d.pause.is_set():
        texte = "robot en mouvement"
    elif not niveau:
        texte = "detection en cours... {:.0f} %".format(100 * d.progression())
    else:
        texte = "{} -- nouvelle detection {:.0f} %".format(
            "boites RAPIDES" if niveau == "rapide" else "boites CONSOLIDEES",
            100 * d.progression())
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, texte, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 220, 255) if niveau == "rapide" else (120, 230, 120), 1, cv2.LINE_AA)
    return img


def carte_profondeur(capture) -> str:
    """Profondeur en fausses couleurs (proche rouge, loin bleu), JPEG base64."""
    import cv2
    z = np.asarray(capture.get("profondeur"), float)
    valides = z > 0
    if not z.size or not valides.any():
        return ""
    lo, hi = np.percentile(z[valides], (2, 98))
    n = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    carte = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)
    carte[~valides] = 0
    return _jpeg(carte)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Saisie guidee par la camera, dans le "
                                             "simulateur en miroir")
    ap.add_argument("--longueur-outil", type=float, default=None)
    ap.add_argument("--vitesse", type=float, default=20.0, help="mm/s (20)")
    ap.add_argument("--port", default=None, help="port serie du robot")
    ap.add_argument("--etalonnage", default=ec.FICHIER)
    ap.add_argument("--hauteur-prise", type=float, default=sp.PIED,
                    help="mm visees entre le bout des doigts et la table ({:.0f})"
                    .format(sp.PIED))
    ap.add_argument("--delai-detection", type=float, default=None,
                    help="relancer la detection N s apres l'ouverture de la pince, "
                         "pendant la remontee (defaut : a l'arrivee en equerre)")
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
                     outil="pince", vitesse=a.vitesse,
                     longueur_outil=sp.longueur_outil(a.longueur_outil, donnees))
    essai = sp.Essai(robot, perc_robot, donnees, demander=ec._demander,
                     pied=a.hauteur_prise)
    det = Detecteur(perc_continue, essai)
    from dobot_sim.passerelle import demander

    def quand_saisie(piece, x, y, z, r):
        # la piece passe de la table a la pince : sa boite quitte la scene,
        # la pince se ferme sur ses flancs et l'emporte
        lacet = np.degrees(np.arctan2(piece.T_robot[1, 0], piece.T_robot[0, 0]))
        dims = boite(piece, perc_continue.types)["dims"]
        dessous = float(piece.T_robot[2, 3] * 1000.0 - dims[2])
        try:
            demander({"saisie": {"tenue": {"dims": dims, "type": piece.type,
                                           "angle": float(ec._angle(lacet - r, 180.0)),
                                           "pied": float(z - dessous)}}})
            with det.verrou:
                autres = [p for p in (det.derniere[1] if det.derniere else [])
                          if p is not piece]
            afficher([boite(p, perc_continue.types) for p in autres])
        except Exception:                               # noqa: BLE001
            pass

    def quand_lache():
        try:
            demander({"saisie": {"tenue": None}})
        except Exception:                               # noqa: BLE001
            pass
    essai.quand_saisie, essai.quand_lache = quand_saisie, quand_lache
    try:
        # la camera, a sa place dans la vue 3D (repere optique, mm)
        demander({"scene": {"camera": {
            "T": donnees["T_robot_camera_mm"], "K": donnees.get("camera_K"),
            "taille": donnees.get("camera_taille")}}})
        if demander({"saisie": {"activer": True}}) is None:
            print("ATTENTION : la fenetre du simulateur ne connait pas le panneau Saisie "
                  "(elle a ete\n  ouverte avant cette version). Fermez-la et relancez ce "
                  "script : il en ouvrira une\n  a jour. En attendant : Entree ici pour "
                  "saisir, f + Entree pour forcer.")
        robot.pince(False)
        essai.equerre()
        det.start()
        flux = Flux(det, perc_continue.types)
        flux.start()
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
            reprise = {"faite": False}

            def reprendre_detection():
                if not reprise["faite"]:
                    reprise["faite"] = True
                    det.epoque += 1
                    det.pause.clear()

            def apres_lacher():
                # --delai-detection : la detection repart peu apres
                # l'ouverture de la pince, pendant que le bras remonte (il
                # peut encore etre dans le champ) ; sans lui, a l'arrivee
                # en equerre
                if a.delai_detection is not None:
                    threading.Timer(a.delai_detection, reprendre_detection).start()
            try:
                t0 = time.time()
                panneau(occupe=True, texte="saisie en cours{}...".format(
                    " (forcee, sans marges)" if forcer else ""))
                vise = essai.un_tour(cible, quand_meme=forcer, interactif=False,
                                     apres_lacher=apres_lacher, controle=False)
                if vise is not None:
                    det.controle = (vise[0], vise[1])
                panneau(occupe=False, forcer=False,
                        texte="tour fait en {:.0f} s ; controle a la prochaine detection "
                              "consolidee".format(time.time() - t0))
            except sp.HorsAtteinte as exc:
                print("  piece HORS D'ATTEINTE ({})".format(exc))
                panneau(occupe=False, forcer=True,
                        texte="<b style='color:#e06c5a'>HORS D'ATTEINTE</b> : {}.<br>"
                              "Deplacez la piece, ou 'Forcer la saisie'.".format(exc))
            except ErreurExecution as exc:
                # le robot (ou les garde-fous) a refuse : on le dit, on
                # tente de revenir en equerre, et on continue
                print("  MOUVEMENT REFUSE : {}".format(exc))
                retour = ""
                try:
                    essai.equerre()
                except Exception as exc2:               # noqa: BLE001
                    retour = ("<br>Le robot ne repond plus aux mouvements ({}) : "
                              "mise en route necessaire (eteindre et rallumer si une "
                              "alarme de planification persiste).".format(exc2))
                panneau(occupe=False, forcer=False,
                        texte="<b style='color:#e06c5a'>MOUVEMENT REFUSE</b> : {}{}"
                              .format(exc, retour))
            finally:
                reprendre_detection()
    except KeyboardInterrupt:
        print("\ninterrompu (robot arrete)")
    except ec.Arret as exc:
        print("arret : {}".format(exc))
    except ErreurExecution as exc:
        print("ECHEC : {}".format(exc))
    finally:
        det.fin.set()
        if "flux" in locals():
            flux.fin.set()
        try:
            demander({"saisie": {"desactiver": True}})
            demander({"scene": {"camera": None}})
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
