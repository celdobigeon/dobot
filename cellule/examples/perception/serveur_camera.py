#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serveur de camera RealSense : ouvre la camera UNE fois et la garde ouverte.

    sudo .venv/bin/python examples/perception/serveur_camera.py [--sans-couleur] [--resolution 848x480]

C'est le SEUL script qui a besoin de sudo (macOS reserve l'acces USB direct
a l'administrateur). Laissez-le tourner dans un terminal a part : les
autres scripts lui demandent des captures, sans sudo, via
camera.capturer(). Ctrl+C l'arrete.

Pourquoi un serveur plutot qu'une capture par script :
  - la camera n'est ouverte qu'une fois : sous macOS, c'est chaque
    arret-redemarrage du flux qui risque de la bloquer ;
  - elle lit en continu : l'exposition reste reglee, et une capture ne
    coute que le temps de ses images (1 s pour 15 images a 15 Hz) ;
  - plus de sudo dans les scripts de travail.

Une fois lance, il se REGLE A DISTANCE, sans sudo (camera.regler(),
camera.redemarrer(), camera.etat()) : resolution, puissance du projecteur,
preset, exposition. Plus besoin de le relancer pour un reglage.

La COULEUR est demandee par defaut (--sans-couleur pour s'en passer). Si
elle n'arrive pas au demarrage (sous macOS, le systeme garde parfois le
capteur couleur pour lui), le serveur se relance tout seul avec la
profondeur et l'infrarouge seulement : la detection n'a pas besoin de la
couleur. (--couleur, l'ancienne option, est encore acceptee, sans effet.)

Il n'ecoute que la machine elle-meme (127.0.0.1). Rien ne bouge : la
camera ne fait que regarder.
"""

import collections
import faulthandler
import gc
import json
import os
import socketserver
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera  # noqa: E402

# le module RealSense peut planter (segmentation fault) : Python dira OU
faulthandler.enable()

try:
    import pyrealsense2 as rs
except ImportError as exc:
    sys.exit("module manquant ({}) : sous macOS, examples/perception/"
             "installer_realsense_macos.sh ; sous Linux, .venv/bin/pip install "
             "pyrealsense2".format(exc))


class Acquisition:
    """Lit la camera en continu et garde les dernieres images."""

    def __init__(self, avec_couleur: bool, largeur: int = camera.LARGEUR,
                 hauteur: int = camera.HAUTEUR):
        self.avec_couleur = avec_couleur
        self.largeur, self.hauteur = int(largeur), int(hauteur)
        self.images = collections.deque(maxlen=60)
        self.compteur = 0
        self.condition = threading.Condition()
        self.erreur = ""
        self.arret = threading.Event()

        self.pipeline = rs.pipeline()
        cfg = self._choisir_mode()
        profil = self.pipeline.start(cfg)
        try:
            self.capteur = profil.get_device().first_depth_sensor()
            self.echelle = camera.regler_capteur(rs, profil)
            intr = (profil.get_stream(rs.stream.depth)
                    .as_video_stream_profile().intrinsics)
        except Exception:
            # un flux laisse demarre garderait la camera : la relance qui
            # suit (sans couleur, ou retour a l'ancien mode) echouerait
            self.pipeline.stop()
            raise
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy],
                           [0, 0, 1]])
        self.coeffs = np.array(intr.coeffs, dtype=float)
        # geometrie de la couleur BRUTE (intrinseques, passage profondeur ->
        # couleur) : sans elle, l'image couleur entiere est inexploitable en
        # 3D. Jamais bloquant : sans geometrie, les captures restent comme
        # avant.
        self.geometrie_couleur = {}
        if avec_couleur:
            try:
                self.geometrie_couleur = geometrie_couleur(profil)
            except Exception as exc:
                print("geometrie de la couleur illisible ({}) : captures sans "
                      "K_couleur ni T_couleur_profondeur".format(exc),
                      flush=True)
        # la couleur est ramenee sur la profondeur : pixel pour pixel
        self.aligner = rs.align(rs.stream.depth) if avec_couleur else None
        self.fil = threading.Thread(target=self._boucle, daemon=True)
        self.fil.start()

    def _boucle(self):
        echecs = 0
        while not self.arret.is_set():
            try:
                fr = self.pipeline.wait_for_frames(3000)
            except RuntimeError as exc:
                echecs += 1
                if echecs >= 3:
                    self.erreur = ("la camera ne repond plus ({}). Debranchez-"
                                   "la, rebranchez-la, relancez le serveur."
                                   .format(exc))
                    with self.condition:
                        self.condition.notify_all()
                    return
                continue
            echecs = 0
            brute = None
            if self.avec_couleur:
                c0 = fr.get_color_frame()
                if c0:
                    # la couleur ENTIERE, telle que le capteur la voit : le
                    # recalage sur la profondeur la rogne et la troue
                    brute = np.asanyarray(c0.get_data()).copy()
            if self.aligner is not None:
                fr = self.aligner.process(fr)
            d = fr.get_depth_frame()
            i = fr.get_infrared_frame(1)
            if not d or not i:
                continue
            image = {"profondeur": np.asanyarray(d.get_data()).copy(),
                     "infrarouge": np.asanyarray(i.get_data()).copy()}
            if self.avec_couleur:
                c = fr.get_color_frame()
                if c:
                    image["couleur"] = np.asanyarray(c.get_data()).copy()
                if brute is not None:
                    image["couleur_brute"] = brute
            with self.condition:
                self.images.append(image)
                self.compteur += 1
                self.condition.notify_all()

    def capture(self, n: int, delai: float = 20.0) -> dict:
        """Mediane des n PROCHAINES images : rien d'anterieur a la demande."""
        n = int(min(max(n, 1), self.images.maxlen))
        with self.condition:
            depart = self.compteur
            fin = time.time() + delai
            while self.compteur - depart < n and not self.erreur:
                reste = fin - time.time()
                if reste <= 0:
                    raise camera.ErreurCamera("pas assez d'images en {:.0f} s"
                                              .format(delai))
                self.condition.wait(reste)
            if self.erreur:
                raise camera.ErreurCamera(self.erreur)
            images = list(self.images)[-n:]
        capture = {
            "profondeur": camera.mediane([im["profondeur"] * self.echelle
                                          for im in images]),
            "infrarouge": images[-1]["infrarouge"],
            "K": self.K, "coeffs": self.coeffs,
        }
        for cle in ("couleur", "couleur_brute"):
            if cle in images[-1]:
                capture[cle] = images[-1][cle]
        if "couleur_brute" in capture:
            capture.update(self.geometrie_couleur)
        return capture

    #: modes essayes, du meilleur au plus modeste, apres celui demande : la
    #: camera n'offre pas les memes selon la liaison (USB 2 ou 3)
    MODES = [(848, 480), (1280, 720), (640, 480), (640, 360), (480, 270),
             (424, 240)]
    CADENCES = [15, 30, 6, 5]

    def _config(self, largeur, hauteur, cadence):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, largeur, hauteur, rs.format.z16,
                          cadence)
        cfg.enable_stream(rs.stream.infrared, 1, largeur, hauteur,
                          rs.format.y8, cadence)
        if self.avec_couleur:
            cfg.enable_stream(rs.stream.color, largeur, hauteur,
                              rs.format.bgr8, cadence)
        return cfg

    def _choisir_mode(self):
        """
        Le mode demande s'il est disponible, sinon le meilleur qui l'est.
        `config.can_resolve` interroge la camera sans demarrer le flux.
        """
        demande = (self.largeur, self.hauteur)
        candidats = [demande] + [m for m in self.MODES if m != demande]
        enveloppe = rs.pipeline_wrapper(self.pipeline)
        for (l, h) in candidats:
            for cadence in self.CADENCES:
                cfg = self._config(l, h, cadence)
                if cfg.can_resolve(enveloppe):
                    if (l, h) != demande:
                        print("{}x{} indisponible (liaison USB 2 ?) : {}x{} a "
                              "{} images/s".format(*demande, l, h, cadence),
                              flush=True)
                    self.largeur, self.hauteur, self.cadence = l, h, cadence
                    return cfg
        raise camera.ErreurCamera("aucun mode profondeur + infrarouge{} "
                                  "disponible".format(
                                      " + couleur" if self.avec_couleur
                                      else ""))

    #: reglages modifiables a chaud : nom -> option RealSense
    OPTIONS = {"laser": "laser_power", "preset": "visual_preset",
               "exposition_auto": "enable_auto_exposure",
               "exposition": "exposure", "projecteur": "emitter_enabled"}

    def regler(self, reglages: dict) -> dict:
        """Applique des options a chaud (sans arreter le flux)."""
        faits = {}
        for nom, valeur in reglages.items():
            option = getattr(rs.option, self.OPTIONS.get(nom, nom), None)
            if option is None or not self.capteur.supports(option):
                raise camera.ErreurCamera("reglage inconnu : " + nom)
            self.capteur.set_option(option, float(valeur))
            faits[nom] = self.capteur.get_option(option)
        return faits

    def etat(self) -> dict:
        e = {"resolution": "{}x{}".format(self.largeur, self.hauteur),
             "cadence": getattr(self, "cadence", None),
             "couleur": self.avec_couleur, "images_lues": self.compteur}
        for nom, attr in self.OPTIONS.items():
            option = getattr(rs.option, attr, None)
            if option is not None and self.capteur.supports(option):
                e[nom] = self.capteur.get_option(option)
        return e

    def fermer(self):
        self.arret.set()
        self.fil.join(timeout=5)
        try:
            self.pipeline.stop()
        except Exception:
            pass


def geometrie_couleur(profil) -> dict:
    """
    Intrinseques de la camera couleur et passage du repere de la profondeur
    a celui de la couleur, lus dans les profils de flux du pipeline demarre.

    Rend {"K_couleur": 3x3, "coeffs_couleur": (5,), "T_couleur_profondeur":
    4x4 en metres} : un point p de la profondeur (repere camera, m) se voit
    dans la couleur en T_couleur_profondeur @ [p, 1], puis se projette par
    K_couleur. librealsense donne la rotation en ordre COLONNE d'abord
    (rs2_extrinsics) : d'ou la transposee.
    """
    p_d = profil.get_stream(rs.stream.depth).as_video_stream_profile()
    p_c = profil.get_stream(rs.stream.color).as_video_stream_profile()
    ic = p_c.intrinsics
    ex = p_d.get_extrinsics_to(p_c)
    T = np.eye(4)
    T[:3, :3] = np.array(ex.rotation, dtype=float).reshape(3, 3).T
    T[:3, 3] = np.array(ex.translation, dtype=float)
    return {"K_couleur": np.array([[ic.fx, 0, ic.ppx], [0, ic.fy, ic.ppy],
                                   [0, 0, 1]], dtype=float),
            "coeffs_couleur": np.array(ic.coeffs, dtype=float),
            # le sens des coefficients depend du modele ("inverse_brown_
            # conrady" : ils servent a la deprojection, pas a la projection)
            "modele_couleur": np.array(str(getattr(ic, "model", ""))),
            "T_couleur_profondeur": T}


def lire_resolution(texte: str):
    try:
        l, h = (int(v) for v in texte.lower().split("x"))
        return l, h
    except ValueError:
        sys.exit("resolution illisible : {} (exemple : 848x480)".format(texte))


def couleur_arrive(acquisition: Acquisition, delai: float = 5.0) -> bool:
    """
    Vrai si une image couleur arrive dans le delai. Les premieres images
    peuvent venir sans elle (le capteur couleur demarre apres la
    profondeur) : on en laisse passer plusieurs avant de conclure.
    """
    fin = time.time() + delai
    while time.time() < fin:
        c = acquisition.capture(1, delai=max(fin - time.time(), 0.5))
        if "couleur" in c:
            return True
    return False


def ouvrir(avec_couleur: bool, resolution) -> Acquisition:
    """
    Demarre le flux et verifie par une premiere capture que la camera
    repond (et, si on l'a demandee, que la couleur arrive). En cas d'echec,
    le flux est arrete avant de lever l'erreur : la camera reste libre pour
    une nouvelle tentative.
    """
    acquisition = Acquisition(avec_couleur, *resolution)
    try:
        acquisition.capture(1, delai=8.0)          # la camera repond-elle ?
        if avec_couleur and not couleur_arrive(acquisition):
            raise camera.ErreurCamera("aucune image couleur")
    except Exception:
        acquisition.fermer()
        raise
    return acquisition


def main():
    # couleur par defaut ; "--couleur" (l'ancienne option) reste accepte
    avec_couleur = "--sans-couleur" not in sys.argv
    resolution = (848, 480)       # la resolution optimale de la D435
    if "--resolution" in sys.argv:
        resolution = lire_resolution(sys.argv[sys.argv.index("--resolution") + 1])

    # Sonde d'abord : sans les droits, pipeline.start() FAIT PLANTER Python
    # sous macOS au lieu de lever une erreur.
    try:
        appareils = rs.context().query_devices()
        if len(appareils) == 0:
            sys.exit("aucune RealSense branchee")
        nom = appareils[0].get_info(rs.camera_info.name)
        try:
            usb = appareils[0].get_info(rs.camera_info.usb_type_descriptor)
            nom += " (USB {})".format(usb)
            if usb.startswith("2"):
                print("ATTENTION : camera en USB {} : resolutions et cadences "
                      "reduites. Cable ou port USB 3 conseille.".format(usb),
                      flush=True)
        except Exception:
            pass
        # ne pas garder ce handle pendant que le pipeline ouvre la camera
        del appareils
        gc.collect()
        time.sleep(1.0)
        try:
            acquisition = ouvrir(avec_couleur, resolution)
        except RuntimeError as exc:                # ErreurCamera comprise
            if not avec_couleur:
                raise
            # Repli : la detection n'a pas besoin de la couleur. Une pause
            # avant de rouvrir : enchainer arret et redemarrage sans
            # souffler est ce qui bloquait la camera sous macOS.
            print("couleur indisponible : profondeur et infrarouge "
                  "seulement. Relance du flux sans la couleur...\n  (essai "
                  "avec la couleur : {})".format(exc), flush=True)
            time.sleep(2.0)
            try:
                acquisition = ouvrir(False, resolution)
            except RuntimeError as exc2:
                sys.exit("camera inaccessible apres l'essai de la couleur : "
                         "{}\n  - debranchez et rebranchez la camera (cable "
                         "USB 3, direct sur l'ordinateur)\n  - puis relancez "
                         "SANS la couleur :\n    sudo .venv/bin/python "
                         "examples/perception/serveur_camera.py --sans-couleur"
                         .format(exc2))
    except (RuntimeError, camera.ErreurCamera) as exc:
        sys.exit("camera inaccessible : {}\n  - sous macOS, lancez avec sudo\n"
                 "  - sinon, debranchez et rebranchez la camera (cable USB 3, "
                 "direct sur l'ordinateur)".format(exc))

    # l'acquisition peut etre REMPLACEE (redemarrage a distance) : on la
    # garde dans une boite que les gestionnaires consultent a chaque demande
    courante = {"acq": acquisition}
    verrou = threading.Lock()

    def redemarrer(d):
        """Change la resolution ou la couleur : arret puis relance du flux."""
        with verrou:
            ancienne = courante["acq"]
            res = lire_resolution(d.get("resolution", "{}x{}".format(
                ancienne.largeur, ancienne.hauteur)))
            couleur = bool(d.get("couleur", ancienne.avec_couleur))
            ancienne.fermer()
            time.sleep(1.0)
            try:
                # la couleur demandee doit arriver, sinon retour a l'ancienne
                # configuration (etat() dirait "couleur" a tort)
                nouvelle = ouvrir(couleur, res)
            except Exception as exc:
                # retour a l'ancienne configuration, qui marchait
                time.sleep(1.0)
                courante["acq"] = Acquisition(ancienne.avec_couleur,
                                              ancienne.largeur, ancienne.hauteur)
                raise camera.ErreurCamera(
                    "redemarrage en {}x{} impossible ({}) : retour en {}x{}"
                    .format(*res, exc, ancienne.largeur, ancienne.hauteur))
            courante["acq"] = nouvelle
            print("camera redemarree : {}x{}{}".format(
                *res, ", couleur" if couleur else ""), flush=True)
            return nouvelle.etat()

    class Gestionnaire(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                demande = json.loads(camera.recevoir_message(self.request))
            except Exception:
                return                              # simple test de presence
            try:
                quoi = demande.get("demande")
                acquisition = courante["acq"]
                if quoi in ("regler", "etat", "redemarrer"):
                    if quoi == "regler":
                        rep = acquisition.regler(demande.get("reglages", {}))
                        print("reglages : {}".format(rep), flush=True)
                    elif quoi == "redemarrer":
                        rep = redemarrer(demande)
                    else:
                        rep = acquisition.etat()
                    camera.envoyer_message(self.request,
                                           json.dumps(rep).encode("utf-8"))
                    return
                if quoi == "capture":
                    debut = time.time()
                    c = acquisition.capture(int(demande.get("images",
                                                            camera.N_IMAGES)))
                    camera.envoyer_message(self.request, camera.empaqueter(c))
                    # une image seule, c'est un flux video : on ne le
                    # journalise pas, il inonderait le terminal
                    if int(demande.get("images", camera.N_IMAGES)) > 1:
                        valides = c["profondeur"] > 0
                        print("capture servie : {:.0f} % de pixels mesures, "
                              "{:.1f} s".format(100 * valides.mean(),
                                                time.time() - debut), flush=True)
                else:
                    camera.envoyer_message(self.request, b"ERREUR demande "
                                           b"inconnue")
            except Exception as exc:
                camera.envoyer_message(self.request,
                                       ("ERREUR " + str(exc)).encode("utf-8"))

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    serveur = socketserver.ThreadingTCPServer((camera.HOTE, camera.PORT),
                                              Gestionnaire)
    serveur.daemon_threads = True
    print("serveur de camera pret : {} en {}x{} sur {}:{}{}. Ctrl+C pour "
          "l'arreter.".format(nom, acquisition.largeur, acquisition.hauteur,
                              camera.HOTE, camera.PORT,
                              ", avec la couleur" if acquisition.avec_couleur
                              else ", SANS la couleur"),
          flush=True)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\narret du serveur", flush=True)
    finally:
        serveur.server_close()
        courante["acq"].fermer()


if __name__ == "__main__":
    main()
