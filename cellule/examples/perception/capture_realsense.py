#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture d'une scene avec la RealSense : profondeur, infrarouge, parametres.

    sudo .venv/bin/python examples/perception/capture_realsense.py [nom] [--couleur]

Enregistre examples/perception/captures/<nom>.npz (defaut : scene) :
  profondeur   en METRES, float32, H x W (0 = pas de mesure) ; mediane de
               N images, nettement moins bruitee qu'une image seule
  infrarouge   image infrarouge (camera de gauche), uint8, H x W : prise
               par le MEME capteur que la profondeur, donc parfaitement
               alignee avec elle, sans recalage
  K            parametres intrinseques de cette camera (fx, fy, cx, cy)
  couleur      image couleur alignee sur la profondeur (pixel pour pixel :
               noire la ou la profondeur manque), SI elle arrive -- sous
               macOS, le systeme garde parfois le capteur couleur
  couleur_brute  la meme image couleur, entiere, non alignee
et un apercu <nom>.png (infrarouge | profondeur en fausses couleurs).

Si le serveur de camera tourne (serveur_camera.py), la capture lui est
demandee : PAS de sudo, et la camera n'est pas rouverte. La couleur y est
alors si le serveur l'a (il la demande par defaut) ; --couleur ne sert
pas dans ce cas. Sinon, le script ouvre la camera lui-meme, et ne tente la
couleur que sur demande (--couleur).

Dans ce second cas, sous macOS, `sudo` est obligatoire : sans lui, la camera refuse de
s'ouvrir ("failed to set power state"). Rien ne bouge : la camera ne
fait que regarder.
"""

import faulthandler
import gc
import os
import sys
import time

# le module RealSense pour macOS plante parfois (segmentation fault) : au
# moins, Python dira OU
faulthandler.enable()

import numpy as np

try:
    import cv2
    import pyrealsense2 as rs
except ImportError as exc:
    sys.exit("module manquant ({}) : .venv/bin/pip install opencv-python, "
             "puis, sous macOS, examples/perception/installer_realsense_macos.sh "
             "(sous Linux : .venv/bin/pip install pyrealsense2)".format(exc))

LARGEUR, HAUTEUR, CADENCE = 640, 480, 15     # tient en USB 2
N_IMAGES = 15                                # pour la mediane
ECHAUFFEMENT = 30                            # images jetees : exposition
PUISSANCE_LASER = 90                         # mW (defaut 150, max 360)


def demarrer(avec_couleur):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, LARGEUR, HAUTEUR, rs.format.z16, CADENCE)
    cfg.enable_stream(rs.stream.infrared, 1, LARGEUR, HAUTEUR, rs.format.y8,
                      CADENCE)
    if avec_couleur:
        cfg.enable_stream(rs.stream.color, LARGEUR, HAUTEUR, rs.format.bgr8,
                          CADENCE)
    profil = pipeline.start(cfg)
    return pipeline, profil


def demarrer_fiable(avec_couleur, essais=3):
    """
    Demarre le flux et attend une premiere image. Sous macOS, un flux qu'on
    vient d'arreter ne repart pas toujours du premier coup : on reessaie,
    avec une pause.
    """
    for k in range(essais):
        pipeline, profil = demarrer(avec_couleur)
        try:
            pipeline.wait_for_frames(5000)
            return pipeline, profil
        except RuntimeError:
            pipeline.stop()
            if k + 1 < essais:
                print("  la camera ne repond pas, nouvel essai...", flush=True)
                time.sleep(2.0)
    raise RuntimeError("aucune image apres {} essais : debranchez et "
                       "rebranchez la camera".format(essais))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nom = args[0] if args else "scene"
    essayer_couleur = "--couleur" in sys.argv

    # le serveur de camera tourne : on lui demande, sans sudo
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import camera
    if camera.serveur_present():
        dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "captures")
        os.makedirs(dossier, exist_ok=True)
        c = camera.capturer()
        chemin = os.path.join(dossier, nom + ".npz")
        np.savez_compressed(chemin, **c)
        print("capture (par le serveur de camera) : {} ({:.0f} % de pixels "
              "mesures)".format(chemin, 100 * (c["profondeur"] > 0).mean()))
        return
    dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "captures")

    # Sonde d'abord la camera : sans les droits, pipeline.start() ne leve
    # pas d'erreur sous macOS, il fait PLANTER Python (segmentation fault).
    try:
        appareils = rs.context().query_devices()
        if len(appareils) == 0:
            sys.exit("aucune RealSense branchee")
        print("camera :", appareils[0].get_info(rs.camera_info.name),
              flush=True)
        # Ne PAS garder ce handle ouvert pendant que le pipeline ouvre la
        # meme camera : deux acces simultanes font planter le module sous
        # macOS. On le libere, et on laisse a l'USB le temps de souffler.
        del appareils
        gc.collect()
        time.sleep(1.0)
    except RuntimeError as exc:
        sys.exit("camera inaccessible : {}\n  sous macOS, lancez avec sudo : "
                 "sudo .venv/bin/python {}".format(exc, " ".join(sys.argv)))
    os.makedirs(dossier, exist_ok=True)

    # Sous macOS, la couleur n'arrive le plus souvent pas (le systeme garde
    # le capteur), et l'arret-redemarrage qui suit laisse parfois la camera
    # muette : on ne la tente que sur demande (--couleur).
    avec_couleur = False
    if essayer_couleur:
        try:
            pipeline, profil = demarrer_fiable(True, essais=1)
            avec_couleur = True
        except RuntimeError:
            # surtout ne PAS redemarrer la camera dans le meme processus :
            # c'est ce redemarrage qui la bloque sous macOS
            sys.exit("couleur indisponible. Debranchez et rebranchez la "
                     "camera, puis relancez SANS --couleur.")
    if not avec_couleur:
        try:
            pipeline, profil = demarrer_fiable(False)
        except RuntimeError as exc:
            sys.exit(str(exc))

    try:
        capteur = profil.get_device().first_depth_sensor()
        echelle = capteur.get_depth_scale()
        # preset "haute densite" (4) : le plus de pixels mesures possible.
        # "Haute precision" (3) en laisse trop vides sur une table sombre.
        if capteur.supports(rs.option.visual_preset):
            try:
                capteur.set_option(rs.option.visual_preset, 4)
            except Exception:
                pass
        # Bois clair + lampe + projecteur : l'infrarouge SATURE sur les
        # pieces, et la profondeur y laisse un trou. Exposition automatique,
        # et projecteur moins fort (il reste assez de motif pour la table).
        for option, valeur in ((rs.option.enable_auto_exposure, 1),
                               (rs.option.laser_power, PUISSANCE_LASER)):
            if capteur.supports(option):
                try:
                    capteur.set_option(option, valeur)
                except Exception:
                    pass
        aligner = rs.align(rs.stream.depth) if avec_couleur else None
        for _ in range(ECHAUFFEMENT):
            pipeline.wait_for_frames(5000)
        profondeurs, ir, couleur, brute = [], None, None, None
        for _ in range(N_IMAGES):
            fr = pipeline.wait_for_frames(5000)
            if avec_couleur:
                c0 = fr.get_color_frame()
                if c0:
                    brute = np.asanyarray(c0.get_data()).copy()
            if aligner is not None:
                fr = aligner.process(fr)
            d = fr.get_depth_frame()
            if not d:
                continue
            profondeurs.append(np.asanyarray(d.get_data()).astype(np.float32)
                               * echelle)
            i = fr.get_infrared_frame(1)
            if i:
                ir = np.asanyarray(i.get_data()).copy()
            if avec_couleur:
                c = fr.get_color_frame()
                if c:
                    couleur = np.asanyarray(c.get_data()).copy()
        intr = d.profile.as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()

    pile = np.stack(profondeurs)
    # mediane des seules mesures valides ; un pixel jamais mesure reste a 0
    pile[pile <= 0] = np.nan
    jamais = np.all(np.isnan(pile), axis=0)
    pile[:, jamais] = 0.0
    profondeur = np.nan_to_num(np.nanmedian(pile, axis=0), nan=0.0)
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    chemin = os.path.join(dossier, nom + ".npz")
    donnees = dict(profondeur=profondeur.astype(np.float32), K=K,
                   coeffs=np.array(intr.coeffs, dtype=float),
                   infrarouge=ir if ir is not None else np.zeros(profondeur.shape, np.uint8))
    if couleur is not None:
        donnees["couleur"] = couleur
    if brute is not None:
        donnees["couleur_brute"] = brute
    np.savez_compressed(chemin, **donnees)

    vue = cv2.applyColorMap(cv2.convertScaleAbs(profondeur, alpha=255 / 1.2),
                            cv2.COLORMAP_JET)
    vue[profondeur == 0] = 0
    gauche = (brute if brute is not None
              else cv2.cvtColor(donnees["infrarouge"], cv2.COLOR_GRAY2BGR))
    apercu = os.path.join(dossier, nom + ".png")
    cv2.imwrite(apercu, np.hstack([gauche, vue]))
    valides = profondeur > 0
    if ir is not None:
        print("infrarouge sature (255) : {:.1f} % des pixels".format(
            100 * (ir >= 250).mean()), flush=True)
    print("capture : {} ({} images, {:.0f} % de pixels mesures, profondeur "
          "{:.2f} a {:.2f} m, couleur {})".format(
              chemin, len(profondeurs), 100 * valides.mean(),
              profondeur[valides].min(), profondeur[valides].max(),
              "oui" if couleur is not None else "non"), flush=True)
    # le fichier appartient a root sous sudo : le rendre a l'utilisateur
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        for f in (chemin, apercu, dossier):
            os.chown(f, int(uid), int(gid))


if __name__ == "__main__":
    main()
