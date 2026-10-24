#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic de la RealSense, etape par etape. Rien ne bouge.

    sudo .venv/bin/python examples/perception/diagnostic_realsense.py

Chaque etape s'annonce AVANT d'etre tentee : si Python plante
(segmentation fault), la derniere ligne affichee dit ou.
"""
import faulthandler
import sys

faulthandler.enable()


def etape(texte):
    print("-> " + texte, flush=True)


etape("import de pyrealsense2")
import pyrealsense2 as rs                       # noqa: E402
print("   version :", getattr(rs, "__version__", "?"), flush=True)

etape("creation du contexte")
ctx = rs.context()

etape("liste des cameras")
appareils = ctx.query_devices()
print("   nombre :", len(appareils), flush=True)
if not len(appareils):
    sys.exit("aucune camera")

d = appareils[0]
for info in ("name", "serial_number", "firmware_version", "usb_type_descriptor"):
    etape("lecture de " + info)
    print("   ", d.get_info(getattr(rs.camera_info, info)), flush=True)

etape("capteurs")
for s in d.query_sensors():
    print("   ", s.get_info(rs.camera_info.name), flush=True)

for nom, flux, fmt in (("profondeur", rs.stream.depth, rs.format.z16),
                       ("couleur", rs.stream.color, rs.format.bgr8)):
    etape("demarrage du flux " + nom + " seul")
    p = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(flux, 640, 480, fmt, 15)
    p.start(cfg)
    etape("lecture d'une image " + nom)
    fr = p.wait_for_frames(10000)
    print("   ok :", [f.profile.stream_name() for f in fr], flush=True)
    p.stop()

print("TOUT EST BON", flush=True)
