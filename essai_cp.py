#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Essai : le Dobot peut-il enchainer des points SANS s'arreter ?

    python essai_cp.py ptp     reference : mode point a point (PTP)
    python essai_cp.py cp      mode trajectoire continue (CP)
    python essai_cp.py cptr    mode CP "temps reel" : on envoie notre propre
                               trajectoire, echantillonnee toutes les 100 ms
                               (ECHEC constate : lent, grezillement)
    python essai_cp.py cp2     CP, VITESSE PAR POINT : aller droit a 30 mm/s
                               (4,0 s attendues), retour en demi-cercle de
                               rayon 60 mm a 80 mm/s (2,4 s attendues),
                               premier point a 1 mm pour absorber l'a-coup
                               (CONSTATE : vitesse par point ignoree, ~50 mm/s)
    python essai_cp.py cp3     meme trajet ; la vitesse passe par CPParams
                               (junctionVel) MIS EN FILE : 30 pour l'aller,
                               80 avant l'arc
                               (CONSTATE : aller exact a 30 ; arc plafonne ~62)
    python essai_cp.py cp4     ligne droite seulement : aller a 80, retour a
                               120 mm/s, acceleration 200 puis 400 (deux
                               allers-retours) -- plafond global ou courbure ?

Meme trajet pour les trois : aller-retour en ligne droite de 120 mm en y,
a x = 220 et z = 80 mm (ventouse a 80 mm au-dessus de la table), par
13 points. Pendant l'essai, la position du robot est relevee toutes les
50 ms : le script MESURE les arrets aux points intermediaires et la duree
reelle.

LE ROBOT BOUGE : chaque essai attend Entree avant de partir. Ctrl+C =
arret d'urgence. Simulateur ferme (le port doit etre libre).
"""

import struct
import sys
import time

import numpy as np

from dobot_sim import DobotMagician, cible
from dobot_sim import outils as ou
from dobot_sim.liaison import (PTP, CommandeDobot, FilePleine,
                               ports_disponibles, resume_alarmes)
from dobot_sim.reel import PontReel

# commandes CP du protocole V1.1.5, section 1.11
CP_PARAMS = 90          # planAcc, junctionVel, acc|period (float), realTimeTrack (uint8)
CP_CMD = 91             # cpMode (uint8, 1 = absolu), x, y, z, velocity (float)

X, Z = 220.0, 80.0      # en coordonnees du SIMULATEUR (pointe de ventouse)
Y_MIN, Y_MAX = -60.0, 60.0
N_POINTS = 13           # points par trajet : tous les 10 mm
DUREE_TRAJET = 3.0      # s, pour le mode temps reel
PERIODE = 0.1           # s, echantillonnage du mode temps reel
VITESSE = 60.0          # mm/s, modes ptp et cp


def trajet_points():
    """Aller puis retour, en points du simulateur (x, y, z)."""
    aller = [(X, y, Z) for y in np.linspace(Y_MIN, Y_MAX, N_POINTS)]
    retour = aller[::-1][1:]
    return aller[1:] + retour


def trajet_temps_reel():
    """
    Aller-retour en DUREE_TRAJET s chacun, loi en polynome de degre 5
    (vitesse et acceleration nulles aux bouts), echantillonne a PERIODE.
    """
    t = np.arange(PERIODE, DUREE_TRAJET + 1e-9, PERIODE) / DUREE_TRAJET
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    aller = [(X, Y_MIN + (Y_MAX - Y_MIN) * v, Z) for v in s]
    retour = [(X, Y_MAX - (Y_MAX - Y_MIN) * v, Z) for v in s]
    return aller + retour


V_ALLER, V_RETOUR = 30.0, 80.0
R_ARC = 60.0


def trajet_cp2():
    """
    [(x, y, z, vitesse)] : un point a 1 mm du depart, l'aller droit tous
    les 10 mm, puis le retour par un demi-cercle cote exterieur (x jusqu'a
    X + R_ARC), un point tous les 5 mm d'arc.
    """
    pts = [(X, Y_MIN + 1.0, Z, V_ALLER)]
    pts += [(X, y, Z, V_ALLER) for y in np.linspace(Y_MIN, Y_MAX, N_POINTS)[1:]]
    n_arc = int(round(np.pi * R_ARC / 5.0))
    for a in np.linspace(0.0, np.pi, n_arc + 1)[1:]:
        pts.append((X + R_ARC * np.sin(a), Y_MAX * np.cos(a), Z, V_RETOUR))
    return pts


def attendre(li, index, releves, t0, limite=60.0):
    """Attend que le controleur ait execute `index`, en relevant la pose."""
    while time.time() - t0 < limite:
        p = li.pose()
        releves.append((time.time() - t0, p.x, p.y, p.z))
        if li.index_courant() >= index and len(releves) > 2:
            # deux releves de plus, pour voir l'arret final
            for _ in range(3):
                time.sleep(0.05)
                p = li.pose()
                releves.append((time.time() - t0, p.x, p.y, p.z))
            return True
        al = li.alarmes()
        if al:
            print("ROBOT EN DEFAUT :", resume_alarmes(al))
            return False
        time.sleep(0.05)
    print("delai depasse")
    return False


def verser(li, commandes, releves, t0, marge=24):
    """
    Envoie les commandes en file au fil de l'eau (la file ne tient que 32
    commandes), tout en relevant la pose. Retourne l'index de la derniere.
    """
    dernier = li.dernier_index
    reste = list(commandes)
    while reste:
        courant = li.index_courant()
        while reste and dernier - courant < marge:
            ident, params = reste[0]
            try:
                li.echanger(ident, params, ecriture=True, en_file=True)
            except FilePleine:
                break                    # on la renverra au tour suivant
            reste.pop(0)
            dernier = li.dernier_index
        p = li.pose()
        releves.append((time.time() - t0, p.x, p.y, p.z))
        time.sleep(0.05)
    return dernier


def analyse(releves, mode, duree_voulue=None):
    t = np.array([r[0] for r in releves])
    xyz = np.array([r[1:] for r in releves])
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
    tm = (t[1:] + t[:-1]) / 2
    en_mouvement = np.where(v > 5.0)[0]
    if not len(en_mouvement):
        print("le robot n'a pas bouge")
        return
    debut, fin = tm[en_mouvement[0]], tm[en_mouvement[-1]]
    milieu = (tm > debut + 0.15) & (tm < fin - 0.15)
    # arrets : passages sous 5 mm/s au milieu du mouvement (hors demi-tour)
    lent = milieu & (v < 5.0)
    arrets = int(np.sum(np.diff(lent.astype(int)) == 1))
    print()
    print("=== resultat ({}) ===".format(mode))
    print("  duree du mouvement : {:.2f} s{}".format(
        fin - debut, "" if duree_voulue is None
        else "   (demandee : {:.2f} s)".format(duree_voulue)))
    print("  vitesse maximale   : {:.0f} mm/s".format(v.max()))
    print("  passages a l'arret en cours de route : {}".format(arrets))
    print("    (le demi-tour en compte normalement 1 ; PTP devrait en "
          "montrer ~{})".format(2 * (N_POINTS - 1) - 1))
    nom = "essai_{}.csv".format(mode)
    np.savetxt(nom, np.column_stack([t, xyz]), delimiter=";",
               header="t;x;y;z (controleur)", comments="", fmt="%.3f")
    print("  releves bruts : {}".format(nom))
    if mode == "cp4":
        # quatre traversees : vitesse de palier de chacune
        ys = xyz[:, 1]
        sens = np.sign(np.diff(ys))
        bords = [0] + [i for i in range(1, len(sens))
                       if sens[i] != 0 and sens[i] != sens[i - 1]
                       and abs(ys[i] - (Y_MAX if sens[i - 1] > 0 else Y_MIN)) < 15]
        bords.append(len(v))
        demandes = ["80 (acc 200)", "120 (acc 200)", "80 (acc 400)", "120 (acc 400)"]
        for k in range(min(4, len(bords) - 1)):
            seg = v[bords[k]:bords[k + 1]]
            seg = seg[seg > 0.5 * seg.max()] if len(seg) else seg
            if len(seg):
                print("  traversee {} : demande {} -> palier {:.0f} mm/s"
                      .format(k + 1, demandes[k], np.median(seg)))
    if mode in ("cp2", "cp3"):
        # duree de chaque branche : l'aller finit au point le plus a y max
        ys = xyz[:, 1]
        i_demi = int(np.argmax(ys))
        mouv = t[(np.r_[0, v] > 5.0)]
        if len(mouv):
            t_debut, t_fin = mouv[0], mouv[-1]
            print("  aller (droit, {:.0f} mm/s) : {:.2f} s   (attendu {:.2f} s)"
                  .format(V_ALLER, t[i_demi] - t_debut,
                          (Y_MAX - Y_MIN) / V_ALLER))
            print("  retour (arc, {:.0f} mm/s)  : {:.2f} s   (attendu {:.2f} s)"
                  .format(V_RETOUR, t_fin - t[i_demi],
                          np.pi * R_ARC / V_RETOUR))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("ptp", "cp", "cptr", "cp2", "cp3", "cp4"):
        print(__doc__)
        return 1

    robot = DobotMagician()
    robot.geo.lt = ou.outil("ventouse").longueur
    for x, y, z, _v in trajet_cp2():
        q = robot.mgi(cible(x, y, z))
        assert not robot.depassements(q) and not robot.collisions(q)

    ports = ports_disponibles()
    if not ports:
        print("aucun port serie")
        return 1
    pont = PontReel(robot, journal=lambda m: print("  [robot] " + m))
    pont.connecter(ports[0][0])
    li = pont.liaison
    try:
        if li.alarmes():
            print("robot en defaut :", resume_alarmes(li.alarmes()))
            print("faites la mise en route dans le simulateur, fermez-le, "
                  "puis relancez")
            return 1
        ctrl = lambda x, y, z: pont.vers_controleur(x, y, z, 0.0)[:3]

        # 1. aller au depart, en PTP, lentement
        li.arreter_file()
        li.vider_file()
        li.definir_ratio(100, 100)
        li.echanger(CommandeDobot.PARAMS_PTP_CARTESIEN,
                    struct.pack("<4f", VITESSE, 100.0, 200.0, 100.0),
                    ecriture=True)
        input("\nLe robot va se placer au depart (x={:.0f}, y={:.0f}, "
              "z={:.0f}). Entree pour continuer, Ctrl+C pour annuler. "
              .format(X, Y_MIN, Z))
        idx = li.ptp(PTP.MOVJ_XYZ, *ctrl(X, Y_MIN, Z), 0.0)
        li.demarrer_file()
        if not attendre(li, idx, [], time.time()):
            return 1
        li.arreter_file()
        li.vider_file()

        # 2. preparer l'essai
        duree_voulue = None
        if mode == "ptp":
            cmds = [(CommandeDobot.PTP,
                     struct.pack("<B4f", PTP.MOVL_XYZ, *ctrl(*p), 0.0))
                    for p in trajet_points()]
        elif mode == "cp4":
            def params(v, a):
                return (CP_PARAMS, struct.pack("<3fB", a, v, a, 0))
            ys = np.linspace(Y_MIN, Y_MAX, N_POINTS)
            cmds = [(CP_CMD, struct.pack("<B4f", 1, *ctrl(X, Y_MIN + 1.0, Z), 0.0))]
            for acc in (200.0, 400.0):
                for v, sens in ((80.0, ys[1:]), (120.0, ys[::-1][1:])):
                    cmds.append(params(v, acc))
                    cmds += [(CP_CMD, struct.pack("<B4f", 1, *ctrl(X, y, Z), v))
                             for y in sens]
        elif mode == "cp3":
            # hypothese : junctionVel fixe la vitesse de croisiere ; mis EN
            # FILE, il change au bon moment de la trajectoire
            def params(v):
                return (CP_PARAMS, struct.pack("<3fB", 200.0, v, 200.0, 0))
            pts = trajet_cp2()
            cmds = [params(V_ALLER)]
            for i, (x, y, z, v) in enumerate(pts):
                if i and v != pts[i - 1][3]:
                    cmds.append(params(v))
                cmds.append((CP_CMD, struct.pack("<B4f", 1, *ctrl(x, y, z), v)))
        elif mode == "cp2":
            li.echanger(CP_PARAMS, struct.pack("<3fB", 200.0, 50.0, 200.0, 0),
                        ecriture=True)
            cmds = [(CP_CMD, struct.pack("<B4f", 1, *ctrl(x, y, z), v))
                    for x, y, z, v in trajet_cp2()]
        elif mode == "cp":
            li.echanger(CP_PARAMS, struct.pack("<3fB", 200.0, 50.0, 200.0, 0),
                        ecriture=True)
            cmds = [(CP_CMD, struct.pack("<B4f", 1, *ctrl(*p), VITESSE))
                    for p in trajet_points()]
        else:
            # realTimeTrack = 1, periode d'interpolation en ms (unite non
            # documentee : c'est aussi ce que l'essai doit nous dire)
            li.echanger(CP_PARAMS, struct.pack("<3fB", 200.0, 50.0,
                                               PERIODE * 1000.0, 1),
                        ecriture=True)
            cmds = [(CP_CMD, struct.pack("<B4f", 1, *ctrl(*p), 0.0))
                    for p in trajet_temps_reel()]
            duree_voulue = 2 * DUREE_TRAJET
        print("{} commandes a envoyer en mode {}".format(len(cmds), mode))

        if mode in ("cp2", "cp3"):
            print("aller droit vers y = +60, retour en demi-cercle cote "
                  "exterieur (jusqu'a x = {:.0f})".format(X + R_ARC))
        input("\nLE ROBOT VA FAIRE L'ALLER-RETOUR ({}). Entree pour lancer, "
              "Ctrl+C = arret d'urgence. ".format(mode))
        releves = []
        t0 = time.time()
        # on met une premiere salve en file, PUIS on demarre : le controleur
        # ne doit pas manquer de points des le debut
        premiere = cmds[:20]
        for ident, params in premiere:
            li.echanger(ident, params, ecriture=True, en_file=True)
        li.demarrer_file()
        dernier = verser(li, cmds[20:], releves, t0)
        attendre(li, dernier, releves, t0)
        analyse(releves, mode, duree_voulue)
        return 0
    except KeyboardInterrupt:
        li.arret_urgence()
        print("\nARRET D'URGENCE")
        return 1
    finally:
        pont.deconnecter()


if __name__ == "__main__":
    sys.exit(main())
