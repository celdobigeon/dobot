#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Un fichier CAO (STEP) -> un maillage OBJ que le simulateur sait afficher.

    python convertir_cao.py "Modele3D/Conveyor belt.stp" --tapis
    python convertir_cao.py Modele3D/Camera.step --finesse 15
    python convertir_cao.py Modele3D/ventouse.stp --boite

Pourquoi convertir
------------------
La CAO decrit des SURFACES exactes -- des plans, des cylindres, des
congés. Une carte graphique ne sait afficher que des triangles. Il faut
donc mailler, et c'est un choix : trop fin, la scene rame et le fichier
pese des dizaines de megaoctets ; trop grossier, les arrondis
deviennent des facettes. Pour du DECOR -- un tapis, un bati -- une
dizaine de milliers de triangles suffit largement.

Le recalage, qui compte autant que le maillage
-----------------------------------------------
Un modele CAO a l'origine que son dessinateur lui a donnee : souvent le
centre de la piece, parfois un coin, rarement ce dont on a besoin. Le
simulateur, lui, attend des conventions precises. `--tapis` recale sur
celle du `Convoyeur` :

    origine a l'ENTREE de la bande, x dans le sens de marche,
    y en travers, z vers le haut, zero a la SURFACE de la bande

Une fois recale, le maillage s'affiche par la seule matrice du tapis :
deplacer le convoyeur dans la cellule ne demande plus de retoucher quoi
que ce soit.

`--boite` n'ecrit rien : il se contente d'annoncer les cotes du fichier.
A faire EN PREMIER sur un modele inconnu, pour verifier l'unite -- un
STEP n'en porte pas, et un facteur mille passe inapercu jusqu'a ce que
le tapis fasse six cents metres de long.
"""

import argparse
import os
import sys

import numpy as np


def ouvrir(chemin, finesse, grossier):
    """Maille le fichier CAO et rend (sommets, faces)."""
    try:
        import gmsh
    except ImportError:
        sys.exit("gmsh est absent : python -m pip install gmsh")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(chemin)
        boite = gmsh.model.getBoundingBox(-1, -1)
        if grossier:
            return None, None, boite

        gmsh.option.setNumber("Mesh.MeshSizeMax", float(finesse))
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(finesse) / 4.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)

        etiquettes, coords, _ = gmsh.model.mesh.getNodes()
        sommets = np.asarray(coords, dtype=float).reshape(-1, 3)
        rang = {int(e): i for i, e in enumerate(etiquettes)}
        morceaux = []
        for dim, tag in gmsh.model.getEntities(2):
            types, _, noeuds = gmsh.model.mesh.getElements(dim, tag)
            for t, n in zip(types, noeuds):
                if t == 2:                     # triangle a trois noeuds
                    morceaux.append(np.asarray(n, dtype=np.int64).reshape(-1, 3))
        if not morceaux:
            sys.exit("aucun triangle produit : le fichier est-il vide ?")
        faces = np.vectorize(rang.get)(np.vstack(morceaux))
        return sommets, faces, boite
    finally:
        gmsh.finalize()


def recaler_tapis(sommets, inverser=False):
    """
    Passe du repere du dessinateur a celui du `Convoyeur`.

    La bande est l'arete la plus longue : c'est elle qui donne le sens de
    marche, sans avoir a le demander. L'origine va a l'une de ses
    extremites -- `--inverser` choisit l'autre, quand l'entree physique
    se trouve du mauvais cote.
    """
    mini, maxi = sommets.min(axis=0), sommets.max(axis=0)
    etendue = maxi - mini
    axe_long = int(np.argmax(etendue))
    axe_haut = 2                                # z du dessinateur, vers le haut
    if axe_long == axe_haut:
        sys.exit("la plus grande dimension est verticale : modele inattendu")
    axe_large = ({0, 1, 2} - {axe_long, axe_haut}).pop()

    x = np.zeros(3); x[axe_long] = -1.0 if inverser else 1.0
    z = np.zeros(3); z[axe_haut] = 1.0
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])              # colonnes = axes du tapis

    origine = np.zeros(3)
    origine[axe_long] = maxi[axe_long] if inverser else mini[axe_long]
    origine[axe_large] = (mini[axe_large] + maxi[axe_large]) / 2.0
    origine[axe_haut] = maxi[axe_haut]          # zero a la SURFACE de la bande

    recales = (sommets - origine) @ R
    print("  recale sur le repere du Convoyeur :")
    print("    sens de marche = axe {} du modele{}"
          .format("xyz"[axe_long], " (inverse)" if inverser else ""))
    print("    longueur {:.0f} mm, largeur {:.0f} mm, hauteur {:.0f} mm"
          .format(etendue[axe_long], etendue[axe_large], etendue[axe_haut]))
    return recales


def ecrire_obj(chemin, sommets, faces, source):
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("# maillage produit par convertir_cao.py\n")
        f.write("# source : {}\n".format(os.path.basename(source)))
        f.write("# {} sommets, {} triangles\n".format(len(sommets), len(faces)))
        for s in sommets:
            f.write("v {:.3f} {:.3f} {:.3f}\n".format(*s))
        for t in faces:
            f.write("f {} {} {}\n".format(t[0] + 1, t[1] + 1, t[2] + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("fichier", help="le STEP a convertir")
    ap.add_argument("--sortie", help="l'OBJ a ecrire (sinon, a cote de la source)")
    ap.add_argument("--finesse", type=float, default=30.0,
                    help="taille de maille en mm (defaut 30 ; 10 pour du fin)")
    ap.add_argument("--tapis", action="store_true",
                    help="recaler sur le repere du Convoyeur")
    ap.add_argument("--inverser", action="store_true",
                    help="avec --tapis : l'entree est a l'autre bout")
    ap.add_argument("--boite", action="store_true",
                    help="annoncer les cotes sans rien ecrire")
    a = ap.parse_args()

    if not os.path.exists(a.fichier):
        sys.exit("fichier introuvable : {}".format(a.fichier))
    print("lecture de {}".format(a.fichier))
    sommets, faces, boite = ouvrir(a.fichier, a.finesse, a.boite)

    xmin, ymin, zmin, xmax, ymax, zmax = boite
    print("  boite englobante : {:.0f} x {:.0f} x {:.0f} (unites du fichier)"
          .format(xmax - xmin, ymax - ymin, zmax - zmin))
    if a.boite:
        print("  (rien n'a ete ecrit : enlevez --boite pour mailler)")
        return

    print("  {} sommets, {} triangles".format(len(sommets), len(faces)))
    if a.tapis:
        sommets = recaler_tapis(sommets, a.inverser)

    sortie = a.sortie or os.path.splitext(a.fichier)[0] + ".obj"
    ecrire_obj(sortie, sommets, faces, a.fichier)
    print("  ecrit : {} ({:.1f} Mo)"
          .format(sortie, os.path.getsize(sortie) / 1e6))


if __name__ == "__main__":
    main()
