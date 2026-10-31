# -*- coding: utf-8 -*-
"""
Rend `dobot_sim` importable depuis l'atelier.

    import cellule_sur_le_chemin        # noqa: F401

A mettre en TETE de tout script de ce dossier, avant d'importer
`dobot_sim`. Une ligne, et l'explication vit ici plutot que recopiee
dix-huit fois.

Pourquoi il faut ca
-------------------
Le depot est range en deux dossiers depuis le 31/10/2026 :

    cellule/   ce qu'on donne aux etudiants -- le paquet, le
               simulateur, les exemples, les procedures
    atelier/   les diagnostics, les essais, les outils de mise au
               point. Ce dossier-ci.

Python ajoute tout seul le dossier du script lance dans son chemin de
recherche. Un diagnostic lance d'ici voit donc `atelier/`, ou le paquet
n'est pas. On ajoute `cellule/`, qui est son voisin.

Pourquoi un module plutot qu'une ligne copiee
---------------------------------------------
Parce qu'une ligne copiee dix-huit fois est dix-huit endroits a
corriger le jour ou l'arborescence bouge encore. Ici, il n'y en a
qu'un.
"""

import os
import sys

#: le dossier donne aux etudiants, voisin de celui-ci
CELLULE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cellule")

if CELLULE not in sys.path:
    sys.path.insert(0, CELLULE)
