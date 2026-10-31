#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alertes du terminal, communes aux deux facons de piloter le robot.

`Pilote.executer_sur_robot()` (un programme joue d'un bloc) et `Commande`
(chaque appel execute tout de suite) parlent a l'operateur de la meme
facon : en JAUNE ce qui merite d'etre vu sans arreter quoi que ce soit, en
ROUGE encadre ce qui arrete le script. Les deux puisent ici, pour qu'un
etudiant passant de l'un a l'autre retrouve les memes signaux.
"""

from __future__ import annotations

import sys

__all__ = ["ErreurExecution"]


class ErreurExecution(RuntimeError):
    """
    Le mouvement demande n'a pas pu etre execute jusqu'au bout.

    Levee par `Pilote.executer_sur_robot()` et par chaque commande de
    `Commande`, apres une alerte rouge dans le terminal. Elle arrete le
    script : la suite partirait d'une posture qui n'est pas celle prevue.
    Pour gerer l'echec soi-meme : `try: ... except ErreurExecution: ...`.
    """


def _couleur(code: str, texte: str) -> str:
    """Couleur ANSI si la sortie est un terminal (VS Code l'est)."""
    return ("\033[{}m{}\033[0m".format(code, texte)
            if getattr(sys.stdout, "isatty", lambda: False)() else texte)


def _signaler(titre: str, lignes=()) -> None:
    """Avertissement : on continue, mais il faut l'avoir vu (jaune)."""
    print(_couleur("1;33", "/!\\ " + titre))
    for ligne in lignes:
        print(_couleur("33", "      " + ligne))


def _alerter(titre: str, lignes=()) -> None:
    """Alerte encadree, en rouge : impossible de la manquer."""
    barre = "!" * 72
    print(_couleur("1;31", barre))
    print(_couleur("1;31", "!!! " + titre))
    for ligne in lignes:
        print(_couleur("31", "!!!     " + ligne))
    print(_couleur("1;31", barre))


def _echouer(titre: str, lignes=()):
    """Alerte, puis ErreurExecution : le script s'arrete la."""
    _alerter(titre, lignes)
    raise ErreurExecution(titre)


def _signaler_urgence() -> None:
    _alerter("ARRET D'URGENCE : mouvement coupe, file du controleur videe",
             ["le robot est arrete la ou il etait ; verifiez sa posture "
              "avant de relancer"])
