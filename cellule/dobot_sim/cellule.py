# -*- coding: utf-8 -*-
"""
Plusieurs robots dans la meme cellule : chacun son cycle, ou ensemble.

Deux facons de travailler a plusieurs, et il faut savoir laquelle on
veut.

**Chacun son cycle.** Les robots sont independants : on leur donne a
chacun son programme, ils partent, on attend qu'ils aient tous fini.
C'est le cas courant -- deux postes qui travaillent en parallele.

    cellule = Cellule()
    alpha = cellule.ajouter("alpha", port="COM4", outil="ventouse")
    beta = cellule.ajouter("beta", port="COM6", outil="pince")

    alpha.movj(A1); alpha.movl(A2)
    beta.movj(B1)

    cellule.connecter()
    cellule.lancer()              # tout le monde part
    cellule.attendre_tous()

**Rendez-vous.** Les bras cooperent sur la meme piece : on decoupe le
travail en ETAPES, et personne ne passe a la suivante avant que tous
aient fini la leur.

    alpha.movj(AU_DESSUS); beta.movj(EN_FACE)
    cellule.etape()               # partent ensemble, on les attend
    alpha.ventouse(True)
    cellule.etape()

Ce que la synchronisation vaut, et ce qu'elle ne vaut pas
----------------------------------------------------------
La file du controleur n'a AUCUNE synchronisation entre machines : il
n'existe pas de commande « pars quand l'autre part ». Tout se fait donc
cote PC, et cela borne la precision :

- le depart est decale du temps qu'il faut pour envoyer une trame a
  chaque robot, soit une dizaine de millisecondes par machine. A
  100 mm/s, cela fait un millimetre. `Cellule.decalage_demarrage` donne
  la valeur MESUREE au dernier depart, plutot qu'une promesse ;
- a l'interieur d'une etape, les bras derivent librement : rien ne les
  recale. Le rendez-vous est a la FIN de l'etape, pas en continu.

Decoupez donc en etapes la ou la coordination compte, et ne comptez pas
sur une trajectoire synchronisee a l'interieur d'un mouvement.

La securite
-----------
Un arret d'urgence vaut pour TOUTE la cellule : arreter un seul bras
pendant que l'autre continue vers lui ne protege de rien. De meme, un
robot qui tombe en defaut arrete les autres -- c'est le comportement par
defaut, et `solidaire=False` le desactive pour les essais.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .pilote import Pilote
from .reel import BLOQUANTS_TELEVERSEMENT, ErreurPont, PontReel

__all__ = ["Cellule", "ErreurCellule", "Poste"]


class ErreurCellule(RuntimeError):
    """Un robot de la cellule a refuse, ou s'est arrete en route."""


@dataclass
class Poste:
    """Un robot de la cellule : son nom, sa liaison, son programme."""

    nom: str
    pilote: Pilote
    pont: PontReel
    port: str = ""
    index_final: int = 0

    @property
    def connecte(self) -> bool:
        return self.pont.connecte

    def en_attente(self) -> bool:
        """Ce poste a-t-il quelque chose a jouer ?"""
        return bool(self.pilote.programme.instructions)


class Cellule:
    """
    Plusieurs robots, une seule poignee.

    `journal` recoit les messages de la cellule ET de chaque poste,
    prefixes par le nom du robot : sans cela on ne sait plus qui parle.
    """

    #: periode de scrutation pendant l'attente, en secondes
    PERIODE = 0.05

    def __init__(self, journal: Optional[Callable[[str], None]] = None,
                 solidaire: bool = True):
        self.journal = journal or (lambda _m: None)
        #: un defaut sur un poste arrete-t-il les autres ?
        self.solidaire = bool(solidaire)
        self.postes: Dict[str, Poste] = {}
        #: ecart MESURE entre le premier et le dernier depart, en secondes
        self.decalage_demarrage = 0.0

    # -- composition --------------------------------------------------------

    def ajouter(self, nom: str, port: str = "", robot=None,
                **kw) -> Pilote:
        """
        Ajoute un robot a la cellule et rend son Pilote.

        Les arguments supplementaires sont ceux de `Pilote` : outil,
        vitesse, lissage... Le port peut attendre `connecter()`.
        """
        if nom in self.postes:
            raise ValueError("deux robots nommes '{}'".format(nom))

        def consigner(message, _nom=nom):
            self.journal("[{}] {}".format(_nom, message))

        pilote = Pilote(robot=robot, nom=kw.pop("nom", nom), **kw)
        pont = PontReel(pilote.robot, journal=consigner)
        self.postes[nom] = Poste(nom=nom, pilote=pilote, pont=pont, port=port)
        return pilote

    def __getitem__(self, nom: str) -> Pilote:
        return self.postes[nom].pilote

    def __len__(self) -> int:
        return len(self.postes)

    @property
    def noms(self) -> List[str]:
        return list(self.postes)

    # -- liaisons -----------------------------------------------------------

    def connecter(self, ports: Optional[Dict[str, str]] = None) -> None:
        """
        Ouvre toutes les liaisons. Rien ne bouge.

        Si une seule echoue, celles deja ouvertes sont refermees : une
        cellule a moitie connectee est un piege, on croirait pouvoir
        lancer le cycle.
        """
        ports = ports or {}
        ouverts: List[Poste] = []
        try:
            for poste in self.postes.values():
                port = ports.get(poste.nom) or poste.port
                if not port:
                    raise ErreurCellule(
                        "aucun port pour le robot '{}'".format(poste.nom))
                poste.port = port
                poste.pont.connecter(port)
                ouverts.append(poste)
                self.journal("[{}] connecte sur {}".format(poste.nom, port))
        except Exception as exc:
            for poste in ouverts:
                try:
                    poste.pont.deconnecter()
                except Exception:
                    pass
            raise ErreurCellule(
                "cellule non connectee : {}".format(exc)) from exc

    def deconnecter(self) -> None:
        for poste in self.postes.values():
            try:
                poste.pont.deconnecter()
            except Exception:
                pass

    def definir_ratio(self, vitesse_pct: float) -> None:
        """Bride toute la cellule a la meme fraction de la vitesse."""
        for poste in self.postes.values():
            if poste.connecte:
                poste.pont.definir_ratio(vitesse_pct)

    # -- securite -----------------------------------------------------------

    def arret_urgence(self) -> None:
        """
        Coupe TOUS les robots, tout de suite.

        On n'arrete pas un bras en laissant l'autre avancer vers lui :
        chaque poste est coupe, meme si l'un d'eux refuse -- l'echec d'un
        arret ne doit pas empecher les suivants.
        """
        echecs = []
        for poste in self.postes.values():
            try:
                if poste.connecte:
                    poste.pont.arret_urgence()
            except Exception as exc:                     # pragma: no cover
                echecs.append("{} : {}".format(poste.nom, exc))
        self.journal("ARRET D'URGENCE de la cellule")
        if echecs:                                       # pragma: no cover
            raise ErreurCellule("arret incomplet -- " + " ; ".join(echecs))

    def defauts(self) -> Dict[str, List[str]]:
        """Les alarmes de chaque poste, par nom. Vide si tout va bien."""
        trouves = {}
        for poste in self.postes.values():
            if not poste.connecte:
                continue
            try:
                alarmes = poste.pont.etat().alarmes
            except Exception as exc:
                alarmes = ["lecture impossible : {}".format(exc)]
            if alarmes:
                trouves[poste.nom] = list(alarmes)
        return trouves

    # -- execution ----------------------------------------------------------

    def controler(self) -> Dict[str, List[str]]:
        """
        Verifie les programmes en attente, sans rien envoyer.

        Rend, par robot, les defauts qui EMPECHENT le cycle -- butees,
        collisions. Les simples reserves ne sont pas remontees ici.
        """
        defauts = {}
        for poste in self.postes.values():
            if not poste.en_attente():
                continue
            bloquants = [a for a in poste.pilote.controler()
                         if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
            if bloquants:
                defauts[poste.nom] = bloquants
        return defauts

    def lancer(self, vitesse: Optional[float] = None,
               forcer: bool = False) -> float:
        """
        Televerse tous les programmes, puis demarre tout le monde.

        Les televersements se font D'ABORD, les demarrages ENSUITE : un
        televersement dure, un demarrage est une seule trame. Les faire
        dans cet ordre ramene l'ecart de depart de quelques secondes a
        quelques millisecondes.

        Rend le decalage MESURE entre le premier et le dernier depart.
        """
        actifs = [p for p in self.postes.values() if p.en_attente()]
        if not actifs:
            return 0.0
        for poste in actifs:
            if not poste.connecte:
                raise ErreurCellule(
                    "le robot '{}' n'est pas connecte".format(poste.nom))

        if not forcer:
            defauts = self.controler()
            if defauts:
                detail = " ; ".join(
                    "{} : {}".format(nom, liste[0])
                    for nom, liste in defauts.items())
                raise ErreurCellule(
                    "cycle refuse, rien n'a ete envoye -- " + detail)

        if vitesse is not None:
            self.definir_ratio(vitesse)

        for poste in actifs:
            poste.index_final = poste.pont.televerser(
                poste.pilote.programme, poste.pilote.q_depart,
                outil=poste.pilote.programme.outil, forcer=forcer)

        debut = time.time()
        for poste in actifs:
            poste.pont.demarrer()
        self.decalage_demarrage = time.time() - debut
        self.journal("cellule lancee : {} robot(s), decalage de depart "
                     "{:.0f} ms".format(len(actifs),
                                        1000 * self.decalage_demarrage))
        return self.decalage_demarrage

    def termine(self) -> bool:
        """Tous les postes ont-ils fini ce qu'on leur a donne ?"""
        return all(p.pont.termine() for p in self.postes.values()
                   if p.connecte)

    def attendre_tous(self, limite: float = 600.0) -> None:
        """
        Bloque jusqu'a ce que tous les robots aient fini.

        Un defaut sur l'un arrete les autres si la cellule est solidaire
        -- la laisser tourner pendant qu'un bras est fige, c'est
        justement le moment ou les deux se rencontrent.
        """
        debut = time.time()
        while time.time() - debut < limite:
            defauts = self.defauts()
            if defauts:
                if self.solidaire:
                    self.arret_urgence()
                detail = " ; ".join("{} : {}".format(nom, liste[0])
                                    for nom, liste in defauts.items())
                raise ErreurCellule("robot en defaut -- " + detail)
            if self.termine():
                return
            time.sleep(self.PERIODE)
        if self.solidaire:
            self.arret_urgence()
        raise ErreurCellule(
            "cycle non termine apres {:.0f} s : cellule arretee".format(limite))

    # -- rendez-vous --------------------------------------------------------

    def etape(self, vitesse: Optional[float] = None,
              limite: float = 600.0) -> float:
        """
        Une ETAPE : chacun joue ce qu'il a en attente, et on se retrouve.

            alpha.movj(AU_DESSUS); beta.movj(EN_FACE)
            cellule.etape()               # partent ensemble, on attend
            alpha.ventouse(True)
            cellule.etape()

        Apres l'etape, chaque programme est vide et REPART DE LA POSE
        ATTEINTE : la suite s'ecrit sans se soucier d'ou le bras en est.

        Rend le decalage de depart mesure, pour qu'on sache ce que vaut
        le « ensemble ».
        """
        decalage = self.lancer(vitesse=vitesse)
        self.attendre_tous(limite=limite)
        for poste in self.postes.values():
            if poste.en_attente():
                poste.pilote.vider_programme()
        return decalage

    # -- confort ------------------------------------------------------------

    def __enter__(self) -> "Cellule":
        return self

    def __exit__(self, *_exc) -> None:
        """A la sortie, on coupe puis on ferme -- meme sur erreur."""
        try:
            self.arret_urgence()
        except Exception:                                # pragma: no cover
            pass
        self.deconnecter()

    def resume(self) -> str:
        lignes = ["cellule : {} robot(s)".format(len(self.postes))]
        for poste in self.postes.values():
            lignes.append(
                "  {:<10} {:<8} {:>4} instruction(s){}"
                .format(poste.nom, poste.port or "(sans port)",
                        len(poste.pilote.programme.instructions),
                        "" if poste.connecte else "   NON CONNECTE"))
        if self.decalage_demarrage:
            lignes.append("  dernier depart : {:.0f} ms d'ecart"
                          .format(1000 * self.decalage_demarrage))
        return "\n".join(lignes)
