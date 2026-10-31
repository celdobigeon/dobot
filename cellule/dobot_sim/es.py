"""
Entrees / sorties tout-ou-rien du Dobot Magician.

Le controleur expose 20 ports multiplexes (EIO1 a EIO20), configurables en
sortie, en entree, en PWM ou en entree analogique. Le simulateur ne
modelise que le tout-ou-rien, qui couvre l'essentiel des travaux
pratiques : commander un verin ou un voyant, attendre l'acquittement d'un
capteur, se synchroniser avec un convoyeur.

Les SORTIES sont pilotees par le programme. Les ENTREES sont pilotees par
l'exterieur : dans le simulateur, c'est vous qui les basculez depuis le
panneau Entrees / Sorties, exactement comme un capteur le ferait.
"""

from __future__ import annotations

from typing import Callable, Dict, List

__all__ = ["EtatES", "NB_PORTS", "PORTS_RESERVES"]

NB_PORTS = 20

#: Affectations usuelles sur le Magician, a titre indicatif dans l'IHM.
#:
#: ATTENTION, verifie sur machine reelle : l'EFFECTEUR -- ventouse, pince,
#: laser -- n'est PAS cable sur ces sorties. Les actionner une a une, de
#: EIO1 a EIO20, n'a aucun effet sur lui. Il possede son propre circuit et
#: ses propres commandes dans le protocole (SetEndEffectorSuctionCup,
#: ...Gripper, ...Laser). Ces ports servent aux peripheriques de la
#: cellule : voyants, verins, convoyeur, capteurs.
PORTS_RESERVES: Dict[int, str] = {
    1: "EIO1  (port GP1)",
    2: "EIO2  (port GP1)",
    3: "EIO3  (port GP2)",
    4: "EIO4  (port GP2)",
    5: "EIO5  (port GP3)",
    6: "EIO6  (port GP3)",
    11: "EIO11 (port SW1)",
    12: "EIO12 (port SW2)",
    15: "EIO15 (port GP4)",
    16: "EIO16 (port GP4)",
    17: "EIO17 (port GP5)",
    18: "EIO18 (port GP5)",
}


class EtatES:
    """
    Etat courant des ports, partage entre le programme et l'IHM.

    Les numeros de port vont de 1 a NB_PORTS, comme sur le materiel.
    """

    def __init__(self, nb_ports: int = NB_PORTS):
        self.nb_ports = int(nb_ports)
        self.sorties: List[bool] = [False] * self.nb_ports
        self.entrees: List[bool] = [False] * self.nb_ports
        self._abonnes: List[Callable[[], None]] = []

    # -- notification ------------------------------------------------------

    def abonner(self, rappel: Callable[[], None]) -> None:
        """Enregistre une fonction appelee a chaque changement d'etat."""
        self._abonnes.append(rappel)

    def _notifier(self) -> None:
        for r in list(self._abonnes):
            try:
                r()
            except Exception:
                pass

    # -- acces -------------------------------------------------------------

    @staticmethod
    def _verifier(port: int, nb: int) -> int:
        p = int(port)
        if not 1 <= p <= nb:
            raise ValueError("port {} hors de 1..{}".format(port, nb))
        return p - 1

    def sortie(self, port: int) -> bool:
        return self.sorties[self._verifier(port, self.nb_ports)]

    def definir_sortie(self, port: int, etat: bool) -> None:
        i = self._verifier(port, self.nb_ports)
        if self.sorties[i] != bool(etat):
            self.sorties[i] = bool(etat)
            self._notifier()

    def entree(self, port: int) -> bool:
        return self.entrees[self._verifier(port, self.nb_ports)]

    def definir_entree(self, port: int, etat: bool) -> None:
        i = self._verifier(port, self.nb_ports)
        if self.entrees[i] != bool(etat):
            self.entrees[i] = bool(etat)
            self._notifier()

    def basculer_entree(self, port: int) -> bool:
        i = self._verifier(port, self.nb_ports)
        self.entrees[i] = not self.entrees[i]
        self._notifier()
        return self.entrees[i]

    # -- utilitaires -------------------------------------------------------

    def reinitialiser(self, entrees: bool = True, sorties: bool = True) -> None:
        if sorties:
            self.sorties = [False] * self.nb_ports
        if entrees:
            self.entrees = [False] * self.nb_ports
        self._notifier()

    def sorties_actives(self) -> List[int]:
        return [i + 1 for i, v in enumerate(self.sorties) if v]

    def entrees_actives(self) -> List[int]:
        return [i + 1 for i, v in enumerate(self.entrees) if v]

    def resume(self) -> str:
        def fmt(nom, vals):
            actifs = [str(i + 1) for i, v in enumerate(vals) if v]
            return "{} : {}".format(nom, ", ".join(actifs) if actifs else "aucune")
        return fmt("sorties actives", self.sorties) + "   |   " \
            + fmt("entrees actives", self.entrees)

    def __repr__(self) -> str:
        return "EtatES({})".format(self.resume())
