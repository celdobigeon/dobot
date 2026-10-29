#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
De combien le tapis depasse apres l'ordre d'arret, et selon quelle loi.

    python diag_depassement.py COM6

LE BRAS NE BOUGE PAS. Le tapis, si. Il vous faut une regle et un repere
colle sur la bande.

Pourquoi cette mesure n'est pas celle qu'on croit
--------------------------------------------------
Chronometrer la distance TOTALE entre « marche » et « arret » ne montre
rien : la rampe de demarrage perd exactement ce que la rampe d'arret
gagne, et le total tombe juste. C'est ce qui s'est passe ici -- 180 mm
mesures pour 180 mm attendus, alors que la piece depassait le capteur de
145 mm. Il faut donc mesurer le depassement SEUL : on lance, on laisse la
vitesse s'etablir, on repere la bande EN MARCHE, et on arrete.

Ce qu'on en tire, et pourquoi deux vitesses
--------------------------------------------
Deux mecanismes donnent le meme depassement a une vitesse donnee et
divergent partout ailleurs :

    RETARD FIXE   la commande met un temps t a agir ; depassement = v.t
                  -> deux fois moins vite, deux fois moins loin
    RAMPE         le moteur decelere a taux constant ; d = v^2/(2a)
                  -> deux fois moins vite, QUATRE fois moins loin

Entre 60 et 20 mm/s, les deux predictions different d'un facteur trois.
Mesurer a deux vitesses suffit donc a trancher -- et la loi decide si
ralentir le tapis resout le probleme ou le deplace.
"""

import sys
import time

from dobot_sim.convoyeur import IMPULSIONS_PAR_MM
from dobot_sim.liaison import ErreurLiaison, LiaisonDobot, ports_disponibles

VITESSES = (60.0, 30.0, 20.0)
ETABLISSEMENT = 4.0            # s de marche avant de reperer la bande


def demander(message: str) -> str:
    try:
        return input(message).strip()
    except EOFError:
        return ""


def nombre(message: str):
    while True:
        rep = demander(message)
        if not rep:
            return None
        try:
            return float(rep.replace(",", "."))
        except ValueError:
            print("    (un nombre, ou Entree pour passer)")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else ""
    if not port:
        dispo = [p for p, _ in ports_disponibles() if p != "COM3"]
        if not dispo:
            sys.exit("aucun port serie")
        port = dispo[0]

    li = LiaisonDobot(port).ouvrir()
    mesures = []
    try:
        print("port {} - firmware {}".format(port, li.version()))
        li.arreter_file()
        li.vider_file()
        li.demarrer_file()
        print(__doc__.split("Pourquoi")[0].strip())

        for vitesse in VITESSES:
            print("\n" + "=" * 64)
            print("VITESSE {:.0f} mm/s".format(vitesse))
            print("=" * 64)
            print("  Le tapis va tourner {:.0f} s pour etablir sa vitesse,"
                  .format(ETABLISSEMENT))
            print("  puis je vous dirai de REPERER la bande en marche --")
            print("  un trait au crayon en face d'un point fixe du bati.")
            print("  J'arrete 2 s plus tard. Vous mesurez ensuite la")
            print("  distance entre votre trait et ce meme point fixe.")
            demander("  Entree pour lancer... ")

            li.moteur_etendu(0, True, vitesse * IMPULSIONS_PAR_MM,
                             en_file=False)
            for reste in range(int(ETABLISSEMENT), 0, -1):
                print("    etablissement... {}".format(reste))
                time.sleep(1.0)
            print("\n    >>> REPEREZ LA BANDE MAINTENANT <<<")
            time.sleep(2.0)
            li.moteur_etendu(0, False, 0.0, en_file=False)
            print("    arret commande.")

            d = nombre("    Distance parcourue APRES l'arret, en mm ? ")
            if d is not None:
                mesures.append((vitesse, d))
                print("    -> {:.0f} mm a {:.0f} mm/s".format(d, vitesse))

        # ------------------------------------------------------- bilan
        print("\n" + "=" * 64)
        print("BILAN")
        print("=" * 64)
        if len(mesures) < 2:
            print("  Moins de deux mesures : impossible de trancher la loi.")
            if mesures:
                v, d = mesures[0]
                print("  Retenez au moins : depassement={:.0f}, "
                      "vitesse_depassement={:.0f}".format(d, v))
            return

        (v0, d0) = mesures[0]
        print("  mesures : {}".format(
            ", ".join("{:.0f} mm a {:.0f} mm/s".format(d, v)
                      for v, d in mesures)))
        err_retard = err_rampe = 0.0
        for v, d in mesures[1:]:
            err_retard += abs(d - d0 * v / v0)
            err_rampe += abs(d - d0 * (v / v0) ** 2)
        print("  ecart au modele RETARD FIXE : {:.0f} mm".format(err_retard))
        print("  ecart au modele RAMPE       : {:.0f} mm".format(err_rampe))

        loi = "retard" if err_retard <= err_rampe else "rampe"
        print("\n  -> loi retenue : {}".format(
            "RETARD FIXE (depassement proportionnel a la vitesse)"
            if loi == "retard"
            else "RAMPE (depassement en vitesse au carre)"))
        if loi == "retard":
            print("     retard equivalent : {:.2f} s".format(d0 / v0))
            print("     Ralentir aide, mais lineairement seulement. Et un")
            print("     retard de plus d'une seconde n'est pas une rampe")
            print("     mecanique : cherchez du cote du firmware.")
        else:
            print("     deceleration : {:.0f} mm/s2".format(v0 ** 2 / (2 * d0)))
            print("     Ralentir aide BEAUCOUP : moitie moins vite, quatre")
            print("     fois moins loin.")

        print("\n  A mettre dans votre Convoyeur :")
        print("      depassement={:.0f},".format(d0))
        print("      vitesse_depassement={:.0f},".format(v0))
        print("      loi_depassement=\"{}\",".format(loi))
        print("\n  Le modele saura alors ou la piece s'immobilise vraiment :")
        print("  TAPIS.abscisse_arret(vitesse) -- le capteur n'est que")
        print("  l'endroit ou on l'a vue passer.")
    finally:
        try:
            li.moteur_etendu(0, False, 0.0, en_file=False)
        except ErreurLiaison:
            pass
        li.arreter_file()
        li.vider_file()
        li.fermer()
        print("\n  tapis arrete, liaison fermee")


if __name__ == "__main__":
    main()
