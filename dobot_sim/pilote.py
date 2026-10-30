"""
Pilotage du simulateur par script Python.

C'est l'interface a utiliser depuis VS Code : on ecrit un fichier, on
l'execute (F5 ou Ctrl+F5), la fenetre de simulation s'ouvre et le robot
joue le mouvement.

    from dobot_sim import Pilote, cible, repere_travail
    from dobot_sim import homogeneous as hg

    MAGASIN = repere_travail(215, -95, 25)        # repere pose sur la table
    PRISE = cible(repere=MAGASIN)                 # matrice homogene 4x4
    DEPOSE = cible(215, 95, 25, rz=45)
    HAUT = hg.translation(0, 0, -85)              # degagement selon l'axe outil

    robot = Pilote(vitesse=120, acceleration=500,           # MOVL, MOVC : mm/s
                   vitesse_articulaire=60)                  # MOVJ : deg/s
    robot.repli()
    robot.movj(PRISE @ HAUT)
    robot.movl(PRISE, vitesse=60)         # descente lente
    robot.ventouse(True)
    robot.movl(PRISE @ HAUT, lissage=60)  # raccord arrondi (simulateur seul)
    robot.movc(via=cible(265, 0, 130), T=DEPOSE @ HAUT)
    robot.movl(DEPOSE)
    robot.ventouse(False)
    robot.repli()
    robot.simuler()

Une cible est une SITUATION, donc une matrice homogene. Les coordonnees
directes restent acceptees, et les composantes omises sont conservees de
l'instruction precedente : `robot.movl(z=25)` descend a la verticale.

Sans interface graphique, les memes objets servent a l'analyse :

    e = robot.trajectoire().echantillonner(0.01)
    print(e["V"].max(), "mm/s")
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Les alertes du terminal et ErreurExecution vivent dans un module commun :
# Commande (dobot_sim.commande) parle a l'operateur exactement de la meme
# facon. Les noms restent importables d'ici, comme avant.
from .alertes import (ErreurExecution, _alerter, _couleur,  # noqa: F401
                      _echouer, _signaler, _signaler_urgence)
from .outils import outil as catalogue_outil
from .program import Instruction, Programme
from .robot import DobotMagician, decomposer_cible
from .trajectory import PROFILS, Trajectoire

__all__ = ["Pilote", "ErreurExecution"]


#: liaison directe gardee ouverte entre deux executer_sur_robot()
_PONT_DIRECT = {}


def _decrire_tapis(tapis) -> dict:
    """
    Ce qu'un tapis doit dire de lui-meme pour traverser la passerelle.

    Quelques nombres, pas un objet : une passerelle transporte des
    donnees, et le simulateur n'a pas a reconstruire un `Convoyeur` pour
    commander un moteur.
    """
    from .convoyeur import IMPULSIONS_PAR_MM, pas_signe_de
    pas = pas_signe_de(tapis, IMPULSIONS_PAR_MM)
    return {"index": int(getattr(tapis, "index_moteur", 0) or 0),
            "pas": pas,
            "capteur_port": getattr(tapis, "capteur_port", None),
            "capteur_eio": getattr(tapis, "capteur_eio", None)}


def _liaison_directe(robot, port: Optional[str]):
    """
    La liaison du script avec le robot, ouverte au premier besoin et
    REUTILISEE ensuite : rouvrir le port a chaque partie couterait pres
    d'une seconde, et ferait perdre le fil d'un cycle. Fermee a la sortie
    du script.
    """
    import atexit

    from .liaison import ports_disponibles
    from .reel import PontReel

    pont = _PONT_DIRECT.get("pont")
    if pont is not None and pont.connecte and (port is None
                                               or pont.liaison.port == port):
        pont.robot = robot
        return pont
    if pont is not None:
        # Changement de port, ou liaison tombee : fermer AVANT de remplacer.
        # Sans cela, un script qui passe d'une machine a l'autre laisserait
        # le premier port ouvert jusqu'a la fin du processus -- et l'atexit,
        # qui ne connait que le dernier pont, ne le libererait jamais.
        try:
            pont.deconnecter()
        except Exception:
            pass
        _PONT_DIRECT.pop("pont", None)
    if port is None:
        ports = ports_disponibles()
        if not ports:
            raise RuntimeError("aucun port serie : branchez et allumez le "
                               "Dobot")
        port = ports[0][0]
    pont = PontReel(robot, journal=lambda m: print("  [robot] " + m))
    try:
        pont.connecter(port)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    # une seule fois pour le processus : le dictionnaire se vide a chaque
    # changement de port, il ne peut donc pas servir de temoin
    if not _PONT_DIRECT.get("atexit"):
        atexit.register(_fermer_liaison_directe)
        _PONT_DIRECT["atexit"] = True
    _PONT_DIRECT["pont"] = pont
    return pont


def _fermer_liaison_directe() -> None:
    pont = _PONT_DIRECT.pop("pont", None)
    if pont is not None:
        try:
            pont.deconnecter()
        except Exception:
            pass


def _script_appelant() -> str:
    """
    Chemin absolu du script lance (F5 dans VS Code), ou "" en console.

    La fenetre le retient : son bouton Recharger pourra ainsi reexecuter
    le script modifie sans repasser par l'editeur.
    """
    import os
    import sys

    principal = sys.modules.get("__main__")
    chemin = getattr(principal, "__file__", "") or ""
    return os.path.abspath(chemin) if chemin.endswith(".py") else ""


class Pilote:
    """
    Constructeur de programme, pense pour l'ecriture au fil du script.

    Les reglages passes au constructeur servent de valeurs par defaut a
    toutes les instructions ; chaque commande peut les redefinir localement.
    """

    def __init__(self,
                 robot: Optional[DobotMagician] = None,
                 nom: str = "script",
                 vitesse: Optional[float] = None,
                 acceleration: Optional[float] = None,
                 vitesse_articulaire: Optional[float] = None,
                 acceleration_articulaire: Optional[float] = None,
                 lissage: float = 0.0,
                 profil: str = "polynome5",
                 depart: Optional[Sequence[float]] = None,
                 outil: str = "ventouse"):
        """
        `vitesse` et `acceleration` valent pour les mouvements CARTESIENS
        (MOVL, MOVC), en mm/s et mm/s2. Les mouvements ARTICULAIRES (MOVJ)
        s'expriment en deg/s et deg/s2, d'ou les deux parametres separes :
        140 mm/s et 140 deg/s n'ont rien de comparable.

        Les valeurs laissees a None reprennent celles de program.py.
        """
        self.robot = robot or DobotMagician()
        longueur = catalogue_outil(outil).longueur
        self.programme = Programme([], nom, outil, longueur_outil=longueur)
        self.robot.geo.lt = longueur
        self.q_depart = np.array(depart if depart is not None
                                 else self.robot.q_home, dtype=float)

        self.vitesse_defaut = {"mm": vitesse, "deg": vitesse_articulaire}
        self.acceleration_defaut = {"mm": acceleration,
                                    "deg": acceleration_articulaire}
        self.lissage_defaut = float(lissage)
        self.profil_defaut = profil

        # derniere cible cartesienne connue, pour les coordonnees omises
        self._pose = list(self.robot.pose(self.q_depart))
        #: le tapis de la cellule, s'il y en a un. Pose une fois, il sert
        #: de defaut a toutes les commandes de convoyeur du programme.
        self.tapis = None
        #: instant (perf_counter) ou le capteur a vu la piece,
        #: pose par attendre_piece_sur_robot
        self.instant_detection = None
        self.tours_scrutation = 0
        #: l'axe lineaire, s'il y en a un, et la position courante du
        #: chariot -- retenue pour que `controler` et la simulation
        #: sachent d'ou le bras travaille
        self.rail = None
        self.rail_courant = 0.0

    # -- reglages -----------------------------------------------------------

    def regler(self, vitesse: Optional[float] = None,
               acceleration: Optional[float] = None,
               vitesse_articulaire: Optional[float] = None,
               acceleration_articulaire: Optional[float] = None,
               lissage: Optional[float] = None,
               profil: Optional[str] = None) -> "Pilote":
        """Change les valeurs par defaut pour la suite du script."""
        if vitesse is not None:
            self.vitesse_defaut["mm"] = float(vitesse)
        if acceleration is not None:
            self.acceleration_defaut["mm"] = float(acceleration)
        if vitesse_articulaire is not None:
            self.vitesse_defaut["deg"] = float(vitesse_articulaire)
        if acceleration_articulaire is not None:
            self.acceleration_defaut["deg"] = float(acceleration_articulaire)
        if lissage is not None:
            self.lissage_defaut = float(lissage)
        if profil is not None:
            if profil not in PROFILS:
                raise ValueError("loi horaire inconnue : {} (choix : {})"
                                 .format(profil, ", ".join(sorted(PROFILS))))
            self.profil_defaut = profil
        return self

    def vider_programme(self) -> "Pilote":
        """
        Repart d'un programme VIDE, sans changer de robot ni de reglages.

        Garde l'outil et sa longueur, le repere objet, les vitesses et le
        lissage par defaut. Le nouveau programme part de la posture ou
        l'ancien a laisse le bras : le controle des butees, la simulation
        et une prochaine `executer_sur_robot()` partent du bon endroit.

            robot.executer_sur_robot(vitesse=50)
            robot.vider_programme()
            robot.movl(...)                  # la suite, depuis la fin
            robot.executer_sur_robot(vitesse=50)
        """
        p = self.programme
        if p.instructions:
            traj = p.construire(self.robot, self.q_depart)
            self.q_depart = np.asarray(traj.q(traj.duree), dtype=float)
            self._pose = list(self.robot.pose(self.q_depart))
        self.programme = Programme([], p.nom, p.outil,
                                   repere_objet=p.repere_objet,
                                   longueur_outil=p.longueur_outil)
        return self

    # -- construction interne ----------------------------------------------

    def _resoudre(self, T, x, y, z, r) -> list:
        """
        Determine la cible (x, y, z, r) d'une commande.

        `T` accepte une matrice homogene 4x4 (la forme recommandee), ou un
        triplet / quadruplet de coordonnees. Les mots-cles x, y, z, r
        surchargent ensuite composante par composante ; ceux qui restent a
        None conservent leur valeur precedente, ce qui permet d'ecrire
        `movl(z=25)` pour une descente verticale.
        """
        if T is not None:
            M = np.asarray(T, dtype=float)
            if M.shape == (4, 4):
                self._pose = list(decomposer_cible(M))
            else:
                v = [float(k) for k in np.ravel(M)]
                if len(v) == 3:
                    v.append(self._pose[3])
                if len(v) != 4:
                    raise ValueError(
                        "cible invalide : attendu une matrice homogene 4x4 "
                        "ou 3 a 4 coordonnees, recu {} valeurs".format(len(v)))
                self._pose = v
        for i, v in enumerate((x, y, z, r)):
            if v is not None:
                self._pose[i] = float(v)
        return list(self._pose)

    def _ajouter(self, type_: str, duree, vitesse, acceleration, lissage,
                 profil, commentaire, repere=None, **extra) -> Instruction:
        par_duree = duree is not None
        # un MOVJ interpole des angles, tout le reste des millimetres
        u = "deg" if type_ == "MOVJ" else "mm"
        ins = Instruction(
            type=type_,
            mode_temps="duree" if par_duree else "vitesse",
            duree=float(duree) if par_duree else 2.0,
            vitesse=float(vitesse if vitesse is not None
                          else (self.vitesse_defaut[u] or 0.0)),
            acceleration=float(acceleration if acceleration is not None
                               else (self.acceleration_defaut[u] or 0.0)),
            lissage=float(lissage if lissage is not None else self.lissage_defaut),
            profil=profil or self.profil_defaut,
            commentaire=commentaire or "",
            repere=repere or "base",
            **extra)
        self.programme.ajouter(ins)
        return ins

    # -- commandes de mouvement --------------------------------------------

    def movj(self, T=None, *, x=None, y=None, z=None, r=None, repere=None, q=None,
             duree=None, vitesse=None, acceleration=None, lissage=None,
             profil=None, rail=None, commentaire="") -> "Pilote":
        """
        Mouvement articulaire : les angles sont interpoles, le chemin du TCP
        n'est pas rectiligne. Vitesse et acceleration sont en deg/s et deg/s2.

            robot.movj(PRISE)                    cible = matrice homogene
            robot.movj(PRISE @ DEGAGEMENT)       composition de matrices
            robot.movj(x=220, y=-90, z=60)       coordonnees directes
            robot.movj(q=[0, 15, 30, 0])         configuration articulaire
        """
        chariot = self._chariot(rail)
        if q is not None:
            ins = self._ajouter("MOVJ", duree, vitesse, acceleration, lissage,
                                profil, commentaire, None, mode="articulaire",
                                q=[float(v) for v in q], **chariot)
            self._pose = list(self.robot.pose(ins.q))
        else:
            self._ajouter("MOVJ", duree, vitesse, acceleration, lissage,
                          profil, commentaire, repere, mode="cartesien",
                          pose=self._resoudre(T, x, y, z, r), **chariot)
        return self

    def _chariot(self, rail) -> dict:
        """
        Les champs de rail d'une instruction, ou rien.

        `rail=None` ne veut pas dire « chariot a zero » : il veut dire
        « ce mouvement ne parle pas du rail ». Sans cette distinction,
        tout programme ecrit sans rail ramenerait le chariot a l'origine
        des qu'un rail serait declare.
        """
        if rail is None:
            return {}
        axe = getattr(self, "rail", None)
        l = float(rail)
        if axe is not None:
            l = axe.borner(l)
            self.rail_courant = l
        return {"rail": l, "avec_rail": True}

    def movl(self, T=None, *, x=None, y=None, z=None, r=None, repere=None,
             duree=None, vitesse=None, acceleration=None, lissage=None,
             profil=None, rail=None, commentaire="") -> "Pilote":
        """
        Mouvement cartesien rectiligne : le TCP suit une droite.
        Vitesse et acceleration sont en mm/s et mm/s2.

            robot.movl(PRISE)                    cible = matrice homogene
            robot.movl(PRISE, z=40)              matrice, avec z force
            robot.movl(z=25)                     descente verticale
        """
        self._ajouter("MOVL", duree, vitesse, acceleration, lissage,
                      profil, commentaire, repere, mode="cartesien",
                      pose=self._resoudre(T, x, y, z, r),
                      **self._chariot(rail))
        return self

    def aller_rail(self, l: float, vitesse=None, lissage=None,
                   commentaire: str = "") -> "Pilote":
        """
        Deplace le CHARIOT seul, le bras gardant sa pose.

            robot.rail = AxeLineaire(course=1000)
            robot.aller_rail(400)              # le chariot y va
            robot.movl(PRISE)                  # le bras prend, de la

        Le rail n'est pas une commande separee sur le controleur : c'est
        la cinquieme coordonnee d'un mouvement point a point. On renvoie
        donc la pose courante du bras, assortie de la nouvelle position
        de chariot -- le bras ne bouge pas, le chariot si.
        """
        return self.movj(x=self._pose[0], y=self._pose[1], z=self._pose[2],
                         r=self._pose[3], rail=l, vitesse=vitesse,
                         lissage=lissage,
                         commentaire=commentaire or "chariot")

    def movc(self, via, T=None, *, x=None, y=None, z=None, r=None, repere=None,
             duree=None, vitesse=None, acceleration=None, lissage=None,
             profil=None, commentaire="") -> "Pilote":
        """
        Mouvement cartesien circulaire : arc passant par `via`.

        `via` et la cible acceptent l'un et l'autre une matrice homogene
        4x4 ou des coordonnees (x, y, z) / (x, y, z, r).
        """
        memoire = list(self._pose)
        point_via = self._resoudre(via, None, None, None, None)
        self._pose = memoire                     # `via` n'est qu'un passage
        self._ajouter("MOVC", duree, vitesse, acceleration, lissage,
                      profil, commentaire, repere, mode="cartesien",
                      via=point_via,
                      pose=self._resoudre(T, x, y, z, r))
        return self

    def trajet(self, points, *, repere=None, vitesse=None, acceleration=None,
               commentaire="") -> "Pilote":
        """
        Trajectoire CONTINUE affine par morceaux : le TCP suit les segments
        de droite qui relient les points, SANS S'ARRETER aux points
        intermediaires, et passe exactement par chacun.

            robot.movj(cible(200, 0, 50))
            robot.trajet([cible(200, 60, 50), cible(260, 60, 50),
                          cible(260, 0, 50), cible(200, 0, 50)],
                         vitesse=80)

        Chaque point est une matrice homogene ou des coordonnees (x, y, z).
        Le robot part de l'arret, accelere jusqu'a `vitesse` (mm/s, 125 au
        plus), ralentit dans les coins -- d'autant plus qu'ils sont vifs --
        et s'arrete au dernier point. Sur le robot reel, c'est le mode CP
        du controleur ; le simulateur en reproduit les vitesses.

        L'outil ne tourne pas pendant un trajet : tous les points gardent
        l'orientation r de depart. Orientez l'outil avant, par un movj.
        """
        from .commande import TOLERANCE_R, VITESSE_CP_MAX, _ecart_angle
        points = list(points)
        if not points:
            raise ValueError("trajet : aucun point")
        r0 = self._pose[3]
        memoire = list(self._pose)
        poses = []
        for p in points:
            pose = self._resoudre(p, None, None, None, None)
            if repere is None and abs(_ecart_angle(r0, pose[3])) > TOLERANCE_R:
                self._pose = memoire
                raise ValueError(
                    "trajet : le point {} demande r = {:.1f} deg, au lieu de "
                    "{:.1f} au depart. L'outil ne tourne pas pendant un "
                    "trajet continu : orientez-le avant, par un movj."
                    .format(len(poses) + 1, pose[3], r0))
            poses.append(pose)
        if vitesse is not None and float(vitesse) > VITESSE_CP_MAX:
            print("trajet : {:.0f} mm/s demandes, limites a {:.0f} mm/s "
                  "(vitesse la plus haute essayee en continu sur le robot)"
                  .format(float(vitesse), VITESSE_CP_MAX))
        for k, pose in enumerate(poses):
            dernier = k == len(poses) - 1
            self._ajouter("MOVL", None, vitesse, acceleration, 0.0,
                          "trapeze", commentaire if k == 0 else "", repere,
                          mode="cartesien", pose=pose, continu=not dernier)
        return self

    def repli(self, **kw) -> "Pilote":
        """Retour a la configuration de repos du robot."""
        kw.setdefault("commentaire", "repli")
        return self.movj(q=list(self.robot.q_home), **kw)

    # -- commandes auxiliaires ---------------------------------------------

    def attendre(self, secondes: float, commentaire: str = "") -> "Pilote":
        """Temporisation. Une attente n'est jamais absorbee par le lissage."""
        self.programme.ajouter(Instruction(type="ATTENTE",
                                           duree=float(secondes),
                                           commentaire=commentaire))
        return self

    def definir_repere_outil(self, T=None) -> "Pilote":
        """
        Repere OUTIL : axe de rotation de l'outil -> pointe, matrice 4x4.

        Permet un deport lateral et une rotation propre, que la simple
        longueur d'outil ne sait pas decrire. `T=None` revient au deport
        vertical deduit de l'outil monte.

            robot.definir_repere_outil(
                hg.composer(hg.translation(18, 0, -78), hg.rot_x(np.pi)))
        """
        self.robot.definir_repere_outil(T)
        self._pose = list(self.robot.pose(self.q_depart))
        return self

    def definir_repere_objet(self, T=None) -> "Pilote":
        """
        Repere OBJET : base -> plan de travail, matrice 4x4.

        Les instructions ecrites avec `repere="objet"` y sont exprimees.
        Deplacer le montage ne demande alors que de changer cette matrice.
        """
        self.programme.definir_repere_objet(T)
        return self

    @property
    def repere_objet(self) -> np.ndarray:
        return self.programme.repere_objet

    def monter_outil(self, nom: str) -> "Pilote":
        """
        Choisit l'outil monte au bout du bras.

        L'outil porte une longueur, qui devient le deport `lt` du modele
        geometrique : a partir de la, la coordonnee z designe la POINTE de
        l'outil et non plus la bride. L'outil vaut pour tout le programme,
        comme sur la machine : appelez cette methode avant les mouvements.
        """
        o = catalogue_outil(nom)
        self.programme.outil = o.nom
        self.programme.longueur_outil = o.longueur
        self.robot.geo.lt = o.longueur
        self._pose = list(self.robot.pose(self.q_depart))
        return self

    def regler_longueur_outil(self, millimetres: float) -> "Pilote":
        """
        Impose la longueur reelle de l'outil, par-dessus celle du catalogue.

        Les valeurs du catalogue sont nominales. Un montage particulier --
        cuvette differente, rallonge, embase imprimee -- change la longueur,
        et avec elle toute la coordonnee z : le robot descend trop bas ou
        pas assez.

        Pour la mesurer : amenez la POINTE au contact du plan de travail,
        lisez le z annonce par le modele, et ajoutez-le a la longueur
        courante. Si le modele annonce +17,3 mm alors que la pointe touche,
        l'outil est 17,3 mm plus long que prevu.

        La valeur est inscrite DANS LE PROGRAMME, et non seulement dans ce
        pilote : sans cela elle serait perdue des que le programme change de
        main -- simulateur, fichier JSON, robot reel -- et le z ne
        designerait plus le meme point.

        A appeler APRES `monter_outil`, qui remet la valeur du catalogue.
        """
        self.robot.geo.lt = float(millimetres)
        self.programme.longueur_outil = float(millimetres)
        self._pose = list(self.robot.pose(self.q_depart))
        return self

    @property
    def outil(self):
        """Outil actuellement monte."""
        return catalogue_outil(self.programme.outil)

    def actionner(self, actif: bool = True, commentaire: str = "",
                  couper: bool = True) -> "Pilote":
        """
        Commande l'outil monte : ventouse aspirante, pince fermee, laser
        allume. Sans effet pour un outil passif comme le porte-stylo.

        Sur le robot reel, ouvrir la pince ou relacher la ventouse fait
        SOUFFLER la pompe ; `couper` (vrai par defaut) l'arrete apres le
        mouvement suivant (ou 1 s apres, en fin de programme), sans pause
        ajoutee. `couper=False` la laisse tourner.
        """
        self.programme.ajouter(Instruction(type="OUTIL", etat=bool(actif),
                                           commentaire=commentaire,
                                           couper=bool(couper)))
        return self

    def ventouse(self, actif: bool = True, commentaire: str = "",
                 couper: bool = True) -> "Pilote":
        """Active ou relache la ventouse (alias lisible de `actionner`)."""
        return self.actionner(actif, commentaire, couper)

    def pince(self, fermee: bool = True, commentaire: str = "",
              couper: bool = True) -> "Pilote":
        """Ferme ou ouvre la pince (alias lisible de `actionner`)."""
        return self.actionner(fermee, commentaire, couper)

    # -- entrees / sorties tout-ou-rien ------------------------------------

    def convoyeur(self, vitesse: float = 0.0, tapis=None,
                  commentaire: str = "") -> "Pilote":
        """
        Regle la vitesse du tapis, en MILLIMETRES PAR SECONDE.

            robot.convoyeur(60, TAPIS)        # avance a 60 mm/s
            robot.movl(TAPIS.cible(450, 0, 5))
            robot.convoyeur(0, TAPIS)         # arret

        Une vitesse nulle arrete le tapis. Une vitesse negative le fait
        reculer. L'instruction prend sa place dans le programme comme une
        sortie : elle part quand la file l'atteint, et le tapis continue
        a tourner pendant les mouvements qui suivent.

        `tapis` est un `Convoyeur` : on lui demande son pas, en
        impulsions par millimetre, et ce pas voyage avec l'instruction
        jusqu'au robot reel. Sans lui, la valeur nominale du kit Dobot
        est retenue -- et un tapis different n'avancerait pas de la
        distance voulue.

        Le tapis du Pilote (`robot.tapis`) sert de defaut : il suffit de
        le poser une fois.
        """
        from .convoyeur import pas_signe_de
        tapis = tapis if tapis is not None else getattr(self, "tapis", None)
        pas = pas_signe_de(tapis)
        index = int(getattr(tapis, "index_moteur", 0) or 0)
        v = float(vitesse)
        self.programme.ajouter(Instruction(
            type="CONVOYEUR", vitesse=v, etat=bool(v), port=index,
            impulsions_par_mm=pas, commentaire=commentaire))
        return self

    def attendre_piece(self, port: int, etat: bool = True,
                       timeout: float = 30.0,
                       commentaire: str = "piece au capteur") -> "Pilote":
        """
        Suspend le programme jusqu'a ce que le capteur du tapis voie une
        piece.

        C'est `attendre_entree` sous un autre nom : le capteur de
        presence du kit convoyeur se cable sur une entree tout-ou-rien,
        et le programme s'y synchronise comme sur n'importe quel capteur
        de cellule. Le nom dit l'intention, ce qui se relit mieux dans un
        programme de TP.

        Le capteur infrarouge Dobot, lui, se branche sur un port GP et se
        lit par `PontReel.capteur_piece()` -- hors file, donc depuis un
        script, pas depuis le programme.
        """
        return self.attendre_entree(port, etat=etat, timeout=timeout,
                                    commentaire=commentaire)

    def sortie(self, port: int, etat: bool = True,
               commentaire: str = "") -> "Pilote":
        """Met la sortie EIO`port` a l'etat demande."""
        self.programme.ajouter(Instruction(type="SORTIE", port=int(port),
                                           etat=bool(etat),
                                           commentaire=commentaire))
        return self

    def attendre_entree(self, port: int, etat: bool = True,
                        timeout: float = 10.0,
                        commentaire: str = "") -> "Pilote":
        """
        Suspend le programme jusqu'a ce que l'entree EIO`port` prenne
        l'etat demande, au plus pendant `timeout` secondes.

        Dans le simulateur, c'est vous qui basculez l'entree depuis le
        panneau Entrees / Sorties : la lecture se fige en attendant.
        """
        self.programme.ajouter(Instruction(type="ENTREE", port=int(port),
                                           etat=bool(etat),
                                           duree=float(timeout),
                                           commentaire=commentaire))
        return self

    # -- resultats ----------------------------------------------------------

    def trajectoire(self) -> Trajectoire:
        """Compile le programme, sans ouvrir d'interface."""
        return self.programme.construire(self.robot, self.q_depart)

    def controler(self, dt: float = 0.02) -> list:
        """Liste des butees violees, points hors d'atteinte et survitesses."""
        return self.trajectoire().controler(dt)

    def resume(self) -> str:
        """Description textuelle du programme et de sa trajectoire."""
        traj = self.trajectoire()
        lignes = ["programme '{}' : {} instructions, {:.2f} s"
                  .format(self.programme.nom, len(self.programme), traj.duree)]
        if traj.lissage_actif:
            lignes[0] += "  ({:.2f} s gagnees par le lissage)".format(
                traj.duree_sans_lissage - traj.duree)
        for i, ins in enumerate(self.programme.instructions):
            duree = traj.segments[i].duree if i < len(traj.segments) else 0.0
            lignes.append("  {:02d}  {:<6.2f}s  {}".format(i + 1, duree,
                                                           ins.resume()))
        avert = traj.controler(0.05)
        if avert:
            lignes.append("  {} avertissement(s) :".format(len(avert)))
            lignes += ["    ! " + a for a in avert[:10]]
        return "\n".join(lignes)

    def enregistrer(self, chemin: str) -> "Pilote":
        self.programme.enregistrer(chemin)
        return self

    def exporter_python(self, chemin: str) -> "Pilote":
        """Ecrit le script de pilotage du robot reel (API DobotDllType)."""
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(self.programme.exporter_python(self.robot))
        return self

    # -- simulation ---------------------------------------------------------

    def pose_sur_robot(self, port: Optional[str] = None) -> np.ndarray:
        """
        La pose de l'OUTIL lue sur la machine, en matrice homogene.

        Le bras ne bouge pas : on lit. C'est le geste de
        l'APPRENTISSAGE -- on amene la ventouse ou elle doit aller, a la
        main, et on demande au robot ou il est. Le point obtenu absorbe
        d'un coup tout ce qu'on ne sait pas mesurer autrement : la place
        exacte du tapis, l'abscisse reelle du capteur, le retard de la
        chaine de scrutation, le decalage entre le bord de la piece que
        le capteur voit et son centre que la ventouse doit viser.

        Un point appris vaut mieux qu'un point calcule a partir de cotes
        relevees au decimetre.
        """
        from .robot import cible as _cible

        pont = _liaison_directe(self.robot, port)
        p = pont.liaison.pose()
        x, y, z, r = pont.depuis_controleur(p.x, p.y, p.z, p.r)
        return _cible(x=x, y=y, z=z, rz=r)

    def angles_sur_robot(self, port: Optional[str] = None) -> np.ndarray:
        """
        Les quatre angles articulaires lus sur la machine, en degres.

        Le bras ne bouge pas : on lit. Pendant du `pose_sur_robot`, mais
        en articulaire -- et c'est ce qu'il faut pour savoir si le robot
        est DEJA a une posture donnee. Un mouvement qui ne produit rien
        parce que la cible est atteinte ressemble, vu de l'atelier, a un
        mouvement qui n'a pas eu lieu.
        """
        return _liaison_directe(self.robot, port).angles_reels()

    def convoyeur_sur_robot(self, vitesse: float = 0.0, tapis=None,
                            port: Optional[str] = None) -> "Pilote":
        """
        Fait tourner le tapis sur la MACHINE, tout de suite, hors
        programme. Le bras ne bouge pas.

        A ne pas confondre avec `convoyeur()`, qui INSCRIT une consigne
        dans le programme. Celle-ci part immediatement, et sert a mener
        un cycle depuis le PC -- ce qui devient obligatoire des que le
        capteur de presence est un infrarouge sur un port GP : la file du
        controleur ne sait pas attendre dessus.

        DEUX CHEMINS, comme `executer_sur_robot`, et pour la meme raison :
        un simulateur ouvert et connecte TIENT le port serie. Sans cela,
        l'exemple du convoyeur ne pourrait pas tourner pendant qu'une
        fenetre est connectee -- alors que c'est justement ce qu'on veut
        en cours, la scene 3D d'un cote et la machine de l'autre.

        1. un simulateur ouvert et connecte : on lui confie la commande,
           il la passe avec SA liaison et anime son tapis en meme temps ;
        2. sinon, le script ouvre la liaison lui-meme et la garde.
        """
        from .passerelle import demander

        tapis = tapis if tapis is not None else self.tapis
        if demander({"tapis": dict(_decrire_tapis(tapis),
                                   action="vitesse",
                                   mm_s=float(vitesse))}) is not None:
            return self
        pont = _liaison_directe(self.robot, port)
        pont.convoyeur(float(vitesse), tapis)
        return self

    def capteur_piece_sur_robot(self, tapis=None,
                                port: Optional[str] = None) -> bool:
        """
        Lit le capteur de presence du tapis, une fois, sur la machine.

        Utile comme TEMOIN apres un ordre d'arret : si la piece quitte
        le faisceau alors qu'on vient de couper le tapis, c'est que la
        bande tourne encore. Le capteur mesure alors ce qu'aucun
        chronometre cote PC ne peut voir -- l'obeissance du materiel.

        Passe par le simulateur s'il tient le port, comme les commandes
        de tapis.
        """
        from .passerelle import demander

        tapis = tapis if tapis is not None else self.tapis
        rep = demander({"tapis": dict(_decrire_tapis(tapis),
                                      action="capteur")})
        if rep is not None:
            return bool(rep.get("vu"))
        pont = _liaison_directe(self.robot, port)
        return bool(pont.capteur_piece(tapis))

    def _lecteur_capteur(self, tapis, port: Optional[str] = None):
        """
        Rend une fonction qui lit le capteur, le chemin etant choisi UNE
        SEULE FOIS.

        Scruter en interrogeant la passerelle a chaque tour ouvrirait une
        connexion locale par lecture. On decide donc au depart -- fenetre
        connectee ou liaison directe -- et la boucle garde ce chemin.

        Le chemin choisi se paie en PRECISION, et il faut le savoir : la
        dispersion de la position d'arret vaut la vitesse du tapis
        multipliee par la periode de scrutation. En direct, une lecture
        coute une dizaine de millisecondes ; par la passerelle, un
        aller-retour de socket s'y ajoute. L'exemple du convoyeur affiche
        la periode mesuree a chaque essai plutot que de la promettre.
        """
        from .passerelle import demander

        charge = {"tapis": dict(_decrire_tapis(tapis), action="capteur")}
        if demander(charge) is not None:
            def par_la_passerelle():
                rep = demander(charge)
                if rep is None:
                    raise RuntimeError(
                        "le simulateur a ferme pendant l'attente du capteur")
                return bool(rep.get("vu"))
            return par_la_passerelle

        pont = _liaison_directe(self.robot, port)
        return lambda: bool(pont.capteur_piece(tapis))

    def attendre_piece_sur_robot(self, timeout: float = 30.0, tapis=None,
                                 etat: bool = True, periode: float = 0.0,
                                 front: bool = True,
                                 port: Optional[str] = None) -> bool:
        """
        Attend que le capteur du tapis voie une piece, en le SCRUTANT
        depuis le PC. Rend True s'il l'a vue, False si le delai expire.

        Pourquoi scruter plutot que laisser la file attendre. Un capteur
        tout-ou-rien cable sur une entree EIO se laisse attendre par le
        controleur : `attendre_piece()` suffit, le programme se suspend
        tout seul et le PC peut meme etre debranche. L'infrarouge du kit,
        lui, se branche sur un port GP et ne se lit que par une commande
        immediate : aucune instruction de la file ne peut s'y
        synchroniser. Le cycle doit alors etre mene depuis le PC, morceau
        par morceau -- c'est le prix du cablage, pas un choix de style.

        CE QUE LA SCRUTATION COUTE EN PRECISION, et comment le calculer.
        La piece continue d'avancer entre deux lectures : on ne la voit
        donc pas la ou elle est, mais la ou elle etait. La dispersion de
        la position d'arret vaut

            vitesse du tapis  x  periode de scrutation

        Sur cette cellule, une lecture du capteur prend 9,7 ms (mesure a
        115200 bauds, firmware 3.7.0) et l'ordre d'arret 9,4 ms. A
        60 mm/s, cela fait environ 1,2 mm de dispersion et 0,6 mm de
        retard moyen -- et jusqu'a 3 mm quand la liaison hoquette.

        D'ou `periode=0` par defaut : on scrute aussi vite que la liaison
        le permet. Mettre une pause n'economise rien d'utile et multiplie
        l'erreur d'autant. Et si la precision ne suffit toujours pas,
        c'est la VITESSE qu'il faut baisser : elle entre directement dans
        le produit ci-dessus.

        UN FRONT, ET NON UN NIVEAU. `front=True` attend d'abord que le
        capteur soit LIBRE, puis qu'il voie quelque chose. C'est la
        difference entre « une piece vient d'arriver » et « il y a une
        piece quelque part devant moi », et elle decide de tout.

        Sur un niveau, une piece deja presente au demarrage du cycle --
        celle que le bras vient de manquer, ou la suivante dans la file
        -- rend la main AUSSITOT : le tapis n'a pas le temps de tourner,
        et la piece qui a declenche peut etre n'importe ou en amont du
        capteur. La prise devient erratique sans que le tapis, le
        capteur ni la cible n'y soient pour rien -- on les soupconne
        tous les trois, longuement, et on les disculpe un par un.

        Le prix du front : il faut que la bande se libere entre deux
        pieces. Des pieces qui se touchent ne seront pas separees, et
        c'est normal -- un capteur de presence ne sait pas les compter.

        Ctrl+C rend la main sans rien casser : le tapis continue, c'est a
        l'appelant de l'arreter.
        """
        import time as _time

        tapis = tapis if tapis is not None else self.tapis
        if tapis is None:
            raise ValueError("aucun tapis : robot.tapis = Convoyeur(...)")
        lire = self._lecteur_capteur(tapis, port)
        fin = _time.time() + float(timeout)

        if front:
            # on laisse d'abord passer ce qui occupe deja le capteur
            libre = False
            while _time.time() < fin:
                if bool(lire()) != bool(etat):
                    libre = True
                    break
                if periode > 0:
                    _time.sleep(periode)
            if not libre:
                self.instant_detection = None
                return False
        tours = 0
        while _time.time() < fin:
            tours += 1
            if bool(lire()) == bool(etat):
                # L'instant de la detection, pour qui veut savoir ce que
                # le PC a coute entre voir la piece et arreter le tapis.
                # Un tapis qui depasse accuse souvent le programme ; le
                # chiffre tranche sans discussion.
                self.instant_detection = _time.perf_counter()
                self.tours_scrutation = tours
                return True
            if periode > 0:
                _time.sleep(periode)
        self.instant_detection = None
        return False

    def executer_sur_robot(self, vitesse: float = 30.0,
                           port: Optional[str] = None) -> bool:
        """
        Execute le programme courant sur le Dobot REEL. LE ROBOT BOUGE, tout
        de suite.

            robot.movj(PRISE @ HAUT)
            robot.movl(PRISE)
            robot.executer_sur_robot(vitesse=50)     # 50 % de la nominale

            robot.vider_programme()                  # on repart a vide...
            robot.movl(PRISE @ HAUT)
            robot.executer_sur_robot(vitesse=80)     # ...pour la suite

        L'appel attend la fin du mouvement avant de rendre la main : un
        script peut ainsi enchainer plusieurs programmes, et decider entre
        deux. Sans `vider_programme()`, un second appel REJOUERAIT tout le
        programme depuis le debut.

        En cas d'echec -- butee, collision, robot en defaut, liaison
        perdue... -- une ALERTE s'affiche dans le terminal et
        `ErreurExecution` arrete le script : la suite ne doit pas partir
        d'une posture qui n'est pas celle prevue. Pour gerer l'echec soi-
        meme : `try: ... except ErreurExecution: ...`. Ctrl+C arrete le
        robot puis le script.

        Deux chemins, choisis tout seuls :

        1. un simulateur est OUVERT ET CONNECTE au robot : il tient le port
           serie, on lui confie donc la partie par la passerelle, et il
           l'execute sans confirmation ; le script suit l'avancement ;
        2. sinon, le script se connecte lui-meme, et GARDE la liaison
           ouverte d'un appel a l'autre (fermee a la fin du script).

        Garde-fous : chaque partie est controlee avant tout envoi (butees,
        collisions) ; un robot en defaut -- c'est le cas juste apres la
        mise sous tension -- ouvre l'assistant de mise en route, et rien ne
        part tant qu'un defaut reste ; Ctrl+C declenche un ARRET D'URGENCE.
        """
        import signal

        from .passerelle import demander
        from .reel import BLOQUANTS_TELEVERSEMENT

        vitesse = float(min(max(vitesse, 1.0), 100.0))

        if not self.programme.instructions:
            print("executer_sur_robot : le programme est vide")
            return True
        partie = self.programme
        q0 = np.asarray(self.q_depart, dtype=float)

        avertissements = partie.construire(self.robot, q0).controler(0.05)
        defauts = [a for a in avertissements
                   if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
        if defauts:
            _echouer("EXECUTION REFUSEE : {} defaut(s) sur la trajectoire, "
                     "rien n'a ete envoye au robot".format(len(defauts)),
                     defauts[:8] + (["... et {} autre(s)".format(
                         len(defauts) - 8)] if len(defauts) > 8 else [])
                     + ["robot.simuler() montre ou le programme coince."])
        reserves = [a for a in avertissements if a not in defauts]
        if reserves:
            _signaler("attention : le programme s'execute, mais avec "
                      "{} reserve(s)".format(len(reserves)), reserves[:5])

        # Le bouton stop de VS Code envoie SIGTERM : on le traite comme
        # Ctrl+C, pour que l'arret d'urgence parte AVANT la fin du processus.
        def _interrompre(*_a):
            raise KeyboardInterrupt
        ancien = None
        if hasattr(signal, "SIGTERM"):
            try:
                ancien = signal.signal(signal.SIGTERM, _interrompre)
            except ValueError:               # pas dans le fil principal
                ancien = None
        try:
            # 1. un simulateur ouvert et connecte prend la main
            if demander({"programme": partie.vers_dict(),
                         "q_depart": [float(v) for v in q0],
                         "script": _script_appelant(),
                         "executer_reel": {"vitesse": vitesse}}) is not None:
                self._suivre_dans_simulateur(len(partie), vitesse)
            else:
                # 2. execution directe depuis le script
                self._executer_directement(partie, q0, vitesse, port)
        finally:
            if ancien is not None:
                signal.signal(signal.SIGTERM, ancien)
        return True

    @staticmethod
    def _suivre_dans_simulateur(n: int, vitesse: float) -> None:
        """Attend la fin d'un programme confie au simulateur ouvert."""
        import time as _time

        from .passerelle import demander

        print("{} instruction(s) confiee(s) au simulateur, a {:.0f} %"
              .format(n, vitesse))
        mise_en_route_signalee = False
        try:
            while True:
                etat = demander({"etat_reel": True})
                if etat is None:
                    _echouer("LE SIMULATEUR NE REPOND PLUS : l'execution a pu "
                             "s'interrompre", ["fenetre fermee pendant le "
                                               "mouvement ?"])
                quoi = etat.get("etat")
                if quoi == "mise_en_route" and not mise_en_route_signalee:
                    print("  robot en defaut : terminez la MISE EN ROUTE "
                          "dans la fenetre du simulateur")
                    mise_en_route_signalee = True
                elif quoi == "termine":
                    return
                elif quoi == "refuse":
                    _echouer("EXECUTION INTERROMPUE : {}".format(
                        etat.get("raison", "?")), etat.get("alarmes", []))
                _time.sleep(0.2)
        except KeyboardInterrupt:
            demander({"arret_urgence": True})
            _signaler_urgence()
            raise

    def preparer_sur_robot(self, port: Optional[str] = None) -> bool:
        """
        Declare la posture d'equerre et reprend l'origine, DANS LA
        LIAISON qui va ensuite travailler. Le bras doit y etre mis a la
        main avant l'appel.

        Pourquoi dans la meme liaison. Sur une machine dont un capteur
        d'angle est fautif -- le J3 de celle du labo -- le controleur
        relit ce capteur A CHAQUE OUVERTURE de liaison et repart sur une
        valeur fausse. Une declaration faite dans une autre liaison, par
        `dobot.py posture equerre` par exemple, est donc perdue des que
        celle-ci se referme.

        L'assistant s'ouvre deja tout seul QUAND DES ALARMES SAUTENT,
        c'est-a-dire quand la lecture fautive sort des butees du
        controleur. Mais une lecture fausse qui reste PLAUSIBLE ne
        declenche rien : le cycle part alors sur un repere faux, en
        silence, et c'est le cas le plus couteux. Appeler ceci le rend
        systematique au lieu de le laisser au hasard.
        """
        from .ui import mise_en_route_autonome

        pont = _liaison_directe(self.robot, port)
        print("  Mettez le bras EN EQUERRE -- arriere vertical, avant-bras")
        print("  horizontal -- avant de valider la mise en route.")
        if not mise_en_route_autonome(pont, self.robot):
            _echouer("MISE EN ROUTE INCOMPLETE : le robot signale encore un "
                     "defaut", ["rien n'a ete execute"])
        return True

    def _verifier_posture_annoncee(self, pont) -> None:
        """
        Refuse de partir si le robot annonce une posture IMPOSSIBLE.

        Un capteur d'angle fautif se trahit ainsi : le controleur annonce
        un axe hors de ses propres butees -- J3 a 152 degres quand la
        butee haute est a 95. Il ne sait alors plus ou est son bras, et
        AUCUNE cible ne peut etre atteinte, si juste soit-elle.

        Releve sur la machine du labo : quatre lectures consecutives, bras
        immobile en equerre, ont donne J3 = -2,5 puis -162,6, -157,8 et
        -161,8. La bonne valeur sortait une fois sur quatre, et les essais
        qui reussissaient etaient ceux-la.

        Sans ce controle, le cycle part et se trompe de cible sans rien
        signaler -- on cherche alors l'erreur dans les reperes, les cotes
        et la file du controleur, longuement.
        """
        try:
            q = pont.angles_reels()
        except Exception:
            return                          # pas de lecture : on n'invente pas
        lim = self.robot.lim
        noms = ("J1 base", "J2 bras arriere", "J3 avant-bras", "J4 outil")
        fautifs = ["{} = {:.1f} deg, hors de [{:.0f} ; {:.0f}]".format(
            noms[i], q[i], lim.q_min[i], lim.q_max[i])
            for i in range(4)
            if not (lim.q_min[i] - 1.0 <= q[i] <= lim.q_max[i] + 1.0)]
        if not fautifs:
            return
        _echouer("LE ROBOT ANNONCE UNE POSTURE IMPOSSIBLE", fautifs + [
            "Le controleur ne sait pas ou est son bras : aucune cible ne",
            "sera atteinte. Un capteur d'angle est probablement fautif.",
            "Remettez le bras en equerre et declarez la posture :",
            "    python dobot.py posture equerre",
            "puis relisez : si la valeur reste impossible, c'est le",
            "capteur, et il faudra le contourner comme sur l'autre robot."])

    def deposer_sur_robot(self, vitesse: float = 30.0,
                          port: Optional[str] = None) -> bool:
        """
        Televerse le programme, le lance, ET REND LA MAIN. PC debranchable.

        C'est la forme AUTONOME, celle d'une vraie cellule : le programme
        part en entier dans la file du controleur, l'execution demarre, et
        l'ordinateur n'a plus rien a faire. `executer_sur_robot`, lui,
        surveille jusqu'au bout et alimente la file au fur et a mesure --
        le robot y reste attache du debut a la fin.

        LA CONDITION, ET ELLE EST DURE. La file du Magician ne tient que
        32 commandes. Ce qui n'y entre pas est garde dans le PC et envoye
        pendant l'execution : un programme qui depasse n'est donc PAS
        autonome, meme s'il demarre et s'il a l'air de bien partir. On
        refuse plutot que de laisser croire.

        Ce qui rend la chose possible, c'est que le capteur se lise en
        ENTREE NUMERIQUE : la file sait attendre dessus (TRIG), alors
        qu'elle ne sait pas attendre sur un port GP. Le cycle n'a plus
        besoin d'etre decoupe depuis le PC.
        """
        from .reel import BLOQUANTS_TELEVERSEMENT

        vitesse = float(min(max(vitesse, 1.0), 100.0))
        if not self.programme.instructions:
            print("deposer_sur_robot : le programme est vide")
            return True
        partie = self.programme
        q0 = np.asarray(self.q_depart, dtype=float)

        defauts = [a for a in partie.construire(self.robot, q0).controler(0.05)
                   if any(m in a for m in BLOQUANTS_TELEVERSEMENT)]
        if defauts:
            _echouer("DEPOT REFUSE : {} defaut(s) sur la trajectoire"
                     .format(len(defauts)), defauts[:8])

        try:
            pont = _liaison_directe(self.robot, port)
        except RuntimeError as exc:
            _echouer("CONNEXION AU ROBOT IMPOSSIBLE", [str(exc)])

        from .liaison import alarmes_bloquantes, resume_alarmes
        if alarmes_bloquantes(pont.liaison.alarmes()):
            _echouer("DEPOT REFUSE : le robot est en defaut",
                     resume_alarmes(pont.liaison.alarmes())
                     + ["python dobot.py mise-en-route"])
        self._verifier_posture_annoncee(pont)
        pont.definir_ratio(vitesse)
        pont.televerser(partie, q0, outil=partie.outil)

        reste = len(getattr(pont, "_reste", []))
        if reste:
            tient = pont.marge
            pont.arret_urgence()            # vide la file : rien ne partira
            _echouer(
                "PROGRAMME TROP LONG POUR ETRE AUTONOME",
                ["{} commande(s) n'entrent pas dans la file du controleur"
                 .format(reste),
                 "elle en tient {}, et le reste serait envoye par le PC"
                 .format(tient),
                 "pendant l'execution -- donc PC branche, donc pas",
                 "autonome. Rien n'a ete lance.",
                 "",
                 "Reduisez le nombre de cycles, ou lancez le programme",
                 "avec executer_sur_robot(), qui alimente la file."])

        pont.demarrer()
        return True

    def _executer_directement(self, partie: Programme, q0, vitesse: float,
                              port: Optional[str]) -> None:
        """Simulateur ferme : le script tient lui-meme la liaison."""
        import time as _time

        from .liaison import alarmes_bloquantes, resume_alarmes

        try:
            pont = _liaison_directe(self.robot, port)
        except RuntimeError as exc:
            _echouer("CONNEXION AU ROBOT IMPOSSIBLE", [str(exc)])
        bouge = False
        try:
            if alarmes_bloquantes(pont.liaison.alarmes()):
                _signaler("robot en defaut : ouverture de l'assistant de mise "
                          "en route", resume_alarmes(pont.liaison.alarmes()))
                from .ui import mise_en_route_autonome
                if not mise_en_route_autonome(pont, self.robot):
                    _echouer("EXECUTION REFUSEE : le robot est toujours en "
                             "defaut apres la mise en route",
                             resume_alarmes(pont.liaison.alarmes()))
            self._verifier_posture_annoncee(pont)
            pont.definir_ratio(vitesse)
            pont.televerser(partie, q0, outil=partie.outil)
            pont.demarrer()
            bouge = True
            while True:
                etat = pont.etat()
                pct = int(100 * etat.progression)
                if etat.alarmes:
                    _echouer("ROBOT EN DEFAUT : execution arretee a {} %"
                             .format(pct), etat.alarmes)
                incident = getattr(pont, "_incident", "")
                if incident:
                    _echouer("ALIMENTATION DE LA FILE INTERROMPUE", [incident])
                if pont.termine():
                    return
                _time.sleep(0.2)
        except KeyboardInterrupt:
            if bouge:
                pont.arret_urgence()
                _signaler_urgence()
            raise
        except ErreurExecution:
            raise
        except Exception as exc:
            _echouer("EXECUTION IMPOSSIBLE", [str(exc)])

    def simuler(self, lancer: bool = True, bloquant: bool = True) -> int:
        """
        Ouvre le simulateur avec ce programme charge.

        `lancer=True` demarre l'animation des l'ouverture. `bloquant=False`
        rend la main sans attendre la fermeture de la fenetre : utile depuis
        une console interactive.

        Trois cas, dans cet ordre :

        1. un simulateur tourne dans un AUTRE processus -- typiquement
           lance a la main pendant qu'on execute ce script depuis VS Code :
           on lui envoie le programme par la passerelle. Il garde ainsi sa
           liaison serie et sa place a l'ecran, la ou une seconde fenetre
           se verrait refuser le port ;
        2. un simulateur tourne dans CE processus (console interactive,
           menu Fichier de l'IHM) : on y charge le programme directement ;
        3. il n'y en a pas : on ouvre une fenetre.
        """
        from pyqtgraph.Qt import QtWidgets

        from .passerelle import envoyer
        from .ui import FenetrePrincipale

        app = QtWidgets.QApplication.instance()
        if app is None and envoyer({"programme": self.programme.vers_dict(),
                                    "q_depart": [float(v) for v in self.q_depart],
                                    "lancer": bool(lancer),
                                    "script": _script_appelant()}):
            print("programme envoye au simulateur deja ouvert "
                  "({} instructions)".format(len(self.programme)))
            return 0

        if app is not None:
            # un simulateur tourne deja (script lance depuis le menu Fichier
            # de l'IHM, ou console interactive) : on y charge le programme
            # au lieu d'ouvrir une seconde fenetre
            for w in app.topLevelWidgets():
                if isinstance(w, FenetrePrincipale):
                    w.charger_programme(self.programme, self.q_depart)
                    if lancer:
                        w.demarrer_lecture()
                    self.fenetre = w
                    return 0
        else:
            app = QtWidgets.QApplication([])
            try:
                app.setStyle("Fusion")
            except Exception:
                pass

        fenetre = FenetrePrincipale()
        fenetre.charger_programme(self.programme, self.q_depart)
        fenetre.show()
        self.fenetre = fenetre

        if lancer:
            fenetre.demarrer_lecture()
        if not bloquant:
            return 0
        return app.exec_() if hasattr(app, "exec_") else app.exec()
