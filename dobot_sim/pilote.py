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
    robot.movl(PRISE @ HAUT, lissage=60)  # raccord arrondi avec le suivant
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

from .outils import outil as catalogue_outil
from .program import Instruction, Programme
from .robot import DobotMagician, decomposer_cible
from .trajectory import PROFILS, Trajectoire

__all__ = ["Pilote"]


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
             profil=None, commentaire="") -> "Pilote":
        """
        Mouvement articulaire : les angles sont interpoles, le chemin du TCP
        n'est pas rectiligne. Vitesse et acceleration sont en deg/s et deg/s2.

            robot.movj(PRISE)                    cible = matrice homogene
            robot.movj(PRISE @ DEGAGEMENT)       composition de matrices
            robot.movj(x=220, y=-90, z=60)       coordonnees directes
            robot.movj(q=[0, 15, 30, 0])         configuration articulaire
        """
        if q is not None:
            ins = self._ajouter("MOVJ", duree, vitesse, acceleration, lissage,
                                profil, commentaire, None, mode="articulaire",
                                q=[float(v) for v in q])
            self._pose = list(self.robot.pose(ins.q))
        else:
            self._ajouter("MOVJ", duree, vitesse, acceleration, lissage,
                          profil, commentaire, repere, mode="cartesien",
                          pose=self._resoudre(T, x, y, z, r))
        return self

    def movl(self, T=None, *, x=None, y=None, z=None, r=None, repere=None,
             duree=None, vitesse=None, acceleration=None, lissage=None,
             profil=None, commentaire="") -> "Pilote":
        """
        Mouvement cartesien rectiligne : le TCP suit une droite.
        Vitesse et acceleration sont en mm/s et mm/s2.

            robot.movl(PRISE)                    cible = matrice homogene
            robot.movl(PRISE, z=40)              matrice, avec z force
            robot.movl(z=25)                     descente verticale
        """
        self._ajouter("MOVL", duree, vitesse, acceleration, lissage,
                      profil, commentaire, repere, mode="cartesien",
                      pose=self._resoudre(T, x, y, z, r))
        return self

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

    def actionner(self, actif: bool = True, commentaire: str = "") -> "Pilote":
        """
        Commande l'outil monte : ventouse aspirante, pince fermee, laser
        allume. Sans effet pour un outil passif comme le porte-stylo.
        """
        self.programme.ajouter(Instruction(type="OUTIL", etat=bool(actif),
                                           commentaire=commentaire))
        return self

    def ventouse(self, actif: bool = True, commentaire: str = "") -> "Pilote":
        """Active ou relache la ventouse (alias lisible de `actionner`)."""
        return self.actionner(actif, commentaire)

    def pince(self, fermee: bool = True, commentaire: str = "") -> "Pilote":
        """Ferme ou ouvre la pince (alias lisible de `actionner`)."""
        return self.actionner(fermee, commentaire)

    # -- entrees / sorties tout-ou-rien ------------------------------------

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
                                    "lancer": bool(lancer)}):
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
