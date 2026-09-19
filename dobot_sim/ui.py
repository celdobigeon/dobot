"""
Interface graphique du simulateur Dobot Magician.

Organisation :
    - au centre  : la vue 3D (ou 2D de secours) ;
    - a gauche   : pilotage articulaire et cartesien ;
    - a droite   : editeur de programme et lecteur de trajectoire ;
    - en bas     : matrices homogenes, jacobienne, courbes et journal.
"""

from __future__ import annotations

import struct
import traceback
from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import homogeneous as hg
from . import meshes as mm
from . import outils as ou
from .es import PORTS_RESERVES, EtatES
from .program import Instruction, Programme, TYPES_INSTRUCTION, programme_demo
from .robot import (DobotMagician, ErreurCinematique, cible, depuis_xyzrpy,
                    vers_xyzrpy)
from .liaison import CommandeDobot, FonctionES, PTP
from .passerelle import Serveur
from .reel import ErreurPont, PontReel
from .trajectory import PROFILS, Trajectoire
from .viewer2d import Vue2D

try:
    from .viewer3d import Vue3D
    ERREUR_OPENGL = ""
except Exception as _exc:                    # pragma: no cover
    Vue3D = None
    ERREUR_OPENGL = "{}: {}".format(type(_exc).__name__, _exc)

QAction = getattr(QtGui, "QAction", None) or QtWidgets.QAction

TITRE = "Simulateur Dobot Magician - pilotage articulaire et cartesien"


def _police_fixe(taille: int = 9) -> QtGui.QFont:
    f = QtGui.QFont("Consolas")
    try:
        f.setStyleHint(QtGui.QFont.Monospace)
    except Exception:
        pass
    f.setPointSize(taille)
    return f


def _spin(mini, maxi, pas=1.0, suffixe="", decimales=2, valeur=0.0):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(mini, maxi)
    s.setSingleStep(pas)
    s.setDecimals(decimales)
    s.setValue(valeur)
    if suffixe:
        s.setSuffix(" " + suffixe)
    s.setKeyboardTracking(False)
    s.setMinimumWidth(92)
    return s


def _bloc_titre(titre: str, contenu) -> QtWidgets.QWidget:
    """Un intitule puis une grille, sur toute la largeur du formulaire."""
    w = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 2, 0, 0)
    v.setSpacing(2)
    lab = QtWidgets.QLabel(titre)
    lab.setStyleSheet("color:#9aa4b0")
    v.addWidget(lab)
    v.addLayout(contenu)
    return w


#: une ligne par type d'instruction, pour les bulles du catalogue
AIDE_INSTRUCTION = {
    "MOVJ": "interpole les ANGLES : chemin libre, rapide, jamais singulier",
    "MOVL": "interpole la MATRICE : le TCP suit une droite",
    "MOVC": "arc de cercle par trois points : depart, passage, arrivee",
    "ATTENTE": "temporisation ; jamais absorbee par un lissage",
    "OUTIL": "commande l'effecteur monte (ventouse, pince, laser)",
    "SORTIE": "met un port EIO a 0 ou 1",
    "ENTREE": "suspend le programme jusqu'a l'etat voulu d'un port EIO",
}

_CACHE_ICONES = {}


def _icone(nom: str) -> QtGui.QIcon:
    """
    Pictogrammes dessines a la volee, sans aucun fichier externe.

    Les trois mouvements sont representes par leur CHEMIN, c'est-a-dire par
    ce qui les distingue vraiment : courbe libre pour MOVJ, droite pour
    MOVL, arc pour MOVC.
    """
    palette = QtWidgets.QApplication.palette()
    encre = palette.color(QtGui.QPalette.WindowText)
    fond = palette.color(QtGui.QPalette.Window)
    cle = (nom, encre.name())
    if cle in _CACHE_ICONES:
        return _CACHE_ICONES[cle]

    pm = QtGui.QPixmap(32, 32)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    trait = QtGui.QPen(encre, 2.4)
    trait.setCapStyle(QtCore.Qt.RoundCap)
    trait.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(trait)
    p.setBrush(QtCore.Qt.NoBrush)

    def pt(x, y, r=2.7):
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(encre)
        p.drawEllipse(QtCore.QPointF(x, y), r, r)
        p.restore()

    def plein(chemin):
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(encre)
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*c) for c in chemin]))
        p.restore()

    if nom == "MOVJ":
        ch = QtGui.QPainterPath(QtCore.QPointF(7, 25))
        ch.cubicTo(8, 9, 19, 5, 25, 9)
        pointille = QtGui.QPen(trait)
        pointille.setStyle(QtCore.Qt.CustomDashLine)
        pointille.setDashPattern([2.2, 2.0])
        p.setPen(pointille)
        p.drawPath(ch)
        p.setPen(trait)
        pt(7, 25)
        pt(25, 9)
    elif nom == "MOVL":
        p.drawLine(QtCore.QPointF(7, 25), QtCore.QPointF(25, 9))
        pt(7, 25)
        pt(25, 9)
    elif nom == "MOVC":
        p.drawArc(QtCore.QRectF(5, 7, 22, 22), 20 * 16, 140 * 16)
        pt(26.3, 14.2)
        pt(16.0, 7.0)
        pt(5.7, 14.2)
    elif nom == "ATTENTE":
        p.drawEllipse(QtCore.QPointF(16, 16), 10.5, 10.5)
        p.drawLine(QtCore.QPointF(16, 16), QtCore.QPointF(16, 9))
        p.drawLine(QtCore.QPointF(16, 16), QtCore.QPointF(21, 18))
    elif nom == "OUTIL":                 # ventouse : tige + cuvette
        p.drawLine(QtCore.QPointF(16, 5), QtCore.QPointF(16, 14))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*c) for c in (
            (10, 25), (13, 14), (19, 14), (22, 25))]))
    elif nom == "SORTIE":                # la fleche SORT du controleur
        p.drawRect(QtCore.QRectF(4, 8, 10, 16))
        p.drawLine(QtCore.QPointF(17, 16), QtCore.QPointF(27, 16))
        plein(((27, 16), (22, 12.5), (22, 19.5)))
    elif nom == "ENTREE":                # la fleche ENTRE dans le controleur
        p.drawRect(QtCore.QRectF(18, 8, 10, 16))
        p.drawLine(QtCore.QPointF(4, 16), QtCore.QPointF(14, 16))
        plein(((14, 16), (9, 12.5), (9, 19.5)))
    elif nom == "lire":
        plein(((11, 7), (26, 16), (11, 25)))
    elif nom == "pause":
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(encre)
        p.drawRect(QtCore.QRectF(10, 8, 5, 16))
        p.drawRect(QtCore.QRectF(18, 8, 5, 16))
        p.restore()
    elif nom == "arreter":
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(encre)
        p.drawRect(QtCore.QRectF(9, 9, 14, 14))
        p.restore()
    elif nom == "pas_suiv":
        plein(((9, 8), (20, 16), (9, 24)))
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(encre)
        p.drawRect(QtCore.QRectF(22, 8, 4, 16))
        p.restore()
    elif nom == "pas_prec":
        plein(((23, 8), (12, 16), (23, 24)))
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(encre)
        p.drawRect(QtCore.QRectF(6, 8, 4, 16))
        p.restore()
    elif nom == "reel":              # antenne : la machine est au bout du fil
        p.drawLine(QtCore.QPointF(16, 27), QtCore.QPointF(16, 14))
        p.drawArc(QtCore.QRectF(9, 9, 14, 14), 30 * 16, 120 * 16)
        p.drawArc(QtCore.QRectF(4, 4, 24, 24), 30 * 16, 120 * 16)
        pt(16, 12, 2.4)
    elif nom == "ajouter":
        p.drawLine(QtCore.QPointF(16, 7), QtCore.QPointF(16, 25))
        p.drawLine(QtCore.QPointF(7, 16), QtCore.QPointF(25, 16))
    elif nom == "dupliquer":
        p.drawRect(QtCore.QRectF(6, 6, 14, 14))
        p.save()
        p.setBrush(fond)
        p.drawRect(QtCore.QRectF(12, 12, 14, 14))
        p.restore()
    elif nom == "supprimer":             # corbeille
        p.drawLine(QtCore.QPointF(7, 10), QtCore.QPointF(25, 10))
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*c) for c in (
            (13, 10), (13, 7), (19, 7), (19, 10))]))
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*c) for c in (
            (10, 10), (11.5, 25), (20.5, 25), (22, 10))]))
    elif nom == "monter":
        p.drawLine(QtCore.QPointF(16, 25), QtCore.QPointF(16, 9))
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*c) for c in (
            (10, 15), (16, 9), (22, 15))]))
    elif nom == "descendre":
        p.drawLine(QtCore.QPointF(16, 7), QtCore.QPointF(16, 23))
        p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*c) for c in (
            (10, 17), (16, 23), (22, 17))]))
    p.end()

    icone = QtGui.QIcon(pm)
    _CACHE_ICONES[cle] = icone
    return icone


# ===========================================================================

class Curseur(QtWidgets.QWidget):
    """
    Curseur + champ numerique synchronises.

    Sert aussi bien aux articulations (degres) qu'aux coordonnees
    cartesiennes (millimetres). `resolution` est le pas du curseur dans
    l'unite affichee ; le champ numerique, lui, reste au 1/100.
    """

    valeurChangee = QtCore.Signal(float) if hasattr(QtCore, "Signal") \
        else QtCore.pyqtSignal(float)

    #: Emis quand l'utilisateur a FINI de bouger : relachement du curseur ou
    #: validation du champ. `valeurChangee` part a chaque pixel parcouru, ce
    #: qui convient au modele 3D mais noierait une machine reelle sous les
    #: ordres. Le pilotage du robot s'accroche donc a ce signal-ci.
    valeurValidee = QtCore.Signal(float) if hasattr(QtCore, "Signal") \
        else QtCore.pyqtSignal(float)

    def __init__(self, nom: str, mini: float, maxi: float, valeur: float = 0.0,
                 suffixe: str = "deg", resolution: float = 0.1,
                 largeur_etiquette: int = 112, couleur: str = ""):
        super().__init__()
        self._interne = False
        self._couleur = couleur            # code couleur de l'axe, s'il y en a
        self._k = 1.0 / float(resolution)      # unite affichee -> pas entiers
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        # 4 px et non 6 : sur trois intervalles et huit curseurs, ces deux
        # pixels decidaient de l'apparition d'une barre de defilement
        lay.setSpacing(4)

        self.etiquette = QtWidgets.QLabel(nom)
        self.etiquette.setMinimumWidth(largeur_etiquette)
        if couleur:
            self.etiquette.setStyleSheet("color:{};font-weight:bold"
                                         .format(couleur))
        self.curseur = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.curseur.setRange(int(round(mini * self._k)), int(round(maxi * self._k)))
        self.curseur.setValue(self._vers_entier(valeur))
        self.curseur.setPageStep(max(int(round(10 * self._k)), 1))
        self.champ = _spin(mini, maxi, 1.0, suffixe, 2, valeur)
        # un peu plus etroit que le defaut : "180,00 deg" y tient encore,
        # et le panneau Commande cesse de deborder
        self.champ.setMinimumWidth(84)

        lay.addWidget(self.etiquette)
        lay.addWidget(self.curseur, 1)
        lay.addWidget(self.champ)

        self.curseur.valueChanged.connect(self._depuis_curseur)
        self.champ.valueChanged.connect(self._depuis_champ)
        self.curseur.sliderReleased.connect(
            lambda: self.valeurValidee.emit(self.valeur()))
        self.champ.editingFinished.connect(
            lambda: self.valeurValidee.emit(self.valeur()))

    def _vers_entier(self, v: float) -> int:
        return int(round(float(v) * self._k))

    def _depuis_curseur(self, v):
        if self._interne:
            return
        self._interne = True
        self.champ.setValue(v / self._k)
        self._interne = False
        self.valeurChangee.emit(v / self._k)

    def _depuis_champ(self, v):
        if self._interne:
            return
        self._interne = True
        self.curseur.setValue(self._vers_entier(v))
        self._interne = False
        self.valeurChangee.emit(float(v))

    def valeur(self) -> float:
        return float(self.champ.value())

    def definir(self, v: float) -> None:
        self._interne = True
        self.champ.setValue(float(v))
        self.curseur.setValue(self._vers_entier(v))
        self._interne = False

    # -- retour visuel d'atteignabilite -------------------------------------

    _probleme = False

    def signaler(self, probleme: bool) -> None:
        """
        Signale une consigne irrealisable.

        L'etiquette portant deja le code couleur de son axe, on la met en
        evidence par le FOND plutot qu'en ecrasant sa couleur.
        """
        if probleme == self._probleme:
            return                     # evite un restylage a chaque image
        self._probleme = probleme
        base = ("color:{};font-weight:bold".format(self._couleur)
                if self._couleur else "")
        if probleme:
            base += (";" if base else "") + \
                "background:#5c2a26;border-radius:3px;padding:1px 3px"
            if not self._couleur:
                base = "color:#e06c5a;font-weight:bold;" + base
        self.etiquette.setStyleSheet(base)


# retro-compatibilite : l'ancien nom reste utilisable
CurseurAngle = Curseur


# ===========================================================================

#: libelles et unites d'une situation, dans l'ordre d'affichage
COMPOSANTES = (("X", "mm"), ("Y", "mm"), ("Z", "mm"),
               ("Rx", "deg"), ("Ry", "deg"), ("Rz", "deg"))


class EditeurPose(QtWidgets.QWidget):
    """
    Saisie d'une situation sous la forme (X, Y, Z, Rx, Ry, Rz).

    C'est la forme d'affichage usuelle en robotique industrielle, et celle
    retenue dans toute l'interface. Les matrices homogenes restent la forme
    d'ECRITURE dans les scripts Python ; ici on montre les six composantes.

    `imposes` fige les composantes que la mecanique ne laisse pas libres :
    sur le Magician, Rx et Ry ne sont jamais commandables.
    """

    modifiee = QtCore.Signal() if hasattr(QtCore, "Signal") \
        else QtCore.pyqtSignal()

    def __init__(self, imposes=None, parent=None):
        super().__init__(parent)
        self._maj = False
        imposes = dict(imposes or {})

        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(5)

        self.champs = []
        for k, (nom, unite) in enumerate(COMPOSANTES):
            ligne, colonne = divmod(k, 3)
            e = QtWidgets.QLabel(nom)
            e.setMinimumWidth(24)
            s = _spin(-10000, 10000, 1.0, unite, 3)
            s.setMinimumWidth(104)
            if k in imposes:
                s.setValue(float(imposes[k]))
                s.setReadOnly(True)
                s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                s.setStyleSheet("color:#888")
                s.setToolTip("impose par la mecanique : non commandable")
                e.setStyleSheet("color:#888")
            else:
                s.valueChanged.connect(self._change)
            lay.addWidget(e, ligne, colonne * 2)
            lay.addWidget(s, ligne, colonne * 2 + 1)
            self.champs.append(s)
        lay.setColumnStretch(6, 1)

    # -- acces ---------------------------------------------------------------

    def valeurs(self) -> np.ndarray:
        return np.array([s.value() for s in self.champs], dtype=float)

    def matrice(self) -> np.ndarray:
        return depuis_xyzrpy(*self.valeurs())

    def definir(self, T) -> None:
        self.definir_valeurs(vers_xyzrpy(T))

    def definir_valeurs(self, v) -> None:
        self._maj = True
        try:
            for s, x in zip(self.champs, np.asarray(v, dtype=float)):
                s.setValue(float(x))
        finally:
            self._maj = False
        self.modifiee.emit()

    def _change(self, _=None) -> None:
        if not self._maj:
            self.modifiee.emit()


class DialogueRepere(QtWidgets.QDialog):
    """Saisie d'un repere ou d'une cible, avec validation immediate."""

    def __init__(self, parent, titre: str, explication: str,
                 T, imposes, validateur):
        super().__init__(parent)
        self.setWindowTitle(titre)
        self._validateur = validateur

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)

        info = QtWidgets.QLabel(explication)
        info.setWordWrap(True)
        info.setStyleSheet("color:#888")
        lay.addWidget(info)

        self.editeur = EditeurPose(imposes)
        self.editeur.modifiee.connect(self._valider)
        lay.addWidget(self.editeur)

        self.etat = QtWidgets.QLabel()
        self.etat.setWordWrap(True)
        self.etat.setMinimumHeight(44)
        self.etat.setAlignment(QtCore.Qt.AlignTop)
        lay.addWidget(self.etat)

        self.boutons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.boutons.accepted.connect(self.accept)
        self.boutons.rejected.connect(self.reject)
        lay.addWidget(self.boutons)

        self.editeur.definir(T)

    def matrice(self) -> np.ndarray:
        return self.editeur.matrice()

    def _valider(self) -> None:
        try:
            valeurs = self._validateur(self.editeur.matrice())
        except Exception as exc:
            self.etat.setText("<b style='color:#e06c5a'>refusee</b><br>" + str(exc))
            self.boutons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
            return
        self.etat.setText(
            "<span style='color:#5aa469'>valide</span> &nbsp; "
            "x={:.2f}  y={:.2f}  z={:.2f}  angle={:.2f} deg"
            .format(*valeurs))
        self.boutons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)


# ===========================================================================

class DialogueCalibrage(QtWidgets.QDialog):
    """Ajustement fin du recalage des maillages CAO sur la cinematique."""

    def __init__(self, parent, config: dict):
        super().__init__(parent)
        self.setWindowTitle("Calibrer le modele 3D")
        self.config = config
        self.resize(640, 460)

        lay = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Les maillages CAO sont livres dans une pose d'assemblage unique.\n"
            "Ces valeurs replacent chaque piece dans le repere de son corps.\n"
            "Modifiez, cliquez Appliquer pour voir le resultat, puis Enregistrer.")
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QtWidgets.QFormLayout()
        self.rot = _spin(-180, 180, 1.0, "deg", 2,
                         config.get("rotation_z_deg", 0.0))
        form.addRow("Rotation CAO -> robot (autour de z)", self.rot)

        ligne = QtWidgets.QHBoxLayout()
        self.q0 = []
        for i, v in enumerate(config.get("pose_assemblage_deg", [0, 0, 0, 0])):
            s = _spin(-180, 180, 0.5, "deg", 2, v)
            self.q0.append(s)
            ligne.addWidget(QtWidgets.QLabel("J{}".format(i + 1)))
            ligne.addWidget(s)
        cont = QtWidgets.QWidget()
        cont.setLayout(ligne)
        form.addRow("Pose d'assemblage du modele", cont)
        lay.addLayout(form)

        self.table = QtWidgets.QTableWidget(len(config.get("pieces", [])), 5)
        self.table.setHorizontalHeaderLabels(
            ["piece", "repere", "tx (mm)", "ty (mm)", "tz (mm)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, p in enumerate(config.get("pieces", [])):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(p.get("nom", "")))
            self.table.item(i, 0).setFlags(QtCore.Qt.ItemIsEnabled)
            for j, val in enumerate([p.get("repere", 0)] + list(p.get("translation", [0, 0, 0]))):
                self.table.setItem(i, j + 1, QtWidgets.QTableWidgetItem(str(val)))
        lay.addWidget(self.table, 1)

        boutons = QtWidgets.QDialogButtonBox()
        self.b_appliquer = boutons.addButton("Appliquer",
                                             QtWidgets.QDialogButtonBox.ApplyRole)
        self.b_enregistrer = boutons.addButton("Enregistrer",
                                               QtWidgets.QDialogButtonBox.AcceptRole)
        boutons.addButton("Fermer", QtWidgets.QDialogButtonBox.RejectRole)
        boutons.rejected.connect(self.reject)
        lay.addWidget(boutons)

    def lire(self) -> dict:
        """Reconstruit un dictionnaire de configuration a partir du formulaire."""
        cfg = dict(self.config)
        cfg["rotation_z_deg"] = self.rot.value()
        cfg["pose_assemblage_deg"] = [s.value() for s in self.q0]
        pieces = []
        for i, p in enumerate(self.config.get("pieces", [])):
            q = dict(p)
            try:
                q["repere"] = int(float(self.table.item(i, 1).text()))
                q["translation"] = [float(self.table.item(i, j).text())
                                    for j in (2, 3, 4)]
            except (ValueError, AttributeError):
                pass
            pieces.append(q)
        cfg["pieces"] = pieces
        return cfg


# ===========================================================================

class FenetrePrincipale(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITRE)
        self.resize(1560, 940)

        self.robot = DobotMagician()
        self.q = np.array(self.robot.q_home, dtype=float)
        self._q_depart = np.array(self.robot.q_home, dtype=float)
        self.programme = programme_demo()
        self.trajectoire: Optional[Trajectoire] = None
        self.config_modele = mm.charger_config()
        self._maj_en_cours = False
        self._pieces_cao: List[mm.Piece] = []

        self.es = EtatES()
        self.outil = ou.outil(self.programme.outil)
        self.robot.geo.lt = self.outil.longueur
        self.outil_actif = False
        self._attente_es = 0.0
        self._segment_attente = -1
        self._maj_es = False
        # vrai quand c'est le code, et non l'utilisateur, qui change la
        # ligne selectionnee : on ne deplace alors pas le robot
        self._selection_programmee = False
        #: panneaux ancres, pour pouvoir les rouvrir apres fermeture
        self.docks: List[QtWidgets.QDockWidget] = []
        #: avertissements du dernier controle de trajectoire, et drapeau
        #: disant que le bandeau parle de la POSE et non du programme
        self._avert_programme: List[str] = []
        self._alerte_posturale = False

        self._creer_vue()
        self._creer_dock_commande()
        self._creer_dock_programme()
        self._creer_docks_bas()
        self._creer_dock_reel()
        self._creer_menus()
        self._proportions_initiales()

        self.minuteur = QtCore.QTimer(self)
        self.minuteur.setInterval(20)
        self.minuteur.timeout.connect(self._pas_animation)
        self.t_courant = 0.0
        self.en_lecture = False

        self._charger_maillages()
        self._recompiler()
        self._appliquer_q(self.q, source="init")

        # Passerelle : un script lance depuis VS Code tourne dans un AUTRE
        # processus. Sans elle il ouvrait une seconde fenetre, a qui le
        # port serie -- tenu par la premiere -- etait refuse.
        self._serveur = Serveur(self._recevoir_programme, journal=self.journal)

        self.journal("Simulateur pret. {}".format(self.robot))

    # -- construction -------------------------------------------------------

    def _proportions_initiales(self) -> None:
        """
        Largeurs de depart des panneaux lateraux.

        Il faut viser ce dont le contenu a BESOIN, pas un minimum arbitraire.
        Une premiere version forcait `minimumWidth()` : les champs de saisie
        des curseurs se retrouvaient coupes de 250 px, donc inaccessibles a
        la souris. Un panneau qu'on ne peut pas utiliser ne fait pas gagner
        de place, il en perd.

        On borne tout de meme a 32 % de la fenetre, pour qu'un panneau
        bavard ne mange pas la vue 3D.
        """
        lateraux = [d for d in self.docks
                    if d.isVisible()
                    and self.dockWidgetArea(d) in (QtCore.Qt.LeftDockWidgetArea,
                                                   QtCore.Qt.RightDockWidgetArea)]
        if not lateraux:
            return
        plafond = max(int(self.width() * 0.32), 380)
        largeurs = [int(np.clip(self._largeur_utile(d), d.minimumWidth(),
                                plafond))
                    for d in lateraux]
        self.resizeDocks(lateraux, largeurs, QtCore.Qt.Horizontal)

    @staticmethod
    def _largeur_utile(dock) -> int:
        """
        Largeur qu'il faut vraiment a un dock pour n'etre pas tronque.

        On vise le minimumSizeHint et non le sizeHint : le second est la
        largeur PREFEREE, qu'une simple etiquette de texte long fait
        exploser. Et quand le contenu defile, on ajoute la barre de
        defilement verticale, sans quoi il manque toujours une dizaine de
        pixels et une barre horizontale apparait.
        """
        w = dock.widget()
        if w is None:
            return dock.minimumWidth()
        besoin = w.minimumSizeHint().width()
        for sa in w.findChildren(QtWidgets.QScrollArea):
            interne = sa.widget()
            if interne is not None:
                besoin = max(besoin,
                             interne.minimumSizeHint().width()
                             + sa.verticalScrollBar().sizeHint().width()
                             + 2 * sa.frameWidth())
        return besoin + 12          # marges du dock

    def _creer_vue(self) -> None:
        if Vue3D is not None:
            try:
                self.vue = Vue3D(self.robot)
                self.mode_3d = True
            except Exception as exc:          # pragma: no cover
                self.vue = Vue2D(self.robot)
                self.mode_3d = False
                globals()["ERREUR_OPENGL"] = str(exc)
        else:                                  # pragma: no cover
            self.vue = Vue2D(self.robot)
            self.mode_3d = False

        # Bandeau d'alerte au-dessus de la vue. Un programme dont les cibles
        # sont hors d'atteinte se joue quand meme -- le robot se replie sur
        # lui-meme sans rien signaler de visible. Le dire franchement.
        self.bandeau = QtWidgets.QLabel()
        self.bandeau.setWordWrap(True)
        self.bandeau.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        self.bandeau.hide()

        # Badge superpose a la vue 3D. Quand une MACHINE est au bout du fil,
        # il faut que cela se voie la ou l'on regarde -- pas seulement dans
        # un panneau lateral que l'on a peut-etre replie.
        self.badge_reel = QtWidgets.QLabel(self.vue)
        self.badge_reel.setTextFormat(QtCore.Qt.RichText)
        self.badge_reel.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.badge_reel.hide()
        self.vue.installEventFilter(self)

        centre = QtWidgets.QWidget()
        pile = QtWidgets.QVBoxLayout(centre)
        pile.setContentsMargins(0, 0, 0, 0)
        pile.setSpacing(0)
        pile.addWidget(self.bandeau)
        pile.addWidget(self.vue, 1)
        self.setCentralWidget(centre)

    def eventFilter(self, objet, evenement):
        """Recale le badge dans le coin de la vue a chaque redimensionnement."""
        if (objet is getattr(self, "vue", None)
                and evenement.type() == QtCore.QEvent.Resize):
            self._placer_badge()
        return super().eventFilter(objet, evenement)

    def _placer_badge(self) -> None:
        if not getattr(self, "badge_reel", None):
            return
        self.badge_reel.adjustSize()
        marge = 12
        self.badge_reel.move(
            max(self.vue.width() - self.badge_reel.width() - marge, marge),
            marge)

    #: apparence du badge selon ce que fait la machine
    BADGES = {
        "pilotage": ("#8a3d0b", "#e8913c", "#ffe8cf", "ROBOT REEL PILOTE",
                     "les curseurs commandent la machine"),
        "execution": ("#7a1f1f", "#c0392b", "#ffe3e3", "ROBOT REEL EN MARCHE",
                      "le programme s'execute sur la machine"),
        "defaut": ("#7a1f1f", "#c0392b", "#ffe3e3", "ROBOT REEL EN DEFAUT",
                   "la file est arretee"),
        "miroir": ("#123246", "#2f6f96", "#cfe3ff", "ROBOT REEL CONNECTE",
                   "le modele suit la machine"),
        "connecte": ("#123246", "#2f6f96", "#cfe3ff", "ROBOT REEL CONNECTE",
                     "liaison ouverte"),
    }

    def _maj_badge_reel(self, etat=None) -> None:
        """
        Affiche l'etat de la machine par-dessus la vue 3D.

        Trois degres : bleu quand la liaison est simplement ouverte, ambre
        quand les curseurs la commandent, rouge quand elle execute ou
        qu'elle est en defaut.
        """
        badge = getattr(self, "badge_reel", None)
        if badge is None:
            return
        if not getattr(self, "pont", None) or not self.pont.connecte:
            badge.hide()
            return

        if etat is not None and etat.alarmes:
            cle = "defaut"
        elif self.chk_piloter.isChecked():
            cle = "pilotage"
        elif etat is not None and 0.0 < etat.progression < 1.0:
            cle = "execution"
        elif self.chk_miroir.isChecked():
            cle = "miroir"
        else:
            cle = "connecte"

        fond, bord, encre, titre, sous_titre = self.BADGES[cle]
        badge.setStyleSheet(
            "QLabel {{ background:{}; color:{}; border:2px solid {};"
            " border-radius:5px; padding:6px 10px; }}".format(fond, encre, bord))
        badge.setText(
            "<b>&#9679; {}</b><br><span style='font-size:11px'>{}</span>"
            .format(titre, sous_titre))
        badge.show()
        badge.raise_()
        self._placer_badge()

    #: ce qui distingue un refus mecanique d'une simple remarque, par famille
    MOTS_COLLISION = ("plan de pose", "embase :", "colonne :")
    MOTS_ATTEINTE = ("hors de [", "hors d'atteinte", "inatteignable",
                     "n'a pas pu etre compile")
    MOTS_BLOQUANTS = (MOTS_COLLISION + MOTS_ATTEINTE
                      + ("Arret de securite", "Robot reel en defaut"))

    def _alerter(self, avertissements: Sequence[str]) -> None:
        """
        Affiche ou efface le bandeau au-dessus de la vue 3D.

        Deux niveaux, parce que les avertissements n'ont pas le meme poids :
        une cible hors des butees ou une collision empechent le mouvement
        (rouge), un arc degenere ou une survitesse ne font que meriter un
        coup d'oeil (ambre).
        """
        if not avertissements:
            self.bandeau.hide()
            return

        bloquants = [a for a in avertissements
                     if any(m in a for m in self.MOTS_BLOQUANTS)]
        if bloquants:
            fond, bord, encre = "#7a1f1f", "#c0392b", "#ffe3e3"
            liste = bloquants
            # la tete decrit le message AFFICHE, c'est-a-dire le premier :
            # sinon on annoncerait une famille et on en montrerait une autre
            if (bloquants[0].startswith("Arret de securite")
                    or bloquants[0].startswith("Robot reel en defaut")):
                tete = ""                    # il porte deja son propre titre
            elif any(m in bloquants[0] for m in self.MOTS_COLLISION):
                tete = "Collision : le bras traverse un volume du robot."
            else:
                tete = ("Trajectoire irrealisable : le robot ne peut pas "
                        "atteindre ces cibles.")
        else:
            fond, bord, encre = "#5c4410", "#c8931e", "#ffeecf"
            tete = "Le mouvement est execute, mais avec une reserve."
            liste = list(avertissements)

        detail = liste[0]
        if len(liste) > 1:
            detail += "   (+ {} autre(s))".format(len(liste) - 1)
        self.bandeau.setStyleSheet(
            "QLabel {{ background:{}; color:{}; padding:7px 10px;"
            " border-bottom:2px solid {}; font-weight:bold; }}"
            .format(fond, encre, bord))
        self.bandeau.setText(
            "{}{}   --   detail dans l'onglet Journal."
            .format(tete + "   " if tete else "", detail))
        self.bandeau.show()

    def _creer_dock_commande(self) -> None:
        dock = QtWidgets.QDockWidget("Commande", self)
        dock.setObjectName("dock_commande")
        # Les trois panneaux sont empiles et visibles simultanement ;
        # chaque titre est une case a cocher qui replie sa section.
        self.sections_commande = {}

        # --- articulaire
        page_a = QtWidgets.QWidget()
        la = QtWidgets.QVBoxLayout(page_a)
        la.setContentsMargins(6, 2, 2, 2)
        self.curseurs: List[Curseur] = []
        for i, nom in enumerate(self.robot.NOMS_ARTICULATIONS):
            c = Curseur(nom, self.robot.lim.q_min[i], self.robot.lim.q_max[i],
                        self.q[i], suffixe="deg", resolution=0.1)
            c.curseur.setToolTip("{} : de {:.0f} a {:.0f} deg"
                                 .format(nom, self.robot.lim.q_min[i],
                                         self.robot.lim.q_max[i]))
            c.valeurChangee.connect(self._depuis_curseurs)
            # le robot reel n'est sollicite qu'au RELACHEMENT
            c.valeurValidee.connect(self._reel_envoyer_pose)
            self.curseurs.append(c)
            la.addWidget(c)
        la.addSpacing(6)
        b = QtWidgets.QHBoxLayout()
        for txt, val in (("Zero", [0, 0, 0, 0]), ("Repli", list(self.robot.q_home))):
            bt = QtWidgets.QPushButton(txt)
            bt.clicked.connect(lambda _=False, v=val: self._appliquer_q(np.array(v, float)))
            b.addWidget(bt)
        la.addLayout(b)

        # --- cartesien
        page_c = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(page_c)
        lc.setContentsMargins(6, 2, 2, 2)
        self.champs_cart = {}
        # meme code couleur que les trieds de la vue 3D : x rouge, y vert,
        # z bleu. r tourne AUTOUR de z, on le rattache donc au bleu, en
        # plus clair pour ne pas le confondre avec la translation.
        for cle, libelle, mini, maxi, suf, res, coul, bulle in (
                ("x", "X  (avant)", -350.0, 350.0, "mm", 0.5, "#e04a4a",
                 "abscisse du TCP, axe x vers l'avant"),
                ("y", "Y  (gauche)", -350.0, 350.0, "mm", 0.5, "#4cbf58",
                 "ordonnee du TCP, axe y vers la gauche"),
                ("z", "Z  (hauteur)", -80.0, 330.0, "mm", 0.5, "#4f80f2",
                 "hauteur du TCP au-dessus du plan de pose"),
                ("r", "Rz  (outil)", -180.0, 180.0, "deg", 0.1, "#7fa8ff",
                 "rotation de l'outil autour de la verticale, Rz = J1 + J4")):
            c = Curseur(libelle, mini, maxi, 0.0, suffixe=suf, resolution=res,
                        couleur=coul)
            c.curseur.setToolTip(bulle)
            c.valeurChangee.connect(self._depuis_cartesien)
            c.valeurValidee.connect(self._reel_envoyer_pose)
            self.champs_cart[cle] = c
            lc.addWidget(c)

        impose = QtWidgets.QLabel(
            "Rx = 180,00 deg   Ry = 0,00 deg\n"
            "imposes par le parallelogramme, non commandables")
        impose.setWordWrap(True)          # sinon il impose sa largeur au dock
        impose.setStyleSheet("color:#888")
        lc.addWidget(impose)

        lc.addSpacing(6)
        self.chk_suivi = QtWidgets.QCheckBox("appliquer le MGI en continu")
        self.chk_suivi.setChecked(True)
        lc.addWidget(self.chk_suivi)

        b = QtWidgets.QHBoxLayout()
        bt = QtWidgets.QPushButton("Appliquer le MGI")
        bt.clicked.connect(lambda: self._depuis_cartesien(force=True))
        b.addWidget(bt)
        bt2 = QtWidgets.QPushButton("Resynchroniser")
        bt2.setToolTip("recopie la pose courante du robot dans les curseurs")
        bt2.clicked.connect(lambda: self._ecrire_cartesien(self.robot.pose(self.q)))
        b.addWidget(bt2)
        lc.addLayout(b)

        aide = QtWidgets.QLabel(
            "Chaque deplacement d'un curseur appelle le modele geometrique "
            "inverse. Si le point demande sort de l'espace atteignable, le "
            "robot ne bouge pas et les axes concernes passent en rouge.")
        aide.setWordWrap(True)
        aide.setStyleSheet("color:#888")
        lc.addSpacing(6)
        lc.addWidget(aide)

        # --- outil
        page_o = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(page_o)
        lo.setContentsMargins(6, 2, 2, 2)
        form_o = QtWidgets.QFormLayout()
        self.cb_outil = QtWidgets.QComboBox()
        for nom in ou.noms_outils():
            self.cb_outil.addItem(ou.CATALOGUE[nom].libelle, nom)
        self.cb_outil.setCurrentIndex(
            max(0, ou.noms_outils().index(self.outil.nom)))
        self.cb_outil.currentIndexChanged.connect(self._changer_outil)
        form_o.addRow("outil monte", self.cb_outil)
        self.lab_longueur = QtWidgets.QLabel()
        form_o.addRow("longueur", self.lab_longueur)
        lo.addLayout(form_o)

        self.lab_outil = QtWidgets.QLabel()
        self.lab_outil.setWordWrap(True)
        self.lab_outil.setStyleSheet("color:#888")
        lo.addWidget(self.lab_outil)

        self.bt_outil = QtWidgets.QPushButton("Actionner")
        self.bt_outil.setCheckable(True)
        self.bt_outil.toggled.connect(self._actionner_outil)
        lo.addWidget(self.bt_outil)

        rappel = QtWidgets.QLabel(
            "La longueur de l'outil alimente le deport lt du modele : une "
            "fois l'outil monte, la coordonnee z designe sa POINTE et non "
            "plus la bride. Changer d'outil change donc le MGI, et l'enveloppe "
            "de travail avec lui.")
        rappel.setWordWrap(True)
        rappel.setStyleSheet("color:#888")
        lo.addSpacing(6)
        lo.addWidget(rappel)

        # --- reperes outil et objet, saisis en matrices 4x4
        page_r = QtWidgets.QWidget()
        lr = QtWidgets.QVBoxLayout(page_r)
        lr.setContentsMargins(6, 2, 2, 2)
        intro = QtWidgets.QLabel(
            "Deux reperes decrivent la cellule. "
            "Le repere OUTIL va de l'axe de rotation de l'outil a sa pointe : "
            "il autorise un deport lateral et une rotation propre, que la "
            "seule longueur ne sait pas decrire. Le repere OBJET va de la base "
            "au plan de travail : les cibles marquees <i>objet</i> y sont "
            "exprimees, et deplacer le montage ne change qu'un repere. "
            "Ils s'affichent et se saisissent ici en (X, Y, Z, Rx, Ry, Rz) ; "
            "dans un script Python ils acceptent aussi une matrice 4x4.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#888")
        lr.addWidget(intro)

        self.lab_rep_outil = QtWidgets.QLabel()
        self.lab_rep_outil.setFont(_police_fixe(9))
        self.lab_rep_objet = QtWidgets.QLabel()
        self.lab_rep_objet.setFont(_police_fixe(9))
        for titre, label, definir, reinit in (
                ("Repere outil", self.lab_rep_outil,
                 self._definir_repere_outil, self._reinit_repere_outil),
                ("Repere objet", self.lab_rep_objet,
                 self._definir_repere_objet, self._reinit_repere_objet)):
            lr.addSpacing(6)
            t = QtWidgets.QLabel("<b>{}</b>".format(titre))
            lr.addWidget(t)
            lr.addWidget(label)
            b = QtWidgets.QHBoxLayout()
            bt = QtWidgets.QPushButton("Modifier...")
            bt.clicked.connect(definir)
            b.addWidget(bt)
            bt2 = QtWidgets.QPushButton("Par defaut")
            bt2.clicked.connect(reinit)
            b.addWidget(bt2)
            lr.addLayout(b)
        lr.addStretch(1)

        # --- assemblage : les trois sections l'une sous l'autre
        empile = QtWidgets.QWidget()
        le = QtWidgets.QVBoxLayout(empile)
        le.setContentsMargins(0, 0, 0, 0)
        le.setSpacing(6)
        for titre, page in (("Articulaire", page_a), ("Cartesien", page_c),
                            ("Outil", page_o), ("Reperes", page_r)):
            boite = QtWidgets.QGroupBox(titre)
            boite.setCheckable(True)
            boite.setChecked(True)
            boite.setToolTip("decocher pour replier cette section")
            lb = QtWidgets.QVBoxLayout(boite)
            lb.setContentsMargins(4, 4, 4, 4)
            lb.addWidget(page)
            boite.toggled.connect(page.setVisible)
            self.sections_commande[titre] = boite
            le.addWidget(boite)
        le.addStretch(1)

        defilement = QtWidgets.QScrollArea()
        defilement.setWidgetResizable(True)
        defilement.setFrameShape(QtWidgets.QFrame.NoFrame)
        defilement.setWidget(empile)

        conteneur = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(conteneur)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.addWidget(defilement, 1)
        self.etat_config = QtWidgets.QLabel()
        self.etat_config.setWordWrap(True)
        self.etat_config.setMinimumHeight(72)
        self.etat_config.setAlignment(QtCore.Qt.AlignTop)
        lv.addWidget(self.etat_config)

        dock.setWidget(conteneur)
        dock.setMinimumWidth(360)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.docks.append(dock)

    def _creer_dock_programme(self) -> None:
        dock = QtWidgets.QDockWidget("Programme", self)
        dock.setObjectName("dock_programme")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)

        # --- barre d'edition : des pictogrammes, pas cinq boutons en toutes
        # lettres qui imposaient a eux seuls la largeur du dock
        self.barre_edition = QtWidgets.QToolBar()
        self.barre_edition.setIconSize(QtCore.QSize(18, 18))
        self.barre_edition.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        # le "+" ajoute un MOVJ ; sa fleche ouvre le catalogue des sept
        # instructions, chacune avec son pictogramme
        self.bt_ajout = QtWidgets.QToolButton()
        self.bt_ajout.setIcon(_icone("ajouter"))
        self.bt_ajout.setToolTip(
            "ajouter une instruction\nla fleche ouvre le catalogue des types")
        self.menu_ajout = QtWidgets.QMenu(self.bt_ajout)
        for t in TYPES_INSTRUCTION:
            a = self.menu_ajout.addAction(_icone(t), t)
            a.setToolTip(AIDE_INSTRUCTION.get(t, ""))
            a.triggered.connect(lambda _=False, t=t: self._ajouter_instruction(t))
        self.bt_ajout.setMenu(self.menu_ajout)
        self.bt_ajout.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.bt_ajout.clicked.connect(lambda: self._ajouter_instruction())
        self.barre_edition.addWidget(self.bt_ajout)

        self.actions_edition = {}
        for nom, bulle, fn in (
                ("dupliquer", "dupliquer l'instruction selectionnee",
                 self._dupliquer),
                ("supprimer", "supprimer l'instruction selectionnee",
                 self._supprimer),
                ("monter", "remonter d'une ligne", lambda: self._deplacer(-1)),
                ("descendre", "descendre d'une ligne",
                 lambda: self._deplacer(1))):
            a = self.barre_edition.addAction(_icone(nom), bulle)
            a.setToolTip(bulle)
            a.triggered.connect(fn)
            self.actions_edition[nom] = a
        lay.addWidget(self.barre_edition)

        self.liste = QtWidgets.QListWidget()
        self.liste.setFont(_police_fixe(9))
        self.liste.setIconSize(QtCore.QSize(16, 16))
        self.liste.currentRowChanged.connect(self._selection_changee)
        lay.addWidget(self.liste, 1)

        # --- formulaire d'edition, repliable : c'est lui qui occupe le plus
        # de hauteur, et on n'en a pas besoin pour regarder les courbes
        boite = QtWidgets.QGroupBox("Instruction selectionnee")
        boite.setCheckable(True)
        boite.setChecked(True)
        boite.setToolTip("decocher pour replier le formulaire")
        lb = QtWidgets.QVBoxLayout(boite)
        lb.setContentsMargins(4, 4, 4, 4)
        page_form = QtWidgets.QWidget()
        lb.addWidget(page_form)
        boite.toggled.connect(page_form.setVisible)
        self.boite_instruction = boite
        form = QtWidgets.QFormLayout(page_form)
        form.setContentsMargins(0, 0, 0, 0)

        self.cb_type = QtWidgets.QComboBox()
        self.cb_type.addItems(TYPES_INSTRUCTION)
        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems(["cartesien", "articulaire"])
        self.cb_profil = QtWidgets.QComboBox()
        self.cb_profil.addItems(sorted(PROFILS.keys()))
        form.addRow("type", self.cb_type)
        form.addRow("mode", self.cb_mode)

        # Six composantes, dans l'ordre X Y Z / Rx Ry Rz. En articulaire les
        # quatre premieres deviennent J1..J4 et les deux dernieres disparaissent.
        self.champs_cible = []
        self.etiq_cible = []
        grille = QtWidgets.QGridLayout()
        grille.setHorizontalSpacing(8)
        for i in range(6):
            lab = QtWidgets.QLabel("")
            lab.setMinimumWidth(24)
            s = _spin(-10000, 10000, 1.0, "", 2)
            s.setMinimumWidth(74)
            self.etiq_cible.append(lab)
            self.champs_cible.append(s)
            ligne, colonne = divmod(i, 3)
            grille.addWidget(lab, ligne, colonne * 2)
            grille.addWidget(s, ligne, colonne * 2 + 1)
        grille.setColumnStretch(6, 1)
        # sur toute la largeur : la colonne d'etiquettes du formulaire
        # coutait une centaine de pixels pour rien
        form.addRow(_bloc_titre("cible", grille))

        ligne_rep = QtWidgets.QHBoxLayout()
        self.cb_repere = QtWidgets.QComboBox()
        self.cb_repere.addItems(["base", "objet"])
        self.cb_repere.setToolTip(
            "repere dans lequel les coordonnees ci-dessus sont exprimees")
        ligne_rep.addWidget(self.cb_repere, 1)
        bt_mat = QtWidgets.QPushButton("Saisir...")
        bt_mat.setToolTip("saisir la cible en (X, Y, Z, Rx, Ry, Rz)")
        bt_mat.clicked.connect(self._saisir_cible)
        ligne_rep.addWidget(bt_mat)
        cont_rep = QtWidgets.QWidget()
        cont_rep.setLayout(ligne_rep)
        form.addRow("repere", cont_rep)

        self.champs_via = []
        gv = QtWidgets.QGridLayout()
        gv.setHorizontalSpacing(8)
        for i, (nom, unite) in enumerate(COMPOSANTES):
            lab = QtWidgets.QLabel(nom)
            lab.setMinimumWidth(24)
            s = _spin(-10000, 10000, 1.0, unite, 2)
            s.setMinimumWidth(74)
            if i in (3, 4):                 # Rx, Ry : imposes
                s.setReadOnly(True)
                s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                s.setStyleSheet("color:#888")
                lab.setStyleSheet("color:#888")
            self.champs_via.append(s)
            ligne, colonne = divmod(i, 3)
            gv.addWidget(lab, ligne, colonne * 2)
            gv.addWidget(s, ligne, colonne * 2 + 1)
        gv.setColumnStretch(6, 1)
        self.bloc_via = _bloc_titre("point de passage", gv)
        form.addRow(self.bloc_via)

        self.cb_temps = QtWidgets.QComboBox()
        self.cb_temps.addItems(["vitesse", "duree"])
        self.cb_temps.setToolTip(
            "vitesse : la duree est calculee a partir de la longueur du chemin\n"
            "duree   : la duree est imposee telle quelle")
        form.addRow("cadence", self.cb_temps)

        self.sp_duree = _spin(0.05, 60.0, 0.25, "s", 2, 2.0)
        form.addRow("duree imposee", self.sp_duree)

        self.sp_vitesse = _spin(1.0, 600.0, 10.0, "mm/s", 1, 120.0)
        self.lab_vitesse = QtWidgets.QLabel("vitesse max")
        form.addRow(self.lab_vitesse, self.sp_vitesse)

        self.sp_accel = _spin(10.0, 5000.0, 50.0, "mm/s2", 0, 500.0)
        self.lab_accel = QtWidgets.QLabel("acceleration max")
        form.addRow(self.lab_accel, self.sp_accel)

        self.sp_lissage = _spin(0.0, 100.0, 5.0, "%", 0, 0.0)
        self.sp_lissage.setToolTip(
            "arrondi du raccord avec l'instruction suivante\n"
            "0 = point d'arret exact, 100 = recouvrement maximal\n"
            "le point de passage n'est alors plus atteint exactement")
        form.addRow("lissage du raccord", self.sp_lissage)

        self.lab_duree_calc = QtWidgets.QLabel("")
        self.lab_duree_calc.setStyleSheet("color:#888")
        form.addRow("", self.lab_duree_calc)

        form.addRow("loi horaire", self.cb_profil)
        self.sp_port = _spin(1, 20, 1.0, "", 0, 1.0)
        self.sp_port.setToolTip("numero de port EIO, de 1 a 20")
        self.lab_port = QtWidgets.QLabel("port EIO")
        form.addRow(self.lab_port, self.sp_port)

        self.chk_outil = QtWidgets.QCheckBox("etat actif")
        form.addRow("", self.chk_outil)
        self.ed_commentaire = QtWidgets.QLineEdit()
        form.addRow("commentaire", self.ed_commentaire)

        bt_capture = QtWidgets.QPushButton("Capturer la pose courante")
        bt_capture.clicked.connect(self._capturer_pose)
        form.addRow("", bt_capture)
        lay.addWidget(boite)

        # --- lecteur
        boite2 = QtWidgets.QGroupBox("Execution")
        l2 = QtWidgets.QVBoxLayout(boite2)
        l2.setContentsMargins(4, 4, 4, 4)

        # deux rangees de boutons en toutes lettres remplacees par une seule
        # barre de pictogrammes, a la maniere d'un lecteur
        self.barre_exec = QtWidgets.QToolBar()
        self.barre_exec.setIconSize(QtCore.QSize(20, 20))
        self.barre_exec.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.act_lecture = self.barre_exec.addAction(_icone("lire"), "Lancer")
        self.act_lecture.setToolTip("lancer ou mettre en pause (Espace)")
        self.act_lecture.triggered.connect(self._basculer_lecture)
        a = self.barre_exec.addAction(_icone("arreter"), "Arreter")
        a.setToolTip("arreter et revenir au debut")
        a.triggered.connect(self._arreter)
        self.barre_exec.addSeparator()
        a = self.barre_exec.addAction(_icone("pas_prec"), "Pas precedent")
        a.setToolTip("reculer d'une instruction (Ctrl+Gauche)")
        a.triggered.connect(lambda: self.pas_a_pas(-1))
        a = self.barre_exec.addAction(_icone("pas_suiv"), "Pas suivant")
        a.setToolTip("avancer d'une instruction (Ctrl+Droite)")
        a.triggered.connect(lambda: self.pas_a_pas(+1))
        self.barre_exec.addSeparator()
        self.sp_facteur = _spin(0.1, 5.0, 0.1, "x", 1, 1.0)
        self.sp_facteur.setMinimumWidth(64)
        self.sp_facteur.setToolTip("facteur de vitesse de lecture")
        self.barre_exec.addWidget(self.sp_facteur)
        l2.addWidget(self.barre_exec)

        self.chk_suivre = QtWidgets.QCheckBox(
            "se positionner sur la ligne selectionnee")
        self.chk_suivre.setChecked(True)
        self.chk_suivre.setToolTip(
            "cliquer une ligne du programme amene le robot sur sa cible")
        l2.addWidget(self.chk_suivre)

        self.curseur_temps = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.curseur_temps.setRange(0, 1000)
        self.curseur_temps.sliderMoved.connect(self._deplacer_temps)
        l2.addWidget(self.curseur_temps)
        self.lab_temps = QtWidgets.QLabel("t = 0.00 s / 0.00 s")
        l2.addWidget(self.lab_temps)
        lay.addWidget(boite2)

        dock.setWidget(w)
        dock.setMinimumWidth(330)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.docks.append(dock)

        for widget in ([self.cb_type, self.cb_mode, self.cb_profil, self.cb_temps,
                        self.cb_repere]
                       + self.champs_cible + self.champs_via
                       + [self.sp_duree, self.sp_vitesse, self.sp_accel,
                          self.sp_lissage, self.sp_port, self.chk_outil,
                          self.ed_commentaire]):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._formulaire_modifie)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.stateChanged.connect(self._formulaire_modifie)
            elif isinstance(widget, QtWidgets.QLineEdit):
                # editingFinished plutot que textChanged : sinon la liste est
                # reconstruite a chaque frappe et le curseur saute
                widget.editingFinished.connect(self._formulaire_modifie)
            else:
                widget.valueChanged.connect(self._formulaire_modifie)

        self._rafraichir_liste()

    def _creer_docks_bas(self) -> None:
        # matrices
        d1 = QtWidgets.QDockWidget("Matrices homogenes", self)
        d1.setObjectName("dock_matrices")
        self.txt_matrices = QtWidgets.QPlainTextEdit()
        self.txt_matrices.setReadOnly(True)
        self.txt_matrices.setFont(_police_fixe(9))
        self.txt_matrices.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        d1.setWidget(self.txt_matrices)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d1)

        # courbes
        d2 = QtWidgets.QDockWidget("Courbes", self)
        d2.setObjectName("dock_courbes")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        self.plot_q = pg.PlotWidget(title="Angles articulaires (deg)")
        self.plot_v = pg.PlotWidget(title="Vitesse du TCP (mm/s) et altitude z (mm)")
        for p in (self.plot_q, self.plot_v):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "temps", units="s")
        self.plot_q.addLegend(offset=(-10, 10))
        couleurs = [(230, 90, 80), (90, 200, 120), (90, 160, 240), (230, 190, 80)]
        self.courbes_q = [self.plot_q.plot(pen=pg.mkPen(couleurs[i], width=2),
                                           name="J{}".format(i + 1)) for i in range(4)]
        self.courbe_v = self.plot_v.plot(pen=pg.mkPen((60, 200, 235), width=2))
        self.courbe_z = self.plot_v.plot(pen=pg.mkPen((250, 150, 60), width=2,
                                                      style=QtCore.Qt.DashLine))
        self.ligne_t1 = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((255, 255, 255, 120)))
        self.ligne_t2 = pg.InfiniteLine(0, angle=90, pen=pg.mkPen((255, 255, 255, 120)))
        self.plot_q.addItem(self.ligne_t1)
        self.plot_v.addItem(self.ligne_t2)
        lay.addWidget(self.plot_q)
        lay.addWidget(self.plot_v)
        d2.setWidget(w)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d2)

        # entrees / sorties
        d4 = QtWidgets.QDockWidget("Entrees / Sorties", self)
        d4.setObjectName("dock_es")
        d4.setWidget(self._creer_panneau_es())
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d4)
        self._dock_es = d4

        # journal
        d3 = QtWidgets.QDockWidget("Journal", self)
        d3.setObjectName("dock_journal")
        self.txt_journal = QtWidgets.QPlainTextEdit()
        self.txt_journal.setReadOnly(True)
        self.txt_journal.setFont(_police_fixe(9))
        d3.setWidget(self.txt_journal)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, d3)

        self.docks += [d1, d2, d4, d3]
        self._dock_courbes = d2
        self.tabifyDockWidget(d1, d2)
        self.tabifyDockWidget(d2, d4)
        self.tabifyDockWidget(d4, d3)
        d1.raise_()
        self.resizeDocks([d1], [300], QtCore.Qt.Vertical)

    # -- panneau robot reel -------------------------------------------------

    def _creer_dock_reel(self) -> None:
        """
        Pilotage de la machine physique.

        Tout ce qui fait bouger le robot demande une confirmation explicite
        et porte un fond rouge : on ne declenche pas un mouvement reel par
        inadvertance en cliquant a cote.
        """
        self.pont = PontReel(self.robot, journal=self.journal)
        self._reel_alarmes_vues: List[str] = []

        dock = QtWidgets.QDockWidget("Robot reel", self)
        dock.setObjectName("dock_reel")
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(6, 6, 6, 6)

        # --- connexion
        boite = QtWidgets.QGroupBox("Liaison")
        fl = QtWidgets.QFormLayout(boite)
        ligne = QtWidgets.QHBoxLayout()
        self.cb_port = QtWidgets.QComboBox()
        self.cb_port.setToolTip("le Magician se presente en USB-SERIE CH340")
        ligne.addWidget(self.cb_port, 1)
        bt_scan = QtWidgets.QPushButton("Chercher")
        bt_scan.clicked.connect(self._reel_chercher_ports)
        ligne.addWidget(bt_scan)
        cont = QtWidgets.QWidget()
        cont.setLayout(ligne)
        fl.addRow("port", cont)

        self.bt_connecter = QtWidgets.QPushButton("Connecter")
        self.bt_connecter.clicked.connect(self._reel_basculer_connexion)
        fl.addRow("", self.bt_connecter)
        self.lab_reel_etat = QtWidgets.QLabel("hors ligne")
        self.lab_reel_etat.setWordWrap(True)
        fl.addRow("etat", self.lab_reel_etat)
        lay.addWidget(boite)

        # --- mouvement
        self.boite_reel_mvt = QtWidgets.QGroupBox("Execution reelle")
        ml = QtWidgets.QVBoxLayout(self.boite_reel_mvt)

        v = QtWidgets.QHBoxLayout()
        v.addWidget(QtWidgets.QLabel("vitesse"))
        self.sp_reel_ratio = _spin(5.0, 100.0, 5.0, "%", 0, 30.0)
        self.sp_reel_ratio.setToolTip(
            "bride TOUTES les vitesses du controleur\n"
            "30 % pour une premiere execution, 100 % une fois le programme sur")
        self.sp_reel_ratio.valueChanged.connect(self._reel_ratio)
        v.addWidget(self.sp_reel_ratio, 1)
        ml.addLayout(v)

        # Declarer la posture remplace la prise d'origine quand un capteur
        # d'angle est fautif : le controleur garde ce qu'on lui annonce et
        # compte ensuite ses pas moteur. Verifie sur firmware 3.7.0.
        self.bt_declarer = QtWidgets.QPushButton(
            "Declarer la posture d'equerre")
        self.bt_declarer.setToolTip(
            "bras arriere VERTICAL, avant-bras HORIZONTAL\n"
            "ne fait RIEN bouger : fixe l'origine sans prise d'origine\n"
            "a refaire apres chaque mise sous tension")
        self.bt_declarer.clicked.connect(self._reel_declarer)
        ml.addWidget(self.bt_declarer)

        self.bt_home = QtWidgets.QPushButton("Prise d'origine")
        self.bt_home.setToolTip("le robot parcourt toute sa course : degagez la zone")
        self.bt_home.clicked.connect(self._reel_home)
        ml.addWidget(self.bt_home)

        self.bt_televerser = QtWidgets.QPushButton("Televerser le programme")
        self.bt_televerser.setToolTip(
            "remplit la file du controleur\nle robot ne bouge pas encore\n"
            "facultatif : Lancer s'en charge tout seul")
        self.bt_televerser.clicked.connect(lambda: self._reel_televerser())
        ml.addWidget(self.bt_televerser)

        b = QtWidgets.QHBoxLayout()
        self.bt_reel_lancer = QtWidgets.QPushButton("Lancer")
        self.bt_reel_lancer.setToolTip(
            "joue le programme sur la machine\n"
            "la file du controleur se vide en s'executant : si elle est\n"
            "epuisee, le programme est televerse automatiquement")
        self.bt_reel_lancer.clicked.connect(self._reel_lancer)
        b.addWidget(self.bt_reel_lancer)
        self.bt_reel_pause = QtWidgets.QPushButton("Suspendre")
        self.bt_reel_pause.clicked.connect(self._reel_suspendre)
        b.addWidget(self.bt_reel_pause)
        ml.addLayout(b)

        self.bt_urgence = QtWidgets.QPushButton("ARRET D'URGENCE")
        self.bt_urgence.setStyleSheet(
            "QPushButton { background:#a11c1c; color:white; font-weight:bold;"
            " padding:8px; }")
        self.bt_urgence.setToolTip("coupe le mouvement en cours et vide la file")
        self.bt_urgence.clicked.connect(self._reel_urgence)
        ml.addWidget(self.bt_urgence)

        self.barre_reel = QtWidgets.QProgressBar()
        self.barre_reel.setRange(0, 100)
        self.barre_reel.setFormat("file : %p %")
        ml.addWidget(self.barre_reel)

        # Defauts de la machine. Sans cela, une alarme fige la file et la
        # barre reste plantee sans que rien ne l'explique.
        self.lab_defauts = QtWidgets.QLabel()
        self.lab_defauts.setWordWrap(True)
        self.lab_defauts.setStyleSheet(
            "QLabel { background:#7a1f1f; color:#ffe3e3; padding:6px;"
            " border:1px solid #c0392b; }")
        self.lab_defauts.hide()
        ml.addWidget(self.lab_defauts)

        self.bt_effacer_defauts = QtWidgets.QPushButton("Effacer les defauts")
        self.bt_effacer_defauts.setToolTip(
            "acquitte les alarmes du controleur\n"
            "la cause reste a corriger : le robot refusera de nouveau")
        self.bt_effacer_defauts.clicked.connect(self._reel_effacer_defauts)
        ml.addWidget(self.bt_effacer_defauts)
        lay.addWidget(self.boite_reel_mvt)

        # --- miroir
        # Les deux sens de circulation, volontairement EXCLUSIFS : si le
        # modele suivait la machine pendant que la machine suit le modele,
        # les curseurs se battraient avec la main de l'operateur.
        self.chk_miroir = QtWidgets.QCheckBox(
            "miroir : le modele 3D suit le robot reel")
        self.chk_miroir.setToolTip(
            "lit les angles de la machine et les applique au modele\n"
            "ce sont les MEMES quatre angles : aucune conversion")
        self.chk_miroir.setChecked(True)
        self.chk_miroir.toggled.connect(self._reel_exclusivite)
        lay.addWidget(self.chk_miroir)

        self.chk_piloter = QtWidgets.QCheckBox(
            "pilotage : le robot reel suit les curseurs et l'outil")
        self.chk_piloter.setStyleSheet("color:#e08a3c; font-weight:bold")
        self.chk_piloter.setToolTip(
            "chaque curseur relache envoie le robot sur la pose affichee\n"
            "articulaire comme cartesien : c'est la meme configuration\n"
            "le bouton Actionner de la section Outil commande aussi\n"
            "l'effecteur reel -- ventouse, pince ou laser\n"
            "LE ROBOT BOUGE a chaque relachement")
        self.chk_piloter.toggled.connect(self._reel_basculer_pilotage)
        lay.addWidget(self.chk_piloter)

        # Les quatre angles en gros : c'est ce qu'on regarde en recalant le
        # bras a la main, une main sur le robot et les yeux sur l'ecran.
        self.lab_angles = QtWidgets.QLabel("")
        self.lab_angles.setFont(_police_fixe(13))
        self.lab_angles.setStyleSheet(
            "QLabel { background:#1d2732; color:#cfe3ff; padding:8px;"
            " border:1px solid #38485c; }")
        lay.addWidget(self.lab_angles)

        self.chk_equerre = QtWidgets.QCheckBox(
            "aide au recalage : ecart a la posture d'equerre")
        self.chk_equerre.setToolTip(
            "affiche l'ecart de J2 et J3 a la posture de reference\n"
            "bras arriere VERTICAL, avant-bras HORIZONTAL")
        lay.addWidget(self.chk_equerre)

        self.lab_reel_pose = QtWidgets.QLabel("")
        self.lab_reel_pose.setFont(_police_fixe(9))
        self.lab_reel_pose.setWordWrap(True)
        lay.addWidget(self.lab_reel_pose)
        lay.addStretch(1)

        dock.setWidget(page)
        dock.setMinimumWidth(300)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.docks.append(dock)
        self._dock_reel = dock

        # Tabule avec PROGRAMME, pas avec Commande. Pour piloter la machine
        # a la main il faut les curseurs ET l'arret d'urgence sous les yeux :
        # les mettre dans le meme jeu d'onglets rendait l'un ou l'autre
        # invisible. L'editeur de programme, lui, ne sert pas pendant qu'on
        # pilote : c'est avec lui qu'il faut partager la place.
        programme = next((d for d in self.docks
                          if d.windowTitle() == "Programme"), None)
        if programme is not None:
            self.tabifyDockWidget(programme, dock)
            programme.raise_()

        self.minuteur_reel = QtCore.QTimer(self)
        self.minuteur_reel.setInterval(200)          # 5 Hz : ~30 ms de liaison
        self.minuteur_reel.timeout.connect(self._reel_rafraichir)

        self._reel_chercher_ports()
        self._reel_maj_boutons()

    def closeEvent(self, evenement) -> None:
        """On ne quitte jamais en laissant la machine en mouvement."""
        serveur = getattr(self, "_serveur", None)
        if serveur is not None:
            # liberer le tube nomme, sinon le prochain simulateur le
            # trouverait occupe par un fantome
            serveur.fermer()
        if getattr(self, "pont", None) is not None and self.pont.connecte:
            try:
                self.minuteur_reel.stop()
                self.pont.arret_urgence()
            except Exception:
                pass
            finally:
                self.pont.deconnecter()
        super().closeEvent(evenement)

    # -- robot reel : actions -----------------------------------------------

    def _reel_chercher_ports(self) -> None:
        from .liaison import ports_disponibles
        courant = self.cb_port.currentText()
        self.cb_port.clear()
        for device, description in ports_disponibles():
            self.cb_port.addItem("{}  -  {}".format(device, description), device)
        if courant:
            i = self.cb_port.findText(courant)
            if i >= 0:
                self.cb_port.setCurrentIndex(i)
        if not self.cb_port.count():
            self.cb_port.addItem("aucun port serie detecte", "")

    def _reel_maj_boutons(self) -> None:
        actif = self.pont.connecte
        self.bt_connecter.setText("Deconnecter" if actif else "Connecter")
        self.boite_reel_mvt.setEnabled(actif)
        self.chk_miroir.setEnabled(actif)
        self.chk_equerre.setEnabled(actif)
        self.chk_piloter.setEnabled(actif)
        if not actif and self.chk_piloter.isChecked():
            self.chk_piloter.setChecked(False)
        if not actif:
            # la machine partie, les entrees redeviennent des interrupteurs
            for case in (getattr(self, "chk_es_reelle", None),
                         getattr(self, "chk_es_entrees", None)):
                if case is not None and case.isChecked():
                    case.setChecked(False)
        self._maj_badge_reel()
        if not actif:
            self.lab_reel_pose.setText("")
            self.lab_angles.setText("hors ligne")
            self.barre_reel.setValue(0)

    def _reel_basculer_connexion(self) -> None:
        if self.pont.connecte:
            self.minuteur_reel.stop()
            self.pont.deconnecter()
            self.lab_reel_etat.setText("hors ligne")
            self._reel_maj_boutons()
            return
        port = self.cb_port.currentData() or ""
        if not port:
            QtWidgets.QMessageBox.warning(
                self, "Robot reel",
                "Aucun port serie. Branchez le Dobot, allumez-le, puis "
                "cliquez sur Chercher.")
            return
        try:
            version = self.pont.connecter(port)
        except Exception as exc:
            self.lab_reel_etat.setText("connexion impossible")
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))
            return
        # Le controleur annonce a la fois une pose et des angles : les
        # confronter verifie que sa geometrie et celle du modele coincident.
        # ATTENTION, ce controle ne dit RIEN de la calibration : pose et
        # angles viennent des memes compteurs, ils restent d'accord entre
        # eux meme si le zero a derive. Pour la prise d'origine, c'est le
        # miroir 3D qui tranche -- ou l'alarme 0x11.
        etat = "connecte sur {}, firmware {}".format(port, version)
        try:
            ecart, _ = self.pont.coherence()
            if ecart < 0.05:
                etat += ("\nmodele et controleur d'accord (ecart {:.3f} mm)"
                         .format(ecart))
            else:
                etat += ("\n<b style='color:#e0a030'>modele et controleur "
                         "divergent de {:.1f} mm : verifiez xBias et la "
                         "geometrie</b>".format(ecart))
                self.journal("robot reel : le controleur et le modele "
                             "divergent de {:.2f} mm".format(ecart))
        except Exception as exc:
            self.journal("controle de coherence impossible : {}".format(exc))
        etat += ("\nmiroir actif : comparez la posture affichee a la vraie. "
                 "Si elles different, faites une prise d'origine.")
        self.lab_reel_etat.setText(etat)

        self._reel_alarmes_vues = []
        self._reel_maj_boutons()
        self._reel_ratio()
        self.minuteur_reel.start()

    def _confirmer_mouvement(self, titre: str, texte: str) -> bool:
        """Garde-fou commun a toute action qui fait bouger la machine."""
        b = QtWidgets.QMessageBox(self)
        b.setIcon(QtWidgets.QMessageBox.Warning)
        b.setWindowTitle(titre)
        b.setText(texte)
        b.setInformativeText(
            "LE ROBOT REEL VA BOUGER.\nDegagez la zone de travail et gardez "
            "l'arret d'urgence a portee.")
        b.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        b.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        b.button(QtWidgets.QMessageBox.Ok).setText("Lancer le mouvement")
        return b.exec_() == QtWidgets.QMessageBox.Ok

    def _reel_ratio(self) -> None:
        if not self.pont.connecte:
            return
        try:
            self.pont.definir_ratio(self.sp_reel_ratio.value())
        except Exception as exc:
            self.journal("reglage du ratio impossible : {}".format(exc))

    #: posture de reference du recalage, convention du controleur verifiee
    #: sur machine reelle : J2 = 0 bras VERTICAL, J3 = 0 avant-bras HORIZONTAL
    POSTURE_EQUERRE = (0.0, 0.0)

    def _reel_declarer(self) -> None:
        """
        Fixe l'origine en DECLARANT la posture, sans prise d'origine.

        Utile quand un capteur d'angle est fautif : la prise d'origine s'y
        fie et part en butee, alors que le controleur, lui, garde ce qu'on
        lui annonce et compte ensuite ses pas moteur.
        """
        b = QtWidgets.QMessageBox(self)
        b.setIcon(QtWidgets.QMessageBox.Question)
        b.setWindowTitle("Declarer la posture")
        b.setText("Le bras est-il a la posture d'equerre ?")
        b.setInformativeText(
            "  -  bras arriere VERTICAL\n"
            "  -  avant-bras HORIZONTAL\n\n"
            "Placez-le a la main, bouton de la fourche enfonce, en "
            "verifiant a l'equerre.\n\n"
            "Rien ne bougera : cette commande ne fait que fixer l'origine. "
            "La precision du resultat est celle de votre placement.")
        b.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        b.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        b.button(QtWidgets.QMessageBox.Ok).setText("Oui, declarer")
        if b.exec_() != QtWidgets.QMessageBox.Ok:
            return
        try:
            avant = self.pont.angles_reels()
            self.pont.liaison.redefinir_pose(*self.POSTURE_EQUERRE)
            apres = self.pont.angles_reels()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))
            return
        self.journal("posture declaree : J2 {:+.2f} -> {:+.2f}, "
                     "J3 {:+.2f} -> {:+.2f}"
                     .format(avant[1], apres[1], avant[2], apres[2]))
        pris = (abs(apres[1] - self.POSTURE_EQUERRE[0]) < 0.5
                and abs(apres[2] - self.POSTURE_EQUERRE[1]) < 0.5)
        QtWidgets.QMessageBox.information(
            self, "Robot reel",
            ("Origine fixee : J2 = {:+.2f}, J3 = {:+.2f}.\n\n"
             "Le controleur compte ses pas a partir de la. A refaire apres "
             "chaque mise sous tension.".format(apres[1], apres[2]))
            if pris else
            "Le controleur n'a pas retenu les valeurs : J2 = {:+.2f}, "
            "J3 = {:+.2f}.".format(apres[1], apres[2]))

    def _reel_home(self) -> None:
        if not self._confirmer_mouvement(
                "Prise d'origine",
                "Le robot va parcourir toute sa course pour retrouver ses "
                "origines."):
            return
        try:
            self.pont.prise_origine()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))

    def _reel_file_epuisee(self) -> bool:
        """
        La file ne contient-elle plus le programme ?

        La file du controleur se CONSOMME : chaque commande executee
        disparait. Apres une execution, il n'y reste rien -- relancer sans
        televerser ne ferait donc rien du tout.
        """
        if self.pont.index_final <= 0:
            return True
        try:
            return self.pont.liaison.index_courant() >= self.pont.index_final
        except Exception:
            return True

    def _reel_televerser(self, silencieux: bool = False) -> bool:
        """
        Remplit la file du controleur. Aucun mouvement a ce stade.

        Retourne vrai si la file contient le programme.
        """
        progres = QtWidgets.QProgressDialog(
            "Televersement...", "Annuler", 0,
            max(len(self.programme.instructions), 1), self)
        progres.setWindowTitle("Robot reel")
        progres.setMinimumDuration(0)
        annule = {"oui": False}

        def avancer(k, n):
            progres.setMaximum(n)
            progres.setValue(k)
            QtWidgets.QApplication.processEvents()
            if progres.wasCanceled():
                annule["oui"] = True
                raise ErreurPont("televersement interrompu par l'operateur")

        try:
            self.pont.televerser(self.programme, self.q_depart(),
                                 progression=avancer, outil=self.outil.nom)
        except ErreurPont as exc:
            progres.close()
            if annule["oui"]:
                try:
                    self.pont.arret_urgence()
                except Exception:
                    pass
                self.journal("televersement annule, file videe")
                return False
            reponse = QtWidgets.QMessageBox.warning(
                self, "Programme refuse", str(exc),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ignore,
                QtWidgets.QMessageBox.Cancel)
            if reponse != QtWidgets.QMessageBox.Ignore:
                return False
            try:
                self.pont.televerser(self.programme, self.q_depart(),
                                     forcer=True, outil=self.outil.nom)
            except Exception as exc2:
                QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc2))
                return False
        except Exception as exc:
            progres.close()
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))
            return False
        progres.close()
        if not silencieux:
            QtWidgets.QMessageBox.information(
                self, "Robot reel",
                "Programme televerse dans la file du controleur.\n"
                "Le robot n'a pas encore bouge : cliquez sur Lancer.")
        return True

    def _reel_lancer(self) -> None:
        """
        Joue le programme sur la machine, autant de fois qu'on veut.

        La file du controleur se consomme a l'execution : apres un cycle
        elle est vide. Plutot que d'exiger un televersement manuel avant
        chaque relance -- pour un programme identique, qui plus est --, on
        la remplit ici quand elle est epuisee. Le bouton veut dire "joue ce
        programme", un point c'est tout.
        """
        if not self.pont.connecte:
            QtWidgets.QMessageBox.warning(
                self, "Robot reel", "Connectez d'abord le robot.")
            return
        a_televerser = self._reel_file_epuisee()
        if not self._confirmer_mouvement(
                "Execution reelle",
                ("Le programme va etre televerse puis execute sur la machine, "
                 "a {:.0f} % de la vitesse nominale." if a_televerser else
                 "Le programme televerse va s'executer sur la machine, a "
                 "{:.0f} % de la vitesse nominale.")
                .format(self.sp_reel_ratio.value())):
            return
        if a_televerser and not self._reel_televerser(silencieux=True):
            return
        try:
            self.pont.demarrer()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))

    def _reel_suspendre(self) -> None:
        try:
            self.pont.suspendre()
        except Exception as exc:
            self.journal("suspension impossible : {}".format(exc))

    # -- pilotage manuel du robot reel depuis les curseurs -------------------

    def _reel_exclusivite(self, miroir_actif: bool) -> None:
        """Miroir et pilotage sont deux sens opposes : jamais les deux."""
        if miroir_actif and self.chk_piloter.isChecked():
            self.chk_piloter.setChecked(False)

    def _reel_basculer_pilotage(self, actif: bool) -> None:
        if not actif:
            self._maj_badge_reel()
            self.journal("pilotage du robot reel desactive")
            return
        if not self.pont.connecte:
            self.chk_piloter.setChecked(False)
            QtWidgets.QMessageBox.warning(
                self, "Robot reel", "Connectez d'abord le robot.")
            return
        if not self._confirmer_mouvement(
                "Pilotage manuel",
                "Chaque curseur relache enverra le robot sur la pose "
                "affichee, a {:.0f} % de la vitesse nominale."
                .format(self.sp_reel_ratio.value())):
            self.chk_piloter.setChecked(False)
            return
        self.chk_miroir.setChecked(False)      # le modele devient le maitre
        self._maj_badge_reel()
        self.journal("pilotage du robot reel ACTIF : chaque curseur relache "
                     "envoie un mouvement")

    def _reel_envoyer_pose(self, _=None) -> None:
        """
        Envoie la configuration affichee au robot reel.

        On transmet les ANGLES, pas les coordonnees, et ce quel que soit le
        curseur manipule : le simulateur a deja resolu le MGI, applique la
        longueur d'outil et verifie les butees. Les angles du modele sont
        exactement ceux du controleur, donc rien ne peut se perdre en
        conversion, et le bras reproduit la posture montree en 3D.
        """
        if not self.chk_piloter.isChecked() or not self.pont.connecte:
            return
        if self._maj_en_cours or self.en_lecture:
            return

        q = np.asarray(self.q, dtype=float)
        defauts = self.robot.defauts(q)
        if defauts:
            self.journal("pilotage refuse : " + defauts[0])
            self._alerte_posturale = True
            self._alerter(["Pilotage refuse -- " + defauts[0]] + defauts[1:])
            return

        try:
            li = self.pont.liaison
            # couper et vider : un relachement annule le precedent plutot
            # que de s'empiler derriere lui
            li.arreter_file()
            li.vider_file()
            li.ptp(PTP.MOVJ_ANGLE, *(float(v) for v in q))
            li.demarrer_file()
        except Exception as exc:
            self.chk_piloter.setChecked(False)
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))
            return
        self.journal("pilotage : J = ({:.2f}, {:.2f}, {:.2f}, {:.2f}) deg"
                     .format(*q))

    def _reel_effacer_defauts(self) -> None:
        """
        Acquitte les alarmes du controleur.

        La cause, elle, n'est pas corrigee : si le robot a refuse une cible,
        il la refusera encore. On vide donc aussi la file, pour ne pas
        repartir au milieu d'un programme.
        """
        try:
            self.pont.liaison.arreter_file()
            self.pont.liaison.effacer_alarmes()
            self.pont.liaison.vider_file()
            self._reel_alarmes_vues = []
            self.lab_defauts.hide()
            self._alerter(())
            self.journal("defauts du robot reel acquittes, file videe")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))

    def _reel_urgence(self) -> None:
        try:
            self.pont.arret_urgence()
            self.journal("ARRET D'URGENCE declenche depuis l'interface")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))

    def _reel_rafraichir(self) -> None:
        """Interroge la machine et, si le miroir est actif, anime le modele."""
        try:
            etat = self.pont.etat()
        except Exception as exc:
            self.minuteur_reel.stop()
            self.lab_reel_etat.setText("liaison perdue")
            self.journal("liaison avec le robot reel perdue : {}".format(exc))
            self.pont.deconnecter()
            self._reel_maj_boutons()
            return

        x, y, z, r = etat.pose_controleur
        xs, ys, zs, rs = self.pont.depuis_controleur(x, y, z, r)

        lignes = ["{:<16}{:+8.2f} deg".format(nom, v) for nom, v
                  in zip(self.robot.NOMS_ARTICULATIONS, etat.angles)]
        if self.chk_equerre.isChecked():
            # posture de reference du recalage : J2 = 0 (bras vertical),
            # J3 = 0 (avant-bras horizontal)
            lignes.append("")
            for i, nom in ((1, "ecart J2"), (2, "ecart J3")):
                e = etat.angles[i]
                marque = "   <-- A L'EQUERRE" if abs(e) < 0.6 else ""
                lignes.append("{:<16}{:+8.2f} deg{}".format(nom, e, marque))
        self.lab_angles.setText("\n".join(lignes))

        self.lab_reel_pose.setText(
            "controleur  x={:7.1f} y={:7.1f} z={:7.1f} r={:6.1f}\n"
            "simulateur  x={:7.1f} y={:7.1f} z={:7.1f} r={:6.1f}\n"
            "file : index {} / {}"
            .format(x, y, z, r, xs, ys, zs, rs,
                    etat.index_courant, etat.index_final))
        self.barre_reel.setValue(int(round(100 * etat.progression)))

        # Un defaut du controleur FIGE la file : la barre s'arrete en cours
        # de route. Le dire haut et fort, sans quoi on cherche longtemps.
        for m in etat.alarmes:
            if m not in self._reel_alarmes_vues:
                self._reel_alarmes_vues.append(m)
                self.journal("ROBOT REEL EN DEFAUT : " + m)
        if etat.alarmes:
            self.lab_defauts.setText(
                "Robot en defaut, la file est arretee a {:.0f} %.\n{}"
                .format(100 * etat.progression, "\n".join(etat.alarmes)))
            self.lab_defauts.show()
            self._alerter(["Robot reel en defaut : " + etat.alarmes[0]]
                          + etat.alarmes[1:])
        else:
            self._reel_alarmes_vues = []
            self.lab_defauts.hide()

        self._maj_badge_reel(etat)

        if self.chk_es_entrees.isChecked():
            self._lire_entrees_reelles()

        # miroir : les angles du controleur sont ceux du modele, tels quels
        if self.chk_miroir.isChecked() and not self.en_lecture:
            self._appliquer_q(np.asarray(etat.angles, dtype=float),
                              source="reel")

    # -- panneau entrees / sorties ------------------------------------------

    STYLE_VOYANT = ("QPushButton{{background:{};color:{};border:1px solid #555;"
                    "border-radius:3px;padding:2px 0;}}")

    def _creer_panneau_es(self) -> QtWidgets.QWidget:
        """
        20 sorties pilotees par le programme, 20 entrees pilotees par vous.

        C'est la difference essentielle : une sortie est un ordre, une
        entree est une information venue de la cellule. Ici vous jouez le
        role des capteurs en basculant les entrees a la main.
        """
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 6, 8, 6)

        entete = QtWidgets.QHBoxLayout()
        entete.addWidget(QtWidgets.QLabel(
            "<b>Sorties</b> : pilotees par le programme, ou cliquez pour les "
            "forcer.  <b>Entrees</b> : cliquez pour simuler un capteur."))
        entete.addStretch(1)
        bt = QtWidgets.QPushButton("Remettre les entrees a zero")
        bt.clicked.connect(lambda: self.es.reinitialiser(entrees=True,
                                                         sorties=False))
        entete.addWidget(bt)
        lay.addLayout(entete)

        # Forcer une sortie sur la MACHINE : c'est ainsi qu'on verifie qu'une
        # electrovanne ou un voyant repond, sans ecrire de programme.
        self.chk_es_reelle = QtWidgets.QCheckBox(
            "repercuter les sorties sur le robot reel")
        self.chk_es_reelle.setStyleSheet("color:#e08a3c; font-weight:bold")
        self.chk_es_reelle.setToolTip(
            "un clic sur une sortie l'applique aussi a la machine\n"
            "le port est configure en sortie au premier usage\n"
            "ATTENTION : cela actionne le materiel branche dessus")
        self.chk_es_reelle.toggled.connect(self._basculer_es_reelle)
        lay.addWidget(self.chk_es_reelle)

        self.chk_es_entrees = QtWidgets.QCheckBox(
            "lire les entrees sur le robot reel")
        self.chk_es_entrees.setToolTip(
            "les boutons d'entree cessent d'etre des interrupteurs :\n"
            "ils affichent l'etat VRAI des capteurs de la cellule\n"
            "lecture seule, aucune commande n'est envoyee")
        self.chk_es_entrees.toggled.connect(self._basculer_entrees_reelles)
        lay.addWidget(self.chk_es_entrees)

        note = QtWidgets.QLabel(
            "L'EFFECTEUR n'est pas ici. Ventouse, pince et laser ne sont pas "
            "cables sur ces sorties : les actionner de EIO1 a EIO20 n'a aucun "
            "effet sur eux. Ils ont leur propre commande, et se pilotent a la "
            "main par le bouton <b>Actionner</b> de la section Outil, dans le "
            "panneau Commande.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa4b0; font-style:italic")
        lay.addWidget(note)

        grille = QtWidgets.QGridLayout()
        grille.setHorizontalSpacing(4)
        grille.setVerticalSpacing(3)
        grille.addWidget(QtWidgets.QLabel("sorties DO"), 0, 0)
        grille.addWidget(QtWidgets.QLabel("entrees DI"), 1, 0)

        self.voyants_sortie: List[QtWidgets.QLabel] = []
        self.boutons_entree: List[QtWidgets.QPushButton] = []
        for p in range(1, self.es.nb_ports + 1):
            infobulle = PORTS_RESERVES.get(p, "EIO{}".format(p))

            v = QtWidgets.QPushButton(str(p))
            v.setCheckable(True)
            v.setMinimumWidth(28)
            v.setMaximumWidth(44)
            v.setToolTip(infobulle + "  -  cliquez pour forcer la sortie")
            v.toggled.connect(lambda etat, port=p:
                              self._forcer_sortie(port, etat))
            grille.addWidget(v, 0, p)
            self.voyants_sortie.append(v)

            b = QtWidgets.QPushButton(str(p))
            b.setCheckable(True)
            b.setMinimumWidth(28)
            b.setMaximumWidth(44)
            b.setToolTip(infobulle + "  -  cliquez pour basculer l'entree")
            b.toggled.connect(lambda etat, port=p: self._basculer_entree(port, etat))
            grille.addWidget(b, 1, p)
            self.boutons_entree.append(b)
        grille.setColumnStretch(self.es.nb_ports + 1, 1)
        lay.addLayout(grille)

        self.lab_es = QtWidgets.QLabel()
        self.lab_es.setStyleSheet("color:#888")
        lay.addWidget(self.lab_es)
        lay.addStretch(1)

        self.es.abonner(self._rafraichir_es)
        self._rafraichir_es()
        return w

    def _rafraichir_es(self) -> None:
        """Met les voyants en accord avec l'etat des ports."""
        self._maj_es = True
        try:
            for i, v in enumerate(self.voyants_sortie):
                actif = self.es.sorties[i]
                v.setChecked(actif)
                v.setStyleSheet(self.STYLE_VOYANT.format(
                    "#d8892a" if actif else "#2c2f34",
                    "#101010" if actif else "#777"))
            for i, b in enumerate(self.boutons_entree):
                b.setChecked(self.es.entrees[i])
        finally:
            self._maj_es = False
        self.lab_es.setText(self.es.resume())

    # -- sorties sur la machine reelle --------------------------------------

    def _basculer_es_reelle(self, actif: bool) -> None:
        if not actif:
            self.journal("les sorties ne sont plus repercutees sur la machine")
            return
        if not getattr(self, "pont", None) or not self.pont.connecte:
            self.chk_es_reelle.setChecked(False)
            QtWidgets.QMessageBox.warning(
                self, "Robot reel", "Connectez d'abord le robot.")
            return
        b = QtWidgets.QMessageBox(self)
        b.setIcon(QtWidgets.QMessageBox.Warning)
        b.setWindowTitle("Sorties reelles")
        b.setText("Les sorties vont agir sur la machine.")
        b.setInformativeText(
            "Un clic sur une sortie actionnera ce qui est branche dessus : "
            "electrovanne, voyant, relais.\n\nVerifiez le cablage avant de "
            "commencer.")
        b.setStandardButtons(QtWidgets.QMessageBox.Cancel
                             | QtWidgets.QMessageBox.Ok)
        b.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if b.exec_() != QtWidgets.QMessageBox.Ok:
            self.chk_es_reelle.setChecked(False)
            return
        self._ports_sortie_configures = set()
        self.journal("les sorties sont desormais repercutees sur la machine")

    # -- entrees lues sur la machine reelle ---------------------------------

    def _basculer_entree(self, port: int, etat: bool) -> None:
        """Bascule une entree a la main, sauf si la machine les fournit."""
        if self._maj_es:
            return
        if self.chk_es_entrees.isChecked():
            return                       # ce sont les vrais capteurs qui parlent
        self.es.definir_entree(int(port), bool(etat))

    def _basculer_entrees_reelles(self, actif: bool) -> None:
        if actif and (not getattr(self, "pont", None) or not self.pont.connecte):
            self.chk_es_entrees.setChecked(False)
            QtWidgets.QMessageBox.warning(
                self, "Robot reel", "Connectez d'abord le robot.")
            return
        for b in self.boutons_entree:
            # en lecture reelle, ce ne sont plus des interrupteurs
            b.setEnabled(not actif)
        self._ports_entree_configures = set()
        self._tour_entrees = 0
        self.journal("entrees {}".format(
            "lues sur la machine" if actif else "revenues en simulation"))

    def _ports_du_programme(self) -> List[int]:
        """Ports EIO attendus par une instruction ENTREE du programme."""
        return sorted({int(i.port) for i in self.programme.instructions
                       if i.type == "ENTREE"})

    def _lire_entrees_reelles(self) -> None:
        """
        Reporte l'etat vrai des capteurs dans le panneau.

        Lire les vingt ports coute vingt allers-retours, soit 200 ms : bien
        trop pour chaque rafraichissement. On lit donc a chaque tour les
        ports ATTENDUS par le programme -- ceux qui comptent -- et on
        balaie le reste a tour de role, quelques-uns a la fois. Un port
        quelconque est ainsi rafraichi en une seconde environ, un port
        utilise cinq fois par seconde.
        """
        li = self.pont.liaison
        prioritaires = self._ports_du_programme()
        restants = [p for p in range(1, self.es.nb_ports + 1)
                    if p not in prioritaires]
        if restants:
            debut = self._tour_entrees % len(restants)
            tour = restants[debut:debut + 3]
            self._tour_entrees = debut + 3
        else:
            tour = []

        deja = getattr(self, "_ports_entree_configures", set())
        for p in prioritaires + tour:
            if p in getattr(self, "_ports_sortie_configures", set()):
                continue             # ce port sert de sortie : on n'y touche pas
            try:
                if p not in deja:
                    li.configurer_port(p, FonctionES.ENTREE)
                    deja.add(p)
                self.es.definir_entree(p, li.entree(p))
            except Exception as exc:
                self.chk_es_entrees.setChecked(False)
                self.journal("lecture des entrees interrompue : {}".format(exc))
                return
        self._ports_entree_configures = deja

    def _forcer_sortie(self, port: int, etat: bool) -> None:
        """
        Force une sortie a la main, pour verifier qu'elle repond.

        C'est la facon la plus simple de tester un cablage : pas besoin
        d'ecrire un programme ni de lancer un cycle.
        """
        if self._maj_es:
            return                       # c'est le programme qui ecrit
        self.es.definir_sortie(int(port), bool(etat))
        if not self.chk_es_reelle.isChecked() or not self.pont.connecte:
            return
        try:
            li = self.pont.liaison
            if port not in getattr(self, "_ports_sortie_configures", set()):
                li.configurer_port(int(port), FonctionES.SORTIE)
                self._ports_sortie_configures = \
                    getattr(self, "_ports_sortie_configures", set()) | {port}
            # immediate, pas en file : on veut l'effet tout de suite
            li.echanger(CommandeDobot.ES_SORTIE,
                        struct.pack("<2B", int(port), 1 if etat else 0),
                        ecriture=True)
        except Exception as exc:
            self.chk_es_reelle.setChecked(False)
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))
            return
        self.journal("sortie reelle EIO{} = {}".format(port, int(bool(etat))))

    # -- outil ---------------------------------------------------------------

    def _changer_outil(self, _index: int = 0) -> None:
        nom = self.cb_outil.currentData() or "aucun"
        self.outil = ou.outil(nom)
        self.programme.outil = self.outil.nom
        # changer d'outil a la main revient a la longueur du catalogue :
        # une longueur mesuree valait pour l'outil precedent
        self.programme.longueur_outil = self.outil.longueur
        self.robot.geo.lt = self.outil.longueur
        self._maj_outil()
        self.vue.invalider_enveloppe()
        self._recompiler()
        # lt a change : la pose affichee designe desormais la pointe
        self._appliquer_q(self.q, source="outil")
        self.journal("outil monte : {} (longueur {:.0f} mm)"
                     .format(self.outil.libelle, self.outil.longueur))

    def _maj_outil(self) -> None:
        """Reconstruit le maillage de l'outil et rafraichit son bandeau."""
        self.vue.definir_outil(self.outil.pieces(self.outil_actif))
        self.lab_longueur.setText("{:.0f} mm  (deport lt du modele)"
                                  .format(self.outil.longueur))
        self.lab_outil.setText(self.outil.description)
        self.bt_outil.setEnabled(self.outil.commandable)
        self._maj_es = True
        try:
            self.bt_outil.setChecked(bool(self.outil_actif))
        finally:
            self._maj_es = False
        self.bt_outil.setText("Outil : " + self.outil.etat_texte(self.outil_actif))

    # -- reperes outil et objet ---------------------------------------------

    @staticmethod
    def _resume_situation(T) -> str:
        """
        Six composantes, pas la matrice : c'est la forme d'affichage.

        Format serre a dessein : en police fixe, chaque caractere de trop
        elargit le panneau Commande, et au-dela de 40 il y faisait
        apparaitre une barre de defilement horizontale. Les largeurs
        retenues suffisent : 7 chiffres pour un millimetre signe
        (-340.00), 6 pour un angle (180.00).
        """
        v = vers_xyzrpy(T)
        return ("X {:7.2f}  Y {:7.2f}  Z {:7.2f} mm\n"
                "Rx {:6.2f} Ry {:6.2f} Rz {:6.2f} deg".format(*v))

    def _maj_reperes(self) -> None:
        # L'annotation va sur SA PROPRE LIGNE. Accolee aux valeurs, elle
        # portait la ligne a 74 caracteres en police fixe : le dock entier
        # s'elargissait a 614 px, une barre de defilement horizontale
        # apparaissait, et les champs de saisie des curseurs se retrouvaient
        # hors d'atteinte de la souris.
        suffixe = "" if self.robot.outil_est_par_defaut else "\npersonnalise"
        self.lab_rep_outil.setText(
            self._resume_situation(self.robot.repere_outil) + suffixe)
        identite = np.allclose(self.programme.repere_objet, np.eye(4))
        self.lab_rep_objet.setText(
            self._resume_situation(self.programme.repere_objet)
            + ("\nidentite : cibles dans la base" if identite else ""))

    def _definir_repere_outil(self) -> None:
        from .robot import verifier_repere_outil
        dlg = DialogueRepere(
            self, "Repere outil",
            "Transformation de l'axe de rotation de l'outil vers sa pointe. "
            "Un deport lateral et une rotation propre sont realisables ; une "
            "inclinaison ne l'est pas, le parallelogramme maintenant l'axe "
            "outil vertical.",
            self.robot.repere_outil, {3: 180.0, 4: 0.0},
            verifier_repere_outil)
        if (dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()):
            self.robot.definir_repere_outil(dlg.matrice())
            self._apres_changement_repere("repere outil redefini")

    def _reinit_repere_outil(self) -> None:
        self.robot.definir_repere_outil(None)
        self._apres_changement_repere("repere outil : retour au deport de l'outil monte")

    def _definir_repere_objet(self) -> None:
        from .robot import verifier_repere_objet
        dlg = DialogueRepere(
            self, "Repere objet",
            "Transformation de la base vers le plan de travail. Les "
            "instructions dont le repere vaut <i>objet</i> y sont exprimees. "
            "Seules une translation et une rotation autour de z sont admises.",
            self.programme.repere_objet, {3: 0.0, 4: 0.0},
            verifier_repere_objet)
        if (dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()):
            self.programme.definir_repere_objet(dlg.matrice())
            self._apres_changement_repere("repere objet redefini")

    def _reinit_repere_objet(self) -> None:
        self.programme.definir_repere_objet(None)
        self._apres_changement_repere("repere objet : identite")

    def _apres_changement_repere(self, message: str) -> None:
        self._maj_reperes()
        self.vue.invalider_enveloppe()
        self._recompiler()
        self._appliquer_q(self.q, source="repere")
        self.journal(message)

    def _saisir_cible(self) -> None:
        """Saisie de la cible courante en (X, Y, Z, Rx, Ry, Rz)."""
        from .robot import decomposer_cible
        ins = self._instruction_courante()
        if ins is None or ins.type not in ("MOVJ", "MOVL", "MOVC"):
            return
        dlg = DialogueRepere(
            self, "Cible de l'instruction {}".format(self.liste.currentRow() + 1),
            "Situation de la cible dans le repere <b>{}</b>. Rx et Ry sont "
            "imposes : le parallelogramme maintient l'axe de l'outil vertical, "
            "dirige vers le bas. Seuls X, Y, Z et Rz sont commandables."
            .format(ins.repere),
            self.robot.mgd(ins.q) if self._articulaire(ins)
            else cible(*ins.pose),
            {3: 180.0, 4: 0.0}, decomposer_cible)
        if not (dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()):
            return
        x, y, z, r = decomposer_cible(dlg.matrice())
        self._maj_en_cours = True
        try:
            self.cb_mode.setCurrentText("cartesien")
            for s, v in zip(self.champs_cible, (x, y, z, r)):
                s.setValue(float(v))
        finally:
            self._maj_en_cours = False
        self._formulaire_modifie()

    def _actionner_outil(self, etat: bool) -> None:
        if self._maj_es:
            return
        self._definir_etat_outil(bool(etat))
        self._reel_actionner_outil(bool(etat))

    def _reel_actionner_outil(self, etat: bool) -> None:
        """
        Actionne l'effecteur de la MACHINE.

        La ventouse, la pince et le laser ne sont PAS cables sur les sorties
        EIO generales : agir sur EIO1..EIO20 n'a aucun effet sur eux. Ils ont
        leur propre commande dans le protocole -- SetEndEffectorSuctionCup,
        Gripper, Laser -- qui pilote le circuit d'effecteur. C'est par la
        qu'il faut passer.

        Envoi IMMEDIAT, pas en file : on veut entendre la pompe tout de
        suite, pas a la fin d'un cycle.
        """
        if not self.chk_piloter.isChecked():
            return
        if not getattr(self, "pont", None) or not self.pont.connecte:
            return
        if not self.outil.commandable:
            self.journal("outil '{}' : aucun actionneur a commander"
                         .format(self.outil.nom))
            return

        identifiants = {"ventouse": CommandeDobot.VENTOUSE,
                        "pince": CommandeDobot.PINCE,
                        "laser": CommandeDobot.LASER}
        ident = identifiants.get(self.outil.nom)
        if ident is None:
            return
        try:
            self.pont.liaison.echanger(
                ident, struct.pack("<2B", 1, 1 if etat else 0), ecriture=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Robot reel", str(exc))
            return
        self.journal("outil reel : {} {}".format(
            self.outil.libelle, self.outil.etat_texte(etat)))

    def _definir_etat_outil(self, etat: bool) -> None:
        if etat == self.outil_actif:
            return
        self.outil_actif = etat
        self._maj_outil()

    def _creer_menus(self) -> None:
        m = self.menuBar()

        fichier = m.addMenu("&Fichier")
        self._action(fichier, "Nouveau programme", self._nouveau, "Ctrl+N")
        self._action(fichier, "Ouvrir...", self._ouvrir, "Ctrl+O")
        self._action(fichier, "Enregistrer sous...", self._enregistrer, "Ctrl+S")
        fichier.addSeparator()
        self._action(fichier, "Executer un script Python...", self._executer_script,
                     "Ctrl+R")
        self._action(fichier, "Rejouer le dernier script", self._rejouer_script,
                     "F5")
        fichier.addSeparator()
        self._action(fichier, "Exporter un script Dobot (DobotDllType)...",
                     self._exporter_python)
        self._action(fichier, "Exporter les consignes en CSV...", self._exporter_csv)
        fichier.addSeparator()
        self._action(fichier, "Quitter", self.close, "Ctrl+Q")

        # Fenetres : sans cela, un panneau ferme par sa croix serait perdu.
        # toggleViewAction() donne une case a cocher qui suit l'etat reel.
        fen = m.addMenu("Fe&netres")
        for dock in self.docks:
            fen.addAction(dock.toggleViewAction())
        fen.addSeparator()
        self._action(fen, "Agrandir les courbes", self._agrandir_courbes,
                     "Ctrl+Shift+C")
        self._action(fen, "Disposition par defaut", self._disposition_defaut)
        self._action(fen, "Tout reafficher", self._reafficher_docks,
                     "Ctrl+Shift+R")

        aff = m.addMenu("&Affichage")
        self.act_maillages = self._bascule(aff, "Maillages CAO", True,
                                           self.vue.afficher_maillages)
        self.act_triedre = self._bascule(
            aff, "Triedre du repere de base  (x rouge, y vert, z bleu)", True,
            self.vue.afficher_triedre_base)
        self.act_reperes = self._bascule(aff, "Reperes des liaisons", False,
                                         self.vue.afficher_reperes)
        self.act_cibles = self._bascule(aff, "Cibles du programme", True,
                                        self._afficher_cibles)
        self.act_reperes_cibles = self._bascule(
            aff, "Reperes des cibles", False, self._afficher_reperes_cibles)
        self.act_squelette = self._bascule(aff, "Squelette", True,
                                           self.vue.afficher_squelette)
        self.act_grille = self._bascule(aff, "Grille", True, self.vue.afficher_grille)
        self.act_enveloppe = self._bascule(aff, "Espace de travail", False,
                                           self.vue.afficher_enveloppe)
        self.act_trace = self._bascule(aff, "Trace du TCP", True, lambda v: None)
        aff.addSeparator()
        self._action(aff, "Zoom avant", lambda: self.vue.zoomer(1.0 / 1.25),
                     "Ctrl++")
        self._action(aff, "Zoom arriere", lambda: self.vue.zoomer(1.25), "Ctrl+-")
        self._action(aff, "Recadrer", self.vue.recadrer, "Ctrl+0")
        aff.addSeparator()
        for nom in ("iso", "face", "cote", "dessus", "arriere"):
            self._action(aff, "Vue " + nom, lambda _=False, n=nom: self.vue.vue(n))
        aff.addSeparator()
        self._action(aff, "Basculer modele CAO / simplifie", self._basculer_modele)
        self._action(aff, "Calibrer le modele 3D...", self._calibrer)
        self._action(aff, "Effacer la trace", self.vue.effacer_trace)

        exe = m.addMenu("&Execution")
        self._action(exe, "Lancer / Pause", self._basculer_lecture, "Space")
        self._action(exe, "Arreter", self._arreter, "Ctrl+.")
        exe.addSeparator()
        self._action(exe, "Pas suivant", lambda: self.pas_a_pas(+1),
                     "Ctrl+Right")
        self._action(exe, "Pas precedent", lambda: self.pas_a_pas(-1),
                     "Ctrl+Left")
        self._action(exe, "Aller a la ligne selectionnee",
                     lambda: self.aller_a_instruction(self.liste.currentRow()),
                     "Ctrl+Return")
        exe.addSeparator()
        self._action(exe, "Robot reel : afficher le panneau",
                     lambda: (self._dock_reel.show(), self._dock_reel.raise_()))
        self._action(exe, "Robot reel : ARRET D'URGENCE", self._reel_urgence,
                     "Ctrl+Shift+Space")
        exe.addSeparator()
        self.act_securite = QAction("Arret de securite (butees et collisions)",
                                    self)
        self.act_securite.setCheckable(True)
        self.act_securite.setChecked(True)
        self.act_securite.setToolTip(
            "coupe l'animation des que le bras sort de ses butees ou entre en "
            "collision\ndecocher pour laisser la faute se produire et "
            "l'observer")
        exe.addAction(self.act_securite)

        aide = m.addMenu("&Aide")
        self._action(aide, "Aide-memoire matrices homogenes", self._aide_matrices)
        self._action(aide, "A propos", self._a_propos)

    def _afficher_cibles(self, actif: bool) -> None:
        """Les trieds suivent les marqueurs : masquer l'un masque l'autre."""
        self.vue.afficher_cibles(bool(actif))
        trieds = getattr(self, "act_reperes_cibles", None)
        if actif and trieds is not None:
            self.vue.afficher_reperes_cibles(trieds.isChecked())

    def _afficher_reperes_cibles(self, actif: bool) -> None:
        marqueurs = getattr(self, "act_cibles", None)
        visible = bool(actif) and (marqueurs is None or marqueurs.isChecked())
        self.vue.afficher_reperes_cibles(visible)

    def _agrandir_courbes(self) -> None:
        """
        Met les courbes au premier plan et leur donne la moitie de la hauteur.

        Le formulaire d'instruction est replie au passage : c'est lui qui
        prenait la place, et on ne l'utilise pas pendant qu'on lit un trace.
        """
        d = self._dock_courbes
        if not d.isVisible():
            d.show()
        d.raise_()
        self.boite_instruction.setChecked(False)
        self.resizeDocks([d], [int(self.height() * 0.55)], QtCore.Qt.Vertical)
        self.journal("courbes agrandies ; formulaire replie "
                     "(Fenetres > Disposition par defaut pour revenir)")

    def _disposition_defaut(self) -> None:
        """Revient aux proportions de depart."""
        self.boite_instruction.setChecked(True)
        self._proportions_initiales()
        self.resizeDocks([self._dock_courbes], [300], QtCore.Qt.Vertical)
        self.journal("disposition par defaut retablie")

    def _reafficher_docks(self) -> None:
        """Rouvre tous les panneaux fermes, sans toucher a leur disposition."""
        caches = [d for d in self.docks if not d.isVisible()]
        for d in caches:
            d.show()
            d.raise_()
        self.journal("panneaux reaffiches : {}".format(
            ", ".join(d.windowTitle() for d in caches) if caches
            else "aucun n'etait ferme"))

    def _action(self, menu, texte, slot, raccourci=""):
        a = QAction(texte, self)
        if raccourci:
            a.setShortcut(raccourci)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _bascule(self, menu, texte, actif, slot):
        a = QAction(texte, self)
        a.setCheckable(True)
        a.setChecked(actif)
        a.toggled.connect(slot)
        menu.addAction(a)
        slot(actif)
        return a

    # -- maillages ----------------------------------------------------------

    def _charger_maillages(self) -> None:
        try:
            pieces = mm.charger_pieces(self.robot, self.config_modele,
                                       journal=self.journal)
        except Exception:                     # pragma: no cover
            self.journal(traceback.format_exc())
            pieces = []
        if not pieces:
            self.journal("modele CAO indisponible : bascule sur le modele simplifie")
            pieces = mm.modele_procedural(self.robot)
            self._pieces_cao = []
        else:
            self._pieces_cao = pieces
        self.vue.definir_pieces(pieces)
        self._maj_outil()
        self._maj_reperes()
        self.vue.maj(self.q)

    def _basculer_modele(self) -> None:
        actuel = getattr(self, "_modele_simplifie", False)
        self._modele_simplifie = not actuel
        if self._modele_simplifie or not self._pieces_cao:
            self.vue.definir_pieces(mm.modele_procedural(self.robot))
            self.journal("affichage du modele simplifie")
        else:
            self.vue.definir_pieces(self._pieces_cao)
            self.journal("affichage du modele CAO")
        self.vue.maj(self.q)

    def _calibrer(self) -> None:
        dlg = DialogueCalibrage(self, self.config_modele)

        def appliquer():
            self.config_modele = dlg.lire()
            self._charger_maillages()

        def enregistrer():
            appliquer()
            try:
                with open(mm.chemin_config_defaut(), "w", encoding="utf-8") as f:
                    import json
                    json.dump(self.config_modele, f, indent=2, ensure_ascii=False)
                self.journal("configuration enregistree dans model_config.json")
            except Exception as exc:
                self.journal("enregistrement impossible : {}".format(exc))

        dlg.b_appliquer.clicked.connect(appliquer)
        dlg.b_enregistrer.clicked.connect(enregistrer)
        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()

    # -- pilotage -----------------------------------------------------------

    def _depuis_curseurs(self, _=None) -> None:
        if self._maj_en_cours:
            return
        q = np.array([c.valeur() for c in self.curseurs], dtype=float)
        self._appliquer_q(q, source="curseurs")

    def _ecrire_cartesien(self, pose) -> None:
        """Recopie une pose (x, y, z, r) dans les curseurs cartesiens."""
        self._maj_en_cours = True
        try:
            for cle, v in zip(("x", "y", "z", "r"), pose):
                self.champs_cart[cle].definir(float(v))
        finally:
            self._maj_en_cours = False
        self._signaler_cartesien(False)

    def _signaler_cartesien(self, probleme: bool) -> None:
        for c in self.champs_cart.values():
            c.signaler(probleme)

    def _depuis_cartesien(self, _=None, force: bool = False) -> None:
        if self._maj_en_cours:
            return
        if not force and not self.chk_suivi.isChecked():
            return
        x = self.champs_cart["x"].valeur()
        y = self.champs_cart["y"].valeur()
        z = self.champs_cart["z"].valeur()
        r = self.champs_cart["r"].valeur()
        try:
            q = self.robot.mgi_pose(x, y, z, r)
        except ErreurCinematique as exc:
            self._signaler_cartesien(True)
            self.etat_config.setText(
                "<b style='color:#e06c5a'>MGI impossible</b><br>{}<br>"
                "<i>le robot reste a sa position precedente</i>".format(exc))
            return
        self._signaler_cartesien(False)
        self._appliquer_q(q, source="cartesien")

    def _appliquer_q(self, q, source: str = "") -> None:
        q = np.asarray(q, dtype=float)
        self.q = q
        self._maj_en_cours = True
        try:
            if source != "curseurs":
                for i, c in enumerate(self.curseurs):
                    c.definir(q[i])
            if source != "cartesien":
                for cle, v in zip(("x", "y", "z", "r"), self.robot.pose(q)):
                    self.champs_cart[cle].definir(float(v))
                self._signaler_cartesien(False)
        finally:
            self._maj_en_cours = False

        self.vue.maj(q, enregistrer_trace=(self.en_lecture
                                           and self.act_trace.isChecked()))
        self._maj_matrices()
        self._maj_etat()

    def _maj_etat(self) -> None:
        depass = self.robot.depassements(self.q)
        self._effacer_alerte_posturale(depass)
        p = self.robot.pose(self.q)
        w = self.robot.manipulabilite(self.q)
        lignes = ["<b>TCP</b> x={:.1f}  y={:.1f}  z={:.1f}  r={:.1f} deg"
                  .format(*p),
                  "portee horizontale = {:.1f} mm".format(float(np.hypot(p[0], p[1]))),
                  "manipulabilite = {:.3g}".format(w)]
        choc = self.robot.collisions(self.q)
        if depass:
            lignes.append("<b style='color:#e06c5a'>butees depassees</b><br>"
                          + "<br>".join(depass))
        if choc:
            lignes.append("<b style='color:#e06c5a'>collision</b><br>"
                          + "<br>".join(choc))
        if not depass and not choc:
            lignes.append("<span style='color:#5aa469'>configuration admissible</span>")
        self.etat_config.setText("<br>".join(lignes))

    def _effacer_alerte_posturale(self, depassements) -> None:
        """
        Retire l'alerte des que la pose redevient saine.

        Un arret de securite laisse un bandeau rouge. Une fois le bras
        ramene dans ses butees -- au curseur, au pas a pas ou a la main --
        ce bandeau ne decrit plus rien : il doit s'effacer tout seul, sans
        obliger a recompiler le programme.

        On ne rend la main qu'aux avertissements du PROGRAMME, qui eux
        restent valables : ils parlent de la trajectoire, pas de l'instant.
        """
        if not getattr(self, "_alerte_posturale", False):
            return
        if depassements or self.robot.collisions(self.q):
            return
        self._alerte_posturale = False
        self._alerter(getattr(self, "_avert_programme", []))
        self.journal("pose revenue dans les butees : alerte effacee")

    def _maj_matrices(self) -> None:
        rel = self.robot.matrices_liaisons(self.q)
        abso = self.robot.reperes(self.q)
        noms_rel = ["T01 = Rz(q1)",
                    "T12 = Tz(d1).Ry(q2)",
                    "T23 = Tz(l1).Ry(q3-q2+90)",
                    "T34 = Tz(l2).Ry(-90-q3)",
                    "T45 = Tx(l3).Rz(q4)",
                    "T5T = Tz(-lt).Rx(180)"]
        blocs = ["q = ({:.2f}, {:.2f}, {:.2f}, {:.2f}) deg".format(*self.q), ""]
        for nom, T in zip(noms_rel, rel):
            blocs.append(hg.formater(T, 3, nom))
            blocs.append("")
        blocs.append(hg.formater(abso[-1], 3,
                                 "T0_TCP = T01.T12.T23.T34.T45.T5T   (MGD)"))
        blocs.append("")
        J = self.robot.jacobienne(self.q)
        blocs.append("Jacobienne  [vx vy vz wz]^T = J . [dq1 dq2 dq3 dq4]^T")
        for i in range(4):
            blocs.append("[" + " ".join("{:>10.3f}".format(J[i, j])
                                        for j in range(4)) + " ]")
        blocs.append("det(J) = {:.4g}   manipulabilite = {:.4g}"
                     .format(float(np.linalg.det(J)),
                             self.robot.manipulabilite(self.q)))
        self.txt_matrices.setPlainText("\n".join(blocs))

    # -- programme ----------------------------------------------------------

    def _rafraichir_liste(self, selection: Optional[int] = None) -> None:
        ligne = self.liste.currentRow() if selection is None else selection
        self.liste.blockSignals(True)
        self.liste.clear()
        for i, ins in enumerate(self.programme.instructions):
            item = QtWidgets.QListWidgetItem(
                _icone(ins.type),
                "{:02d}  {}".format(i + 1, ins.resume(self.outil)))
            item.setToolTip(AIDE_INSTRUCTION.get(ins.type, ""))
            self.liste.addItem(item)
        if 0 <= ligne < self.liste.count():
            self.liste.setCurrentRow(ligne)
        elif self.liste.count():
            self.liste.setCurrentRow(0)
        self.liste.blockSignals(False)
        # rafraichissement interne : la selection ne doit pas faire bouger
        # le robot, sinon la moindre edition le deplacerait
        self._selection_programmee = True
        try:
            self._selection_changee(self.liste.currentRow())
        finally:
            self._selection_programmee = False

    def _instruction_courante(self) -> Optional[Instruction]:
        i = self.liste.currentRow()
        if 0 <= i < len(self.programme.instructions):
            return self.programme.instructions[i]
        return None

    def _selection_changee(self, _ligne: int) -> None:
        ins = self._instruction_courante()
        if ins is None:
            return
        self._maj_en_cours = True
        try:
            self.cb_type.setCurrentText(ins.type)
            self.cb_mode.setCurrentText(ins.mode)
            self.cb_profil.setCurrentText(ins.profil)
            self.cb_temps.setCurrentText(ins.mode_temps)
            self.cb_repere.setCurrentText(ins.repere)
            self.sp_duree.setValue(ins.duree)
            self.sp_vitesse.setValue(ins.vitesse_effective())
            self.sp_accel.setValue(ins.acceleration_effective())
            self.sp_lissage.setValue(ins.lissage)
            self.sp_port.setValue(int(ins.port))
            self.chk_outil.setChecked(bool(ins.etat))
            self.ed_commentaire.setText(ins.commentaire)
            self._ecrire_cible(ins)
            self._ecrire_via(ins)
            self._adapter_formulaire(ins)
        finally:
            self._maj_en_cours = False

        # selection faite a la main : on amene le robot sur la cible de la
        # ligne, ce qui rend le programme inspectable instruction par
        # instruction sans avoir a le rejouer
        if (not self._selection_programmee and not self.en_lecture
                and self.chk_suivre.isChecked()):
            self.aller_a_instruction(_ligne)

    # -- correspondance champs <-> modele -----------------------------------
    #
    # L'affichage est en (X, Y, Z, Rx, Ry, Rz) ; le modele ne stocke que
    # (x, y, z, r) parce que Rx et Ry sont imposes par la mecanique. Ces
    # quatre fonctions font la traduction, dans un seul endroit.

    @staticmethod
    def _articulaire(ins: Instruction) -> bool:
        return ins.mode == "articulaire" and ins.type != "MOVC"

    def _ecrire_cible(self, ins: Instruction) -> None:
        if self._articulaire(ins):
            valeurs = list(ins.q) + [0.0, 0.0]
        else:
            x, y, z, r = ins.pose
            valeurs = [x, y, z, 180.0, 0.0, r]
        for s, v in zip(self.champs_cible, valeurs):
            s.setValue(float(v))

    def _lire_cible(self, ins: Instruction) -> list:
        v = [float(s.value()) for s in self.champs_cible]
        return v[:4] if self._articulaire(ins) else [v[0], v[1], v[2], v[5]]

    def _ecrire_via(self, ins: Instruction) -> None:
        x, y, z, r = ins.via
        for s, v in zip(self.champs_via, (x, y, z, 180.0, 0.0, r)):
            s.setValue(float(v))

    def _lire_via(self) -> list:
        v = [float(s.value()) for s in self.champs_via]
        return [v[0], v[1], v[2], v[5]]

    def _adapter_formulaire(self, ins: Instruction) -> None:
        mouvement = ins.type in ("MOVJ", "MOVL", "MOVC")
        articulaire = self._articulaire(ins)
        if articulaire:
            libelles = [("J1", "deg"), ("J2", "deg"), ("J3", "deg"),
                        ("J4", "deg"), ("", ""), ("", "")]
        else:
            libelles = list(COMPOSANTES)
        for k, (lab, s) in enumerate(zip(self.etiq_cible, self.champs_cible)):
            nom, unite = libelles[k]
            visible = bool(nom)
            lab.setVisible(visible)
            s.setVisible(visible)
            lab.setText(nom)
            s.setSuffix((" " + unite) if unite else "")
            # Rx et Ry ne sont pas commandables : lecture seule, en gris
            impose = (not articulaire) and k in (3, 4)
            s.setReadOnly(impose)
            s.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons if impose
                               else QtWidgets.QAbstractSpinBox.UpDownArrows)
            s.setStyleSheet("color:#888" if impose else "")
            lab.setStyleSheet("color:#888" if impose else "")
            s.setToolTip("impose par le parallelogramme : non commandable"
                         if impose else "")
            s.setEnabled(mouvement)
        self.bloc_via.setEnabled(ins.type == "MOVC")
        self.cb_repere.setEnabled(mouvement and ins.mode == "cartesien")
        self.cb_mode.setEnabled(ins.type in ("MOVJ", "MOVL"))
        self.cb_profil.setEnabled(mouvement)
        self.chk_outil.setEnabled(ins.type == "OUTIL")

        # cadence : soit on impose la duree, soit on impose vitesse et
        # acceleration et la duree en decoule
        par_vitesse = mouvement and ins.mode_temps == "vitesse"
        self.cb_temps.setEnabled(mouvement)
        self.sp_vitesse.setEnabled(par_vitesse)
        self.sp_accel.setEnabled(par_vitesse)
        self.sp_lissage.setEnabled(mouvement)

        es = ins.type in ("SORTIE", "ENTREE")
        self.sp_port.setEnabled(es)
        self.lab_port.setText("port EIO" if es else "port EIO (sans objet)")
        self.chk_outil.setEnabled(ins.type in ("OUTIL", "SORTIE", "ENTREE"))
        if ins.type == "ENTREE":
            self.chk_outil.setText("attendre l'etat actif (1)")
        elif ins.type == "SORTIE":
            self.chk_outil.setText("mettre la sortie a 1")
        else:
            self.chk_outil.setText("outil actif (" + self.outil.libelle_actif + ")")
        self.sp_duree.setEnabled(ins.type in ("ATTENTE", "ENTREE")
                                 or (mouvement and ins.mode_temps != "vitesse"))

        # un MOVJ interpole des angles, un MOVL ou un MOVC des millimetres
        u = ins.unite
        self.lab_vitesse.setText("vitesse max ({}/s)".format(u))
        self.lab_accel.setText("acceleration ({}/s2)".format(u))
        self.sp_vitesse.setSuffix(" {}/s".format(u))
        self.sp_accel.setSuffix(" {}/s2".format(u))
        self._maj_duree_calculee()

    def _maj_duree_calculee(self) -> None:
        """Affiche la duree et la longueur reellement obtenues pour la ligne."""
        i = self.liste.currentRow()
        traj = self.trajectoire
        if traj is None or not (0 <= i < len(traj.segments)):
            self.lab_duree_calc.setText("")
            return
        seg = traj.segments[i]
        txt = "chemin {:.1f} {} parcouru en {:.2f} s".format(
            seg.longueur, seg.unite, seg.duree)
        if i < len(traj.recouvrements) and traj.recouvrements[i] > 1e-9:
            txt += "   -   raccord lisse sur {:.2f} s".format(traj.recouvrements[i])
        self.lab_duree_calc.setText(txt)

    def _formulaire_modifie(self, *_):
        if self._maj_en_cours:
            return
        ins = self._instruction_courante()
        if ins is None:
            return
        unite_avant = ins.unite
        ins.type = self.cb_type.currentText()
        ins.mode = self.cb_mode.currentText()
        ins.profil = self.cb_profil.currentText()
        ins.mode_temps = self.cb_temps.currentText()
        ins.repere = self.cb_repere.currentText()
        ins.duree = float(self.sp_duree.value())
        ins.vitesse = float(self.sp_vitesse.value())
        ins.acceleration = float(self.sp_accel.value())
        ins.lissage = float(self.sp_lissage.value())
        ins.port = int(self.sp_port.value())
        ins.etat = bool(self.chk_outil.isChecked())
        ins.commentaire = self.ed_commentaire.text()
        valeurs = self._lire_cible(ins)
        if self._articulaire(ins):
            ins.q = valeurs
        else:
            ins.pose = valeurs
        ins.via = self._lire_via()

        if ins.unite != unite_avant:
            # on est passe des degres aux millimetres ou l'inverse : les
            # anciennes consignes n'ont plus de sens, on reprend les defauts
            ins.vitesse = 0.0
            ins.acceleration = 0.0
            self._maj_en_cours = True
            try:
                self.sp_vitesse.setValue(ins.vitesse_effective())
                self.sp_accel.setValue(ins.acceleration_effective())
            finally:
                self._maj_en_cours = False

        self._adapter_formulaire(ins)
        self._rafraichir_liste()
        self._recompiler()

    def _ajouter_instruction(self, type_ins: str = "MOVJ") -> None:
        if type_ins not in TYPES_INSTRUCTION:
            type_ins = "MOVJ"
        p = self.robot.pose(self.q)
        # une ATTENTE n'a pas de chemin : sa duree est forcement imposee
        ins = Instruction(type=type_ins, mode="cartesien",
                          pose=[float(v) for v in p],
                          q=[float(v) for v in self.q],
                          mode_temps=("duree" if type_ins == "ATTENTE"
                                      else "vitesse"),
                          duree=2.0)
        i = self.liste.currentRow() + 1
        self.programme.ajouter(ins, i)
        self._rafraichir_liste(i)
        self._recompiler()

    def _dupliquer(self) -> None:
        ins = self._instruction_courante()
        if ins is None:
            return
        import copy
        i = self.liste.currentRow() + 1
        self.programme.ajouter(copy.deepcopy(ins), i)
        self._rafraichir_liste(i)
        self._recompiler()

    def _supprimer(self) -> None:
        i = self.liste.currentRow()
        if i < 0:
            return
        self.programme.supprimer(i)
        self._rafraichir_liste(min(i, len(self.programme.instructions) - 1))
        self._recompiler()

    def _deplacer(self, delta: int) -> None:
        i = self.liste.currentRow()
        j = self.programme.deplacer(i, delta)
        self._rafraichir_liste(j)
        self._recompiler()

    def _capturer_pose(self) -> None:
        ins = self._instruction_courante()
        if ins is None:
            return
        self._maj_en_cours = True
        try:
            if self._articulaire(ins):
                valeurs = list(self.q) + [0.0, 0.0]
            else:
                x, y, z, r = self.robot.pose(self.q)
                valeurs = [x, y, z, 180.0, 0.0, r]
            for s, v in zip(self.champs_cible, valeurs):
                s.setValue(float(v))
        finally:
            self._maj_en_cours = False
        self._formulaire_modifie()

    # -- compilation et animation -------------------------------------------

    def _recompiler(self) -> None:
        try:
            self.trajectoire = self.programme.construire(self.robot, self.q_depart())
        except Exception:
            self.journal(traceback.format_exc())
            self.trajectoire = None
            self._alerter(["le programme n'a pas pu etre compile "
                           "(trace complete dans le Journal)"])
            return
        ech = self.trajectoire.echantillonner(0.02) if len(self.trajectoire) else None
        if ech is None:
            self.vue.definir_chemin(None)
            self.vue.definir_cibles(None)
            for c in self.courbes_q:
                c.setData([], [])
            self.courbe_v.setData([], [])
            self.courbe_z.setData([], [])
            self.lab_temps.setText("programme vide")
            self._alerter(())
            return

        self.vue.definir_chemin(ech["P"])
        # les cibles sont des SITUATIONS : on transmet les matrices, pour
        # pouvoir afficher l'orientation commandee et pas seulement le point
        cibles = []
        for ins in self.programme.instructions:
            if ins.type not in ("MOVJ", "MOVL", "MOVC"):
                continue
            try:
                cibles.append(ins.matrice(self.robot, self.programme.repere_objet))
            except Exception:
                continue
            if ins.type == "MOVC":
                cibles.append(ins.matrice_via(self.programme.repere_objet))
        self.vue.definir_cibles(cibles or None)

        for i, c in enumerate(self.courbes_q):
            c.setData(ech["t"], ech["Q"][:, i])
        self.courbe_v.setData(ech["t"], ech["V"])
        self.courbe_z.setData(ech["t"], ech["P"][:, 2])

        avert = self.trajectoire.controler(0.05)
        # on retient les avertissements du PROGRAMME a part : une alerte
        # levee par la pose courante s'efface quand la pose redevient saine,
        # mais elle ne doit pas emporter ceux-la avec elle
        self._avert_programme = list(avert)
        self._alerte_posturale = False
        self._alerter(avert)
        if avert:
            self.journal("--- controle de la trajectoire ---")
            for a in avert[:15]:
                self.journal("  " + a)
        txt = "t = {:.2f} s / {:.2f} s".format(self.t_courant,
                                               self.trajectoire.duree)
        if self.trajectoire.lissage_actif:
            gagne = self.trajectoire.duree_sans_lissage - self.trajectoire.duree
            txt += "   (lissage : {:.2f} s gagnees)".format(gagne)
        self.lab_temps.setText(txt)
        self._maj_duree_calculee()

    def q_depart(self) -> np.ndarray:
        return np.array(self._q_depart, dtype=float)

    # -- points d'entree pour le pilotage par script ------------------------

    def charger_programme(self, programme: Programme,
                          q_depart: Optional[np.ndarray] = None) -> None:
        """
        Remplace le programme affiche. Utilise par dobot_sim.Pilote pour
        injecter un programme ecrit dans un script.
        """
        self.programme = programme
        if q_depart is not None:
            self._q_depart = np.asarray(q_depart, dtype=float).copy()
        # le programme porte son outil : on le remonte sur le robot
        if programme.outil != self.outil.nom:
            self._maj_es = True
            try:
                self.cb_outil.setCurrentIndex(
                    ou.noms_outils().index(programme.outil))
            except ValueError:
                pass
            finally:
                self._maj_es = False
            self.outil = ou.outil(programme.outil)
            self.robot.geo.lt = self.outil.longueur
            self._maj_outil()
            self.vue.invalider_enveloppe()

        # Le programme charge fixe entierement l'etat de l'outil, longueur
        # comprise : il peut porter une longueur MESUREE, differente de la
        # valeur nominale. Sans cela le z du simulateur et celui du robot
        # reel ne designeraient pas le meme point. Un programme qui n'en
        # porte pas -- un JSON d'avant cette version -- revient a la valeur
        # du catalogue, plutot que d'heriter de celle du programme d'avant.
        mesuree = getattr(programme, "longueur_outil", None)
        voulue = float(self.outil.longueur if mesuree is None else mesuree)
        if abs(voulue - self.robot.geo.lt) > 1e-9:
            precedente = self.robot.geo.lt
            self.robot.geo.lt = voulue
            self._maj_outil()
            self.vue.invalider_enveloppe()
            self.journal("longueur d'outil : {:.2f} -> {:.2f} mm ({})"
                         .format(precedente, voulue,
                                 "catalogue" if mesuree is None
                                 else "mesuree, portee par le programme"))
        self._arreter()
        self._rafraichir_liste(0)
        self._recompiler()
        self.journal("programme '{}' charge : {} instructions, {:.2f} s"
                     .format(programme.nom, len(programme),
                             self.trajectoire.duree if self.trajectoire else 0.0))

    def _recevoir_programme(self, charge) -> None:
        """
        Prend un programme envoye par un script d'un autre processus.

        C'est le pendant de `Pilote.simuler()` : le script serialise son
        programme, la fenetre deja ouverte le reconstruit et le charge.
        Elle garde ainsi sa liaison serie, ses reglages et sa place a
        l'ecran.
        """
        prog = Programme.depuis_dict(charge["programme"])
        depart = charge.get("q_depart")
        self.charger_programme(prog, np.asarray(depart, dtype=float)
                               if depart is not None else None)
        self.journal("programme '{}' recu d'un script exterieur : "
                     "{} instructions".format(prog.nom, len(prog)))
        # la fenetre est peut-etre derriere l'editeur : on la remonte
        self.setWindowState((self.windowState() & ~QtCore.Qt.WindowMinimized)
                            | QtCore.Qt.WindowActive)
        self.raise_()
        self.activateWindow()
        if charge.get("lancer"):
            self.demarrer_lecture()

    def demarrer_lecture(self) -> None:
        """Lance l'animation si elle ne tourne pas deja."""
        if not self.en_lecture:
            self._basculer_lecture()

    def _basculer_lecture(self) -> None:
        if self.trajectoire is None or not len(self.trajectoire):
            return
        self.en_lecture = not self.en_lecture
        self._maj_bouton_lecture()
        if self.en_lecture:
            if self.t_courant >= self.trajectoire.duree - 1e-9:
                self.t_courant = 0.0
                self.vue.effacer_trace()
            self.minuteur.start()
        else:
            self.minuteur.stop()

    def _maj_bouton_lecture(self) -> None:
        """Le bouton unique bascule entre lire et pause."""
        lire = not self.en_lecture
        self.act_lecture.setIcon(_icone("lire" if lire else "pause"))
        self.act_lecture.setText("Lancer" if lire else "Pause")

    def _arreter(self) -> None:
        self.en_lecture = False
        self.minuteur.stop()
        self._maj_bouton_lecture()
        self.t_courant = 0.0
        self._attente_es = 0.0
        self._segment_attente = -1
        self.es.reinitialiser(entrees=False, sorties=True)
        self._definir_etat_outil(False)
        self.vue.effacer_trace()
        self._aller_a(0.0)

    def _pas_animation(self) -> None:
        if self.trajectoire is None or not len(self.trajectoire):
            return
        dt = self.minuteur.interval() / 1000.0 * self.sp_facteur.value()

        # une instruction ENTREE fige le temps tant que le capteur n'a pas
        # bascule : c'est bien le programme qui attend, pas l'animation
        i, _ = self.trajectoire.localiser(self.t_courant)
        ins = getattr(self.trajectoire.segments[i], "instruction", None)
        if ins is not None and ins.type == "ENTREE":
            if i != self._segment_attente:
                self._segment_attente = i
                self._attente_es = 0.0
            if self.es.entree(ins.port) == bool(ins.etat):
                # condition remplie : on ne consomme pas le delai restant
                self.t_courant = (self.trajectoire.debuts[i]
                                  + self.trajectoire.segments[i].duree)
                self._segment_attente = -1
                self._aller_a(self.t_courant)
                return
            self._attente_es += dt
            if self._attente_es < ins.duree:
                self.t_courant = self.trajectoire.debuts[i]
                self._aller_a(self.t_courant)
                self.lab_temps.setText(
                    "ATTENTE entree EIO{} = {}   ({:.1f} s ecoulees sur {:.1f})"
                    .format(ins.port, 1 if ins.etat else 0,
                            self._attente_es, ins.duree))
                return
            self.journal("ENTREE EIO{} : delai de {:.1f} s depasse, "
                         "le programme continue".format(ins.port, ins.duree))
            self._segment_attente = -1
        else:
            self._segment_attente = -1

        self.t_courant += dt
        if self.t_courant >= self.trajectoire.duree:
            self.t_courant = self.trajectoire.duree
            self._basculer_lecture()

        # arret de securite : on regarde la pose AVANT de la jouer
        if self.act_securite.isChecked():
            defauts = self.robot.defauts(self.trajectoire.q(self.t_courant))
            if defauts:
                self._arret_securite(defauts)
                return
        self._aller_a(self.t_courant)

    def _arret_securite(self, defauts: Sequence[str]) -> None:
        """
        Coupe l'animation sur une faute, et laisse le robot DESSUS.

        Rester sur la pose fautive est volontaire : c'est en la voyant qu'on
        comprend ce qui cloche. Le robot reel, lui, se serait arrete avant.
        """
        if self.en_lecture:
            self._basculer_lecture()
        self._aller_a(self.t_courant)

        i = int(np.clip(self.trajectoire.localiser(self.t_courant)[0], 0,
                        len(self.trajectoire) - 1))
        tete = ("Arret de securite a t = {:.2f} s, instruction {}/{}"
                .format(self.t_courant, i + 1, len(self.trajectoire)))
        self.journal("--- " + tete + " ---")
        for d in defauts:
            self.journal("  " + d)
        # alerte liee a la POSE : elle s'effacera d'elle-meme des que le
        # bras sera revenu dans ses butees
        self._alerte_posturale = True
        self._alerter([tete + " -- " + defauts[0]] + list(defauts[1:]))

        # boite NON modale : elle se voit sans figer l'application
        self._boite_securite = QtWidgets.QMessageBox(self)
        self._boite_securite.setIcon(QtWidgets.QMessageBox.Critical)
        self._boite_securite.setWindowTitle("Arret de securite")
        self._boite_securite.setText(tete)
        self._boite_securite.setInformativeText("\n".join(defauts))
        self._boite_securite.setDetailedText(
            "Le mouvement a ete interrompu : le robot reste sur la pose "
            "fautive pour que vous puissiez l'observer.\n\n"
            "Corrigez la cible, ou decochez Execution > Arret de securite "
            "pour laisser la simulation aller au bout.")
        self._boite_securite.setModal(False)
        self._boite_securite.show()

    # -- pas a pas -----------------------------------------------------------

    def pas_a_pas(self, sens: int = 1) -> None:
        """
        Avance ou recule d'une instruction, et s'y arrete.

        On se place a la FIN du segment vise : le robot montre donc la
        cible atteinte par l'instruction, pas son point de depart.
        """
        if self.trajectoire is None or not len(self.trajectoire):
            return
        if self.en_lecture:
            self._basculer_lecture()          # le pas a pas met en pause

        # On raisonne sur les instants de FIN d'instruction, strictement
        # croissants meme avec du lissage. Passer par localiser() serait
        # piegeux : a une jonction, deux segments sont actifs a la fois.
        fins = [self.trajectoire.debuts[k] + s.duree
                for k, s in enumerate(self.trajectoire.segments)]
        eps = 1e-6
        if sens > 0:
            suivants = [k for k, t in enumerate(fins) if t > self.t_courant + eps]
            self.aller_a_instruction(suivants[0] if suivants else len(fins) - 1)
            return
        precedents = [k for k, t in enumerate(fins) if t < self.t_courant - eps]
        if precedents:
            self.aller_a_instruction(precedents[-1])
        else:
            # deja au premier point d'arret : on revient au depart
            self.t_courant = 0.0
            self._aller_a(0.0)

    def aller_a_instruction(self, i: int) -> None:
        """Amene le robot sur la cible de l'instruction i et selectionne sa ligne."""
        if self.trajectoire is None or not (0 <= i < len(self.trajectoire)):
            return
        seg = self.trajectoire.segments[i]
        self.t_courant = self.trajectoire.debuts[i] + seg.duree
        self._aller_a(self.t_courant, segment=i)
        if self.liste.currentRow() != i:
            self._selection_programmee = True
            try:
                self.liste.setCurrentRow(i)
            finally:
                self._selection_programmee = False

    def _deplacer_temps(self, v: int) -> None:
        if self.trajectoire is None:
            return
        self.t_courant = self.trajectoire.duree * v / 1000.0
        self._aller_a(self.t_courant)

    def _aller_a(self, t: float, segment: Optional[int] = None) -> None:
        """
        Place le robot a l'instant t.

        `segment` force l'indice affiche : a une jonction, deux segments
        sont actifs et localiser() renvoie le suivant, ce qui afficherait
        l'instruction d'apres celle sur laquelle on vient de s'arreter.
        """
        if self.trajectoire is None or not len(self.trajectoire):
            return
        # les sorties et l'etat de l'outil sont des fonctions du temps,
        # au meme titre que la configuration articulaire
        for p, v in enumerate(self.programme.etat_sorties(t, self.es.nb_ports), 1):
            self.es.definir_sortie(p, v)
        self._definir_etat_outil(self.programme.etat_outil(t))

        q = self.trajectoire.q(t)
        self._appliquer_q(q, source="animation")
        duree = max(self.trajectoire.duree, 1e-9)
        self.curseur_temps.blockSignals(True)
        self.curseur_temps.setValue(int(1000 * t / duree))
        self.curseur_temps.blockSignals(False)
        self.ligne_t1.setValue(t)
        self.ligne_t2.setValue(t)
        i = segment if segment is not None else self.trajectoire.localiser(t)[0]
        i = int(np.clip(i, 0, len(self.trajectoire) - 1))
        self.lab_temps.setText("t = {:.2f} s / {:.2f} s   -   instruction {}/{}  {}"
                               .format(t, self.trajectoire.duree, i + 1,
                                       len(self.trajectoire),
                                       self.trajectoire.segments[i].type_mouvement))

    # -- fichiers -----------------------------------------------------------

    def _nouveau(self) -> None:
        self.programme = Programme([], "nouveau")
        self._rafraichir_liste(0)
        self._recompiler()

    def _ouvrir(self) -> None:
        chemin, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Ouvrir un programme", "", "Programme Dobot (*.json)")
        if not chemin:
            return
        try:
            self.programme = Programme.charger(chemin)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Ouverture", str(exc))
            return
        self.journal("programme charge : " + chemin)
        self._rafraichir_liste(0)
        self._recompiler()

    def _enregistrer(self) -> None:
        chemin, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Enregistrer le programme", self.programme.nom + ".json",
            "Programme Dobot (*.json)")
        if not chemin:
            return
        self.programme.enregistrer(chemin)
        self.journal("programme enregistre : " + chemin)

    def _executer_script(self, chemin: str = "") -> None:
        """
        Execute un script Python de pilotage dans le simulateur.

        Le script est celui que l'on ecrit dans VS Code : il construit un
        Pilote et appelle simuler(). Comme une fenetre est deja ouverte,
        simuler() y charge le programme au lieu d'en ouvrir une seconde.
        """
        if not chemin:
            chemin, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Executer un script de pilotage", "", "Script Python (*.py)")
        if not chemin:
            return
        self._dernier_script = chemin
        espace = {"__name__": "__main__", "__file__": chemin}
        self.journal("--- execution de {} ---".format(chemin))
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                code = compile(f.read(), chemin, "exec")
            exec(code, espace)                      # noqa: S102 (script de l'utilisateur)
        except Exception:
            self.journal(traceback.format_exc())
            QtWidgets.QMessageBox.warning(
                self, "Script", "Le script a echoue.\n\n"
                                "Le detail est dans l'onglet Journal.")
            return
        self.journal("script termine")

    def _rejouer_script(self) -> None:
        chemin = getattr(self, "_dernier_script", "")
        if chemin:
            self._executer_script(chemin)
        else:
            self._executer_script()

    def _exporter_python(self) -> None:
        chemin, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter un script Dobot", "programme_dobot.py",
            "Script Python (*.py)")
        if not chemin:
            return
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(self.programme.exporter_python(self.robot))
        self.journal("script exporte : " + chemin)

    def _exporter_csv(self) -> None:
        chemin, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exporter les consignes", "trajectoire.csv", "CSV (*.csv)")
        if not chemin:
            return
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(self.programme.exporter_csv(self.robot, 0.05, self.q_depart()))
        self.journal("consignes exportees : " + chemin)

    # -- aide ---------------------------------------------------------------

    def _aide_matrices(self) -> None:
        texte = """<h3>Matrices homogenes : aide-memoire</h3>
<pre>
        [ R  p ]        R : rotation 3x3      p : position 3x1
    T = [ 0  1 ]        T : situation d'un repere par rapport a un autre

Composition        T02 = T01 . T12
Inverse            T^-1 = [ R^T  -R^T p ]
Point transforme   p' = R . p + t
</pre>
<h4>Chaine du Dobot Magician</h4>
<pre>
T01 = Rz(q1)                     rotation de la colonne
T12 = Tz(138) . Ry(q2)           epaule
T23 = Tz(135) . Ry(q3-q2+90)     coude
T34 = Tz(147) . Ry(-90-q3)       poignet
T45 = Tx(61)  . Rz(q4)           axe outil
T5T = Rx(180)                    z de l'outil vers le bas
</pre>
<p>La somme des rotations autour de Y vaut zero : c'est l'effet du
parallelogramme, qui maintient l'outil vertical.</p>
<h4>Modele geometrique inverse</h4>
<pre>
q1 = atan2(y, x)
rp = sqrt(x^2+y^2) - 61        zp = z - 138        D = hypot(rp, zp)
cos(beta) = (l1^2 + D^2 - l2^2) / (2 l1 D)      (loi des cosinus)
q2 = atan2(rp, zp) - beta
q3 = atan2(l1 cos q2 - zp, rp - l1 sin q2)
q4 = r - q1
</pre>
<h4>Modele cinematique</h4>
<pre>
[vx vy vz wz]^T = J(q) . [dq1 dq2 dq3 dq4]^T
Singularite quand det(J) -> 0 : TCP sur l'axe vertical, bras tendu ou replie.
</pre>"""
        b = QtWidgets.QMessageBox(self)
        b.setWindowTitle("Aide-memoire")
        b.setTextFormat(QtCore.Qt.RichText)
        b.setText(texte)
        b.exec_() if hasattr(b, "exec_") else b.exec()

    def _a_propos(self) -> None:
        moteur = "OpenGL (pyqtgraph)" if self.mode_3d else \
                 "projections 2D (OpenGL indisponible)"
        msg = ("<b>Simulateur Dobot Magician</b><br><br>"
               "Pilotage virtuel, trajectoires articulaires et cartesiennes "
               "construites sur les matrices homogenes.<br><br>"
               "Rendu : {}<br>"
               "Geometrie : d1={:.0f}, l1={:.0f}, l2={:.0f}, l3={:.0f} mm"
               .format(moteur, self.robot.geo.d1, self.robot.geo.l1,
                       self.robot.geo.l2, self.robot.geo.l3))
        if ERREUR_OPENGL:
            msg += "<br><br><i>OpenGL : {}</i>".format(ERREUR_OPENGL)
        QtWidgets.QMessageBox.about(self, "A propos", msg)

    # -- divers -------------------------------------------------------------

    def journal(self, message: str) -> None:
        self.txt_journal.appendPlainText(str(message))


def lancer() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    f = FenetrePrincipale()
    f.show()
    return app.exec_() if hasattr(app, "exec_") else app.exec()
