"""
Simulateur Dobot Magician.

Pilotage virtuel d'un Dobot Magician a partir de son modele CAO :
modeles geometriques direct et inverse, jacobienne, trajectoires
articulaires et cartesiennes, le tout construit sur les matrices
homogenes 4x4.

Modules
-------
homogeneous  algebre SE(3) : construction, composition, inversion,
             interpolation, Denavit-Hartenberg
robot        modele du Magician : MGD, MGI, jacobienne, butees
trajectory   lois horaires et segments MOVJ / MOVL / MOVC
program      programme robot editable, JSON, export vers le robot reel
meshes       lecture des OBJ et recalage dans les reperes des corps
viewer3d     vue OpenGL ; viewer2d : repli en projections
pilote       un programme decrit, puis joue d'un bloc (simulateur, robot)
commande     chaque appel execute tout de suite, et attend la fin
ui           interface graphique

Exemple minimal
---------------
>>> from dobot_sim import DobotMagician
>>> r = DobotMagician()
>>> T = r.mgd([0, 20, 30, 0])          # modele geometrique direct
>>> q = r.mgi(T)                       # modele geometrique inverse
"""

import os as _os
import sys as _sys

# Linux sous Wayland (Ubuntu par defaut) : PyQt5 installe par pip n'a pas de
# greffon Wayland et ouvre sa fenetre via XWayland, avec un contexte GLX.
# PyOpenGL, voyant une session Wayland, choisirait EGL : ses appels partiraient
# alors dans le mauvais contexte et la vue 3D resterait vide. On aligne les
# deux sur X11/GLX. A faire avant tout import d'OpenGL ; setdefault laisse a
# l'utilisateur le dernier mot via les variables d'environnement.
if _sys.platform.startswith("linux") and _os.environ.get("DISPLAY"):
    _os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

from .homogeneous import (composer, formater, identite, interpoler, inverse,
                          position_de, rot_x, rot_y, rot_z, rotation_de,
                          translation)
from .pilote import ErreurExecution, Pilote
from .commande import Commande
from .program import Instruction, Programme
from .robot import (DobotMagician, Encombrement, ErreurCinematique, Geometrie,
                    Limites, cible, decomposer_cible, depuis_xyzrpy,
                    repere_travail, vers_xyzrpy, verifier_repere_objet,
                    verifier_repere_outil)
from .axe_lineaire import AxeLineaire
from .cellule import Cellule, ErreurCellule, Poste
from .convoyeur import Convoyeur, Piece
from .pieces import Manutention, PieceLibre, piece_posee
from .image import ErreurImage, lire_image, programme_depuis_image
from .svg import ErreurSVG, lire_svg, programme_depuis_svg
from .trace import programme_depuis_dessin, programme_depuis_polylignes
from .trajectory import (ProfilPolynomial5, ProfilTrapeze, SegmentArticulaire,
                         SegmentCirculaire, SegmentLineaire, Trajectoire,
                         creer_profil, creer_profil_vitesse)

__version__ = "1.0.0"

__all__ = [
    "Pilote", "Commande", "ErreurExecution", "cible", "repere_travail", "decomposer_cible",
    "vers_xyzrpy", "depuis_xyzrpy",
    "verifier_repere_outil", "verifier_repere_objet",
    "DobotMagician", "Geometrie", "Limites", "Encombrement",
    "ErreurCinematique",
    "Trajectoire", "SegmentArticulaire", "SegmentLineaire", "SegmentCirculaire",
    "ProfilTrapeze", "ProfilPolynomial5", "creer_profil", "creer_profil_vitesse",
    "Programme", "Instruction",
    "programme_depuis_svg", "lire_svg", "ErreurSVG",
    "programme_depuis_image", "lire_image", "ErreurImage",
    "programme_depuis_dessin", "programme_depuis_polylignes",
    "Convoyeur", "Piece", "AxeLineaire",
    "PieceLibre", "piece_posee", "Manutention",
    "Cellule", "ErreurCellule", "Poste",
    "identite", "translation", "rot_x", "rot_y", "rot_z",
    "composer", "inverse", "interpoler", "rotation_de", "position_de", "formater",
    "__version__",
]
