"""
Vue 3D temps reel du robot (pyqtgraph + OpenGL).

Chaque piece du maillage est un GLMeshItem dont la matrice de transformation
est simplement la matrice homogene absolue du repere qui la porte : animer
le robot revient donc a recalculer le MGD puis a pousser les matrices vers
la carte graphique.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

from . import homogeneous as hg
from .meshes import Piece
from .robot import DobotMagician

__all__ = ["Vue3D", "sonder_opengl"]

COULEUR_X = (0.90, 0.25, 0.25, 1.0)
COULEUR_Y = (0.30, 0.75, 0.35, 1.0)
COULEUR_Z = (0.30, 0.50, 0.95, 1.0)

#: gris fonce neutre : assez clair pour ne pas ecraser les pieces noires
#: du robot, assez sombre pour que le blanc de la machine ressorte
FOND = (52, 55, 61)
COULEUR_GRILLE = (118, 124, 136, 150)


# ---------------------------------------------------------------------------
# Nuanceur
#
# Le nuanceur 'shaded' livre avec pyqtgraph n'a que 0,2 d'ambiant et une
# seule source : un corps blanc y apparait gris fonce. On en enregistre un
# autre, a deux sources plus un ambiant genereux, pour retrouver le blanc
# de la machine. gl_FrontFacing retourne la normale des facettes vues de
# dos, ce qui rend le rendu insensible au sens d'enroulement.
# ---------------------------------------------------------------------------

NUANCEUR = "dobot"

try:
    from pyqtgraph.opengl import shaders as _shaders

    if NUANCEUR not in _shaders.ShaderProgram.names:
        _shaders.ShaderProgram(NUANCEUR, [
            _shaders.VertexShader("""
                uniform mat4 u_mvp;
                uniform mat3 u_normal;
                attribute vec4 a_position;
                attribute vec3 a_normal;
                attribute vec4 a_color;
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    v_normal = normalize(u_normal * a_normal);
                    v_color = a_color;
                    gl_Position = u_mvp * a_position;
                }
            """),
            _shaders.FragmentShader("""
                #ifdef GL_ES
                precision mediump float;
                #endif
                varying vec4 v_color;
                varying vec3 v_normal;
                void main() {
                    vec3 n = normalize(v_normal);
                    if (!gl_FrontFacing) n = -n;
                    // frontale : |n.z| eclaire ce qui fait face a la camera
                    // sans dependre du sens de l'axe de vue, et assombrit
                    // les silhouettes, ce qui detache les volumes
                    float frontale = abs(n.z);
                    float haute = max(dot(n, normalize(vec3(-0.45, 0.78, 0.0))), 0.0);
                    float i = clamp(0.38 + 0.45 * frontale + 0.30 * haute,
                                    0.0, 1.06);
                    vec3 rgb = clamp(v_color.rgb * i, 0.0, 1.0);
                    gl_FragColor = vec4(rgb, v_color.a);
                }
            """),
        ])
except Exception:                       # pragma: no cover
    NUANCEUR = "shaded"                 # repli sur le nuanceur standard


def sonder_opengl() -> tuple:
    """
    Verifie, AVANT de construire la vue, que le rendu 3D marchera vraiment.

    Renvoie ``(erreur, pilote)`` : ``erreur`` vide si tout va bien, sinon la
    raison ; ``pilote`` decrit la carte et la version d'OpenGL.

    Un widget OpenGL ne leve aucune exception quand le rendu echoue : la
    fenetre reste simplement vide. D'ou cette sonde : on ouvre un contexte
    hors ecran avec Qt, puis on interroge ce MEME contexte par PyOpenGL. Si
    PyOpenGL ne le voit pas (cas Wayland/EGL contre XWayland/GLX), ou si la
    version est trop ancienne pour les nuanceurs, la vue 2D prend le relais.
    Exige qu'une QApplication existe deja.
    """
    from pyqtgraph.Qt import QtGui

    ctx = QtGui.QOpenGLContext()
    if not ctx.create():
        return "impossible de creer un contexte OpenGL", ""
    surface = QtGui.QOffscreenSurface()
    surface.setFormat(ctx.format())
    surface.create()
    if not surface.isValid() or not ctx.makeCurrent(surface):
        return "contexte OpenGL inutilisable", ""
    try:
        fmt = ctx.format()
        if (fmt.majorVersion(), fmt.minorVersion()) < (2, 0):
            return ("OpenGL {}.{} : la version 2.0 est requise"
                    .format(fmt.majorVersion(), fmt.minorVersion())), ""
        from OpenGL import GL
        version = GL.glGetString(GL.GL_VERSION)
        rendu = GL.glGetString(GL.GL_RENDERER)
        if not version:
            import os
            return ("PyOpenGL ne voit pas le contexte de Qt (plateforme {!r})"
                    .format(os.environ.get("PYOPENGL_PLATFORM", "auto"))), ""
        pilote = "{} -- OpenGL {}".format(
            (rendu or b"?").decode(errors="replace"),
            version.decode(errors="replace"))
        return "", pilote
    except Exception as exc:
        return "{}: {}".format(type(exc).__name__, exc), ""
    finally:
        ctx.doneCurrent()


def _toujours_visible(item, profondeur: int = 100) -> None:
    """
    Dessine l'element PAR-DESSUS la scene.

    Deux conditions, et il faut les deux : desactiver le test de
    profondeur, ET passer l'element en fin d'ordre de trace, sinon les
    maillages ajoutes ensuite le recouvrent quand meme. Le triedre de base
    a son origine au centre du socle : sans cela il y serait enterre.
    C'est le comportement d'un gizmo d'orientation de logiciel de CAO.
    """
    item.setDepthValue(profondeur)
    try:
        from OpenGL import GL
        item.setGLOptions({
            GL.GL_DEPTH_TEST: False,
            GL.GL_BLEND: True,
            GL.GL_ALPHA_TEST: False,
            GL.GL_CULL_FACE: False,
            "glBlendFunc": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA),
        })
    except Exception:                       # pragma: no cover
        pass


def _qmat(T) -> pg.Transform3D:
    """Matrice homogene numpy -> Transform3D de pyqtgraph."""
    T = np.asarray(T, dtype=float)
    return pg.Transform3D(*[float(v) for v in T.reshape(-1)])


def _trieds(taille: float = 40.0):
    """Sommets et couleurs d'un triedre (3 segments colores)."""
    pos = np.array([[0, 0, 0], [taille, 0, 0],
                    [0, 0, 0], [0, taille, 0],
                    [0, 0, 0], [0, 0, taille]], dtype=float)
    col = np.array([COULEUR_X, COULEUR_X, COULEUR_Y, COULEUR_Y,
                    COULEUR_Z, COULEUR_Z], dtype=float)
    return pos, col


class Vue3D(gl.GLViewWidget):
    """Widget OpenGL affichant le robot, ses reperes et ses trajectoires."""

    def __init__(self, robot: DobotMagician, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.pieces: List[Piece] = []
        self.pieces_outil: List[Piece] = []
        self._items_pieces: List[gl.GLMeshItem] = []
        self._items_outil: List[gl.GLMeshItem] = []
        self._items_reperes: List[gl.GLLinePlotItem] = []
        self._q = np.array(robot.q_home, dtype=float)
        self._trace: List[np.ndarray] = []
        self._enveloppe_prete = False
        self._item_enveloppe: Optional[gl.GLMeshItem] = None
        self._maillages_visibles = True

        self.setBackgroundColor(pg.mkColor(*FOND))
        self.setCameraPosition(distance=620, elevation=22, azimuth=35)
        self.opts["center"] = pg.Vector(60, 0, 130)

        self._creer_decor()
        self._creer_annotations()

    # -- decor ---------------------------------------------------------------

    def _creer_decor(self) -> None:
        self.grille = gl.GLGridItem()
        self.grille.setSize(x=700, y=700)
        self.grille.setSpacing(x=25, y=25)
        try:                                   # setColor absent des versions anciennes
            self.grille.setColor(COULEUR_GRILLE)
        except Exception:
            pass
        self.addItem(self.grille)

        # Triedre du repere de base, avec le code couleur x rouge, y vert,
        # z bleu. Les etiquettes evitent d'avoir a memoriser la convention.
        taille = 90.0
        pos, col = _trieds(taille)
        self.triedre_base = gl.GLLinePlotItem(pos=pos, color=col, width=3.5,
                                              antialias=True, mode="lines")
        _toujours_visible(self.triedre_base)
        self.addItem(self.triedre_base)

        self.etiquettes_axes = []
        for nom, couleur, direction in (("x", COULEUR_X, (1, 0, 0)),
                                        ("y", COULEUR_Y, (0, 1, 0)),
                                        ("z", COULEUR_Z, (0, 0, 1))):
            p = np.asarray(direction, dtype=float) * (taille + 14.0)
            try:
                item = gl.GLTextItem(pos=p, text=nom,
                                     color=pg.mkColor([int(255 * c)
                                                       for c in couleur[:3]]))
            except Exception:          # GLTextItem absent des vieilles versions
                break
            _toujours_visible(item)
            self.addItem(item)
            self.etiquettes_axes.append(item)

    def _creer_annotations(self) -> None:
        vide = np.zeros((0, 3))

        self.squelette = gl.GLLinePlotItem(pos=vide, color=(1.0, 0.85, 0.2, 1.0),
                                           width=3.0, antialias=True,
                                           mode="line_strip")
        self.addItem(self.squelette)

        self.chemin = gl.GLLinePlotItem(pos=vide, color=(0.20, 0.85, 0.95, 0.95),
                                        width=2.5, antialias=True, mode="line_strip")
        self.addItem(self.chemin)

        self.trace = gl.GLLinePlotItem(pos=vide, color=(1.0, 0.45, 0.15, 1.0),
                                       width=3.0, antialias=True, mode="line_strip")
        self.addItem(self.trace)

        self.cibles = gl.GLScatterPlotItem(pos=vide, size=9.0,
                                           color=(1.0, 0.95, 0.3, 1.0))
        self.addItem(self.cibles)

        # trieds des cibles : un seul item, les sommets sont pre-transformes
        # puisque les cibles ne bougent pas entre deux compilations
        self.reperes_cibles = gl.GLLinePlotItem(pos=vide, color=vide,
                                                width=2.0, antialias=True,
                                                mode="lines")
        self.reperes_cibles.setVisible(False)
        self.addItem(self.reperes_cibles)

        self.enveloppe = gl.GLLinePlotItem(pos=vide, color=(0.45, 0.55, 0.70, 0.55),
                                           width=1.0, antialias=True, mode="lines")
        self.enveloppe.setVisible(False)
        self.addItem(self.enveloppe)

        pos, col = _trieds(45.0)
        self.triedre_outil = gl.GLLinePlotItem(pos=pos, color=col, width=3.5,
                                               antialias=True, mode="lines")
        self.addItem(self.triedre_outil)

    # -- maillages -----------------------------------------------------------

    def _construire_items(self, pieces: Sequence[Piece]) -> List[gl.GLMeshItem]:
        items = []
        for p in pieces:
            md = gl.MeshData(vertexes=p.sommets.astype(np.float32),
                             faces=p.faces.astype(np.uint32))
            item = gl.GLMeshItem(meshdata=md, smooth=True, color=p.couleur,
                                 shader=NUANCEUR, glOptions="opaque")
            item.setVisible(p.visible and self._maillages_visibles)
            self.addItem(item)
            items.append(item)
        return items

    def definir_pieces(self, pieces: Sequence[Piece]) -> None:
        """Remplace les maillages du ROBOT (l'outil est gere a part)."""
        for item in self._items_pieces:
            self.removeItem(item)
        self.pieces = list(pieces)
        self._items_pieces = self._construire_items(self.pieces)
        self._creer_items_reperes()
        self.maj(self._q)

    def definir_outil(self, pieces: Sequence[Piece]) -> None:
        """
        Remplace les maillages de l'OUTIL seul.

        Separe du robot pour que le changement d'etat d'une ventouse ou
        d'une pince, frequent pendant une animation, ne reconstruise pas
        les 55 000 facettes du bras.
        """
        for item in self._items_outil:
            self.removeItem(item)
        self.pieces_outil = list(pieces)
        self._items_outil = self._construire_items(self.pieces_outil)
        self.maj(self._q)

    def _creer_items_reperes(self) -> None:
        for item in self._items_reperes:
            self.removeItem(item)
        self._items_reperes = []
        pos, col = _trieds(35.0)
        for _ in range(len(self.robot.NOMS_REPERES)):
            item = gl.GLLinePlotItem(pos=pos, color=col, width=2.0,
                                     antialias=True, mode="lines")
            item.setVisible(False)
            self.addItem(item)
            self._items_reperes.append(item)

    # -- mise a jour ---------------------------------------------------------

    def maj(self, q_deg: Sequence[float], enregistrer_trace: bool = False) -> None:
        """Place le robot dans la configuration q (degres)."""
        self._q = np.asarray(q_deg, dtype=float).copy()
        reperes = self.robot.reperes(self._q)

        for piece, item in zip(self.pieces + self.pieces_outil,
                               self._items_pieces + self._items_outil):
            item.setTransform(_qmat(reperes[piece.repere]))

        for k, item in enumerate(self._items_reperes):
            if k < len(reperes):
                item.setTransform(_qmat(reperes[k]))

        self.triedre_outil.setTransform(_qmat(reperes[-1]))

        pts = np.array([hg.position_de(T) for T in reperes])
        self.squelette.setData(pos=pts)

        if enregistrer_trace:
            self._trace.append(pts[-1].copy())
            if len(self._trace) > 4000:
                self._trace = self._trace[-4000:]
            self.trace.setData(pos=np.array(self._trace))

    def effacer_trace(self) -> None:
        self._trace = []
        self.trace.setData(pos=np.zeros((0, 3)))

    # -- annotations ---------------------------------------------------------

    def definir_chemin(self, points: Optional[np.ndarray]) -> None:
        """Trace le chemin theorique du TCP, tableau (N, 3) ou None."""
        if points is None or len(points) == 0:
            self.chemin.setData(pos=np.zeros((0, 3)))
        else:
            self.chemin.setData(pos=np.asarray(points, dtype=float))

    def definir_cibles(self, poses: Optional[Sequence[np.ndarray]],
                       taille: float = 26.0) -> None:
        """
        Place les marqueurs et les trieds des cibles du programme.

        `poses` est une suite de matrices homogenes 4x4 : on affiche donc
        l'orientation commandee, et pas seulement le point vise. L'axe z
        d'une cible pointe vers le bas, c'est la direction d'approche.
        """
        if poses is None or len(poses) == 0:
            self.cibles.setData(pos=np.zeros((0, 3)))
            self.reperes_cibles.setData(pos=np.zeros((0, 3)),
                                        color=np.zeros((0, 4)))
            return

        origines, sommets, couleurs = [], [], []
        for T in poses:
            T = np.asarray(T, dtype=float)
            p = T[:3, 3]
            origines.append(p)
            for axe, couleur in enumerate((COULEUR_X, COULEUR_Y, COULEUR_Z)):
                sommets += [p, p + taille * T[:3, axe]]
                couleurs += [couleur, couleur]
        self.cibles.setData(pos=np.array(origines, dtype=float))
        self.reperes_cibles.setData(pos=np.array(sommets, dtype=float),
                                    color=np.array(couleurs, dtype=float))

    def afficher_reperes(self, actif: bool) -> None:
        for item in self._items_reperes:
            item.setVisible(actif)

    def afficher_reperes_cibles(self, actif: bool) -> None:
        self.reperes_cibles.setVisible(actif)

    def afficher_cibles(self, actif: bool) -> None:
        self.cibles.setVisible(actif)
        if not actif:
            self.reperes_cibles.setVisible(False)

    def afficher_squelette(self, actif: bool) -> None:
        self.squelette.setVisible(actif)

    def afficher_maillages(self, actif: bool) -> None:
        self._maillages_visibles = bool(actif)
        for item in self._items_pieces + self._items_outil:
            item.setVisible(actif)

    def afficher_grille(self, actif: bool) -> None:
        self.grille.setVisible(actif)

    def afficher_triedre_base(self, actif: bool) -> None:
        """Triedre du repere de base : x rouge, y vert, z bleu."""
        self.triedre_base.setVisible(actif)
        for item in self.etiquettes_axes:
            item.setVisible(actif)

    def afficher_enveloppe(self, actif: bool) -> None:
        if actif and not self._enveloppe_prete:
            self._construire_enveloppe()
            self._enveloppe_prete = True
        self.enveloppe.setVisible(actif)
        if self._item_enveloppe is not None:
            self._item_enveloppe.setVisible(actif)

    def invalider_enveloppe(self) -> None:
        """A appeler apres un changement d'outil ou de geometrie."""
        self._enveloppe_prete = False
        visible = self.enveloppe.visible()
        if self._item_enveloppe is not None:
            self.removeItem(self._item_enveloppe)
            self._item_enveloppe = None
        if visible:
            self.afficher_enveloppe(True)

    def _contour_meridien(self, n_pts: int = 48) -> np.ndarray:
        """
        Contour du domaine atteignable dans le plan meridien (rho, z).

        Le domaine articulaire (q2, q3) est un rectangle ; son CONTOUR,
        transporte par le modele geometrique direct, donne le bord de la
        section de l'espace de travail.
        """
        r = self.robot
        g = r.geo
        q2a, q2b = r.lim.q_min[1], r.lim.q_max[1]
        q3a, q3b = r.lim.q_min[2], r.lim.q_max[2]
        u = np.linspace(0.0, 1.0, n_pts)
        bords = [
            (q2a + (q2b - q2a) * u, np.full(n_pts, q3a)),
            (np.full(n_pts, q2b), q3a + (q3b - q3a) * u),
            (q2b + (q2a - q2b) * u, np.full(n_pts, q3b)),
            (np.full(n_pts, q2a), q3b + (q3a - q3b) * u),
        ]
        q2 = np.concatenate([b[0] for b in bords])
        q3 = np.concatenate([b[1] for b in bords])
        d = np.pi / 180.0
        rho = g.l1 * np.sin(q2 * d) + g.l2 * np.cos(q3 * d) + g.l3
        z = g.d1 + g.l1 * np.cos(q2 * d) - g.l2 * np.sin(q3 * d) - g.lt
        return np.column_stack([rho, z])

    def _construire_enveloppe(self, n_meridiens: int = 33) -> None:
        """
        Enveloppe de travail : le contour meridien balaye autour de l'axe
        J1, entre les butees de J1. Une surface translucide donne le volume,
        des lignes en marquent le bord.
        """
        r = self.robot
        contour = self._contour_meridien()
        rho, z = contour[:, 0], contour[:, 1]
        n = len(rho)
        d = np.pi / 180.0
        azimuts = np.linspace(r.lim.q_min[0], r.lim.q_max[0], n_meridiens) * d

        # --- surface de revolution
        ca, sa = np.cos(azimuts), np.sin(azimuts)
        X = np.outer(ca, rho).ravel()
        Y = np.outer(sa, rho).ravel()
        Z = np.tile(z, n_meridiens)
        V = np.column_stack([X, Y, Z]).astype(np.float32)
        faces = []
        for a in range(n_meridiens - 1):
            for i in range(n):
                j = (i + 1) % n
                p00, p01 = a * n + i, a * n + j
                p10, p11 = (a + 1) * n + i, (a + 1) * n + j
                faces += [[p00, p10, p11], [p00, p11, p01]]
        F = np.asarray(faces, dtype=np.uint32)

        if self._item_enveloppe is not None:
            self.removeItem(self._item_enveloppe)
        md = gl.MeshData(vertexes=V, faces=F)
        self._item_enveloppe = gl.GLMeshItem(
            meshdata=md, smooth=True, color=(0.25, 0.55, 0.85, 0.10),
            shader="balloon", glOptions="additive")
        self.addItem(self._item_enveloppe)

        # --- bords : les deux sections extremes et quelques meridiens
        segments = []
        indices = [0, n_meridiens - 1] + list(
            range(0, n_meridiens, max(n_meridiens // 6, 1)))
        for k in sorted(set(indices)):
            a = azimuts[k]
            pts = np.column_stack([rho * np.cos(a), rho * np.sin(a), z])
            pts = np.vstack([pts, pts[:1]])
            for i in range(len(pts) - 1):
                segments += [pts[i], pts[i + 1]]
        # quelques paralleles, pour lire le volume
        for k in range(0, n, max(n // 10, 1)):
            cercle = np.column_stack([rho[k] * np.cos(azimuts),
                                      rho[k] * np.sin(azimuts),
                                      np.full(n_meridiens, z[k])])
            for i in range(len(cercle) - 1):
                segments += [cercle[i], cercle[i + 1]]
        self.enveloppe.setData(pos=np.array(segments, dtype=float))

    # -- points de vue -------------------------------------------------------

    #: bornes de distance camera, en mm
    DISTANCE_MIN = 90.0
    DISTANCE_MAX = 2600.0
    CADRAGE = dict(distance=620.0, centre=(60.0, 0.0, 130.0))

    def vue(self, nom: str) -> None:
        """Change l'angle de vue en CONSERVANT le zoom courant."""
        angles = {"iso": (22, 35), "face": (0, 0), "cote": (0, 90),
                  "dessus": (89, 0), "arriere": (10, 180)}
        el, az = angles.get(nom, (22, 35))
        self.setCameraPosition(elevation=el, azimuth=az)
        self.update()

    def zoomer(self, facteur: float) -> None:
        """
        Rapproche (facteur < 1) ou eloigne (facteur > 1) la camera.

        Meme effet que la molette, mais accessible au clavier et au menu :
        sur pave tactile la molette n'est pas toujours commode.
        """
        d = float(self.opts.get("distance", self.CADRAGE["distance"]))
        self.setCameraPosition(
            distance=float(np.clip(d * float(facteur),
                                   self.DISTANCE_MIN, self.DISTANCE_MAX)))
        self.update()

    def recadrer(self) -> None:
        """Revient au cadrage d'origine, robot entier visible."""
        self.setCameraPosition(distance=self.CADRAGE["distance"])
        self.opts["center"] = pg.Vector(*self.CADRAGE["centre"])
        self.update()

    def keyPressEvent(self, ev):                 # pragma: no cover (IHM)
        """+ et - zooment, 0 recadre, meme sans passer par le menu."""
        touche = ev.key()
        if touche in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.zoomer(1.0 / 1.25)
        elif touche in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self.zoomer(1.25)
        elif touche == QtCore.Qt.Key_0:
            self.recadrer()
        else:
            super().keyPressEvent(ev)
