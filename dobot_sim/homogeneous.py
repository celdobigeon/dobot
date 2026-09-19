"""
Algebre des matrices homogenes 4x4 (groupe SE(3)).

Toutes les fonctions manipulent des tableaux numpy de forme (4, 4) :

        [ R  p ]      R : matrice de rotation 3x3 (SO(3))
    T = [ 0  1 ]      p : vecteur position 3x1 (mm)

Convention : les angles sont en RADIANS dans ce module.
Les modules de plus haut niveau (robot, trajectoire, IHM) travaillent
en degres, comme l'API officielle du Dobot.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "identite", "translation", "rot_x", "rot_y", "rot_z", "rot_axe",
    "depuis_rp", "rotation_de", "position_de", "inverse", "composer",
    "transformer_point", "transformer_points",
    "rpy_vers_rotation", "rotation_vers_rpy",
    "rotation_vers_axe_angle", "axe_angle_vers_rotation",
    "rotation_vers_quaternion", "quaternion_vers_rotation",
    "slerp", "interpoler", "erreur_pose", "est_homogene",
    "dh_standard", "dh_modifie", "formater",
]

EPS = 1e-12


# ---------------------------------------------------------------------------
# Constructeurs elementaires
# ---------------------------------------------------------------------------

def identite() -> np.ndarray:
    """Matrice homogene identite."""
    return np.eye(4, dtype=float)


def translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Translation pure Trans(x, y, z)."""
    T = np.eye(4, dtype=float)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def rot_x(theta: float) -> np.ndarray:
    """Rotation pure autour de l'axe X, angle en radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rot_y(theta: float) -> np.ndarray:
    """Rotation pure autour de l'axe Y, angle en radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=float)


def rot_z(theta: float) -> np.ndarray:
    """Rotation pure autour de l'axe Z, angle en radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def rot_axe(axe, theta: float) -> np.ndarray:
    """Rotation d'angle theta autour d'un axe quelconque (formule de Rodrigues)."""
    u = np.asarray(axe, dtype=float)
    n = np.linalg.norm(u)
    if n < EPS:
        raise ValueError("axe de rotation nul")
    u = u / n
    K = np.array([[0.0, -u[2], u[1]],
                  [u[2], 0.0, -u[0]],
                  [-u[1], u[0], 0.0]])
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return depuis_rp(R, (0.0, 0.0, 0.0))


def depuis_rp(R, p) -> np.ndarray:
    """Construit T a partir d'une rotation 3x3 et d'une position 3x1."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


# ---------------------------------------------------------------------------
# Extraction / operations
# ---------------------------------------------------------------------------

def rotation_de(T) -> np.ndarray:
    """Partie rotation 3x3 de T."""
    return np.asarray(T, dtype=float)[:3, :3].copy()


def position_de(T) -> np.ndarray:
    """Partie position 3x1 de T (mm)."""
    return np.asarray(T, dtype=float)[:3, 3].copy()


def inverse(T) -> np.ndarray:
    """
    Inverse analytique d'une matrice homogene :

        T^-1 = [ R^T   -R^T p ]
               [  0       1   ]

    Bien plus rapide et plus stable que np.linalg.inv.
    """
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def composer(*matrices) -> np.ndarray:
    """Produit T1 @ T2 @ ... @ Tn."""
    out = np.eye(4, dtype=float)
    for T in matrices:
        out = out @ np.asarray(T, dtype=float)
    return out


def transformer_point(T, p) -> np.ndarray:
    """Applique T a un point 3D (coordonnees homogenes implicites)."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def transformer_points(T, P) -> np.ndarray:
    """Applique T a un nuage de points de forme (N, 3)."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    return P @ T[:3, :3].T + T[:3, 3]


def est_homogene(T, tol: float = 1e-6) -> bool:
    """Verifie que T est bien une matrice homogene valide (R orthonormale directe)."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return abs(np.linalg.det(R) - 1.0) < tol


# ---------------------------------------------------------------------------
# Representations de l'orientation
# ---------------------------------------------------------------------------

def rpy_vers_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Angles de Roulis-Tangage-Lacet (convention XYZ fixes) vers matrice homogene.

        R = Rz(yaw) . Ry(pitch) . Rx(roll)
    """
    return composer(rot_z(yaw), rot_y(pitch), rot_x(roll))


def rotation_vers_rpy(T) -> tuple:
    """Matrice -> (roll, pitch, yaw) en radians. Gere le blocage de cardan."""
    R = rotation_de(T)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) > 1.0 - 1e-9:          # blocage de cardan
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def rotation_vers_axe_angle(T) -> tuple:
    """Matrice -> (axe unitaire, angle en radians)."""
    R = rotation_de(T)
    cos_t = (np.trace(R) - 1.0) / 2.0
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if theta > np.pi - 1e-6:
        # Cas theta = pi : R est symetrique, on lit l'axe sur la diagonale
        d = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(d))
        axe = np.zeros(3)
        axe[k] = d[k]
        for i in range(3):
            if i != k:
                axe[i] = (R[i, k] + R[k, i]) / (4.0 * d[k])
        return axe / np.linalg.norm(axe), theta
    axe = np.array([R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return axe, theta


def axe_angle_vers_rotation(axe, angle: float) -> np.ndarray:
    """(axe, angle) -> matrice homogene."""
    return rot_axe(axe, angle)


def rotation_vers_quaternion(T) -> np.ndarray:
    """Matrice -> quaternion [w, x, y, z] unitaire (methode de Shepperd)."""
    R = rotation_de(T)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def quaternion_vers_rotation(q) -> np.ndarray:
    """Quaternion [w, x, y, z] -> matrice homogene."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return depuis_rp(R, (0, 0, 0))


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def slerp(T0, T1, s: float) -> np.ndarray:
    """
    Interpolation spherique lineaire entre deux orientations.

    s = 0 -> orientation de T0, s = 1 -> orientation de T1.
    La position renvoyee est nulle : seule la rotation est interpolee.
    """
    q0 = rotation_vers_quaternion(T0)
    q1 = rotation_vers_quaternion(T1)
    d = float(np.dot(q0, q1))
    if d < 0.0:                 # chemin le plus court
        q1 = -q1
        d = -d
    d = float(np.clip(d, -1.0, 1.0))
    if d > 1.0 - 1e-9:          # quasi colineaires : interpolation lineaire
        q = q0 + s * (q1 - q0)
    else:
        theta = np.arccos(d)
        st = np.sin(theta)
        q = (np.sin((1.0 - s) * theta) / st) * q0 + (np.sin(s * theta) / st) * q1
    return quaternion_vers_rotation(q)


def interpoler(T0, T1, s: float) -> np.ndarray:
    """
    Interpolation d'une pose complete : position lineaire + orientation SLERP.

    C'est l'operation de base d'un mouvement cartesien rectiligne (MOVL).
    """
    s = float(np.clip(s, 0.0, 1.0))
    p0 = position_de(T0)
    p1 = position_de(T1)
    R = rotation_de(slerp(T0, T1, s))
    return depuis_rp(R, p0 + s * (p1 - p0))


def erreur_pose(T_mes, T_ref) -> tuple:
    """
    Ecart entre deux poses : (erreur de position en mm, erreur d'orientation en rad).
    """
    dp = position_de(T_mes) - position_de(T_ref)
    dR = rotation_de(T_ref).T @ rotation_de(T_mes)
    _, angle = rotation_vers_axe_angle(depuis_rp(dR, (0, 0, 0)))
    return float(np.linalg.norm(dp)), float(angle)


# ---------------------------------------------------------------------------
# Parametres de Denavit-Hartenberg
# ---------------------------------------------------------------------------

def dh_standard(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Matrice de passage de Denavit-Hartenberg classique (convention de 1955) :

        T = Rot_z(theta) . Trans_z(d) . Trans_x(a) . Rot_x(alpha)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def dh_modifie(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """
    Matrice de passage de Denavit-Hartenberg modifiee (Khalil-Kleinfinger),
    convention majoritairement enseignee en France :

        T = Rot_x(alpha) . Trans_x(a) . Rot_z(theta) . Trans_z(d)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st, 0.0, a],
        [ca * st, ca * ct, -sa, -d * sa],
        [sa * st, sa * ct, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def formater(T, decimales: int = 3, titre: str = "") -> str:
    """Rend une matrice homogene lisible dans une console ou un QTextEdit."""
    T = np.asarray(T, dtype=float)
    lignes = []
    if titre:
        lignes.append(titre)
    for i in range(4):
        cellules = []
        for j in range(4):
            v = T[i, j]
            if abs(v) < 10.0 ** (-decimales):
                v = 0.0
            cellules.append("{0:>10.{1}f}".format(v, decimales))
        lignes.append("[" + " ".join(cellules) + " ]")
    return "\n".join(lignes)
