"""Rotation and orientation conventions.

TORQUE stores orientation as an active actor-to-world WXYZ quaternion and, for
model state, as the 6D (Gram-Schmidt) representation ``rot6d`` of that rotation.
All functions operate on ``numpy`` arrays; leading batch dimensions are
preserved.
"""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternion(s) to 3x3 active rotation matrix.

    Args:
        q: ``(..., 4)`` array of ``[w, x, y, z]`` quaternions.

    Returns:
        ``(..., 3, 3)`` rotation matrices.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[..., 0, 1] = 2.0 * (xy - wz)
    out[..., 0, 2] = 2.0 * (xz + wy)
    out[..., 1, 0] = 2.0 * (xy + wz)
    out[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[..., 1, 2] = 2.0 * (yz - wx)
    out[..., 2, 0] = 2.0 * (xz - wy)
    out[..., 2, 1] = 2.0 * (yz + wx)
    out[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Encode a 3x3 rotation matrix as the first two columns (``rot6d``).

    Args:
        R: ``(..., 3, 3)`` rotation matrices.

    Returns:
        ``(..., 6)`` rot6d vectors.
    """
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Project a rot6d vector onto SO(3) via Gram-Schmidt.

    Args:
        r6: ``(..., 6)`` rot6d vectors ``[a1 | a2]``.

    Returns:
        ``(..., 3, 3)`` valid SO(3) rotation matrices.
    """
    r6 = np.asarray(r6, dtype=np.float64)
    a1 = r6[..., :3]
    a2 = r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)
