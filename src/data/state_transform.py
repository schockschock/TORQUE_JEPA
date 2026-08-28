"""Observation-End Frame state transform.

Primary methods condition on the observation-end COM world location while
predicting position and orientation relative to the observation end. Observed
state carries absolute orientation; future targets carry orientation relative to
the observation-end actor-to-world rotation ``R_e``.

State vector layout (15 dims): ``[position(3) | rot6d(6) | linear velocity(3) |
angular velocity(3)]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.rotations import matrix_to_rot6d, quat_wxyz_to_matrix, rot6d_to_matrix

STATE_DIM = 15
POS_DIM = 3
ROT6D_DIM = 6
VEL_DIM = 3
ANGVEL_DIM = 3


@dataclass
class TrajectoryState:
    """Synchronized per-frame trajectory state loaded from a ``state.npz``."""

    time_s: np.ndarray
    position_com_world_m: np.ndarray
    orientation_actor_to_world_wxyz: np.ndarray
    linear_velocity_com_world_m_s: np.ndarray
    angular_velocity_actor_rad_s: np.ndarray


def _rotation_matrices(state: TrajectoryState) -> np.ndarray:
    return quat_wxyz_to_matrix(state.orientation_actor_to_world_wxyz)


def observed_state(state: TrajectoryState, e: int, indices: list[int]) -> np.ndarray:
    """Observed kinematic state at native frames ``indices`` (all ``t <= e``)."""
    p_e = state.position_com_world_m[e]
    R = _rotation_matrices(state)
    out = np.empty((len(indices), STATE_DIM), dtype=np.float64)
    out[:, 0:3] = state.position_com_world_m[indices] - p_e
    out[:, 3:9] = matrix_to_rot6d(R[indices])
    out[:, 9:12] = state.linear_velocity_com_world_m_s[indices]
    out[:, 12:15] = state.angular_velocity_actor_rad_s[indices]
    return out


def target_state(state: TrajectoryState, e: int, indices: list[int]) -> np.ndarray:
    """Future target state at native frames ``indices`` (all ``t > e``)."""
    p_e = state.position_com_world_m[e]
    R = _rotation_matrices(state)
    R_e = R[e]
    R_rel = np.einsum("ij,njk->nik", R_e.T, R[indices])
    out = np.empty((len(indices), STATE_DIM), dtype=np.float64)
    out[:, 0:3] = state.position_com_world_m[indices] - p_e
    out[:, 3:9] = matrix_to_rot6d(R_rel)
    out[:, 9:12] = state.linear_velocity_com_world_m_s[indices]
    out[:, 12:15] = state.angular_velocity_actor_rad_s[indices]
    return out


def observation_end(state: TrajectoryState, e: int) -> dict:
    """Observation-end world location and actor-to-world rotation."""
    R = _rotation_matrices(state)
    return {
        "p_com_world": state.position_com_world_m[e].copy(),
        "r_actor_to_world": R[e].copy(),
    }


def reconstruct_world_positions(
    relative_positions: np.ndarray, p_com_world_e: np.ndarray
) -> np.ndarray:
    """Add the observation-end world location back to relative displacements."""
    return relative_positions + p_com_world_e


def reconstruct_orientations(
    r_actor_to_world_e: np.ndarray, rot6d_relative: np.ndarray
) -> np.ndarray:
    """Recover world orientations from relative rot6d: ``R_t = R_e R_relative``."""
    R_rel = rot6d_to_matrix(rot6d_relative)
    return np.einsum("ij,...jk->...ik", r_actor_to_world_e, R_rel)
