import numpy as np
import pytest

from data.rotations import quat_wxyz_to_matrix
from data.state_transform import (
    TrajectoryState,
    observed_state,
    observation_end,
    reconstruct_world_positions,
    reconstruct_orientations,
    target_state,
)


def _rot_z(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def make_state(n=200, seed=0):
    rng = np.random.default_rng(seed)
    time_s = np.arange(n) / 120.0
    t = time_s
    # freefall-like parabola with a horizontal drift
    position = np.stack(
        [
            0.3 * t,
            0.1 * t,
            1.0 - 0.5 * 9.81 * t**2,
        ],
        axis=-1,
    )
    # smoothly varying rotation about z plus a fixed tilt
    thetas = 0.2 * t
    quats = []
    for theta in thetas:
        qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
        qx = np.array([np.cos(0.3 / 2), np.sin(0.3 / 2), 0.0, 0.0])
        # compose qx * qz (active rotation: tilt then spin)
        w1, x1, y1, z1 = qx
        w2, x2, y2, z2 = qz
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        quats.append([w, x, y, z])
    quats = np.array(quats)
    lin_vel = np.stack([0.3 * np.ones(n), 0.1 * np.ones(n), -9.81 * t], axis=-1)
    ang_vel = np.stack([0.2 * np.ones(n), np.zeros(n), np.zeros(n)], axis=-1)
    return TrajectoryState(
        time_s=time_s,
        position_com_world_m=position,
        orientation_actor_to_world_wxyz=quats,
        linear_velocity_com_world_m_s=lin_vel,
        angular_velocity_actor_rad_s=ang_vel,
    )


def test_observed_position_is_relative_to_end():
    st = make_state()
    e = 60
    idx = list(range(30, 61, 2))
    obs = observed_state(st, e, idx)
    p_e = st.position_com_world_m[e]
    assert obs.shape == (16, 15)
    assert np.allclose(obs[:, 0:3], st.position_com_world_m[idx] - p_e)


def test_observed_rotation_is_absolute_rot6d():
    st = make_state()
    e = 60
    idx = list(range(30, 61, 2))
    obs = observed_state(st, e, idx)
    R = quat_wxyz_to_matrix(st.orientation_actor_to_world_wxyz)
    expected = np.concatenate([R[idx][:, :, 0], R[idx][:, :, 1]], axis=-1)
    assert np.allclose(obs[:, 3:9], expected)


def test_observed_velocity_and_angular_velocity_passthrough():
    st = make_state()
    e = 60
    idx = list(range(30, 61, 2))
    obs = observed_state(st, e, idx)
    assert np.allclose(obs[:, 9:12], st.linear_velocity_com_world_m_s[idx])
    assert np.allclose(obs[:, 12:15], st.angular_velocity_actor_rad_s[idx])


def test_target_position_is_relative_to_end():
    st = make_state()
    e = 60
    idx = list(range(62, 121, 2))
    tgt = target_state(st, e, idx)
    p_e = st.position_com_world_m[e]
    assert tgt.shape == (30, 15)
    assert np.allclose(tgt[:, 0:3], st.position_com_world_m[idx] - p_e)


def test_target_rotation_is_relative():
    st = make_state()
    e = 60
    idx = list(range(62, 121, 2))
    tgt = target_state(st, e, idx)
    R = quat_wxyz_to_matrix(st.orientation_actor_to_world_wxyz)
    R_e = R[e]
    R_rel = np.einsum("ij,njk->nik", R_e.T, R[idx])
    expected = np.concatenate([R_rel[:, :, 0], R_rel[:, :, 1]], axis=-1)
    assert np.allclose(tgt[:, 3:9], expected)


def test_target_relative_rotation_identity_when_static():
    n = 200
    st = TrajectoryState(
        time_s=np.arange(n) / 120.0,
        position_com_world_m=np.zeros((n, 3)),
        orientation_actor_to_world_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        linear_velocity_com_world_m_s=np.zeros((n, 3)),
        angular_velocity_actor_rad_s=np.zeros((n, 3)),
    )
    e = 60
    idx = list(range(62, 121, 2))
    tgt = target_state(st, e, idx)
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(tgt[:, 3:9], identity_rot6d)


def test_observation_end_fields():
    st = make_state()
    e = 60
    oe = observation_end(st, e)
    R = quat_wxyz_to_matrix(st.orientation_actor_to_world_wxyz)
    assert np.allclose(oe["p_com_world"], st.position_com_world_m[e])
    assert np.allclose(oe["r_actor_to_world"], R[e])


def test_reconstruct_world_positions_round_trip():
    st = make_state()
    e = 60
    idx = list(range(62, 121, 2))
    tgt = target_state(st, e, idx)
    world = reconstruct_world_positions(tgt[:, 0:3], st.position_com_world_m[e])
    assert np.allclose(world, st.position_com_world_m[idx])


def test_reconstruct_orientations_round_trip():
    st = make_state()
    e = 60
    idx = list(range(62, 121, 2))
    tgt = target_state(st, e, idx)
    R = quat_wxyz_to_matrix(st.orientation_actor_to_world_wxyz)
    R_e = R[e]
    world = reconstruct_orientations(R_e, tgt[:, 3:9])
    assert world.shape == (30, 3, 3)
    assert np.allclose(world, R[idx], atol=1e-6)


def test_reconstruct_orientations_valid_so3():
    st = make_state()
    e = 60
    idx = list(range(62, 121, 2))
    tgt = target_state(st, e, idx)
    world = reconstruct_orientations(
        quat_wxyz_to_matrix(st.orientation_actor_to_world_wxyz)[e], tgt[:, 3:9]
    )
    for i in range(len(world)):
        assert np.allclose(world[i] @ world[i].T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(world[i]), 1.0, atol=1e-6)
