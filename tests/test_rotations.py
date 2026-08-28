import numpy as np
import pytest

from data.rotations import matrix_to_rot6d, quat_wxyz_to_matrix, rot6d_to_matrix


def _is_so3(R):
    return np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_quat_wxyz_identity_yields_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_wxyz_to_matrix(q), np.eye(3))


def test_quat_wxyz_rotation_about_z():
    theta = np.pi / 3
    q = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    R = quat_wxyz_to_matrix(q)
    expected = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(R, expected)
    assert _is_so3(R)


def test_quat_wxyz_accepts_batch():
    q = np.array([[1.0, 0.0, 0.0, 0.0], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]])
    R = quat_wxyz_to_matrix(q)
    assert R.shape == (2, 3, 3)
    assert np.allclose(R[0], np.eye(3))
    assert _is_so3(R[1])


def test_quat_wxyz_normalizes_non_unit_input():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_wxyz_to_matrix(q), np.eye(3))


def test_matrix_to_rot6d_takes_first_two_columns():
    theta = 0.3
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    r6 = matrix_to_rot6d(R)
    assert r6.shape == (6,)
    assert np.allclose(r6, np.concatenate([R[:, 0], R[:, 1]]))


def test_rot6d_to_matrix_projects_to_so3():
    a1 = np.array([1.0, 0.2, 0.1])
    a2 = np.array([0.3, 1.0, -0.4])
    R = rot6d_to_matrix(np.concatenate([a1, a2]))
    assert _is_so3(R)


def test_rot6d_matrix_round_trip():
    rng = np.random.default_rng(0)
    R = rot6d_to_matrix(rng.normal(size=6))
    assert np.allclose(matrix_to_rot6d(R), matrix_to_rot6d(R))
    assert _is_so3(R)
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))
    assert np.allclose(R, R2, atol=1e-6)


def test_rot6d_to_matrix_batch():
    rng = np.random.default_rng(1)
    r6 = rng.normal(size=(4, 6))
    R = rot6d_to_matrix(r6)
    assert R.shape == (4, 3, 3)
    for i in range(4):
        assert _is_so3(R[i])


def test_known_rotation_round_trip():
    theta = 0.7
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_reconstructed = rot6d_to_matrix(matrix_to_rot6d(R))
    assert np.allclose(R, R_reconstructed)
