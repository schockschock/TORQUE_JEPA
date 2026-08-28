"""Integration tests against the real TORQUE pilot data.

Skipped when the dataset root is absent (e.g. CI, other machines).
"""

import os

import numpy as np
import pytest

from data.rotations import quat_wxyz_to_matrix
from data.state_transform import reconstruct_world_positions, reconstruct_orientations
from data.TorqueDataset import (
    TorqueDataset,
    TrajectoryRecord,
    load_trajectory_state,
)

DATA_ROOT = os.environ.get("TORQUE_DATA_ROOT", "/data2/adrien/TORQUE")
TRAJ_DIR = os.path.join(DATA_ROOT, "v2", "freefall", "2R1-1", "pose0_angvel0")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TRAJ_DIR), reason="TORQUE pilot data not available"
)


def _record():
    return TrajectoryRecord(
        trajectory_id="freefall/2R1-1/pose0_angvel0",
        object_id="2R1-1",
        scenario="freefall",
        condition_id="pose0_angvel0",
        split="train",
        state_path=os.path.join(TRAJ_DIR, "state.npz"),
    )


def test_real_state_loads_and_round_trips():
    rec = _record()
    ds = TorqueDataset([rec])
    assert len(ds) > 0

    state = load_trajectory_state(rec.state_path)
    n = len(state.time_s)

    for idx in [0, len(ds) // 2, len(ds) - 1]:
        sample = ds[idx]
        _, e = ds._window_index[idx]
        fut_idx = [e + off for off in range(2, 61, 2)]
        R = quat_wxyz_to_matrix(state.orientation_actor_to_world_wxyz)

        world_pos = reconstruct_world_positions(
            sample["target"]["state"][:, 0:3], sample["observation_end"]["p_com_world"]
        )
        world_rot = reconstruct_orientations(
            sample["observation_end"]["r_actor_to_world"], sample["target"]["state"][:, 3:9]
        )

        assert np.allclose(world_pos, state.position_com_world_m[fut_idx], atol=1e-4)
        assert np.allclose(world_rot, R[fut_idx], atol=1e-4)


def test_real_observed_rot6d_matches_source():
    rec = _record()
    ds = TorqueDataset([rec])
    raw = np.load(rec.state_path, allow_pickle=True)
    sample = ds[0]
    _, e = ds._window_index[0]
    obs_idx = [e + off for off in range(-30, 1, 2)]
    assert np.allclose(sample["past"]["state"][:, 3:9], raw["rotation_actor_to_world_6d"][obs_idx], atol=1e-6)


def test_real_timestamp_spans():
    rec = _record()
    ds = TorqueDataset([rec])
    sample = ds[0]
    obs_ts = sample["past"]["timestamps"]
    fut_ts = sample["target"]["timestamps"]
    assert np.isclose(obs_ts[-1] - obs_ts[0], 0.25)
    assert np.isclose(fut_ts[-1] - fut_ts[0], 0.50 - 2 / 120.0)
    assert np.isclose(fut_ts[0] - obs_ts[-1], 2 / 120.0)
