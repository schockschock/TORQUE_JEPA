import numpy as np
import pytest

from data.state_transform import TrajectoryState
from data.TorqueDataset import (
    TorqueDataset,
    TrajectoryGroupedBatchSampler,
    TrajectoryRecord,
)
from data.rotations import quat_wxyz_to_matrix


def _synthetic_state(n=200, seed=0):
    rng = np.random.default_rng(seed)
    time_s = np.arange(n) / 120.0
    t = time_s
    position = np.stack([0.3 * t, 0.1 * t, 1.0 - 0.5 * 9.81 * t**2], axis=-1)
    quats = []
    for theta in 0.2 * t:
        qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
        qx = np.array([np.cos(0.3 / 2), np.sin(0.3 / 2), 0.0, 0.0])
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


def _make_records(n=3):
    return [
        TrajectoryRecord(
            trajectory_id=f"freefall/obj{i}/pose0_angvel0",
            object_id=f"obj{i}",
            scenario="freefall",
            condition_id="pose0_angvel0",
            split="train",
            state_path=f"/fake/obj{i}/state.npz",
        )
        for i in range(n)
    ]


def _loader(records):
    states = {rec.trajectory_id: _synthetic_state(seed=i) for i, rec in enumerate(records)}
    return lambda rec: states[rec.trajectory_id]


def _make_dataset(n=3, stats=None):
    records = _make_records(n)
    ds = TorqueDataset(records, normalization_stats=stats, state_loader=_loader(records))
    return ds, records


def test_window_index_is_deterministic():
    ds1, _ = _make_dataset(3)
    ds2, _ = _make_dataset(3)
    assert ds1.window_record_indices == ds2.window_record_indices
    assert [e for _, e in ds1._window_index] == [e for _, e in ds2._window_index]


def test_sample_structure_and_shapes():
    ds, _ = _make_dataset(3)
    s = ds[0]
    assert s["past"]["state"].shape == (16, 15)
    assert s["past"]["timestamps"].shape == (16,)
    assert s["past"]["availability_mask"].shape == (16, 3)
    assert s["past"]["rgb_a"] is None
    assert s["past"]["depth_a"] is None
    assert s["target"]["state"].shape == (30, 15)
    assert s["target"]["timestamps"].shape == (30,)
    assert s["observation_end"]["p_com_world"].shape == (3,)
    assert s["observation_end"]["r_actor_to_world"].shape == (3, 3)
    assert s["static"]["geometry_available"] is False
    assert s["static"]["utonia_global_mean"] is None


def test_sample_metadata():
    ds, records = _make_dataset(3)
    s = ds[0]
    m = s["metadata"]
    assert m["object_id"] == "obj0"
    assert m["trajectory_id"] == "freefall/obj0/pose0_angvel0"
    assert m["scenario"] == "freefall"
    assert m["condition_id"] == "pose0_angvel0"
    assert m["split"] == "train"


def test_sample_world_reconstruction_round_trip():
    ds, records = _make_dataset(3)
    states = {rec.trajectory_id: _synthetic_state(seed=i) for i, rec in enumerate(records)}
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        s = ds[idx]
        rec_idx, e = ds._window_index[idx]
        rec = records[rec_idx]
        st = states[rec.trajectory_id]
        p_e = s["observation_end"]["p_com_world"]
        R_e = s["observation_end"]["r_actor_to_world"]
        # reconstruct target positions and orientations
        from data.state_transform import reconstruct_world_positions, reconstruct_orientations

        world_pos = reconstruct_world_positions(s["target"]["state"][:, 0:3], p_e)
        world_rot = reconstruct_orientations(R_e, s["target"]["state"][:, 3:9])
        fut_idx = [e + off for off in range(2, 61, 2)]
        R = quat_wxyz_to_matrix(st.orientation_actor_to_world_wxyz)
        assert np.allclose(world_pos, st.position_com_world_m[fut_idx])
        assert np.allclose(world_rot, R[fut_idx], atol=1e-6)


def test_sampler_no_duplicate_trajectory_per_batch():
    ds, _ = _make_dataset(5)
    sampler = TrajectoryGroupedBatchSampler(ds, batch_size=4, seed=0)
    for batch in sampler:
        recs = [ds._window_index[i][0] for i in batch]
        assert len(recs) == len(set(recs))


def test_sampler_uses_every_window_exactly_once():
    ds, _ = _make_dataset(5)
    sampler = TrajectoryGroupedBatchSampler(ds, batch_size=4, seed=1)
    seen = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(ds)))


def test_sampler_deterministic_given_seed():
    ds, _ = _make_dataset(5)
    a = [list(b) for b in TrajectoryGroupedBatchSampler(ds, batch_size=4, seed=7)]
    b = [list(b) for b in TrajectoryGroupedBatchSampler(ds, batch_size=4, seed=7)]
    assert a == b


def test_sampler_len_matches_iteration():
    ds, _ = _make_dataset(5)
    for drop_last in [False, True]:
        sampler = TrajectoryGroupedBatchSampler(ds, batch_size=4, seed=3, drop_last=drop_last)
        assert len(sampler) == sum(1 for _ in sampler)


def test_sampler_drop_last_yields_full_batches_or_none():
    ds, _ = _make_dataset(5)
    sampler = TrajectoryGroupedBatchSampler(ds, batch_size=4, seed=2, drop_last=True)
    for batch in sampler:
        assert len(batch) == 4


def test_dataset_requires_enough_frames():
    records = [
        TrajectoryRecord(
            trajectory_id="freefall/obj0/pose0_angvel0",
            object_id="obj0",
            scenario="freefall",
            condition_id="pose0_angvel0",
            split="train",
            state_path="/fake/obj0/state.npz",
        )
    ]
    short = TrajectoryState(
        time_s=np.arange(50) / 120.0,
        position_com_world_m=np.zeros((50, 3)),
        orientation_actor_to_world_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (50, 1)),
        linear_velocity_com_world_m_s=np.zeros((50, 3)),
        angular_velocity_actor_rad_s=np.zeros((50, 3)),
    )
    ds = TorqueDataset(records, state_loader=lambda rec: short)
    assert len(ds) == 0
