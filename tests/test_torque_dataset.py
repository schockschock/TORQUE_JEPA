import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from data.TorqueDataset import (
    DatasetConfig,
    TorqueDataset,
    TrajectoryBatchSampler,
    make_dataloader,
    rot6d_to_matrix,
    torque_collate,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _write_trajectory(
    root,
    scenario,
    object_id,
    condition_id,
    *,
    split="train",
    frame_count=100,
    visible=True,
    depth_value=2.0,
    status="completed",
):
    run = root / "v2" / scenario / object_id / condition_id
    run.mkdir(parents=True, exist_ok=True)
    frames = np.arange(frame_count, dtype=np.int64)
    times = frames / 120.0
    q = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    rotation = _quaternion_to_matrix(q)
    rotations = np.repeat(rotation[None], frame_count, axis=0)
    rotation_6d = np.concatenate((rotations[:, :, 0], rotations[:, :, 1]), axis=1)
    position = np.stack((frames * 0.01, frames * 0.02, 1.0 + frames * 0.03), axis=1)
    velocity = np.tile(np.array([1.2, 2.4, 3.6]), (frame_count, 1))
    omega = np.tile(np.array([0.1, 0.2, 0.3]), (frame_count, 1))
    np.savez(
        run / "state.npz",
        frame_index=frames,
        time_s=times,
        position_com_world_m=position,
        orientation_actor_to_world_wxyz=np.tile(q, (frame_count, 1)),
        rotation_actor_to_world_6d=rotation_6d,
        linear_velocity_com_world_m_s=velocity,
        angular_velocity_actor_rad_s=omega,
        angular_velocity_world_rad_s=omega,
        contact_active=np.zeros(frame_count, dtype=bool),
    )
    np.savez(
        run / "modality_timestamps.npz",
        frame_index=frames.astype(np.int32),
        state_time_s=times,
        rgb_a_time_s=times,
        depth_a_time_s=times,
        frustum_visible_a=np.full(frame_count, visible, dtype=bool),
        visible_pixels_a=np.full(frame_count, 100, dtype=np.int32),
    )
    np.savez(run / "contacts.npz", frame_index=frames)
    _write_json(run / "events.json", {})
    _write_json(
        run / "metadata.json",
        {
            "status": status,
            "frames_captured": frame_count,
            "dataset_version": "2.0",
            "object_id": object_id,
            "condition_id": condition_id,
            "scenario": scenario,
            "object_split": split,
        },
    )
    _write_json(run / "validation.json", {"valid": True, "frame_count": frame_count})
    rgb = run / "rgb_a"
    depth = run / "depth_a"
    rgb.mkdir(exist_ok=True)
    depth.mkdir(exist_ok=True)
    for frame in frames:
        Image.new("RGB", (8, 6), (frame % 255, 20, 30)).save(rgb / f"frame_{frame:06d}.jpg")
        np.savez(depth / f"frame_{frame:06d}.npz", depth=np.full((6, 8), depth_value, dtype=np.float32))


def _make_root(tmp_path, trajectories):
    root = tmp_path / "torque"
    manifests = root / "v2" / "manifests"
    manifests.mkdir(parents=True)
    conditions = []
    split_objects = {"train": [], "validation": [], "test": []}
    for scenario, object_id, condition_id, split in trajectories:
        conditions.append(
            {
                "scenario": scenario,
                "object_id": object_id,
                "condition_id": condition_id,
                "parameters": {},
            }
        )
        if object_id not in split_objects[split]:
            split_objects[split].append(object_id)
        (root / "assets").mkdir(exist_ok=True)
        (root / "3D_features").mkdir(exist_ok=True)
        (root / "assets" / f"{object_id}.json").write_text("{}")
        np.savez(
            root / "3D_features" / f"{object_id}.npz",
            global_mean=np.full(1386, float(len(split_objects[split])), dtype=np.float32),
            global_max=np.full(1386, 2.0, dtype=np.float32),
        )
    _write_json(manifests / "ordinary_schedule.json", {"conditions": conditions})
    _write_json(manifests / "object_split.json", {"splits": split_objects})
    return root


def _config(root, **kwargs):
    values = {"data_root": str(root), "split": "train", "modalities": ("state",)}
    values.update(kwargs)
    return DatasetConfig(**values)


def test_dataset_freezes_ordered_cohort_and_uses_only_eligible_trajectories(tmp_path):
    trajectories = [
        ("freefall", "obj-b", "condition-2", "train"),
        ("freefall", "obj-a", "condition-1", "train"),
        ("freefall", "obj-c", "condition-3", "validation"),
    ]
    root = _make_root(tmp_path, trajectories)
    for scenario, object_id, condition_id, split in trajectories:
        _write_trajectory(root, scenario, object_id, condition_id, split=split)
    _write_trajectory(root, "freefall", "obj-d", "too-short", frame_count=90)

    dataset = TorqueDataset(_config(root))
    assert dataset.trajectory_ids == (
        "freefall/obj-a/condition-1",
        "freefall/obj-b/condition-2",
    )
    assert dataset.trajectory_paths == tuple(record.run_dir for record in dataset.cohort)
    old_windows = len(dataset)

    _write_trajectory(root, "freefall", "obj-a", "condition-later", split="train")
    assert dataset.trajectory_ids == tuple(record.trajectory_id for record in dataset.cohort)
    assert len(dataset) == old_windows


def test_state_sample_has_documented_shapes_and_reconstructs_world_trajectory(tmp_path):
    root = _make_root(tmp_path, [("freefall", "obj-a", "condition-1", "train")])
    _write_trajectory(root, "freefall", "obj-a", "condition-1")
    dataset = TorqueDataset(_config(root))

    sample = dataset[0]
    assert sample["past"]["state"].shape == (16, 15)
    assert sample["target"]["state"].shape == (30, 15)
    assert sample["past"]["timestamps"].shape == (16,)
    assert sample["target"]["timestamps"].shape == (30,)
    assert sample["past"]["state"].dtype == torch.float32
    assert np.allclose(sample["past"]["state"][-1, :3].numpy(), 0.0)

    end = sample["observation_end"]
    reconstructed_position = end["p_com_world"] + sample["target"]["state"][:, :3]
    assert np.allclose(
        reconstructed_position.numpy(),
        np.stack((np.arange(32, 92, 2) * 0.01, np.arange(32, 92, 2) * 0.02, 1.0 + np.arange(32, 92, 2) * 0.03), axis=1),
    )
    reconstructed_rotation = np.einsum(
        "ij,njk->nik",
        end["r_actor_to_world"].numpy(),
        rot6d_to_matrix(sample["target"]["state"][:, 3:9].numpy()),
    )
    assert np.allclose(reconstructed_rotation, np.repeat(end["r_actor_to_world"].numpy()[None], 30, axis=0))


def test_modalities_are_lazy_and_camera_visibility_drops_invalid_windows(tmp_path):
    root = _make_root(tmp_path, [("freefall", "obj-a", "condition-1", "train")])
    _write_trajectory(root, "freefall", "obj-a", "condition-1", visible=False)
    dataset = TorqueDataset(
        _config(root, modalities=("state", "rgb_a", "depth_a"), invalid_window_policy="drop")
    )
    assert len(dataset) == 0

    _write_trajectory(root, "freefall", "obj-a", "condition-1", visible=True)
    dataset = TorqueDataset(_config(root, modalities=("state", "rgb_a", "depth_a")))
    sample = dataset[0]
    assert sample["past"]["rgb_a"].shape == (16, 3, 224, 224)
    assert sample["past"]["depth_a"].shape == (16, 1, 224, 224)
    assert sample["past"]["depth_valid_a"].dtype == sample["past"]["availability_mask"].dtype
    assert sample["past"]["availability_mask"].shape == (16, 3)
    assert sample["past"]["availability_mask"][:, 1:].all()

    geometry_dataset = TorqueDataset(_config(root, modalities=("state", "utonia")))
    geometry_sample = geometry_dataset[0]
    assert geometry_sample["static"]["utonia_global_mean"].shape == (1386,)


def test_training_batch_sampler_never_repeats_a_trajectory(tmp_path):
    trajectories = [
        ("freefall", "obj-a", "condition-1", "train"),
        ("freefall", "obj-b", "condition-2", "train"),
    ]
    root = _make_root(tmp_path, trajectories)
    for scenario, object_id, condition_id, split in trajectories:
        _write_trajectory(root, scenario, object_id, condition_id, split=split)
    dataset = TorqueDataset(_config(root))
    sampler = TrajectoryBatchSampler(dataset, batch_size=2, shuffle=False)

    for batch in sampler:
        ids = [dataset.windows[index].trajectory_id for index in batch]
        assert len(ids) == len(set(ids))

    one_trajectory_sampler = TrajectoryBatchSampler(dataset, batch_size=100)
    assert len(one_trajectory_sampler) == len(list(one_trajectory_sampler))

    batch = torque_collate([dataset[0], dataset[1]])
    assert batch["past"]["state"].shape == (2, 16, 15)
    assert batch["past"]["rgb_a"] is None


def test_evaluation_uses_one_horizon_stride_and_dataloader_collates(tmp_path):
    trajectories = [
        ("freefall", "obj-a", "condition-1", "train"),
        ("freefall", "obj-b", "condition-2", "train"),
    ]
    root = _make_root(tmp_path, trajectories)
    for scenario, object_id, condition_id, split in trajectories:
        _write_trajectory(root, scenario, object_id, condition_id, split=split, frame_count=160)
    dataset = TorqueDataset(_config(root, mode="evaluation"))
    assert [window.native_frame_end for window in dataset.windows] == [30, 90, 30, 90]

    loader = make_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["past"]["state"].shape == (2, 16, 15)
    assert batch["target"]["state"].shape == (2, 30, 15)
    assert batch["metadata"]["trajectory_id"] == list(dataset.trajectory_ids)


def test_normalization_uses_training_objects_for_validation_samples(tmp_path):
    trajectories = [
        ("freefall", "obj-train", "condition-1", "train"),
        ("freefall", "obj-validation", "condition-2", "validation"),
    ]
    root = _make_root(tmp_path, trajectories)
    _write_trajectory(root, "freefall", "obj-train", "condition-1", split="train", depth_value=2.0)
    _write_trajectory(
        root,
        "freefall",
        "obj-validation",
        "condition-2",
        split="validation",
        depth_value=10.0,
    )
    dataset = TorqueDataset(
        _config(root, split="validation", modalities=("state", "depth_a"))
    )
    assert dataset.normalization_stats.depth_mean == pytest.approx(2.0)
    assert dataset[0]["past"]["depth_a"].mean().item() == pytest.approx(8.0)


def test_missing_object_split_is_not_eligible(tmp_path):
    root = _make_root(tmp_path, [("freefall", "obj-a", "condition-1", "train")])
    _write_trajectory(root, "freefall", "obj-a", "condition-1")
    split_path = root / "v2" / "manifests" / "object_split.json"
    _write_json(split_path, {"splits": {"train": []}})
    dataset = TorqueDataset(_config(root))
    assert len(dataset.cohort) == 0
    assert dataset.exclusions[0]["reason"] == "missing_object_split"
