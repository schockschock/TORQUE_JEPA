"""Unit tests for the frozen trajectory manifest.

Exercises condition decomposition, eligibility gates, determinism, summary
counts, and audit rendering against synthetic on-disk trajectories.
"""

import json
import os

import numpy as np
import pytest

from data.trajectory_manifest import (
    ManifestConfig,
    build_manifest,
    decompose_condition,
    inspect_trajectory,
    render_audit,
    sha256_bytes,
    sha256_file,
    trajectory_id,
    write_json,
)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _write_state(path, frame_count):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        frame_index=np.arange(frame_count),
        time_s=np.arange(frame_count) / 120.0,
    )


def _make_trajectory(
    run_dir,
    object_id,
    condition_id,
    frame_count=153,
    status="completed",
    valid=True,
    termination="settled",
    dataset_version="2.0",
    warnings=None,
    omit=None,
):
    omit = omit or set()
    os.makedirs(run_dir, exist_ok=True)

    if "metadata" not in omit:
        _write_json(
            os.path.join(run_dir, "metadata.json"),
            {
                "status": status,
                "termination_reason": termination,
                "dataset_version": dataset_version,
                "frames_captured": frame_count,
                "object_id": object_id,
                "condition_id": condition_id,
                "validation_warnings": warnings or [],
            },
        )
    if "validation" not in omit:
        _write_json(
            os.path.join(run_dir, "validation.json"),
            {"valid": valid, "frame_count": frame_count, "warnings": warnings or []},
        )
    if "state" not in omit:
        _write_state(os.path.join(run_dir, "state.npz"), frame_count)
    if "modality_timestamps" not in omit:
        _write_state(os.path.join(run_dir, "modality_timestamps.npz"), frame_count)
    if "contacts" not in omit:
        _write_state(os.path.join(run_dir, "contacts.npz"), frame_count)
    if "events" not in omit:
        _write_json(os.path.join(run_dir, "events.json"), {})
    for sensor in ("rgb_a", "depth_a"):
        if sensor not in omit:
            d = os.path.join(run_dir, sensor)
            os.makedirs(d, exist_ok=True)
            for i in range(frame_count):
                open(os.path.join(d, f"frame_{i:06d}.jpg"), "w").close()


def _make_dataset(tmp_path, scenarios=("freefall", "ramp")):
    root = tmp_path / "data"
    assets = root / "assets"
    utonia = root / "3D_features"
    manifests = root / "v2" / "manifests"
    for d in (assets, utonia, manifests):
        d.mkdir(parents=True, exist_ok=True)

    conditions = []
    schedule = {
        "freefall": [
            {"condition_id": "pose0_angvel0", "parameters": {"orientation_wxyz": [1, 0, 0, 0], "angular_velocity_world_rad_s": [0, 0, 0]}},
            {"condition_id": "pose1_angvel1", "parameters": {"orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0, 0], "angular_velocity_world_rad_s": [0, 0, 2]}},
        ],
        "ramp": [
            {"condition_id": "angle20deg_pose0", "parameters": {"ramp_angle_deg": 20.0, "relative_orientation_wxyz": [1, 0, 0, 0]}},
            {"condition_id": "angle30deg_pose1", "parameters": {"ramp_angle_deg": 30.0, "relative_orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0, 0]}},
        ],
    }
    for scen in scenarios:
        for cond in schedule[scen]:
            conditions.append(
                {
                    "scenario": scen,
                    "object_id": "objA",
                    "condition_id": cond["condition_id"],
                    "parameters": cond["parameters"],
                }
            )

    schedule_doc = {"condition_count": len(conditions), "conditions": conditions}
    _write_json(manifests / "ordinary_schedule.json", schedule_doc)
    _write_json(
        manifests / "object_split.json",
        {"splits": {"train": ["objA"], "validation": ["objB"], "test": ["objC"]}},
    )

    (assets / "objA.json").write_text("{}")
    (utonia / "objA.npz").write_bytes(b"utonia-a")

    for cond in conditions:
        run_dir = root / "v2" / cond["scenario"] / cond["object_id"] / cond["condition_id"]
        _make_trajectory(run_dir, "objA", cond["condition_id"])

    return root, conditions


def _config(root, **overrides):
    kwargs = dict(
        data_root=str(root),
        schedule_path=str(root / "v2" / "manifests" / "ordinary_schedule.json"),
        split_path=str(root / "v2" / "manifests" / "object_split.json"),
        scenarios=("freefall", "ramp"),
        geometry_dir=str(root / "assets"),
        utonia_dir=str(root / "3D_features"),
    )
    kwargs.update(overrides)
    return ManifestConfig(**kwargs)


# --- condition decomposition ---


def test_decompose_freefall():
    out = decompose_condition(
        "freefall",
        {"orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], "angular_velocity_world_rad_s": [0.0, 0.0, 2.0]},
    )
    assert out["initial_condition_id"] == "pose1_angvel1"
    assert out["scenario_condition_id"] is None
    assert out["scenario_condition_parameters"] == {}
    assert set(out["initial_condition_parameters"]) == {"orientation_wxyz", "angular_velocity_world_rad_s"}


def test_decompose_ramp():
    out = decompose_condition(
        "ramp",
        {"ramp_angle_deg": 20.0, "relative_orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]},
    )
    assert out["scenario_condition_id"] == "angle20deg"
    assert out["initial_condition_id"] == "pose1"
    assert out["scenario_condition_parameters"] == {"ramp_angle_deg": 20.0}
    assert out["initial_condition_parameters"] == {"relative_orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]}


def test_decompose_unknown_level_falls_back_to_canonical():
    out = decompose_condition(
        "ramp",
        {"ramp_angle_deg": 20.0, "relative_orientation_wxyz": [0.9, 0.1, 0.2, 0.3]},
    )
    assert out["scenario_condition_id"] == "angle20deg"
    assert "relative_orientation_wxyz" in out["initial_condition_id"]


# --- hashing ---


def test_sha256_deterministic(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"content-123")
    assert sha256_file(str(p)) == sha256_file(str(p))
    assert sha256_bytes(b"content-123") == sha256_file(str(p))


# --- eligibility ---


def test_inspect_valid(tmp_path):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0")
    result = inspect_trajectory(str(run_dir))
    assert result.included is True
    assert result.reason is None
    assert result.flags["frame_count"] == 153
    assert result.flags["status"] == "completed"


def test_inspect_missing_dir(tmp_path):
    result = inspect_trajectory(str(tmp_path / "nope"))
    assert result.included is False
    assert result.reason == "missing_run_directory"


def test_inspect_status_not_completed(tmp_path):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0", status="failed")
    result = inspect_trajectory(str(run_dir))
    assert result.reason == "metadata_status_not_completed"


def test_inspect_validation_not_valid(tmp_path):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0", valid=False)
    result = inspect_trajectory(str(run_dir))
    assert result.reason == "validation_not_valid"


@pytest.mark.parametrize("omit,reason", [
    ("state", "missing_artifact:state"),
    ("contacts", "missing_artifact:contacts"),
    ("modality_timestamps", "missing_artifact:modality_timestamps"),
    ("events", "missing_artifact:events"),
    ("rgb_a", "missing_artifact:rgb_a"),
    ("depth_a", "missing_artifact:depth_a"),
    ("metadata", "missing_artifact:metadata"),
    ("validation", "missing_artifact:validation"),
])
def test_inspect_missing_artifact(tmp_path, omit, reason):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0", omit={omit})
    result = inspect_trajectory(str(run_dir))
    assert result.reason == reason


def test_inspect_insufficient_duration(tmp_path):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0", frame_count=80)
    result = inspect_trajectory(str(run_dir))
    assert result.reason == "insufficient_duration"


def test_inspect_frame_count_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0", frame_count=153)
    # validation reports a different count than the state arrays
    _write_json(
        os.path.join(str(run_dir), "validation.json"),
        {"valid": True, "frame_count": 120},
    )
    result = inspect_trajectory(str(run_dir))
    assert result.reason == "frame_count_mismatch"


def test_inspect_records_warnings(tmp_path):
    run_dir = tmp_path / "run"
    _make_trajectory(str(run_dir), "objA", "pose0_angvel0", warnings=["w1"])
    result = inspect_trajectory(str(run_dir))
    assert "w1" in result.warnings


# --- manifest build ---


def test_build_manifest_counts(tmp_path):
    root, _ = _make_dataset(tmp_path)
    manifest = build_manifest(_config(root))
    s = manifest["summary"]
    assert s["total_scheduled"] == 4
    assert s["included"] == 4
    assert s["excluded"] == 0
    assert s["by_scenario"]["freefall"]["included"] == 2
    assert s["by_scenario"]["ramp"]["included"] == 2
    assert manifest["dataset_version"] == "2.0"


def test_build_manifest_deterministic(tmp_path):
    root, _ = _make_dataset(tmp_path)
    m1 = build_manifest(_config(root))
    m2 = build_manifest(_config(root))
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_json(m1, str(a))
    write_json(m2, str(b))
    assert a.read_bytes() == b.read_bytes()


def test_build_manifest_sorted(tmp_path):
    root, _ = _make_dataset(tmp_path)
    manifest = build_manifest(_config(root))
    ids = [t["trajectory_id"] for t in manifest["trajectories"]]
    assert ids == sorted(ids)
    excl = [e["trajectory_id"] for e in manifest["excluded"]]
    assert excl == sorted(excl)


def test_build_manifest_entry_fields(tmp_path):
    root, _ = _make_dataset(tmp_path)
    manifest = build_manifest(_config(root))
    t = manifest["trajectories"][0]
    assert t["object_id"] == "objA"
    assert t["split"] == "train"
    assert t["condition_id"] in {"pose0_angvel0", "pose1_angvel1", "angle20deg_pose0", "angle30deg_pose1"}
    assert "content_hashes" in t
    assert set(t["content_hashes"]) == {"state", "metadata", "validation", "contacts", "modality_timestamps", "events"}
    assert all(len(h) == 64 for h in t["content_hashes"].values())


def test_build_manifest_excludes_missing_geometry(tmp_path):
    root, _ = _make_dataset(tmp_path)
    (root / "assets" / "objA.json").unlink()
    manifest = build_manifest(_config(root))
    assert manifest["summary"]["included"] == 0
    reasons = {e["reason"] for e in manifest["excluded"]}
    assert reasons == {"missing_geometry"}


def test_build_manifest_excludes_missing_utonia(tmp_path):
    root, _ = _make_dataset(tmp_path)
    (root / "3D_features" / "objA.npz").unlink()
    manifest = build_manifest(_config(root))
    reasons = {e["reason"] for e in manifest["excluded"]}
    assert reasons == {"missing_utonia"}


def test_build_manifest_excludes_missing_run(tmp_path):
    root, _ = _make_dataset(tmp_path)
    import shutil

    shutil.rmtree(root / "v2" / "freefall" / "objA" / "pose0_angvel0")
    manifest = build_manifest(_config(root))
    reasons = {e["reason"] for e in manifest["excluded"]}
    assert "missing_run_directory" in reasons
    assert manifest["summary"]["included"] == 3


def test_build_manifest_records_schedule_and_split_hash(tmp_path):
    root, _ = _make_dataset(tmp_path)
    manifest = build_manifest(_config(root))
    assert len(manifest["schedule_hash"]) == 64
    assert len(manifest["split_hash"]) == 64


# --- audit ---


def test_render_audit_contains_sections(tmp_path):
    root, _ = _make_dataset(tmp_path)
    manifest = build_manifest(_config(root))
    text = render_audit(manifest)
    for section in ("Summary", "Object Split", "Objects", "Conditions", "Termination Reasons", "Warnings", "Excluded Runs"):
        assert section in text
    assert "pose0_angvel0" in text


def test_trajectory_id_format():
    assert trajectory_id("freefall", "2R1-1", "pose0_angvel0") == "freefall/2R1-1/pose0_angvel0"
