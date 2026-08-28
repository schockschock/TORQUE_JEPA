"""Integration test against the real TORQUE pilot data.

Skipped when the dataset root is absent (e.g. CI, other machines). Builds a
manifest over a small real subset to verify the gates and determinism against
genuine simulator artifacts.
"""

import os

import pytest

from data.trajectory_manifest import (
    ManifestConfig,
    build_manifest,
    inspect_trajectory,
    render_audit,
)

DATA_ROOT = os.environ.get("TORQUE_DATA_ROOT", "/data2/adrien/TORQUE")
TRAJ_DIR = os.path.join(DATA_ROOT, "v2", "freefall", "3R1-8", "pose1_angvel1")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TRAJ_DIR), reason="TORQUE pilot data not available"
)


def _config(tmp_path, scenarios=("freefall",), object_ids=("3R1-8",)):
    schedule_path = tmp_path / "schedule.json"
    split_path = tmp_path / "split.json"

    import json

    src = json.load(open(os.path.join(DATA_ROOT, "v2", "manifests", "ordinary_schedule.json")))
    conds = [
        c for c in src["conditions"]
        if c["scenario"] in scenarios and c["object_id"] in object_ids
    ]
    with open(schedule_path, "w") as f:
        json.dump({"condition_count": len(conds), "conditions": conds}, f)

    split = {"splits": {"train": [], "validation": [], "test": []}}
    for oid in object_ids:
        split["splits"]["train"].append(oid)
    with open(split_path, "w") as f:
        json.dump(split, f)

    return ManifestConfig(
        data_root=DATA_ROOT,
        schedule_path=str(schedule_path),
        split_path=str(split_path),
        scenarios=scenarios,
        geometry_dir=os.path.join(DATA_ROOT, "assets"),
        utonia_dir=os.path.join(DATA_ROOT, "3D_features"),
    )


def test_real_trajectory_is_eligible():
    result = inspect_trajectory(TRAJ_DIR)
    assert result.included is True
    assert result.flags["frame_count"] >= 91
    assert result.flags["status"] == "completed"


def test_real_manifest_builds_and_is_deterministic(tmp_path):
    cfg = _config(tmp_path, scenarios=("freefall",), object_ids=("3R1-8",))
    m1 = build_manifest(cfg)
    m2 = build_manifest(cfg)

    assert m1 == m2
    assert m1["summary"]["included"] > 0
    assert m1["dataset_version"] == "2.0"

    entry = m1["trajectories"][0]
    assert entry["object_id"] == "3R1-8"
    assert entry["scenario"] == "freefall"
    assert entry["initial_condition_id"] == entry["condition_id"]
    assert entry["scenario_condition_id"] is None
    assert entry["split"] == "train"
    assert len(entry["content_hashes"]["state"]) == 64

    text = render_audit(m1)
    assert "Summary" in text
