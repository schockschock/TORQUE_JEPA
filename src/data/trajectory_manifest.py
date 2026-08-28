"""Frozen, versioned TORQUE trajectory manifest.

``build_manifest`` enumerates scheduled trajectories, applies the eligibility
gates (completion, validation, synchronized artifacts, duration), decomposes
each trajectory condition into initial and scenario conditions, records object
splits, per-artifact SHA-256 hashes, and explicit exclusion reasons. The result
is an immutable snapshot: regenerating against unchanged inputs yields
byte-identical contents and summary counts.

The manifest is trajectory-level only. Window indexing is a later stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

MANIFEST_FORMAT_VERSION = 1
KIND_TRAJECTORY = "trajectory"

# A full observation window (30 native frames) plus the observation end plus a
# full forecast horizon (60 native frames) requires at least 91 native frames.
DEFAULT_MIN_FRAMES = 30 + 1 + 60

# Scenario-condition keys partition each scenario's ``condition_parameters``
# into a scenario-specific configuration versus the initial kinematic state.
SCENARIO_CONDITION_KEYS: dict[str, tuple[str, ...]] = {
    "freefall": (),
    "ramp": ("ramp_angle_deg",),
}

# Per-trajectory artifacts that must exist and be synchronized.
REQUIRED_FILES = ("state", "contacts", "modality_timestamps", "events")
REQUIRED_DIRS = ("rgb_a", "depth_a")

_FILE_SUFFIX = {
    "state": "state.npz",
    "metadata": "metadata.json",
    "validation": "validation.json",
    "contacts": "contacts.npz",
    "modality_timestamps": "modality_timestamps.npz",
    "events": "events.json",
}

# Discrete initial-condition levels used to derive stable, human-readable
# decomposed condition ids from parameter values. Values below the tolerance
# fall back to a canonical JSON label so unknown levels stay identifiable.
_ORIENTATION_LEVELS = {
    "pose0": (1.0, 0.0, 0.0, 0.0),
    "pose1": (0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
    "pose2": (0.7071067811865476, 0.0, 0.7071067811865476, 0.0),
}
_ANGVEL_LEVELS = {
    "angvel0": (0.0, 0.0, 0.0),
    "angvel1": (0.0, 0.0, 2.0),
    "angvel2": (2.0, 1.5, 0.5),
}
_LEVEL_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ManifestConfig:
    data_root: str
    schedule_path: str
    split_path: str
    scenarios: tuple[str, ...]
    geometry_dir: str
    utonia_dir: str
    dataset_version: str | None = None
    min_frames: int = DEFAULT_MIN_FRAMES
    scope: str = "pilot"
    generated_at: str | None = None


@dataclass
class EligibilityResult:
    included: bool
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_label(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _nearest_level(value: Any, levels: dict[str, tuple[float, ...]], sign_invariant: bool) -> str | None:
    v = [float(x) for x in value]
    best_label: str | None = None
    best_dist: float | None = None
    for label, ref in levels.items():
        ref = [float(x) for x in ref]
        dist = math.dist(v, ref)
        if sign_invariant:
            dist = min(dist, math.dist([-x for x in v], ref))
        if best_dist is None or dist < best_dist:
            best_label, best_dist = label, dist
    if best_label is not None and best_dist is not None and best_dist <= _LEVEL_TOLERANCE:
        return best_label
    return None


def _orientation_label(value: Any) -> str | None:
    return _nearest_level(value, _ORIENTATION_LEVELS, sign_invariant=True)


def _angvel_label(value: Any) -> str | None:
    return _nearest_level(value, _ANGVEL_LEVELS, sign_invariant=False)


def _scenario_condition_label(scenario: str, params: dict[str, Any]) -> str:
    if scenario == "ramp" and "ramp_angle_deg" in params:
        return f"angle{params['ramp_angle_deg']:g}deg"
    return _canonical_label(params)


def _initial_condition_label(scenario: str, params: dict[str, Any]) -> str:
    if scenario == "freefall":
        if "orientation_wxyz" in params and "angular_velocity_world_rad_s" in params:
            pose = _orientation_label(params["orientation_wxyz"])
            angvel = _angvel_label(params["angular_velocity_world_rad_s"])
            if pose and angvel:
                return f"{pose}_{angvel}"
    elif scenario == "ramp":
        if "relative_orientation_wxyz" in params:
            pose = _orientation_label(params["relative_orientation_wxyz"])
            if pose:
                return pose
    return _canonical_label(params)


def decompose_condition(
    scenario: str, condition_parameters: dict[str, Any]
) -> dict[str, Any]:
    """Split a trajectory condition into initial and scenario conditions."""
    scenario_keys = SCENARIO_CONDITION_KEYS.get(scenario, ())
    scenario_params = {k: condition_parameters[k] for k in scenario_keys if k in condition_parameters}
    initial_params = {k: v for k, v in condition_parameters.items() if k not in scenario_keys}

    return {
        "initial_condition_id": _initial_condition_label(scenario, initial_params) if initial_params else None,
        "initial_condition_parameters": initial_params,
        "scenario_condition_id": _scenario_condition_label(scenario, scenario_params) if scenario_params else None,
        "scenario_condition_parameters": scenario_params,
    }


def trajectory_id(scenario: str, object_id: str, condition_id: str) -> str:
    return f"{scenario}/{object_id}/{condition_id}"


def _excluded_entry(
    scenario: str, object_id: str, condition_id: str, reason: str | None
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id(scenario, object_id, condition_id),
        "object_id": object_id,
        "scenario": scenario,
        "condition_id": condition_id,
        "reason": reason,
    }


def resolve_paths(
    data_root: str,
    scenario: str,
    object_id: str,
    condition_id: str,
    geometry_dir: str,
    utonia_dir: str,
) -> dict[str, str]:
    run_dir = os.path.join(data_root, "v2", scenario, object_id, condition_id)
    paths = {"run_dir": run_dir}
    for name, suffix in _FILE_SUFFIX.items():
        paths[f"{name}_path"] = os.path.join(run_dir, suffix)
    paths["rgb_a_path"] = os.path.join(run_dir, "rgb_a")
    paths["depth_a_path"] = os.path.join(run_dir, "depth_a")
    paths["geometry_path"] = os.path.join(geometry_dir, f"{object_id}.json")
    paths["utonia_path"] = os.path.join(utonia_dir, f"{object_id}.npz")
    return paths


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _dir_file_count(path: str) -> int:
    try:
        return sum(1 for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)))
    except OSError:
        return 0


def inspect_trajectory(run_dir: str, min_frames: int = DEFAULT_MIN_FRAMES) -> EligibilityResult:
    """Apply the per-trajectory eligibility gates to a run directory."""
    warnings: list[str] = []
    flags: dict[str, Any] = {}

    if not os.path.isdir(run_dir):
        return EligibilityResult(False, "missing_run_directory", warnings, flags)

    metadata = _read_json(os.path.join(run_dir, "metadata.json"))
    if metadata is None:
        return EligibilityResult(False, "missing_artifact:metadata", warnings, flags)
    status = metadata.get("status")
    flags["status"] = status
    flags["termination_reason"] = metadata.get("termination_reason")
    warnings.extend(metadata.get("validation_warnings", []) or [])

    if status != "completed":
        return EligibilityResult(False, "metadata_status_not_completed", warnings, flags)

    validation = _read_json(os.path.join(run_dir, "validation.json"))
    if validation is None:
        return EligibilityResult(False, "missing_artifact:validation", warnings, flags)
    warnings.extend(validation.get("warnings", []) or [])
    flags["valid"] = validation.get("valid")
    if validation.get("valid") is not True:
        return EligibilityResult(False, "validation_not_valid", warnings, flags)

    for name in REQUIRED_FILES:
        path = os.path.join(run_dir, _FILE_SUFFIX[name])
        if not os.path.isfile(path):
            return EligibilityResult(False, f"missing_artifact:{name}", warnings, flags)

    for name in REQUIRED_DIRS:
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            return EligibilityResult(False, f"missing_artifact:{name}", warnings, flags)
        flags[f"{name}_frame_count"] = _dir_file_count(path)

    try:
        state_frame_count = int(len(np.load(os.path.join(run_dir, "state.npz"))["frame_index"]))
        mt_frame_count = int(
            len(np.load(os.path.join(run_dir, "modality_timestamps.npz"))["frame_index"])
        )
    except (OSError, ValueError, KeyError):
        return EligibilityResult(False, "unreadable_state_arrays", warnings, flags)

    validation_frame_count = validation.get("frame_count")
    if validation_frame_count is None:
        validation_frame_count = metadata.get("frames_captured")
    if validation_frame_count is None:
        return EligibilityResult(False, "frame_count_missing", warnings, flags)

    flags["frame_count"] = state_frame_count
    if not (
        state_frame_count == mt_frame_count
        and (validation_frame_count is None or state_frame_count == validation_frame_count)
    ):
        return EligibilityResult(False, "frame_count_mismatch", warnings, flags)

    if state_frame_count < min_frames:
        return EligibilityResult(False, "insufficient_duration", warnings, flags)

    for name in REQUIRED_DIRS:
        expected = state_frame_count
        if flags.get(f"{name}_frame_count", expected) != expected:
            warnings.append(f"{name}_frame_count_mismatch")

    return EligibilityResult(True, None, warnings, flags)


def _content_hashes(paths: dict[str, str], include: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in include:
        path = paths.get(f"{name}_path")
        if path and os.path.isfile(path):
            hashes[name] = sha256_file(path)
    return hashes


def _load_split_map(split_path: str) -> dict[str, str]:
    split = _read_json(split_path) or {}
    out: dict[str, str] = {}
    for partition, object_ids in (split.get("splits") or {}).items():
        for oid in object_ids:
            out[oid] = partition
    return out


def build_manifest(config: ManifestConfig) -> dict[str, Any]:
    """Build a deterministic, immutable trajectory manifest snapshot."""
    schedule = _read_json(config.schedule_path) or {}
    split_doc = _read_json(config.split_path) or {}
    split_map = _load_split_map(config.split_path)
    split_hash = sha256_file(config.split_path)
    schedule_hash = sha256_file(config.schedule_path)

    scheduled = [
        c
        for c in schedule.get("conditions", [])
        if c.get("scenario") in config.scenarios
    ]
    scheduled.sort(key=lambda c: trajectory_id(c["scenario"], c["object_id"], c["condition_id"]))

    objects: dict[str, dict[str, Any]] = {}
    for oid in sorted({c["object_id"] for c in scheduled}):
        geometry_path = os.path.join(config.geometry_dir, f"{oid}.json")
        utonia_path = os.path.join(config.utonia_dir, f"{oid}.npz")
        objects[oid] = {
            "split": split_map.get(oid),
            "geometry_path": geometry_path,
            "utonia_path": utonia_path,
            "geometry_hash": sha256_file(geometry_path) if os.path.isfile(geometry_path) else None,
            "utonia_hash": sha256_file(utonia_path) if os.path.isfile(utonia_path) else None,
            "scheduled_trajectories": 0,
            "included_trajectories": 0,
        }

    trajectories: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    dataset_versions: set[str] = set()

    for cond in scheduled:
        oid = cond["object_id"]
        scen = cond["scenario"]
        cid = cond["condition_id"]
        tid = trajectory_id(scen, oid, cid)
        paths = resolve_paths(
            config.data_root, scen, oid, cid, config.geometry_dir, config.utonia_dir
        )
        objects[oid]["scheduled_trajectories"] += 1

        geometry_ok = os.path.isfile(paths["geometry_path"])
        utonia_ok = os.path.isfile(paths["utonia_path"])
        if not geometry_ok:
            excluded.append(_excluded_entry(scen, oid, cid, "missing_geometry"))
            continue
        if not utonia_ok:
            excluded.append(_excluded_entry(scen, oid, cid, "missing_utonia"))
            continue

        result = inspect_trajectory(paths["run_dir"], config.min_frames)
        if not result.included:
            excluded.append(_excluded_entry(scen, oid, cid, result.reason))
            continue

        decomp = decompose_condition(scen, cond.get("parameters", {}))
        metadata = _read_json(paths["metadata_path"]) or {}
        dataset_versions.add(metadata.get("dataset_version") or "unknown")

        entry: dict[str, Any] = {
            "trajectory_id": tid,
            "object_id": oid,
            "scenario": scen,
            "condition_id": cid,
            "condition_parameters": cond.get("parameters", {}),
            "initial_condition_id": decomp["initial_condition_id"],
            "initial_condition_parameters": decomp["initial_condition_parameters"],
            "scenario_condition_id": decomp["scenario_condition_id"],
            "scenario_condition_parameters": decomp["scenario_condition_parameters"],
            "split": split_map.get(oid),
            "geometry_path": paths["geometry_path"],
            "utonia_path": paths["utonia_path"],
            "state_path": paths["state_path"],
            "rgb_a_path": paths["rgb_a_path"],
            "depth_a_path": paths["depth_a_path"],
            "metadata_path": paths["metadata_path"],
            "validation_path": paths["validation_path"],
            "contact_path": paths["contacts_path"],
            "modality_timestamps_path": paths["modality_timestamps_path"],
            "events_path": paths["events_path"],
            "frame_count": result.flags.get("frame_count"),
            "termination_reason": result.flags.get("termination_reason"),
            "validity_flags": {k: v for k, v in result.flags.items() if k != "termination_reason"},
            "warnings": sorted(set(result.warnings)),
            "content_hashes": _content_hashes(
                paths, ("state", "metadata", "validation", "contacts", "modality_timestamps", "events")
            ),
        }
        objects[oid]["included_trajectories"] += 1
        trajectories.append(entry)

    trajectories.sort(key=lambda t: t["trajectory_id"])
    excluded.sort(key=lambda e: e["trajectory_id"])

    summary = _summarize(trajectories, excluded, scheduled, split_map)
    if config.dataset_version is not None:
        dataset_version: str | None = config.dataset_version
        dataset_versions: list[str] | None = None
    elif len(dataset_versions) == 1:
        dataset_version = dataset_versions.pop()
    else:
        dataset_version = None

    result: dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "kind": KIND_TRAJECTORY,
        "scope": config.scope,
        "dataset_version": dataset_version,
        "data_root": config.data_root,
        "schedule_path": config.schedule_path,
        "schedule_hash": schedule_hash,
        "split_path": config.split_path,
        "split_hash": split_hash,
        "split_seed": split_doc.get("split_seed"),
        "split_unit": split_doc.get("split_unit"),
        "object_population_version": split_doc.get("object_population_version"),
        "policy": {
            "scenarios": list(config.scenarios),
            "min_frames": config.min_frames,
            "required_files": list(REQUIRED_FILES),
            "required_dirs": list(REQUIRED_DIRS),
        },
        "summary": summary,
        "objects": objects,
        "trajectories": trajectories,
        "excluded": excluded,
    }
    if config.dataset_version is None and dataset_version is None and dataset_versions:
        result["dataset_versions"] = sorted(dataset_versions)
    if config.generated_at is not None:
        result["generated_at"] = config.generated_at
    return result


def _summarize(
    trajectories: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    scheduled: list[dict[str, Any]],
    split_map: dict[str, str],
) -> dict[str, Any]:
    by_scenario: dict[str, dict[str, int]] = {}
    exclusion_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    trajectories_with_warnings = 0
    by_condition: dict[str, dict[str, dict[str, int]]] = {}
    by_split: dict[str, dict[str, int]] = {}

    for c in scheduled:
        scen = c["scenario"]
        s = by_scenario.setdefault(scen, {"scheduled": 0, "included": 0, "excluded": 0})
        s["scheduled"] += 1
        cond = by_condition.setdefault(scen, {}).setdefault(c["condition_id"], {"scheduled": 0, "included": 0})
        cond["scheduled"] += 1

    for e in excluded:
        by_scenario[e["scenario"]]["excluded"] += 1
        exclusion_counts[e["reason"]] = exclusion_counts.get(e["reason"], 0) + 1

    for t in trajectories:
        scen = t["scenario"]
        by_scenario[scen]["included"] += 1
        by_condition[scen][t["condition_id"]]["included"] += 1
        split = t["split"]
        sp = by_split.setdefault(split, {"scheduled": 0, "included": 0})
        sp["included"] += 1
        reason = t["termination_reason"]
        if reason is not None:
            termination_counts[str(reason)] = termination_counts.get(str(reason), 0) + 1
        if t.get("warnings"):
            trajectories_with_warnings += 1
        for w in t.get("warnings", []):
            warning_counts[str(w)] = warning_counts.get(str(w), 0) + 1

    for c in scheduled:
        scen = c["scenario"]
        oid = c["object_id"]
        split = split_map.get(oid)
        if split is not None:
            by_split.setdefault(split, {"scheduled": 0, "included": 0})["scheduled"] += 1

    return {
        "total_scheduled": len(scheduled),
        "included": len(trajectories),
        "excluded": len(excluded),
        "by_scenario": by_scenario,
        "by_split": by_split,
        "exclusion_reason_counts": exclusion_counts,
        "termination_reason_counts": termination_counts,
        "warning_counts": warning_counts,
        "trajectories_with_warnings": trajectories_with_warnings,
        "by_condition": by_condition,
    }


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write("\n")


def render_audit(manifest: dict[str, Any]) -> str:
    """Render a human-readable audit report from a manifest."""
    summary = manifest["summary"]
    objects = manifest["objects"]
    lines: list[str] = []
    lines.append("# TORQUE Pilot Trajectory Snapshot Audit")
    lines.append("")
    lines.append(f"- scope: `{manifest['scope']}`")
    lines.append(f"- dataset_version: `{manifest['dataset_version']}`")
    lines.append(f"- format_version: `{manifest['format_version']}`")
    lines.append(f"- schedule_hash: `{manifest['schedule_hash']}`")
    lines.append(f"- split_hash: `{manifest['split_hash']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- scheduled: {summary['total_scheduled']} "
        f"(included {summary['included']}, excluded {summary['excluded']})"
    )
    lines.append("")
    lines.append("| scenario | scheduled | included | excluded |")
    lines.append("| --- | ---: | ---: | ---: |")
    for scen, counts in sorted(summary["by_scenario"].items()):
        lines.append(
            f"| {scen} | {counts['scheduled']} | {counts['included']} | {counts['excluded']} |"
        )
    lines.append("")
    lines.append("## Object Split")
    lines.append("")
    lines.append("| split | scheduled | included |")
    lines.append("| --- | ---: | ---: |")
    for split, counts in sorted(summary["by_split"].items()):
        lines.append(f"| {split} | {counts['scheduled']} | {counts['included']} |")
    lines.append("")
    lines.append("## Objects")
    lines.append("")
    included_objects = {oid: o for oid, o in objects.items() if o["included_trajectories"] > 0}
    missing_objects = sorted(oid for oid, o in objects.items() if o["included_trajectories"] == 0)
    lines.append(f"- objects with included trajectories: {len(included_objects)}")
    lines.append(f"- objects without pilot trajectories: {len(missing_objects)}")
    lines.append("")
    lines.append("| split | objects | included trajectories |")
    lines.append("| --- | ---: | ---: |")
    by_split_objects: dict[str, list[str]] = {}
    by_split_included: dict[str, int] = {}
    for oid, o in sorted(included_objects.items()):
        by_split_objects.setdefault(o["split"], []).append(oid)
        by_split_included[o["split"]] = by_split_included.get(o["split"], 0) + o["included_trajectories"]
    for split in sorted(by_split_objects):
        lines.append(
            f"| {split} | {len(by_split_objects[split])} | {by_split_included[split]} |"
        )
    if missing_objects:
        lines.append("")
        lines.append(f"Objects without pilot trajectories: {', '.join(missing_objects)}")
    lines.append("")
    lines.append("## Conditions")
    lines.append("")
    for scen in sorted(summary["by_condition"]):
        lines.append(f"### {scen}")
        lines.append("")
        lines.append("| condition_id | scheduled | included |")
        lines.append("| --- | ---: | ---: |")
        for cid in sorted(summary["by_condition"][scen]):
            counts = summary["by_condition"][scen][cid]
            lines.append(f"| {cid} | {counts['scheduled']} | {counts['included']} |")
        lines.append("")
    lines.append("## Termination Reasons")
    lines.append("")
    if summary["termination_reason_counts"]:
        lines.append("| termination_reason | count |")
        lines.append("| --- | ---: |")
        for reason, count in sorted(summary["termination_reason_counts"].items()):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("_(none recorded)_")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if summary["warning_counts"]:
        lines.append(f"- trajectories with warnings: {summary['trajectories_with_warnings']}")
        lines.append("")
        lines.append("| warning | count |")
        lines.append("| --- | ---: |")
        for warning, count in sorted(summary["warning_counts"].items()):
            lines.append(f"| {warning} | {count} |")
    else:
        lines.append("_(none recorded)_")
    lines.append("")
    lines.append("## Excluded Runs")
    lines.append("")
    if manifest["excluded"]:
        lines.append("| reason | count |")
        lines.append("| --- | ---: |")
        for reason, count in sorted(summary["exclusion_reason_counts"].items()):
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        for reason in sorted(summary["exclusion_reason_counts"]):
            entries = [e for e in manifest["excluded"] if e["reason"] == reason]
            if reason == "missing_run_directory":
                objects_by_scenario: dict[str, set[str]] = {}
                for e in entries:
                    objects_by_scenario.setdefault(e["scenario"], set()).add(e["object_id"])
                lines.append(f"### {reason} (objects not yet simulated)")
                lines.append("")
                for scen in sorted(objects_by_scenario):
                    lines.append(f"- {scen}: {', '.join(sorted(objects_by_scenario[scen]))}")
                lines.append("")
            else:
                lines.append(f"### {reason}")
                lines.append("")
                lines.append("| trajectory_id |")
                lines.append("| --- |")
                for e in entries:
                    lines.append(f"| {e['trajectory_id']} |")
                lines.append("")
    else:
        lines.append("_(none)_")
        lines.append("")
    return "\n".join(lines)
