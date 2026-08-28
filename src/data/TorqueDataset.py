"""Configurable TORQUE Dataset and trajectory-aware DataLoader utilities.

The dataset discovers a deterministic :class:`Dataset Cohort` at
initialization.  It then indexes fixed 120 Hz trajectories at 60 Hz and loads
only the modalities requested by the caller.  The public sample shape follows
``docs/data_requirements.md``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # Keep discovery helpers importable in environments without PyTorch.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - the training package requires torch.
    torch = None  # type: ignore[assignment]

    class TorchDataset:  # type: ignore[no-redef]
        pass


CAPTURE_FREQUENCY_HZ = 120
MODEL_FREQUENCY_HZ = 60
MODEL_STRIDE = CAPTURE_FREQUENCY_HZ // MODEL_FREQUENCY_HZ
OBSERVATION_SECONDS = 0.25
FORECAST_SECONDS = 0.50
OBSERVATION_SAMPLES = 16
PREDICTION_SAMPLES = 30
OBSERVATION_NATIVE_FRAMES = 30
FORECAST_NATIVE_FRAMES = 60
MINIMUM_FRAMES = OBSERVATION_NATIVE_FRAMES + 1 + FORECAST_NATIVE_FRAMES
DEFAULT_SCENARIOS = ("freefall", "object_throw", "ramp", "conveyor")
MODALITIES = frozenset({"state", "rgb_a", "depth_a", "utonia"})
_CONTACT_PHASE_ORDER = (
    "no_contact",
    "contact_onset",
    "sustained_contact",
    "post_contact",
    "unknown",
)
_FRAME_RE = re.compile(r"^frame_(\d+)(?:\.jpg|\.jpeg|\.png|\.npz|\.npy)$")


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for :class:`TorqueDataset`.

    ``state`` is always included because it supplies both the forecasting
    target and the observation-end reconstruction anchor.  The other three
    modalities are loaded only when named in ``modalities``.
    """

    data_root: str
    split: str | None = "train"
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    modalities: tuple[str, ...] = ("state",)
    schedule_path: str | None = None
    split_path: str | None = None
    geometry_dir: str | None = None
    utonia_dir: str | None = None
    mode: str = "train"
    min_frames: int = MINIMUM_FRAMES
    invalid_window_policy: str = "drop"
    sync_tolerance_s: float = 1e-7
    image_size: tuple[int, int] = (224, 224)
    evaluation_stride_samples: int = PREDICTION_SAMPLES
    normalize_depth: bool = True
    normalize_utonia: bool = True
    normalization_pixels_per_object: int = 100_000

    def __post_init__(self) -> None:
        scenarios = (self.scenarios,) if isinstance(self.scenarios, str) else tuple(self.scenarios)
        modalities = (self.modalities,) if isinstance(self.modalities, str) else tuple(self.modalities)
        object.__setattr__(self, "scenarios", scenarios)
        normalized = tuple(_normalize_modality(value) for value in modalities)
        unknown = sorted(set(normalized) - MODALITIES)
        if unknown:
            raise ValueError(f"unknown TORQUE modality: {', '.join(unknown)}")
        object.__setattr__(self, "modalities", tuple(dict.fromkeys((*normalized, "state"))))
        if self.mode not in {"train", "evaluation", "eval"}:
            raise ValueError("mode must be 'train' or 'evaluation'")
        if self.invalid_window_policy not in {"drop", "skip", "keep", "error"}:
            raise ValueError("invalid_window_policy must be drop, keep, or error")
        if self.evaluation_stride_samples < 1:
            raise ValueError("evaluation_stride_samples must be positive")
        if self.min_frames < MINIMUM_FRAMES:
            object.__setattr__(self, "min_frames", MINIMUM_FRAMES)
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("image_size must contain two positive dimensions")

    @property
    def required_modalities(self) -> frozenset[str]:
        return frozenset(self.modalities)

    @property
    def paths(self) -> dict[str, str]:
        root = os.path.abspath(self.data_root)
        return {
            "schedule": os.path.abspath(
                self.schedule_path
                or os.path.join(root, "v2", "manifests", "ordinary_schedule.json")
            ),
            "split": os.path.abspath(
                self.split_path
                or os.path.join(root, "v2", "manifests", "object_split.json")
            ),
            "geometry": os.path.abspath(self.geometry_dir or os.path.join(root, "assets")),
            "utonia": os.path.abspath(
                self.utonia_dir or os.path.join(root, "3D_features")
            ),
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    """One immutable entry in the Dataset Cohort."""

    trajectory_id: str
    scenario: str
    object_id: str
    condition_id: str
    split: str
    run_dir: str
    state_path: str
    modality_timestamps_path: str
    metadata_path: str
    validation_path: str
    contacts_path: str | None
    events_path: str | None
    rgb_a_path: str | None
    depth_a_path: str | None
    geometry_path: str
    utonia_path: str | None
    frame_count: int
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    frustum_visible_a: tuple[bool, ...] | None
    contact_active: tuple[bool, ...] | None
    condition_parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowIndex:
    """Deterministic native-frame bounds for one forecast sample."""

    trajectory_index: int
    trajectory_id: str
    observation_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    endpoint_index: int
    native_frame_start: int
    native_frame_end: int
    native_target_end: int
    observation_timestamps: tuple[float, ...]
    target_timestamps: tuple[float, ...]
    contact_phase: str
    validity_flags: Mapping[str, bool]

    @property
    def end_index(self) -> int:
        """Compatibility alias for callers that call the endpoint ``end``."""

        return self.endpoint_index

    @property
    def timestamps(self) -> tuple[float, ...]:
        """All sampled timestamps, observation first and forecast second."""

        return self.observation_timestamps + self.target_timestamps


@dataclass(frozen=True)
class NormalizationStats:
    """Statistics fitted from training objects only."""

    depth_mean: float = 0.0
    depth_std: float = 1.0
    utonia_mean: np.ndarray | None = None
    utonia_std: np.ndarray | None = None
    utonia_max_mean: np.ndarray | None = None
    utonia_max_std: np.ndarray | None = None


def _normalize_modality(value: str) -> str:
    aliases = {
        "kinematic_state": "state",
        "rgb": "rgb_a",
        "camera_a_rgb": "rgb_a",
        "depth": "depth_a",
        "camera_a_depth": "depth_a",
        "depth_valid_a": "depth_a",
        "utonia_features": "utonia",
        "geometry": "utonia",
    }
    return aliases.get(str(value).lower(), str(value).lower())


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _load_npz(path: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"could not read NPZ artifact {path}: {exc}") from exc


def _load_npz_keys(path: str, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Load selected NPZ arrays without materializing unrelated large arrays."""

    wanted = tuple(keys)
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in wanted if key in loaded.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"could not read NPZ artifact {path}: {exc}") from exc


def _first_array(data: Mapping[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def _trajectory_id(scenario: str, object_id: str, condition_id: str) -> str:
    return f"{scenario}/{object_id}/{condition_id}"


def _split_map(split_doc: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split, object_ids in (split_doc.get("splits") or {}).items():
        if not isinstance(object_ids, list):
            raise ValueError(f"split {split!r} must contain a list of object IDs")
        for object_id in object_ids:
            object_id = str(object_id)
            previous = result.get(object_id)
            if previous is not None and previous != split:
                raise ValueError(f"object {object_id!r} appears in multiple splits")
            result[object_id] = str(split)
    return result


def _frame_files(directory: str, suffixes: tuple[str, ...]) -> set[int]:
    try:
        names = os.listdir(directory)
    except OSError:
        return set()
    accepted = set(suffixes)
    frames: set[int] = set()
    for name in names:
        match = _FRAME_RE.match(name)
        if match and Path(name).suffix.lower() in accepted:
            frames.add(int(match.group(1)))
    return frames


def _frame_path(directory: str, frame_index: int, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        path = os.path.join(directory, f"frame_{frame_index:06d}{suffix}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"no frame {frame_index} in {directory} with suffix {suffixes}"
    )


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float64).eps):
        raise ValueError("orientation quaternion contains a zero-norm value")
    q = q / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), axis=-1),
            np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), axis=-1),
            np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), axis=-1),
        ),
        axis=-2,
    )


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Return the first two rotation columns in the project’s 6D order."""

    matrix = np.asarray(matrix)
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Project a first-two-columns representation back to an SO(3) matrix."""

    values = np.asarray(values, dtype=np.float64)
    first, second = values[..., :3], values[..., 3:6]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def _state_rotations(state: Mapping[str, np.ndarray]) -> np.ndarray:
    quaternion = _first_array(
        state,
        "orientation_actor_to_world_wxyz",
        "orientation_wxyz",
    )
    if quaternion is not None:
        if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
            raise ValueError("orientation quaternion must have shape [frames, 4]")
        return _quaternion_to_matrix(quaternion)
    rotation_6d = _first_array(
        state,
        "rotation_actor_to_world_6d",
        "rotation_6d",
    )
    if rotation_6d is None or rotation_6d.ndim != 2 or rotation_6d.shape[-1] != 6:
        raise ValueError("state artifact has no supported actor-to-world orientation")
    return rot6d_to_matrix(rotation_6d)


def _torch_tensor(array: np.ndarray, dtype: Any | None = None) -> Any:
    if torch is None:  # pragma: no cover - exercised only without torch installed.
        raise ImportError("TorqueDataset samples require PyTorch")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return tensor.to(dtype=dtype) if dtype is not None else tensor


class TorqueDataset(TorchDataset):
    """PyTorch Dataset over a fixed, deterministic TORQUE Dataset Cohort."""

    def __init__(
        self,
        config_or_data_root: DatasetConfig | str | os.PathLike[str] | None = None,
        *,
        config: DatasetConfig | None = None,
        data_root: str | os.PathLike[str] | None = None,
        split: str | None = "train",
        scenarios: Iterable[str] = DEFAULT_SCENARIOS,
        modalities: Iterable[str] = ("state",),
        mode: str = "train",
        **config_overrides: Any,
    ) -> None:
        if config is not None and config_or_data_root is not None:
            raise TypeError("pass either config or a positional dataset root, not both")
        if config is None and isinstance(config_or_data_root, DatasetConfig):
            config = config_or_data_root
        if config is None:
            root = data_root if data_root is not None else config_or_data_root
            if root is None:
                raise TypeError("TorqueDataset requires DatasetConfig or data_root")
            scenario_values = (scenarios,) if isinstance(scenarios, str) else tuple(scenarios)
            modality_values = (modalities,) if isinstance(modalities, str) else tuple(modalities)
            config = DatasetConfig(
                data_root=os.fspath(root),
                split=split,
                scenarios=scenario_values,
                modalities=modality_values,
                mode=mode,
                **config_overrides,
            )
        elif config_overrides:
            raise TypeError("configuration overrides cannot be combined with DatasetConfig")

        self.config = config
        self._all_eligible_cohort, self.exclusions = self._discover_cohort()
        self.cohort = tuple(
            record
            for record in self._all_eligible_cohort
            if self.config.split in {None, "all"} or record.split == self.config.split
        )
        self.trajectory_ids = tuple(record.trajectory_id for record in self.cohort)
        self.trajectory_paths = tuple(record.run_dir for record in self.cohort)
        self._record_by_id = {record.trajectory_id: record for record in self.cohort}
        self.normalization_stats = self._fit_normalization()
        self.windows = tuple(self._index_windows())
        self.window_index = self.windows

    def dataloader(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Return a trajectory-aware PyTorch DataLoader for this dataset."""

        return make_dataloader(
            self,
            batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            num_workers=num_workers,
            **kwargs,
        )

    def _discover_cohort(self) -> tuple[tuple[TrajectoryRecord, ...], tuple[dict[str, Any], ...]]:
        paths = self.config.paths
        schedule = _read_json(paths["schedule"])
        split_doc = _read_json(paths["split"])
        split_map = _split_map(split_doc)
        conditions = [
            condition
            for condition in schedule.get("conditions", [])
            if condition.get("scenario") in self.config.scenarios
        ]
        conditions.sort(
            key=lambda condition: _trajectory_id(
                str(condition.get("scenario", "")),
                str(condition.get("object_id", "")),
                str(condition.get("condition_id", "")),
            )
        )

        eligible: list[TrajectoryRecord] = []
        exclusions: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for condition in conditions:
            scenario = str(condition.get("scenario", ""))
            object_id = str(condition.get("object_id", ""))
            condition_id = str(condition.get("condition_id", ""))
            trajectory_id = _trajectory_id(scenario, object_id, condition_id)
            exclusion_base = {
                "trajectory_id": trajectory_id,
                "scenario": scenario,
                "object_id": object_id,
                "condition_id": condition_id,
            }
            if trajectory_id in seen_ids:
                exclusions.append({**exclusion_base, "reason": "duplicate_trajectory_id"})
                continue
            seen_ids.add(trajectory_id)
            split = split_map.get(object_id)
            if split is None:
                exclusions.append({**exclusion_base, "reason": "missing_object_split"})
                continue
            run_dir = os.path.abspath(
                os.path.join(self.config.data_root, "v2", scenario, object_id, condition_id)
            )
            record, reason = self._inspect_candidate(
                trajectory_id=trajectory_id,
                scenario=scenario,
                object_id=object_id,
                condition_id=condition_id,
                split=split,
                run_dir=run_dir,
                geometry_dir=paths["geometry"],
                utonia_dir=paths["utonia"],
                condition_parameters=condition.get("parameters", {}),
            )
            if record is None:
                exclusions.append({**exclusion_base, "reason": reason})
            else:
                eligible.append(record)
        eligible.sort(key=lambda record: record.trajectory_id)
        exclusions.sort(key=lambda entry: entry["trajectory_id"])
        return tuple(eligible), tuple(exclusions)

    def _inspect_candidate(
        self,
        *,
        trajectory_id: str,
        scenario: str,
        object_id: str,
        condition_id: str,
        split: str,
        run_dir: str,
        geometry_dir: str,
        utonia_dir: str,
        condition_parameters: Mapping[str, Any],
    ) -> tuple[TrajectoryRecord | None, str | None]:
        if not os.path.isdir(run_dir):
            return None, "missing_run_directory"
        paths = {
            "state": os.path.join(run_dir, "state.npz"),
            "modality_timestamps": os.path.join(run_dir, "modality_timestamps.npz"),
            "metadata": os.path.join(run_dir, "metadata.json"),
            "validation": os.path.join(run_dir, "validation.json"),
            "contacts": os.path.join(run_dir, "contacts.npz"),
            "events": os.path.join(run_dir, "events.json"),
            "rgb_a": os.path.join(run_dir, "rgb_a"),
            "depth_a": os.path.join(run_dir, "depth_a"),
        }
        for name in ("metadata", "validation", "state", "modality_timestamps"):
            if not os.path.isfile(paths[name]):
                return None, f"missing_artifact:{name}"
        try:
            metadata = _read_json(paths["metadata"])
        except (OSError, ValueError):
            return None, "unreadable_artifact:metadata"
        if metadata.get("status") != "completed":
            return None, "metadata_status_not_completed"
        try:
            validation = _read_json(paths["validation"])
        except (OSError, ValueError):
            return None, "unreadable_artifact:validation"
        if validation.get("valid") is not True:
            return None, "validation_not_valid"
        try:
            state = _load_npz(paths["state"])
            modality_timestamps = _load_npz(paths["modality_timestamps"])
        except ValueError:
            return None, "unreadable_state_arrays"

        frame_indices = _first_array(state, "frame_index")
        timestamps = _first_array(state, "time_s", "timestamps")
        if frame_indices is None or timestamps is None:
            return None, "missing_state_timeline"
        frame_indices = np.asarray(frame_indices).reshape(-1)
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        frame_count = len(frame_indices)
        if frame_count == 0 or len(timestamps) != frame_count:
            return None, "frame_count_mismatch"
        if not np.issubdtype(frame_indices.dtype, np.integer):
            frame_indices = frame_indices.astype(np.int64)
        else:
            frame_indices = frame_indices.astype(np.int64, copy=False)
        if not np.array_equal(frame_indices, np.arange(frame_indices[0], frame_indices[0] + frame_count)):
            return None, "noncontiguous_frame_index"
        validation_count = validation.get("frame_count", metadata.get("frames_captured"))
        if validation_count is not None and int(validation_count) != frame_count:
            return None, "frame_count_mismatch"
        if frame_count < self.config.min_frames:
            return None, "insufficient_duration"

        for field_name, field_value in (
            ("position_com_world_m", _first_array(state, "position_com_world_m", "position_com_world")),
            ("linear_velocity_com_world_m_s", _first_array(state, "linear_velocity_com_world_m_s", "linear_velocity_com_world")),
        ):
            if field_value is None or field_value.shape != (frame_count, 3):
                return None, f"invalid_state_field:{field_name}"
        quaternion = _first_array(state, "orientation_actor_to_world_wxyz", "orientation_wxyz")
        rotation_6d = _first_array(state, "rotation_actor_to_world_6d", "rotation_6d")
        if quaternion is not None:
            if quaternion.shape != (frame_count, 4):
                return None, "invalid_state_field:orientation"
        elif rotation_6d is None or rotation_6d.shape != (frame_count, 6):
            return None, "invalid_state_field:orientation"
        actor_angular_velocity = _first_array(
            state, "angular_velocity_actor_rad_s", "angular_velocity_actor"
        )
        world_angular_velocity = _first_array(
            state, "angular_velocity_world_rad_s", "angular_velocity_world"
        )
        if actor_angular_velocity is None and world_angular_velocity is None:
            return None, "invalid_state_field:angular_velocity"
        if actor_angular_velocity is not None and actor_angular_velocity.shape != (frame_count, 3):
            return None, "invalid_state_field:angular_velocity_actor"
        if world_angular_velocity is not None and world_angular_velocity.shape != (frame_count, 3):
            return None, "invalid_state_field:angular_velocity_world"

        modality_frames = _first_array(modality_timestamps, "frame_index")
        if modality_frames is None or not np.array_equal(
            np.asarray(modality_frames).reshape(-1).astype(np.int64), frame_indices
        ):
            return None, "frame_count_mismatch"
        state_times = _first_array(modality_timestamps, "state_time_s")
        if state_times is None:
            return None, "missing_artifact:state_time_s"
        if not _synchronized(state_times, timestamps, self.config.sync_tolerance_s):
            return None, "timestamp_mismatch:state"
        if len(timestamps) > 1 and not np.allclose(
            np.diff(timestamps), 1.0 / CAPTURE_FREQUENCY_HZ, atol=1e-6, rtol=0
        ):
            return None, "timestamp_rate_mismatch"

        frustum_visible: tuple[bool, ...] | None = None
        contact_active = _first_array(state, "contact_active")
        if contact_active is not None and len(contact_active) != frame_count:
            return None, "frame_count_mismatch"
        contact_active_tuple = (
            tuple(bool(value) for value in np.asarray(contact_active).reshape(-1))
            if contact_active is not None
            else None
        )
        required_frames = set(int(value) for value in frame_indices)
        if "rgb_a" in self.config.required_modalities or "depth_a" in self.config.required_modalities:
            for modality, timestamp_key, suffixes in (
                ("rgb_a", "rgb_a_time_s", (".jpg", ".jpeg", ".png")),
                ("depth_a", "depth_a_time_s", (".npz", ".npy")),
            ):
                if modality not in self.config.required_modalities:
                    continue
                if not os.path.isdir(paths[modality]):
                    return None, f"missing_artifact:{modality}"
                if not required_frames.issubset(_frame_files(paths[modality], suffixes)):
                    return None, f"incomplete_artifact:{modality}"
                modality_times = _first_array(modality_timestamps, timestamp_key)
                if modality_times is None or not _synchronized(
                    modality_times, timestamps, self.config.sync_tolerance_s
                ):
                    return None, f"timestamp_mismatch:{modality}"
            visibility = _first_array(
                modality_timestamps, "frustum_visible_a", "camera_a_visible"
            )
            if visibility is None:
                visible_pixels = _first_array(modality_timestamps, "visible_pixels_a")
                if visible_pixels is not None:
                    visibility = np.asarray(visible_pixels) > 0
            if visibility is None or len(visibility) != frame_count:
                return None, "missing_artifact:camera_a_visibility"
            frustum_visible = tuple(bool(value) for value in np.asarray(visibility).reshape(-1))

        utonia_path: str | None = None
        if "utonia" in self.config.required_modalities:
            candidate = os.path.abspath(os.path.join(utonia_dir, f"{object_id}.npz"))
            if not os.path.isfile(candidate):
                return None, "missing_artifact:utonia"
            try:
                utonia = _load_npz_keys(candidate, ("global_mean", "global_max"))
                mean = _first_array(utonia, "global_mean")
                maximum = _first_array(utonia, "global_max")
                if (
                    mean is None
                    or maximum is None
                    or mean.ndim != 1
                    or mean.shape != (1386,)
                    or maximum.shape != mean.shape
                ):
                    return None, "invalid_artifact:utonia"
            except ValueError:
                return None, "unreadable_artifact:utonia"
            utonia_path = candidate

        return (
            TrajectoryRecord(
                trajectory_id=trajectory_id,
                scenario=scenario,
                object_id=object_id,
                condition_id=condition_id,
                split=split,
                run_dir=run_dir,
                state_path=os.path.abspath(paths["state"]),
                modality_timestamps_path=os.path.abspath(paths["modality_timestamps"]),
                metadata_path=os.path.abspath(paths["metadata"]),
                validation_path=os.path.abspath(paths["validation"]),
                contacts_path=os.path.abspath(paths["contacts"]) if os.path.isfile(paths["contacts"]) else None,
                events_path=os.path.abspath(paths["events"]) if os.path.isfile(paths["events"]) else None,
                rgb_a_path=os.path.abspath(paths["rgb_a"]) if "rgb_a" in self.config.required_modalities else None,
                depth_a_path=os.path.abspath(paths["depth_a"]) if "depth_a" in self.config.required_modalities else None,
                geometry_path=os.path.abspath(os.path.join(geometry_dir, f"{object_id}.json")),
                utonia_path=utonia_path,
                frame_count=frame_count,
                frame_indices=tuple(int(value) for value in frame_indices),
                timestamps=tuple(float(value) for value in timestamps),
                frustum_visible_a=frustum_visible,
                contact_active=contact_active_tuple,
                condition_parameters=dict(condition_parameters),
            ),
            None,
        )

    def _fit_normalization(self) -> NormalizationStats:
        train_records: dict[str, TrajectoryRecord] = {}
        for record in self._all_eligible_cohort:
            if record.split == "train":
                train_records.setdefault(record.object_id, record)

        depth_mean = 0.0
        depth_std = 1.0
        if "depth_a" in self.config.required_modalities and self.config.normalize_depth:
            values: list[np.ndarray] = []
            limit = max(1, self.config.normalization_pixels_per_object)
            for record in (train_records[key] for key in sorted(train_records)):
                if record.depth_a_path is None:
                    continue
                frame = record.frame_indices[0]
                try:
                    path = _frame_path(record.depth_a_path, frame, (".npz", ".npy"))
                    depth, valid = _read_depth(path)
                except (OSError, ValueError, KeyError):
                    continue
                finite = depth[valid]
                if len(finite) > limit:
                    finite = finite[:: int(np.ceil(len(finite) / limit))]
                if len(finite):
                    values.append(finite.astype(np.float64, copy=False))
            if values:
                flattened = np.concatenate(values)
                depth_mean = float(np.mean(flattened))
                depth_std = float(np.std(flattened))
                if not np.isfinite(depth_std) or depth_std < 1e-8:
                    depth_std = 1.0

        utonia_mean: np.ndarray | None = None
        utonia_std: np.ndarray | None = None
        utonia_max_mean: np.ndarray | None = None
        utonia_max_std: np.ndarray | None = None
        if "utonia" in self.config.required_modalities and self.config.normalize_utonia:
            pooled_mean: list[np.ndarray] = []
            pooled_max: list[np.ndarray] = []
            for record in (train_records[key] for key in sorted(train_records)):
                if record.utonia_path is None:
                    continue
                try:
                    features = _load_npz_keys(record.utonia_path, ("global_mean", "global_max"))
                    mean = _first_array(features, "global_mean")
                    maximum = _first_array(features, "global_max")
                except ValueError:
                    continue
                if mean is not None and maximum is not None:
                    pooled_mean.append(mean.astype(np.float64))
                    pooled_max.append(maximum.astype(np.float64))
            if pooled_mean:
                mean_matrix = np.stack(pooled_mean)
                max_matrix = np.stack(pooled_max)
                utonia_mean = np.mean(mean_matrix, axis=0).astype(np.float32)
                utonia_std = np.std(mean_matrix, axis=0).astype(np.float32)
                utonia_std[utonia_std < 1e-8] = 1.0
                utonia_max_mean = np.mean(max_matrix, axis=0).astype(np.float32)
                utonia_max_std = np.std(max_matrix, axis=0).astype(np.float32)
                utonia_max_std[utonia_max_std < 1e-8] = 1.0
        return NormalizationStats(
            depth_mean,
            depth_std,
            utonia_mean,
            utonia_std,
            utonia_max_mean,
            utonia_max_std,
        )

    def _index_windows(self) -> list[WindowIndex]:
        windows: list[WindowIndex] = []
        for trajectory_index, record in enumerate(self.cohort):
            endpoints = range(
                OBSERVATION_NATIVE_FRAMES,
                record.frame_count - FORECAST_NATIVE_FRAMES,
            )
            if self.config.mode in {"evaluation", "eval"}:
                endpoints = range(
                    OBSERVATION_NATIVE_FRAMES,
                    record.frame_count - FORECAST_NATIVE_FRAMES,
                    self.config.evaluation_stride_samples * MODEL_STRIDE,
                )
            for endpoint in endpoints:
                if record.frame_indices[endpoint] % MODEL_STRIDE:
                    continue
                observations = tuple(range(endpoint - OBSERVATION_NATIVE_FRAMES, endpoint + 1, MODEL_STRIDE))
                targets = tuple(range(endpoint + MODEL_STRIDE, endpoint + FORECAST_NATIVE_FRAMES + 1, MODEL_STRIDE))
                observation_visible = (
                    record.frustum_visible_a is None
                    or all(record.frustum_visible_a[index] for index in observations)
                )
                validity = {
                    "complete_future": targets[-1] < record.frame_count,
                    "observation_visible": observation_visible,
                    "valid": observation_visible,
                }
                if not validity["complete_future"]:
                    continue
                if not observation_visible:
                    if self.config.invalid_window_policy == "error":
                        raise ValueError(
                            f"invalid Camera A visibility window in {record.trajectory_id}"
                        )
                    if self.config.invalid_window_policy in {"drop", "skip"}:
                        continue
                windows.append(
                    WindowIndex(
                        trajectory_index=trajectory_index,
                        trajectory_id=record.trajectory_id,
                        observation_indices=observations,
                        target_indices=targets,
                        endpoint_index=endpoint,
                        native_frame_start=record.frame_indices[observations[0]],
                        native_frame_end=record.frame_indices[endpoint],
                        native_target_end=record.frame_indices[targets[-1]],
                        observation_timestamps=tuple(record.timestamps[index] for index in observations),
                        target_timestamps=tuple(record.timestamps[index] for index in targets),
                        contact_phase=self._contact_phase(record, endpoint),
                        validity_flags=validity,
                    )
                )
        return windows

    @staticmethod
    def _contact_phase(record: TrajectoryRecord, endpoint: int) -> str:
        if record.events_path is not None:
            try:
                events = _read_json(record.events_path)
                phases = events.get("phases", {})
                onset = _phase_index(phases.get("contact_onset"))
                rebound = _phase_index(phases.get("rebound_or_contact_end"))
                if onset is not None and endpoint < onset:
                    return "no_contact"
                if onset is not None and endpoint == onset:
                    return "contact_onset"
                if rebound is not None and endpoint >= rebound:
                    return "post_contact"
                if onset is not None:
                    return "sustained_contact"
            except (OSError, ValueError):
                pass
        if record.contact_active is not None:
            active = record.contact_active[endpoint]
            if active and (endpoint == 0 or not record.contact_active[endpoint - 1]):
                return "contact_onset"
            if active:
                return "sustained_contact"
            if any(record.contact_active[:endpoint]):
                return "post_contact"
            return "no_contact"
        return "unknown"

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        record = self.cohort[window.trajectory_index]
        state = _load_npz(record.state_path)
        observations = np.asarray(window.observation_indices, dtype=np.int64)
        targets = np.asarray(window.target_indices, dtype=np.int64)
        rotations = _state_rotations(state)
        position = _required_state_array(
            state, "position_com_world_m", "position_com_world", shape_last=3
        )
        velocity = _required_state_array(
            state,
            "linear_velocity_com_world_m_s",
            "linear_velocity_com_world",
            shape_last=3,
        )
        angular_velocity = _first_array(
            state, "angular_velocity_actor_rad_s", "angular_velocity_actor"
        )
        if angular_velocity is None:
            world_angular_velocity = _required_state_array(
                state, "angular_velocity_world_rad_s", "angular_velocity_world", shape_last=3
            )
            angular_velocity = np.einsum("nij,nj->ni", np.swapaxes(rotations, -1, -2), world_angular_velocity)
        angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        p_end = np.asarray(position[window.endpoint_index], dtype=np.float64)
        r_end = rotations[window.endpoint_index]

        observed_state = _make_state(
            position[observations] - p_end,
            rotations[observations],
            velocity[observations],
            angular_velocity[observations],
        )
        relative_rotations = np.einsum("ij,njk->nik", r_end.T, rotations[targets])
        target_state = _make_state(
            position[targets] - p_end,
            relative_rotations,
            velocity[targets],
            angular_velocity[targets],
        )
        availability = np.zeros((OBSERVATION_SAMPLES, 3), dtype=bool)
        availability[:, 0] = True
        past: dict[str, Any] = {
            "state": _torch_tensor(observed_state.astype(np.float32)),
            "rgb_a": None,
            "depth_a": None,
            "depth_valid_a": None,
            "timestamps": _torch_tensor(np.asarray(record.timestamps, dtype=np.float32)[observations]),
            "availability_mask": _torch_tensor(availability),
        }
        if "rgb_a" in self.config.required_modalities:
            rgb = self._load_rgb(record, observations)
            past["rgb_a"] = _torch_tensor(rgb.astype(np.float32))
            past["availability_mask"][:, 1] = True
        if "depth_a" in self.config.required_modalities:
            depth, valid = self._load_depth(record, observations)
            past["depth_a"] = _torch_tensor(depth.astype(np.float32))
            past["depth_valid_a"] = _torch_tensor(valid)
            past["availability_mask"][:, 2] = True

        static: dict[str, Any] = {
            "utonia_global_mean": None,
            "utonia_global_max": None,
            "geometry_available": False,
        }
        if "utonia" in self.config.required_modalities:
            features = _load_npz_keys(record.utonia_path or "", ("global_mean", "global_max"))
            mean = _first_array(features, "global_mean")
            maximum = _first_array(features, "global_max")
            if mean is None or maximum is None:
                raise ValueError(f"Utonia artifact has no pooled features: {record.utonia_path}")
            if self.config.normalize_utonia and self.normalization_stats.utonia_mean is not None:
                mean = (mean - self.normalization_stats.utonia_mean) / self.normalization_stats.utonia_std
                maximum = (maximum - self.normalization_stats.utonia_max_mean) / self.normalization_stats.utonia_max_std
            static["utonia_global_mean"] = _torch_tensor(np.asarray(mean, dtype=np.float32))
            static["utonia_global_max"] = _torch_tensor(np.asarray(maximum, dtype=np.float32))
            static["geometry_available"] = True

        return {
            "past": past,
            "static": static,
            "observation_end": {
                "p_com_world": _torch_tensor(p_end.astype(np.float32)),
                "r_actor_to_world": _torch_tensor(r_end.astype(np.float32)),
            },
            "target": {
                "state": _torch_tensor(target_state.astype(np.float32)),
                "timestamps": _torch_tensor(np.asarray(record.timestamps, dtype=np.float32)[targets]),
            },
            "metadata": {
                "object_id": record.object_id,
                "trajectory_id": record.trajectory_id,
                "scenario": record.scenario,
                "condition_id": record.condition_id,
                "split": record.split,
                "contact_phase": window.contact_phase,
                "validity_flags": dict(window.validity_flags),
            },
        }

    def _load_rgb(self, record: TrajectoryRecord, indices: np.ndarray) -> np.ndarray:
        if record.rgb_a_path is None:
            raise RuntimeError("RGB A was not requested during discovery")
        height, width = self.config.image_size
        frames: list[np.ndarray] = []
        for index in indices:
            frame = record.frame_indices[int(index)]
            path = _frame_path(record.rgb_a_path, frame, (".jpg", ".jpeg", ".png"))
            with Image.open(path) as image:
                image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
                frames.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return np.stack(frames)

    def _load_depth(self, record: TrajectoryRecord, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if record.depth_a_path is None:
            raise RuntimeError("depth A was not requested during discovery")
        height, width = self.config.image_size
        depths: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index in indices:
            frame = record.frame_indices[int(index)]
            path = _frame_path(record.depth_a_path, frame, (".npz", ".npy"))
            depth, valid = _read_depth(path)
            weighted = np.where(valid, depth, 0.0).astype(np.float32)
            valid_image = Image.fromarray(valid.astype(np.float32), mode="F")
            weight_image = Image.fromarray(weighted, mode="F")
            resized_valid = np.asarray(
                valid_image.resize((width, height), Image.Resampling.NEAREST), dtype=np.float32
            ) > 0.5
            resized_fraction = np.asarray(
                valid_image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
            )
            resized_weight = np.asarray(
                weight_image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
            )
            resized = np.zeros_like(resized_weight, dtype=np.float32)
            valid_pixels = resized_valid & (resized_fraction > 1e-6)
            resized[valid_pixels] = resized_weight[valid_pixels] / resized_fraction[valid_pixels]
            if self.config.normalize_depth:
                resized[valid_pixels] = (
                    resized[valid_pixels] - self.normalization_stats.depth_mean
                ) / self.normalization_stats.depth_std
            depths.append(resized[None])
            masks.append(resized_valid[None])
        return np.stack(depths), np.stack(masks)


def _synchronized(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    return len(left) == len(right) and bool(np.allclose(left, right, atol=tolerance, rtol=0))


def _required_state_array(
    state: Mapping[str, np.ndarray], *names: str, shape_last: int
) -> np.ndarray:
    value = _first_array(state, *names)
    if value is None or value.ndim != 2 or value.shape[-1] != shape_last:
        raise ValueError(f"state artifact is missing one of: {', '.join(names)}")
    return np.asarray(value, dtype=np.float64)


def _make_state(
    position: np.ndarray,
    rotations: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(position, dtype=np.float64),
            matrix_to_rot6d(np.asarray(rotations, dtype=np.float64)),
            np.asarray(velocity, dtype=np.float64),
            np.asarray(angular_velocity, dtype=np.float64),
        ),
        axis=-1,
    )


def _phase_index(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    index = value.get("state_index")
    return int(index) if index is not None else None


def _read_depth(path: str) -> tuple[np.ndarray, np.ndarray]:
    if path.endswith(".npy"):
        depth = np.asarray(np.load(path, allow_pickle=False))
    else:
        loaded = _load_npz(path)
        depth = _first_array(loaded, "depth")
        if depth is None and len(loaded) == 1:
            depth = next(iter(loaded.values()))
        if depth is None:
            raise ValueError(f"depth artifact has no depth array: {path}")
    if depth.ndim != 2:
        raise ValueError(f"depth frame must be two-dimensional: {path}")
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0)
    return depth, valid


class TrajectoryBatchSampler:
    """Batch windows while guaranteeing one window per trajectory per batch."""

    def __init__(
        self,
        dataset: TorqueDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0
        self._groups: dict[str, list[int]] = {}
        for index, window in enumerate(dataset.windows):
            self._groups.setdefault(window.trajectory_id, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        grouped_windows: list[dict[str, list[int]]] = []
        for _, indices in sorted(self._groups.items()):
            by_phase: dict[str, list[int]] = {}
            for index in indices:
                phase = self.dataset.windows[index].contact_phase
                by_phase.setdefault(phase, []).append(index)
            if self.shuffle:
                for phase_indices in by_phase.values():
                    rng.shuffle(phase_indices)
            grouped_windows.append(by_phase)
        group_order = list(range(len(grouped_windows)))
        if self.shuffle:
            rng.shuffle(group_order)
        phases = list(_CONTACT_PHASE_ORDER)
        extra_phases = {
            phase for by_phase in grouped_windows for phase in by_phase if phase not in phases
        }
        phases.extend(sorted(extra_phases))
        pointers = [dict.fromkeys(by_phase, 0) for by_phase in grouped_windows]
        phase_cursor = 0
        while any(
            pointer[phase] < len(by_phase[phase])
            for by_phase, pointer in zip(grouped_windows, pointers)
            for phase in by_phase
        ):
            batch: list[int] = []
            selected_groups: set[int] = set()
            for slot in range(self.batch_size):
                desired_phase = phases[(phase_cursor + slot) % len(phases)]
                available_groups = [
                    group_index
                    for group_index in group_order
                    if group_index not in selected_groups
                    and any(
                        pointers[group_index][phase] < len(grouped_windows[group_index][phase])
                        for phase in grouped_windows[group_index]
                    )
                ]
                if not available_groups:
                    break
                preferred = [
                    group_index
                    for group_index in available_groups
                    if pointers[group_index].get(desired_phase, 0)
                    < len(grouped_windows[group_index].get(desired_phase, []))
                ]
                group_index = (preferred or available_groups)[0]
                available_phases = [
                    phase
                    for phase in phases
                    if pointers[group_index].get(phase, 0)
                    < len(grouped_windows[group_index].get(phase, []))
                ]
                phase = desired_phase if desired_phase in available_phases else available_phases[0]
                batch.append(grouped_windows[group_index][phase][pointers[group_index][phase]])
                pointers[group_index][phase] += 1
                selected_groups.add(group_index)
            if not batch:
                break
            phase_cursor = (phase_cursor + len(batch)) % len(phases)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())


def torque_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate TORQUE samples while preserving ``None`` for unrequested inputs."""

    if not samples:
        raise ValueError("cannot collate an empty TORQUE batch")
    first = samples[0]
    if isinstance(first, Mapping):
        return {key: torque_collate([sample[key] for sample in samples]) for key in first}
    if all(sample is None for sample in samples):
        return None
    if any(sample is None for sample in samples):
        raise ValueError("a TORQUE modality is inconsistently present in one batch")
    if torch is not None and all(torch.is_tensor(sample) for sample in samples):
        return torch.stack(list(samples))
    if all(isinstance(sample, (bool, np.bool_)) for sample in samples):
        return _torch_tensor(np.asarray(samples, dtype=bool)) if torch is not None else list(samples)
    if all(isinstance(sample, (int, float, str)) for sample in samples):
        return list(samples)
    if all(isinstance(sample, Mapping) for sample in samples):
        keys = first.keys()
        return {key: torque_collate([sample[key] for sample in samples]) for key in keys}
    return list(samples)


def make_dataloader(
    dataset: TorqueDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
    num_workers: int = 0,
    **kwargs: Any,
) -> Any:
    """Build a DataLoader using :class:`TrajectoryBatchSampler`."""

    if torch is None:  # pragma: no cover
        raise ImportError("make_dataloader requires PyTorch")
    from torch.utils.data import DataLoader

    sampler = TrajectoryBatchSampler(
        dataset, batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=torque_collate,
        num_workers=num_workers,
        **kwargs,
    )


TorqueDataLoader = make_dataloader


__all__ = [
    "CAPTURE_FREQUENCY_HZ",
    "DatasetConfig",
    "FORECAST_NATIVE_FRAMES",
    "MINIMUM_FRAMES",
    "MODEL_STRIDE",
    "NormalizationStats",
    "TorqueDataset",
    "TorqueDataLoader",
    "TrajectoryBatchSampler",
    "TrajectoryRecord",
    "WindowIndex",
    "make_dataloader",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "torque_collate",
]
