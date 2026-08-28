"""Canonical state-only forecasting samples.

``TorqueDataset`` turns validated pilot trajectories into Forecasting Samples
using the canonical 16-observation / 30-target window and Observation-End Frame
semantics. It conditions on the observation-end COM world location while every
forecast output is expressed relative to the observation end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from data.normalization import StateNormalizationStats, apply_state_normalization
from data.state_transform import (
    TrajectoryState,
    observed_state,
    observation_end,
    target_state,
)
from data.windows import (
    WindowConfig,
    eligible_endpoints,
    future_indices,
    observation_indices,
)


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    object_id: str
    scenario: str
    condition_id: str
    split: str
    state_path: str


def load_trajectory_state(state_path: str) -> TrajectoryState:
    """Load the synchronized state arrays used by the dataset from ``state.npz``."""
    d = np.load(state_path, allow_pickle=True)
    return TrajectoryState(
        time_s=d["time_s"].astype(np.float64),
        position_com_world_m=d["position_com_world_m"].astype(np.float64),
        orientation_actor_to_world_wxyz=d["orientation_actor_to_world_wxyz"].astype(np.float64),
        linear_velocity_com_world_m_s=d["linear_velocity_com_world_m_s"].astype(np.float64),
        angular_velocity_actor_rad_s=d["angular_velocity_actor_rad_s"].astype(np.float64),
    )


class TorqueDataset:
    """Indexes eligible windows across trajectories and yields canonical samples."""

    def __init__(
        self,
        records: list[TrajectoryRecord],
        window_config: WindowConfig | None = None,
        normalization_stats: StateNormalizationStats | None = None,
        state_loader: Callable[[TrajectoryRecord], TrajectoryState] | None = None,
    ):
        self.records = list(records)
        self.cfg = window_config or WindowConfig()
        self.stats = normalization_stats
        self._state_loader = state_loader or (lambda rec: load_trajectory_state(rec.state_path))
        self._states: dict[int, TrajectoryState] = {}
        self._window_index: list[tuple[int, int]] = []
        self._build_index()

    def _load(self, record_idx: int) -> TrajectoryState:
        if record_idx not in self._states:
            self._states[record_idx] = self._state_loader(self.records[record_idx])
        return self._states[record_idx]

    def _build_index(self) -> None:
        for record_idx, _ in enumerate(self.records):
            state = self._load(record_idx)
            for e in eligible_endpoints(len(state.time_s), self.cfg):
                self._window_index.append((record_idx, e))

    @property
    def window_record_indices(self) -> list[int]:
        return [record_idx for record_idx, _ in self._window_index]

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> dict:
        record_idx, e = self._window_index[index]
        record = self.records[record_idx]
        state = self._load(record_idx)
        obs_idx = observation_indices(e)
        fut_idx = future_indices(e)
        oe = observation_end(state, e)
        sample = {
            "past": {
                "state": observed_state(state, e, obs_idx).astype(np.float32),
                "rgb_a": None,
                "depth_a": None,
                "depth_valid_a": None,
                "timestamps": state.time_s[obs_idx].astype(np.float32),
                "availability_mask": np.tile([True, False, False], (len(obs_idx), 1)),
            },
            "static": {
                "utonia_global_mean": None,
                "utonia_global_max": None,
                "geometry_available": False,
            },
            "observation_end": {
                "p_com_world": oe["p_com_world"].astype(np.float32),
                "r_actor_to_world": oe["r_actor_to_world"].astype(np.float32),
            },
            "target": {
                "state": target_state(state, e, fut_idx).astype(np.float32),
                "timestamps": state.time_s[fut_idx].astype(np.float32),
            },
            "metadata": {
                "object_id": record.object_id,
                "trajectory_id": record.trajectory_id,
                "scenario": record.scenario,
                "condition_id": record.condition_id,
                "split": record.split,
            },
        }
        if self.stats is not None:
            sample = apply_state_normalization(sample, self.stats)
        return sample


class TrajectoryGroupedBatchSampler:
    """Yields batches with at most one window per trajectory.

    Each epoch is a seeded, deterministic pass: every trajectory's windows are
    shuffled, then batches are assembled by drawing one window from each of up to
    ``batch_size`` non-empty trajectories at a time. Every window is used exactly
    once per epoch and no batch repeats a trajectory. Consequently a batch can
    only be full when ``batch_size`` does not exceed the number of trajectories.
    """

    def __init__(
        self,
        dataset: TorqueDataset,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        by_trajectory: dict[int, list[int]] = {}
        for window_idx, record_idx in enumerate(self.dataset.window_record_indices):
            by_trajectory.setdefault(record_idx, []).append(window_idx)

        rng = np.random.default_rng(self.seed)
        queues: dict[int, list[int]] = {}
        for record_idx, windows in by_trajectory.items():
            windows = list(windows)
            rng.shuffle(windows)
            queues[record_idx] = windows
        pointer = {record_idx: 0 for record_idx in queues}

        batches: list[list[int]] = []
        while True:
            order = list(queues.keys())
            rng.shuffle(order)
            batch: list[int] = []
            for record_idx in order:
                if pointer[record_idx] < len(queues[record_idx]):
                    batch.append(queues[record_idx][pointer[record_idx]])
                    pointer[record_idx] += 1
                    if len(batch) == self.batch_size:
                        break
            if not batch:
                break
            batches.append(batch)

        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        return batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
