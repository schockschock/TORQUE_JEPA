"""Canonical observation/future window indexing.

The simulator captures state, RGB, and depth synchronously at 120 Hz. Models
consume a stride-two 60 Hz view: a 0.25 s observation window (16 samples) and a
0.50 s forecast horizon (30 samples). For an even native observation-end frame
``e``, observation samples are ``e-30, e-28, ..., e`` and future samples are
``e+2, e+4, ..., e+60``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    observation_seconds: float = 0.25
    prediction_seconds: float = 0.50
    observation_samples: int = 16
    prediction_samples: int = 30
    capture_frequency_hz: int = 120
    model_frequency_hz: int = 60
    position_reference: str = "com"
    coordinate_frame: str = "observation_end_translated_world"
    prediction_mode: str = "autoregressive"
    rotation_representation: str = "rot6d"

    @property
    def stride(self) -> int:
        return self.capture_frequency_hz // self.model_frequency_hz

    @property
    def observation_span_frames(self) -> int:
        return int(self.observation_seconds * self.capture_frequency_hz)

    @property
    def prediction_span_frames(self) -> int:
        return int(self.prediction_seconds * self.capture_frequency_hz)


def observation_offsets() -> range:
    """Native-frame offsets from the observation end ``e`` (inclusive)."""
    span = int(0.25 * 120)
    stride = 2
    return range(-span, 1, stride)


def future_offsets() -> range:
    """Native-frame offsets from the observation end ``e`` (exclusive)."""
    span = int(0.50 * 120)
    stride = 2
    return range(stride, span + 1, stride)


def observation_indices(e: int) -> list[int]:
    return [e + off for off in observation_offsets()]


def future_indices(e: int) -> list[int]:
    return [e + off for off in future_offsets()]


def eligible_endpoints(frame_count: int, cfg: WindowConfig | None = None) -> list[int]:
    """Every eligible 60 Hz observation endpoint in ``[0, frame_count)``.

    An endpoint ``e`` is eligible when a full observation window (``e-30``) and a
    full forecast horizon (``e+60``) fall inside the native frame range.
    """
    cfg = cfg or WindowConfig()
    obs_lo = abs(min(observation_offsets()))
    fut_hi = max(future_offsets())
    first = obs_lo
    last = frame_count - 1 - fut_hi
    if last < first:
        return []
    return list(range(first, last + 1, cfg.stride))
