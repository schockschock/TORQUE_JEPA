import numpy as np
import pytest

from data.windows import (
    WindowConfig,
    eligible_endpoints,
    future_offsets,
    observation_offsets,
)


def test_observation_offsets_are_sixteen_stride_two():
    off = observation_offsets()
    assert list(off) == list(range(-30, 1, 2))
    assert len(off) == 16


def test_future_offsets_are_thirty_stride_two():
    off = future_offsets()
    assert list(off) == list(range(2, 61, 2))
    assert len(off) == 30


def test_config_defaults_match_canonical_window():
    cfg = WindowConfig()
    assert cfg.observation_samples == 16
    assert cfg.prediction_samples == 30
    assert cfg.capture_frequency_hz == 120
    assert cfg.model_frequency_hz == 60


def test_stride_is_ratio_of_frequencies():
    cfg = WindowConfig()
    assert cfg.stride == 2


def test_eligible_endpoints_bounds():
    cfg = WindowConfig()
    frame_count = 163
    endpoints = eligible_endpoints(frame_count, cfg)
    assert len(endpoints) > 0
    assert endpoints[0] == 30
    assert endpoints[-1] == 102
    assert all((e % cfg.stride) == 0 for e in endpoints)
    assert all(e - 30 >= 0 for e in endpoints)
    assert all(e + 60 <= frame_count - 1 for e in endpoints)


def test_eligible_endpoints_empty_when_too_short():
    cfg = WindowConfig()
    assert eligible_endpoints(90, cfg) == []


def test_observation_and_future_timestamps_span_correct_seconds():
    cfg = WindowConfig()
    time_s = np.arange(163) / 120.0
    e = 60
    obs_idx = [e + off for off in observation_offsets()]
    fut_idx = [e + off for off in future_offsets()]
    assert np.isclose(time_s[e] - time_s[obs_idx[0]], 0.25)
    assert np.isclose(time_s[fut_idx[-1]] - time_s[e], 0.50)
