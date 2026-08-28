import numpy as np
import pytest

from data.normalization import (
    StateNormalizationStats,
    apply_state_normalization,
    compute_state_stats,
)


def make_sample(pos, vel, angvel, p_e):
    n_obs, n_tgt = 1, 1
    state = np.zeros((n_obs, 15))
    state[:, 0:3] = pos
    state[:, 9:12] = vel
    state[:, 12:15] = angvel
    tgt = np.zeros((n_tgt, 15))
    tgt[:, 0:3] = pos
    tgt[:, 9:12] = vel
    tgt[:, 12:15] = angvel
    return {
        "past": {"state": state},
        "target": {"state": tgt},
        "observation_end": {"p_com_world": p_e, "r_actor_to_world": np.eye(3)},
    }


def test_compute_stats_means_and_stds():
    s0 = make_sample(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), np.zeros(3), np.array([5.0, 6.0, 7.0]))
    s1 = make_sample(np.array([3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0]), np.zeros(3), np.array([7.0, 8.0, 9.0]))
    stats = compute_state_stats([s0, s1])
    assert np.allclose(stats.position_mean, [2.0, 3.0, 4.0])
    assert np.allclose(stats.position_std, [1.0, 1.0, 1.0])
    assert np.allclose(stats.velocity_mean, [1.0, 2.0, 3.0])
    assert np.allclose(stats.world_location_mean, [6.0, 7.0, 8.0])


def test_apply_normalizes_position_velocity_angvel():
    s = make_sample(np.array([2.0, 4.0, 6.0]), np.array([0.0, 2.0, 4.0]), np.array([1.0, 1.0, 1.0]), np.array([10.0, 20.0, 30.0]))
    stats = StateNormalizationStats(
        position_mean=np.array([1.0, 2.0, 3.0]),
        position_std=np.array([1.0, 2.0, 3.0]),
        velocity_mean=np.array([0.0, 1.0, 2.0]),
        velocity_std=np.array([2.0, 2.0, 2.0]),
        angular_velocity_mean=np.zeros(3),
        angular_velocity_std=np.array([1.0, 1.0, 1.0]),
        world_location_mean=np.array([10.0, 20.0, 30.0]),
        world_location_std=np.array([10.0, 10.0, 10.0]),
    )
    out = apply_state_normalization(s, stats)
    assert np.allclose(out["past"]["state"][:, 0:3], [1.0, 1.0, 1.0])
    assert np.allclose(out["past"]["state"][:, 9:12], [0.0, 0.5, 1.0])
    assert np.allclose(out["target"]["state"][:, 12:15], [1.0, 1.0, 1.0])
    assert np.allclose(out["observation_end"]["p_com_world"], [0.0, 0.0, 0.0])


def test_apply_leaves_rot6d_untouched():
    s = make_sample(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    s["past"]["state"][:, 3:9] = np.arange(6).reshape(1, 6)
    stats = StateNormalizationStats(
        position_mean=np.zeros(3),
        position_std=np.ones(3),
        velocity_mean=np.zeros(3),
        velocity_std=np.ones(3),
        angular_velocity_mean=np.zeros(3),
        angular_velocity_std=np.ones(3),
        world_location_mean=np.zeros(3),
        world_location_std=np.ones(3),
    )
    out = apply_state_normalization(s, stats)
    assert np.array_equal(out["past"]["state"][:, 3:9], np.arange(6).reshape(1, 6))


def test_apply_does_not_mutate_input():
    s = make_sample(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.zeros(3), np.array([5.0, 6.0, 7.0]))
    stats = StateNormalizationStats(
        position_mean=np.zeros(3),
        position_std=np.array([2.0, 2.0, 2.0]),
        velocity_mean=np.zeros(3),
        velocity_std=np.ones(3),
        angular_velocity_mean=np.zeros(3),
        angular_velocity_std=np.ones(3),
        world_location_mean=np.zeros(3),
        world_location_std=np.ones(3),
    )
    apply_state_normalization(s, stats)
    assert np.allclose(s["past"]["state"][:, 0:3], [1.0, 2.0, 3.0])


def test_zero_std_does_not_produce_nan():
    s = make_sample(np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]), np.zeros(3), np.array([3.0, 3.0, 3.0]))
    stats = compute_state_stats([s, s])
    out = apply_state_normalization(s, stats)
    assert np.all(np.isfinite(out["past"]["state"]))
    assert np.all(np.isfinite(out["target"]["state"]))
