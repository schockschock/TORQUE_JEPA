"""Train-only normalization for forecasting samples.

Displacement, world linear velocity, actor angular velocity, and the
observation-end world location are normalized by separate statistics fitted on
training objects only. rot6d is never scalar-standardized. Statistics use the
population standard deviation (ddof=0); zero-variance components fall back to
unit scale so normalized values stay finite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POS_SLICE = slice(0, 3)
VEL_SLICE = slice(9, 12)
ANGVEL_SLICE = slice(12, 15)


@dataclass
class StateNormalizationStats:
    position_mean: np.ndarray
    position_std: np.ndarray
    velocity_mean: np.ndarray
    velocity_std: np.ndarray
    angular_velocity_mean: np.ndarray
    angular_velocity_std: np.ndarray
    world_location_mean: np.ndarray
    world_location_std: np.ndarray


def _pooled(samples, key):
    return np.concatenate([s[key] for s in samples], axis=0)


def compute_state_stats(samples) -> StateNormalizationStats:
    """Fit normalization statistics over an iterable of raw samples."""
    samples = list(samples)

    def _concat(accessor):
        parts = []
        for s in samples:
            parts.append(accessor(s))
        return np.concatenate(parts, axis=0)

    pos = _concat(lambda s: np.concatenate([s["past"]["state"][:, POS_SLICE], s["target"]["state"][:, POS_SLICE]], axis=0))
    vel = _concat(lambda s: np.concatenate([s["past"]["state"][:, VEL_SLICE], s["target"]["state"][:, VEL_SLICE]], axis=0))
    angvel = _concat(lambda s: np.concatenate([s["past"]["state"][:, ANGVEL_SLICE], s["target"]["state"][:, ANGVEL_SLICE]], axis=0))
    world = _concat(lambda s: s["observation_end"]["p_com_world"][None, :])

    def _stats(x):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        return mean, std

    pos_mean, pos_std = _stats(pos)
    vel_mean, vel_std = _stats(vel)
    angvel_mean, angvel_std = _stats(angvel)
    world_mean, world_std = _stats(world)
    return StateNormalizationStats(
        position_mean=pos_mean,
        position_std=pos_std,
        velocity_mean=vel_mean,
        velocity_std=vel_std,
        angular_velocity_mean=angvel_mean,
        angular_velocity_std=angvel_std,
        world_location_mean=world_mean,
        world_location_std=world_std,
    )


def _normalize(x, mean, std):
    return (x - mean) / std


def apply_state_normalization(sample: dict, stats: StateNormalizationStats) -> dict:
    """Return a copy of ``sample`` with state components normalized. rot6d is untouched."""
    out = {
        "past": dict(sample["past"]),
        "target": dict(sample["target"]),
        "observation_end": dict(sample["observation_end"]),
        "metadata": sample.get("metadata"),
        "static": sample.get("static"),
    }
    past = out["past"]["state"].copy()
    tgt = out["target"]["state"].copy()
    past[:, POS_SLICE] = _normalize(past[:, POS_SLICE], stats.position_mean, stats.position_std)
    past[:, VEL_SLICE] = _normalize(past[:, VEL_SLICE], stats.velocity_mean, stats.velocity_std)
    past[:, ANGVEL_SLICE] = _normalize(past[:, ANGVEL_SLICE], stats.angular_velocity_mean, stats.angular_velocity_std)
    tgt[:, POS_SLICE] = _normalize(tgt[:, POS_SLICE], stats.position_mean, stats.position_std)
    tgt[:, VEL_SLICE] = _normalize(tgt[:, VEL_SLICE], stats.velocity_mean, stats.velocity_std)
    tgt[:, ANGVEL_SLICE] = _normalize(tgt[:, ANGVEL_SLICE], stats.angular_velocity_mean, stats.angular_velocity_std)
    out["past"]["state"] = past
    out["target"]["state"] = tgt
    out["observation_end"]["p_com_world"] = _normalize(
        sample["observation_end"]["p_com_world"], stats.world_location_mean, stats.world_location_std
    )
    return out
