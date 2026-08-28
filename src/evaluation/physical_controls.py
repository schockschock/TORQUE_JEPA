"""Physical controls for TORQUE Kinematic-State Forecasting."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch


def forecast_controls(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Forecast a canonical TORQUE batch with the physical control methods.

    Returned states use the canonical future-target representation: COM
    displacement and orientation relative to the Observation-End Frame, world
    linear velocity, and actor-frame angular velocity.
    """

    past_state = batch["past"]["state"]
    target_timestamps = batch["target"]["timestamps"]
    if not torch.is_tensor(past_state) or past_state.ndim != 3 or past_state.shape[-1] != 15:
        raise ValueError("past.state must have shape [batch, observation, 15]")
    if not torch.is_tensor(target_timestamps) or target_timestamps.ndim != 2:
        raise ValueError("target.timestamps must have shape [batch, horizon]")
    if target_timestamps.shape[0] != past_state.shape[0]:
        raise ValueError("past.state and target.timestamps must share a batch dimension")

    observation_timestamps = batch["past"].get("timestamps")
    observation_end = batch["observation_end"]
    r_end = observation_end["r_actor_to_world"]
    if (
        not torch.is_tensor(observation_timestamps)
        or observation_timestamps.ndim != 2
        or observation_timestamps.shape[0] != past_state.shape[0]
    ):
        raise ValueError("past.timestamps must have shape [batch, observation]")
    if (
        not torch.is_tensor(r_end)
        or r_end.shape != (past_state.shape[0], 3, 3)
    ):
        raise ValueError("observation_end.r_actor_to_world must have shape [batch, 3, 3]")

    last_state = past_state[:, -1]
    persistence = last_state.unsqueeze(1).expand(-1, target_timestamps.shape[1], -1).clone()
    persistence_relative_rotation = (
        r_end.transpose(-1, -2) @ _rot6d_to_matrix(last_state[:, 3:9])
    )
    persistence[..., 3:9] = _matrix_to_rot6d(persistence_relative_rotation).unsqueeze(1)
    elapsed = target_timestamps - observation_timestamps[:, -1:]
    linear_velocity_world = last_state[:, 9:12]
    angular_velocity_world = torch.einsum("bij,bj->bi", r_end, last_state[:, 12:15])
    rotation_increment = _rotation_exp(elapsed.unsqueeze(-1) * angular_velocity_world.unsqueeze(1))
    world_rotation = rotation_increment @ r_end.unsqueeze(1)
    relative_rotation = r_end.transpose(-1, -2).unsqueeze(1) @ world_rotation
    angular_velocity_actor = torch.einsum(
        "bhji,bj->bhi", world_rotation, angular_velocity_world
    )
    constant_velocity = torch.cat(
        (
            elapsed.unsqueeze(-1) * linear_velocity_world.unsqueeze(1),
            _matrix_to_rot6d(relative_rotation),
            linear_velocity_world.unsqueeze(1).expand(-1, elapsed.shape[1], -1),
            angular_velocity_actor,
        ),
        dim=-1,
    )
    return {"persistence": persistence, "constant_velocity": constant_velocity}


def evaluate_forecasts(
    forecasts: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    *,
    bootstrap_samples: int = 1_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate target-format forecasts after reconstructing world quantities.

    Windows are reduced to trajectories, trajectories to scenarios within each
    object, and scenarios to equally weighted objects. Confidence intervals
    bootstrap objects, preserving the required object-disjoint unit of
    uncertainty.
    """

    target_state = batch["target"]["state"]
    observation_end = batch["observation_end"]
    r_end = observation_end["r_actor_to_world"]
    p_end = observation_end["p_com_world"]
    if not torch.is_tensor(target_state) or target_state.ndim != 3 or target_state.shape[-1] != 15:
        raise ValueError("target.state must have shape [batch, horizon, 15]")
    if not torch.is_tensor(r_end) or r_end.shape != (target_state.shape[0], 3, 3):
        raise ValueError("observation_end.r_actor_to_world must have shape [batch, 3, 3]")
    if not torch.is_tensor(p_end) or p_end.shape != (target_state.shape[0], 3):
        raise ValueError("observation_end.p_com_world must have shape [batch, 3]")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    metadata = _metadata_for_batch(batch["metadata"], target_state.shape[0])
    target_world = _reconstruct_world_state(target_state, p_end, r_end)
    controls: dict[str, Any] = {}
    for name, prediction in forecasts.items():
        if (
            not torch.is_tensor(prediction)
            or prediction.shape != target_state.shape
        ):
            raise ValueError(
                f"forecast {name!r} must have shape {tuple(target_state.shape)}"
            )
        prediction = prediction.to(dtype=target_state.dtype, device=target_state.device)
        prediction_world = _reconstruct_world_state(prediction, p_end, r_end)
        errors = {
            "position_m": torch.linalg.vector_norm(
                prediction_world["position_m"] - target_world["position_m"], dim=-1
            ),
            "orientation_deg": _orientation_error_degrees(
                prediction_world["orientation"], target_world["orientation"]
            ),
            "linear_velocity_m_s": torch.linalg.vector_norm(
                prediction_world["linear_velocity_m_s"] - target_world["linear_velocity_m_s"], dim=-1
            ),
            "angular_velocity_rad_s": torch.linalg.vector_norm(
                prediction_world["angular_velocity_rad_s"]
                - target_world["angular_velocity_rad_s"],
                dim=-1,
            ),
        }
        controls[str(name)] = _make_control_report(
            errors, metadata, bootstrap_samples, bootstrap_seed
        )

    return {
        "controls": controls,
        "cohort": {
            "trajectory_ids": sorted(set(metadata["trajectory_id"])),
            "object_ids": sorted(set(metadata["object_id"])),
            "window_count": target_state.shape[0],
        },
        "reduction": "window -> trajectory -> scenario -> object",
        "bootstrap": {
            "unit": "object",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
        },
    }


def evaluate_controls(
    batches: Iterable[Mapping[str, Any]],
    *,
    forecast: Callable[[Mapping[str, Any]], Mapping[str, torch.Tensor]] = forecast_controls,
    bootstrap_samples: int = 1_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Forecast and evaluate every batch in an evaluation DataLoader.

    Forecasts are collected before evaluation so windows from the same
    trajectory are reduced together even when a DataLoader splits them across
    batches. The returned cohort identifiers make a pilot result rerunnable
    against the same fixed Dataset Cohort.
    """

    materialized_batches = list(batches)
    if not materialized_batches:
        raise ValueError("cannot evaluate an empty sequence of batches")
    per_batch_forecasts = [forecast(batch) for batch in materialized_batches]
    control_names = tuple(per_batch_forecasts[0])
    if not control_names or any(tuple(predictions) != control_names for predictions in per_batch_forecasts):
        raise ValueError("every batch must provide the same non-empty set of control forecasts")
    combined_forecasts = {
        name: torch.cat([predictions[name] for predictions in per_batch_forecasts], dim=0)
        for name in control_names
    }
    return evaluate_forecasts(
        combined_forecasts,
        _concatenate_batches(materialized_batches),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _make_control_report(
    errors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, list[str]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_object = {
        name: _reduce_to_objects(values, metadata).detach().to(device="cpu", dtype=torch.float64)
        for name, values in errors.items()
    }
    horizon = {name: values.mean(dim=0).tolist() for name, values in per_object.items()}
    summaries = {
        "position_ade_m": per_object["position_m"].mean(dim=1),
        "position_fde_m": per_object["position_m"][:, -1],
        "orientation_mean_deg": per_object["orientation_deg"].mean(dim=1),
        "orientation_final_deg": per_object["orientation_deg"][:, -1],
        "linear_velocity_mean_m_s": per_object["linear_velocity_m_s"].mean(dim=1),
        "linear_velocity_final_m_s": per_object["linear_velocity_m_s"][:, -1],
        "angular_velocity_mean_rad_s": per_object["angular_velocity_rad_s"].mean(dim=1),
        "angular_velocity_final_rad_s": per_object["angular_velocity_rad_s"][:, -1],
    }
    summary = {name: float(values.mean()) for name, values in summaries.items()}
    uncertainty = {
        name: _bootstrap_interval(values, bootstrap_samples, bootstrap_seed)
        for name, values in summaries.items()
    }
    return {
        "horizon": horizon,
        "summary": summary,
        "uncertainty": uncertainty,
        "object_count": next(iter(per_object.values())).shape[0],
    }


def _concatenate_batches(batches: list[Mapping[str, Any]]) -> dict[str, Any]:
    def concatenate(path: tuple[str, ...]) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for batch in batches:
            value: Any = batch
            for key in path:
                value = value[key]
            if not torch.is_tensor(value):
                raise ValueError(f"{'.'.join(path)} must be a tensor in every batch")
            values.append(value)
        return torch.cat(values, dim=0)

    metadata: dict[str, list[str]] = {}
    for field in ("object_id", "trajectory_id", "scenario"):
        values: list[str] = []
        for batch in batches:
            field_values = batch["metadata"].get(field)
            if isinstance(field_values, str):
                field_values = [field_values]
            if not isinstance(field_values, (list, tuple)):
                raise ValueError(f"metadata.{field} must contain one value per batch item")
            values.extend(str(value) for value in field_values)
        metadata[field] = values
    return {
        "past": {
            "state": concatenate(("past", "state")),
            "timestamps": concatenate(("past", "timestamps")),
        },
        "observation_end": {
            "p_com_world": concatenate(("observation_end", "p_com_world")),
            "r_actor_to_world": concatenate(("observation_end", "r_actor_to_world")),
        },
        "target": {
            "state": concatenate(("target", "state")),
            "timestamps": concatenate(("target", "timestamps")),
        },
        "metadata": metadata,
    }


def _metadata_for_batch(metadata: Mapping[str, Any], batch_size: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field in ("object_id", "trajectory_id", "scenario"):
        values = metadata.get(field)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)) or len(values) != batch_size:
            raise ValueError(f"metadata.{field} must contain one value per batch item")
        result[field] = [str(value) for value in values]
    return result


def _reduce_to_objects(values: torch.Tensor, metadata: Mapping[str, list[str]]) -> torch.Tensor:
    """Apply the required window -> trajectory -> scenario -> object reduction."""

    trajectories: dict[str, list[torch.Tensor]] = {}
    trajectory_context: dict[str, tuple[str, str]] = {}
    for index, trajectory_id in enumerate(metadata["trajectory_id"]):
        context = (metadata["object_id"][index], metadata["scenario"][index])
        previous = trajectory_context.setdefault(trajectory_id, context)
        if previous != context:
            raise ValueError(f"trajectory {trajectory_id!r} has inconsistent object or scenario metadata")
        trajectories.setdefault(trajectory_id, []).append(values[index])

    scenarios: dict[tuple[str, str], list[torch.Tensor]] = {}
    for trajectory_id, windows in trajectories.items():
        scenarios.setdefault(trajectory_context[trajectory_id], []).append(torch.stack(windows).mean(dim=0))
    objects: dict[str, list[torch.Tensor]] = {}
    for (object_id, _), scenario_trajectories in scenarios.items():
        objects.setdefault(object_id, []).append(torch.stack(scenario_trajectories).mean(dim=0))
    return torch.stack([torch.stack(objects[object_id]).mean(dim=0) for object_id in sorted(objects)])


def _bootstrap_interval(values: torch.Tensor, samples: int, seed: int) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        values.shape[0], (samples, values.shape[0]), generator=generator, device="cpu"
    )
    bootstrap_means = values[indices].mean(dim=1)
    lower, upper = torch.quantile(
        bootstrap_means, torch.tensor([0.025, 0.975], dtype=values.dtype)
    )
    return {"lower": float(lower), "upper": float(upper)}


def _reconstruct_world_state(
    state: torch.Tensor, p_end: torch.Tensor, r_end: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Reconstruct world quantities from canonical future-target state tensors."""

    orientation = r_end.unsqueeze(1) @ _rot6d_to_matrix(state[..., 3:9])
    return {
        "position_m": p_end.unsqueeze(1) + state[..., :3],
        "orientation": orientation,
        "linear_velocity_m_s": state[..., 9:12],
        "angular_velocity_rad_s": _world_angular_velocity(state, orientation),
    }


def _world_angular_velocity(state: torch.Tensor, world_rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhij,bhj->bhi", world_rotation, state[..., 12:15])


def _orientation_error_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _rot6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
    first = values[..., :3]
    second = values[..., 3:6]
    eps = torch.finfo(values.dtype).eps
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    first_fallback = torch.zeros_like(first)
    first_fallback[..., 0] = 1.0
    first = torch.where(first_norm > eps, first / first_norm.clamp_min(eps), first_fallback)
    second = second - (first * second).sum(dim=-1, keepdim=True) * first
    second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    fallback_axis = torch.nn.functional.one_hot(
        first.abs().argmin(dim=-1), num_classes=3
    ).to(dtype=values.dtype)
    fallback_second = fallback_axis - (first * fallback_axis).sum(dim=-1, keepdim=True) * first
    second = torch.where(
        second_norm > eps, second / second_norm.clamp_min(eps),
        fallback_second / torch.linalg.vector_norm(fallback_second, dim=-1, keepdim=True),
    )
    return torch.stack((first, second, torch.linalg.cross(first, second, dim=-1)), dim=-1)


def _matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _rotation_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Return Rodrigues' SO(3) exponential for batched rotation vectors."""

    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    angle_squared = angle.square()
    safe_angle = angle.clamp_min(torch.finfo(rotation_vector.dtype).eps)
    safe_angle_squared = angle_squared.clamp_min(torch.finfo(rotation_vector.dtype).eps)
    sine_over_angle = torch.where(
        angle < 1e-4,
        1.0 - angle_squared / 6.0,
        torch.sin(angle) / safe_angle,
    )
    one_minus_cosine_over_angle_squared = torch.where(
        angle < 1e-4,
        0.5 - angle_squared / 24.0,
        (1.0 - torch.cos(angle)) / safe_angle_squared,
    )
    x, y, z = rotation_vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return (
        identity
        + sine_over_angle.unsqueeze(-1) * skew
        + one_minus_cosine_over_angle_squared.unsqueeze(-1) * (skew @ skew)
    )
