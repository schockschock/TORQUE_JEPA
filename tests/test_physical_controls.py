import pytest
import torch

from evaluation.physical_controls import (
    evaluate_controls,
    evaluate_forecasts,
    forecast_controls,
)


def _canonical_batch():
    """One canonical batch with an identity observation-end orientation."""

    past_state = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.5, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.5, 0.0, 0.0, 1.0],
            ]
        ]
    )
    return {
        "past": {
            "state": past_state,
            "timestamps": torch.tensor([[0.0, 0.25]]),
        },
        "observation_end": {
            "p_com_world": torch.tensor([[3.0, 4.0, 5.0]]),
            "r_actor_to_world": torch.eye(3).unsqueeze(0),
        },
        "target": {"timestamps": torch.tensor([[0.5, 0.75, 1.0]])},
        "metadata": {
            "object_id": ["object-a"],
            "trajectory_id": ["freefall/object-a/condition-a"],
            "scenario": ["freefall"],
        },
    }


def test_persistence_emits_all_four_kinematic_state_families_at_every_horizon():
    forecasts = forecast_controls(_canonical_batch())

    persistence = forecasts["persistence"]
    assert persistence.shape == (1, 3, 15)
    expected_state = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, -1.0, 0.5, 0.0, 0.0, 1.0]
    )
    assert torch.equal(persistence[0], expected_state.expand(3, -1))


def test_persistence_converts_observation_orientation_to_future_target_coordinates():
    batch = _canonical_batch()
    r_end = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    batch["observation_end"]["r_actor_to_world"] = r_end
    batch["past"]["state"][:, :, 3:9] = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])

    persistence = forecast_controls(batch)["persistence"]

    assert torch.equal(
        persistence[0, :, 3:9],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(3, -1),
    )


def test_constant_velocity_integrates_world_position_and_orientation_from_observation_end():
    forecasts = forecast_controls(_canonical_batch())

    constant_velocity = forecasts["constant_velocity"]
    elapsed = torch.tensor([0.25, 0.5, 0.75])
    assert torch.allclose(
        constant_velocity[0, :, :3],
        torch.stack((2.0 * elapsed, -elapsed, 0.5 * elapsed), dim=-1),
    )
    assert torch.allclose(
        constant_velocity[0, :, 3:9],
        torch.stack(
            (
                torch.cos(elapsed),
                torch.sin(elapsed),
                torch.zeros_like(elapsed),
                -torch.sin(elapsed),
                torch.cos(elapsed),
                torch.zeros_like(elapsed),
            ),
            dim=-1,
        ),
    )
    assert torch.allclose(constant_velocity[0, :, 9:12], torch.tensor([2.0, -1.0, 0.5]))
    assert torch.allclose(constant_velocity[0, :, 12:15], torch.tensor([0.0, 0.0, 1.0]))


def test_evaluation_reconstructs_all_state_families_and_reports_zero_error_for_exact_forecast():
    batch = _canonical_batch()
    forecast = forecast_controls(batch)["constant_velocity"]
    batch["target"]["state"] = forecast.clone()

    report = evaluate_forecasts({"constant_velocity": forecast}, batch, bootstrap_samples=32)

    metrics = report["controls"]["constant_velocity"]
    assert metrics["horizon"]["position_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert metrics["horizon"]["orientation_deg"] == pytest.approx([0.0, 0.0, 0.0])
    assert metrics["horizon"]["linear_velocity_m_s"] == pytest.approx([0.0, 0.0, 0.0])
    assert metrics["horizon"]["angular_velocity_rad_s"] == pytest.approx([0.0, 0.0, 0.0])
    assert metrics["summary"]["position_ade_m"] == pytest.approx(0.0)
    assert metrics["summary"]["position_fde_m"] == pytest.approx(0.0)
    assert metrics["uncertainty"]["position_ade_m"] == {"lower": 0.0, "upper": 0.0}


def test_control_evaluation_reduces_windows_through_trajectory_scenario_and_object():
    def batch(errors, object_ids, trajectory_ids, scenarios):
        count = len(errors)
        target = torch.zeros((count, 1, 15))
        target[:, :, 3] = 1.0
        target[:, :, 7] = 1.0
        prediction = target.clone()
        prediction[:, :, 0] = torch.tensor(errors).unsqueeze(1)
        return {
            "past": {
                "state": target.clone(),
                "timestamps": torch.zeros((count, 1)),
            },
            "observation_end": {
                "p_com_world": torch.zeros((count, 3)),
                "r_actor_to_world": torch.eye(3).expand(count, -1, -1).clone(),
            },
            "target": {"state": target, "timestamps": torch.ones((count, 1))},
            "metadata": {
                "object_id": object_ids,
                "trajectory_id": trajectory_ids,
                "scenario": scenarios,
            },
            "expected_prediction": prediction,
        }

    first = batch(
        [1.0, 3.0, 10.0],
        ["object-a", "object-a", "object-a"],
        ["trajectory-a", "trajectory-a", "trajectory-b"],
        ["freefall", "freefall", "ramp"],
    )
    second = batch(
        [4.0], ["object-b"], ["trajectory-c"], ["freefall"]
    )
    batches = [first, second]
    report = evaluate_controls(
        batches,
        forecast=lambda value: {"test": value["expected_prediction"]},
        bootstrap_samples=64,
        bootstrap_seed=17,
    )

    metrics = report["controls"]["test"]
    # object-a: mean((1 + 3) / 2, 10) = 6; object-b: 4; object macro = 5.
    assert metrics["horizon"]["position_m"] == pytest.approx([5.0])
    assert metrics["summary"]["position_ade_m"] == pytest.approx(5.0)
    assert metrics["summary"]["position_fde_m"] == pytest.approx(5.0)
    assert metrics["object_count"] == 2
    assert report["cohort"]["trajectory_ids"] == ["trajectory-a", "trajectory-b", "trajectory-c"]
    assert report["bootstrap"] == {
        "unit": "object",
        "samples": 64,
        "seed": 17,
        "confidence_level": 0.95,
    }
