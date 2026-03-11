"""Tests for the two-level configuration-conditional calibration core."""

from __future__ import annotations

import numpy as np

from measurement_calibration.spectral_calibration import (
    calibrate_sensor_observations,
    evaluate_persistent_calibration,
    fit_two_level_calibration,
)

from tests._synthetic_two_level import build_synthetic_two_level_fixture


def test_fit_two_level_calibration_tracks_campaign_parameters() -> None:
    """Offline fitting should beat naive campaign baselines on synthetic data."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )

    assert np.all(np.isfinite(result.objective_history))
    assert result.objective_history[-1] < result.objective_history[0]

    gain_improvement_count = 0
    floor_improvement_count = 0
    for campaign_state in result.campaign_states:
        true_gain = fixture.true_gain_by_campaign[campaign_state.campaign_label]
        true_floor = fixture.true_floor_by_campaign[campaign_state.campaign_label]
        true_variance = fixture.true_variance_by_campaign[campaign_state.campaign_label]

        fitted_gain_rmse = float(
            np.sqrt(np.mean((campaign_state.gain_power - true_gain) ** 2))
        )
        baseline_gain_rmse = float(np.sqrt(np.mean((1.0 - true_gain) ** 2)))
        fitted_floor_rmse = float(
            np.sqrt(np.mean((campaign_state.additive_noise_power - true_floor) ** 2))
        )
        baseline_floor_rmse = float(np.sqrt(np.mean(true_floor**2)))
        fitted_variance_rmse = float(
            np.sqrt(np.mean((campaign_state.residual_variance_power2 - true_variance) ** 2))
        )
        baseline_variance_rmse = float(np.sqrt(np.mean(true_variance**2)))

        assert np.allclose(
            np.mean(campaign_state.deviation_log_gain, axis=0),
            0.0,
            atol=1.0e-8,
        )
        gain_improvement_count += int(fitted_gain_rmse < baseline_gain_rmse)
        floor_improvement_count += int(fitted_floor_rmse < baseline_floor_rmse)
        assert fitted_variance_rmse < 2.0 * baseline_variance_rmse

    assert gain_improvement_count >= 2
    assert floor_improvement_count == len(result.campaign_states)


def test_persistent_gain_respects_global_reference_convention() -> None:
    """Deployment-scale gain curves should remain centered on the global reference."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )

    first_campaign = fixture.corpus.campaigns[0]
    gain_curves = np.stack(
        [
            np.log(
                evaluate_persistent_calibration(
                    result=result,
                    sensor_id=sensor_id,
                    configuration=first_campaign.configuration,
                    frequency_hz=first_campaign.frequency_hz,
                ).gain_power
            )
            for sensor_id in result.sensor_ids
        ],
        axis=0,
    )
    weighted_mean_log_gain = np.sum(
        result.sensor_reference_weight[:, np.newaxis] * gain_curves,
        axis=0,
    )

    assert np.allclose(weighted_mean_log_gain, 0.0, atol=1.0e-8)


def test_calibrate_sensor_observations_improves_single_sensor_deployment() -> None:
    """Deployment should improve latent-spectrum recovery without any common field."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )

    deployment = calibrate_sensor_observations(
        result=result,
        sensor_id=fixture.deployment_sensor_id,
        configuration=fixture.deployment_configuration,
        frequency_hz=fixture.deployment_frequency_hz,
        observations_power=fixture.deployment_observations_power,
    )

    raw_rmse = float(
        np.sqrt(
            np.mean(
                (fixture.deployment_observations_power - fixture.deployment_true_latent_power)
                ** 2
            )
        )
    )
    corrected_rmse = float(
        np.sqrt(
            np.mean(
                (deployment.calibrated_power - fixture.deployment_true_latent_power)
                ** 2
            )
        )
    )

    assert corrected_rmse < 0.75 * raw_rmse
    assert deployment.propagated_variance_power2.shape == deployment.calibrated_power.shape
    assert np.all(deployment.propagated_variance_power2 > 0.0)
