"""Tests for the two-level configuration-conditional calibration core."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from measurement_calibration.spectral_calibration import (
    CampaignConfiguration,
    _accumulate_campaign_objective_and_gradients,
    _forward_campaign,
    _initialize_campaign_states,
    _initialize_persistent_parameters,
    _refresh_campaign_latent_and_variance,
    _resolve_effective_variance_floor_power2,
    _resolve_sensor_reference_weight,
    _zero_gradient_dict,
    calibrate_sensor_observations,
    evaluate_persistent_calibration,
    fit_two_level_calibration,
)

from tests._synthetic_two_level import build_synthetic_two_level_fixture


def _build_first_campaign_gradient_check_state() -> tuple[
    Any,
    Any,
    Any,
    np.ndarray,
    float,
]:
    """Build one deterministic single-campaign state for gradient checks."""

    fixture = build_synthetic_two_level_fixture()
    raw_configuration_features = np.stack(
        [
            campaign.configuration.to_feature_vector()
            for campaign in fixture.corpus.campaigns
        ],
        axis=0,
    )
    all_observations_power = np.concatenate(
        [
            campaign.observations_power.reshape(-1)
            for campaign in fixture.corpus.campaigns
        ]
    )
    configuration_feature_mean = np.mean(raw_configuration_features, axis=0)
    configuration_feature_scale = np.std(raw_configuration_features, axis=0)
    configuration_feature_scale = np.where(
        configuration_feature_scale > 1.0e-12,
        configuration_feature_scale,
        1.0,
    )
    frequency_min_hz = float(
        min(np.min(campaign.frequency_hz) for campaign in fixture.corpus.campaigns)
    )
    frequency_max_hz = float(
        max(np.max(campaign.frequency_hz) for campaign in fixture.corpus.campaigns)
    )
    sensor_reference_weight = _resolve_sensor_reference_weight(
        sensor_ids=fixture.corpus.sensor_ids,
        sensor_reference_weight_by_id=None,
    )
    variance_floor_power2 = _resolve_effective_variance_floor_power2(
        all_observations_power=all_observations_power,
        fit_config=fixture.fit_config,
    )
    parameter_state = _initialize_persistent_parameters(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
        all_observations_power=all_observations_power,
        variance_floor_power2=variance_floor_power2,
        configuration_feature_mean=configuration_feature_mean,
        configuration_feature_scale=configuration_feature_scale,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    campaign_states = _initialize_campaign_states(
        corpus=fixture.corpus,
        sensor_index_by_id={
            sensor_id: sensor_index
            for sensor_index, sensor_id in enumerate(fixture.corpus.sensor_ids)
        },
        basis_config=fixture.basis_config,
        parameter_state=parameter_state,
        sensor_reference_weight=sensor_reference_weight,
        fit_config=fixture.fit_config,
        variance_floor_power2=variance_floor_power2,
        configuration_feature_mean=configuration_feature_mean,
        configuration_feature_scale=configuration_feature_scale,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    first_campaign_state = campaign_states[0]
    _refresh_campaign_latent_and_variance(
        campaign_state=first_campaign_state,
        parameter_state=parameter_state,
        sensor_reference_weight=sensor_reference_weight,
        fit_config=fixture.fit_config,
        variance_floor_power2=variance_floor_power2,
    )
    return (
        fixture,
        parameter_state,
        first_campaign_state,
        sensor_reference_weight,
        variance_floor_power2,
    )


def _single_campaign_objective(
    fixture: Any,
    parameter_state: Any,
    campaign_state: Any,
    sensor_reference_weight: np.ndarray,
    variance_floor_power2: float,
) -> float:
    """Evaluate one campaign objective with a fixed latent state."""

    forward_cache = _forward_campaign(
        parameter_state=parameter_state,
        campaign_state=campaign_state,
        sensor_reference_weight=sensor_reference_weight,
    )
    return _accumulate_campaign_objective_and_gradients(
        campaign_index=0,
        campaign_state=campaign_state,
        forward_cache=forward_cache,
        parameter_state=parameter_state,
        gradients=_zero_gradient_dict(
            parameter_state=parameter_state,
            campaign_states=[campaign_state],
        ),
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
        variance_floor_power2=variance_floor_power2,
        sensor_reference_weight=sensor_reference_weight,
    )


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
    assert result.fit_diagnostics.selected_objective_value == pytest.approx(
        float(np.min(result.objective_history))
    )
    assert result.fit_diagnostics.selected_outer_iteration == int(
        np.argmin(result.objective_history)
    )
    assert result.fit_diagnostics.n_completed_outer_iterations == len(
        result.objective_history
    )
    assert len(result.fit_diagnostics.max_gradient_norm_by_outer_iteration) == len(
        result.objective_history
    )
    assert np.all(
        np.isfinite(
            np.asarray(
                result.fit_diagnostics.max_gradient_norm_by_outer_iteration,
                dtype=np.float64,
            )
        )
    )
    assert np.all(
        np.asarray(
            result.fit_diagnostics.max_gradient_norm_by_outer_iteration,
            dtype=np.float64,
        )
        >= 0.0
    )
    assert result.effective_variance_floor_power2 is not None
    assert result.effective_variance_floor_power2 > fixture.fit_config.sigma_min

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
            np.sqrt(
                np.mean((campaign_state.residual_variance_power2 - true_variance) ** 2)
            )
        )
        baseline_variance_rmse = float(np.sqrt(np.mean(true_variance**2)))

        assert np.allclose(
            np.mean(campaign_state.deviation_log_gain, axis=0),
            0.0,
            atol=1.0e-8,
        )
        assert np.min(campaign_state.residual_variance_power2) >= (
            result.effective_variance_floor_power2 - 1.0e-12
        )
        gain_improvement_count += int(fitted_gain_rmse < baseline_gain_rmse)
        floor_improvement_count += int(fitted_floor_rmse < baseline_floor_rmse)
        assert fitted_variance_rmse < 2.0 * baseline_variance_rmse

    assert gain_improvement_count >= 2
    assert floor_improvement_count == len(result.campaign_states)


@pytest.mark.parametrize(
    ("parameter_name", "parameter_index"),
    [
        ("gain_head_bias", (0,)),
        ("variance_head_bias", (0,)),
    ],
)
def test_single_campaign_gradients_match_finite_differences(
    parameter_name: str,
    parameter_index: tuple[int, ...],
) -> None:
    """Hand-derived objective gradients should match finite differences."""

    (
        fixture,
        parameter_state,
        campaign_state,
        sensor_reference_weight,
        variance_floor_power2,
    ) = _build_first_campaign_gradient_check_state()
    gradients = _zero_gradient_dict(
        parameter_state=parameter_state,
        campaign_states=[campaign_state],
    )
    forward_cache = _forward_campaign(
        parameter_state=parameter_state,
        campaign_state=campaign_state,
        sensor_reference_weight=sensor_reference_weight,
    )
    _accumulate_campaign_objective_and_gradients(
        campaign_index=0,
        campaign_state=campaign_state,
        forward_cache=forward_cache,
        parameter_state=parameter_state,
        gradients=gradients,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
        variance_floor_power2=variance_floor_power2,
        sensor_reference_weight=sensor_reference_weight,
    )

    parameter = getattr(parameter_state, parameter_name)
    step_size = 1.0e-6
    parameter[parameter_index] += step_size
    positive_objective = _single_campaign_objective(
        fixture=fixture,
        parameter_state=parameter_state,
        campaign_state=campaign_state,
        sensor_reference_weight=sensor_reference_weight,
        variance_floor_power2=variance_floor_power2,
    )
    parameter[parameter_index] -= 2.0 * step_size
    negative_objective = _single_campaign_objective(
        fixture=fixture,
        parameter_state=parameter_state,
        campaign_state=campaign_state,
        sensor_reference_weight=sensor_reference_weight,
        variance_floor_power2=variance_floor_power2,
    )
    parameter[parameter_index] += step_size

    numerical_gradient = (positive_objective - negative_objective) / (2.0 * step_size)
    analytic_gradient = float(gradients[parameter_name][parameter_index])

    assert analytic_gradient == pytest.approx(
        numerical_gradient,
        rel=5.0e-4,
        abs=1.0e-5,
    )


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
                (
                    fixture.deployment_observations_power
                    - fixture.deployment_true_latent_power
                )
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
    assert (
        deployment.propagated_variance_power2.shape == deployment.calibrated_power.shape
    )
    assert np.all(deployment.propagated_variance_power2 > 0.0)
    assert deployment.uncertainty_scope == "observation_noise_only"


def test_evaluate_persistent_calibration_rejects_frequency_extrapolation_by_default() -> (
    None
):
    """Deployment evaluation should refuse silent frequency extrapolation."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )
    out_of_support_frequency_hz = np.asarray(
        [result.frequency_min_hz - 1.0e6, result.frequency_max_hz + 1.0e6],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="frequency grid lies outside"):
        evaluate_persistent_calibration(
            result=result,
            sensor_id=fixture.deployment_sensor_id,
            configuration=fixture.deployment_configuration,
            frequency_hz=out_of_support_frequency_hz,
        )

    curves = evaluate_persistent_calibration(
        result=result,
        sensor_id=fixture.deployment_sensor_id,
        configuration=fixture.deployment_configuration,
        frequency_hz=out_of_support_frequency_hz,
        allow_frequency_extrapolation=True,
    )

    assert curves.trust_diagnostics.frequency_extrapolation_detected is True
    assert curves.trust_diagnostics.n_frequencies_below_support == 1
    assert curves.trust_diagnostics.n_frequencies_above_support == 1
    assert len(curves.trust_diagnostics.standardized_configuration) == len(
        fixture.deployment_configuration.to_feature_vector()
    )


def test_evaluate_persistent_calibration_flags_configuration_ood() -> None:
    """Deployment evaluation should expose configuration OOD diagnostics."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )
    ood_configuration = CampaignConfiguration(
        central_frequency_hz=140.0e6,
        span_hz=fixture.deployment_configuration.span_hz,
        resolution_bandwidth_hz=fixture.deployment_configuration.resolution_bandwidth_hz,
        lna_gain_db=fixture.deployment_configuration.lna_gain_db,
        vga_gain_db=fixture.deployment_configuration.vga_gain_db,
        acquisition_interval_s=fixture.deployment_configuration.acquisition_interval_s,
        antenna_amplifier_enabled=fixture.deployment_configuration.antenna_amplifier_enabled,
    )

    with pytest.raises(ValueError, match="outside the stored training envelope"):
        evaluate_persistent_calibration(
            result=result,
            sensor_id=fixture.deployment_sensor_id,
            configuration=ood_configuration,
            frequency_hz=fixture.deployment_frequency_hz,
            allow_configuration_ood=False,
        )

    curves = evaluate_persistent_calibration(
        result=result,
        sensor_id=fixture.deployment_sensor_id,
        configuration=ood_configuration,
        frequency_hz=fixture.deployment_frequency_hz,
        allow_configuration_ood=True,
    )

    assert curves.trust_diagnostics.configuration_support_available is True
    assert curves.trust_diagnostics.configuration_out_of_distribution is True
    assert "central_frequency_hz" in curves.trust_diagnostics.out_of_range_feature_names
    assert curves.trust_diagnostics.max_abs_standardized_feature == pytest.approx(
        max(abs(value) for value in curves.trust_diagnostics.standardized_configuration)
    )
