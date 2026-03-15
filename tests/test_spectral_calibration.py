"""Tests for the two-level configuration-conditional calibration core."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import measurement_calibration.spectral_calibration as spectral_calibration_core
from measurement_calibration.spectral_calibration import (
    CampaignConfiguration,
    _configuration_geometry_support,
    _accumulate_campaign_objective_and_gradients,
    _forward_campaign,
    _initialize_campaign_states,
    _initialize_persistent_parameters,
    _inverse_softplus,
    _optimizer_parameter_dict,
    _refresh_campaign_latent_and_variance,
    _resolve_effective_variance_floor_power2,
    _resolve_sensor_reference_weight,
    _same_scene_correlation_penalty_and_gradients,
    _same_scene_consistency_penalty_and_gradients,
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


def _max_gain_edge_jump_ratio(gain_power: np.ndarray) -> float:
    """Recompute the gain-edge instability diagnostic used by the core."""

    gain_jumps = np.abs(np.diff(np.asarray(gain_power, dtype=np.float64), axis=1))
    if gain_jumps.size == 0:
        return 0.0
    boundary_jumps = np.concatenate(
        [
            gain_jumps[:, :1].reshape(-1),
            gain_jumps[:, -1:].reshape(-1),
        ]
    )
    if gain_jumps.shape[1] > 2:
        interior_jumps = gain_jumps[:, 1:-1].reshape(-1)
    else:
        interior_jumps = gain_jumps.reshape(-1)
    if interior_jumps.size == 0:
        return 0.0
    return float(
        np.max(boundary_jumps) / max(float(np.median(interior_jumps)), 1.0e-12)
    )


def _mean_persistent_only_same_scene_dispersion(
    *,
    result: Any,
    fixture: Any,
    log_floor_power: float,
) -> float:
    """Compute the persistent-only same-scene log-dispersion on a corpus.

    This helper mirrors the quantity regularized by the offline consistency
    term, but it evaluates the fitted deployment-scale laws on the frozen
    synthetic campaigns so tests can compare fits trained with different
    weights.
    """

    per_campaign_dispersion: list[float] = []
    for campaign in fixture.corpus.campaigns:
        transformed_corrected_power_by_sensor: list[np.ndarray] = []
        for sensor_id, observations_power in zip(
            campaign.sensor_ids,
            campaign.observations_power,
            strict=True,
        ):
            curves = evaluate_persistent_calibration(
                result=result,
                sensor_id=sensor_id,
                configuration=campaign.configuration,
                frequency_hz=campaign.frequency_hz,
            )
            persistent_only_corrected_power = (
                np.maximum(
                    observations_power - curves.additive_noise_power[np.newaxis, :],
                    0.0,
                )
                / curves.gain_power[np.newaxis, :]
            )
            transformed_corrected_power_by_sensor.append(
                np.log(persistent_only_corrected_power + log_floor_power)
            )

        transformed_corrected_power = np.stack(
            transformed_corrected_power_by_sensor,
            axis=0,
        )
        centered_transformed_power = transformed_corrected_power - np.mean(
            transformed_corrected_power,
            axis=0,
            keepdims=True,
        )
        per_campaign_dispersion.append(float(np.mean(centered_transformed_power**2)))

    return float(np.mean(per_campaign_dispersion))


def _mean_persistent_only_same_scene_pairwise_correlation(
    *,
    result: Any,
    fixture: Any,
    log_floor_power: float,
) -> float:
    """Compute the mean persistent-only same-scene pairwise correlation.

    The correlation is computed across frequency, per aligned acquisition, and
    then averaged over all sensor pairs and campaigns. This mirrors the
    quantity regularized by the new offline correlation penalty.
    """

    per_campaign_correlation: list[float] = []
    for campaign in fixture.corpus.campaigns:
        transformed_corrected_power_by_sensor: list[np.ndarray] = []
        for sensor_id, observations_power in zip(
            campaign.sensor_ids,
            campaign.observations_power,
            strict=True,
        ):
            curves = evaluate_persistent_calibration(
                result=result,
                sensor_id=sensor_id,
                configuration=campaign.configuration,
                frequency_hz=campaign.frequency_hz,
            )
            persistent_only_corrected_power = (
                np.maximum(
                    observations_power - curves.additive_noise_power[np.newaxis, :],
                    0.0,
                )
                / curves.gain_power[np.newaxis, :]
            )
            transformed_corrected_power_by_sensor.append(
                np.log(persistent_only_corrected_power + log_floor_power)
            )

        transformed_corrected_power = np.stack(
            transformed_corrected_power_by_sensor,
            axis=0,
        )
        record_correlations: list[float] = []
        for acquisition_index in range(transformed_corrected_power.shape[1]):
            transformed_record = transformed_corrected_power[:, acquisition_index, :]
            centered_record = transformed_record - np.mean(
                transformed_record,
                axis=1,
                keepdims=True,
            )
            norms = np.sqrt(np.sum(centered_record**2, axis=1))
            for sensor_index in range(centered_record.shape[0] - 1):
                for other_sensor_index in range(
                    sensor_index + 1,
                    centered_record.shape[0],
                ):
                    left = centered_record[sensor_index]
                    right = centered_record[other_sensor_index]
                    if norms[sensor_index] <= 1.0e-6 and norms[other_sensor_index] <= (
                        1.0e-6
                    ):
                        record_correlations.append(1.0)
                        continue
                    record_correlations.append(
                        float(
                            np.dot(left, right)
                            / max(norms[sensor_index], 1.0e-6)
                            / max(norms[other_sensor_index], 1.0e-6)
                        )
                    )
        per_campaign_correlation.append(float(np.mean(record_correlations)))

    return float(np.mean(per_campaign_correlation))


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
    objective_history = np.asarray(result.objective_history, dtype=np.float64)
    objective_deltas = np.diff(objective_history)
    expected_n_objective_increases = int(np.sum(objective_deltas > 0.0))
    assert (
        result.fit_diagnostics.n_objective_increases == expected_n_objective_increases
    )
    if expected_n_objective_increases == 0:
        assert result.fit_diagnostics.max_objective_increase_ratio == pytest.approx(0.0)
    else:
        expected_max_objective_increase_ratio = float(
            np.max(
                objective_deltas[objective_deltas > 0.0]
                / np.maximum(
                    np.abs(objective_history[:-1][objective_deltas > 0.0]), 1.0
                )
            )
        )
        assert result.fit_diagnostics.max_objective_increase_ratio == pytest.approx(
            expected_max_objective_increase_ratio
        )
    assert result.effective_variance_floor_power2 is not None
    assert result.effective_variance_floor_power2 > fixture.fit_config.sigma_min

    gain_improvement_count = 0
    floor_improvement_count = 0
    campaign_by_label = {
        campaign.campaign_label: campaign for campaign in fixture.corpus.campaigns
    }
    expected_campaign_objective_fraction = max(
        float(campaign_state.objective_value)
        for campaign_state in result.campaign_states
    ) / max(abs(result.fit_diagnostics.selected_objective_value), 1.0e-12)
    expected_max_residual_variance_ratio = 0.0
    expected_max_gain_edge_jump_ratio = 0.0
    for campaign_state in result.campaign_states:
        raw_campaign = campaign_by_label[campaign_state.campaign_label]
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
        expected_max_residual_variance_ratio = max(
            expected_max_residual_variance_ratio,
            float(np.max(campaign_state.residual_variance_power2))
            / max(float(np.var(raw_campaign.observations_power)), 1.0e-12),
        )
        expected_max_gain_edge_jump_ratio = max(
            expected_max_gain_edge_jump_ratio,
            _max_gain_edge_jump_ratio(campaign_state.gain_power),
        )
        gain_improvement_count += int(fitted_gain_rmse < baseline_gain_rmse)
        floor_improvement_count += int(fitted_floor_rmse < baseline_floor_rmse)
        assert fitted_variance_rmse < 2.0 * baseline_variance_rmse

    assert gain_improvement_count >= 2
    assert floor_improvement_count == len(result.campaign_states)
    expected_max_embedding_norm = float(
        np.max(np.linalg.norm(result.sensor_embeddings, axis=1))
    )
    assert result.fit_diagnostics.final_max_sensor_embedding_norm == pytest.approx(
        expected_max_embedding_norm
    )
    assert (
        result.fit_diagnostics.final_max_campaign_objective_fraction
        == pytest.approx(expected_campaign_objective_fraction)
    )
    assert result.fit_diagnostics.final_max_residual_variance_ratio == pytest.approx(
        expected_max_residual_variance_ratio
    )
    assert result.fit_diagnostics.final_max_gain_edge_jump_ratio == pytest.approx(
        expected_max_gain_edge_jump_ratio
    )


@pytest.mark.parametrize(
    ("target_owner", "parameter_name", "parameter_index"),
    [
        ("parameter_state", "sensor_embeddings", (0, 0)),
        ("parameter_state", "configuration_encoder_weight", (0, 0)),
        ("parameter_state", "gain_head_bias", (0,)),
        ("parameter_state", "floor_head_weight", (0, 0)),
        ("parameter_state", "variance_head_bias", (0,)),
        ("campaign_state", "delta_log_gain", (0, 0)),
        ("campaign_state", "delta_floor_parameter", (0, 0)),
        ("campaign_state", "delta_variance_parameter", (0, 0)),
    ],
)
def test_single_campaign_gradients_match_finite_differences(
    target_owner: str,
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

    optimizer_parameters = _optimizer_parameter_dict(
        parameter_state=parameter_state,
        campaign_states=[campaign_state],
    )
    gradient_name = (
        parameter_name
        if target_owner == "parameter_state"
        else f"campaign_0_{parameter_name}"
    )
    parameter = optimizer_parameters[gradient_name]
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
    analytic_gradient = float(gradients[gradient_name][parameter_index])

    assert analytic_gradient == pytest.approx(
        numerical_gradient,
        rel=5.0e-4,
        abs=1.0e-5,
    )


def test_same_scene_consistency_penalty_vanishes_for_identical_corrected_spectra() -> (
    None
):
    """Persistent-only consensus loss should vanish when sensors already agree."""

    latent_power = np.asarray(
        [
            [0.35, 0.55, 0.75],
            [0.40, 0.50, 0.70],
        ],
        dtype=np.float64,
    )
    gain_power = np.asarray(
        [
            [1.2, 0.9, 1.1],
            [0.8, 1.4, 1.3],
            [1.5, 1.1, 0.7],
        ],
        dtype=np.float64,
    )
    additive_noise_power = np.asarray(
        [
            [0.02, 0.01, 0.03],
            [0.04, 0.02, 0.01],
            [0.03, 0.05, 0.02],
        ],
        dtype=np.float64,
    )
    observations_power = (
        gain_power[:, np.newaxis, :] * latent_power[np.newaxis, :, :]
        + additive_noise_power[:, np.newaxis, :]
    )

    penalty_value, gradient_log_gain, gradient_floor_parameter = (
        _same_scene_consistency_penalty_and_gradients(
            observations_power=observations_power,
            persistent_log_gain=np.log(gain_power),
            persistent_floor_parameter=_inverse_softplus(additive_noise_power),
            log_floor_power=1.0e-12,
        )
    )

    assert penalty_value == pytest.approx(0.0, abs=1.0e-12)
    assert np.allclose(gradient_log_gain, 0.0, atol=1.0e-12)
    assert np.allclose(gradient_floor_parameter, 0.0, atol=1.0e-12)


def test_same_scene_consistency_penalty_avoids_backend_prod_on_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consistency penalty should not call backend ``prod`` on a shape tuple."""

    class _ProdGuardNamespace:
        """Proxy NumPy while rejecting ``prod`` on Python shape tuples."""

        def __getattr__(self, name: str) -> Any:
            return getattr(np, name)

        def prod(self, values: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(values, tuple):
                raise AssertionError("backend prod received a shape tuple")
            return np.prod(values, *args, **kwargs)

    latent_power = np.asarray([[0.35, 0.55]], dtype=np.float64)
    gain_power = np.asarray([[1.2, 0.9], [0.8, 1.4]], dtype=np.float64)
    additive_noise_power = np.asarray([[0.02, 0.01], [0.04, 0.02]], dtype=np.float64)
    observations_power = (
        gain_power[:, np.newaxis, :] * latent_power[np.newaxis, :, :]
        + additive_noise_power[:, np.newaxis, :]
    )

    monkeypatch.setattr(spectral_calibration_core, "np", _ProdGuardNamespace())
    penalty_value, gradient_log_gain, gradient_floor_parameter = (
        _same_scene_consistency_penalty_and_gradients(
            observations_power=observations_power,
            persistent_log_gain=np.log(gain_power),
            persistent_floor_parameter=_inverse_softplus(additive_noise_power),
            log_floor_power=1.0e-12,
        )
    )

    assert penalty_value >= 0.0
    assert gradient_log_gain.shape == gain_power.shape
    assert gradient_floor_parameter.shape == gain_power.shape


def test_same_scene_correlation_penalty_vanishes_for_identical_log_shapes() -> None:
    """Shape-only correlation loss should ignore per-sensor level offsets."""

    base_spectrum = np.asarray(
        [
            [0.35, 0.55, 0.80, 1.10],
            [0.42, 0.60, 0.88, 1.20],
        ],
        dtype=np.float64,
    )
    corrected_power_by_sensor = np.asarray(
        [
            base_spectrum,
            2.5 * base_spectrum,
            0.4 * base_spectrum,
        ],
        dtype=np.float64,
    )

    penalty_value, gradient_log_gain, gradient_floor_parameter = (
        _same_scene_correlation_penalty_and_gradients(
            observations_power=corrected_power_by_sensor,
            persistent_log_gain=np.zeros((3, 4), dtype=np.float64),
            persistent_floor_parameter=_inverse_softplus(
                np.zeros((3, 4), dtype=np.float64)
            ),
            log_floor_power=1.0e-12,
        )
    )

    assert penalty_value == pytest.approx(0.0, abs=1.0e-12)
    assert np.allclose(gradient_log_gain, 0.0, atol=1.0e-12)
    assert np.allclose(gradient_floor_parameter, 0.0, atol=1.0e-12)


def test_same_scene_correlation_penalty_gradients_match_finite_differences() -> None:
    """Correlation-penalty gradients should match finite differences."""

    observations_power = np.asarray(
        [
            [
                [0.80, 1.10, 0.95, 1.25],
                [0.75, 1.05, 0.90, 1.15],
            ],
            [
                [0.55, 0.92, 0.78, 1.05],
                [0.60, 0.88, 0.82, 1.00],
            ],
        ],
        dtype=np.float64,
    )
    persistent_log_gain = np.log(
        np.asarray(
            [
                [1.15, 0.95, 1.05, 1.10],
                [0.90, 1.05, 0.98, 1.08],
            ],
            dtype=np.float64,
        )
    )
    persistent_floor_parameter = _inverse_softplus(
        np.asarray(
            [
                [0.03, 0.02, 0.01, 0.04],
                [0.01, 0.03, 0.02, 0.02],
            ],
            dtype=np.float64,
        )
    )

    penalty_value, gradient_log_gain, gradient_floor_parameter = (
        _same_scene_correlation_penalty_and_gradients(
            observations_power=observations_power,
            persistent_log_gain=persistent_log_gain,
            persistent_floor_parameter=persistent_floor_parameter,
            log_floor_power=1.0e-12,
        )
    )
    assert penalty_value >= 0.0

    step_size = 1.0e-6
    perturbed_log_gain = persistent_log_gain.copy()
    perturbed_log_gain[0, 1] += step_size
    positive_log_gain_objective = _same_scene_correlation_penalty_and_gradients(
        observations_power=observations_power,
        persistent_log_gain=perturbed_log_gain,
        persistent_floor_parameter=persistent_floor_parameter,
        log_floor_power=1.0e-12,
    )[0]
    perturbed_log_gain[0, 1] -= 2.0 * step_size
    negative_log_gain_objective = _same_scene_correlation_penalty_and_gradients(
        observations_power=observations_power,
        persistent_log_gain=perturbed_log_gain,
        persistent_floor_parameter=persistent_floor_parameter,
        log_floor_power=1.0e-12,
    )[0]
    numerical_log_gain_gradient = (
        positive_log_gain_objective - negative_log_gain_objective
    ) / (2.0 * step_size)
    assert gradient_log_gain[0, 1] == pytest.approx(
        numerical_log_gain_gradient,
        rel=5.0e-4,
        abs=1.0e-5,
    )

    perturbed_floor_parameter = persistent_floor_parameter.copy()
    perturbed_floor_parameter[1, 2] += step_size
    positive_floor_objective = _same_scene_correlation_penalty_and_gradients(
        observations_power=observations_power,
        persistent_log_gain=persistent_log_gain,
        persistent_floor_parameter=perturbed_floor_parameter,
        log_floor_power=1.0e-12,
    )[0]
    perturbed_floor_parameter[1, 2] -= 2.0 * step_size
    negative_floor_objective = _same_scene_correlation_penalty_and_gradients(
        observations_power=observations_power,
        persistent_log_gain=persistent_log_gain,
        persistent_floor_parameter=perturbed_floor_parameter,
        log_floor_power=1.0e-12,
    )[0]
    numerical_floor_gradient = (
        positive_floor_objective - negative_floor_objective
    ) / (2.0 * step_size)
    assert gradient_floor_parameter[1, 2] == pytest.approx(
        numerical_floor_gradient,
        rel=5.0e-4,
        abs=1.0e-5,
    )


def test_large_consistency_weight_reduces_same_scene_dispersion() -> None:
    """A stronger consistency weight should reduce fitted same-scene spread.

    The synthetic fixture already approximately satisfies the model, so the
    improvement is modest. The important regression is directional: if the
    penalty path stops influencing the persistent deployment map, the
    persistent-only dispersion will stop decreasing when the weight is raised
    by orders of magnitude.
    """

    fixture = build_synthetic_two_level_fixture()
    no_penalty_result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=replace(fixture.fit_config, lambda_consistency=0.0),
    )
    strong_penalty_result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=replace(fixture.fit_config, lambda_consistency=500.0),
    )

    no_penalty_dispersion = _mean_persistent_only_same_scene_dispersion(
        result=no_penalty_result,
        fixture=fixture,
        log_floor_power=fixture.fit_config.consistency_log_floor_power,
    )
    strong_penalty_dispersion = _mean_persistent_only_same_scene_dispersion(
        result=strong_penalty_result,
        fixture=fixture,
        log_floor_power=fixture.fit_config.consistency_log_floor_power,
    )

    assert strong_penalty_dispersion < no_penalty_dispersion


def test_large_correlation_weight_increases_same_scene_pairwise_correlation() -> None:
    """A stronger correlation weight should improve fitted same-scene shape agreement."""

    fixture = build_synthetic_two_level_fixture()
    no_penalty_result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=replace(fixture.fit_config, lambda_correlation=0.0),
    )
    strong_penalty_result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=replace(fixture.fit_config, lambda_correlation=500.0),
    )

    no_penalty_correlation = _mean_persistent_only_same_scene_pairwise_correlation(
        result=no_penalty_result,
        fixture=fixture,
        log_floor_power=fixture.fit_config.consistency_log_floor_power,
    )
    strong_penalty_correlation = (
        _mean_persistent_only_same_scene_pairwise_correlation(
            result=strong_penalty_result,
            fixture=fixture,
            log_floor_power=fixture.fit_config.consistency_log_floor_power,
        )
    )

    assert strong_penalty_correlation > no_penalty_correlation


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
    assert curves.trust_diagnostics.overall_out_of_distribution is True
    assert "central_frequency_hz" in curves.trust_diagnostics.out_of_range_feature_names
    assert curves.trust_diagnostics.max_abs_standardized_feature == pytest.approx(
        max(abs(value) for value in curves.trust_diagnostics.standardized_configuration)
    )


def test_evaluate_persistent_calibration_flags_geometric_configuration_ood() -> None:
    """Deployment diagnostics should catch within-box but off-manifold inputs."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )
    geometric_ood_configuration = CampaignConfiguration(
        central_frequency_hz=98.0e6,
        span_hz=16.0e6,
        resolution_bandwidth_hz=20.0e3,
        lna_gain_db=0.0,
        vga_gain_db=60.0,
        acquisition_interval_s=180.0,
        antenna_amplifier_enabled=True,
    )

    with pytest.raises(ValueError, match="Mahalanobis distance"):
        evaluate_persistent_calibration(
            result=result,
            sensor_id=fixture.deployment_sensor_id,
            configuration=geometric_ood_configuration,
            frequency_hz=fixture.deployment_frequency_hz,
            allow_configuration_ood=False,
        )

    curves = evaluate_persistent_calibration(
        result=result,
        sensor_id=fixture.deployment_sensor_id,
        configuration=geometric_ood_configuration,
        frequency_hz=fixture.deployment_frequency_hz,
        allow_configuration_ood=True,
    )

    assert curves.trust_diagnostics.configuration_support_available is True
    assert curves.trust_diagnostics.configuration_out_of_distribution is False
    assert curves.trust_diagnostics.configuration_geometry_support_available is True
    assert curves.trust_diagnostics.configuration_geometric_out_of_distribution is True
    assert curves.trust_diagnostics.configuration_mahalanobis_distance is not None
    assert curves.trust_diagnostics.configuration_mahalanobis_threshold is not None
    assert (
        curves.trust_diagnostics.configuration_mahalanobis_tail_probability is not None
    )
    assert (
        curves.trust_diagnostics.configuration_mahalanobis_distance
        > curves.trust_diagnostics.configuration_mahalanobis_threshold
    )
    assert curves.trust_diagnostics.configuration_mahalanobis_tail_probability < 0.05
    assert curves.trust_diagnostics.overall_out_of_distribution is True


def test_configuration_geometry_support_falls_back_without_hermitian_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geometry support should tolerate backends whose ``pinv`` lacks ``hermitian``."""

    original_pinv = spectral_calibration_core.np.linalg.pinv

    def pinv_without_hermitian(*args: Any, **kwargs: Any) -> np.ndarray:
        if "hermitian" in kwargs:
            raise TypeError("pinv() got an unexpected keyword argument 'hermitian'")
        return original_pinv(*args, **kwargs)

    monkeypatch.setattr(
        spectral_calibration_core.np.linalg,
        "pinv",
        pinv_without_hermitian,
    )

    precision_matrix, threshold, effective_rank = _configuration_geometry_support(
        standardized_configuration_features=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, -0.5, 0.25],
                [-0.75, 0.5, -0.1],
            ],
            dtype=np.float64,
        )
    )

    assert precision_matrix.shape == (3, 3)
    assert np.all(np.isfinite(precision_matrix))
    assert threshold > 0.0
    assert effective_rank >= 1
