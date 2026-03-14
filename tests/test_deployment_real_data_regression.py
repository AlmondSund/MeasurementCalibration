"""Frozen real-data regression tests for deployment workflows.

These checks intentionally reuse the checked-in production artifact and the
checked-in campaign CSVs. They validate the deployment boundary without
retraining the model, which keeps the regression suite light enough for local
development while still exercising the real data path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from measurement_calibration import (
    DEFAULT_CAMPAIGNS_DATA_DIR,
    DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR,
    DEFAULT_PRODUCTION_ARTIFACT_DIR,
    calibrate_sensor_observations,
    evaluate_persistent_calibration,
    load_notebook_workflow_config,
    load_two_level_calibration_artifact,
    prepare_calibration_campaign,
    resolve_global_excluded_sensor_ids_by_campaign,
)


_REAL_DEPLOYMENT_REGRESSION_EXPECTATIONS = {
    "test-calibration": {
        "mean_gain_power": 1.069190787498796,
        "mean_gain_sensor_spread": 0.15165675058808825,
        "mean_floor_power": 2.3771144232745977e-08,
        "mean_variance_power": 2.3929261670524444,
        "mean_calibrated_power": 0.026014931445245526,
        "mean_calibrated_sensor_spread": 0.009673948019876123,
        "calibrated_power_p99": 0.4169910561568298,
    },
    "MeasurementCalibration": {
        "mean_gain_power": 1.0310154754960368,
        "mean_gain_sensor_spread": 0.2191542323899645,
        "mean_floor_power": 2.5304899120139988e-08,
        "mean_variance_power": 2.351468353741056,
        "mean_calibrated_power": 0.027065611326458218,
        "mean_calibrated_sensor_spread": 0.014087971101656872,
        "calibrated_power_p99": 0.43994379081154994,
    },
}


def _curve_roughness(values: np.ndarray) -> float:
    """Return the RMS second difference of a one-dimensional curve."""

    second_difference = np.diff(np.asarray(values, dtype=np.float64), n=2)
    if second_difference.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(second_difference**2)))


@pytest.mark.parametrize(
    ("deployment_campaign_label", "expected_metrics"),
    tuple(_REAL_DEPLOYMENT_REGRESSION_EXPECTATIONS.items()),
)
def test_checked_in_production_artifact_supports_real_deployment_campaigns(
    deployment_campaign_label: str,
    expected_metrics: dict[str, float],
) -> None:
    """The saved artifact should keep both real deployment campaigns in-support.

    This regression intentionally uses the checked-in production artifact and a
    frozen real campaign. It does not prove scientific optimality, but it
    does anchor the current deployment behavior to quantitative real-data
    expectations instead of only shape/finiteness checks.

    The repository ships a promoted artifact that is expected to remain within
    the stored configuration envelope for both deployment-like campaigns that
    share the operational VGA=62 configuration. If this stops being true, the
    checked-in production bundle is no longer aligned with the project
    requirement that deployment runs stay in-support.
    """

    repo_root = Path(__file__).resolve().parents[1]
    workflow_config = load_notebook_workflow_config(
        repo_root / DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR
    )
    excluded_sensor_ids_by_campaign = resolve_global_excluded_sensor_ids_by_campaign(
        campaign_labels=workflow_config.workflow_campaign_labels,
        excluded_sensor_ids=workflow_config.excluded_sensor_ids,
        campaigns_root=repo_root / DEFAULT_CAMPAIGNS_DATA_DIR,
    )
    prepared_campaign = prepare_calibration_campaign(
        campaign_label=deployment_campaign_label,
        campaigns_root=repo_root / DEFAULT_CAMPAIGNS_DATA_DIR,
        excluded_sensor_ids=excluded_sensor_ids_by_campaign[deployment_campaign_label],
        excluded_leading_measurements_per_sensor=(
            workflow_config.excluded_leading_measurements_per_sensor
        ),
    )
    artifact = load_two_level_calibration_artifact(
        repo_root / DEFAULT_PRODUCTION_ARTIFACT_DIR
    )

    campaign = prepared_campaign.campaign
    gain_curves: list[np.ndarray] = []
    floor_curves: list[np.ndarray] = []
    variance_curves: list[np.ndarray] = []
    calibrated_power_by_sensor: list[np.ndarray] = []

    for sensor_id in campaign.sensor_ids:
        sensor_index = campaign.sensor_ids.index(sensor_id)
        raw_power = campaign.observations_power[sensor_index]
        curves = evaluate_persistent_calibration(
            result=artifact.result,
            sensor_id=sensor_id,
            configuration=campaign.configuration,
            frequency_hz=campaign.frequency_hz,
        )
        deployment = calibrate_sensor_observations(
            result=artifact.result,
            sensor_id=sensor_id,
            configuration=campaign.configuration,
            frequency_hz=campaign.frequency_hz,
            observations_power=raw_power,
        )

        assert curves.sensor_id == sensor_id
        assert curves.trust_diagnostics.frequency_extrapolation_detected is False
        assert curves.trust_diagnostics.configuration_support_available is True
        assert curves.trust_diagnostics.configuration_out_of_distribution is False
        assert curves.trust_diagnostics.overall_out_of_distribution is False
        assert len(curves.trust_diagnostics.standardized_configuration) == 7
        assert curves.trust_diagnostics.configuration_geometry_support_available is True
        assert curves.trust_diagnostics.configuration_geometric_out_of_distribution is (
            False
        )
        assert curves.trust_diagnostics.out_of_range_feature_names == ()
        assert curves.trust_diagnostics.configuration_mahalanobis_distance is not None
        assert curves.trust_diagnostics.configuration_mahalanobis_threshold is not None
        assert (
            curves.trust_diagnostics.configuration_mahalanobis_distance
            < curves.trust_diagnostics.configuration_mahalanobis_threshold
        )
        assert deployment.calibrated_power.shape == raw_power.shape
        assert deployment.propagated_variance_power2.shape == raw_power.shape
        assert deployment.uncertainty_scope == "observation_noise_only"
        assert np.all(np.isfinite(curves.gain_power))
        assert np.all(np.isfinite(curves.additive_noise_power))
        assert np.all(np.isfinite(curves.residual_variance_power2))
        assert np.all(np.isfinite(deployment.calibrated_power))
        assert np.all(np.isfinite(deployment.propagated_variance_power2))
        assert np.all(deployment.propagated_variance_power2 > 0.0)
        assert not np.allclose(deployment.calibrated_power, raw_power)

        gain_curves.append(curves.gain_power)
        floor_curves.append(curves.additive_noise_power)
        variance_curves.append(curves.residual_variance_power2)
        calibrated_power_by_sensor.append(deployment.calibrated_power)

    gain_stack = np.stack(gain_curves, axis=0)
    calibrated_power_stack = np.stack(calibrated_power_by_sensor, axis=0)

    max_gain_roughness = max(_curve_roughness(curve) for curve in gain_curves)
    max_floor_roughness = max(_curve_roughness(curve) for curve in floor_curves)
    max_variance_roughness = max(_curve_roughness(curve) for curve in variance_curves)
    mean_gain_power = float(np.mean(gain_stack))
    mean_gain_sensor_spread = float(np.mean(np.std(gain_stack, axis=0)))
    mean_floor_power = float(np.mean(np.stack(floor_curves, axis=0)))
    mean_variance_power = float(np.mean(np.stack(variance_curves, axis=0)))
    mean_calibrated_power = float(np.mean(calibrated_power_stack))
    mean_calibrated_sensor_spread = float(
        np.mean(np.std(calibrated_power_stack, axis=0))
    )
    calibrated_power_p99 = float(np.quantile(calibrated_power_stack, 0.99))

    assert max_gain_roughness < 1.0e-4
    assert max_floor_roughness < 1.0e-10
    assert max_variance_roughness < 1.0e-4
    assert mean_gain_power == pytest.approx(
        expected_metrics["mean_gain_power"],
        rel=1.0e-3,
    )
    assert mean_gain_sensor_spread == pytest.approx(
        expected_metrics["mean_gain_sensor_spread"],
        rel=1.0e-3,
    )
    assert mean_floor_power == pytest.approx(
        expected_metrics["mean_floor_power"],
        rel=1.0e-3,
    )
    assert mean_variance_power == pytest.approx(
        expected_metrics["mean_variance_power"],
        rel=1.0e-3,
    )
    assert mean_calibrated_power == pytest.approx(
        expected_metrics["mean_calibrated_power"],
        rel=1.0e-3,
    )
    assert mean_calibrated_sensor_spread == pytest.approx(
        expected_metrics["mean_calibrated_sensor_spread"],
        rel=1.0e-3,
    )
    assert calibrated_power_p99 == pytest.approx(
        expected_metrics["calibrated_power_p99"],
        rel=1.0e-3,
    )


@pytest.mark.parametrize(
    "deployment_campaign_label",
    tuple(_REAL_DEPLOYMENT_REGRESSION_EXPECTATIONS),
)
def test_checked_in_production_artifact_accepts_real_deployment_campaigns_in_strict_mode(
    deployment_campaign_label: str,
) -> None:
    """Strict deployment mode should accept the supported real campaigns."""

    repo_root = Path(__file__).resolve().parents[1]
    workflow_config = load_notebook_workflow_config(
        repo_root / DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR
    )
    excluded_sensor_ids_by_campaign = resolve_global_excluded_sensor_ids_by_campaign(
        campaign_labels=workflow_config.workflow_campaign_labels,
        excluded_sensor_ids=workflow_config.excluded_sensor_ids,
        campaigns_root=repo_root / DEFAULT_CAMPAIGNS_DATA_DIR,
    )
    prepared_campaign = prepare_calibration_campaign(
        campaign_label=deployment_campaign_label,
        campaigns_root=repo_root / DEFAULT_CAMPAIGNS_DATA_DIR,
        excluded_sensor_ids=excluded_sensor_ids_by_campaign[deployment_campaign_label],
        excluded_leading_measurements_per_sensor=(
            workflow_config.excluded_leading_measurements_per_sensor
        ),
    )
    artifact = load_two_level_calibration_artifact(
        repo_root / DEFAULT_PRODUCTION_ARTIFACT_DIR
    )

    curves = evaluate_persistent_calibration(
        result=artifact.result,
        sensor_id=prepared_campaign.campaign.sensor_ids[0],
        configuration=prepared_campaign.campaign.configuration,
        frequency_hz=prepared_campaign.campaign.frequency_hz,
        allow_configuration_ood=False,
    )

    assert curves.trust_diagnostics.overall_out_of_distribution is False
