"""Frozen real-data regression tests for deployment workflows.

These checks intentionally reuse the checked-in production artifact and the
checked-in campaign CSVs. They validate the deployment boundary without
retraining the model, which keeps the regression suite light enough for local
development while still exercising the real data path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_checked_in_production_artifact_calibrates_real_deployment_campaign() -> None:
    """The saved production artifact should calibrate the held-out campaign."""

    repo_root = Path(__file__).resolve().parents[1]
    workflow_config = load_notebook_workflow_config(
        repo_root / DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR
    )
    excluded_sensor_ids_by_campaign = resolve_global_excluded_sensor_ids_by_campaign(
        campaign_labels=workflow_config.workflow_campaign_labels,
        excluded_sensor_ids=workflow_config.excluded_sensor_ids,
        campaigns_root=repo_root / DEFAULT_CAMPAIGNS_DATA_DIR,
    )
    deployment_campaign_label = workflow_config.testing_campaign_labels[0]
    prepared_campaign = prepare_calibration_campaign(
        campaign_label=deployment_campaign_label,
        campaigns_root=repo_root / DEFAULT_CAMPAIGNS_DATA_DIR,
        excluded_sensor_ids=excluded_sensor_ids_by_campaign[deployment_campaign_label],
    )
    artifact = load_two_level_calibration_artifact(
        repo_root / DEFAULT_PRODUCTION_ARTIFACT_DIR
    )

    sensor_id = (
        "Node3"
        if "Node3" in prepared_campaign.campaign.sensor_ids
        else prepared_campaign.reliable_sensor_id
    )
    sensor_index = prepared_campaign.campaign.sensor_ids.index(sensor_id)
    raw_power = prepared_campaign.campaign.observations_power[sensor_index]

    curves = evaluate_persistent_calibration(
        result=artifact.result,
        sensor_id=sensor_id,
        configuration=prepared_campaign.campaign.configuration,
        frequency_hz=prepared_campaign.campaign.frequency_hz,
    )
    deployment = calibrate_sensor_observations(
        result=artifact.result,
        sensor_id=sensor_id,
        configuration=prepared_campaign.campaign.configuration,
        frequency_hz=prepared_campaign.campaign.frequency_hz,
        observations_power=raw_power,
    )

    assert curves.sensor_id == sensor_id
    assert curves.trust_diagnostics.frequency_extrapolation_detected is False
    assert curves.trust_diagnostics.configuration_support_available is True
    assert curves.trust_diagnostics.configuration_out_of_distribution is False
    assert len(curves.trust_diagnostics.standardized_configuration) == 7
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
