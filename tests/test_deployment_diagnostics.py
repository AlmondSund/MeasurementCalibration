"""Tests for campaign-level deployment diagnostics used by notebook overlays."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from measurement_calibration import (
    CalibrationCampaign,
    CampaignConfiguration,
    DeploymentCalibrationResult,
    PersistentCalibrationCurves,
    build_cross_node_campaign_animation_data,
    format_cross_node_overlay_title,
    power_db_to_linear,
    resolve_cross_node_overlay_limits_db,
)


def _build_synthetic_campaign() -> CalibrationCampaign:
    """Build a small deterministic same-scene campaign for overlay tests."""

    observations_db = np.asarray(
        [
            [[0.0, 2.0], [1.0, 3.0]],
            [[3.0, 5.0], [4.0, 6.0]],
            [[6.0, 8.0], [7.0, 9.0]],
        ],
        dtype=np.float64,
    )
    return CalibrationCampaign(
        campaign_label="synthetic-campaign",
        sensor_ids=("Node1", "Node2", "Node3"),
        frequency_hz=np.asarray([100.0, 200.0], dtype=np.float64),
        observations_power=power_db_to_linear(observations_db),
        configuration=CampaignConfiguration(
            central_frequency_hz=100.0,
            span_hz=50.0,
            resolution_bandwidth_hz=1.0,
            lna_gain_db=0.0,
            vga_gain_db=0.0,
            acquisition_interval_s=1.0,
            antenna_amplifier_enabled=False,
        ),
        reliable_sensor_id="Node1",
    )


def test_build_cross_node_campaign_animation_data_tracks_framewise_rmse() -> None:
    """The helper should preserve sensor order and compute record-level RMSE."""

    campaign = _build_synthetic_campaign()
    calibrated_db_by_sensor = {
        "Node1": np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
        "Node2": np.asarray([[2.0, 3.0], [3.0, 4.0]], dtype=np.float64),
        "Node3": np.asarray([[3.0, 4.0], [4.0, 5.0]], dtype=np.float64),
    }

    def fake_calibrate_sensor(
        sensor_id: str,
        observations_power: np.ndarray,
    ) -> DeploymentCalibrationResult:
        """Return a deterministic calibrated tensor for one synthetic sensor."""

        del observations_power
        return DeploymentCalibrationResult(
            curves=cast(PersistentCalibrationCurves, None),
            calibrated_power=power_db_to_linear(calibrated_db_by_sensor[sensor_id]),
            propagated_variance_power2=np.ones((2, 2), dtype=np.float64),
            uncertainty_scope="observation_noise_only",
        )

    animation_data = build_cross_node_campaign_animation_data(
        campaign=campaign,
        calibrate_sensor=fake_calibrate_sensor,
    )

    assert animation_data.campaign_label == "synthetic-campaign"
    assert animation_data.sensor_ids == ("Node1", "Node2", "Node3")
    assert animation_data.raw_power_db.shape == (3, 2, 2)
    assert animation_data.calibrated_power_db.shape == (3, 2, 2)
    assert animation_data.n_sensors == 3
    assert animation_data.n_records == 2
    assert np.allclose(
        animation_data.calibrated_power_db[1],
        calibrated_db_by_sensor["Node2"],
    )
    assert animation_data.record_alignments[0].mean_pairwise_raw_rmse_db == (
        pytest.approx(4.0)
    )
    assert animation_data.record_alignments[1].mean_pairwise_raw_rmse_db == (
        pytest.approx(4.0)
    )
    assert animation_data.record_alignments[0].mean_pairwise_calibrated_rmse_db == (
        pytest.approx(4.0 / 3.0)
    )
    assert animation_data.record_alignments[1].mean_pairwise_calibrated_rmse_db == (
        pytest.approx(4.0 / 3.0)
    )
    assert format_cross_node_overlay_title(animation_data, 1) == (
        "Cross-Node PSD Overlays | synthetic-campaign | record 2/2 | "
        "mean pairwise RMSE raw=4.00 dB, cal=1.33 dB"
    )
    assert resolve_cross_node_overlay_limits_db(
        animation_data,
        padding_db=0.5,
    ) == pytest.approx((-0.5, 9.5))


def test_build_cross_node_campaign_animation_data_rejects_shape_changes() -> None:
    """The injected calibrator must preserve the per-sensor observation shape."""

    campaign = _build_synthetic_campaign()

    def wrong_shape_calibrator(
        sensor_id: str,
        observations_power: np.ndarray,
    ) -> DeploymentCalibrationResult:
        """Return an invalid tensor shape to exercise the validation path."""

        del sensor_id, observations_power
        return DeploymentCalibrationResult(
            curves=cast(PersistentCalibrationCurves, None),
            calibrated_power=np.ones((1, 2), dtype=np.float64),
            propagated_variance_power2=np.ones((1, 2), dtype=np.float64),
            uncertainty_scope="observation_noise_only",
        )

    with pytest.raises(
        ValueError,
        match="calibrate_sensor must preserve the per-sensor observation shape",
    ):
        build_cross_node_campaign_animation_data(
            campaign=campaign,
            calibrate_sensor=wrong_shape_calibrator,
        )


def test_format_cross_node_overlay_title_rejects_out_of_range_records() -> None:
    """Title formatting should fail fast on invalid record indices."""

    campaign = _build_synthetic_campaign()

    def identity_calibrator(
        sensor_id: str,
        observations_power: np.ndarray,
    ) -> DeploymentCalibrationResult:
        """Return the unmodified observations for bounds-check testing."""

        del sensor_id
        return DeploymentCalibrationResult(
            curves=cast(PersistentCalibrationCurves, None),
            calibrated_power=observations_power,
            propagated_variance_power2=np.ones_like(observations_power),
            uncertainty_scope="observation_noise_only",
        )

    animation_data = build_cross_node_campaign_animation_data(
        campaign=campaign,
        calibrate_sensor=identity_calibrator,
    )

    with pytest.raises(IndexError, match="record_index must be in \\[0, 2\\)"):
        format_cross_node_overlay_title(animation_data, 2)
