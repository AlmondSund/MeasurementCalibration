"""Tests for the RBW calibration adapter and training workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from measurement_calibration.artifacts import load_spectral_calibration_artifact
from measurement_calibration.rbw_calibration import (
    RbwCalibrationValidationSummary,
    RbwSensorValidationSummary,
    _select_reliable_sensor_id,
    _select_rbw_qc_retrain_sensor_ids,
    evaluate_rbw_calibration_holdout,
    fit_and_save_rbw_calibration_model,
    identify_rbw_qc_outlier_sensor_ids,
    prepare_rbw_calibration_dataset,
)
from measurement_calibration.sensor_ranking import (
    RbwAcquisitionDataset,
    SensorRankingResult,
)
from measurement_calibration.spectral_calibration import (
    SpectralCalibrationResult,
    compute_network_consensus,
    power_db_to_linear,
    power_linear_to_db,
)


def test_prepare_rbw_calibration_dataset_excludes_node9_and_converts_to_linear() -> (
    None
):
    """RBW preparation should drop Node9 and build a valid calibration dataset."""

    dataset = _synthetic_rbw_dataset()

    preparation = prepare_rbw_calibration_dataset(
        dataset,
        excluded_sensor_ids_by_label={},
    )
    calibration_dataset = preparation.calibration_dataset

    assert calibration_dataset.sensor_ids == ("Node1", "Node3", "Node4", "Node5")
    assert preparation.excluded_sensor_ids == ("Node9",)
    assert calibration_dataset.observations_power.shape == (4, 5, 8)
    assert np.allclose(
        calibration_dataset.observations_power,
        power_db_to_linear(dataset.observations_db[:4]),
    )
    assert np.allclose(calibration_dataset.nominal_gain_power, 1.0)
    assert np.array_equal(
        calibration_dataset.experiment_timestamps_ms,
        np.median(dataset.timestamps_ms[:4], axis=0).astype(np.int64),
    )
    assert calibration_dataset.selected_band_hz == (88.0e6, 108.0e6)
    assert all(
        calibration_dataset.sensor_shifts[sensor_id] == 0
        for sensor_id in calibration_dataset.sensor_ids
    )
    assert all(
        np.array_equal(
            calibration_dataset.source_row_indices[sensor_id],
            np.arange(dataset.n_records, dtype=np.int64),
        )
        for sensor_id in calibration_dataset.sensor_ids
    )


def test_prepare_rbw_calibration_dataset_uses_first_non_outlier_recordwise_sensor() -> (
    None
):
    """RBW preparation should keep the first record-wise non-outlier as anchor."""

    preparation = prepare_rbw_calibration_dataset(
        _synthetic_rbw_dataset(),
        excluded_sensor_ids_by_label={},
    )
    expected_reliable_sensor_id = next(
        sensor_id
        for sensor_id in preparation.ranking_result.ranking_sensor_ids
        if sensor_id not in preparation.distribution_outlier_sensor_ids
    )

    assert "Node4" in preparation.distribution_outlier_sensor_ids
    assert preparation.reliable_sensor_id == expected_reliable_sensor_id


def test_prepare_rbw_calibration_dataset_applies_default_per_rbw_exclusions() -> None:
    """RBW preparation should apply the repository default per-RBW exclusions."""

    preparation = prepare_rbw_calibration_dataset(_synthetic_rbw_dataset())

    assert preparation.calibration_dataset.sensor_ids == ("Node1", "Node3", "Node4")
    assert preparation.excluded_sensor_ids == ("Node5", "Node9")


def test_select_reliable_sensor_id_skips_distribution_outlier_recordwise_winner() -> (
    None
):
    """The anchor-selection helper should skip a ranked winner flagged as outlier."""

    ranking_result = SensorRankingResult(
        rbw_label="10K",
        sensor_ids=("Node1", "Node3", "Node4", "Node5"),
        per_record_score=np.zeros((4, 3), dtype=np.float64),
        average_score=np.asarray([2.8, 2.9, 3.0, 2.6], dtype=np.float64),
        average_correlation=np.asarray(
            [0.93333333, 0.96666667, 1.0, 0.86666667],
            dtype=np.float64,
        ),
        noise_floor_db=np.zeros((4, 3), dtype=np.float64),
        global_noise_floor_db=np.zeros(3, dtype=np.float64),
        per_record_correlation=np.repeat(
            np.eye(4, dtype=np.float64)[np.newaxis, :, :],
            3,
            axis=0,
        ),
        ranking_sensor_ids=("Node4", "Node3", "Node1", "Node5"),
    )

    reliable_sensor_id = _select_reliable_sensor_id(
        ranking_result=ranking_result,
        distribution_outlier_sensor_ids=("Node4",),
    )

    assert reliable_sensor_id == "Node3"


def test_fit_and_save_rbw_calibration_model_writes_manifest_without_response_dir(
    tmp_path: Path,
) -> None:
    """RBW artifact persistence should not require a nominal-response directory."""

    preparation = prepare_rbw_calibration_dataset(
        _synthetic_rbw_dataset(),
        excluded_sensor_ids_by_label={},
    )
    fit_result = fit_and_save_rbw_calibration_model(
        preparation=preparation,
        output_dir=tmp_path / "rbw-artifact",
        acquisition_dir=tmp_path / "RBW_acquisitions" / preparation.rbw_label,
        fit_config={
            "n_iterations": 3,
            "lambda_gain_smooth": 10.0,
            "lambda_noise_smooth": 10.0,
            "lambda_gain_reference": 3.0,
            "lambda_noise_reference": 20.0,
            "lambda_reliable_anchor": 1.0,
            "reliable_weight_boost": 1.05,
            "low_information_threshold_ratio": 0.10,
            "low_information_weight": 0.05,
        },
        test_fraction=0.4,
        auto_retrain_after_qc=False,
    )
    loaded = load_spectral_calibration_artifact(fit_result.artifact.output_dir)

    assert loaded.manifest["response_dir"] is None
    assert loaded.manifest["reference_sensor_id"] == preparation.reliable_sensor_id
    assert loaded.manifest["reliable_sensor_id"] == preparation.reliable_sensor_id
    assert loaded.manifest["excluded_sensor_ids"] == ["Node9"]
    assert loaded.manifest["dataset"]["sensor_ids"] == list(
        preparation.calibration_dataset.sensor_ids
    )
    assert loaded.manifest["fit_config"]["max_correction_db"] == 12.0
    assert loaded.manifest["fit_config"]["min_variance"] == 1.0e-14
    assert loaded.manifest["extra_summary"]["corrected_to_raw_dispersion_ratio"] > 0.0
    assert np.isfinite(fit_result.corrected_to_raw_dispersion_ratio)
    assert fit_result.corrected_mean_sensor_std_db > 0.0
    assert fit_result.raw_mean_sensor_std_db > 0.0


def test_evaluate_rbw_calibration_holdout_uses_leave_one_out_consensus() -> None:
    """Per-sensor RBW QC should compare against leave-one-sensor-out consensus."""

    preparation = prepare_rbw_calibration_dataset(
        _synthetic_rbw_dataset(),
        excluded_sensor_ids_by_label={},
    )
    dataset = preparation.calibration_dataset
    train_indices = np.asarray([0, 1], dtype=np.int64)
    test_indices = np.asarray([2, 3, 4], dtype=np.int64)
    n_sensors = len(dataset.sensor_ids)
    n_frequencies = dataset.frequency_hz.size

    result = SpectralCalibrationResult(
        sensor_ids=dataset.sensor_ids,
        frequency_hz=dataset.frequency_hz,
        gain_power=np.ones((n_sensors, n_frequencies), dtype=np.float64),
        additive_noise_power=np.zeros((n_sensors, n_frequencies), dtype=np.float64),
        residual_variance_power2=np.ones((n_sensors, n_frequencies), dtype=np.float64),
        latent_spectra_power=np.mean(
            dataset.observations_power[:, train_indices, :],
            axis=0,
        ),
        nominal_gain_power=np.ones((n_sensors, n_frequencies), dtype=np.float64),
        correction_gain_power=np.ones((n_sensors, n_frequencies), dtype=np.float64),
        train_indices=train_indices,
        test_indices=test_indices,
        objective_history=np.asarray([1.0], dtype=np.float64),
        latent_variation_power2=np.var(
            dataset.observations_power[:, train_indices, :],
            axis=1,
        ).mean(axis=0),
        frequency_information_weight=np.ones(n_frequencies, dtype=np.float64),
        information_weight=np.ones((n_sensors, n_frequencies), dtype=np.float64),
        frequency_low_information_mask=np.zeros(n_frequencies, dtype=bool),
        low_information_mask=np.zeros((n_sensors, n_frequencies), dtype=bool),
        gain_at_correction_bound_mask=np.zeros((n_sensors, n_frequencies), dtype=bool),
        noise_zero_mask=np.zeros((n_sensors, n_frequencies), dtype=bool),
        solver_nonfinite_step_count=np.zeros(n_sensors, dtype=np.int64),
    )

    validation = evaluate_rbw_calibration_holdout(preparation, result)
    raw_test_power = dataset.observations_power[:, test_indices, :]
    for sensor_index, row in enumerate(validation.sensor_rows):
        other_sensor_mask = np.arange(n_sensors) != sensor_index
        leave_one_out_consensus_power = compute_network_consensus(
            corrected_power=raw_test_power[other_sensor_mask],
            residual_variance_power2=result.residual_variance_power2[other_sensor_mask],
        )
        expected_residual_db = power_linear_to_db(
            raw_test_power[sensor_index]
        ) - power_linear_to_db(leave_one_out_consensus_power)
        assert row.rmse_to_consensus_db == pytest.approx(
            float(np.sqrt(np.mean(expected_residual_db**2)))
        )
        assert row.mean_bias_to_consensus_db == pytest.approx(
            float(np.mean(expected_residual_db))
        )


def test_identify_rbw_qc_outlier_sensor_ids_flags_invalid_or_high_rmse_sensors() -> (
    None
):
    """RBW QC should flag non-anchor sensors with invalid or outlying behavior."""

    validation = RbwCalibrationValidationSummary(
        raw_mean_sensor_std_db=1.8,
        corrected_mean_sensor_std_db=0.7,
        corrected_to_raw_dispersion_ratio=0.39,
        sensor_rows=(
            RbwSensorValidationSummary(
                sensor_id="Node1",
                mean_bias_to_consensus_db=0.0,
                rmse_to_consensus_db=0.25,
                invalid_corrected_fraction=0.0,
                median_information_weight=0.95,
                low_information_fraction=0.01,
                gain_cap_fraction=0.0,
                alignment_median_error_ms=25.0,
            ),
            RbwSensorValidationSummary(
                sensor_id="Node3",
                mean_bias_to_consensus_db=0.1,
                rmse_to_consensus_db=0.45,
                invalid_corrected_fraction=0.0,
                median_information_weight=0.83,
                low_information_fraction=0.02,
                gain_cap_fraction=0.0,
                alignment_median_error_ms=40.0,
            ),
            RbwSensorValidationSummary(
                sensor_id="Node5",
                mean_bias_to_consensus_db=-0.7,
                rmse_to_consensus_db=2.9,
                invalid_corrected_fraction=2.0e-3,
                median_information_weight=0.44,
                low_information_fraction=0.32,
                gain_cap_fraction=0.12,
                alignment_median_error_ms=2800.0,
            ),
        ),
    )

    flagged_sensor_ids = identify_rbw_qc_outlier_sensor_ids(
        validation=validation,
        reliable_sensor_id="Node1",
    )

    assert flagged_sensor_ids == ("Node5",)


def test_select_rbw_qc_retrain_sensor_ids_keeps_only_worst_candidate() -> None:
    """RBW QC retraining should remove the single worst sensor per pass."""

    validation = RbwCalibrationValidationSummary(
        raw_mean_sensor_std_db=1.8,
        corrected_mean_sensor_std_db=0.7,
        corrected_to_raw_dispersion_ratio=0.39,
        sensor_rows=(
            RbwSensorValidationSummary(
                sensor_id="Node1",
                mean_bias_to_consensus_db=0.0,
                rmse_to_consensus_db=0.25,
                invalid_corrected_fraction=0.0,
                median_information_weight=0.95,
                low_information_fraction=0.01,
                gain_cap_fraction=0.0,
                alignment_median_error_ms=25.0,
            ),
            RbwSensorValidationSummary(
                sensor_id="Node3",
                mean_bias_to_consensus_db=-0.3,
                rmse_to_consensus_db=1.6,
                invalid_corrected_fraction=0.003,
                median_information_weight=0.60,
                low_information_fraction=0.20,
                gain_cap_fraction=0.02,
                alignment_median_error_ms=900.0,
            ),
            RbwSensorValidationSummary(
                sensor_id="Node5",
                mean_bias_to_consensus_db=-0.7,
                rmse_to_consensus_db=2.9,
                invalid_corrected_fraction=0.010,
                median_information_weight=0.44,
                low_information_fraction=0.32,
                gain_cap_fraction=0.12,
                alignment_median_error_ms=2800.0,
            ),
        ),
    )

    selected_sensor_ids = _select_rbw_qc_retrain_sensor_ids(
        validation=validation,
        candidate_sensor_ids=("Node3", "Node5"),
        retained_sensor_ids=("Node1", "Node3", "Node5"),
    )

    assert selected_sensor_ids == ("Node5",)
    assert (
        _select_rbw_qc_retrain_sensor_ids(
            validation=validation,
            candidate_sensor_ids=("Node3",),
            retained_sensor_ids=("Node1", "Node3"),
        )
        == ()
    )


def _synthetic_rbw_dataset() -> RbwAcquisitionDataset:
    """Build a deterministic RBW dataset with a shifted-histogram outlier."""

    base_records = np.asarray(
        [
            [-63.0, -61.0, -58.0, -54.0, -50.0, -49.0, -52.0, -57.0],
            [-62.5, -60.3, -57.1, -53.4, -49.7, -48.8, -51.6, -56.3],
            [-64.1, -61.7, -58.9, -55.1, -51.0, -50.2, -53.4, -58.0],
            [-63.6, -61.4, -58.3, -54.7, -50.5, -49.6, -52.8, -57.5],
            [-62.8, -60.7, -57.8, -53.8, -49.4, -48.6, -51.9, -56.8],
        ],
        dtype=np.float64,
    )
    observations_db = np.stack(
        [
            base_records
            + np.asarray(
                [
                    [0.15, -0.10, 0.08, -0.05, 0.03, -0.06, 0.05, -0.02],
                    [0.05, -0.04, 0.09, -0.02, 0.02, -0.03, 0.03, 0.01],
                    [0.12, -0.07, 0.04, -0.05, 0.04, -0.05, 0.06, -0.03],
                    [0.10, -0.08, 0.06, -0.03, 0.03, -0.03, 0.04, -0.01],
                    [0.09, -0.06, 0.08, -0.04, 0.03, -0.04, 0.05, -0.02],
                ],
                dtype=np.float64,
            ),
            base_records
            + np.asarray(
                [
                    [0.22, -0.18, 0.11, -0.09, 0.05, -0.09, 0.08, -0.05],
                    [0.18, -0.15, 0.14, -0.08, 0.04, -0.08, 0.07, -0.03],
                    [0.24, -0.20, 0.12, -0.10, 0.05, -0.10, 0.09, -0.05],
                    [0.20, -0.17, 0.13, -0.08, 0.04, -0.08, 0.07, -0.04],
                    [0.19, -0.15, 0.14, -0.09, 0.05, -0.09, 0.08, -0.04],
                ],
                dtype=np.float64,
            ),
            base_records + 8.0,
            base_records
            + np.asarray(
                [
                    [0.40, -0.35, 0.20, -0.15, 0.09, -0.18, 0.12, -0.08],
                    [0.32, -0.28, 0.23, -0.14, 0.08, -0.16, 0.11, -0.06],
                    [0.42, -0.38, 0.21, -0.17, 0.10, -0.19, 0.13, -0.08],
                    [0.36, -0.31, 0.22, -0.15, 0.09, -0.17, 0.12, -0.07],
                    [0.34, -0.30, 0.22, -0.16, 0.09, -0.17, 0.12, -0.07],
                ],
                dtype=np.float64,
            ),
            np.flip(base_records, axis=1) + 12.0,
        ],
        axis=0,
    )
    timestamps_ms = np.asarray(
        [
            [1_000, 2_000, 3_000, 4_000, 5_000],
            [1_040, 2_040, 3_040, 4_040, 5_040],
            [980, 1_980, 2_980, 3_980, 4_980],
            [1_020, 2_020, 3_020, 4_020, 5_020],
            [1_060, 2_060, 3_060, 4_060, 5_060],
        ],
        dtype=np.int64,
    )
    return RbwAcquisitionDataset(
        rbw_label="10K",
        sensor_ids=("Node1", "Node3", "Node4", "Node5", "Node9"),
        frequency_hz=np.linspace(88.0e6, 108.0e6, observations_db.shape[2]),
        observations_db=observations_db,
        timestamps_ms=timestamps_ms,
    )
