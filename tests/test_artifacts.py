"""Tests for calibration artifact serialization and reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from measurement_calibration.artifacts import (
    load_spectral_calibration_artifact,
    save_spectral_calibration_artifact,
)
from measurement_calibration.spectral_calibration import (
    CalibrationDataset,
    fit_spectral_calibration,
)


def test_save_and_load_spectral_calibration_artifact_round_trip(
    tmp_path: Path,
) -> None:
    """Saved artifacts should reload into the same fitted calibration result."""

    dataset = _synthetic_dataset()
    train_indices = np.arange(0, 4, dtype=np.int64)
    test_indices = np.arange(4, dataset.observations_power.shape[1], dtype=np.int64)
    fit_config = {
        "n_iterations": 4,
        "lambda_gain_smooth": 12.0,
        "lambda_noise_smooth": 10.0,
        "lambda_gain_reference": 3.0,
        "lambda_noise_reference": 20.0,
        "lambda_reliable_anchor": 1.0,
        "reliable_weight_boost": 1.05,
        "low_information_threshold_ratio": 0.10,
        "low_information_weight": 0.05,
    }

    result = fit_spectral_calibration(
        observations_power=dataset.observations_power,
        frequency_hz=dataset.frequency_hz,
        sensor_ids=dataset.sensor_ids,
        nominal_gain_power=dataset.nominal_gain_power,
        train_indices=train_indices,
        test_indices=test_indices,
        reliable_sensor_id="Node1",
        n_iterations=int(fit_config["n_iterations"]),
        lambda_gain_smooth=float(fit_config["lambda_gain_smooth"]),
        lambda_noise_smooth=float(fit_config["lambda_noise_smooth"]),
        lambda_gain_reference=float(fit_config["lambda_gain_reference"]),
        lambda_noise_reference=float(fit_config["lambda_noise_reference"]),
        lambda_reliable_anchor=float(fit_config["lambda_reliable_anchor"]),
        reliable_weight_boost=float(fit_config["reliable_weight_boost"]),
        low_information_threshold_ratio=float(
            fit_config["low_information_threshold_ratio"]
        ),
        low_information_weight=float(fit_config["low_information_weight"]),
    )

    artifact = save_spectral_calibration_artifact(
        output_dir=tmp_path / "artifact",
        result=result,
        dataset=dataset,
        acquisition_dir=tmp_path / "acquisitions",
        response_dir=tmp_path / "responses",
        reference_sensor_id="Node1",
        reliable_sensor_id="Node1",
        excluded_sensor_ids=(),
        fit_config=fit_config,
        extra_summary={"fit_duration_s": 0.25, "test_fraction": 1.0 / 3.0},
    )
    loaded = load_spectral_calibration_artifact(artifact.output_dir)

    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["dataset"]["sensor_ids"] == list(dataset.sensor_ids)
    assert loaded.manifest["result_summary"]["train_experiments"] == 4
    assert loaded.manifest["result_summary"]["test_experiments"] == 2
    assert loaded.manifest["extra_summary"]["fit_duration_s"] == 0.25

    assert loaded.result.sensor_ids == result.sensor_ids
    assert np.allclose(loaded.result.frequency_hz, result.frequency_hz)
    assert np.allclose(loaded.result.gain_power, result.gain_power)
    assert np.allclose(
        loaded.result.additive_noise_power,
        result.additive_noise_power,
    )
    assert np.allclose(
        loaded.result.residual_variance_power2,
        result.residual_variance_power2,
    )
    assert np.allclose(
        loaded.result.latent_spectra_power,
        result.latent_spectra_power,
    )
    assert np.allclose(loaded.result.nominal_gain_power, result.nominal_gain_power)
    assert np.allclose(
        loaded.result.correction_gain_power,
        result.correction_gain_power,
    )
    assert np.array_equal(loaded.result.train_indices, result.train_indices)
    assert np.array_equal(loaded.result.test_indices, result.test_indices)
    assert np.allclose(
        loaded.result.objective_history,
        result.objective_history,
    )
    assert np.allclose(
        loaded.result.latent_variation_power2,
        result.latent_variation_power2,
    )
    assert np.allclose(
        loaded.result.frequency_information_weight,
        result.frequency_information_weight,
    )
    assert np.allclose(loaded.result.information_weight, result.information_weight)
    assert np.array_equal(
        loaded.result.frequency_low_information_mask,
        result.frequency_low_information_mask,
    )
    assert np.array_equal(
        loaded.result.low_information_mask,
        result.low_information_mask,
    )
    assert np.array_equal(
        loaded.result.gain_at_correction_bound_mask,
        result.gain_at_correction_bound_mask,
    )
    assert np.array_equal(loaded.result.noise_zero_mask, result.noise_zero_mask)

    with loaded.sensor_summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == len(dataset.sensor_ids)
    assert rows[0]["sensor_id"] == "Node1"
    assert "median_total_gain_db" in rows[0]
    assert "low_information_fraction" in rows[0]


def _synthetic_dataset() -> CalibrationDataset:
    """Build a deterministic synthetic dataset for artifact serialization tests."""

    rng = np.random.default_rng(7)
    sensor_ids = ("Node1", "Node2", "Node3")
    n_sensors = len(sensor_ids)
    n_experiments = 6
    n_frequencies = 12

    frequency_hz = np.linspace(88.0e6, 108.0e6, n_frequencies)
    frequency_phase = np.linspace(0.0, 1.0, n_frequencies)
    nominal_gain_power = np.ones((n_sensors, n_frequencies), dtype=np.float64)

    true_log_gain = np.vstack(
        [
            np.zeros(n_frequencies),
            0.08 * np.sin(2.0 * np.pi * frequency_phase),
            -0.05 * np.cos(2.0 * np.pi * frequency_phase),
        ]
    )
    true_log_gain -= np.mean(true_log_gain, axis=0, keepdims=True)
    true_gain = nominal_gain_power * np.exp(true_log_gain)

    additive_noise_power = np.asarray(
        [
            np.linspace(0.010, 0.015, n_frequencies),
            np.linspace(0.012, 0.017, n_frequencies),
            np.linspace(0.011, 0.016, n_frequencies),
        ],
        dtype=np.float64,
    )
    latent_spectra_power = np.vstack(
        [
            0.8
            + 0.10 * np.sin(2.0 * np.pi * frequency_phase * (1.0 + experiment / 4.0))
            + 0.03 * experiment
            for experiment in range(n_experiments)
        ]
    )
    observations_power = (
        true_gain[:, np.newaxis, :] * latent_spectra_power[np.newaxis, :, :]
        + additive_noise_power[:, np.newaxis, :]
        + 0.004 * rng.normal(size=(n_sensors, n_experiments, n_frequencies))
    )
    observations_power = np.clip(observations_power, 1.0e-6, None)

    return CalibrationDataset(
        sensor_ids=sensor_ids,
        frequency_hz=frequency_hz,
        observations_power=observations_power,
        nominal_gain_power=nominal_gain_power,
        experiment_timestamps_ms=np.arange(n_experiments, dtype=np.int64) * 1_000,
        selected_band_hz=(88.0e6, 108.0e6),
        sensor_shifts={sensor_id: 0 for sensor_id in sensor_ids},
        alignment_median_error_ms={sensor_id: 0.0 for sensor_id in sensor_ids},
        source_row_indices={
            sensor_id: np.arange(n_experiments, dtype=np.int64)
            for sensor_id in sensor_ids
        },
    )
