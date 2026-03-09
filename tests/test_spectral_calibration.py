"""Tests for the spectral calibration framework."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from measurement_calibration.spectral_calibration import (
    apply_deployed_calibration,
    compute_network_consensus,
    fit_spectral_calibration,
    load_calibration_dataset,
    power_db_to_linear,
    power_linear_to_db,
)


def test_load_calibration_dataset_aligns_common_band(tmp_path: Path) -> None:
    """The loader should retain only the shared band and align shifted sensors."""

    acquisition_dir = tmp_path / "aquisitions"
    response_dir = tmp_path / "responses"
    acquisition_dir.mkdir()
    response_dir.mkdir()

    sensor_a_rows = [
        _acquisition_row(
            timestamp_ms=1_000,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[0.0, 1.0, 2.0, 3.0],
        ),
        _acquisition_row(
            timestamp_ms=2_000,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[1.0, 2.0, 3.0, 4.0],
        ),
        _acquisition_row(
            timestamp_ms=3_000,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[2.0, 3.0, 4.0, 5.0],
        ),
    ]
    sensor_b_rows = [
        _acquisition_row(
            timestamp_ms=100,
            start_hz=200.0,
            end_hz=204.0,
            power_db=[10.0, 10.0, 10.0, 10.0],
        ),
        _acquisition_row(
            timestamp_ms=2_010,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[0.5, 1.5, 2.5, 3.5],
        ),
        _acquisition_row(
            timestamp_ms=3_010,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[1.5, 2.5, 3.5, 4.5],
        ),
    ]

    _write_acquisition_csv(acquisition_dir / "Node1-Bogota.csv", sensor_a_rows)
    _write_acquisition_csv(acquisition_dir / "Node2-Bogota.csv", sensor_b_rows)
    _write_response_csv(response_dir / "Node1-Bogota-response.csv")
    _write_response_csv(response_dir / "Node2-Bogota-response.csv")

    dataset = load_calibration_dataset(
        acquisition_dir=acquisition_dir,
        response_dir=response_dir,
        reference_sensor_id="Node1-Bogota",
        max_alignment_shift=2,
    )

    assert dataset.selected_band_hz == (100.0, 104.0)
    assert dataset.observations_power.shape == (2, 2, 4)
    assert dataset.sensor_shifts["Node2-Bogota"] == -1
    assert np.array_equal(dataset.source_row_indices["Node1-Bogota"], np.array([1, 2]))
    assert np.array_equal(dataset.source_row_indices["Node2-Bogota"], np.array([0, 1]))


def test_load_calibration_dataset_excludes_requested_sensors(tmp_path: Path) -> None:
    """The loader should remove explicitly excluded sensors before alignment."""

    acquisition_dir = tmp_path / "aquisitions"
    response_dir = tmp_path / "responses"
    acquisition_dir.mkdir()
    response_dir.mkdir()

    base_rows = [
        _acquisition_row(
            timestamp_ms=1_000,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[0.0, 1.0, 2.0, 3.0],
        ),
        _acquisition_row(
            timestamp_ms=2_000,
            start_hz=100.0,
            end_hz=104.0,
            power_db=[1.0, 2.0, 3.0, 4.0],
        ),
    ]

    for sensor_id in ("Node1-Bogota", "Node2-Bogota", "Node3-Bogota"):
        _write_acquisition_csv(acquisition_dir / f"{sensor_id}.csv", base_rows)
        _write_response_csv(response_dir / f"{sensor_id}-response.csv")

    dataset = load_calibration_dataset(
        acquisition_dir=acquisition_dir,
        response_dir=response_dir,
        reference_sensor_id="Node1-Bogota",
        excluded_sensor_ids=("Node3-Bogota",),
    )

    assert dataset.sensor_ids == ("Node1-Bogota", "Node2-Bogota")
    assert "Node3-Bogota" not in dataset.sensor_shifts
    assert "Node3-Bogota" not in dataset.source_row_indices
    assert dataset.observations_power.shape == (2, 2, 4)


def test_fit_spectral_calibration_reduces_common_field_dispersion() -> None:
    """Synthetic calibration should reduce held-out inter-sensor dispersion."""

    rng = np.random.default_rng(12)
    n_sensors = 4
    n_experiments = 18
    n_frequencies = 64
    sensor_ids = tuple(f"Node{index + 1}" for index in range(n_sensors))
    frequency_hz = np.linspace(88.0e6, 108.0e6, n_frequencies)
    frequency_phase = np.linspace(0.0, 1.0, n_frequencies)

    true_log_gain = np.vstack(
        [
            0.18 * np.sin(2.0 * np.pi * frequency_phase + phase)
            for phase in np.linspace(0.0, np.pi, n_sensors, endpoint=False)
        ]
    )
    true_log_gain -= np.mean(true_log_gain, axis=0, keepdims=True)
    true_gain = np.exp(true_log_gain)

    true_noise = 0.02 + 0.01 * np.vstack(
        [
            1.0 + 0.2 * np.cos(2.0 * np.pi * frequency_phase + phase)
            for phase in np.linspace(0.0, np.pi / 2.0, n_sensors)
        ]
    )

    latent_spectra = np.vstack(
        [
            0.8
            + 0.15 * np.sin(2.0 * np.pi * frequency_phase * (1.0 + experiment / 6.0))
            + 0.05 * experiment
            for experiment in range(n_experiments)
        ]
    )
    latent_spectra = np.clip(latent_spectra, 0.1, None)

    noise_scale = 0.015
    observations = (
        true_gain[:, np.newaxis, :] * latent_spectra[np.newaxis, :, :]
        + true_noise[:, np.newaxis, :]
        + noise_scale * rng.normal(size=(n_sensors, n_experiments, n_frequencies))
    )
    observations = np.clip(observations, 1.0e-6, None)

    train_indices = np.arange(0, 12)
    test_indices = np.arange(12, n_experiments)
    result = fit_spectral_calibration(
        observations_power=observations,
        frequency_hz=frequency_hz,
        sensor_ids=sensor_ids,
        nominal_gain_power=np.ones((n_sensors, n_frequencies)),
        train_indices=train_indices,
        test_indices=test_indices,
        reliable_sensor_id="Node1",
        n_iterations=10,
        lambda_gain_smooth=30.0,
        lambda_noise_smooth=10.0,
    )

    corrected_test = apply_deployed_calibration(
        observations_power=observations[:, test_indices, :],
        gain_power=result.gain_power,
        additive_noise_power=result.additive_noise_power,
    )
    raw_dispersion_db = float(
        np.mean(np.std(power_linear_to_db(observations[:, test_indices, :]), axis=0))
    )
    corrected_dispersion_db = float(
        np.mean(np.std(power_linear_to_db(corrected_test), axis=0))
    )

    assert corrected_dispersion_db < raw_dispersion_db
    assert np.allclose(np.mean(np.log(result.gain_power), axis=0), 0.0, atol=1.0e-8)
    assert np.all(np.isfinite(result.objective_history))
    assert result.objective_history[-1] < result.objective_history[0]


def test_apply_deployed_calibration_and_consensus_shapes() -> None:
    """Deployment correction and consensus fusion should preserve array semantics."""

    observations = power_db_to_linear(
        np.asarray(
            [
                [[-30.0, -29.0, -28.0], [-31.0, -30.0, -29.0]],
                [[-29.0, -28.0, -27.0], [-30.0, -29.0, -28.0]],
            ]
        )
    )
    gain = np.asarray([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]])
    noise = np.asarray([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    sigma2 = np.asarray([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

    corrected = apply_deployed_calibration(observations, gain, noise)
    consensus = compute_network_consensus(corrected, sigma2)

    assert corrected.shape == observations.shape
    assert consensus.shape == (2, 3)
    assert np.all(consensus >= 0.0)


def test_apply_deployed_calibration_broadcasts_sensor_curves() -> None:
    """Deployment correction should support 2D and higher-rank sensor-first tensors."""

    gain = np.asarray([[2.0, 4.0, 8.0], [1.0, 2.0, 4.0]])
    noise = np.asarray([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])

    observations_2d = np.asarray([[5.0, 9.0, 17.0], [1.5, 2.5, 4.5]])
    corrected_2d = apply_deployed_calibration(observations_2d, gain, noise)

    expected_2d = np.asarray([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    assert corrected_2d.shape == observations_2d.shape
    assert np.allclose(corrected_2d, expected_2d)

    template = np.asarray(
        [
            [[2.0, 1.5, 1.0], [0.5, 0.25, 0.125]],
            [[1.0, 0.5, 0.25], [3.0, 2.0, 1.0]],
        ]
    )
    observations_4d = (
        gain[:, np.newaxis, np.newaxis, :] * template
        + noise[:, np.newaxis, np.newaxis, :]
    )
    corrected_4d = apply_deployed_calibration(observations_4d, gain, noise)

    assert corrected_4d.shape == observations_4d.shape
    assert np.allclose(corrected_4d, template)


def _acquisition_row(
    timestamp_ms: int,
    start_hz: float,
    end_hz: float,
    power_db: list[float],
) -> dict[str, str]:
    """Build a minimal acquisition CSV row used by the loader test."""

    return {
        "id": "1",
        "mac": "00:00:00:00:00:00",
        "campaign_id": "176",
        "pxx": json.dumps(power_db),
        "start_freq_hz": f"{start_hz}",
        "end_freq_hz": f"{end_hz}",
        "timestamp": f"{timestamp_ms}",
        "lat": "",
        "lng": "",
        "excursion_peak_to_peak_hz": "",
        "excursion_peak_deviation_hz": "",
        "excursion_rms_deviation_hz": "",
        "depth_peak_to_peak": "",
        "depth_peak_deviation": "",
        "depth_rms_deviation": "",
        "created_at": f"{timestamp_ms}",
    }


def _write_acquisition_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a minimal acquisition CSV fixture."""

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_response_csv(path: Path) -> None:
    """Write a nominal response CSV fixture with trailing summary rows."""

    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["central_freq_hz", "pwr_dBm", "error_dB"])
        writer.writerow([100.0, -25.0, -1.0])
        writer.writerow([200.0, -25.0, -2.0])
        writer.writerow(["mean", -25.5, -1.5])
