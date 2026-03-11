"""Tests for the spectral calibration numerical core."""

from __future__ import annotations

import numpy as np
import pytest

from measurement_calibration.spectral_calibration import (
    apply_deployed_calibration,
    compute_network_consensus,
    fit_spectral_calibration,
    make_holdout_split,
    power_db_to_linear,
    power_linear_to_db,
)


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
    assert result.latent_variation_power2.shape == (n_frequencies,)
    assert result.frequency_information_weight.shape == (n_frequencies,)
    assert result.information_weight.shape == (n_sensors, n_frequencies)
    assert result.frequency_low_information_mask.shape == (n_frequencies,)
    assert result.low_information_mask.shape == (n_sensors, n_frequencies)
    assert result.gain_at_correction_bound_mask.shape == (n_sensors, n_frequencies)
    assert result.noise_zero_mask.shape == (n_sensors, n_frequencies)
    assert not np.any(result.noise_zero_mask)


def test_fit_spectral_calibration_flags_low_information_bins() -> None:
    """Weakly varying calibration spectra should trigger conservative updates."""

    n_sensors = 3
    n_experiments = 8
    n_frequencies = 24
    sensor_ids = tuple(f"Node{index + 1}" for index in range(n_sensors))
    frequency_hz = np.linspace(88.0e6, 108.0e6, n_frequencies)

    # With no across-experiment spectral variation, gain and additive noise are
    # not separately identifiable. The estimator should detect that regime and
    # stay close to its conservative references instead of inventing large
    # corrections.
    latent_spectra = np.full((n_experiments, n_frequencies), 0.75, dtype=np.float64)
    additive_noise = np.asarray(
        [
            np.linspace(0.01, 0.015, n_frequencies),
            np.linspace(0.012, 0.017, n_frequencies),
            np.linspace(0.011, 0.016, n_frequencies),
        ]
    )
    observations = latent_spectra[np.newaxis, :, :] + additive_noise[:, np.newaxis, :]

    result = fit_spectral_calibration(
        observations_power=observations,
        frequency_hz=frequency_hz,
        sensor_ids=sensor_ids,
        nominal_gain_power=np.ones((n_sensors, n_frequencies)),
        reliable_sensor_id="Node1",
        n_iterations=4,
        lambda_gain_smooth=15.0,
        lambda_noise_smooth=30.0,
        lambda_gain_reference=8.0,
        lambda_noise_reference=80.0,
        low_information_threshold_ratio=0.5,
        low_information_weight=0.02,
    )

    assert np.all(result.frequency_low_information_mask)
    assert np.all(result.low_information_mask)
    assert np.allclose(result.frequency_information_weight, 0.02)
    assert np.allclose(result.information_weight, 0.02)
    assert np.max(np.abs(np.log(result.correction_gain_power))) < 0.12
    assert np.all(np.isfinite(result.additive_noise_power))
    assert np.all(result.additive_noise_power > 0.0)
    assert not np.any(result.noise_zero_mask)


def test_fit_spectral_calibration_downweights_sensor_specific_mismatch() -> None:
    """A noisy sensor should receive lower local information weight than clean peers."""

    rng = np.random.default_rng(21)
    n_sensors = 3
    n_experiments = 14
    n_frequencies = 40
    sensor_ids = tuple(f"Node{index + 1}" for index in range(n_sensors))
    frequency_hz = np.linspace(88.0e6, 108.0e6, n_frequencies)
    frequency_phase = np.linspace(0.0, 1.0, n_frequencies)

    latent_spectra = np.vstack(
        [
            0.7
            + 0.12 * np.sin(2.0 * np.pi * frequency_phase * (1.0 + experiment / 5.0))
            + 0.03 * experiment
            for experiment in range(n_experiments)
        ]
    )
    latent_spectra = np.clip(latent_spectra, 0.15, None)
    true_noise = 0.01 + 0.002 * np.vstack(
        [
            1.0 + 0.1 * np.cos(2.0 * np.pi * frequency_phase + phase)
            for phase in (0.0, 0.5, 1.0)
        ]
    )

    observations = (
        latent_spectra[np.newaxis, :, :]
        + true_noise[:, np.newaxis, :]
        + 0.006 * rng.normal(size=(n_sensors, n_experiments, n_frequencies))
    )
    observations[1, :, 12:24] += 0.08 * rng.normal(size=(n_experiments, 12))
    observations = np.clip(observations, 1.0e-6, None)

    result = fit_spectral_calibration(
        observations_power=observations,
        frequency_hz=frequency_hz,
        sensor_ids=sensor_ids,
        nominal_gain_power=np.ones((n_sensors, n_frequencies)),
        reliable_sensor_id="Node1",
        n_iterations=6,
        lambda_gain_smooth=20.0,
        lambda_noise_smooth=15.0,
        lambda_gain_reference=5.0,
        lambda_noise_reference=40.0,
    )

    noisy_slice = slice(12, 24)
    noisy_sensor_weight = np.median(result.information_weight[1, noisy_slice])
    clean_sensor_weight = np.median(result.information_weight[0, noisy_slice])

    assert np.median(result.frequency_information_weight[noisy_slice]) > 0.75
    assert noisy_sensor_weight < 0.85 * clean_sensor_weight
    assert np.mean(result.low_information_mask[1, noisy_slice]) > np.mean(
        result.low_information_mask[0, noisy_slice]
    )


def test_fit_spectral_calibration_reports_gain_cap_hits() -> None:
    """Residual-gain cap diagnostics should mark saturated bins explicitly."""

    n_sensors = 3
    n_experiments = 10
    n_frequencies = 28
    sensor_ids = tuple(f"Node{index + 1}" for index in range(n_sensors))
    frequency_hz = np.linspace(88.0e6, 108.0e6, n_frequencies)
    frequency_phase = np.linspace(0.0, 1.0, n_frequencies)

    true_log_gain = np.vstack(
        [
            np.zeros(n_frequencies),
            0.55 * np.sin(2.0 * np.pi * frequency_phase),
            -0.45 * np.cos(2.0 * np.pi * frequency_phase),
        ]
    )
    true_log_gain -= np.mean(true_log_gain, axis=0, keepdims=True)
    true_gain = np.exp(true_log_gain)
    latent_spectra = np.vstack(
        [
            0.9
            + 0.2 * np.sin(2.0 * np.pi * frequency_phase * (1.0 + experiment / 7.0))
            + 0.04 * experiment
            for experiment in range(n_experiments)
        ]
    )
    latent_spectra = np.clip(latent_spectra, 0.2, None)

    observations = true_gain[:, np.newaxis, :] * latent_spectra[np.newaxis, :, :] + 0.01

    result = fit_spectral_calibration(
        observations_power=np.clip(observations, 1.0e-6, None),
        frequency_hz=frequency_hz,
        sensor_ids=sensor_ids,
        nominal_gain_power=np.ones((n_sensors, n_frequencies)),
        reliable_sensor_id="Node1",
        n_iterations=6,
        lambda_gain_smooth=10.0,
        lambda_noise_smooth=10.0,
        lambda_gain_reference=2.0,
        lambda_noise_reference=20.0,
        max_correction_db=0.5,
    )

    assert np.any(result.gain_at_correction_bound_mask[1:])
    max_log_correction = np.log(10.0 ** (0.5 / 10.0))
    assert (
        np.max(np.abs(np.log(result.correction_gain_power)))
        <= max_log_correction + 1.0e-8
    )


def test_fit_spectral_calibration_validates_train_and_test_indices() -> None:
    """Explicit train/test splits must be unique, in-bounds, and disjoint."""

    observations = np.full((2, 4, 3), 1.0, dtype=np.float64)
    frequency_hz = np.asarray([100.0, 101.5, 103.0], dtype=np.float64)
    sensor_ids = ("Node1", "Node2")

    with pytest.raises(ValueError, match="disjoint"):
        fit_spectral_calibration(
            observations_power=observations,
            frequency_hz=frequency_hz,
            sensor_ids=sensor_ids,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([1, 2], dtype=np.int64),
            reliable_sensor_id="Node1",
        )

    with pytest.raises(ValueError, match="test_indices must be unique"):
        fit_spectral_calibration(
            observations_power=observations,
            frequency_hz=frequency_hz,
            sensor_ids=sensor_ids,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([2, 2], dtype=np.int64),
            reliable_sensor_id="Node1",
        )

    with pytest.raises(ValueError, match="test_indices must lie within"):
        fit_spectral_calibration(
            observations_power=observations,
            frequency_hz=frequency_hz,
            sensor_ids=sensor_ids,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            test_indices=np.asarray([4], dtype=np.int64),
            reliable_sensor_id="Node1",
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_spectral_calibration(
            observations_power=observations,
            frequency_hz=np.asarray([100.0, 100.5, 100.5], dtype=np.float64),
            sensor_ids=sensor_ids,
            reliable_sensor_id="Node1",
        )


def test_make_holdout_split_random_is_reproducible_and_disjoint() -> None:
    """Random hold-out splits should remain deterministic under a fixed seed."""

    train_indices_a, test_indices_a = make_holdout_split(
        n_experiments=10,
        test_fraction=0.3,
        strategy="random",
        random_seed=17,
    )
    train_indices_b, test_indices_b = make_holdout_split(
        n_experiments=10,
        test_fraction=0.3,
        strategy="random",
        random_seed=17,
    )

    assert np.array_equal(train_indices_a, train_indices_b)
    assert np.array_equal(test_indices_a, test_indices_b)
    assert np.intersect1d(train_indices_a, test_indices_a).size == 0
    assert train_indices_a.size + test_indices_a.size == 10


def test_fit_spectral_calibration_warns_and_counts_nonfinite_solver_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-finite sparse solves should warn and record the fallback count."""

    observations = np.full((2, 4, 5), 1.0, dtype=np.float64)
    frequency_hz = np.linspace(100.0, 104.0, observations.shape[2])
    sensor_ids = ("Node1", "Node2")

    def _return_nan_solution(*_args: object, **_kwargs: object) -> np.ndarray:
        """Return a non-finite sparse-solver output for every requested system."""

        return np.full(observations.shape[2] * 2, np.nan, dtype=np.float64)

    monkeypatch.setattr(
        "measurement_calibration.spectral_calibration.spsolve",
        _return_nan_solution,
    )

    with pytest.warns(RuntimeWarning, match="non-finite"):
        result = fit_spectral_calibration(
            observations_power=observations,
            frequency_hz=frequency_hz,
            sensor_ids=sensor_ids,
            reliable_sensor_id="Node1",
            n_iterations=2,
        )

    assert np.array_equal(result.solver_nonfinite_step_count, np.asarray([2, 2]))
    assert np.allclose(result.correction_gain_power, 1.0)
    assert np.all(result.additive_noise_power > 0.0)


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
    noise = np.asarray([[1.0e-5, 1.0e-5, 1.0e-5], [2.0e-5, 2.0e-5, 2.0e-5]])
    sigma2 = np.asarray([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])

    corrected = apply_deployed_calibration(observations, gain, noise)
    consensus = compute_network_consensus(corrected, sigma2)

    assert corrected.shape == observations.shape
    assert consensus.shape == (2, 3)
    assert np.all(consensus >= 0.0)


def test_compute_network_consensus_ignores_zero_clipped_bins() -> None:
    """Consensus should drop zero-clipped bins instead of treating them as votes."""

    corrected = np.asarray(
        [
            [[1.0, 0.0, 2.0, 0.0]],
            [[3.0, 4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    sigma2 = np.ones((2, 4), dtype=np.float64)

    consensus = compute_network_consensus(corrected, sigma2)

    assert np.allclose(consensus[0, :3], np.asarray([2.0, 4.0, 2.0]))
    assert np.isnan(consensus[0, 3])


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
