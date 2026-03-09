"""Spectral calibration framework for common-field sensor experiments.

This module implements the latent-variable calibration model described in
``docs/main.tex``. The core model assumes that, during calibration, every
sensor observes the same latent spectrum while keeping sensor-specific gain,
additive noise floor, and residual variance curves.

The implementation keeps the numerical core separate from notebook orchestration:

- CSV parsing and experiment alignment live in :func:`load_calibration_dataset`.
- Alternating estimation of the latent spectra and sensor parameters lives in
  :func:`fit_spectral_calibration`.
- Deployment-time correction and consensus fusion live in
  :func:`apply_deployed_calibration` and :func:`compute_network_consensus`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import factorized


FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]

_EPSILON = 1.0e-12


@dataclass(frozen=True)
class CalibrationDataset:
    """Aligned calibration dataset on a common frequency grid.

    Parameters
    ----------
    sensor_ids:
        Ordered sensor identifiers, usually derived from the acquisition file
        names.
    frequency_hz:
        Frequency-bin centers for the aligned PSD vectors [Hz].
    observations_power:
        Observed PSD values in linear power units with shape
        ``(n_sensors, n_experiments, n_frequencies)``.
    nominal_gain_power:
        Nominal multiplicative gain curves interpolated onto the calibration
        grid. These are dimensionless power ratios on the same shape as one
        sensor-frequency panel, i.e. ``(n_sensors, n_frequencies)``.
    experiment_timestamps_ms:
        Reference timestamps for the aligned experiments [ms since epoch].
    selected_band_hz:
        Common acquisition band retained for calibration ``(start_hz, end_hz)``.
    sensor_shifts:
        Integer index shifts used to align each sensor sequence against the
        reference sensor. A negative value means that sensor ``s`` starts later
        than the reference sequence.
    alignment_median_error_ms:
        Median absolute timestamp mismatch after applying the chosen shift [ms].
    source_row_indices:
        Original row indices used from each acquisition file after alignment.
    """

    sensor_ids: tuple[str, ...]
    frequency_hz: FloatArray
    observations_power: FloatArray
    nominal_gain_power: FloatArray
    experiment_timestamps_ms: IndexArray
    selected_band_hz: tuple[float, float]
    sensor_shifts: dict[str, int]
    alignment_median_error_ms: dict[str, float]
    source_row_indices: dict[str, IndexArray]


@dataclass(frozen=True)
class SpectralCalibrationResult:
    """Estimated sensor calibration parameters and latent spectra.

    Parameters
    ----------
    sensor_ids:
        Ordered sensor identifiers matching the fitted observation tensor.
    frequency_hz:
        Frequency-bin centers [Hz].
    gain_power:
        Estimated multiplicative sensor gain curves ``G_s(f)`` in linear power
        scale with shape ``(n_sensors, n_frequencies)``.
    additive_noise_power:
        Estimated additive sensor noise floors ``N_s(f)`` in linear power scale
        with shape ``(n_sensors, n_frequencies)``.
    residual_variance_power2:
        Estimated residual variances ``sigma_s^2(f)`` in squared linear power
        units with shape ``(n_sensors, n_frequencies)``.
    latent_spectra_power:
        Estimated latent common spectra ``S_k(f)`` on the network reference
        scale with shape ``(n_experiments, n_frequencies)``.
    nominal_gain_power:
        Nominal gain curves used in the reparameterization. When no nominal
        curves are provided, this field is an all-ones array.
    correction_gain_power:
        Residual multiplicative corrections such that
        ``gain_power = nominal_gain_power * correction_gain_power``.
    train_indices:
        Experiment indices used for parameter fitting.
    test_indices:
        Held-out experiment indices reserved for validation.
    objective_history:
        Penalized objective value after each alternating-estimation iteration.
    """

    sensor_ids: tuple[str, ...]
    frequency_hz: FloatArray
    gain_power: FloatArray
    additive_noise_power: FloatArray
    residual_variance_power2: FloatArray
    latent_spectra_power: FloatArray
    nominal_gain_power: FloatArray
    correction_gain_power: FloatArray
    train_indices: IndexArray
    test_indices: IndexArray
    objective_history: FloatArray


@dataclass(frozen=True)
class _AcquisitionRecord:
    """Single acquisition row parsed from the CSV files."""

    timestamp_ms: int
    start_freq_hz: float
    end_freq_hz: float
    power_db: FloatArray


def power_db_to_linear(
    power_db: FloatArray,  # PSD values in dB-like units
) -> FloatArray:  # Linear power ratio on the same array shape
    """Convert dB-like power values into linear power ratios."""

    return np.power(10.0, np.asarray(power_db, dtype=np.float64) / 10.0)


def power_linear_to_db(
    power_linear: FloatArray,  # Linear PSD or gain values
) -> FloatArray:  # dB representation with clipping at a tiny positive floor
    """Convert linear power values to dB after explicit positivity clipping."""

    return 10.0 * np.log10(
        np.clip(np.asarray(power_linear, dtype=np.float64), _EPSILON, None)
    )


def load_calibration_dataset(
    acquisition_dir: Path,  # Directory with sensor acquisition CSV files
    response_dir: Path,  # Directory with nominal response CSV files
    reference_sensor_id: str = "Node1-Bogota",  # Sensor used as alignment reference
    max_alignment_shift: int = 3,  # Maximum integer experiment offset to search
) -> CalibrationDataset:  # Aligned tensor ready for calibration
    """Load, align, and validate the common-field calibration dataset.

    The loader performs the minimum boundary work needed by the notebook:

    - Parse the PSD vectors stored inside the acquisition CSV rows.
    - Select the acquisition band shared by all sensors.
    - Align the per-sensor experiment sequences using small integer index shifts.
    - Interpolate nominal frequency-response curves onto the acquisition grid.

    Parameters
    ----------
    acquisition_dir:
        Directory containing one acquisition CSV per sensor.
    response_dir:
        Directory containing one nominal response CSV per sensor.
    reference_sensor_id:
        Sensor used to define the experiment ordering and reference timestamps.
    max_alignment_shift:
        Maximum absolute index shift considered when aligning each sensor to the
        reference sequence. This keeps the alignment logic simple and explicit.

    Returns
    -------
    CalibrationDataset
        Structured dataset with aligned PSD tensors in linear power units.

    Raises
    ------
    FileNotFoundError
        If the acquisition or response directories do not exist.
    ValueError
        If no common acquisition band exists or the aligned tensor is empty.
    """

    if not acquisition_dir.exists():
        raise FileNotFoundError(
            f"Acquisition directory does not exist: {acquisition_dir}"
        )
    if not response_dir.exists():
        raise FileNotFoundError(f"Response directory does not exist: {response_dir}")

    acquisition_paths = sorted(acquisition_dir.glob("*.csv"))
    if not acquisition_paths:
        raise ValueError(f"No acquisition CSV files were found in {acquisition_dir}")

    records_by_sensor = {
        path.stem: _load_acquisition_records(path) for path in acquisition_paths
    }
    sensor_ids = tuple(sorted(records_by_sensor))

    if reference_sensor_id not in records_by_sensor:
        raise ValueError(
            f"Reference sensor {reference_sensor_id!r} was not found in the acquisition set"
        )

    selected_band_hz = _select_common_band(records_by_sensor)
    filtered_records = {
        sensor_id: [
            record
            for record in sensor_records
            if math.isclose(record.start_freq_hz, selected_band_hz[0])
            and math.isclose(record.end_freq_hz, selected_band_hz[1])
        ]
        for sensor_id, sensor_records in records_by_sensor.items()
    }

    shifts, alignment_errors = _infer_alignment_shifts(
        records_by_sensor=filtered_records,
        reference_sensor_id=reference_sensor_id,
        max_alignment_shift=max_alignment_shift,
    )
    source_row_indices, reference_indices = _build_common_indices(
        lengths_by_sensor={
            sensor_id: len(records) for sensor_id, records in filtered_records.items()
        },
        sensor_shifts=shifts,
        reference_sensor_id=reference_sensor_id,
    )

    if reference_indices.size == 0:
        raise ValueError(
            "The aligned experiment set is empty after applying the sensor shifts"
        )

    reference_record = filtered_records[reference_sensor_id][int(reference_indices[0])]
    frequency_hz = _build_frequency_grid(
        start_freq_hz=reference_record.start_freq_hz,
        end_freq_hz=reference_record.end_freq_hz,
        n_bins=reference_record.power_db.size,
    )

    observations_power = []
    for sensor_id in sensor_ids:
        sensor_records = filtered_records[sensor_id]
        aligned_power_db = np.stack(
            [
                sensor_records[int(index)].power_db
                for index in source_row_indices[sensor_id]
            ],
            axis=0,
        )
        observations_power.append(power_db_to_linear(aligned_power_db))

    nominal_gain_power = np.stack(
        [
            _load_nominal_gain_curve(
                response_path=response_dir / f"{sensor_id}-response.csv",
                target_frequency_hz=frequency_hz,
            )
            for sensor_id in sensor_ids
        ],
        axis=0,
    )

    # Normalize the nominal curves to the network reference scale so the
    # identifiability constraint is consistent from the start.
    nominal_gain_power = _normalize_geometric_mean(nominal_gain_power)

    experiment_timestamps_ms = np.asarray(
        [
            filtered_records[reference_sensor_id][int(index)].timestamp_ms
            for index in reference_indices
        ],
        dtype=np.int64,
    )

    dataset = CalibrationDataset(
        sensor_ids=sensor_ids,
        frequency_hz=frequency_hz,
        observations_power=np.stack(observations_power, axis=0),
        nominal_gain_power=nominal_gain_power,
        experiment_timestamps_ms=experiment_timestamps_ms,
        selected_band_hz=selected_band_hz,
        sensor_shifts=shifts,
        alignment_median_error_ms=alignment_errors,
        source_row_indices=source_row_indices,
    )
    _validate_observation_tensor(dataset.observations_power)
    return dataset


def make_holdout_split(
    n_experiments: int,  # Number of aligned calibration experiments
    test_fraction: float = 0.2,  # Fraction held out for validation
) -> tuple[IndexArray, IndexArray]:  # Train and test experiment indices
    """Create a deterministic train/test split over entire experiments."""

    if n_experiments < 2:
        raise ValueError("At least two experiments are required for a train/test split")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must lie in (0, 1), received {test_fraction}")

    n_test = max(1, int(round(n_experiments * test_fraction)))
    n_test = min(n_test, n_experiments - 1)
    test_start = n_experiments - n_test
    train_indices = np.arange(0, test_start, dtype=np.int64)
    test_indices = np.arange(test_start, n_experiments, dtype=np.int64)
    return train_indices, test_indices


def fit_spectral_calibration(
    observations_power: FloatArray,  # Tensor with shape (sensors, experiments, frequencies)
    frequency_hz: FloatArray,  # Frequency grid [Hz]
    sensor_ids: tuple[str, ...],  # Ordered sensor identifiers
    nominal_gain_power: FloatArray | None = None,  # Optional nominal gain curves
    train_indices: IndexArray | None = None,  # Experiment indices used for fitting
    test_indices: IndexArray | None = None,  # Held-out experiments for diagnostics
    reliable_sensor_id: str = "Node7-Bogota",  # Softly anchored sensor
    n_iterations: int = 8,  # Alternating-estimation iterations
    lambda_gain_smooth: float = 250.0,  # Strength of the gain smoothness penalty
    lambda_noise_smooth: float = 50.0,  # Strength of the noise smoothness penalty
    lambda_reliable_anchor: float = 1.0,  # Penalty that keeps the reliable sensor close to nominal
    reliable_weight_boost: float = 1.0,  # Optional soft reliability boost in the latent-spectrum update
    max_correction_db: float
    | None = 12.0,  # Bound on the residual correction around the nominal curve [dB]
    min_variance: float = 1.0e-14,  # Numerical floor for residual variances
) -> SpectralCalibrationResult:  # Estimated gains, noise floors, and latent spectra
    """Fit the alternating spectral calibration model in linear power scale.

    The implementation follows the structure of ``docs/main.tex``:

    - latent spectra are updated by weighted least squares;
    - per-sensor gain/noise parameters are updated by affine regression at each
      frequency with explicit non-negativity handling;
    - second-difference penalties smooth the gain and noise curves over
      frequency;
    - identifiability is enforced by requiring a unit geometric-mean gain at
      each frequency.

    Parameters
    ----------
    observations_power:
        Observed PSD tensor with shape ``(n_sensors, n_experiments, n_frequencies)``.
    frequency_hz:
        Frequency-bin centers [Hz]. Only the length is used numerically, but the
        array is kept for consistency and downstream plotting.
    sensor_ids:
        Ordered sensor identifiers matching the first axis of
        ``observations_power``.
    nominal_gain_power:
        Optional nominal frequency-response curves. If provided, the model
        estimates a multiplicative residual correction around these curves.
    train_indices:
        Experiment indices used to fit the parameters. Defaults to all
        experiments when omitted.
    test_indices:
        Held-out experiment indices stored in the result for notebook
        diagnostics. These do not affect the fit.
    reliable_sensor_id:
        Sensor receiving the soft anchor regularization described in the
        theoretical framework.
    n_iterations:
        Number of alternating-estimation iterations.
    lambda_gain_smooth:
        Second-difference penalty on the log-gain correction curves.
    lambda_noise_smooth:
        Second-difference penalty on the additive noise curves.
    lambda_reliable_anchor:
        Soft anchor that keeps the reliable sensor close to the nominal gain
        curve, or to unity when no nominal curve is provided.
    reliable_weight_boost:
        Optional multiplicative boost on the reliable sensor weight in the
        latent-spectrum update. This is a practical surrogate for the soft
        variance prior in the document.
    max_correction_db:
        Optional bound on the residual multiplicative correction relative to the
        nominal response. This encodes the modeling assumption that the
        laboratory curve captures the dominant structure and the in-situ
        correction should remain comparatively small.
    min_variance:
        Lower numerical bound for the residual variances.

    Returns
    -------
    SpectralCalibrationResult
        Estimated calibration parameters and latent spectra.
    """

    observations_power = np.asarray(observations_power, dtype=np.float64)
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    _validate_observation_tensor(observations_power)

    n_sensors, n_experiments, n_frequencies = observations_power.shape
    if len(sensor_ids) != n_sensors:
        raise ValueError(
            "sensor_ids length must match the first axis of observations_power"
        )
    if frequency_hz.shape != (n_frequencies,):
        raise ValueError("frequency_hz must have shape (n_frequencies,)")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if reliable_weight_boost <= 0.0:
        raise ValueError("reliable_weight_boost must be strictly positive")
    if max_correction_db is not None and max_correction_db <= 0.0:
        raise ValueError("max_correction_db must be strictly positive when provided")

    if nominal_gain_power is None:
        nominal_gain_power = np.ones((n_sensors, n_frequencies), dtype=np.float64)
    else:
        nominal_gain_power = np.asarray(nominal_gain_power, dtype=np.float64)
        if nominal_gain_power.shape != (n_sensors, n_frequencies):
            raise ValueError(
                "nominal_gain_power must have shape (n_sensors, n_frequencies)"
            )
        nominal_gain_power = np.clip(nominal_gain_power, _EPSILON, None)
        nominal_gain_power = _normalize_geometric_mean(nominal_gain_power)

    if train_indices is None:
        train_indices = np.arange(n_experiments, dtype=np.int64)
    else:
        train_indices = np.asarray(train_indices, dtype=np.int64)
    if test_indices is None:
        test_indices = np.setdiff1d(
            np.arange(n_experiments, dtype=np.int64), train_indices
        )
    else:
        test_indices = np.asarray(test_indices, dtype=np.int64)

    if train_indices.size == 0:
        raise ValueError("train_indices cannot be empty")
    if np.unique(train_indices).size != train_indices.size:
        raise ValueError("train_indices must be unique")

    reliable_sensor_index = _sensor_index(
        sensor_ids=sensor_ids, sensor_id=reliable_sensor_id
    )
    train_observations = observations_power[:, train_indices, :]
    max_log_correction = (
        None
        if max_correction_db is None
        else np.log(10.0 ** (max_correction_db / 10.0))
    )

    # Initialize the model from the nominal curves and a conservative low
    # quantile of the observations as the starting additive floor.
    gain_power = nominal_gain_power.copy()
    additive_noise_power = 0.10 * np.quantile(train_observations, 0.05, axis=1)
    latent_spectra_power = np.median(
        train_observations / gain_power[:, np.newaxis, :], axis=0
    )
    latent_spectra_power = np.clip(latent_spectra_power, _EPSILON, None)

    residuals = (
        train_observations
        - gain_power[:, np.newaxis, :] * latent_spectra_power[np.newaxis, :, :]
    )
    residual_variance_power2 = np.maximum(np.mean(residuals**2, axis=1), min_variance)

    second_difference = _second_difference_operator(n_frequencies)
    smooth_penalty = (second_difference.T @ second_difference).tocsc()
    identity = sparse.eye(n_frequencies, format="csc")
    gain_solver = factorized(identity + lambda_gain_smooth * smooth_penalty)
    anchored_gain_solver = factorized(
        identity
        + lambda_gain_smooth * smooth_penalty
        + lambda_reliable_anchor * identity
    )
    noise_solver = factorized(identity + lambda_noise_smooth * smooth_penalty)

    objective_history = []

    for _ in range(n_iterations):
        weights = 1.0 / np.clip(residual_variance_power2, min_variance, None)
        weights[reliable_sensor_index] *= reliable_weight_boost

        # Update the latent spectra by the weighted least-squares formula.
        numerator = np.sum(
            weights[:, np.newaxis, :]
            * gain_power[:, np.newaxis, :]
            * (train_observations - additive_noise_power[:, np.newaxis, :]),
            axis=0,
        )
        denominator = np.sum(weights * gain_power**2, axis=0, keepdims=True)
        latent_spectra_power = np.clip(
            numerator / np.clip(denominator, _EPSILON, None), _EPSILON, None
        )

        raw_gain_power = np.empty_like(gain_power)
        raw_noise_power = np.empty_like(additive_noise_power)

        # Update each sensor curve frequency-by-frequency using an affine
        # nonnegative regression in the latent spectrum.
        for sensor_index in range(n_sensors):
            gain_fit, noise_fit = _fit_sensor_affine_curve(
                latent_spectra_power=latent_spectra_power,
                observations_power=train_observations[sensor_index],
            )
            raw_gain_power[sensor_index] = gain_fit
            raw_noise_power[sensor_index] = noise_fit

        correction_gain_power = np.clip(
            raw_gain_power / nominal_gain_power, _EPSILON, None
        )
        smoothed_log_correction = np.empty_like(correction_gain_power)

        # Smooth the gain correction in log-domain and the noise floor in linear
        # domain so the estimates respect the intended frequency regularity.
        for sensor_index in range(n_sensors):
            log_correction = np.log(correction_gain_power[sensor_index])
            if max_log_correction is not None:
                log_correction = np.clip(
                    log_correction, -max_log_correction, max_log_correction
                )
            if sensor_index == reliable_sensor_index:
                smoothed_log_correction[sensor_index] = anchored_gain_solver(
                    log_correction
                )
            else:
                smoothed_log_correction[sensor_index] = gain_solver(log_correction)
            additive_noise_power[sensor_index] = np.clip(
                noise_solver(raw_noise_power[sensor_index]), 0.0, None
            )

        if max_log_correction is not None:
            smoothed_log_correction = np.clip(
                smoothed_log_correction, -max_log_correction, max_log_correction
            )
        correction_gain_power = np.exp(smoothed_log_correction)
        gain_power = nominal_gain_power * correction_gain_power

        # Enforce the identifiability constraint by moving the common frequency
        # scale into the latent spectrum.
        log_gain = np.log(np.clip(gain_power, _EPSILON, None))
        log_gain_mean = np.mean(log_gain, axis=0, keepdims=True)
        scale_factor = np.exp(log_gain_mean)
        gain_power = gain_power / scale_factor
        latent_spectra_power = latent_spectra_power * scale_factor[0]
        correction_gain_power = gain_power / nominal_gain_power

        residuals = (
            train_observations
            - gain_power[:, np.newaxis, :] * latent_spectra_power[np.newaxis, :, :]
            - additive_noise_power[:, np.newaxis, :]
        )
        residual_variance_power2 = np.maximum(
            np.mean(residuals**2, axis=1), min_variance
        )
        objective_history.append(
            _penalized_objective(
                residuals=residuals,
                residual_variance_power2=residual_variance_power2,
                correction_gain_power=correction_gain_power,
                additive_noise_power=additive_noise_power,
                reliable_sensor_index=reliable_sensor_index,
                second_difference=second_difference,
                lambda_gain_smooth=lambda_gain_smooth,
                lambda_noise_smooth=lambda_noise_smooth,
                lambda_reliable_anchor=lambda_reliable_anchor,
            )
        )

    return SpectralCalibrationResult(
        sensor_ids=sensor_ids,
        frequency_hz=frequency_hz,
        gain_power=gain_power,
        additive_noise_power=additive_noise_power,
        residual_variance_power2=residual_variance_power2,
        latent_spectra_power=latent_spectra_power,
        nominal_gain_power=nominal_gain_power,
        correction_gain_power=correction_gain_power,
        train_indices=train_indices,
        test_indices=test_indices,
        objective_history=np.asarray(objective_history, dtype=np.float64),
    )


def apply_deployed_calibration(
    observations_power: FloatArray,  # Observed PSD values with shape (..., frequencies) or (sensors, ..., frequencies)
    gain_power: FloatArray,  # Gain curves with shape (sensors, frequencies)
    additive_noise_power: FloatArray,  # Noise floor curves with shape (sensors, frequencies)
    enforce_nonnegative: bool = True,  # Whether to truncate negative corrected spectra
) -> FloatArray:  # Corrected PSD estimates on the common network scale
    """Apply the deployment-time calibration map to new observations.

    The mapping corresponds to equation (deployed calibration) in the document:

    ``S_hat = (Y - N) / G``.

    Parameters
    ----------
    observations_power:
        New PSD observations. The first axis must be the sensor axis when the
        array is at least two-dimensional.
    gain_power:
        Estimated multiplicative gain curves ``G_s(f)``.
    additive_noise_power:
        Estimated additive noise floors ``N_s(f)``.
    enforce_nonnegative:
        If ``True``, apply the truncated nonnegativity version of the mapping.
    """

    observations_power = np.asarray(observations_power, dtype=np.float64)
    gain_power = np.asarray(gain_power, dtype=np.float64)
    additive_noise_power = np.asarray(additive_noise_power, dtype=np.float64)

    if gain_power.shape != additive_noise_power.shape:
        raise ValueError(
            "gain_power and additive_noise_power must share the same shape"
        )
    if observations_power.shape[0] != gain_power.shape[0]:
        raise ValueError(
            "The first axis of observations_power must match the sensor axis"
        )

    corrected = (observations_power - additive_noise_power[:, np.newaxis, :]) / np.clip(
        gain_power[:, np.newaxis, :], _EPSILON, None
    )
    if enforce_nonnegative:
        corrected = np.clip(corrected, 0.0, None)
    return corrected


def compute_network_consensus(
    corrected_power: FloatArray,  # Corrected PSD tensor with shape (sensors, experiments, frequencies)
    residual_variance_power2: FloatArray,  # Sensor reliability variances with shape (sensors, frequencies)
) -> FloatArray:  # Weighted consensus PSD with shape (experiments, frequencies)
    """Fuse corrected common-field spectra into a reliability-weighted consensus."""

    corrected_power = np.asarray(corrected_power, dtype=np.float64)
    residual_variance_power2 = np.asarray(residual_variance_power2, dtype=np.float64)
    if corrected_power.ndim != 3:
        raise ValueError(
            "corrected_power must have shape (sensors, experiments, frequencies)"
        )
    if residual_variance_power2.shape != (
        corrected_power.shape[0],
        corrected_power.shape[2],
    ):
        raise ValueError(
            "residual_variance_power2 must have shape (sensors, frequencies)"
        )

    weights = 1.0 / np.clip(residual_variance_power2, _EPSILON, None)
    numerator = np.sum(weights[:, np.newaxis, :] * corrected_power, axis=0)
    denominator = np.sum(weights, axis=0, keepdims=True)
    return numerator / np.clip(denominator, _EPSILON, None)


def _load_acquisition_records(
    path: Path,  # Acquisition CSV path
) -> list[_AcquisitionRecord]:  # Parsed rows sorted by timestamp
    """Parse the acquisition CSV into timestamped PSD vectors."""

    csv.field_size_limit(sys.maxsize)
    records = []
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            power_db = np.asarray(json.loads(row["pxx"]), dtype=np.float64)
            records.append(
                _AcquisitionRecord(
                    timestamp_ms=int(row["timestamp"]),
                    start_freq_hz=float(row["start_freq_hz"]),
                    end_freq_hz=float(row["end_freq_hz"]),
                    power_db=power_db,
                )
            )
    records.sort(key=lambda record: record.timestamp_ms)
    return records


def _load_nominal_gain_curve(
    response_path: Path,  # Nominal response CSV path
    target_frequency_hz: FloatArray,  # Frequency grid used by the calibration tensor
) -> FloatArray:  # Interpolated nominal gain in linear power scale
    """Load a nominal response CSV and interpolate its dB error curve."""

    if not response_path.exists():
        raise FileNotFoundError(f"Nominal response CSV does not exist: {response_path}")

    numeric_rows: list[tuple[float, float]] = []
    with response_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                frequency_hz = float(row["central_freq_hz"])
                error_db = float(row["error_dB"])
            except (TypeError, ValueError):
                continue
            numeric_rows.append((frequency_hz, error_db))

    if not numeric_rows:
        raise ValueError(f"No numeric response rows were found in {response_path}")

    numeric_rows.sort(key=lambda item: item[0])
    response_frequency_hz = np.asarray(
        [item[0] for item in numeric_rows], dtype=np.float64
    )
    response_error_db = np.asarray([item[1] for item in numeric_rows], dtype=np.float64)
    interpolated_error_db = np.interp(
        x=np.asarray(target_frequency_hz, dtype=np.float64),
        xp=response_frequency_hz,
        fp=response_error_db,
        left=response_error_db[0],
        right=response_error_db[-1],
    )
    return power_db_to_linear(interpolated_error_db)


def _select_common_band(
    records_by_sensor: dict[
        str, list[_AcquisitionRecord]
    ],  # Parsed acquisition rows for each sensor
) -> tuple[float, float]:  # Shared acquisition band with the widest coverage
    """Select the frequency band that is shared by every sensor."""

    coverage_by_band: dict[tuple[float, float], dict[str, int]] = {}
    for sensor_id, sensor_records in records_by_sensor.items():
        counts = Counter(
            (record.start_freq_hz, record.end_freq_hz) for record in sensor_records
        )
        for band_hz, count in counts.items():
            coverage_by_band.setdefault(band_hz, {})[sensor_id] = count

    shared_bands = [
        (band_hz, counts_by_sensor)
        for band_hz, counts_by_sensor in coverage_by_band.items()
        if len(counts_by_sensor) == len(records_by_sensor)
    ]
    if not shared_bands:
        raise ValueError("No acquisition band is shared by all sensors")

    best_band_hz, _ = max(
        shared_bands,
        key=lambda item: (min(item[1].values()), sum(item[1].values())),
    )
    return best_band_hz


def _infer_alignment_shifts(
    records_by_sensor: dict[
        str, list[_AcquisitionRecord]
    ],  # Filtered same-band records for each sensor
    reference_sensor_id: str,  # Sensor used as alignment reference
    max_alignment_shift: int,  # Search range for the integer index shift
) -> tuple[
    dict[str, int], dict[str, float]
]:  # Shift per sensor and median absolute timing error [ms]
    """Infer a small integer index shift for each sensor sequence."""

    reference_records = records_by_sensor[reference_sensor_id]
    shifts: dict[str, int] = {}
    alignment_errors: dict[str, float] = {}

    for sensor_id, sensor_records in records_by_sensor.items():
        if sensor_id == reference_sensor_id:
            shifts[sensor_id] = 0
            alignment_errors[sensor_id] = 0.0
            continue

        best_shift = 0
        best_error = float("inf")

        for shift in range(-max_alignment_shift, max_alignment_shift + 1):
            time_differences = [
                sensor_records[sensor_index].timestamp_ms
                - reference_records[reference_index].timestamp_ms
                for reference_index in range(len(reference_records))
                if 0 <= (sensor_index := reference_index + shift) < len(sensor_records)
            ]
            if not time_differences:
                continue
            median_abs_error_ms = float(
                median(abs(delta_ms) for delta_ms in time_differences)
            )
            if median_abs_error_ms < best_error or (
                math.isclose(median_abs_error_ms, best_error)
                and abs(shift) < abs(best_shift)
            ):
                best_shift = shift
                best_error = median_abs_error_ms

        shifts[sensor_id] = best_shift
        alignment_errors[sensor_id] = best_error

    return shifts, alignment_errors


def _build_common_indices(
    lengths_by_sensor: dict[
        str, int
    ],  # Number of available rows per sensor after same-band filtering
    sensor_shifts: dict[
        str, int
    ],  # Integer index shift relative to the reference sensor
    reference_sensor_id: str,  # Reference sensor name
) -> tuple[
    dict[str, IndexArray], IndexArray
]:  # Per-sensor source indices and reference indices
    """Compute the common experiment indices that remain valid for all sensors."""

    reference_length = lengths_by_sensor[reference_sensor_id]
    start_index = max(max(0, -shift) for shift in sensor_shifts.values())
    stop_index = min(
        min(reference_length, lengths_by_sensor[sensor_id] - shift)
        for sensor_id, shift in sensor_shifts.items()
    )

    if stop_index <= start_index:
        raise ValueError(
            "No common aligned experiments remain after the sensor shifts were applied"
        )

    reference_indices = np.arange(start_index, stop_index, dtype=np.int64)
    source_row_indices = {
        sensor_id: reference_indices + shift
        for sensor_id, shift in sensor_shifts.items()
    }
    return source_row_indices, reference_indices


def _build_frequency_grid(
    start_freq_hz: float,  # Lower band edge [Hz]
    end_freq_hz: float,  # Upper band edge [Hz]
    n_bins: int,  # Number of PSD bins
) -> FloatArray:  # Frequency-bin centers [Hz]
    """Build equally spaced frequency-bin centers from the acquisition band."""

    bin_width_hz = (end_freq_hz - start_freq_hz) / n_bins
    return start_freq_hz + (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width_hz


def _fit_sensor_affine_curve(
    latent_spectra_power: FloatArray,  # Latent spectra with shape (experiments, frequencies)
    observations_power: FloatArray,  # Sensor observations with shape (experiments, frequencies)
) -> tuple[FloatArray, FloatArray]:  # Gain and additive noise curves for one sensor
    """Fit the nonnegative affine model ``y = g x + n`` for every frequency bin."""

    x = latent_spectra_power
    y = observations_power
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    x_var = np.sum(x_centered**2, axis=0)
    xy_cov = np.sum(x_centered * y_centered, axis=0)

    gain_ols = np.divide(
        xy_cov, x_var, out=np.zeros_like(x_var), where=x_var > _EPSILON
    )
    noise_ols = y_mean - gain_ols * x_mean

    # Compute the boundary solutions needed when the unconstrained affine fit
    # violates the nonnegativity constraints.
    gain_with_zero_noise = np.maximum(
        np.sum(x * y, axis=0) / np.clip(np.sum(x * x, axis=0), _EPSILON, None), 0.0
    )
    noise_with_zero_gain = np.maximum(y_mean, 0.0)

    residuals_zero_noise = y - gain_with_zero_noise[np.newaxis, :] * x
    residuals_zero_gain = y - noise_with_zero_gain[np.newaxis, :]
    sse_zero_noise = np.sum(residuals_zero_noise**2, axis=0)
    sse_zero_gain = np.sum(residuals_zero_gain**2, axis=0)

    gain = gain_ols.copy()
    noise = noise_ols.copy()

    invalid_mask = (gain_ols < 0.0) | (noise_ols < 0.0)
    choose_zero_noise = invalid_mask & (sse_zero_noise <= sse_zero_gain)
    choose_zero_gain = invalid_mask & ~choose_zero_noise

    gain[choose_zero_noise] = gain_with_zero_noise[choose_zero_noise]
    noise[choose_zero_noise] = 0.0
    gain[choose_zero_gain] = 0.0
    noise[choose_zero_gain] = noise_with_zero_gain[choose_zero_gain]

    gain = np.clip(gain, _EPSILON, None)
    noise = np.clip(noise, 0.0, None)
    return gain, noise


def _second_difference_operator(
    n_frequencies: int,  # Length of the frequency grid
) -> sparse.csc_matrix:  # Sparse finite-difference matrix
    """Build the second-order finite-difference operator used for smoothing."""

    if n_frequencies < 3:
        return sparse.csc_matrix((0, n_frequencies))
    return sparse.diags(
        diagonals=[
            np.ones(n_frequencies - 2, dtype=np.float64),
            -2.0 * np.ones(n_frequencies - 2, dtype=np.float64),
            np.ones(n_frequencies - 2, dtype=np.float64),
        ],
        offsets=[0, 1, 2],
        shape=(n_frequencies - 2, n_frequencies),
        format="csc",
    )


def _penalized_objective(
    residuals: FloatArray,  # Training residual tensor
    residual_variance_power2: FloatArray,  # Estimated residual variances
    correction_gain_power: FloatArray,  # Residual multiplicative gain correction
    additive_noise_power: FloatArray,  # Estimated additive noise floor
    reliable_sensor_index: int,  # Index of the softly anchored sensor
    second_difference: sparse.csc_matrix,  # Finite-difference operator
    lambda_gain_smooth: float,  # Log-gain smoothness weight
    lambda_noise_smooth: float,  # Noise smoothness weight
    lambda_reliable_anchor: float,  # Reliable sensor anchor weight
) -> float:  # Penalized objective value
    """Evaluate the penalized negative log-likelihood used by the estimator."""

    nll = 0.5 * np.sum(
        np.log(np.clip(residual_variance_power2, _EPSILON, None))[:, np.newaxis, :]
        + residuals**2
        / np.clip(residual_variance_power2, _EPSILON, None)[:, np.newaxis, :]
    )

    if second_difference.shape[0] == 0:
        gain_penalty = 0.0
        noise_penalty = 0.0
    else:
        gain_penalty = float(
            np.sum(
                (
                    second_difference
                    @ np.log(np.clip(correction_gain_power.T, _EPSILON, None))
                )
                ** 2
            )
        )
        noise_penalty = float(np.sum((second_difference @ additive_noise_power.T) ** 2))

    reliable_penalty = float(
        np.sum(
            np.log(
                np.clip(correction_gain_power[reliable_sensor_index], _EPSILON, None)
            )
            ** 2
        )
    )
    return (
        nll
        + lambda_gain_smooth * gain_penalty
        + lambda_noise_smooth * noise_penalty
        + lambda_reliable_anchor * reliable_penalty
    )


def _normalize_geometric_mean(
    gain_power: FloatArray,  # Gain curves with shape (sensors, frequencies)
) -> FloatArray:  # Gain curves rescaled to unit geometric mean per frequency
    """Normalize gain curves so their geometric mean is one at each frequency."""

    log_gain = np.log(np.clip(gain_power, _EPSILON, None))
    return np.exp(log_gain - np.mean(log_gain, axis=0, keepdims=True))


def _sensor_index(
    sensor_ids: tuple[str, ...],  # Ordered sensor identifiers
    sensor_id: str,  # Sensor to locate
) -> int:  # Integer index of the requested sensor
    """Return the sensor index or raise a clear error."""

    try:
        return sensor_ids.index(sensor_id)
    except ValueError as error:
        raise ValueError(
            f"Sensor {sensor_id!r} was not found in {sensor_ids}"
        ) from error


def _validate_observation_tensor(
    observations_power: FloatArray,  # PSD tensor to validate
) -> None:  # Raises if the tensor breaks the model assumptions
    """Validate tensor shape, finiteness, and positivity assumptions."""

    if observations_power.ndim != 3:
        raise ValueError(
            "observations_power must have shape (sensors, experiments, frequencies)"
        )
    if observations_power.shape[0] < 2:
        raise ValueError("At least two sensors are required for calibration")
    if observations_power.shape[1] < 2:
        raise ValueError("At least two calibration experiments are required")
    if observations_power.shape[2] < 3:
        raise ValueError(
            "At least three frequency bins are required for second-difference smoothing"
        )
    if not np.all(np.isfinite(observations_power)):
        raise ValueError("observations_power contains non-finite values")
    if np.any(observations_power <= 0.0):
        raise ValueError(
            "observations_power must be strictly positive in linear power scale"
        )
