"""Spectral calibration framework for aligned sensor experiments.

This module implements the latent-variable calibration model described in
``docs/main.tex``. The core model assumes that, during calibration, every
sensor observes the same latent spectrum while keeping sensor-specific gain,
additive noise floor, and residual variance curves.

The implementation keeps the numerical core separate from notebook orchestration:

- Dataset-specific loading and alignment live in higher-level adapters, such
  as :mod:`measurement_calibration.campaign_calibration`.
- Alternating estimation of the latent spectra and sensor parameters lives in
  :func:`fit_spectral_calibration`.
- Deployment-time correction and consensus fusion live in
  :func:`apply_deployed_calibration` and :func:`compute_network_consensus`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve


FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]

_EPSILON = 1.0e-12
_SPECTRAL_FIT_CONFIG_PARAMETER_NAMES = (
    "n_iterations",
    "lambda_gain_smooth",
    "lambda_noise_smooth",
    "lambda_gain_reference",
    "lambda_noise_reference",
    "lambda_reliable_anchor",
    "reliable_weight_boost",
    "max_correction_db",
    "low_information_threshold_ratio",
    "low_information_weight",
    "min_variance",
)


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
    latent_variation_power2:
        Across-experiment latent-spectrum variance at each frequency on the
        current network scale. This is a direct identifiability diagnostic:
        bins with very small latent variation do not support a stable
        gain-versus-noise separation.
    frequency_information_weight:
        Network-wide identifiability weights with shape ``(n_frequencies,)``.
        These summarize how much across-experiment latent variation exists at
        each frequency, regardless of sensor-specific residual quality.
    information_weight:
        Sensor-specific confidence weights in ``[0, 1]`` with shape
        ``(n_sensors, n_frequencies)`` used during the joint sensor updates.
        These combine the network-wide latent-variation diagnostic with a local
        affine-design conditioning metric and a residual-SNR term, so noisy
        sensors can be downweighted even when the global latent variation is
        adequate.
    frequency_low_information_mask:
        Boolean mask with shape ``(n_frequencies,)`` that flags globally weak
        bins from the latent-variation diagnostic alone.
    low_information_mask:
        Sensor-specific Boolean mask with shape ``(n_sensors, n_frequencies)``
        that flags bins where the full local identifiability weight remains low.
    gain_at_correction_bound_mask:
        Boolean mask with shape ``(n_sensors, n_frequencies)`` that marks
        residual gain corrections that saturate the configured log-gain cap.
    noise_zero_mask:
        Boolean mask with shape ``(n_sensors, n_frequencies)`` that marks
        additive-noise estimates numerically indistinguishable from zero. With
        the softplus parameterization this should usually remain false, which is
        itself a useful boundary diagnostic.
    solver_nonfinite_step_count:
        Per-sensor count of Gauss-Newton updates whose sparse linear solve
        returned non-finite values, forcing the fitter to reuse the previous
        iterate for that sensor instead of accepting an unstable step.
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
    latent_variation_power2: FloatArray
    frequency_information_weight: FloatArray
    information_weight: FloatArray
    frequency_low_information_mask: NDArray[np.bool_]
    low_information_mask: NDArray[np.bool_]
    gain_at_correction_bound_mask: NDArray[np.bool_]
    noise_zero_mask: NDArray[np.bool_]
    solver_nonfinite_step_count: IndexArray


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


def _softplus(
    values: FloatArray,  # Unconstrained real-valued parameter
) -> FloatArray:  # Strictly positive mapped quantity on the same shape
    """Map unconstrained values to positive reals with a stable softplus."""

    values = np.asarray(values, dtype=np.float64)
    return np.logaddexp(0.0, values)


def _inverse_softplus(
    positive_values: FloatArray,  # Strictly positive target values
) -> FloatArray:  # Unconstrained parameter whose softplus returns the target
    """Invert the softplus map for positive values in a numerically stable way."""

    positive_values = np.clip(
        np.asarray(positive_values, dtype=np.float64), _EPSILON, None
    )
    large_mask = positive_values > 20.0
    inverse = np.empty_like(positive_values)
    inverse[large_mask] = positive_values[large_mask]
    inverse[~large_mask] = np.log(np.expm1(positive_values[~large_mask]))
    return inverse


def _sigmoid(
    values: FloatArray,  # Unconstrained real-valued parameter
) -> FloatArray:  # Logistic derivative in the interval (0, 1)
    """Evaluate the logistic function with overflow-safe branches."""

    values = np.asarray(values, dtype=np.float64)
    positive_mask = values >= 0.0
    negative_mask = ~positive_mask
    result = np.empty_like(values)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(values[negative_mask])
    result[negative_mask] = exp_values / (1.0 + exp_values)
    return result


def make_holdout_split(
    n_experiments: int,  # Number of aligned calibration experiments
    test_fraction: float = 0.2,  # Fraction held out for validation
    strategy: str = "tail",  # Split policy: chronological tail or seeded random
    random_seed: int | None = None,  # Optional seed for randomized splits
) -> tuple[IndexArray, IndexArray]:  # Train and test experiment indices
    """Create a train/test split over whole calibration experiments.

    Parameters
    ----------
    n_experiments:
        Number of aligned experiments available for calibration.
    test_fraction:
        Fraction of experiments reserved for hold-out diagnostics.
    strategy:
        Split policy. ``"tail"`` keeps the final block as hold-out, which is
        useful when chronology matters. ``"random"`` draws a reproducible
        experiment-level split when ``random_seed`` is provided.
    random_seed:
        Seed used only when ``strategy="random"``.
    """

    if n_experiments < 2:
        raise ValueError("At least two experiments are required for a train/test split")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must lie in (0, 1), received {test_fraction}")
    if strategy not in {"tail", "random"}:
        raise ValueError(
            f"strategy must be either 'tail' or 'random', received {strategy!r}."
        )

    n_test = max(1, int(round(n_experiments * test_fraction)))
    n_test = min(n_test, n_experiments - 1)
    all_indices = np.arange(n_experiments, dtype=np.int64)
    if strategy == "tail":
        test_start = n_experiments - n_test
        train_indices = all_indices[:test_start]
        test_indices = all_indices[test_start:]
    else:
        rng = np.random.default_rng(random_seed)
        test_indices = np.sort(
            np.asarray(
                rng.choice(n_experiments, size=n_test, replace=False),
                dtype=np.int64,
            )
        )
        train_mask = np.ones(n_experiments, dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_indices[train_mask]
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
    lambda_gain_reference: float = 2.0,  # Conservative damping toward the previous gain correction
    lambda_noise_reference: float = 20.0,  # Conservative damping toward the previous noise floor
    lambda_reliable_anchor: float = 1.0,  # Penalty that keeps the reliable sensor close to nominal
    reliable_weight_boost: float = 1.0,  # Optional soft reliability boost in the latent-spectrum update
    max_correction_db: float
    | None = 12.0,  # Bound on the residual correction around the nominal curve [dB]
    low_information_threshold_ratio: float = 0.05,  # Latent-variance threshold relative to the median informative bin
    low_information_weight: float = 0.10,  # Residual data weight applied to low-information bins
    min_variance: float = 1.0e-14,  # Numerical floor for residual variances
) -> SpectralCalibrationResult:  # Estimated gains, noise floors, and latent spectra
    """Fit the alternating spectral calibration model with structured priors.

    The implementation follows the structure of ``docs/main.tex``:

    - latent spectra are updated by weighted least squares;
    - each sensor update performs one frequency-coupled Gauss-Newton step on
      the residual log-gain correction and a positive softplus noise
      parameterization together;
    - second-difference penalties smooth the residual log-gain curves over
      frequency, which matches the multiplicative geometry of the transfer
      function in dB space;
    - global low-information bins are detected from latent-spectrum variation,
      then refined into sensor-specific weights using the local affine-design
      conditioning and residual-SNR diagnostics;
    - identifiability is enforced by requiring a unit geometric-mean gain at
      each frequency.

    The optimizer remains an alternating fixed-iteration scheme rather than a
    convergence-controlled trust-region method. The hardening in this module
    focuses on making that approximation explicit and observable: frequency
    spacing now enters the smoothness operator directly, train/test splits are
    validated for disjointness and bounds, and failed sparse solves emit a
    warning while recording how often the previous iterate had to be reused.

    Parameters
    ----------
    observations_power:
        Observed PSD tensor with shape ``(n_sensors, n_experiments, n_frequencies)``.
    frequency_hz:
        Strictly increasing frequency-bin centers [Hz]. The actual spacing is
        used in the smoothness operator, so irregular grids keep their true
        geometry instead of being treated like unit-spaced indices.
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
        Second-difference penalty on the residual log-gain-correction curves.
    lambda_noise_smooth:
        Second-difference penalty on the unconstrained softplus noise
        parameter. This keeps the positive noise floor smooth without forcing
        hard clipping at zero.
    lambda_gain_reference:
        Conservative quadratic penalty that keeps each log-gain-correction
        update close to the previous iteration. This reduces bin-to-bin
        oscillations when the calibration data are only weakly informative.
    lambda_noise_reference:
        Conservative quadratic penalty that keeps each softplus noise
        parameter close to the previous iteration. In practice this makes the
        additive floor harder to move than the multiplicative gain.
    lambda_reliable_anchor:
        Soft anchor that keeps the reliable sensor close to the nominal gain
        curve, or to unity when no nominal curve is provided.
    reliable_weight_boost:
        Optional multiplicative boost on the reliable sensor weight in the
        latent-spectrum update. This is a practical surrogate for the soft
        variance prior in the document.
    max_correction_db:
        Optional bound on the residual multiplicative correction relative to the
        nominal response. The cap is enforced in log-gain space, which keeps
        the constraint symmetric in dB.
    low_information_threshold_ratio:
        Fraction of the median latent-spectrum variance used to define
        globally weak bins. Frequencies below this threshold are treated as
        poorly identifiable before the sensor-specific conditioning diagnostics
        are applied.
    low_information_weight:
        Residual data-weight floor assigned to low-information bins. Values
        closer to zero make the joint sensor update rely more heavily on
        smoothness and reference penalties at those frequencies.
    min_variance:
        Lower numerical bound for the residual variances.

    Returns
    -------
    SpectralCalibrationResult
        Estimated calibration parameters and latent spectra.

    Warns
    -----
    RuntimeWarning
        If one of the per-sensor sparse Gauss-Newton systems returns a
        non-finite solution and the previous iterate must be reused.
    """

    observations_power = np.asarray(observations_power, dtype=np.float64)
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    _validate_observation_tensor(observations_power)

    n_sensors, n_experiments, n_frequencies = observations_power.shape
    if len(sensor_ids) != n_sensors:
        raise ValueError(
            "sensor_ids length must match the first axis of observations_power"
        )
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("sensor_ids must be unique")
    if frequency_hz.shape != (n_frequencies,):
        raise ValueError("frequency_hz must have shape (n_frequencies,)")
    if not np.all(np.isfinite(frequency_hz)):
        raise ValueError("frequency_hz contains non-finite values")
    if np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError("frequency_hz must be strictly increasing")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if lambda_gain_smooth < 0.0:
        raise ValueError("lambda_gain_smooth cannot be negative")
    if lambda_noise_smooth < 0.0:
        raise ValueError("lambda_noise_smooth cannot be negative")
    if lambda_gain_reference < 0.0:
        raise ValueError("lambda_gain_reference cannot be negative")
    if lambda_noise_reference < 0.0:
        raise ValueError("lambda_noise_reference cannot be negative")
    if reliable_weight_boost <= 0.0:
        raise ValueError("reliable_weight_boost must be strictly positive")
    if max_correction_db is not None and max_correction_db <= 0.0:
        raise ValueError("max_correction_db must be strictly positive when provided")
    if low_information_threshold_ratio <= 0.0:
        raise ValueError("low_information_threshold_ratio must be strictly positive")
    if not 0.0 < low_information_weight <= 1.0:
        raise ValueError("low_information_weight must lie in the interval (0, 1]")

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

    train_indices, test_indices = _resolve_experiment_split_indices(
        n_experiments=n_experiments,
        train_indices=train_indices,
        test_indices=test_indices,
    )

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
    # quantile of the observations as the starting additive floor. The solver
    # operates on residual log-gain corrections and a softplus noise parameter,
    # but the stored state remains in physically meaningful power units.
    log_correction_gain = np.zeros((n_sensors, n_frequencies), dtype=np.float64)
    gain_power = nominal_gain_power * np.exp(log_correction_gain)
    additive_noise_power = 0.10 * np.quantile(train_observations, 0.05, axis=1)
    noise_parameter = _inverse_softplus(additive_noise_power)
    latent_spectra_power = np.median(
        (train_observations - additive_noise_power[:, np.newaxis, :])
        / np.clip(gain_power[:, np.newaxis, :], _EPSILON, None),
        axis=0,
    )
    latent_spectra_power = np.clip(latent_spectra_power, _EPSILON, None)

    residuals = (
        train_observations
        - gain_power[:, np.newaxis, :] * latent_spectra_power[np.newaxis, :, :]
        - additive_noise_power[:, np.newaxis, :]
    )
    residual_variance_power2 = np.maximum(np.mean(residuals**2, axis=1), min_variance)

    second_difference = _second_difference_operator(frequency_hz)
    smooth_penalty = (second_difference.T @ second_difference).tocsc()
    correction_gain_power = np.exp(log_correction_gain)
    latent_variation_power2 = np.zeros(n_frequencies, dtype=np.float64)
    frequency_information_weight = np.ones(n_frequencies, dtype=np.float64)
    information_weight = np.ones((n_sensors, n_frequencies), dtype=np.float64)
    frequency_low_information_mask = np.zeros(n_frequencies, dtype=bool)
    low_information_mask = np.zeros((n_sensors, n_frequencies), dtype=bool)
    gain_at_correction_bound_mask = np.zeros((n_sensors, n_frequencies), dtype=bool)
    noise_zero_mask = np.zeros((n_sensors, n_frequencies), dtype=bool)
    solver_nonfinite_step_count = np.zeros(n_sensors, dtype=np.int64)

    objective_history = []

    for iteration_index in range(n_iterations):
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

        (
            latent_variation_power2,
            frequency_information_weight,
            frequency_low_information_mask,
        ) = _frequency_information_weights(
            latent_spectra_power=latent_spectra_power,
            low_information_threshold_ratio=low_information_threshold_ratio,
            low_information_weight=low_information_weight,
        )

        (
            information_weight,
            low_information_mask,
        ) = _sensor_information_weights(
            latent_spectra_power=latent_spectra_power,
            gain_power=gain_power,
            noise_parameter=noise_parameter,
            residual_variance_power2=residual_variance_power2,
            frequency_information_weight=frequency_information_weight,
            frequency_low_information_mask=frequency_low_information_mask,
            low_information_weight=low_information_weight,
        )

        raw_log_correction = np.empty_like(log_correction_gain)
        raw_noise_parameter = np.empty_like(noise_parameter)
        raw_gain_at_bound_mask = np.zeros_like(gain_at_correction_bound_mask)

        # Update each sensor with a single frequency-coupled sparse solve around
        # the current nonlinear parameterization. This keeps the scalability of
        # the original joint system while moving the smoothness prior to the
        # correct log-gain geometry.
        for sensor_index in range(n_sensors):
            gain_reference_weight = lambda_gain_reference
            log_correction_target = log_correction_gain[sensor_index].copy()
            if sensor_index == reliable_sensor_index:
                total_reference_weight = gain_reference_weight + lambda_reliable_anchor
                if total_reference_weight > 0.0:
                    log_correction_target = (
                        gain_reference_weight * log_correction_target
                        + lambda_reliable_anchor * 0.0
                    ) / total_reference_weight
                    gain_reference_weight = total_reference_weight

            (
                log_correction_fit,
                noise_parameter_fit,
                sensor_gain_at_bound_mask,
                reused_previous_iterate,
            ) = _fit_sensor_joint_curves(
                latent_spectra_power=latent_spectra_power,
                observations_power=train_observations[sensor_index],
                nominal_gain_power=nominal_gain_power[sensor_index],
                log_correction_linearization=log_correction_gain[sensor_index],
                log_correction_target=log_correction_target,
                noise_parameter_linearization=noise_parameter[sensor_index],
                noise_parameter_target=noise_parameter[sensor_index],
                observation_weight=weights[sensor_index],
                information_weight=information_weight[sensor_index],
                smooth_penalty=smooth_penalty,
                lambda_gain_smooth=lambda_gain_smooth,
                lambda_noise_smooth=lambda_noise_smooth,
                lambda_gain_reference=gain_reference_weight,
                lambda_noise_reference=lambda_noise_reference,
                max_log_correction=max_log_correction,
            )
            if reused_previous_iterate:
                solver_nonfinite_step_count[sensor_index] += 1
                warnings.warn(
                    "Sparse Gauss-Newton solve returned non-finite values for "
                    f"sensor {sensor_ids[sensor_index]!r} at alternating "
                    f"iteration {iteration_index + 1}; the previous iterate was "
                    "reused for that sensor.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raw_log_correction[sensor_index] = log_correction_fit
            raw_noise_parameter[sensor_index] = noise_parameter_fit
            raw_gain_at_bound_mask[sensor_index] = sensor_gain_at_bound_mask

        projected_log_correction = _project_log_corrections(
            raw_log_correction,
            max_log_correction=max_log_correction,
        )
        common_log_shift = np.mean(
            raw_log_correction - projected_log_correction,
            axis=0,
            keepdims=True,
        )
        log_correction_gain = projected_log_correction
        noise_parameter = raw_noise_parameter
        correction_gain_power = np.exp(log_correction_gain)
        gain_power = nominal_gain_power * correction_gain_power
        additive_noise_power = _softplus(noise_parameter)

        # Enforce the identifiability constraint by moving the common frequency
        # scale that was removed from the gain curves into the latent spectrum.
        latent_spectra_power = latent_spectra_power * np.exp(common_log_shift[0])
        correction_gain_power = np.exp(log_correction_gain)
        gain_power = nominal_gain_power * correction_gain_power

        if max_log_correction is None:
            gain_at_correction_bound_mask = raw_gain_at_bound_mask
        else:
            gain_at_correction_bound_mask = raw_gain_at_bound_mask | np.isclose(
                np.abs(log_correction_gain),
                max_log_correction,
                rtol=0.0,
                atol=1.0e-8,
            )
        noise_zero_mask = additive_noise_power <= _EPSILON

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
                log_correction_gain=log_correction_gain,
                noise_parameter=noise_parameter,
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
        latent_variation_power2=latent_variation_power2,
        frequency_information_weight=frequency_information_weight,
        information_weight=information_weight,
        frequency_low_information_mask=frequency_low_information_mask,
        low_information_mask=low_information_mask,
        gain_at_correction_bound_mask=gain_at_correction_bound_mask,
        noise_zero_mask=noise_zero_mask,
        solver_nonfinite_step_count=solver_nonfinite_step_count,
    )


def resolve_spectral_fit_config(
    fit_config: Mapping[str, int | float | None] | None = None,  # Optional overrides
) -> dict[str, int | float | None]:  # Fully resolved fitter configuration
    """Return the full fitter configuration after applying explicit overrides.

    This helper makes artifact manifests audit-friendly by expanding partial
    configuration dictionaries to the exact keyword arguments consumed by
    :func:`fit_spectral_calibration`.
    """

    fit_config = {} if fit_config is None else dict(fit_config)
    unknown_keys = sorted(set(fit_config) - set(_SPECTRAL_FIT_CONFIG_PARAMETER_NAMES))
    if unknown_keys:
        raise ValueError(
            f"fit_config contains unsupported spectral fitter keys: {unknown_keys}."
        )

    signature = inspect.signature(fit_spectral_calibration)
    resolved_fit_config: dict[str, int | float | None] = {}
    for name in _SPECTRAL_FIT_CONFIG_PARAMETER_NAMES:
        default_value = signature.parameters[name].default
        raw_value = fit_config.get(name, default_value)
        if raw_value is None:
            if default_value is not None:
                resolved_fit_config[name] = None
            else:
                resolved_fit_config[name] = None
            continue
        if isinstance(default_value, (int, np.integer)) and not isinstance(
            default_value, bool
        ):
            resolved_fit_config[name] = int(raw_value)
        else:
            resolved_fit_config[name] = float(raw_value)
    return resolved_fit_config


def apply_deployed_calibration(
    observations_power: FloatArray,  # Observed PSD values with shape (sensors, ..., frequencies) or (frequencies,) for a single sensor
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
        New PSD observations. The general shape is
        ``(n_sensors, ..., n_frequencies)`` where the first axis is the sensor
        axis and the last axis is frequency. A one-dimensional input is accepted
        only for the single-sensor case ``(n_frequencies,)``.
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

    if gain_power.ndim != 2:
        raise ValueError("gain_power must have shape (sensors, frequencies)")
    if gain_power.shape != additive_noise_power.shape:
        raise ValueError(
            "gain_power and additive_noise_power must share the same shape"
        )
    if observations_power.ndim == 0:
        raise ValueError("observations_power must have at least one dimension")

    if observations_power.ndim == 1:
        if gain_power.shape[0] != 1:
            raise ValueError(
                "One-dimensional observations_power is only valid for a single sensor"
            )
        if observations_power.shape[0] != gain_power.shape[1]:
            raise ValueError(
                "The frequency axis of observations_power must match gain_power"
            )
        gain_view = gain_power[0]
        noise_view = additive_noise_power[0]
    else:
        if observations_power.shape[0] != gain_power.shape[0]:
            raise ValueError(
                "The first axis of observations_power must match the sensor axis"
            )
        if observations_power.shape[-1] != gain_power.shape[1]:
            raise ValueError(
                "The last axis of observations_power must match the frequency axis"
            )

        # Keep the sensor axis explicit and broadcast the calibration curves
        # across any intermediate deployment dimensions.
        curve_shape = (
            (gain_power.shape[0],)
            + (1,) * (observations_power.ndim - 2)
            + (gain_power.shape[1],)
        )
        gain_view = np.reshape(gain_power, curve_shape)
        noise_view = np.reshape(additive_noise_power, curve_shape)

    corrected = (observations_power - noise_view) / np.clip(gain_view, _EPSILON, None)
    if enforce_nonnegative:
        corrected = np.clip(corrected, 0.0, None)
    return corrected


def compute_network_consensus(
    corrected_power: FloatArray,  # Corrected PSD tensor with shape (sensors, experiments, frequencies)
    residual_variance_power2: FloatArray,  # Sensor reliability variances with shape (sensors, frequencies)
    valid_mask: NDArray[np.bool_] | None = None,  # Optional per-sensor validity mask
) -> FloatArray:  # Weighted consensus PSD with shape (experiments, frequencies)
    """Fuse corrected common-field spectra into a reliability-weighted consensus.

    Parameters
    ----------
    corrected_power:
        Corrected PSD tensor with shape ``(n_sensors, n_experiments, n_frequencies)``.
    residual_variance_power2:
        Sensor reliability variances with shape ``(n_sensors, n_frequencies)``.
    valid_mask:
        Optional Boolean mask with the same shape as ``corrected_power``. When
        provided, invalid sensor bins are excluded from the weighted average
        instead of being allowed to pull the consensus toward zero. Even when
        no explicit mask is supplied, non-finite and non-positive corrected
        powers are treated as invalid because deployment clipping can map
        ``Y <= N`` bins to exact zeros that are not physically meaningful
        consensus votes.

    Returns
    -------
    FloatArray
        Weighted consensus PSD with shape ``(n_experiments, n_frequencies)``.
        Bins with no valid contributing sensors are returned as ``NaN`` so
        downstream plots and diagnostics can display them as undefined instead
        of as artificial deep notches.
    """

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
    if valid_mask is None:
        valid_mask_array = np.isfinite(corrected_power) & (corrected_power > 0.0)
    else:
        valid_mask_array = np.asarray(valid_mask, dtype=bool)
        if valid_mask_array.shape != corrected_power.shape:
            raise ValueError("valid_mask must have the same shape as corrected_power")
        valid_mask_array = (
            valid_mask_array & np.isfinite(corrected_power) & (corrected_power > 0.0)
        )

    weights = 1.0 / np.clip(residual_variance_power2, _EPSILON, None)
    effective_weight = weights[:, np.newaxis, :] * valid_mask_array
    numerator = np.sum(effective_weight * corrected_power, axis=0)
    denominator = np.sum(effective_weight, axis=0)
    consensus = np.full(corrected_power.shape[1:], np.nan, dtype=np.float64)
    valid_consensus_mask = denominator > 0.0
    consensus[valid_consensus_mask] = (
        numerator[valid_consensus_mask] / denominator[valid_consensus_mask]
    )
    return consensus


def _resolve_experiment_split_indices(
    n_experiments: int,  # Total number of available experiments
    train_indices: IndexArray | None,  # Requested training indices
    test_indices: IndexArray | None,  # Requested held-out indices
) -> tuple[IndexArray, IndexArray]:  # Validated train and test index arrays
    """Resolve and validate the train/test experiment split.

    The fitter accepts explicit train and test indices so notebook workflows
    can control chronological or grouped validation. This helper keeps that
    flexibility while enforcing the minimum contract required for trustworthy
    diagnostics: both sets must be one-dimensional, in bounds, unique, and
    mutually disjoint.
    """

    all_indices = np.arange(n_experiments, dtype=np.int64)
    resolved_train_indices = (
        all_indices
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64)
    )
    _validate_experiment_index_array(
        indices=resolved_train_indices,
        n_experiments=n_experiments,
        name="train_indices",
        allow_empty=False,
    )

    resolved_test_indices = (
        np.setdiff1d(all_indices, resolved_train_indices, assume_unique=False)
        if test_indices is None
        else np.asarray(test_indices, dtype=np.int64)
    )
    _validate_experiment_index_array(
        indices=resolved_test_indices,
        n_experiments=n_experiments,
        name="test_indices",
        allow_empty=True,
    )

    overlapping_indices = np.intersect1d(
        resolved_train_indices,
        resolved_test_indices,
        assume_unique=False,
    )
    if overlapping_indices.size != 0:
        raise ValueError(
            "train_indices and test_indices must be disjoint, but overlap at "
            f"{overlapping_indices.tolist()}."
        )
    return resolved_train_indices, resolved_test_indices


def _validate_experiment_index_array(
    indices: IndexArray,  # Candidate experiment indices
    n_experiments: int,  # Exclusive upper bound on valid indices
    name: str,  # Human-readable array name for error messages
    allow_empty: bool,  # Whether zero-length arrays are acceptable
) -> None:  # Raises if the index array breaks the split contract
    """Validate one train/test index array against the experiment count."""

    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and indices.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must be unique")
    if indices.size == 0:
        return
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= n_experiments:
        raise ValueError(
            f"{name} must lie within [0, {n_experiments - 1}], received "
            f"{indices.tolist()}."
        )


def _fit_sensor_joint_curves(
    latent_spectra_power: FloatArray,  # Latent spectra with shape (experiments, frequencies)
    observations_power: FloatArray,  # Sensor observations with shape (experiments, frequencies)
    nominal_gain_power: FloatArray,  # Nominal multiplicative response for the sensor
    log_correction_linearization: FloatArray,  # Current residual log-gain iterate used for linearization
    log_correction_target: FloatArray,  # Conservative target for the residual log-gain correction
    noise_parameter_linearization: FloatArray,  # Current unconstrained noise parameter used for linearization
    noise_parameter_target: FloatArray,  # Conservative target for the noise parameter
    observation_weight: FloatArray,  # Residual-variance weights for this sensor
    information_weight: FloatArray,  # Sensor-specific identifiability weights for each frequency bin
    smooth_penalty: sparse.csc_matrix,  # Second-difference Gram matrix over frequency
    lambda_gain_smooth: float,  # Log-gain smoothness penalty
    lambda_noise_smooth: float,  # Noise-parameter smoothness penalty
    lambda_gain_reference: float,  # Damping toward the previous log correction
    lambda_noise_reference: float,  # Damping toward the previous noise parameter
    max_log_correction: float
    | None,  # Optional symmetric log bound on the residual gain correction
) -> tuple[
    FloatArray, FloatArray, NDArray[np.bool_], bool
]:  # Updated log-gain correction, noise parameter, cap mask, and fallback flag
    """Solve one conservative frequency-coupled gain/noise update for a sensor.

    The state is nonlinear in both the multiplicative gain and the positive
    additive floor, so this helper performs one sparse Gauss-Newton step around
    the current iterate. Frequencies remain coupled through the second-
    difference penalties, while low-information bins are downweighted via
    ``information_weight`` so the update falls back to the conservative
    references instead of producing unstable boundary solutions.
    """

    n_experiments, n_frequencies = latent_spectra_power.shape
    effective_weight = observation_weight * information_weight
    gain_power = nominal_gain_power * np.exp(log_correction_linearization)
    noise_power = _softplus(noise_parameter_linearization)
    noise_jacobian = _sigmoid(noise_parameter_linearization)

    # Linearize the nonlinear model around the current iterate:
    # y ≈ g_ref * S + n_ref + (g_ref * S) * δ_gamma + sigmoid(eta_ref) * δ_eta.
    gain_design = gain_power[np.newaxis, :] * latent_spectra_power
    residual_rhs = observations_power - gain_design - noise_power[np.newaxis, :]

    # The normal equations remain sparse because the data term is diagonal in
    # frequency and only the smoothness penalty couples neighboring bins.
    gain_data_diagonal = effective_weight * np.sum(gain_design**2, axis=0)
    gain_noise_cross = effective_weight * noise_jacobian * np.sum(gain_design, axis=0)
    noise_data_diagonal = effective_weight * float(n_experiments) * noise_jacobian**2
    gain_rhs = effective_weight * np.sum(gain_design * residual_rhs, axis=0)
    noise_rhs = effective_weight * noise_jacobian * np.sum(residual_rhs, axis=0)

    gain_block = sparse.diags(gain_data_diagonal + lambda_gain_reference, format="csc")
    noise_block = sparse.diags(
        noise_data_diagonal + lambda_noise_reference, format="csc"
    )
    cross_block = sparse.diags(gain_noise_cross, format="csc")

    if smooth_penalty.shape[0] != 0:
        gain_block = gain_block + lambda_gain_smooth * smooth_penalty
        noise_block = noise_block + lambda_noise_smooth * smooth_penalty
        gain_rhs = gain_rhs - lambda_gain_smooth * (
            smooth_penalty @ log_correction_linearization
        )
        noise_rhs = noise_rhs - lambda_noise_smooth * (
            smooth_penalty @ noise_parameter_linearization
        )

    system_matrix = sparse.bmat(
        [[gain_block, cross_block], [cross_block, noise_block]], format="csc"
    )
    system_rhs = np.concatenate(
        [
            gain_rhs
            + lambda_gain_reference
            * (log_correction_target - log_correction_linearization),
            noise_rhs
            + lambda_noise_reference
            * (noise_parameter_target - noise_parameter_linearization),
        ]
    )
    solution = spsolve(system_matrix, system_rhs)
    if not np.all(np.isfinite(solution)):
        empty_mask = np.zeros(n_frequencies, dtype=bool)
        return (
            log_correction_linearization.copy(),
            noise_parameter_linearization.copy(),
            empty_mask,
            True,
        )

    delta_log_correction = np.asarray(solution[:n_frequencies], dtype=np.float64)
    delta_noise_parameter = np.asarray(solution[n_frequencies:], dtype=np.float64)
    log_correction = log_correction_linearization + delta_log_correction
    noise_parameter = noise_parameter_linearization + delta_noise_parameter

    if max_log_correction is None:
        gain_at_bound_mask = np.zeros(n_frequencies, dtype=bool)
    else:
        log_correction = np.clip(
            log_correction, -max_log_correction, max_log_correction
        )
        gain_at_bound_mask = np.isclose(
            np.abs(log_correction),
            max_log_correction,
            rtol=0.0,
            atol=1.0e-8,
        )
    return log_correction, noise_parameter, gain_at_bound_mask, False


def _frequency_information_weights(
    latent_spectra_power: FloatArray,  # Latent spectra with shape (experiments, frequencies)
    low_information_threshold_ratio: float,  # Threshold relative to the median informative variance
    low_information_weight: float,  # Minimum data weight for weakly informative bins
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Estimate which frequency bins are poorly identified by the calibration set.

    The affine separation between multiplicative gain and additive noise relies
    on variation in the latent spectra across calibration experiments. When that
    variation is very small, many gain/noise pairs explain the data nearly
    equally well, so the sensor update must lean on regularization instead of
    trusting the raw affine decomposition.
    """

    latent_variation_power2 = np.var(latent_spectra_power, axis=0)
    positive_variation = latent_variation_power2[latent_variation_power2 > _EPSILON]
    if positive_variation.size == 0:
        frequency_information_weight = np.full(
            latent_variation_power2.shape, low_information_weight
        )
        frequency_low_information_mask = np.ones(
            latent_variation_power2.shape, dtype=bool
        )
        return (
            latent_variation_power2,
            frequency_information_weight,
            frequency_low_information_mask,
        )

    variance_threshold = max(
        float(np.median(positive_variation)) * low_information_threshold_ratio,
        _EPSILON,
    )
    frequency_information_weight = np.clip(
        latent_variation_power2 / variance_threshold,
        low_information_weight,
        1.0,
    )
    frequency_low_information_mask = latent_variation_power2 < variance_threshold
    return (
        latent_variation_power2,
        frequency_information_weight,
        frequency_low_information_mask,
    )


def _sensor_information_weights(
    latent_spectra_power: FloatArray,  # Latent spectra with shape (experiments, frequencies)
    gain_power: FloatArray,  # Current gain curves with shape (sensors, frequencies)
    noise_parameter: FloatArray,  # Current unconstrained noise parameter
    residual_variance_power2: FloatArray,  # Sensor residual variances with shape (sensors, frequencies)
    frequency_information_weight: FloatArray,  # Network-wide latent-variation weights
    frequency_low_information_mask: NDArray[np.bool_],  # Global low-information mask
    low_information_weight: float,  # Minimum data weight for weakly informative bins
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Refine global identifiability diagnostics into sensor-specific weights.

    The global latent-variation diagnostic is necessary but not sufficient:
    one sensor can still be poorly conditioned at a bin because its local gain
    scale, noise sensitivity, or residual variance makes the affine
    gain-versus-noise decomposition fragile. This helper combines three terms:

    - the frequency-only latent-variation weight;
    - a local affine-design balance term based on the columns ``[G_s S_k, 1]``;
    - a sensor-specific residual-SNR term.
    """

    n_sensors, n_frequencies = gain_power.shape
    n_experiments = latent_spectra_power.shape[0]
    latent_centered = latent_spectra_power - np.mean(
        latent_spectra_power, axis=0, keepdims=True
    )

    # The affine design is identifiable only when both the varying signal term
    # and the intercept-sensitive noise term carry comparable energy.
    centered_signal_energy = gain_power**2 * np.sum(
        latent_centered**2, axis=0, keepdims=True
    )
    intercept_energy = float(n_experiments) * _sigmoid(noise_parameter) ** 2
    design_balance = np.minimum(centered_signal_energy, intercept_energy) / np.clip(
        np.maximum(centered_signal_energy, intercept_energy),
        _EPSILON,
        None,
    )

    # Downweight sensors whose modeled latent variation is small relative to
    # their current residual variance, since those bins are still effectively
    # dominated by mismatch or noise.
    signal_variation_power2 = gain_power**2 * np.var(
        latent_spectra_power, axis=0, keepdims=True
    )
    residual_snr = signal_variation_power2 / np.clip(
        signal_variation_power2 + residual_variance_power2,
        _EPSILON,
        None,
    )

    positive_design_balance = design_balance[design_balance > _EPSILON]
    design_reference = max(
        _EPSILON
        if positive_design_balance.size == 0
        else float(np.median(positive_design_balance)),
        _EPSILON,
    )
    positive_residual_snr = residual_snr[residual_snr > _EPSILON]
    residual_reference = max(
        _EPSILON
        if positive_residual_snr.size == 0
        else float(np.median(positive_residual_snr)),
        _EPSILON,
    )
    normalized_design_balance = np.clip(design_balance / design_reference, 0.0, 1.0)
    normalized_residual_snr = np.clip(residual_snr / residual_reference, 0.0, 1.0)

    raw_information = (
        frequency_information_weight[np.newaxis, :]
        * normalized_design_balance
        * normalized_residual_snr
    )
    information_weight = np.clip(raw_information, low_information_weight, 1.0)
    # The notebook uses this mask for visual warnings, so keep the threshold
    # high enough to mark bins that are merely marginal, not only catastrophic.
    low_information_mask = frequency_low_information_mask[np.newaxis, :] | (
        information_weight <= max(low_information_weight, 0.60)
    )

    if information_weight.shape != (n_sensors, n_frequencies):
        raise ValueError("sensor-specific information weights have an unexpected shape")
    return information_weight, low_information_mask


def _second_difference_operator(
    frequency_hz: FloatArray,  # Strictly increasing frequency grid [Hz]
) -> sparse.csc_matrix:  # Sparse finite-difference matrix
    """Build a curvature operator that respects non-uniform frequency spacing.

    The operator is built on a frequency axis normalized by the median bin
    spacing. That keeps the smoothness hyperparameters on roughly the same
    scale as the original unit-grid implementation while still respecting
    irregular spacing when the acquisition grid is not uniform.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    n_frequencies = frequency_hz.size
    if n_frequencies < 3:
        return sparse.csc_matrix((0, n_frequencies))

    spacing_hz = np.diff(frequency_hz)
    median_spacing_hz = float(np.median(spacing_hz))
    normalized_frequency = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            np.cumsum(spacing_hz / median_spacing_hz),
        ]
    )

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for left_index in range(n_frequencies - 2):
        spacing_left = float(
            normalized_frequency[left_index + 1] - normalized_frequency[left_index]
        )
        spacing_right = float(
            normalized_frequency[left_index + 2] - normalized_frequency[left_index + 1]
        )
        denominator = spacing_left + spacing_right
        row_coefficients = (
            2.0 / (spacing_left * denominator),
            -2.0 / (spacing_left * spacing_right),
            2.0 / (spacing_right * denominator),
        )
        for offset, coefficient in enumerate(row_coefficients):
            row_indices.append(left_index)
            column_indices.append(left_index + offset)
            values.append(coefficient)

    return sparse.csc_matrix(
        (values, (row_indices, column_indices)),
        shape=(n_frequencies - 2, n_frequencies),
    )


def _penalized_objective(
    residuals: FloatArray,  # Training residual tensor
    residual_variance_power2: FloatArray,  # Estimated residual variances
    log_correction_gain: FloatArray,  # Residual log-gain correction
    noise_parameter: FloatArray,  # Unconstrained softplus noise parameter
    reliable_sensor_index: int,  # Index of the softly anchored sensor
    second_difference: sparse.csc_matrix,  # Finite-difference operator
    lambda_gain_smooth: float,  # Log-gain smoothness weight
    lambda_noise_smooth: float,  # Noise-parameter smoothness weight
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
        gain_penalty = float(np.sum((second_difference @ log_correction_gain.T) ** 2))
        noise_penalty = float(np.sum((second_difference @ noise_parameter.T) ** 2))

    reliable_penalty = float(np.sum(log_correction_gain[reliable_sensor_index] ** 2))
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


def _project_log_corrections(
    log_correction_gain: FloatArray,  # Residual log-gain corrections
    max_log_correction: float | None,  # Optional symmetric absolute bound
) -> FloatArray:  # Mean-zero corrections that respect the optional box bound
    """Project log-gain corrections onto the identifiability and cap constraints.

    The calibration model is identifiable only up to a common multiplicative
    frequency scale, so the residual log-gain curves are projected to zero mean
    across sensors at each frequency. When a correction cap is configured, the
    projection also enforces the symmetric box constraint in log space.
    """

    log_correction_gain = np.asarray(log_correction_gain, dtype=np.float64)
    if max_log_correction is None:
        return log_correction_gain - np.mean(log_correction_gain, axis=0, keepdims=True)

    lower_offset = (
        np.min(log_correction_gain, axis=0, keepdims=True) - max_log_correction
    )
    upper_offset = (
        np.max(log_correction_gain, axis=0, keepdims=True) + max_log_correction
    )

    # The clipped mean is monotone in the scalar offset, so a vectorized
    # bisection finds the unique mean-zero projection at each frequency.
    for _ in range(48):
        midpoint = 0.5 * (lower_offset + upper_offset)
        clipped = np.clip(
            log_correction_gain - midpoint,
            -max_log_correction,
            max_log_correction,
        )
        positive_mean_mask = np.mean(clipped, axis=0, keepdims=True) > 0.0
        lower_offset = np.where(positive_mean_mask, midpoint, lower_offset)
        upper_offset = np.where(positive_mean_mask, upper_offset, midpoint)

    offset = 0.5 * (lower_offset + upper_offset)
    return np.clip(
        log_correction_gain - offset,
        -max_log_correction,
        max_log_correction,
    )


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
