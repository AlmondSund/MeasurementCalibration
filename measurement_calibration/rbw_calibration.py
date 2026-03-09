"""RBW-specific adapters and QC helpers for the spectral calibration workflow.

This module keeps the RBW calibration path structurally aligned with the
existing common-field calibration pipeline:

- :mod:`measurement_calibration.sensor_ranking` remains responsible for loading
  the row-aligned RBW acquisitions and for computing the ranking diagnostics
  that justify sensor-selection decisions.
- :mod:`measurement_calibration.spectral_calibration` remains responsible for
  the actual latent-variable calibration fit.
- This module adapts RBW acquisitions into the calibration-core data contract,
  applies explicit per-RBW exclusions, evaluates hold-out behavior, and can
  retrain after automatically removing clear post-fit sensor outliers.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import cast

import numpy as np

from .artifacts import SavedCalibrationArtifact, save_spectral_calibration_artifact
from .sensor_ranking import (
    RbwAcquisitionDataset,
    SensorDistributionDiagnostics,
    SensorRankingResult,
    build_distribution_summary_rows,
    load_rbw_acquisition_datasets,
    rank_sensors_by_cumulative_correlation,
    summarize_psd_distribution,
)
from .spectral_calibration import (
    CalibrationDataset,
    SpectralCalibrationResult,
    apply_deployed_calibration,
    compute_network_consensus,
    fit_spectral_calibration,
    make_holdout_split,
    power_db_to_linear,
    power_linear_to_db,
    resolve_spectral_fit_config,
)


DEFAULT_RBW_EXCLUDED_SENSOR_IDS = ("Node9",)
DEFAULT_RBW_EXCLUDED_SENSOR_IDS_BY_LABEL = {"10K": ("Node5",)}
DEFAULT_RBW_INVALID_CORRECTED_FRACTION_THRESHOLD = 1.0e-3
DEFAULT_RBW_RMSE_OUTLIER_IQR_MULTIPLIER = 1.5
DEFAULT_RBW_MIN_RMSE_EXCESS_DB = 1.0


@dataclass(frozen=True)
class RbwCalibrationPreparation:
    """Prepared RBW subset ready for calibration fitting.

    Parameters
    ----------
    rbw_label:
        RBW subset identifier such as ``"10K"``.
    source_dataset:
        Original parsed RBW subset before any manual or QC-driven exclusions.
        Keeping the source dataset explicit makes a second-pass retrain
        deterministic because the preparation can be rebuilt without relying on
        hidden global state.
    calibration_dataset:
        RBW observations converted into the generic calibration-dataset shape
        expected by :func:`fit_spectral_calibration`.
    ranking_result:
        Record-wise cumulative-correlation diagnostic computed after the final
        manual exclusions for this preparation pass.
    distribution_diagnostics:
        Campaign-wide PSD-distribution diagnostic computed on the same retained
        sensors.
    reliable_sensor_id:
        Sensor chosen as the soft anchor. The selection rule is intentionally
        conservative: prefer the best record-wise sensor that is not flagged as
        a distribution-shape outlier, then fall back to the record-wise winner
        if every retained sensor is an outlier.
    excluded_sensor_ids:
        Sensor identifiers explicitly removed before fitting this preparation.
    distribution_outlier_sensor_ids:
        Retained sensors whose PSD distributions still disagree with the rest
        of the campaign according to the histogram diagnostic.
    ranking_histogram_bins, distribution_histogram_bins:
        Notebook-derived ranking configuration kept so a QC retrain can rebuild
        the preparation with the same diagnostics settings.
    """

    rbw_label: str
    source_dataset: RbwAcquisitionDataset
    calibration_dataset: CalibrationDataset
    ranking_result: SensorRankingResult
    distribution_diagnostics: SensorDistributionDiagnostics
    reliable_sensor_id: str
    excluded_sensor_ids: tuple[str, ...]
    distribution_outlier_sensor_ids: tuple[str, ...]
    ranking_histogram_bins: int
    distribution_histogram_bins: int


@dataclass(frozen=True)
class RbwSensorValidationSummary:
    """Hold-out validation metrics for one retained RBW sensor.

    Parameters
    ----------
    sensor_id:
        Sensor identifier matching the fitted result.
    mean_bias_to_consensus_db:
        Average residual offset versus the hold-out consensus [dB].
    rmse_to_consensus_db:
        Root-mean-square residual versus the hold-out consensus [dB].
    invalid_corrected_fraction:
        Fraction of hold-out bins where the deployed correction would require
        subtracting more additive noise than the observed power, i.e.
        ``Y <= N`` before the nonnegativity truncation.
    median_information_weight:
        Median sensor information weight used during fitting.
    low_information_fraction:
        Fraction of fitted bins that remain low-information for this sensor.
    gain_cap_fraction:
        Fraction of bins where the fitted gain correction hits the configured
        cap.
    alignment_median_error_ms:
        Median timing mismatch between the sensor and the campaign median
        timestamps [ms].
    """

    sensor_id: str
    mean_bias_to_consensus_db: float
    rmse_to_consensus_db: float
    invalid_corrected_fraction: float
    median_information_weight: float
    low_information_fraction: float
    gain_cap_fraction: float
    alignment_median_error_ms: float


@dataclass(frozen=True)
class RbwCalibrationValidationSummary:
    """Compact hold-out validation summary for one RBW fit."""

    raw_mean_sensor_std_db: float
    corrected_mean_sensor_std_db: float
    corrected_to_raw_dispersion_ratio: float
    sensor_rows: tuple[RbwSensorValidationSummary, ...]


@dataclass(frozen=True)
class RbwCalibrationFitResult:
    """Stored RBW model plus the fitted validation and QC metadata.

    Parameters
    ----------
    preparation:
        Final RBW preparation used for the stored fit. If automatic QC retrain
        removes sensors, this field reflects the final retained subset.
    artifact:
        Saved artifact bundle on disk.
    validation:
        Compact hold-out validation summary for the final stored fit.
    fit_duration_s:
        Total wall-clock fitting time across every training pass [s].
    qc_auto_excluded_sensor_ids:
        Sensors removed automatically after the initial fit due to clear
        hold-out QC failures.
    n_fit_passes:
        Number of fit passes used to produce the final artifact.
    """

    preparation: RbwCalibrationPreparation
    artifact: SavedCalibrationArtifact
    validation: RbwCalibrationValidationSummary
    fit_duration_s: float
    qc_auto_excluded_sensor_ids: tuple[str, ...]
    n_fit_passes: int

    @property
    def raw_mean_sensor_std_db(self) -> float:
        """Return the final raw hold-out mean sensor dispersion [dB]."""

        return self.validation.raw_mean_sensor_std_db

    @property
    def corrected_mean_sensor_std_db(self) -> float:
        """Return the final corrected hold-out mean sensor dispersion [dB]."""

        return self.validation.corrected_mean_sensor_std_db

    @property
    def corrected_to_raw_dispersion_ratio(self) -> float:
        """Return the final corrected-to-raw dispersion ratio."""

        return self.validation.corrected_to_raw_dispersion_ratio


def exclude_rbw_sensors(
    dataset: RbwAcquisitionDataset,  # One RBW subset to filter
    excluded_sensor_ids: Collection[str],  # Sensor identifiers to remove
) -> RbwAcquisitionDataset:  # Filtered copy that preserves record alignment
    """Return a filtered RBW dataset without the requested sensors."""

    excluded_sensor_id_set = {
        str(sensor_id) for sensor_id in excluded_sensor_ids if str(sensor_id)
    }
    retained_indices = [
        sensor_index
        for sensor_index, sensor_id in enumerate(dataset.sensor_ids)
        if sensor_id not in excluded_sensor_id_set
    ]
    if len(retained_indices) < 2:
        raise ValueError(
            "RBW calibration requires at least two retained sensors after "
            f"exclusions, but {dataset.rbw_label} keeps {len(retained_indices)}."
        )

    return RbwAcquisitionDataset(
        rbw_label=dataset.rbw_label,
        sensor_ids=tuple(dataset.sensor_ids[index] for index in retained_indices),
        frequency_hz=np.asarray(dataset.frequency_hz, dtype=np.float64),
        observations_db=np.asarray(
            dataset.observations_db[retained_indices],
            dtype=np.float64,
        ),
        timestamps_ms=np.asarray(
            dataset.timestamps_ms[retained_indices],
            dtype=np.int64,
        ),
    )


def resolve_rbw_excluded_sensor_ids(
    rbw_label: str,  # RBW subset label such as ``10K``
    excluded_sensor_ids: Collection[str],  # Global exclusions applied to every RBW
    excluded_sensor_ids_by_label: Mapping[str, Collection[str]]
    | None = None,  # Optional per-RBW exclusions
) -> tuple[str, ...]:  # Combined explicit exclusions for the requested RBW
    """Combine global and per-RBW explicit exclusions into one stable tuple.

    Parameters
    ----------
    rbw_label:
        RBW subset label such as ``"10K"``.
    excluded_sensor_ids:
        Global sensor exclusions applied to every RBW campaign.
    excluded_sensor_ids_by_label:
        Optional campaign-specific exclusions. When omitted, the repository
        defaults are applied so the standard RBW workflow remains safe.

    Returns
    -------
    tuple[str, ...]
        Stable sorted exclusions for the requested RBW subset.
    """

    if excluded_sensor_ids_by_label is None:
        excluded_sensor_ids_by_label = DEFAULT_RBW_EXCLUDED_SENSOR_IDS_BY_LABEL
    sensor_ids = {str(sensor_id) for sensor_id in excluded_sensor_ids if str(sensor_id)}
    if excluded_sensor_ids_by_label is not None:
        sensor_ids.update(
            str(sensor_id)
            for sensor_id in excluded_sensor_ids_by_label.get(rbw_label, ())
            if str(sensor_id)
        )
    return tuple(sorted(sensor_ids))


def prepare_rbw_calibration_dataset(
    dataset: RbwAcquisitionDataset,  # One RBW subset to adapt for calibration
    excluded_sensor_ids: Collection[str] = DEFAULT_RBW_EXCLUDED_SENSOR_IDS,
    excluded_sensor_ids_by_label: Mapping[str, Collection[str]] | None = None,
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
) -> RbwCalibrationPreparation:  # Prepared calibration inputs and diagnostics
    """Prepare one RBW subset for calibration fitting.

    The RBW acquisitions are already row-aligned by record index, so no
    sequence-shift search is needed here. The adapter therefore:

    1. combines global and per-RBW explicit exclusions,
    2. recomputes the ranking diagnostics on the retained subset,
    3. chooses a reliable anchor sensor using the combined record-wise and
       distribution-shape diagnostics, and
    4. converts the dB PSD tensor into the positive linear-power tensor
       required by :func:`fit_spectral_calibration`.

    Parameters
    ----------
    dataset:
        Parsed RBW subset with one PSD cube per retained sensor.
    excluded_sensor_ids:
        Global exclusions applied to every RBW subset.
    excluded_sensor_ids_by_label:
        Optional per-RBW exclusions. When omitted, the repository defaults are
        applied through :func:`resolve_rbw_excluded_sensor_ids`.
    ranking_histogram_bins, distribution_histogram_bins:
        Histogram resolutions used by the ranking notebook diagnostics.

    Returns
    -------
    RbwCalibrationPreparation
        Prepared dataset plus the diagnostics needed to explain the selection
        and fitting decisions.
    """

    # Resolve the explicit exclusions first so every downstream diagnostic sees
    # the exact sensor set that will be passed into the fitter.
    resolved_excluded_sensor_ids = resolve_rbw_excluded_sensor_ids(
        dataset.rbw_label,
        excluded_sensor_ids=excluded_sensor_ids,
        excluded_sensor_ids_by_label=excluded_sensor_ids_by_label,
    )
    filtered_dataset = exclude_rbw_sensors(dataset, resolved_excluded_sensor_ids)
    ranking_result = rank_sensors_by_cumulative_correlation(
        filtered_dataset,
        histogram_bins=ranking_histogram_bins,
    )
    distribution_diagnostics = summarize_psd_distribution(
        filtered_dataset,
        histogram_bins=distribution_histogram_bins,
    )

    distribution_rows = build_distribution_summary_rows(distribution_diagnostics)
    distribution_outlier_sensor_ids = tuple(
        str(row["sensor_id"])
        for row in distribution_rows
        if bool(row["is_low_similarity_outlier"])
    )
    reliable_sensor_id = _select_reliable_sensor_id(
        ranking_result=ranking_result,
        distribution_outlier_sensor_ids=distribution_outlier_sensor_ids,
    )

    # RBW records are compared row by row across sensors, so the record index is
    # already the alignment key. We summarize the campaign timestamp for each
    # record by the median across sensors to avoid picking one sensor as a
    # hidden temporal reference.
    experiment_timestamps_ms = np.median(
        filtered_dataset.timestamps_ms,
        axis=0,
    ).astype(np.int64)
    alignment_median_error_ms = {
        sensor_id: float(
            np.median(
                np.abs(
                    filtered_dataset.timestamps_ms[sensor_index]
                    - experiment_timestamps_ms
                )
            )
        )
        for sensor_index, sensor_id in enumerate(filtered_dataset.sensor_ids)
    }
    calibration_dataset = CalibrationDataset(
        sensor_ids=filtered_dataset.sensor_ids,
        frequency_hz=np.asarray(filtered_dataset.frequency_hz, dtype=np.float64),
        observations_power=power_db_to_linear(filtered_dataset.observations_db),
        nominal_gain_power=np.ones(
            (len(filtered_dataset.sensor_ids), filtered_dataset.n_frequencies),
            dtype=np.float64,
        ),
        experiment_timestamps_ms=experiment_timestamps_ms,
        selected_band_hz=(
            float(filtered_dataset.frequency_hz[0]),
            float(filtered_dataset.frequency_hz[-1]),
        ),
        sensor_shifts={sensor_id: 0 for sensor_id in filtered_dataset.sensor_ids},
        alignment_median_error_ms=alignment_median_error_ms,
        source_row_indices={
            sensor_id: np.arange(filtered_dataset.n_records, dtype=np.int64)
            for sensor_id in filtered_dataset.sensor_ids
        },
    )

    return RbwCalibrationPreparation(
        rbw_label=filtered_dataset.rbw_label,
        source_dataset=dataset,
        calibration_dataset=calibration_dataset,
        ranking_result=ranking_result,
        distribution_diagnostics=distribution_diagnostics,
        reliable_sensor_id=reliable_sensor_id,
        excluded_sensor_ids=tuple(
            sensor_id
            for sensor_id in resolved_excluded_sensor_ids
            if sensor_id in dataset.sensor_ids
        ),
        distribution_outlier_sensor_ids=distribution_outlier_sensor_ids,
        ranking_histogram_bins=int(ranking_histogram_bins),
        distribution_histogram_bins=int(distribution_histogram_bins),
    )


def load_rbw_calibration_preparations(
    root_dir: Path,  # Root directory with one subdirectory per RBW subset
    excluded_sensor_ids: Collection[str] = DEFAULT_RBW_EXCLUDED_SENSOR_IDS,
    excluded_sensor_ids_by_label: Mapping[str, Collection[str]] | None = None,
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
) -> dict[str, RbwCalibrationPreparation]:  # Prepared datasets indexed by RBW label
    """Load every RBW subset and adapt it for calibration fitting.

    Parameters
    ----------
    root_dir:
        Directory that contains one subdirectory per RBW campaign.
    excluded_sensor_ids:
        Global exclusions applied to every RBW subset.
    excluded_sensor_ids_by_label:
        Optional per-RBW exclusions. Repository defaults are used when omitted.
    ranking_histogram_bins, distribution_histogram_bins:
        Histogram resolutions forwarded to
        :func:`prepare_rbw_calibration_dataset`.

    Returns
    -------
    dict[str, RbwCalibrationPreparation]
        Prepared RBW subsets keyed by their label.
    """

    preparations: dict[str, RbwCalibrationPreparation] = {}
    for rbw_label, dataset in sorted(load_rbw_acquisition_datasets(root_dir).items()):
        preparations[rbw_label] = prepare_rbw_calibration_dataset(
            dataset,
            excluded_sensor_ids=excluded_sensor_ids,
            excluded_sensor_ids_by_label=excluded_sensor_ids_by_label,
            ranking_histogram_bins=ranking_histogram_bins,
            distribution_histogram_bins=distribution_histogram_bins,
        )
    return preparations


def build_rbw_preparation_rows(
    preparations: Mapping[str, RbwCalibrationPreparation],  # Prepared datasets by RBW
) -> list[dict[str, float | int | str]]:  # Table-friendly summary rows
    """Build a compact summary of the RBW preparation decisions.

    Parameters
    ----------
    preparations:
        Prepared RBW subsets keyed by campaign label.

    Returns
    -------
    list[dict[str, float | int | str]]
        Notebook-friendly rows describing retained sensors, exclusions, and the
        selected reliable anchor.
    """

    rows: list[dict[str, float | int | str]] = []
    for rbw_label in sorted(preparations):
        preparation = preparations[rbw_label]
        recordwise_winner = preparation.ranking_result.ranking_sensor_ids[0]
        distribution_winner = preparation.distribution_diagnostics.ranking_sensor_ids[0]
        rows.append(
            {
                "rbw": rbw_label,
                "sensor_ids": ", ".join(preparation.calibration_dataset.sensor_ids),
                "reliable_sensor_id": preparation.reliable_sensor_id,
                "recordwise_winner_sensor_id": recordwise_winner,
                "distribution_winner_sensor_id": distribution_winner,
                "distribution_outlier_sensor_ids": (
                    ", ".join(preparation.distribution_outlier_sensor_ids)
                    if preparation.distribution_outlier_sensor_ids
                    else ""
                ),
                "excluded_sensor_ids": ", ".join(preparation.excluded_sensor_ids),
                "n_sensors": len(preparation.calibration_dataset.sensor_ids),
                "n_records": int(
                    preparation.calibration_dataset.observations_power.shape[1]
                ),
                "n_frequencies": int(preparation.calibration_dataset.frequency_hz.size),
                "recordwise_best_mean_correlation": float(
                    np.max(preparation.ranking_result.average_correlation)
                ),
                "distribution_best_similarity": float(
                    np.max(
                        preparation.distribution_diagnostics.normalized_similarity_score
                    )
                ),
            }
        )
    return rows


def evaluate_rbw_calibration_holdout(
    preparation: RbwCalibrationPreparation,  # Final RBW preparation
    result: SpectralCalibrationResult,  # Fitted RBW calibration result
) -> RbwCalibrationValidationSummary:  # Compact hold-out validation metrics
    """Evaluate one RBW fit on its held-out records.

    The returned summary is intentionally small but decision-oriented:

    - overall raw versus corrected dispersion,
    - per-sensor RMSE and bias to a leave-one-sensor-out corrected consensus, and
    - the fraction of bins where the deployed correction is forced to clip
      because the fitted additive noise exceeds the observed power.

    Parameters
    ----------
    preparation:
        Final RBW preparation used to fit the model.
    result:
        Stored spectral calibration result for the same preparation.

    Returns
    -------
    RbwCalibrationValidationSummary
        Hold-out summary used for QC decisions and notebook reporting.
    """

    dataset = preparation.calibration_dataset
    # Evaluate the same deployment-time correction that will be applied to raw
    # RBW measurements, including the nonnegativity truncation in the adapter.
    raw_test_power = dataset.observations_power[:, result.test_indices, :]
    corrected_test_power = apply_deployed_calibration(
        observations_power=raw_test_power,
        gain_power=result.gain_power,
        additive_noise_power=result.additive_noise_power,
    )

    raw_test_db = power_linear_to_db(raw_test_power)
    corrected_test_db = power_linear_to_db(corrected_test_power)
    invalid_corrected_mask = (
        raw_test_power <= result.additive_noise_power[:, np.newaxis, :]
    )

    # The per-sensor rows expose enough information to diagnose whether a
    # sensor is unstable because of timing mismatch, low information, gain
    # saturation, or invalid deployed corrections.
    sensor_rows: list[RbwSensorValidationSummary] = []
    retained_sensor_indices = np.arange(len(result.sensor_ids), dtype=np.int64)
    for sensor_index, sensor_id in enumerate(result.sensor_ids):
        other_sensor_indices = retained_sensor_indices[
            retained_sensor_indices != sensor_index
        ]
        leave_one_out_consensus_power = compute_network_consensus(
            corrected_power=corrected_test_power[other_sensor_indices],
            residual_variance_power2=result.residual_variance_power2[
                other_sensor_indices
            ],
        )
        leave_one_out_consensus_db = power_linear_to_db(leave_one_out_consensus_power)
        corrected_residual_db = (
            corrected_test_db[sensor_index] - leave_one_out_consensus_db
        )
        sensor_rows.append(
            RbwSensorValidationSummary(
                sensor_id=sensor_id,
                mean_bias_to_consensus_db=float(np.mean(corrected_residual_db)),
                rmse_to_consensus_db=float(np.sqrt(np.mean(corrected_residual_db**2))),
                invalid_corrected_fraction=float(
                    np.mean(invalid_corrected_mask[sensor_index])
                ),
                median_information_weight=float(
                    np.median(result.information_weight[sensor_index])
                ),
                low_information_fraction=float(
                    np.mean(result.low_information_mask[sensor_index])
                ),
                gain_cap_fraction=float(
                    np.mean(result.gain_at_correction_bound_mask[sensor_index])
                ),
                alignment_median_error_ms=float(
                    dataset.alignment_median_error_ms[sensor_id]
                ),
            )
        )

    raw_mean_sensor_std_db = float(np.mean(np.std(raw_test_db, axis=0)))
    corrected_mean_sensor_std_db = float(np.mean(np.std(corrected_test_db, axis=0)))
    corrected_to_raw_dispersion_ratio = (
        float("nan")
        if raw_mean_sensor_std_db <= 0.0
        else float(corrected_mean_sensor_std_db / raw_mean_sensor_std_db)
    )
    return RbwCalibrationValidationSummary(
        raw_mean_sensor_std_db=raw_mean_sensor_std_db,
        corrected_mean_sensor_std_db=corrected_mean_sensor_std_db,
        corrected_to_raw_dispersion_ratio=corrected_to_raw_dispersion_ratio,
        sensor_rows=tuple(sensor_rows),
    )


def build_rbw_sensor_validation_rows(
    validation: RbwCalibrationValidationSummary,  # Final hold-out validation summary
) -> list[dict[str, float | str]]:  # Table-friendly per-sensor validation rows
    """Build notebook-friendly rows from an RBW hold-out validation summary.

    Parameters
    ----------
    validation:
        Hold-out validation summary produced by
        :func:`evaluate_rbw_calibration_holdout`.

    Returns
    -------
    list[dict[str, float | str]]
        One table row per retained sensor, sorted by RMSE to the corrected
        consensus so the worst sensors are easy to spot.
    """

    rows: list[dict[str, float | str]] = [
        {
            "sensor_id": row.sensor_id,
            "mean_bias_to_consensus_db": row.mean_bias_to_consensus_db,
            "rmse_to_consensus_db": row.rmse_to_consensus_db,
            "invalid_corrected_fraction": row.invalid_corrected_fraction,
            "median_information_weight": row.median_information_weight,
            "low_information_fraction": row.low_information_fraction,
            "gain_cap_fraction": row.gain_cap_fraction,
            "alignment_median_error_ms": row.alignment_median_error_ms,
        }
        for row in validation.sensor_rows
    ]
    rows.sort(key=lambda row: float(row["rmse_to_consensus_db"]))
    return rows


def identify_rbw_qc_outlier_sensor_ids(
    validation: RbwCalibrationValidationSummary,  # Hold-out validation summary to inspect
    reliable_sensor_id: str,  # Soft anchor that should not be auto-removed here
    invalid_corrected_fraction_threshold: float = DEFAULT_RBW_INVALID_CORRECTED_FRACTION_THRESHOLD,
    rmse_outlier_iqr_multiplier: float = DEFAULT_RBW_RMSE_OUTLIER_IQR_MULTIPLIER,
    min_rmse_excess_db: float = DEFAULT_RBW_MIN_RMSE_EXCESS_DB,
) -> tuple[str, ...]:  # Sensors that should be dropped before a retrain
    """Identify sensors that fail a conservative RBW post-fit QC pass.

    The automatic removal rule is intentionally narrow. A sensor is flagged
    only when it is not the reliable anchor and at least one of these holds:

    - its deployed correction produces too many invalid ``Y <= N`` bins on the
      hold-out set, or
    - its RMSE to the corrected consensus is a robust outlier by an
      interquartile-range rule, with an additional minimum absolute margin in
      dB so tiny spreads do not trigger needless exclusions.

    Parameters
    ----------
    validation:
        Hold-out validation summary for the current fit.
    reliable_sensor_id:
        Selected anchor sensor. This sensor is intentionally protected from the
        automatic removal rule.
    invalid_corrected_fraction_threshold:
        Maximum tolerated fraction of clipped ``Y <= N`` bins.
    rmse_outlier_iqr_multiplier:
        IQR multiplier used for the robust RMSE threshold.
    min_rmse_excess_db:
        Minimum absolute RMSE margin [dB] above the median before a sensor can
        be removed.

    Returns
    -------
    tuple[str, ...]
        Sensor identifiers that should be dropped before the next fit pass.
    """

    if invalid_corrected_fraction_threshold < 0.0:
        raise ValueError("invalid_corrected_fraction_threshold cannot be negative")
    if rmse_outlier_iqr_multiplier < 0.0:
        raise ValueError("rmse_outlier_iqr_multiplier cannot be negative")
    if min_rmse_excess_db < 0.0:
        raise ValueError("min_rmse_excess_db cannot be negative")

    # Use a robust RMSE threshold so one moderately noisy sensor does not force
    # a retrain unless it is clearly separated from the rest of the campaign.
    rmse_values = np.asarray(
        [row.rmse_to_consensus_db for row in validation.sensor_rows],
        dtype=np.float64,
    )
    rmse_q1, rmse_q3 = np.quantile(rmse_values, [0.25, 0.75])
    rmse_iqr = float(rmse_q3 - rmse_q1)
    rmse_threshold_db = max(
        float(np.median(rmse_values) + min_rmse_excess_db),
        float(rmse_q3 + rmse_outlier_iqr_multiplier * rmse_iqr),
    )

    qc_outlier_sensor_ids: list[str] = []
    for row in validation.sensor_rows:
        if row.sensor_id == reliable_sensor_id:
            continue
        has_invalid_corrections = (
            row.invalid_corrected_fraction > invalid_corrected_fraction_threshold
        )
        has_rmse_outlier = row.rmse_to_consensus_db > rmse_threshold_db
        if has_invalid_corrections or has_rmse_outlier:
            qc_outlier_sensor_ids.append(row.sensor_id)

    return tuple(qc_outlier_sensor_ids)


def _select_rbw_qc_retrain_sensor_ids(
    validation: RbwCalibrationValidationSummary,
    candidate_sensor_ids: Collection[str],
    retained_sensor_ids: Collection[str],
) -> tuple[str, ...]:
    """Select the next QC exclusions while preserving a viable fit.

    The retrain loop intentionally removes at most one sensor per pass. This
    avoids collapsing a weak campaign by excluding every flagged sensor at
    once, while still letting the worst offender be removed before the next
    hold-out evaluation.
    """

    max_removable_sensor_count = len(tuple(retained_sensor_ids)) - 2
    if max_removable_sensor_count <= 0:
        return ()

    candidate_sensor_id_set = {
        str(sensor_id) for sensor_id in candidate_sensor_ids if str(sensor_id)
    }
    candidate_rows = [
        row
        for row in validation.sensor_rows
        if row.sensor_id in candidate_sensor_id_set
    ]
    if not candidate_rows:
        return ()

    candidate_rows.sort(
        key=lambda row: (
            row.invalid_corrected_fraction,
            row.rmse_to_consensus_db,
            row.alignment_median_error_ms,
        ),
        reverse=True,
    )
    return (candidate_rows[0].sensor_id,)


def fit_and_save_rbw_calibration_model(
    preparation: RbwCalibrationPreparation,  # Prepared RBW subset and anchor choice
    output_dir: Path,  # Destination directory for the artifact bundle
    acquisition_dir: Path,  # Source RBW acquisition directory for this subset
    fit_config: Mapping[str, int | float],  # Numerical fitting configuration
    test_fraction: float = 0.2,  # Fraction of records reserved for hold-out validation
    split_strategy: str = "tail",  # Hold-out policy forwarded to make_holdout_split
    split_random_seed: int | None = None,  # Optional seed for randomized hold-out
    auto_retrain_after_qc: bool = True,  # Whether to rerun after dropping QC outliers
    invalid_corrected_fraction_threshold: float = DEFAULT_RBW_INVALID_CORRECTED_FRACTION_THRESHOLD,
    rmse_outlier_iqr_multiplier: float = DEFAULT_RBW_RMSE_OUTLIER_IQR_MULTIPLIER,
    min_rmse_excess_db: float = DEFAULT_RBW_MIN_RMSE_EXCESS_DB,
    max_qc_retrain_rounds: int = 3,  # Maximum number of extra QC retraining passes
) -> RbwCalibrationFitResult:  # Saved artifact plus compact validation metrics
    """Fit and save one RBW calibration model.

    When ``auto_retrain_after_qc`` is enabled, the function performs
    additional preparation/fit passes after the initial fit if the hold-out QC
    identifies clear sensor failures. Each retrain round removes only the
    single worst failing non-anchor sensor, which is conservative enough to
    avoid collapsing weak campaigns by over-pruning in one step.

    Parameters
    ----------
    preparation:
        Prepared RBW subset returned by :func:`prepare_rbw_calibration_dataset`.
    output_dir:
        Destination directory for the saved artifact bundle.
    acquisition_dir:
        RBW subset directory that supplied the acquisition CSV files.
    fit_config:
        Numerical fitting configuration forwarded to
        :func:`fit_spectral_calibration`.
    test_fraction:
        Fraction of RBW records reserved for hold-out diagnostics.
    split_strategy:
        Hold-out split policy. Use ``"random"`` with a fixed seed to avoid a
        chronology-biased trailing block when campaign order is not meaningful.
    split_random_seed:
        Optional seed used when ``split_strategy="random"``.
    auto_retrain_after_qc:
        Whether to drop clear hold-out QC failures and refit.
    invalid_corrected_fraction_threshold, rmse_outlier_iqr_multiplier, min_rmse_excess_db:
        Conservative post-fit QC thresholds used to decide whether a sensor
        should be auto-excluded.
    max_qc_retrain_rounds:
        Maximum number of additional fit passes after the initial fit.

    Returns
    -------
    RbwCalibrationFitResult
        Saved artifact path, final preparation, hold-out validation summary,
        and QC metadata.

    Side Effects
    ------------
    Writes the saved artifact bundle and sensor-summary CSV files under
    ``output_dir``.
    """

    current_preparation = preparation
    resolved_fit_config = resolve_spectral_fit_config(fit_config)
    total_fit_duration_s = 0.0
    qc_auto_excluded_sensor_ids: list[str] = []
    n_fit_passes = 0

    # Fit once, inspect the hold-out behavior, and optionally refit after
    # removing the clearly failing non-anchor sensors.
    while True:
        n_fit_passes += 1
        result, validation, fit_duration_s = _fit_rbw_model_once(
            preparation=current_preparation,
            fit_config=resolved_fit_config,
            test_fraction=test_fraction,
            split_strategy=split_strategy,
            split_random_seed=split_random_seed,
        )
        total_fit_duration_s += fit_duration_s

        if not auto_retrain_after_qc or n_fit_passes > max_qc_retrain_rounds:
            break

        qc_outlier_sensor_ids = identify_rbw_qc_outlier_sensor_ids(
            validation=validation,
            reliable_sensor_id=current_preparation.reliable_sensor_id,
            invalid_corrected_fraction_threshold=invalid_corrected_fraction_threshold,
            rmse_outlier_iqr_multiplier=rmse_outlier_iqr_multiplier,
            min_rmse_excess_db=min_rmse_excess_db,
        )
        qc_outlier_sensor_ids = tuple(
            sensor_id
            for sensor_id in qc_outlier_sensor_ids
            if sensor_id not in current_preparation.excluded_sensor_ids
        )
        qc_outlier_sensor_ids = _select_rbw_qc_retrain_sensor_ids(
            validation=validation,
            candidate_sensor_ids=qc_outlier_sensor_ids,
            retained_sensor_ids=current_preparation.calibration_dataset.sensor_ids,
        )
        if not qc_outlier_sensor_ids:
            break

        qc_auto_excluded_sensor_ids.extend(qc_outlier_sensor_ids)
        current_preparation = prepare_rbw_calibration_dataset(
            current_preparation.source_dataset,
            excluded_sensor_ids=tuple(
                sorted(
                    {
                        *current_preparation.excluded_sensor_ids,
                        *qc_outlier_sensor_ids,
                    }
                )
            ),
            excluded_sensor_ids_by_label=None,
            ranking_histogram_bins=current_preparation.ranking_histogram_bins,
            distribution_histogram_bins=current_preparation.distribution_histogram_bins,
        )

    # Persist only the final fit pass so the artifact on disk matches the
    # validated sensor set that survived the QC loop.
    artifact = save_spectral_calibration_artifact(
        output_dir=output_dir,
        result=result,
        dataset=current_preparation.calibration_dataset,
        acquisition_dir=acquisition_dir,
        response_dir=None,
        reference_sensor_id=current_preparation.reliable_sensor_id,
        reliable_sensor_id=current_preparation.reliable_sensor_id,
        excluded_sensor_ids=current_preparation.excluded_sensor_ids,
        fit_config=resolved_fit_config,
        extra_summary={
            "fit_duration_s": total_fit_duration_s,
            "test_fraction": float(test_fraction),
            "recordwise_best_mean_correlation": float(
                np.max(current_preparation.ranking_result.average_correlation)
            ),
            "distribution_best_similarity": float(
                np.max(
                    current_preparation.distribution_diagnostics.normalized_similarity_score
                )
            ),
            "distribution_outlier_count": float(
                len(current_preparation.distribution_outlier_sensor_ids)
            ),
            "raw_mean_sensor_std_db": validation.raw_mean_sensor_std_db,
            "corrected_mean_sensor_std_db": validation.corrected_mean_sensor_std_db,
            "corrected_to_raw_dispersion_ratio": (
                validation.corrected_to_raw_dispersion_ratio
            ),
            "max_invalid_corrected_fraction": float(
                max(row.invalid_corrected_fraction for row in validation.sensor_rows)
            ),
            "max_rmse_to_consensus_db": float(
                max(row.rmse_to_consensus_db for row in validation.sensor_rows)
            ),
            "manual_excluded_sensor_count": float(
                len(current_preparation.excluded_sensor_ids)
                - len(set(qc_auto_excluded_sensor_ids))
            ),
            "auto_qc_excluded_sensor_count": float(
                len(set(qc_auto_excluded_sensor_ids))
            ),
            "n_fit_passes": float(n_fit_passes),
        },
    )

    return RbwCalibrationFitResult(
        preparation=current_preparation,
        artifact=artifact,
        validation=validation,
        fit_duration_s=total_fit_duration_s,
        qc_auto_excluded_sensor_ids=tuple(sorted(set(qc_auto_excluded_sensor_ids))),
        n_fit_passes=n_fit_passes,
    )


def _fit_rbw_model_once(
    preparation: RbwCalibrationPreparation,  # RBW preparation for one fit pass
    fit_config: Mapping[
        str, int | float | None
    ],  # Fully resolved numerical fitting configuration
    test_fraction: float,  # Fraction of records reserved for hold-out diagnostics
    split_strategy: str,  # Hold-out policy forwarded to make_holdout_split
    split_random_seed: int | None,  # Optional seed for randomized hold-out
) -> tuple[
    SpectralCalibrationResult,
    RbwCalibrationValidationSummary,
    float,
]:  # Fitted result, validation, and wall-clock duration [s]
    """Fit one RBW preparation once and evaluate its hold-out behavior."""

    dataset = preparation.calibration_dataset
    train_indices, test_indices = make_holdout_split(
        n_experiments=dataset.observations_power.shape[1],
        test_fraction=test_fraction,
        strategy=split_strategy,
        random_seed=split_random_seed,
    )

    start_time = time.perf_counter()
    result = fit_spectral_calibration(
        observations_power=dataset.observations_power,
        frequency_hz=dataset.frequency_hz,
        sensor_ids=dataset.sensor_ids,
        nominal_gain_power=dataset.nominal_gain_power,
        train_indices=train_indices,
        test_indices=test_indices,
        reliable_sensor_id=preparation.reliable_sensor_id,
        n_iterations=int(cast(int, fit_config["n_iterations"])),
        lambda_gain_smooth=float(cast(float, fit_config["lambda_gain_smooth"])),
        lambda_noise_smooth=float(cast(float, fit_config["lambda_noise_smooth"])),
        lambda_gain_reference=float(cast(float, fit_config["lambda_gain_reference"])),
        lambda_noise_reference=float(cast(float, fit_config["lambda_noise_reference"])),
        lambda_reliable_anchor=float(cast(float, fit_config["lambda_reliable_anchor"])),
        reliable_weight_boost=float(cast(float, fit_config["reliable_weight_boost"])),
        max_correction_db=cast(float | None, fit_config["max_correction_db"]),
        low_information_threshold_ratio=float(
            cast(float, fit_config["low_information_threshold_ratio"])
        ),
        low_information_weight=float(cast(float, fit_config["low_information_weight"])),
        min_variance=float(cast(float, fit_config["min_variance"])),
    )
    fit_duration_s = time.perf_counter() - start_time
    validation = evaluate_rbw_calibration_holdout(preparation, result)
    return result, validation, fit_duration_s


def _select_reliable_sensor_id(
    ranking_result: SensorRankingResult,
    distribution_outlier_sensor_ids: Collection[str],
) -> str:
    """Choose the RBW reliable sensor from the ranking diagnostics."""

    outlier_sensor_id_set = {
        str(sensor_id) for sensor_id in distribution_outlier_sensor_ids
    }
    for sensor_id in ranking_result.ranking_sensor_ids:
        if sensor_id not in outlier_sensor_id_set:
            return sensor_id
    return ranking_result.ranking_sensor_ids[0]


__all__ = [
    "DEFAULT_RBW_EXCLUDED_SENSOR_IDS",
    "DEFAULT_RBW_EXCLUDED_SENSOR_IDS_BY_LABEL",
    "DEFAULT_RBW_INVALID_CORRECTED_FRACTION_THRESHOLD",
    "DEFAULT_RBW_MIN_RMSE_EXCESS_DB",
    "DEFAULT_RBW_RMSE_OUTLIER_IQR_MULTIPLIER",
    "RbwCalibrationFitResult",
    "RbwCalibrationPreparation",
    "RbwCalibrationValidationSummary",
    "RbwSensorValidationSummary",
    "build_rbw_preparation_rows",
    "build_rbw_sensor_validation_rows",
    "evaluate_rbw_calibration_holdout",
    "exclude_rbw_sensors",
    "fit_and_save_rbw_calibration_model",
    "identify_rbw_qc_outlier_sensor_ids",
    "load_rbw_calibration_preparations",
    "prepare_rbw_calibration_dataset",
    "resolve_rbw_excluded_sensor_ids",
]
