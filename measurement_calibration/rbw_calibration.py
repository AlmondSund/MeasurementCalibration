"""RBW-specific adapters for the spectral calibration workflow.

This module keeps the RBW calibration path structurally aligned with the
existing common-field calibration pipeline:

- :mod:`measurement_calibration.sensor_ranking` remains responsible for loading
  the row-aligned RBW acquisitions and for computing the ranking diagnostics
  that justify sensor-selection decisions.
- :mod:`measurement_calibration.spectral_calibration` remains responsible for
  the actual latent-variable calibration fit.
- This module only adapts RBW acquisitions into the calibration-core data
  contract, chooses a conservative reliable-anchor sensor, and coordinates the
  artifact persistence inputs.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
import time

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
    apply_deployed_calibration,
    compute_network_consensus,
    fit_spectral_calibration,
    make_holdout_split,
    power_db_to_linear,
    power_linear_to_db,
)


DEFAULT_RBW_EXCLUDED_SENSOR_IDS = ("Node9",)


@dataclass(frozen=True)
class RbwCalibrationPreparation:
    """Prepared RBW subset ready for calibration fitting.

    Parameters
    ----------
    rbw_label:
        RBW subset identifier such as ``"10K"``.
    calibration_dataset:
        RBW observations converted into the generic calibration-dataset shape
        expected by :func:`fit_spectral_calibration`.
    ranking_result:
        Record-wise cumulative-correlation diagnostic computed after the
        requested sensor exclusions.
    distribution_diagnostics:
        Campaign-wide PSD-distribution diagnostic computed on the same retained
        sensors.
    reliable_sensor_id:
        Sensor chosen as the soft anchor. The selection rule is intentionally
        conservative: prefer the best record-wise sensor that is not flagged as
        a distribution-shape outlier, then fall back to the record-wise winner
        if every retained sensor is an outlier.
    excluded_sensor_ids:
        Sensor identifiers explicitly removed before preparation.
    distribution_outlier_sensor_ids:
        Retained sensors whose PSD distributions still disagree with the rest
        of the campaign according to the histogram diagnostic.
    """

    rbw_label: str
    calibration_dataset: CalibrationDataset
    ranking_result: SensorRankingResult
    distribution_diagnostics: SensorDistributionDiagnostics
    reliable_sensor_id: str
    excluded_sensor_ids: tuple[str, ...]
    distribution_outlier_sensor_ids: tuple[str, ...]


@dataclass(frozen=True)
class RbwCalibrationFitResult:
    """Stored RBW model plus the fitted result kept in memory for validation.

    Parameters
    ----------
    preparation:
        Prepared RBW subset used for the fit.
    artifact:
        Saved artifact bundle on disk.
    raw_mean_sensor_std_db:
        Mean inter-sensor hold-out dispersion before applying calibration [dB].
    corrected_mean_sensor_std_db:
        Mean inter-sensor hold-out dispersion after calibration [dB].
    corrected_to_raw_dispersion_ratio:
        Hold-out dispersion ratio. Values below one indicate improvement.
    fit_duration_s:
        Wall-clock training time [s].
    """

    preparation: RbwCalibrationPreparation
    artifact: SavedCalibrationArtifact
    raw_mean_sensor_std_db: float
    corrected_mean_sensor_std_db: float
    corrected_to_raw_dispersion_ratio: float
    fit_duration_s: float


def exclude_rbw_sensors(
    dataset: RbwAcquisitionDataset,  # One RBW subset to filter
    excluded_sensor_ids: Collection[str],  # Sensor identifiers to remove
) -> RbwAcquisitionDataset:  # Filtered copy that preserves record alignment
    """Return a filtered RBW dataset without the requested sensors.

    Parameters
    ----------
    dataset:
        Parsed RBW subset whose first axis corresponds to sensors.
    excluded_sensor_ids:
        Sensor identifiers to remove. Missing identifiers are ignored so the
        helper can be reused across RBW subsets with different sensor sets.

    Returns
    -------
    RbwAcquisitionDataset
        Filtered dataset with the same record and frequency grids.

    Raises
    ------
    ValueError
        If fewer than two sensors remain after filtering.
    """

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
            dataset.timestamps_ms[retained_indices], dtype=np.int64
        ),
    )


def prepare_rbw_calibration_dataset(
    dataset: RbwAcquisitionDataset,  # One RBW subset to adapt for calibration
    excluded_sensor_ids: Collection[str] = DEFAULT_RBW_EXCLUDED_SENSOR_IDS,
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
) -> RbwCalibrationPreparation:  # Prepared calibration inputs and diagnostics
    """Prepare one RBW subset for calibration fitting.

    The RBW acquisitions are already row-aligned by record index, so no
    sequence-shift search is needed here. The adapter therefore:

    1. removes explicitly excluded sensors,
    2. recomputes the ranking diagnostics on the retained subset,
    3. chooses a reliable anchor sensor using the combined record-wise and
       distribution-shape diagnostics, and
    4. converts the dB PSD tensor into the positive linear-power tensor
       required by :func:`fit_spectral_calibration`.
    """

    filtered_dataset = exclude_rbw_sensors(dataset, excluded_sensor_ids)
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
        calibration_dataset=calibration_dataset,
        ranking_result=ranking_result,
        distribution_diagnostics=distribution_diagnostics,
        reliable_sensor_id=reliable_sensor_id,
        excluded_sensor_ids=tuple(
            sensor_id
            for sensor_id in sorted(
                {str(sensor_id) for sensor_id in excluded_sensor_ids}
            )
            if sensor_id in dataset.sensor_ids
        ),
        distribution_outlier_sensor_ids=distribution_outlier_sensor_ids,
    )


def load_rbw_calibration_preparations(
    root_dir: Path,  # Root directory with one subdirectory per RBW subset
    excluded_sensor_ids: Collection[str] = DEFAULT_RBW_EXCLUDED_SENSOR_IDS,
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
) -> dict[str, RbwCalibrationPreparation]:  # Prepared datasets indexed by RBW label
    """Load every RBW subset and adapt it for calibration fitting."""

    preparations: dict[str, RbwCalibrationPreparation] = {}
    for rbw_label, dataset in sorted(load_rbw_acquisition_datasets(root_dir).items()):
        preparations[rbw_label] = prepare_rbw_calibration_dataset(
            dataset,
            excluded_sensor_ids=excluded_sensor_ids,
            ranking_histogram_bins=ranking_histogram_bins,
            distribution_histogram_bins=distribution_histogram_bins,
        )
    return preparations


def build_rbw_preparation_rows(
    preparations: Mapping[str, RbwCalibrationPreparation],  # Prepared datasets by RBW
) -> list[dict[str, float | int | str]]:  # Table-friendly summary rows
    """Build a compact summary of the RBW preparation decisions.

    The returned rows are designed for notebook display and CSV-style
    inspection. They expose which sensors were retained, which sensor became
    the reliable anchor, and whether the record-wise winner was also flagged as
    a distribution outlier.
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


def fit_and_save_rbw_calibration_model(
    preparation: RbwCalibrationPreparation,  # Prepared RBW subset and anchor choice
    output_dir: Path,  # Destination directory for the artifact bundle
    acquisition_dir: Path,  # Source RBW acquisition directory for this subset
    fit_config: Mapping[str, int | float],  # Numerical fitting configuration
    test_fraction: float = 0.2,  # Fraction of records reserved for hold-out validation
) -> RbwCalibrationFitResult:  # Saved artifact plus compact validation metrics
    """Fit and save one RBW calibration model.

    Parameters
    ----------
    preparation:
        Prepared RBW subset returned by :func:`prepare_rbw_calibration_dataset`.
    output_dir:
        Destination directory for the saved artifact.
    acquisition_dir:
        RBW subset directory that supplied the acquisition CSV files.
    fit_config:
        Numerical fit configuration forwarded to
        :func:`fit_spectral_calibration`.
    test_fraction:
        Fraction of RBW records reserved for hold-out diagnostics.

    Returns
    -------
    RbwCalibrationFitResult
        Saved artifact plus the compact hold-out metrics most useful for
        notebook summaries.
    """

    dataset = preparation.calibration_dataset
    train_indices, test_indices = make_holdout_split(
        n_experiments=dataset.observations_power.shape[1],
        test_fraction=test_fraction,
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
    fit_duration_s = time.perf_counter() - start_time

    raw_test_power = dataset.observations_power[:, result.test_indices, :]
    corrected_test_power = apply_deployed_calibration(
        observations_power=raw_test_power,
        gain_power=result.gain_power,
        additive_noise_power=result.additive_noise_power,
    )
    compute_network_consensus(
        corrected_power=corrected_test_power,
        residual_variance_power2=result.residual_variance_power2,
    )
    raw_mean_sensor_std_db = float(
        np.mean(np.std(power_linear_to_db(raw_test_power), axis=0))
    )
    corrected_mean_sensor_std_db = float(
        np.mean(np.std(power_linear_to_db(corrected_test_power), axis=0))
    )
    corrected_to_raw_dispersion_ratio = float(
        corrected_mean_sensor_std_db / raw_mean_sensor_std_db
    )

    artifact = save_spectral_calibration_artifact(
        output_dir=output_dir,
        result=result,
        dataset=dataset,
        acquisition_dir=acquisition_dir,
        response_dir=None,
        reference_sensor_id=preparation.reliable_sensor_id,
        reliable_sensor_id=preparation.reliable_sensor_id,
        excluded_sensor_ids=preparation.excluded_sensor_ids,
        fit_config=fit_config,
        extra_summary={
            "fit_duration_s": fit_duration_s,
            "test_fraction": float(test_fraction),
            "recordwise_best_mean_correlation": float(
                np.max(preparation.ranking_result.average_correlation)
            ),
            "distribution_best_similarity": float(
                np.max(preparation.distribution_diagnostics.normalized_similarity_score)
            ),
            "distribution_outlier_count": float(
                len(preparation.distribution_outlier_sensor_ids)
            ),
            "raw_mean_sensor_std_db": raw_mean_sensor_std_db,
            "corrected_mean_sensor_std_db": corrected_mean_sensor_std_db,
            "corrected_to_raw_dispersion_ratio": corrected_to_raw_dispersion_ratio,
        },
    )

    return RbwCalibrationFitResult(
        preparation=preparation,
        artifact=artifact,
        raw_mean_sensor_std_db=raw_mean_sensor_std_db,
        corrected_mean_sensor_std_db=corrected_mean_sensor_std_db,
        corrected_to_raw_dispersion_ratio=corrected_to_raw_dispersion_ratio,
        fit_duration_s=fit_duration_s,
    )


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
    "RbwCalibrationFitResult",
    "RbwCalibrationPreparation",
    "build_rbw_preparation_rows",
    "exclude_rbw_sensors",
    "fit_and_save_rbw_calibration_model",
    "load_rbw_calibration_preparations",
    "prepare_rbw_calibration_dataset",
]
