"""Campaign-specific adapters for the spectral calibration workflow.

This module bridges the dynamic campaign datasets stored under
``data/campaigns/<campaign_label>`` with the project's reusable spectral
calibration model:

- :mod:`measurement_calibration.sensor_ranking` owns CSV loading, timestamp
  alignment, and campaign diagnostics.
- :mod:`measurement_calibration.spectral_calibration` owns the latent-variable
  calibration fit.
- This module adapts one aligned campaign into the calibration data contract,
  chooses a conservative reliable anchor sensor, evaluates hold-out behavior,
  and persists the fitted artifact bundle under ``models/``.
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
    DEFAULT_CAMPAIGNS_DATA_DIR,
    CampaignAlignmentDiagnostics,
    FileSystemCampaignSensorDataRepository,
    RbwAcquisitionDataset,
    SensorDistributionDiagnostics,
    SensorMeasurementSeries,
    SensorRankingResult,
    align_campaign_sensor_series,
    build_distribution_summary_rows,
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


DEFAULT_CAMPAIGN_CALIBRATION_MODELS_DIR = (
    Path("models") / "campaign_spectral_calibration"
)


@dataclass(frozen=True)
class CampaignCalibrationPreparation:
    """Prepared campaign dataset ready for calibration fitting.

    Parameters
    ----------
    campaign_label:
        Campaign identifier resolved from ``data/campaigns/<campaign_label>``.
    campaign_dir:
        Directory that supplied the campaign CSV files.
    aligned_dataset:
        Timestamp-aligned campaign PSD tensor in dB space.
    calibration_dataset:
        Aligned campaign converted into the generic calibration-data contract.
    ranking_result:
        Record-wise cumulative-correlation diagnostic for the retained sensors.
    distribution_diagnostics:
        Campaign-wide PSD-distribution diagnostic on the same retained sensors.
    alignment_diagnostics:
        Summary of the timestamp matching that produced ``aligned_dataset``.
    reliable_sensor_id:
        Soft anchor sensor chosen from the ranking diagnostics. The rule is
        conservative: use the highest record-wise sensor that is not flagged as
        a distribution-shape outlier, and fall back to the record-wise winner
        if every retained sensor is an outlier.
    excluded_sensor_ids:
        Sensors explicitly removed before alignment and fitting.
    distribution_outlier_sensor_ids:
        Retained sensors that still disagree with the campaign-wide PSD
        distribution according to the histogram diagnostic.
    ranking_histogram_bins, distribution_histogram_bins:
        Diagnostic configuration stored so the exact preparation can be audited
        and reproduced from the notebook or a later script.
    """

    campaign_label: str
    campaign_dir: Path
    aligned_dataset: RbwAcquisitionDataset
    calibration_dataset: CalibrationDataset
    ranking_result: SensorRankingResult
    distribution_diagnostics: SensorDistributionDiagnostics
    alignment_diagnostics: CampaignAlignmentDiagnostics
    reliable_sensor_id: str
    excluded_sensor_ids: tuple[str, ...]
    distribution_outlier_sensor_ids: tuple[str, ...]
    ranking_histogram_bins: int
    distribution_histogram_bins: int


@dataclass(frozen=True)
class CampaignSensorValidationSummary:
    """Hold-out validation metrics for one retained campaign sensor.

    Parameters
    ----------
    sensor_id:
        Sensor identifier matching the fitted result.
    mean_bias_to_consensus_db:
        Average residual offset versus the leave-one-out corrected consensus
        on the hold-out experiments [dB].
    rmse_to_consensus_db:
        Root-mean-square residual versus that corrected consensus [dB].
    invalid_corrected_fraction:
        Fraction of hold-out bins where deployment-time correction would
        require subtracting more additive noise than the observed power.
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
class CampaignCalibrationValidationSummary:
    """Compact hold-out validation summary for one fitted campaign model."""

    raw_mean_sensor_std_db: float
    corrected_mean_sensor_std_db: float
    corrected_to_raw_dispersion_ratio: float
    sensor_rows: tuple[CampaignSensorValidationSummary, ...]


@dataclass(frozen=True)
class CampaignCalibrationFitResult:
    """Stored campaign model plus the fitted validation metadata.

    Parameters
    ----------
    preparation:
        Campaign preparation used to produce the stored fit.
    artifact:
        Saved artifact bundle on disk.
    validation:
        Hold-out validation summary for the stored fit.
    fit_duration_s:
        Wall-clock fitting time [s].
    """

    preparation: CampaignCalibrationPreparation
    artifact: SavedCalibrationArtifact
    validation: CampaignCalibrationValidationSummary
    fit_duration_s: float


@dataclass(frozen=True)
class CampaignDeploymentInputs:
    """Shared-sensor deployment tensors for a stored campaign model.

    Parameters
    ----------
    shared_sensor_ids:
        Sensors present in both the stored model and the deployment campaign,
        ordered to match the stored model parameters.
    observations_power:
        Deployment observations restricted to ``shared_sensor_ids`` and ordered
        consistently with the returned model tensors.
    gain_power, additive_noise_power, residual_variance_power2:
        Stored calibration parameters restricted to ``shared_sensor_ids``.
    experiment_timestamps_ms:
        Deployment experiment timestamps [ms] after any campaign alignment.
    frequency_hz:
        Shared frequency grid [Hz] validated against the stored model.
    """

    shared_sensor_ids: tuple[str, ...]
    observations_power: np.ndarray
    gain_power: np.ndarray
    additive_noise_power: np.ndarray
    residual_variance_power2: np.ndarray
    experiment_timestamps_ms: np.ndarray
    frequency_hz: np.ndarray


def build_campaign_calibration_output_dir(
    campaign_label: str,  # Campaign label chosen by the caller
    models_root: Path = DEFAULT_CAMPAIGN_CALIBRATION_MODELS_DIR,  # Root under models/
) -> Path:
    """Build the model-output directory for one campaign calibration artifact."""

    if not str(campaign_label).strip():
        raise ValueError("campaign_label must be a non-empty string")
    return Path(models_root) / str(campaign_label)


def prepare_campaign_calibration_dataset(
    campaign_label: str,  # Dataset label under data/campaigns/
    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR,  # Root with campaign directories
    excluded_sensor_ids: Collection[str] = (),
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
    alignment_tolerance_ms: int | None = None,
    sensor_file_pattern: str = "*.csv",
) -> CampaignCalibrationPreparation:
    """Prepare one stored campaign for spectral calibration fitting.

    The adapter keeps the model boundary explicit:

    1. load raw per-sensor campaign series from disk,
    2. apply explicit exclusions before timestamp alignment,
    3. align rows across the retained sensor set,
    4. compute the same ranking diagnostics used for campaign QC, and
    5. convert the aligned dB PSD tensor into the linear-power calibration
       tensor required by :func:`fit_spectral_calibration`.
    """

    repository = FileSystemCampaignSensorDataRepository(
        campaigns_root=campaigns_root,
        sensor_file_pattern=sensor_file_pattern,
    )
    campaign_dir = Path(campaigns_root) / str(campaign_label)
    sensor_series_by_id = repository.load_campaign_sensor_series(campaign_label)
    retained_sensor_series_by_id, resolved_excluded_sensor_ids = (
        _exclude_campaign_sensor_series(
            sensor_series_by_id=sensor_series_by_id,
            excluded_sensor_ids=excluded_sensor_ids,
            campaign_label=campaign_label,
        )
    )

    aligned_dataset, alignment_diagnostics = align_campaign_sensor_series(
        campaign_label=campaign_label,
        sensor_series_by_id=retained_sensor_series_by_id,
        alignment_tolerance_ms=alignment_tolerance_ms,
    )
    ranking_result = rank_sensors_by_cumulative_correlation(
        aligned_dataset,
        histogram_bins=ranking_histogram_bins,
    )
    distribution_diagnostics = summarize_psd_distribution(
        aligned_dataset,
        histogram_bins=distribution_histogram_bins,
    )
    distribution_outlier_sensor_ids = _distribution_outlier_sensor_ids(
        distribution_diagnostics
    )
    reliable_sensor_id = _select_reliable_sensor_id(
        ranking_result=ranking_result,
        distribution_outlier_sensor_ids=distribution_outlier_sensor_ids,
    )

    # Use the campaign median timestamp as the experiment reference so the
    # calibration dataset does not implicitly privilege one retained sensor.
    experiment_timestamps_ms = np.median(
        aligned_dataset.timestamps_ms,
        axis=0,
    ).astype(np.int64)
    alignment_median_error_ms = {
        sensor_id: float(
            np.median(
                np.abs(
                    aligned_dataset.timestamps_ms[sensor_index]
                    - experiment_timestamps_ms
                )
            )
        )
        for sensor_index, sensor_id in enumerate(aligned_dataset.sensor_ids)
    }
    calibration_dataset = CalibrationDataset(
        sensor_ids=aligned_dataset.sensor_ids,
        frequency_hz=np.asarray(aligned_dataset.frequency_hz, dtype=np.float64),
        observations_power=power_db_to_linear(aligned_dataset.observations_db),
        nominal_gain_power=np.ones(
            (aligned_dataset.n_sensors, aligned_dataset.n_frequencies),
            dtype=np.float64,
        ),
        experiment_timestamps_ms=experiment_timestamps_ms,
        selected_band_hz=(
            float(aligned_dataset.frequency_hz[0]),
            float(aligned_dataset.frequency_hz[-1]),
        ),
        sensor_shifts={sensor_id: 0 for sensor_id in aligned_dataset.sensor_ids},
        alignment_median_error_ms=alignment_median_error_ms,
        source_row_indices={
            sensor_id: np.asarray(
                alignment_diagnostics.aligned_row_indices[sensor_index],
                dtype=np.int64,
            )
            for sensor_index, sensor_id in enumerate(aligned_dataset.sensor_ids)
        },
    )

    return CampaignCalibrationPreparation(
        campaign_label=str(campaign_label),
        campaign_dir=campaign_dir,
        aligned_dataset=aligned_dataset,
        calibration_dataset=calibration_dataset,
        ranking_result=ranking_result,
        distribution_diagnostics=distribution_diagnostics,
        alignment_diagnostics=alignment_diagnostics,
        reliable_sensor_id=reliable_sensor_id,
        excluded_sensor_ids=resolved_excluded_sensor_ids,
        distribution_outlier_sensor_ids=distribution_outlier_sensor_ids,
        ranking_histogram_bins=int(ranking_histogram_bins),
        distribution_histogram_bins=int(distribution_histogram_bins),
    )


def evaluate_campaign_calibration_holdout(
    preparation: CampaignCalibrationPreparation,  # Final campaign preparation
    result: SpectralCalibrationResult,  # Fitted campaign calibration result
) -> CampaignCalibrationValidationSummary:
    """Evaluate one campaign fit on its held-out experiments."""

    dataset = preparation.calibration_dataset
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

    sensor_rows: list[CampaignSensorValidationSummary] = []
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
            valid_mask=~invalid_corrected_mask[other_sensor_indices],
        )
        leave_one_out_consensus_db = power_linear_to_db(leave_one_out_consensus_power)
        corrected_residual_db = (
            corrected_test_db[sensor_index] - leave_one_out_consensus_db
        )
        sensor_rows.append(
            CampaignSensorValidationSummary(
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
    return CampaignCalibrationValidationSummary(
        raw_mean_sensor_std_db=raw_mean_sensor_std_db,
        corrected_mean_sensor_std_db=corrected_mean_sensor_std_db,
        corrected_to_raw_dispersion_ratio=corrected_to_raw_dispersion_ratio,
        sensor_rows=tuple(sensor_rows),
    )


def build_campaign_sensor_validation_rows(
    validation: CampaignCalibrationValidationSummary,  # Final hold-out summary
) -> list[dict[str, float | str]]:
    """Build notebook-friendly rows from a campaign hold-out summary."""

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


def fit_and_save_campaign_calibration_model(
    preparation: CampaignCalibrationPreparation,  # Prepared campaign inputs
    output_dir: Path,  # Destination directory for the artifact bundle
    fit_config: Mapping[str, int | float | None],  # Numerical fitting configuration
    test_fraction: float = 0.2,  # Fraction of experiments reserved for hold-out
    split_strategy: str = "tail",  # Hold-out policy
    split_random_seed: int | None = None,  # Optional seed for randomized hold-out
) -> CampaignCalibrationFitResult:
    """Fit and save one campaign calibration model under ``models/``."""

    dataset = preparation.calibration_dataset
    resolved_fit_config = resolve_spectral_fit_config(fit_config)
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
        n_iterations=int(cast(int, resolved_fit_config["n_iterations"])),
        lambda_gain_smooth=float(
            cast(float, resolved_fit_config["lambda_gain_smooth"])
        ),
        lambda_noise_smooth=float(
            cast(float, resolved_fit_config["lambda_noise_smooth"])
        ),
        lambda_gain_reference=float(
            cast(float, resolved_fit_config["lambda_gain_reference"])
        ),
        lambda_noise_reference=float(
            cast(float, resolved_fit_config["lambda_noise_reference"])
        ),
        lambda_reliable_anchor=float(
            cast(float, resolved_fit_config["lambda_reliable_anchor"])
        ),
        reliable_weight_boost=float(
            cast(float, resolved_fit_config["reliable_weight_boost"])
        ),
        max_correction_db=cast(float | None, resolved_fit_config["max_correction_db"]),
        low_information_threshold_ratio=float(
            cast(float, resolved_fit_config["low_information_threshold_ratio"])
        ),
        low_information_weight=float(
            cast(float, resolved_fit_config["low_information_weight"])
        ),
        min_variance=float(cast(float, resolved_fit_config["min_variance"])),
    )
    fit_duration_s = time.perf_counter() - start_time
    validation = evaluate_campaign_calibration_holdout(preparation, result)

    artifact = save_spectral_calibration_artifact(
        output_dir=output_dir,
        result=result,
        dataset=dataset,
        acquisition_dir=preparation.campaign_dir,
        response_dir=None,
        reference_sensor_id=preparation.alignment_diagnostics.anchor_sensor_id,
        reliable_sensor_id=preparation.reliable_sensor_id,
        excluded_sensor_ids=preparation.excluded_sensor_ids,
        fit_config=resolved_fit_config,
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
            "raw_mean_sensor_std_db": validation.raw_mean_sensor_std_db,
            "corrected_mean_sensor_std_db": validation.corrected_mean_sensor_std_db,
            "corrected_to_raw_dispersion_ratio": (
                validation.corrected_to_raw_dispersion_ratio
            ),
        },
    )

    return CampaignCalibrationFitResult(
        preparation=preparation,
        artifact=artifact,
        validation=validation,
        fit_duration_s=fit_duration_s,
    )


def build_campaign_deployment_inputs(
    preparation: CampaignCalibrationPreparation,  # Deployment campaign preparation
    trained_result: SpectralCalibrationResult,  # Stored or in-memory trained model
    min_shared_sensors: int = 2,  # Minimum overlap required for deployment
    frequency_atol_hz: float = 0.0,  # Absolute frequency tolerance [Hz]
    frequency_rtol: float = 0.0,  # Relative frequency tolerance
) -> CampaignDeploymentInputs:
    """Match a deployment campaign against a stored model on shared sensors.

    The calibration artifact and the deployment campaign may not expose the
    same sensor inventory. This helper keeps deployment logic explicit and
    deterministic by:

    1. selecting the sensor overlap in the stored-model order,
    2. reordering the deployment observations to the same order, and
    3. validating that both sides use a compatible frequency grid.
    """

    deployment_dataset = preparation.calibration_dataset
    trained_sensor_index_by_id = {
        sensor_id: sensor_index
        for sensor_index, sensor_id in enumerate(trained_result.sensor_ids)
    }
    deployment_sensor_index_by_id = {
        sensor_id: sensor_index
        for sensor_index, sensor_id in enumerate(deployment_dataset.sensor_ids)
    }
    shared_sensor_ids = tuple(
        sensor_id
        for sensor_id in trained_result.sensor_ids
        if sensor_id in deployment_sensor_index_by_id
    )
    if len(shared_sensor_ids) < int(min_shared_sensors):
        raise ValueError(
            "Campaign deployment requires at least "
            f"{min_shared_sensors} shared sensors, but only found "
            f"{len(shared_sensor_ids)}: {shared_sensor_ids}"
        )

    if trained_result.frequency_hz.shape != deployment_dataset.frequency_hz.shape:
        raise ValueError(
            "Stored model and deployment campaign use incompatible frequency-grid "
            "shapes: "
            f"{trained_result.frequency_hz.shape} != "
            f"{deployment_dataset.frequency_hz.shape}"
        )
    if not np.allclose(
        trained_result.frequency_hz,
        deployment_dataset.frequency_hz,
        rtol=frequency_rtol,
        atol=frequency_atol_hz,
    ):
        raise ValueError(
            "Stored model and deployment campaign use different frequency grids."
        )

    trained_indices = np.asarray(
        [trained_sensor_index_by_id[sensor_id] for sensor_id in shared_sensor_ids],
        dtype=np.int64,
    )
    deployment_indices = np.asarray(
        [deployment_sensor_index_by_id[sensor_id] for sensor_id in shared_sensor_ids],
        dtype=np.int64,
    )
    return CampaignDeploymentInputs(
        shared_sensor_ids=shared_sensor_ids,
        observations_power=deployment_dataset.observations_power[deployment_indices],
        gain_power=trained_result.gain_power[trained_indices],
        additive_noise_power=trained_result.additive_noise_power[trained_indices],
        residual_variance_power2=trained_result.residual_variance_power2[
            trained_indices
        ],
        experiment_timestamps_ms=deployment_dataset.experiment_timestamps_ms,
        frequency_hz=deployment_dataset.frequency_hz,
    )


def _exclude_campaign_sensor_series(
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    excluded_sensor_ids: Collection[str],
    campaign_label: str,
) -> tuple[dict[str, SensorMeasurementSeries], tuple[str, ...]]:
    """Filter explicit sensor exclusions while validating the requested ids."""

    available_sensor_ids = tuple(sorted(sensor_series_by_id))
    excluded_sensor_id_set = {
        str(sensor_id) for sensor_id in excluded_sensor_ids if str(sensor_id)
    }
    unknown_sensor_ids = sorted(excluded_sensor_id_set.difference(available_sensor_ids))
    if unknown_sensor_ids:
        raise ValueError(
            "excluded_sensor_ids contains unknown sensors for "
            f"{campaign_label!r}: {unknown_sensor_ids}"
        )

    retained_sensor_series_by_id = {
        sensor_id: sensor_series
        for sensor_id, sensor_series in sensor_series_by_id.items()
        if sensor_id not in excluded_sensor_id_set
    }
    if len(retained_sensor_series_by_id) < 2:
        raise ValueError(
            "Campaign calibration requires at least two retained sensors after "
            f"exclusions, but {campaign_label!r} keeps "
            f"{len(retained_sensor_series_by_id)}."
        )

    return retained_sensor_series_by_id, tuple(sorted(excluded_sensor_id_set))


def _distribution_outlier_sensor_ids(
    diagnostics: SensorDistributionDiagnostics,
) -> tuple[str, ...]:
    """Extract low-similarity distribution outliers from the histogram rows."""

    distribution_rows = build_distribution_summary_rows(diagnostics)
    return tuple(
        str(row["sensor_id"])
        for row in distribution_rows
        if bool(row["is_low_similarity_outlier"])
    )


def _select_reliable_sensor_id(
    ranking_result: SensorRankingResult,
    distribution_outlier_sensor_ids: Collection[str],
) -> str:
    """Choose the reliable anchor sensor from the ranking diagnostics."""

    outlier_sensor_id_set = {
        str(sensor_id)
        for sensor_id in distribution_outlier_sensor_ids
        if str(sensor_id)
    }
    for sensor_id in ranking_result.ranking_sensor_ids:
        if sensor_id not in outlier_sensor_id_set:
            return sensor_id
    return ranking_result.ranking_sensor_ids[0]


__all__ = [
    "CampaignDeploymentInputs",
    "DEFAULT_CAMPAIGN_CALIBRATION_MODELS_DIR",
    "CampaignCalibrationFitResult",
    "CampaignCalibrationPreparation",
    "CampaignCalibrationValidationSummary",
    "CampaignSensorValidationSummary",
    "build_campaign_calibration_output_dir",
    "build_campaign_deployment_inputs",
    "build_campaign_sensor_validation_rows",
    "evaluate_campaign_calibration_holdout",
    "fit_and_save_campaign_calibration_model",
    "prepare_campaign_calibration_dataset",
]
