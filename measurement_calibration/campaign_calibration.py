"""Campaign-oriented adapters for the two-level calibration workflow.

This module keeps filesystem and orchestration concerns away from the
numerical core:

- metadata parsing and per-campaign preparation live here;
- the corpus-level trainer is called through an explicit service function;
- ranking diagnostics remain optional preparation aids, not part of the core
  mathematical model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import csv
import re
import time

import numpy as np

from .artifacts import (
    DEFAULT_ARTIFACT_PARAMETERS_FILENAME,
    SavedCalibrationArtifact,
    save_two_level_calibration_artifact,
)
from .sensor_ranking import (
    DEFAULT_CAMPAIGNS_DATA_DIR,
    CampaignAlignmentDiagnostics,
    FileSystemCampaignSensorDataRepository,
    RbwAcquisitionDataset,
    SensorDistributionDiagnostics,
    SensorMeasurementSeries,
    SensorRankingResult,
    align_campaign_sensor_series_with_pruning,
    rank_sensors_by_cumulative_correlation,
    summarize_psd_distribution,
)
from .spectral_calibration import (
    CalibrationCampaign,
    CalibrationCorpus,
    CampaignConfiguration,
    FrequencyBasisConfig,
    PersistentModelConfig,
    TwoLevelCalibrationResult,
    TwoLevelFitConfig,
    build_calibration_corpus,
    fit_two_level_calibration,
    power_db_to_linear,
)
from .notebook_workflow_configuration import fingerprint_notebook_workflow_config


DEFAULT_CONFIGURATION_CONDITIONAL_MODELS_DIR = (
    Path("models") / "configuration_conditional_calibration"
)
_DEFAULT_METADATA_FILENAME = "metadata.csv"


@dataclass(frozen=True)
class PreparedCalibrationCampaign:
    """Prepared offline campaign with diagnostics and parsed configuration.

    Parameters
    ----------
    campaign_label:
        Campaign identifier under ``data/campaigns/<campaign_label>``.
    campaign_dir:
        Directory that supplied the campaign files.
    metadata_path:
        Campaign metadata CSV path used to build ``campaign.configuration``.
    aligned_dataset:
        Timestamp-aligned dB-domain PSD tensor kept for diagnostics.
    campaign:
        Linear-power campaign representation consumed by the numerical core.
    ranking_result:
        Record-wise correlation diagnostic for the aligned sensors.
    distribution_diagnostics:
        Campaign-wide PSD-distribution diagnostic for the aligned sensors.
    alignment_diagnostics:
        Summary of the timestamp alignment stage.
    reliable_sensor_id:
        Reliable sensor selected from the ranking and distribution diagnostics.
    excluded_sensor_ids:
        Sensors explicitly removed before alignment.
    alignment_pruned_sensor_ids:
        Sensors automatically pruned because the full retained roster had no
        timestamp-aligned record groups in common.
    distribution_outlier_sensor_ids:
        Retained sensors flagged as PSD-distribution outliers.
    ranking_histogram_bins, distribution_histogram_bins:
        Diagnostic configuration retained for auditability.
    """

    campaign_label: str
    campaign_dir: Path
    metadata_path: Path
    aligned_dataset: RbwAcquisitionDataset
    campaign: CalibrationCampaign
    ranking_result: SensorRankingResult
    distribution_diagnostics: SensorDistributionDiagnostics
    alignment_diagnostics: CampaignAlignmentDiagnostics
    reliable_sensor_id: str
    excluded_sensor_ids: tuple[str, ...]
    alignment_pruned_sensor_ids: tuple[str, ...]
    distribution_outlier_sensor_ids: tuple[str, ...]
    ranking_histogram_bins: int
    distribution_histogram_bins: int


@dataclass(frozen=True)
class CalibrationCorpusPreparation:
    """Prepared collection of campaigns ready for corpus-level training."""

    campaigns_root: Path
    prepared_campaigns: tuple[PreparedCalibrationCampaign, ...]
    corpus: CalibrationCorpus


@dataclass(frozen=True)
class CalibrationCorpusFitResult:
    """Stored corpus fit plus the artifact written to disk.

    Parameters
    ----------
    preparation:
        Prepared corpus used for fitting.
    result:
        Fitted two-level calibration model.
    artifact:
        Persisted artifact bundle written under ``models/``.
    fit_duration_s:
        Wall-clock training time [s].
    """

    preparation: CalibrationCorpusPreparation
    result: TwoLevelCalibrationResult
    artifact: SavedCalibrationArtifact
    fit_duration_s: float


def build_corpus_calibration_output_dir(
    model_label: str,  # Human-readable label for the saved model
    models_root: Path = DEFAULT_CONFIGURATION_CONDITIONAL_MODELS_DIR,
) -> Path:
    """Build the output directory for one saved corpus-level calibration model."""

    if not str(model_label).strip():
        raise ValueError("model_label must be a non-empty string")
    return Path(models_root) / str(model_label)


def load_campaign_configuration(
    campaign_dir: Path,  # Directory that contains ``metadata.csv``
    metadata_filename: str = _DEFAULT_METADATA_FILENAME,
) -> CampaignConfiguration:
    """Parse the campaign metadata CSV into a typed configuration object.

    The repository currently ships a placeholder schema, but campaign metadata
    can also be extracted directly from the remote API client. Callers should
    therefore not depend on exact presentation labels. This parser resolves a
    small alias set for each required field and normalizes units to the
    physics-facing contract used by the numerical core.
    """

    metadata_path = Path(campaign_dir) / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Campaign metadata CSV does not exist: {metadata_path}"
        )

    with metadata_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        try:
            row = next(reader)
        except StopIteration as error:
            raise ValueError(
                f"Campaign metadata CSV is empty: {metadata_path}"
            ) from error
    if row is None:
        raise ValueError(f"Campaign metadata CSV is empty: {metadata_path}")

    normalized_row = {
        _normalize_metadata_key(key): value
        for key, value in row.items()
        if key is not None
    }

    return CampaignConfiguration(
        central_frequency_hz=_read_scaled_metadata_value(
            normalized_row,
            (
                ("centralfreqmhz", 1.0e6),
                ("centralfrequencymhz", 1.0e6),
                ("centerfreqmhz", 1.0e6),
                ("centerfrequencymhz", 1.0e6),
                ("centralfreqhz", 1.0),
                ("centralfrequencyhz", 1.0),
                ("centerfreqhz", 1.0),
                ("centerfrequencyhz", 1.0),
                ("centerfrequency", 1.0),
            ),
            field_name="central_frequency_hz",
        ),
        span_hz=_read_scaled_metadata_value(
            normalized_row,
            (
                ("spanmhz", 1.0e6),
                ("bandwidthmhz", 1.0e6),
                ("span", 1.0e6),
                ("spanhz", 1.0),
                ("bandwidthhz", 1.0),
            ),
            field_name="span_hz",
        ),
        resolution_bandwidth_hz=_read_scaled_metadata_value(
            normalized_row,
            (
                ("rbwkhz", 1.0e3),
                ("resolutionbandwidthkhz", 1.0e3),
                ("rbw", 1.0),
                ("rbwhz", 1.0),
                ("resolutionbandwidthhz", 1.0),
            ),
            field_name="resolution_bandwidth_hz",
        ),
        lna_gain_db=_read_scaled_metadata_value(
            normalized_row,
            (
                ("lnagaindb", 1.0),
                ("lnagain", 1.0),
                ("lna_gain_db", 1.0),
            ),
            field_name="lna_gain_db",
        ),
        vga_gain_db=_read_scaled_metadata_value(
            normalized_row,
            (
                ("vgagaindb", 1.0),
                ("vgagain", 1.0),
                ("vga_gain_db", 1.0),
            ),
            field_name="vga_gain_db",
        ),
        acquisition_interval_s=_read_scaled_metadata_value(
            normalized_row,
            (
                ("acquisitionfreqminutes", 60.0),
                ("acquisitionintervalminutes", 60.0),
                ("intervalseconds", 1.0),
                ("acquisitionfreqseconds", 1.0),
                ("acquisitionintervalseconds", 1.0),
            ),
            field_name="acquisition_interval_s",
        ),
        antenna_amplifier_enabled=_read_boolean_metadata_value(
            normalized_row,
            (
                "antennaamp",
                "antennaamplifierenabled",
                "antenna_amp",
            ),
            field_name="antenna_amplifier_enabled",
        ),
    )


def prepare_calibration_campaign(
    campaign_label: str,  # Campaign label under ``data/campaigns/``
    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR,
    excluded_sensor_ids: Collection[str] = (),
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
    alignment_tolerance_ms: int | None = None,
    sensor_file_pattern: str = "*.csv",
    metadata_filename: str = _DEFAULT_METADATA_FILENAME,
    allow_alignment_pruning: bool = True,
) -> PreparedCalibrationCampaign:
    """Prepare one campaign for the offline corpus.

    Responsibilities
    ----------------
    1. read and validate campaign metadata;
    2. load raw per-sensor series and apply explicit exclusions;
    3. align rows by timestamp, pruning only the sensors that destroy all
       shared overlap when ``allow_alignment_pruning`` is enabled;
    4. compute ranking diagnostics used for the reliable-sensor annotation;
    5. convert the aligned dB tensor to the linear-power campaign contract.

    Parameters
    ----------
    allow_alignment_pruning:
        Whether campaign preparation may automatically prune sensors when the
        full retained roster has no shared aligned record groups. The pruned
        sensors remain visible on the returned ``PreparedCalibrationCampaign``
        so the fallback stays auditable.
    """

    repository = FileSystemCampaignSensorDataRepository(
        campaigns_root=campaigns_root,
        sensor_file_pattern=sensor_file_pattern,
    )
    campaign_dir = Path(campaigns_root) / str(campaign_label)
    metadata_path = campaign_dir / metadata_filename
    configuration = load_campaign_configuration(
        campaign_dir=campaign_dir,
        metadata_filename=metadata_filename,
    )
    sensor_series_by_id = repository.load_campaign_sensor_series(campaign_label)
    retained_sensor_series_by_id, resolved_excluded_sensor_ids = (
        _exclude_campaign_sensor_series(
            sensor_series_by_id=sensor_series_by_id,
            excluded_sensor_ids=excluded_sensor_ids,
            campaign_label=campaign_label,
        )
    )

    aligned_alignment_result = align_campaign_sensor_series_with_pruning(
        campaign_label=campaign_label,
        sensor_series_by_id=retained_sensor_series_by_id,
        alignment_tolerance_ms=alignment_tolerance_ms,
        allow_pruning=allow_alignment_pruning,
    )
    aligned_dataset = aligned_alignment_result.dataset
    alignment_diagnostics = aligned_alignment_result.diagnostics
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
    campaign = CalibrationCampaign(
        campaign_label=str(campaign_label),
        sensor_ids=aligned_dataset.sensor_ids,
        frequency_hz=np.asarray(aligned_dataset.frequency_hz, dtype=np.float64),
        observations_power=power_db_to_linear(aligned_dataset.observations_db),
        configuration=configuration,
        reliable_sensor_id=reliable_sensor_id,
    )
    return PreparedCalibrationCampaign(
        campaign_label=str(campaign_label),
        campaign_dir=campaign_dir,
        metadata_path=metadata_path,
        aligned_dataset=aligned_dataset,
        campaign=campaign,
        ranking_result=ranking_result,
        distribution_diagnostics=distribution_diagnostics,
        alignment_diagnostics=alignment_diagnostics,
        reliable_sensor_id=reliable_sensor_id,
        excluded_sensor_ids=resolved_excluded_sensor_ids,
        alignment_pruned_sensor_ids=aligned_alignment_result.pruned_sensor_ids,
        distribution_outlier_sensor_ids=distribution_outlier_sensor_ids,
        ranking_histogram_bins=int(ranking_histogram_bins),
        distribution_histogram_bins=int(distribution_histogram_bins),
    )


def prepare_calibration_corpus(
    campaign_labels: Sequence[str] | None = None,
    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR,
    excluded_sensor_ids_by_campaign: Mapping[str, Collection[str]] | None = None,
    ranking_histogram_bins: int = 50,
    distribution_histogram_bins: int = 300,
    alignment_tolerance_ms: int | None = None,
    sensor_file_pattern: str = "*.csv",
    metadata_filename: str = _DEFAULT_METADATA_FILENAME,
    allow_alignment_pruning: bool = True,
) -> CalibrationCorpusPreparation:
    """Prepare a corpus of campaigns for the new offline calibration model.

    Parameters
    ----------
    allow_alignment_pruning:
        Whether each campaign may drop only the sensors that prevent any shared
        aligned record groups from existing. This is enabled by default because
        field corpora often contain a small number of sparse or shifted sensor
        files that would otherwise make the whole campaign unusable.
    """

    repository = FileSystemCampaignSensorDataRepository(
        campaigns_root=campaigns_root,
        sensor_file_pattern=sensor_file_pattern,
    )
    resolved_campaign_labels = (
        repository.list_campaign_labels()
        if campaign_labels is None
        else tuple(str(campaign_label) for campaign_label in campaign_labels)
    )
    if not resolved_campaign_labels:
        raise ValueError("No campaign labels were provided for corpus preparation")

    excluded_sensor_ids_by_campaign = (
        {}
        if excluded_sensor_ids_by_campaign is None
        else dict(excluded_sensor_ids_by_campaign)
    )
    prepared_campaigns = tuple(
        prepare_calibration_campaign(
            campaign_label=campaign_label,
            campaigns_root=campaigns_root,
            excluded_sensor_ids=excluded_sensor_ids_by_campaign.get(campaign_label, ()),
            ranking_histogram_bins=ranking_histogram_bins,
            distribution_histogram_bins=distribution_histogram_bins,
            alignment_tolerance_ms=alignment_tolerance_ms,
            sensor_file_pattern=sensor_file_pattern,
            metadata_filename=metadata_filename,
            allow_alignment_pruning=allow_alignment_pruning,
        )
        for campaign_label in resolved_campaign_labels
    )
    corpus = build_calibration_corpus(
        [prepared_campaign.campaign for prepared_campaign in prepared_campaigns]
    )
    return CalibrationCorpusPreparation(
        campaigns_root=Path(campaigns_root),
        prepared_campaigns=prepared_campaigns,
        corpus=corpus,
    )


def resolve_global_excluded_sensor_ids_by_campaign(
    campaign_labels: Sequence[str],  # Campaigns that participate in the workflow
    excluded_sensor_ids: Collection[
        str
    ],  # Global node exclusions requested by the caller
    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR,
    sensor_file_pattern: str = "*.csv",
) -> dict[str, tuple[str, ...]]:
    """Resolve one global exclusion list into campaign-specific exclusions.

    Purpose
    -------
    Notebook and application workflows often want to express exclusions once,
    for example ``["Node9", "Node10"]``, and apply them consistently across a
    multi-campaign training or deployment path. Campaigns rarely expose the
    exact same sensor roster, so this helper intersects the requested global
    exclusions with each campaign's available sensors.

    Parameters
    ----------
    campaign_labels:
        Campaign labels that should participate in the analysis.
    excluded_sensor_ids:
        Global sensor ids to exclude whenever they exist in a selected
        campaign.
    campaigns_root:
        Root directory that contains ``data/campaigns/<campaign_label>``.
    sensor_file_pattern:
        Glob pattern used to discover sensor CSV files for each campaign.

    Returns
    -------
    dict[str, tuple[str, ...]]
        One entry per requested campaign label, containing the sensor ids that
        should actually be excluded for that campaign.

    Side Effects
    ------------
    Performs filesystem reads through ``FileSystemCampaignSensorDataRepository``
    to discover each campaign's available sensor files.

    Raises
    ------
    ValueError
        If a requested excluded sensor id does not exist in any selected
        campaign. This keeps global typo handling explicit instead of silently
        accepting dead configuration.
    """

    resolved_campaign_labels = tuple(
        str(campaign_label) for campaign_label in campaign_labels
    )
    requested_excluded_sensor_ids = tuple(
        sorted(
            {
                str(sensor_id).strip()
                for sensor_id in excluded_sensor_ids
                if str(sensor_id).strip()
            }
        )
    )
    if not resolved_campaign_labels:
        return {}
    if not requested_excluded_sensor_ids:
        return {campaign_label: () for campaign_label in resolved_campaign_labels}

    repository = FileSystemCampaignSensorDataRepository(
        campaigns_root=campaigns_root,
        sensor_file_pattern=sensor_file_pattern,
    )
    resolved_excluded_sensor_ids_by_campaign: dict[str, tuple[str, ...]] = {}
    matched_sensor_id_set: set[str] = set()

    for campaign_label in resolved_campaign_labels:
        available_sensor_ids = tuple(
            sorted(repository.load_campaign_sensor_series(campaign_label))
        )
        resolved_sensor_ids = tuple(
            sensor_id
            for sensor_id in requested_excluded_sensor_ids
            if sensor_id in available_sensor_ids
        )
        matched_sensor_id_set.update(resolved_sensor_ids)
        resolved_excluded_sensor_ids_by_campaign[campaign_label] = resolved_sensor_ids

    unknown_sensor_ids = sorted(
        set(requested_excluded_sensor_ids).difference(matched_sensor_id_set)
    )
    if unknown_sensor_ids:
        raise ValueError(
            "Global excluded_sensor_ids contains sensors that do not exist in any "
            f"selected campaign: {unknown_sensor_ids}"
        )

    return resolved_excluded_sensor_ids_by_campaign


def fit_and_save_calibration_corpus_model(
    preparation: CalibrationCorpusPreparation,  # Prepared offline corpus
    output_dir: Path,  # Destination directory for the artifact bundle
    basis_config: FrequencyBasisConfig | None = None,
    model_config: PersistentModelConfig | None = None,
    fit_config: TwoLevelFitConfig | None = None,
    sensor_reference_weight_by_id: Mapping[str, float] | None = None,
    extra_summary: Mapping[str, int | float] | None = None,
    parameters_filename: str = DEFAULT_ARTIFACT_PARAMETERS_FILENAME,
    workflow_config_dir: Path | None = None,
) -> CalibrationCorpusFitResult:
    """Fit and persist one corpus-level configuration-conditional model.

    Purpose
    -------
    This service layer keeps notebook and application workflows away from the
    numerical fitting details. The domain layer returns a fitted result, while
    this boundary function measures wall-clock fit time and persists the
    resulting artifact bundle to disk.

    Parameters
    ----------
    preparation:
        Prepared offline corpus consumed by the numerical trainer.
    output_dir:
        Directory that will receive the saved artifact bundle.
    basis_config, model_config, fit_config:
        Optional model hyperparameters forwarded to the numerical core.
    sensor_reference_weight_by_id:
        Optional sensor-specific weighting override.
    extra_summary:
        Optional scalar metadata persisted into the artifact manifest.
    parameters_filename:
        Simple ``.npz`` filename used for the serialized parameter archive
        inside ``output_dir``.
    workflow_config_dir:
        Optional notebook workflow configuration directory. When provided, the
        saved artifact provenance stores a hash of the exact workflow files
        that selected the training and testing campaigns.

    Returns
    -------
    CalibrationCorpusFitResult
        Structured fit result that contains the prepared corpus, numerical
        result, saved artifact metadata, and measured fit duration.

    Side Effects
    ------------
    Runs the numerical training routine and writes the artifact bundle to disk.
    """

    start_time = time.perf_counter()
    result = fit_two_level_calibration(
        corpus=preparation.corpus,
        basis_config=basis_config,
        model_config=model_config,
        fit_config=fit_config,
        sensor_reference_weight_by_id=sensor_reference_weight_by_id,
    )
    fit_duration_s = time.perf_counter() - start_time
    summary_payload = dict(extra_summary or {})
    summary_payload["fit_duration_s"] = fit_duration_s
    summary_payload["n_campaigns"] = len(preparation.prepared_campaigns)
    workflow_config_fingerprint = (
        None
        if workflow_config_dir is None
        else fingerprint_notebook_workflow_config(workflow_config_dir)
    )
    artifact = save_two_level_calibration_artifact(
        output_dir=output_dir,
        result=result,
        extra_summary=summary_payload,
        parameters_filename=parameters_filename,
        workflow_config_fingerprint=workflow_config_fingerprint,
    )
    return CalibrationCorpusFitResult(
        preparation=preparation,
        result=result,
        artifact=artifact,
        fit_duration_s=fit_duration_s,
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
            f"At least two sensors must remain after exclusions for {campaign_label!r}"
        )

    return retained_sensor_series_by_id, tuple(sorted(excluded_sensor_id_set))


def _distribution_outlier_sensor_ids(
    distribution_diagnostics: SensorDistributionDiagnostics,
) -> tuple[str, ...]:
    """Return sensors flagged as low-similarity PSD-distribution outliers."""

    outlier_sensor_ids = [
        sensor_id
        for sensor_id, score in zip(
            distribution_diagnostics.sensor_ids,
            distribution_diagnostics.cross_similarity_score,
            strict=True,
        )
        if score < distribution_diagnostics.outlier_threshold
    ]
    return tuple(outlier_sensor_ids)


def _select_reliable_sensor_id(
    ranking_result: SensorRankingResult,
    distribution_outlier_sensor_ids: Collection[str],
) -> str:
    """Choose the reliable sensor using the conservative two-stage rule."""

    outlier_sensor_id_set = set(distribution_outlier_sensor_ids)
    for sensor_id in ranking_result.ranking_sensor_ids:
        if sensor_id not in outlier_sensor_id_set:
            return sensor_id
    return ranking_result.ranking_sensor_ids[0]


def _normalize_metadata_key(key: str) -> str:
    """Normalize a metadata column label so aliases remain presentation-agnostic."""

    return re.sub(r"[^a-z0-9]+", "", str(key).strip().lower())


def _read_scaled_metadata_value(
    normalized_row: Mapping[str, str | None],
    aliases_with_scale: Sequence[tuple[str, float]],
    field_name: str,
) -> float:
    """Read one numeric metadata value and convert it to the target units."""

    for alias, scale in aliases_with_scale:
        raw_value = normalized_row.get(alias)
        if raw_value is None or not str(raw_value).strip():
            continue
        try:
            return float(raw_value) * float(scale)
        except ValueError as error:
            raise ValueError(
                f"Campaign metadata field {field_name!r} contains an invalid value: "
                f"{raw_value!r}"
            ) from error
    alias_labels = ", ".join(alias for alias, _ in aliases_with_scale)
    raise ValueError(
        f"Campaign metadata is missing the required field {field_name!r}. "
        f"Tried aliases: {alias_labels}"
    )


def _read_boolean_metadata_value(
    normalized_row: Mapping[str, str | None],
    aliases: Sequence[str],
    field_name: str,
) -> bool:
    """Read one boolean metadata field using a permissive string parser."""

    for alias in aliases:
        raw_value = normalized_row.get(alias)
        if raw_value is None or not str(raw_value).strip():
            continue
        normalized_value = str(raw_value).strip().lower()
        if normalized_value in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized_value in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(
            f"Campaign metadata field {field_name!r} contains an invalid boolean "
            f"value: {raw_value!r}"
        )
    alias_labels = ", ".join(aliases)
    raise ValueError(
        f"Campaign metadata is missing the required field {field_name!r}. "
        f"Tried aliases: {alias_labels}"
    )


__all__ = [
    "CalibrationCorpusFitResult",
    "CalibrationCorpusPreparation",
    "DEFAULT_CONFIGURATION_CONDITIONAL_MODELS_DIR",
    "PreparedCalibrationCampaign",
    "build_corpus_calibration_output_dir",
    "fit_and_save_calibration_corpus_model",
    "load_campaign_configuration",
    "prepare_calibration_campaign",
    "prepare_calibration_corpus",
    "resolve_global_excluded_sensor_ids_by_campaign",
]
