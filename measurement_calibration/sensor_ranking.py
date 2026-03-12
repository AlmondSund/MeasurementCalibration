"""Correlation-based sensor ranking helpers for RBW and campaign datasets.

The original exploratory notebooks focused on RBW-specific directory trees
where each subdirectory already contained row-aligned sensor CSV files. The
current module keeps those helpers intact, but it also provides a more general
campaign-analysis framework for datasets stored under ``data/campaigns/``:

- infrastructure/repository code loads per-sensor CSV files from disk,
- orchestration code aligns campaign rows by timestamp when necessary, and
- domain code computes the distribution and record-wise ranking diagnostics.

The ranking mathematics remain pure and deterministic. Filesystem access lives
in explicit loader and repository boundaries so the core ranking logic remains
easy to test and reason about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import csv
from itertools import combinations
import json
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]

_STD_EPSILON = 1.0e-8
DEFAULT_CAMPAIGNS_DATA_DIR = Path("data") / "campaigns"
_DEFAULT_CAMPAIGN_SENSOR_FILE_PATTERN = "*.csv"
_METADATA_FILENAME = "metadata.csv"
_MIN_ALIGNMENT_TOLERANCE_MS = 250
_MAX_ALIGNMENT_TOLERANCE_MS = 5_000
_ALIGNMENT_TOLERANCE_PERIOD_FRACTION = 0.25
_MAX_EXACT_ALIGNMENT_SUBSET_SEARCH_SENSOR_COUNT = 12
_NO_SHARED_ALIGNMENT_ERROR_PREFIX = (
    "Campaign alignment could not find any record groups shared across all sensors"
)


@dataclass(frozen=True)
class SensorMeasurementSeries:
    """Raw measurement rows for one sensor before campaign alignment.

    Parameters
    ----------
    sensor_id:
        Sensor identifier derived from the CSV stem or the caller's mapping.
    frequency_hz:
        Shared PSD frequency grid for every record in this file [Hz].
    observations_db:
        PSD values in dB with shape ``(n_records, n_frequencies)``.
    timestamps_ms:
        Record timestamps with shape ``(n_records,)`` [ms].
    source_row_indices:
        Original row positions retained from the CSV file. This is useful when
        a caller sorts or aligns rows and still wants to trace each aligned
        record back to the source file.
    """

    sensor_id: str
    frequency_hz: FloatArray
    observations_db: FloatArray
    timestamps_ms: IndexArray
    source_row_indices: IndexArray

    @property
    def n_records(self) -> int:
        """Return the number of records available for this sensor."""

        return int(self.observations_db.shape[0])

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency bins per PSD record."""

        return int(self.observations_db.shape[1])


@dataclass(frozen=True)
class RbwAcquisitionDataset:
    """Row-aligned acquisition subset for one dataset label.

    Parameters
    ----------
    rbw_label:
        Legacy label name kept for backward compatibility. The same field is
        also used as the generic dataset label for campaign analysis, for
        example ``"10K"`` or ``"test-calibration"``.
    sensor_ids:
        Ordered sensor identifiers derived from the CSV file names.
    frequency_hz:
        Shared PSD frequency grid [Hz].
    observations_db:
        PSD values in dB with shape ``(n_sensors, n_records, n_frequencies)``.
    timestamps_ms:
        Acquisition timestamps with shape ``(n_sensors, n_records)`` [ms].
    """

    rbw_label: str
    sensor_ids: tuple[str, ...]
    frequency_hz: FloatArray
    observations_db: FloatArray
    timestamps_ms: IndexArray

    @property
    def n_sensors(self) -> int:
        """Return the number of sensors in the RBW subset."""

        return int(self.observations_db.shape[0])

    @property
    def n_records(self) -> int:
        """Return the number of row-aligned records per sensor."""

        return int(self.observations_db.shape[1])

    @property
    def n_frequencies(self) -> int:
        """Return the number of frequency bins per PSD vector."""

        return int(self.observations_db.shape[2])

    @property
    def dataset_label(self) -> str:
        """Return the generic dataset label used by campaign analysis."""

        return self.rbw_label


@dataclass(frozen=True)
class SensorRankingResult:
    """Correlation-based ranking output for one aligned dataset.

    Parameters
    ----------
    rbw_label:
        Legacy label name kept for backward compatibility with the RBW path.
    sensor_ids:
        Ordered sensor identifiers matching the score arrays.
    per_record_score:
        Cumulative agreement scores with shape ``(n_sensors, n_records)``.
        Each value is the sum of a sensor's pairwise Pearson correlations to
        the remaining sensors on one aligned record.
    average_score:
        Mean cumulative agreement score per sensor across all records.
    average_correlation:
        ``average_score`` normalized by ``n_sensors - 1`` so it is easier to
        interpret on the Pearson correlation scale.
    noise_floor_db:
        Per-record histogram-mode noise-floor estimate for every sensor [dB].
    global_noise_floor_db:
        Record-wise average noise-floor target used for recentering [dB].
    per_record_correlation:
        Pairwise Pearson correlation matrices with shape
        ``(n_records, n_sensors, n_sensors)`` after the noise-floor recentering
        and standardization steps. This keeps the notebook-side diagnostics
        aligned with the actual ranking metric rather than recomputing it from
        scratch with slightly different assumptions.
    ranking_sensor_ids:
        Sensor identifiers sorted from best to worst mean score.
    """

    rbw_label: str
    sensor_ids: tuple[str, ...]
    per_record_score: FloatArray
    average_score: FloatArray
    average_correlation: FloatArray
    noise_floor_db: FloatArray
    global_noise_floor_db: FloatArray
    per_record_correlation: FloatArray
    ranking_sensor_ids: tuple[str, ...]

    @property
    def dataset_label(self) -> str:
        """Return the generic dataset label used by campaign analysis."""

        return self.rbw_label


@dataclass(frozen=True)
class SensorDistributionDiagnostics:
    """Dataset-wide PSD distribution diagnostics for one aligned dataset.

    Parameters
    ----------
    rbw_label:
        Legacy label name kept for backward compatibility with the RBW path.
    sensor_ids:
        Ordered sensor identifiers matching the density and score arrays.
    bin_edges_db:
        Histogram bin edges used for the density estimates [dB].
    bin_centers_db:
        Histogram bin centers corresponding to the density vectors [dB].
    global_density:
        Probability-density estimate built from every PSD value across sensors.
    per_sensor_density:
        Per-sensor probability densities with shape ``(n_sensors, n_bins)``.
    correlation_matrix:
        Pearson correlation matrix between the per-sensor density curves.
    mean_db, std_db, min_db, max_db:
        Basic descriptive statistics of the flattened PSD values per sensor [dB].
    value_count:
        Number of PSD samples contributed by each sensor to the histogram view.
    cross_similarity_score:
        Row-wise sum of ``correlation_matrix`` minus one, matching the
        histogram-shape ranking used in the exploratory notebooks.
    normalized_similarity_score:
        ``cross_similarity_score`` normalized by ``n_sensors - 1`` so it lives
        on the Pearson correlation scale.
    outlier_threshold:
        Heuristic threshold ``mean(score) - std(score)`` used to flag sensors
        whose PSD distributions disagree with the rest of the network.
    clipped_fraction:
        Fraction of PSD values outside ``value_range_db`` that were excluded
        from the histogram counts. It is zero when the range is inferred from
        the data itself.
    ranking_sensor_ids:
        Sensor identifiers sorted from best to worst distribution similarity.
    """

    rbw_label: str
    sensor_ids: tuple[str, ...]
    bin_edges_db: FloatArray
    bin_centers_db: FloatArray
    global_density: FloatArray
    per_sensor_density: FloatArray
    correlation_matrix: FloatArray
    mean_db: FloatArray
    std_db: FloatArray
    min_db: FloatArray
    max_db: FloatArray
    value_count: IndexArray
    cross_similarity_score: FloatArray
    normalized_similarity_score: FloatArray
    outlier_threshold: float
    clipped_fraction: float
    ranking_sensor_ids: tuple[str, ...]

    @property
    def dataset_label(self) -> str:
        """Return the generic dataset label used by campaign analysis."""

        return self.rbw_label


@dataclass(frozen=True)
class CampaignAlignmentDiagnostics:
    """Timestamp-alignment diagnostics for one campaign dataset.

    Parameters
    ----------
    campaign_label:
        Campaign label resolved from ``data/campaigns/<campaign_label>``.
    sensor_ids:
        Ordered sensor identifiers that survived campaign loading.
    anchor_sensor_id:
        Sensor used as the alignment anchor. The implementation chooses the
        shortest sensor series so the alignment only keeps record groups that
        are observable across the full retained network.
    alignment_tolerance_ms:
        Maximum timestamp mismatch allowed when matching one record group [ms].
    source_record_count:
        Number of raw CSV rows available per sensor before alignment.
    dropped_record_count:
        Number of rows discarded per sensor because they could not be matched
        across the full sensor set while respecting the tolerance.
    aligned_row_indices:
        Source row positions retained for each sensor after sorting and
        alignment. The tuple order follows ``sensor_ids``.
    record_time_spread_ms:
        Timestamp spread across sensors for each aligned record group [ms].
    """

    campaign_label: str
    sensor_ids: tuple[str, ...]
    anchor_sensor_id: str
    alignment_tolerance_ms: int
    source_record_count: IndexArray
    dropped_record_count: IndexArray
    aligned_row_indices: tuple[IndexArray, ...]
    record_time_spread_ms: FloatArray

    @property
    def aligned_record_count(self) -> int:
        """Return the number of aligned campaign records retained."""

        if not self.aligned_row_indices:
            return 0
        return int(self.aligned_row_indices[0].size)

    @property
    def mean_record_time_spread_ms(self) -> float:
        """Return the mean cross-sensor timestamp spread of aligned records."""

        if self.record_time_spread_ms.size == 0:
            return 0.0
        return float(np.mean(self.record_time_spread_ms))

    @property
    def max_record_time_spread_ms(self) -> float:
        """Return the maximum cross-sensor timestamp spread of aligned records."""

        if self.record_time_spread_ms.size == 0:
            return 0.0
        return float(np.max(self.record_time_spread_ms))


@dataclass(frozen=True)
class AlignmentPrunedCampaignDataset:
    """Aligned campaign dataset plus the sensors pruned to make it feasible.

    Parameters
    ----------
    dataset:
        Row-aligned dataset built from the retained sensor subset.
    diagnostics:
        Alignment diagnostics for the retained subset.
    pruned_sensor_ids:
        Sensors removed because no full-network aligned record groups existed.
        This keeps the fallback explicit instead of quietly pretending the
        original campaign roster was usable as-is.
    """

    dataset: RbwAcquisitionDataset
    diagnostics: CampaignAlignmentDiagnostics
    pruned_sensor_ids: tuple[str, ...]


@dataclass(frozen=True)
class SensorRankingAnalysisConfig:
    """Configuration for campaign-oriented sensor ranking analysis.

    Parameters
    ----------
    ranking_histogram_bins:
        Histogram bins used by the record-wise ranking noise-floor estimate.
    distribution_histogram_bins:
        Histogram bins used by the dataset-wide PSD distribution diagnostic.
    campaign_sensor_file_pattern:
        Glob pattern used to discover sensor CSV files inside one campaign
        directory.
    alignment_tolerance_ms:
        Optional explicit timestamp tolerance used when aligning campaign rows.
        When omitted, the analyzer infers a conservative tolerance from the
        median campaign sampling period.
    """

    ranking_histogram_bins: int = 50
    distribution_histogram_bins: int = 300
    campaign_sensor_file_pattern: str = _DEFAULT_CAMPAIGN_SENSOR_FILE_PATTERN
    alignment_tolerance_ms: int | None = None

    def __post_init__(self) -> None:
        """Validate the campaign-analysis configuration at construction time."""

        if self.ranking_histogram_bins < 1:
            raise ValueError("ranking_histogram_bins must be at least 1")
        if self.distribution_histogram_bins < 1:
            raise ValueError("distribution_histogram_bins must be at least 1")
        if not self.campaign_sensor_file_pattern.strip():
            raise ValueError("campaign_sensor_file_pattern must be non-empty")
        if self.alignment_tolerance_ms is not None and self.alignment_tolerance_ms < 0:
            raise ValueError("alignment_tolerance_ms cannot be negative")


@dataclass(frozen=True)
class CampaignSensorRankingAnalysis:
    """End-to-end ranking analysis for one dataset under ``data/campaigns/``.

    Parameters
    ----------
    campaign_label:
        Campaign label supplied by the caller.
    dataset:
        Timestamp-aligned dataset passed into the ranking core.
    ranking_result:
        Record-wise cumulative-correlation ranking output.
    distribution_diagnostics:
        Dataset-wide PSD distribution diagnostic for the same aligned dataset.
    alignment_diagnostics:
        Summary of how raw campaign rows were aligned before ranking.
    """

    campaign_label: str
    dataset: RbwAcquisitionDataset
    ranking_result: SensorRankingResult
    distribution_diagnostics: SensorDistributionDiagnostics
    alignment_diagnostics: CampaignAlignmentDiagnostics


class CampaignSensorDataRepository(Protocol):
    """Boundary that supplies per-sensor campaign series to the analyzer."""

    def list_campaign_labels(self) -> tuple[str, ...]:
        """Return available campaign labels that contain at least one CSV."""

    def load_campaign_sensor_series(
        self,
        campaign_label: str,  # Campaign label stored under data/campaigns/
    ) -> dict[str, SensorMeasurementSeries]:
        """Load one campaign into raw per-sensor measurement series."""


@dataclass(frozen=True)
class FileSystemCampaignSensorDataRepository:
    """Filesystem-backed repository for ``data/campaigns/<campaign_label>``.

    Parameters
    ----------
    campaigns_root:
        Root directory that contains one subdirectory per campaign label.
    sensor_file_pattern:
        Glob pattern used to discover sensor CSV files inside each campaign
        directory. The default accepts any CSV so the repository adapts to
        campaigns that do not use the historical ``Node*.csv`` naming.
    """

    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR
    sensor_file_pattern: str = _DEFAULT_CAMPAIGN_SENSOR_FILE_PATTERN

    def list_campaign_labels(self) -> tuple[str, ...]:
        """Return campaign labels that contain at least one matching CSV file."""

        root_dir = Path(self.campaigns_root)
        if not root_dir.exists():
            raise FileNotFoundError(f"Campaign data root does not exist: {root_dir}")

        labels = [
            campaign_dir.name
            for campaign_dir in sorted(
                path for path in root_dir.iterdir() if path.is_dir()
            )
            if any(
                path.is_file() and path.name.lower() != _METADATA_FILENAME
                for path in campaign_dir.glob(self.sensor_file_pattern)
            )
        ]
        return tuple(labels)

    def load_campaign_sensor_series(
        self,
        campaign_label: str,  # Campaign label stored under data/campaigns/
    ) -> dict[str, SensorMeasurementSeries]:
        """Load one campaign directory into raw, timestamp-sorted sensor series."""

        campaign_dir = Path(self.campaigns_root) / str(campaign_label)
        if not campaign_dir.exists():
            raise FileNotFoundError(
                f"Campaign directory does not exist: {campaign_dir}"
            )
        return _load_sensor_series_directory(
            campaign_dir,
            sensor_file_pattern=self.sensor_file_pattern,
            sort_by_timestamp=True,
        )


@dataclass(frozen=True)
class SensorRankingAnalyzer:
    """Service layer that runs dynamic campaign ranking analysis.

    The analyzer owns the application workflow:

    1. load raw per-sensor campaign series from the repository,
    2. align rows by timestamp to satisfy the ranking core invariants, and
    3. compute both the record-wise and distribution-wide diagnostics.
    """

    repository: CampaignSensorDataRepository
    config: SensorRankingAnalysisConfig = SensorRankingAnalysisConfig()

    def analyze_campaign(
        self,
        campaign_label: str,  # Label under data/campaigns/
    ) -> CampaignSensorRankingAnalysis:
        """Analyze one campaign label end to end.

        Side Effects
        ------------
        Filesystem reads occur through the injected repository. No writes occur.
        """

        sensor_series = self.repository.load_campaign_sensor_series(campaign_label)
        dataset, alignment_diagnostics = align_campaign_sensor_series(
            campaign_label=campaign_label,
            sensor_series_by_id=sensor_series,
            alignment_tolerance_ms=self.config.alignment_tolerance_ms,
        )
        ranking_result = rank_sensors_by_cumulative_correlation(
            dataset,
            histogram_bins=self.config.ranking_histogram_bins,
        )
        distribution_diagnostics = summarize_psd_distribution(
            dataset,
            histogram_bins=self.config.distribution_histogram_bins,
        )
        return CampaignSensorRankingAnalysis(
            campaign_label=str(campaign_label),
            dataset=dataset,
            ranking_result=ranking_result,
            distribution_diagnostics=distribution_diagnostics,
            alignment_diagnostics=alignment_diagnostics,
        )

    def analyze_all_campaigns(
        self,
    ) -> dict[str, CampaignSensorRankingAnalysis]:
        """Analyze every campaign label exposed by the repository."""

        analyses: dict[str, CampaignSensorRankingAnalysis] = {}
        for campaign_label in self.repository.list_campaign_labels():
            analyses[campaign_label] = self.analyze_campaign(campaign_label)
        if not analyses:
            raise ValueError("No campaign datasets were found for sensor ranking")
        return analyses


def load_rbw_acquisition_datasets(
    root_dir: Path,  # Directory that contains one subdirectory per RBW
) -> dict[str, RbwAcquisitionDataset]:  # Parsed RBW datasets indexed by label
    """Load every RBW acquisition subset from a directory tree.

    The loader expects a structure such as ``root_dir / "10K" / "Node1.csv"``.
    Every sensor inside one RBW directory must expose the same number of
    aligned records and the same PSD frequency grid, because the ranking
    compares sensors row by row.
    """

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"RBW acquisition root does not exist: {root_dir}")

    datasets: dict[str, RbwAcquisitionDataset] = {}
    for rbw_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        try:
            sensor_series = _load_sensor_series_directory(
                rbw_dir,
                sensor_file_pattern="Node*.csv",
                sort_by_timestamp=False,
            )
        except ValueError:
            continue
        datasets[rbw_dir.name] = _stack_aligned_sensor_series(
            dataset_label=rbw_dir.name,
            sensor_series_by_id=sensor_series,
        )

    if not datasets:
        raise ValueError(f"No RBW acquisition CSV files were found under {root_dir}")
    return datasets


def analyze_campaign_sensor_ranking(
    campaign_label: str,  # Dataset label under data/campaigns/
    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR,  # Root containing campaign dirs
    config: SensorRankingAnalysisConfig | None = None,
) -> CampaignSensorRankingAnalysis:
    """Analyze one stored campaign under ``data/campaigns/<campaign_label>``.

    This is the convenience entry point for callers that only need one
    campaign analysis and do not want to instantiate the repository and
    analyzer classes explicitly.
    """

    resolved_config = SensorRankingAnalysisConfig() if config is None else config
    repository = FileSystemCampaignSensorDataRepository(
        campaigns_root=campaigns_root,
        sensor_file_pattern=resolved_config.campaign_sensor_file_pattern,
    )
    analyzer = SensorRankingAnalyzer(repository=repository, config=resolved_config)
    return analyzer.analyze_campaign(campaign_label)


def analyze_all_campaign_sensor_rankings(
    campaigns_root: Path = DEFAULT_CAMPAIGNS_DATA_DIR,  # Root containing campaign dirs
    config: SensorRankingAnalysisConfig | None = None,
) -> dict[str, CampaignSensorRankingAnalysis]:
    """Analyze every campaign dataset stored under ``data/campaigns/``."""

    resolved_config = SensorRankingAnalysisConfig() if config is None else config
    repository = FileSystemCampaignSensorDataRepository(
        campaigns_root=campaigns_root,
        sensor_file_pattern=resolved_config.campaign_sensor_file_pattern,
    )
    analyzer = SensorRankingAnalyzer(repository=repository, config=resolved_config)
    return analyzer.analyze_all_campaigns()


def build_campaign_alignment_rows(
    diagnostics: CampaignAlignmentDiagnostics,  # Alignment summary for one campaign
) -> list[dict[str, float | int | str | bool]]:
    """Build a table-friendly summary of campaign timestamp alignment."""

    rows: list[dict[str, float | int | str | bool]] = []
    aligned_record_count = diagnostics.aligned_record_count
    for sensor_index, sensor_id in enumerate(diagnostics.sensor_ids):
        source_record_count = int(diagnostics.source_record_count[sensor_index])
        dropped_record_count = int(diagnostics.dropped_record_count[sensor_index])
        rows.append(
            {
                "campaign_label": diagnostics.campaign_label,
                "sensor_id": sensor_id,
                "is_anchor_sensor": sensor_id == diagnostics.anchor_sensor_id,
                "source_records": source_record_count,
                "aligned_records": aligned_record_count,
                "dropped_records": dropped_record_count,
                "retained_fraction": (
                    float(aligned_record_count / source_record_count)
                    if source_record_count > 0
                    else 0.0
                ),
                "alignment_tolerance_ms": diagnostics.alignment_tolerance_ms,
                "mean_record_time_spread_ms": diagnostics.mean_record_time_spread_ms,
                "max_record_time_spread_ms": diagnostics.max_record_time_spread_ms,
            }
        )
    return rows


def align_campaign_sensor_series(
    campaign_label: str,  # Campaign label resolved by the caller
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],  # Raw per-sensor rows
    alignment_tolerance_ms: int | None = None,
) -> tuple[RbwAcquisitionDataset, CampaignAlignmentDiagnostics]:
    """Align a campaign's raw sensor rows by timestamp.

    The ranking core requires one row-aligned PSD tensor. Campaign downloads do
    not guarantee that every sensor has the same number of rows, so this
    adapter keeps only record groups that can be matched across the full sensor
    set within a configurable timestamp tolerance.
    """

    if len(sensor_series_by_id) < 2:
        raise ValueError("Campaign ranking requires at least two sensors")

    sorted_series_by_id = {
        sensor_id: _sort_sensor_series_by_timestamp(series)
        for sensor_id, series in sensor_series_by_id.items()
    }
    sensor_ids = tuple(sorted(sorted_series_by_id))
    reference_frequency_hz = _validate_shared_frequency_grid(
        sorted_series_by_id,
        context_label=str(campaign_label),
    )
    resolved_tolerance_ms = _infer_alignment_tolerance_ms(
        sorted_series_by_id=sensor_series_by_id,
        alignment_tolerance_ms=alignment_tolerance_ms,
    )

    # Use the shortest sensor as the anchor so every retained group is
    # observable across the full campaign sensor set.
    anchor_sensor_id = min(
        sensor_ids,
        key=lambda sensor_id: (
            sorted_series_by_id[sensor_id].n_records,
            sensor_id,
        ),
    )
    anchor_series = sorted_series_by_id[anchor_sensor_id]
    next_start_by_sensor = {sensor_id: 0 for sensor_id in sensor_ids}
    aligned_row_indices_by_sensor: dict[str, list[int]] = {
        sensor_id: [] for sensor_id in sensor_ids
    }
    aligned_timestamps_by_sensor: dict[str, list[int]] = {
        sensor_id: [] for sensor_id in sensor_ids
    }
    aligned_observations_by_sensor: dict[str, list[FloatArray]] = {
        sensor_id: [] for sensor_id in sensor_ids
    }

    # Greedy matching preserves time order and drops unmatched rows instead of
    # silently fabricating pairings across distant timestamps.
    for anchor_position, anchor_timestamp_ms in enumerate(anchor_series.timestamps_ms):
        candidate_positions = {anchor_sensor_id: anchor_position}
        is_match = True

        for sensor_id in sensor_ids:
            if sensor_id == anchor_sensor_id:
                continue

            sensor_series = sorted_series_by_id[sensor_id]
            candidate_position = _find_nearest_timestamp_index(
                sorted_timestamps_ms=sensor_series.timestamps_ms,
                target_timestamp_ms=int(anchor_timestamp_ms),
                start_index=next_start_by_sensor[sensor_id],
                alignment_tolerance_ms=resolved_tolerance_ms,
            )
            if candidate_position is None:
                is_match = False
                break

            candidate_positions[sensor_id] = candidate_position

        if not is_match:
            continue

        for sensor_id in sensor_ids:
            sensor_series = sorted_series_by_id[sensor_id]
            candidate_position = candidate_positions[sensor_id]
            next_start_by_sensor[sensor_id] = candidate_position + 1
            aligned_row_indices_by_sensor[sensor_id].append(
                int(sensor_series.source_row_indices[candidate_position])
            )
            aligned_timestamps_by_sensor[sensor_id].append(
                int(sensor_series.timestamps_ms[candidate_position])
            )
            aligned_observations_by_sensor[sensor_id].append(
                np.asarray(
                    sensor_series.observations_db[candidate_position], dtype=np.float64
                )
            )

    if not aligned_row_indices_by_sensor[anchor_sensor_id]:
        raise ValueError(
            "Campaign alignment could not find any record groups shared across "
            f"all sensors in {campaign_label!r}"
        )

    observations_db = np.stack(
        [
            np.stack(aligned_observations_by_sensor[sensor_id], axis=0)
            for sensor_id in sensor_ids
        ],
        axis=0,
    )
    timestamps_ms = np.asarray(
        [aligned_timestamps_by_sensor[sensor_id] for sensor_id in sensor_ids],
        dtype=np.int64,
    )
    aligned_row_indices = tuple(
        np.asarray(aligned_row_indices_by_sensor[sensor_id], dtype=np.int64)
        for sensor_id in sensor_ids
    )
    source_record_count = np.asarray(
        [sorted_series_by_id[sensor_id].n_records for sensor_id in sensor_ids],
        dtype=np.int64,
    )
    dropped_record_count = source_record_count - int(aligned_row_indices[0].size)
    record_time_spread_ms = (
        np.max(timestamps_ms, axis=0) - np.min(timestamps_ms, axis=0)
    ).astype(np.float64)

    dataset = RbwAcquisitionDataset(
        rbw_label=str(campaign_label),
        sensor_ids=sensor_ids,
        frequency_hz=np.asarray(reference_frequency_hz, dtype=np.float64),
        observations_db=np.asarray(observations_db, dtype=np.float64),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
    )
    diagnostics = CampaignAlignmentDiagnostics(
        campaign_label=str(campaign_label),
        sensor_ids=sensor_ids,
        anchor_sensor_id=anchor_sensor_id,
        alignment_tolerance_ms=int(resolved_tolerance_ms),
        source_record_count=source_record_count,
        dropped_record_count=np.asarray(dropped_record_count, dtype=np.int64),
        aligned_row_indices=aligned_row_indices,
        record_time_spread_ms=np.asarray(record_time_spread_ms, dtype=np.float64),
    )
    return dataset, diagnostics


def align_campaign_sensor_series_with_pruning(
    campaign_label: str,  # Campaign label resolved by the caller
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],  # Raw per-sensor rows
    alignment_tolerance_ms: int | None = None,
    *,
    allow_pruning: bool = True,  # Whether to prune sensors when full overlap fails
) -> AlignmentPrunedCampaignDataset:
    """Align a campaign, pruning only the sensors that destroy shared overlap.

    Purpose
    -------
    Real campaign downloads can contain one or two sparse or time-shifted
    sensors that prevent any record group from being shared across the full
    retained sensor roster. The strict ``align_campaign_sensor_series`` helper
    should keep failing loudly in that situation because ranking workflows may
    want the full-set guarantee. Corpus preparation, however, benefits from a
    more resilient policy: keep the largest alignable subset and make the
    pruned sensors explicit.

    Parameters
    ----------
    campaign_label:
        Campaign label resolved by the caller.
    sensor_series_by_id:
        Raw per-sensor measurement rows before timestamp alignment.
    alignment_tolerance_ms:
        Optional explicit timestamp mismatch tolerance [ms].
    allow_pruning:
        Whether the helper may prune sensors after a strict full-set alignment
        failure. When ``False`` this function behaves like the strict aligner
        while still returning a structured result object on success.

    Returns
    -------
    AlignmentPrunedCampaignDataset
        Aligned dataset, diagnostics for the retained sensor subset, and the
        sensors that had to be pruned to recover a usable campaign.
    """

    dataset, diagnostics = _try_align_campaign_subset(
        campaign_label=campaign_label,
        sensor_series_by_id=sensor_series_by_id,
        retained_sensor_ids=tuple(sorted(sensor_series_by_id)),
        alignment_tolerance_ms=alignment_tolerance_ms,
        raise_on_no_shared_overlap=not allow_pruning,
    )
    if dataset is not None and diagnostics is not None:
        return AlignmentPrunedCampaignDataset(
            dataset=dataset,
            diagnostics=diagnostics,
            pruned_sensor_ids=(),
        )
    if not allow_pruning:
        raise AssertionError("Strict alignment should have raised before this point")

    subset_resolution = _find_best_alignable_sensor_subset(
        campaign_label=campaign_label,
        sensor_series_by_id=sensor_series_by_id,
        alignment_tolerance_ms=alignment_tolerance_ms,
    )
    if subset_resolution is None:
        raise ValueError(
            "Campaign alignment could not find any record groups shared across "
            "any retained sensor subset with at least two sensors in "
            f"{campaign_label!r}"
        )

    return AlignmentPrunedCampaignDataset(
        dataset=subset_resolution.dataset,
        diagnostics=subset_resolution.diagnostics,
        pruned_sensor_ids=subset_resolution.pruned_sensor_ids,
    )


@dataclass(frozen=True)
class _AlignmentSubsetResolution:
    """Successful alignment result for one retained sensor subset."""

    dataset: RbwAcquisitionDataset
    diagnostics: CampaignAlignmentDiagnostics
    pruned_sensor_ids: tuple[str, ...]


def _find_best_alignable_sensor_subset(
    campaign_label: str,
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    alignment_tolerance_ms: int | None,
) -> _AlignmentSubsetResolution | None:
    """Return the largest, best-aligned sensor subset for one campaign.

    The search favors:
    1. more retained sensors;
    2. more aligned record groups;
    3. higher retained-row fractions across the surviving sensors.

    Exact subset search is cheap for the small sensor counts used in the
    current calibration corpus. A greedy fallback is kept for larger rosters so
    this helper stays bounded even if future campaigns grow substantially.
    """

    sensor_ids = tuple(sorted(sensor_series_by_id))
    if len(sensor_ids) <= _MAX_EXACT_ALIGNMENT_SUBSET_SEARCH_SENSOR_COUNT:
        return _find_best_alignable_sensor_subset_exact(
            campaign_label=campaign_label,
            sensor_series_by_id=sensor_series_by_id,
            alignment_tolerance_ms=alignment_tolerance_ms,
        )
    return _find_best_alignable_sensor_subset_greedy(
        campaign_label=campaign_label,
        sensor_series_by_id=sensor_series_by_id,
        alignment_tolerance_ms=alignment_tolerance_ms,
    )


def _find_best_alignable_sensor_subset_exact(
    campaign_label: str,
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    alignment_tolerance_ms: int | None,
) -> _AlignmentSubsetResolution | None:
    """Exhaustively search the best alignable subset for modest sensor counts."""

    sensor_ids = tuple(sorted(sensor_series_by_id))
    best_resolution: _AlignmentSubsetResolution | None = None

    for subset_size in range(len(sensor_ids) - 1, 1, -1):
        best_resolution = None
        for retained_sensor_ids in combinations(sensor_ids, subset_size):
            candidate = _resolve_alignment_subset_candidate(
                campaign_label=campaign_label,
                sensor_series_by_id=sensor_series_by_id,
                retained_sensor_ids=retained_sensor_ids,
                alignment_tolerance_ms=alignment_tolerance_ms,
            )
            if candidate is None:
                continue
            if best_resolution is None or _is_better_alignment_subset(
                candidate=candidate,
                reference=best_resolution,
            ):
                best_resolution = candidate
        if best_resolution is not None:
            return best_resolution

    return None


def _find_best_alignable_sensor_subset_greedy(
    campaign_label: str,
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    alignment_tolerance_ms: int | None,
) -> _AlignmentSubsetResolution | None:
    """Greedily prune low-support sensors until a usable subset appears."""

    sorted_series_by_id = {
        sensor_id: _sort_sensor_series_by_timestamp(series)
        for sensor_id, series in sensor_series_by_id.items()
    }
    resolved_tolerance_ms = _infer_alignment_tolerance_ms(
        sorted_series_by_id=sorted_series_by_id,
        alignment_tolerance_ms=alignment_tolerance_ms,
    )
    current_sensor_ids = tuple(sorted(sorted_series_by_id))

    while len(current_sensor_ids) > 2:
        best_resolution: _AlignmentSubsetResolution | None = None

        # First prefer the smallest one-sensor pruning that already succeeds.
        for sensor_id in current_sensor_ids:
            retained_sensor_ids = tuple(
                candidate_sensor_id
                for candidate_sensor_id in current_sensor_ids
                if candidate_sensor_id != sensor_id
            )
            candidate = _resolve_alignment_subset_candidate(
                campaign_label=campaign_label,
                sensor_series_by_id=sorted_series_by_id,
                retained_sensor_ids=retained_sensor_ids,
                alignment_tolerance_ms=resolved_tolerance_ms,
            )
            if candidate is None:
                continue
            if best_resolution is None or _is_better_alignment_subset(
                candidate=candidate,
                reference=best_resolution,
            ):
                best_resolution = candidate

        if best_resolution is not None:
            return best_resolution

        # When every one-sensor pruning still fails, remove the sensor with
        # the weakest pairwise timestamp support and try again on the smaller
        # roster.
        sensor_id_to_prune = _select_low_support_sensor_id(
            current_sensor_ids=current_sensor_ids,
            sorted_series_by_id=sorted_series_by_id,
            alignment_tolerance_ms=resolved_tolerance_ms,
        )
        current_sensor_ids = tuple(
            sensor_id
            for sensor_id in current_sensor_ids
            if sensor_id != sensor_id_to_prune
        )

    return _resolve_alignment_subset_candidate(
        campaign_label=campaign_label,
        sensor_series_by_id=sorted_series_by_id,
        retained_sensor_ids=current_sensor_ids,
        alignment_tolerance_ms=resolved_tolerance_ms,
    )


def _resolve_alignment_subset_candidate(
    campaign_label: str,
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    retained_sensor_ids: tuple[str, ...],
    alignment_tolerance_ms: int | None,
) -> _AlignmentSubsetResolution | None:
    """Return one successful retained subset or ``None`` when it still fails."""

    dataset, diagnostics = _try_align_campaign_subset(
        campaign_label=campaign_label,
        sensor_series_by_id=sensor_series_by_id,
        retained_sensor_ids=retained_sensor_ids,
        alignment_tolerance_ms=alignment_tolerance_ms,
        raise_on_no_shared_overlap=False,
    )
    if dataset is None or diagnostics is None:
        return None

    pruned_sensor_ids = tuple(
        sensor_id
        for sensor_id in sorted(sensor_series_by_id)
        if sensor_id not in set(retained_sensor_ids)
    )
    return _AlignmentSubsetResolution(
        dataset=dataset,
        diagnostics=diagnostics,
        pruned_sensor_ids=pruned_sensor_ids,
    )


def _try_align_campaign_subset(
    campaign_label: str,
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    retained_sensor_ids: tuple[str, ...],
    alignment_tolerance_ms: int | None,
    *,
    raise_on_no_shared_overlap: bool,
) -> tuple[RbwAcquisitionDataset | None, CampaignAlignmentDiagnostics | None]:
    """Try to align one retained sensor subset without masking other failures."""

    retained_sensor_series_by_id = {
        sensor_id: sensor_series_by_id[sensor_id] for sensor_id in retained_sensor_ids
    }
    try:
        return align_campaign_sensor_series(
            campaign_label=campaign_label,
            sensor_series_by_id=retained_sensor_series_by_id,
            alignment_tolerance_ms=alignment_tolerance_ms,
        )
    except ValueError as error:
        if not _is_no_shared_alignment_error(error):
            raise
        if raise_on_no_shared_overlap:
            raise
    return None, None


def _is_no_shared_alignment_error(error: ValueError) -> bool:
    """Return whether the error means no shared aligned record groups exist."""

    return str(error).startswith(_NO_SHARED_ALIGNMENT_ERROR_PREFIX)


def _is_better_alignment_subset(
    candidate: _AlignmentSubsetResolution,
    reference: _AlignmentSubsetResolution,
) -> bool:
    """Return whether ``candidate`` is preferable to ``reference``."""

    candidate_score = _alignment_subset_score(candidate)
    reference_score = _alignment_subset_score(reference)
    if candidate_score != reference_score:
        return candidate_score > reference_score
    return candidate.dataset.sensor_ids < reference.dataset.sensor_ids


def _alignment_subset_score(
    resolution: _AlignmentSubsetResolution,
) -> tuple[int, int, float, float]:
    """Build a deterministic score for comparing successful retained subsets."""

    aligned_record_count = resolution.dataset.n_records
    retained_fraction = aligned_record_count / np.maximum(
        resolution.diagnostics.source_record_count.astype(np.float64),
        1.0,
    )
    return (
        len(resolution.dataset.sensor_ids),
        aligned_record_count,
        float(np.mean(retained_fraction)),
        float(np.min(retained_fraction)),
    )


def _select_low_support_sensor_id(
    current_sensor_ids: tuple[str, ...],
    sorted_series_by_id: Mapping[str, SensorMeasurementSeries],
    alignment_tolerance_ms: int,
) -> str:
    """Return the sensor that contributes the least timestamp-overlap support."""

    lowest_support_signature: tuple[int, int, str] | None = None
    lowest_support_sensor_id: str | None = None
    for sensor_id in current_sensor_ids:
        pairwise_support_count = 0
        for other_sensor_id in current_sensor_ids:
            if other_sensor_id == sensor_id:
                continue
            pairwise_support_count += _count_pairwise_timestamp_matches(
                first_timestamps_ms=sorted_series_by_id[sensor_id].timestamps_ms,
                second_timestamps_ms=sorted_series_by_id[other_sensor_id].timestamps_ms,
                alignment_tolerance_ms=alignment_tolerance_ms,
            )
        support_signature = (
            pairwise_support_count,
            sorted_series_by_id[sensor_id].n_records,
            sensor_id,
        )
        if (
            lowest_support_signature is None
            or support_signature < lowest_support_signature
        ):
            lowest_support_signature = support_signature
            lowest_support_sensor_id = sensor_id

    if lowest_support_sensor_id is None:
        raise AssertionError("At least one sensor id must be available for pruning")
    return lowest_support_sensor_id


def _count_pairwise_timestamp_matches(
    first_timestamps_ms: IndexArray,
    second_timestamps_ms: IndexArray,
    alignment_tolerance_ms: int,
) -> int:
    """Count monotone pairwise timestamp matches within the alignment tolerance."""

    first_index = 0
    second_index = 0
    n_matches = 0

    while (
        first_index < first_timestamps_ms.size
        and second_index < second_timestamps_ms.size
    ):
        first_timestamp_ms = int(first_timestamps_ms[first_index])
        second_timestamp_ms = int(second_timestamps_ms[second_index])
        timestamp_delta_ms = second_timestamp_ms - first_timestamp_ms

        if abs(timestamp_delta_ms) <= alignment_tolerance_ms:
            n_matches += 1
            first_index += 1
            second_index += 1
            continue

        if second_timestamp_ms < first_timestamp_ms - alignment_tolerance_ms:
            second_index += 1
        else:
            first_index += 1

    return n_matches


def build_dataset_summary_rows(
    datasets: Mapping[str, RbwAcquisitionDataset],  # Parsed RBW datasets
) -> list[dict[str, float | int | str]]:  # Table-friendly per-RBW summary rows
    """Build a compact per-RBW integrity summary.

    The exploratory ``DatasetFM-v0`` notebook spent substantial effort checking
    whether records were aligned well enough to justify cross-sensor
    comparisons. This helper preserves that intent in a deterministic way by
    reporting shared shape information and timestamp spread statistics.
    """

    rows: list[dict[str, float | int | str]] = []
    for rbw_label in sorted(datasets):
        dataset = datasets[rbw_label]
        timestamp_spread_s = (
            np.max(dataset.timestamps_ms, axis=0)
            - np.min(dataset.timestamps_ms, axis=0)
        ) / 1_000.0
        frequency_step_hz = float(np.mean(np.diff(dataset.frequency_hz)))
        rows.append(
            {
                "rbw": rbw_label,
                "n_sensors": dataset.n_sensors,
                "n_records": dataset.n_records,
                "n_frequencies": dataset.n_frequencies,
                "frequency_step_hz": frequency_step_hz,
                "frequency_span_mhz": float(
                    (dataset.frequency_hz[-1] - dataset.frequency_hz[0]) / 1.0e6
                ),
                "mean_record_time_spread_s": float(np.mean(timestamp_spread_s)),
                "max_record_time_spread_s": float(np.max(timestamp_spread_s)),
                "sensor_ids": ", ".join(dataset.sensor_ids),
            }
        )
    return rows


def build_sensor_integrity_rows(
    dataset: RbwAcquisitionDataset,  # One RBW subset to summarize sensor by sensor
) -> list[dict[str, float | int | str]]:  # Table-friendly per-sensor summary rows
    """Build sensor-level descriptive statistics for one RBW subset.

    The returned rows expose the most useful checks from the exploratory
    notebooks without mutating or cleaning the raw data in place:

    - timestamp coverage per sensor,
    - total acquisition span [s], and
    - flattened PSD range and dispersion [dB].
    """

    observations_db = np.asarray(dataset.observations_db, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []

    for sensor_index, sensor_id in enumerate(dataset.sensor_ids):
        flattened_power_db = observations_db[sensor_index].reshape(-1)
        timestamps_ms = dataset.timestamps_ms[sensor_index]
        rows.append(
            {
                "sensor_id": sensor_id,
                "records": int(dataset.n_records),
                "timestamp_start_ms": int(timestamps_ms[0]),
                "timestamp_end_ms": int(timestamps_ms[-1]),
                "acquisition_span_s": float(
                    (timestamps_ms[-1] - timestamps_ms[0]) / 1_000.0
                ),
                "mean_psd_db": float(np.mean(flattened_power_db)),
                "std_psd_db": float(np.std(flattened_power_db)),
                "min_psd_db": float(np.min(flattened_power_db)),
                "max_psd_db": float(np.max(flattened_power_db)),
            }
        )
    return rows


def summarize_psd_distribution(
    dataset: RbwAcquisitionDataset,  # One RBW subset with aligned PSD tensors
    histogram_bins: int = 300,  # Number of bins used for the density estimates
    value_range_db: tuple[float, float] | None = None,  # Optional histogram window [dB]
) -> SensorDistributionDiagnostics:  # Dataset-wide PSD distribution diagnostics
    """Summarize PSD distributions across every record of one RBW subset.

    This reproduces the dataset-level histogram analysis from
    ``DatasetFM-v0``/``DatasetFM-v1`` as a complementary diagnostic to the
    per-record ranking from ``DatasetFM-v3``. The goal is different:

    - the ranking core checks whether sensors agree on record-by-record PSD
      shape after recentring, while
    - this function checks whether their *overall* PSD distributions have a
      similar shape across the whole campaign.

    Parameters
    ----------
    dataset:
        Parsed RBW subset with shape ``(n_sensors, n_records, n_frequencies)``.
    histogram_bins:
        Number of bins used for the density estimates. Must be at least one.
    value_range_db:
        Optional ``(low_db, high_db)`` range that clips histogram accumulation
        to a shared dB window. When omitted, the range is inferred from the
        observed data so no values are clipped.
    """

    if histogram_bins < 1:
        raise ValueError("histogram_bins must be at least 1")

    observations_db = np.asarray(dataset.observations_db, dtype=np.float64)
    if observations_db.ndim != 3:
        raise ValueError(
            "dataset.observations_db must have shape "
            "(n_sensors, n_records, n_frequencies)"
        )
    if observations_db.shape[0] < 2:
        raise ValueError(
            "At least two sensors are required for distribution diagnostics"
        )
    if not np.all(np.isfinite(observations_db)):
        raise ValueError("dataset.observations_db must contain only finite values")

    flattened_power_db = observations_db.reshape(observations_db.shape[0], -1)
    if value_range_db is None:
        lower_db = float(np.min(flattened_power_db))
        upper_db = float(np.max(flattened_power_db))
    else:
        lower_db, upper_db = map(float, value_range_db)
        if (
            not np.isfinite(lower_db)
            or not np.isfinite(upper_db)
            or lower_db >= upper_db
        ):
            raise ValueError(
                "value_range_db must satisfy low < high with finite bounds"
            )

    bin_edges_db = np.linspace(lower_db, upper_db, histogram_bins + 1, dtype=np.float64)
    bin_centers_db = 0.5 * (bin_edges_db[:-1] + bin_edges_db[1:])
    bin_width_db = float(bin_edges_db[1] - bin_edges_db[0])

    per_sensor_density = np.empty(
        (dataset.n_sensors, histogram_bins),
        dtype=np.float64,
    )
    global_counts = np.zeros(histogram_bins, dtype=np.int64)
    mean_db = np.empty(dataset.n_sensors, dtype=np.float64)
    std_db = np.empty(dataset.n_sensors, dtype=np.float64)
    min_db = np.empty(dataset.n_sensors, dtype=np.float64)
    max_db = np.empty(dataset.n_sensors, dtype=np.float64)
    value_count = np.empty(dataset.n_sensors, dtype=np.int64)
    clipped_count = 0

    # Reuse the exact same bin edges for every sensor so correlation compares
    # histogram shape rather than arbitrary bin-placement differences.
    for sensor_index in range(dataset.n_sensors):
        values_db = flattened_power_db[sensor_index]
        counts, _ = np.histogram(values_db, bins=bin_edges_db)
        count_sum = int(np.sum(counts))
        if count_sum == 0:
            raise ValueError(
                "The requested histogram range excludes every PSD value; "
                "choose a wider value_range_db"
            )

        clipped_count += int(np.sum((values_db < lower_db) | (values_db > upper_db)))
        global_counts += counts
        per_sensor_density[sensor_index] = counts / float(count_sum * bin_width_db)
        mean_db[sensor_index] = float(np.mean(values_db))
        std_db[sensor_index] = float(np.std(values_db))
        min_db[sensor_index] = float(np.min(values_db))
        max_db[sensor_index] = float(np.max(values_db))
        value_count[sensor_index] = int(values_db.size)

    global_density = global_counts / float(np.sum(global_counts) * bin_width_db)

    # Small campaigns can produce nearly constant density curves. Standardizing
    # with an explicit epsilon avoids NaN correlations from zero-variance rows.
    centered_density = per_sensor_density - np.mean(
        per_sensor_density,
        axis=1,
        keepdims=True,
    )
    density_scale = np.std(centered_density, axis=1, keepdims=True)
    standardized_density = centered_density / np.clip(
        density_scale,
        _STD_EPSILON,
        None,
    )
    correlation_matrix = (
        standardized_density @ standardized_density.T / float(histogram_bins)
    )
    correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)
    np.fill_diagonal(correlation_matrix, 1.0)

    cross_similarity_score = np.sum(correlation_matrix, axis=1) - 1.0
    normalized_similarity_score = cross_similarity_score / float(dataset.n_sensors - 1)
    outlier_threshold = float(
        np.mean(cross_similarity_score) - np.std(cross_similarity_score)
    )
    ranking_order = np.argsort(cross_similarity_score)[::-1]
    ranking_sensor_ids = tuple(dataset.sensor_ids[index] for index in ranking_order)

    return SensorDistributionDiagnostics(
        rbw_label=dataset.rbw_label,
        sensor_ids=dataset.sensor_ids,
        bin_edges_db=bin_edges_db,
        bin_centers_db=bin_centers_db,
        global_density=global_density,
        per_sensor_density=per_sensor_density,
        correlation_matrix=correlation_matrix,
        mean_db=mean_db,
        std_db=std_db,
        min_db=min_db,
        max_db=max_db,
        value_count=value_count,
        cross_similarity_score=cross_similarity_score,
        normalized_similarity_score=normalized_similarity_score,
        outlier_threshold=outlier_threshold,
        clipped_fraction=float(clipped_count / float(np.sum(value_count))),
        ranking_sensor_ids=ranking_sensor_ids,
    )


def estimate_histogram_noise_floor_db(
    power_db: FloatArray,  # One PSD vector in dB
    histogram_bins: int = 50,  # Number of histogram bins used for the mode heuristic
) -> float:  # Histogram-mode noise floor [dB]
    """Estimate a PSD noise floor from the dominant histogram bin.

    The returned value intentionally mirrors the exploratory notebook logic:
    it uses the left edge of the most populated histogram bin rather than a
    fitted statistical model so the ranking stays comparable to the original
    analysis.
    """

    power_db = np.asarray(power_db, dtype=np.float64)
    if power_db.ndim != 1:
        raise ValueError("power_db must be a one-dimensional PSD vector")
    if power_db.size == 0:
        raise ValueError("power_db must contain at least one sample")
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be at least 1")
    if not np.all(np.isfinite(power_db)):
        raise ValueError("power_db must contain only finite numeric values")

    counts, bin_edges = np.histogram(power_db, bins=int(histogram_bins))
    return float(bin_edges[int(np.argmax(counts))])


def rank_sensors_by_cumulative_correlation(
    dataset: RbwAcquisitionDataset,  # Parsed RBW subset with row-aligned PSD vectors
    histogram_bins: int = 50,  # Histogram bins for the noise-floor estimate
    epsilon: float = _STD_EPSILON,  # Stabilizer for zero-variance PSD vectors
) -> SensorRankingResult:  # Ranking diagnostics for the RBW subset
    """Rank sensors by network agreement on one RBW subset.

    The method follows the exploratory ``check_out/DatasetFM-v3.ipynb`` logic:

    1. Estimate a per-sensor noise floor on each record from the PSD histogram.
    2. Shift every PSD so all sensors share the same record-wise mean noise floor.
    3. Standardize each PSD shape to zero mean and unit variance.
    4. Compute the pairwise Pearson correlation matrix.
    5. Use the row sum minus one as the cumulative agreement score.

    No I/O occurs inside the ranking core. The function operates only on the
    already-parsed numeric dataset, which keeps the behavior deterministic and
    easy to test.
    """

    if histogram_bins < 1:
        raise ValueError("histogram_bins must be at least 1")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")

    observations_db = np.asarray(dataset.observations_db, dtype=np.float64)
    if observations_db.ndim != 3:
        raise ValueError(
            "dataset.observations_db must have shape "
            "(n_sensors, n_records, n_frequencies)"
        )
    if observations_db.shape[0] < 2:
        raise ValueError("At least two sensors are required for correlation ranking")
    if observations_db.shape[1] < 1:
        raise ValueError("At least one record is required for correlation ranking")
    if observations_db.shape[2] < 2:
        raise ValueError(
            "At least two frequency bins are required for correlation ranking"
        )
    if not np.all(np.isfinite(observations_db)):
        raise ValueError("dataset.observations_db must contain only finite values")

    n_sensors, n_records, n_frequencies = observations_db.shape
    per_record_score = np.empty((n_sensors, n_records), dtype=np.float64)
    noise_floor_db = np.empty((n_sensors, n_records), dtype=np.float64)
    global_noise_floor_db = np.empty(n_records, dtype=np.float64)
    per_record_correlation = np.empty(
        (n_records, n_sensors, n_sensors),
        dtype=np.float64,
    )

    for record_index in range(n_records):
        record_panel = observations_db[:, record_index, :]
        for sensor_index in range(n_sensors):
            noise_floor_db[sensor_index, record_index] = (
                estimate_histogram_noise_floor_db(
                    record_panel[sensor_index],
                    histogram_bins=histogram_bins,
                )
            )

        # Recenter every PSD to the same average floor before comparing shape.
        global_noise_floor_db[record_index] = float(
            np.mean(noise_floor_db[:, record_index])
        )
        recentered_panel = (
            record_panel
            + (global_noise_floor_db[record_index] - noise_floor_db[:, record_index])[
                :, np.newaxis
            ]
        )

        # Correlation should reflect PSD shape, not absolute offset or scale.
        centered_panel = recentered_panel - np.mean(
            recentered_panel,
            axis=1,
            keepdims=True,
        )
        scale = np.std(centered_panel, axis=1, keepdims=True)
        standardized_panel = centered_panel / np.clip(scale, epsilon, None)

        correlation_matrix = (
            standardized_panel @ standardized_panel.T / float(n_frequencies)
        )
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)
        np.fill_diagonal(correlation_matrix, 1.0)
        per_record_correlation[record_index] = correlation_matrix
        per_record_score[:, record_index] = np.sum(correlation_matrix, axis=1) - 1.0

    average_score = np.mean(per_record_score, axis=1)
    average_correlation = average_score / float(n_sensors - 1)
    ranking_order = np.argsort(average_score)[::-1]
    ranking_sensor_ids = tuple(dataset.sensor_ids[index] for index in ranking_order)

    return SensorRankingResult(
        rbw_label=dataset.rbw_label,
        sensor_ids=dataset.sensor_ids,
        per_record_score=per_record_score,
        average_score=average_score,
        average_correlation=average_correlation,
        noise_floor_db=noise_floor_db,
        global_noise_floor_db=global_noise_floor_db,
        per_record_correlation=per_record_correlation,
        ranking_sensor_ids=ranking_sensor_ids,
    )


def build_sensor_ranking_rows(
    result: SensorRankingResult,  # Ranking result for one RBW subset
) -> list[dict[str, float | int | str]]:  # Notebook-friendly ranking rows
    """Build a table-friendly ranking summary sorted from best to worst sensor."""

    order = np.argsort(result.average_score)[::-1]
    rows: list[dict[str, float | int | str]] = []
    for rank_index, sensor_index in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank_index,
                "sensor_id": result.sensor_ids[sensor_index],
                "mean_score": float(result.average_score[sensor_index]),
                "mean_correlation": float(result.average_correlation[sensor_index]),
                "score_std": float(np.std(result.per_record_score[sensor_index])),
                "mean_noise_floor_db": float(
                    np.mean(result.noise_floor_db[sensor_index])
                ),
                "records": int(result.per_record_score.shape[1]),
            }
        )
    return rows


def build_distribution_summary_rows(
    diagnostics: SensorDistributionDiagnostics,  # Dataset-wide histogram diagnostics
) -> list[dict[str, bool | float | int | str]]:  # Table-friendly distribution rows
    """Build a ranking table for the PSD-distribution diagnostics.

    The output mirrors the histogram-shape ranking from the exploratory
    notebooks while surfacing a few additional descriptive statistics so the
    user can tell *why* a sensor stands out.
    """

    order = np.argsort(diagnostics.cross_similarity_score)[::-1]
    rows: list[dict[str, bool | float | int | str]] = []
    for rank_index, sensor_index in enumerate(order, start=1):
        score = float(diagnostics.cross_similarity_score[sensor_index])
        rows.append(
            {
                "rank": rank_index,
                "sensor_id": diagnostics.sensor_ids[sensor_index],
                "distribution_similarity": score,
                "normalized_similarity": float(
                    diagnostics.normalized_similarity_score[sensor_index]
                ),
                "mean_psd_db": float(diagnostics.mean_db[sensor_index]),
                "std_psd_db": float(diagnostics.std_db[sensor_index]),
                "min_psd_db": float(diagnostics.min_db[sensor_index]),
                "max_psd_db": float(diagnostics.max_db[sensor_index]),
                "value_count": int(diagnostics.value_count[sensor_index]),
                "is_low_similarity_outlier": bool(
                    score < diagnostics.outlier_threshold
                ),
            }
        )
    return rows


def build_score_stability_rows(
    result: SensorRankingResult,  # Ranking result for one RBW subset
) -> list[dict[str, float | int | str]]:  # Table-friendly record-wise stability rows
    """Build a record-wise ranking stability summary.

    Mean scores alone can hide whether the winner is dominant on nearly every
    record or only marginally ahead after averaging. This helper quantifies the
    stability of the ranking with per-record rank positions and top-1 counts.
    """

    n_sensors, n_records = result.per_record_score.shape
    order = np.argsort(result.average_score)[::-1]
    per_record_order = np.argsort(result.per_record_score, axis=0)[::-1]
    per_record_rank = np.empty((n_sensors, n_records), dtype=np.int64)

    # Rank is assigned independently on each record so we can summarize how
    # often a sensor really wins rather than only having a slightly larger mean.
    for record_index in range(n_records):
        per_record_rank[per_record_order[:, record_index], record_index] = np.arange(
            1,
            n_sensors + 1,
            dtype=np.int64,
        )

    rows: list[dict[str, float | int | str]] = []
    for rank_index, sensor_index in enumerate(order, start=1):
        score_series = result.per_record_score[sensor_index]
        rank_series = per_record_rank[sensor_index]
        rows.append(
            {
                "rank": rank_index,
                "sensor_id": result.sensor_ids[sensor_index],
                "mean_score": float(result.average_score[sensor_index]),
                "mean_correlation": float(result.average_correlation[sensor_index]),
                "score_std": float(np.std(score_series)),
                "score_min": float(np.min(score_series)),
                "score_max": float(np.max(score_series)),
                "mean_rank": float(np.mean(rank_series)),
                "best_rank": int(np.min(rank_series)),
                "worst_rank": int(np.max(rank_series)),
                "top_1_count": int(np.sum(rank_series == 1)),
                "top_1_fraction": float(np.mean(rank_series == 1)),
            }
        )
    return rows


def build_rbw_overview_rows(
    results_by_rbw: Mapping[str, SensorRankingResult],  # Per-RBW ranking results
) -> list[dict[str, float | int | str]]:  # Overview rows sorted by RBW label
    """Build a compact per-RBW overview table.

    The overview highlights the best-ranked sensor on each RBW subset together
    with the corresponding mean score and normalized correlation.
    """

    rows: list[dict[str, float | int | str]] = []
    for rbw_label in sorted(results_by_rbw):
        result = results_by_rbw[rbw_label]
        best_sensor_index = int(np.argmax(result.average_score))
        rows.append(
            {
                "rbw": rbw_label,
                "best_sensor_id": result.sensor_ids[best_sensor_index],
                "best_mean_score": float(result.average_score[best_sensor_index]),
                "best_mean_correlation": float(
                    result.average_correlation[best_sensor_index]
                ),
                "n_sensors": int(len(result.sensor_ids)),
                "n_records": int(result.per_record_score.shape[1]),
            }
        )
    return rows


def _load_sensor_series_directory(
    dataset_dir: Path,  # Directory with one CSV file per sensor
    sensor_file_pattern: str,  # Glob used to discover sensor files
    sort_by_timestamp: bool,
) -> dict[str, SensorMeasurementSeries]:
    """Load every sensor CSV from one directory into parsed series objects."""

    sensor_files = sorted(
        path
        for path in Path(dataset_dir).glob(sensor_file_pattern)
        if path.is_file() and path.name.lower() != _METADATA_FILENAME
    )
    if not sensor_files:
        raise ValueError(
            f"No sensor CSV files matching {sensor_file_pattern!r} were found in "
            f"{dataset_dir}"
        )

    sensor_series_by_id: dict[str, SensorMeasurementSeries] = {}
    for path in sensor_files:
        sensor_series = _load_sensor_csv(path, sort_by_timestamp=sort_by_timestamp)
        if sensor_series.sensor_id in sensor_series_by_id:
            raise ValueError(
                f"Duplicate sensor label {sensor_series.sensor_id!r} in {dataset_dir}"
            )
        sensor_series_by_id[sensor_series.sensor_id] = sensor_series
    return sensor_series_by_id


def _stack_aligned_sensor_series(
    dataset_label: str,  # Output label used in the aligned dataset
    sensor_series_by_id: Mapping[
        str, SensorMeasurementSeries
    ],  # Parsed per-sensor rows
) -> RbwAcquisitionDataset:
    """Stack already aligned sensor series into the ranking dataset tensor."""

    if len(sensor_series_by_id) < 2:
        raise ValueError("At least two sensors are required to build a dataset")

    sensor_ids = tuple(sorted(sensor_series_by_id))
    reference_frequency_hz = _validate_shared_frequency_grid(
        sensor_series_by_id,
        context_label=str(dataset_label),
    )
    reference_record_count = sensor_series_by_id[sensor_ids[0]].n_records

    # The RBW workflow assumes row ``k`` refers to the same capture across all
    # sensors, so record counts must agree exactly at this boundary.
    for sensor_id in sensor_ids[1:]:
        sensor_series = sensor_series_by_id[sensor_id]
        if sensor_series.n_records != reference_record_count:
            raise ValueError(
                "All sensors inside one aligned dataset must have the same "
                f"number of records; {sensor_ids[0]} has {reference_record_count} "
                f"but {sensor_id} has {sensor_series.n_records}"
            )

    return RbwAcquisitionDataset(
        rbw_label=str(dataset_label),
        sensor_ids=sensor_ids,
        frequency_hz=np.asarray(reference_frequency_hz, dtype=np.float64),
        observations_db=np.stack(
            [
                sensor_series_by_id[sensor_id].observations_db
                for sensor_id in sensor_ids
            ],
            axis=0,
        ),
        timestamps_ms=np.stack(
            [sensor_series_by_id[sensor_id].timestamps_ms for sensor_id in sensor_ids],
            axis=0,
        ),
    )


def _validate_shared_frequency_grid(
    sensor_series_by_id: Mapping[str, SensorMeasurementSeries],
    context_label: str,
) -> FloatArray:
    """Validate that every sensor shares the same PSD frequency grid."""

    sensor_ids = tuple(sorted(sensor_series_by_id))
    reference_frequency_hz = sensor_series_by_id[sensor_ids[0]].frequency_hz
    for sensor_id in sensor_ids[1:]:
        sensor_frequency_hz = sensor_series_by_id[sensor_id].frequency_hz
        if not np.allclose(
            sensor_frequency_hz,
            reference_frequency_hz,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "All sensors inside one dataset must share the same frequency "
                f"grid; {sensor_id} differs in {context_label}"
            )
    return np.asarray(reference_frequency_hz, dtype=np.float64)


def _sort_sensor_series_by_timestamp(
    sensor_series: SensorMeasurementSeries,
) -> SensorMeasurementSeries:
    """Return a copy of the sensor series sorted by timestamp."""

    if sensor_series.n_records <= 1:
        return sensor_series

    sorted_order = np.argsort(sensor_series.timestamps_ms, kind="stable")
    return SensorMeasurementSeries(
        sensor_id=sensor_series.sensor_id,
        frequency_hz=np.asarray(sensor_series.frequency_hz, dtype=np.float64),
        observations_db=np.asarray(
            sensor_series.observations_db[sorted_order],
            dtype=np.float64,
        ),
        timestamps_ms=np.asarray(
            sensor_series.timestamps_ms[sorted_order], dtype=np.int64
        ),
        source_row_indices=np.asarray(
            sensor_series.source_row_indices[sorted_order],
            dtype=np.int64,
        ),
    )


def _infer_alignment_tolerance_ms(
    sorted_series_by_id: Mapping[str, SensorMeasurementSeries],  # Raw or sorted series
    alignment_tolerance_ms: int | None,
) -> int:
    """Infer a conservative timestamp-alignment tolerance for one campaign."""

    if alignment_tolerance_ms is not None:
        return int(alignment_tolerance_ms)

    positive_deltas_ms: list[float] = []
    for sensor_series in sorted_series_by_id.values():
        if sensor_series.n_records < 2:
            continue
        deltas_ms = np.diff(np.sort(sensor_series.timestamps_ms.astype(np.float64)))
        positive_deltas_ms.extend(float(delta) for delta in deltas_ms if delta > 0.0)

    if not positive_deltas_ms:
        return _MAX_ALIGNMENT_TOLERANCE_MS

    median_period_ms = float(
        np.median(np.asarray(positive_deltas_ms, dtype=np.float64))
    )
    inferred_tolerance_ms = median_period_ms * _ALIGNMENT_TOLERANCE_PERIOD_FRACTION
    clipped_tolerance_ms = min(
        max(inferred_tolerance_ms, float(_MIN_ALIGNMENT_TOLERANCE_MS)),
        float(_MAX_ALIGNMENT_TOLERANCE_MS),
    )
    return int(round(clipped_tolerance_ms))


def _find_nearest_timestamp_index(
    sorted_timestamps_ms: IndexArray,  # Timestamp-sorted series for one sensor
    target_timestamp_ms: int,  # Anchor timestamp used for matching
    start_index: int,  # First index still available for monotone matching
    alignment_tolerance_ms: int,  # Maximum allowed absolute mismatch [ms]
) -> int | None:
    """Return the nearest admissible timestamp index or ``None`` if none fits."""

    if start_index >= int(sorted_timestamps_ms.size):
        return None

    candidate_slice = sorted_timestamps_ms[start_index:]
    insertion_index = int(
        np.searchsorted(candidate_slice, target_timestamp_ms, side="left")
    )
    candidate_indices: list[int] = []
    for relative_index in (insertion_index - 1, insertion_index):
        if 0 <= relative_index < int(candidate_slice.size):
            candidate_indices.append(start_index + relative_index)
    if not candidate_indices:
        return None

    best_index = min(
        candidate_indices,
        key=lambda candidate_index: (
            abs(int(sorted_timestamps_ms[candidate_index]) - int(target_timestamp_ms)),
            candidate_index,
        ),
    )
    if (
        abs(int(sorted_timestamps_ms[best_index]) - int(target_timestamp_ms))
        > alignment_tolerance_ms
    ):
        return None
    return best_index


def _load_sensor_csv(
    path: Path,  # Acquisition CSV path
    sort_by_timestamp: bool = False,
) -> SensorMeasurementSeries:  # Parsed observations for one sensor file
    """Load one acquisition CSV into a numeric sensor series.

    Every row must provide a JSON-encoded ``pxx`` vector, ``start_freq_hz``,
    ``end_freq_hz``, and ``timestamp``. All rows inside the file are required to
    share the same frequency grid because the ranking compares records directly.
    """

    csv.field_size_limit(sys.maxsize)
    observations_db: list[FloatArray] = []
    timestamps_ms: list[int] = []
    reference_frequency_hz: FloatArray | None = None

    with path.open(newline="", encoding="utf-8") as csv_file:
        for row_index, row in enumerate(csv.DictReader(csv_file)):
            try:
                power_db = np.asarray(json.loads(row["pxx"]), dtype=np.float64)
                start_freq_hz = float(row["start_freq_hz"])
                end_freq_hz = float(row["end_freq_hz"])
                timestamp_ms = int(row["timestamp"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid RBW acquisition row in {path}: {row}"
                ) from error

            if power_db.ndim != 1 or power_db.size < 2:
                raise ValueError(
                    f"{path} row {row_index} must contain a one-dimensional PSD vector"
                )
            if not np.all(np.isfinite(power_db)):
                raise ValueError(
                    f"{path} row {row_index} contains non-finite PSD values"
                )

            frequency_hz = np.linspace(
                start_freq_hz,
                end_freq_hz,
                power_db.size,
                dtype=np.float64,
            )
            if reference_frequency_hz is None:
                reference_frequency_hz = frequency_hz
            elif not np.allclose(
                frequency_hz,
                reference_frequency_hz,
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    f"{path} row {row_index} does not match the file frequency grid"
                )

            observations_db.append(power_db)
            timestamps_ms.append(timestamp_ms)

    if reference_frequency_hz is None:
        raise ValueError(f"Acquisition CSV does not contain any rows: {path}")

    sensor_series = SensorMeasurementSeries(
        sensor_id=path.stem,
        frequency_hz=np.asarray(reference_frequency_hz, dtype=np.float64),
        observations_db=np.stack(observations_db, axis=0),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
        source_row_indices=np.arange(len(observations_db), dtype=np.int64),
    )
    if sort_by_timestamp:
        return _sort_sensor_series_by_timestamp(sensor_series)
    return sensor_series


__all__ = [
    "AlignmentPrunedCampaignDataset",
    "CampaignAlignmentDiagnostics",
    "CampaignSensorDataRepository",
    "CampaignSensorRankingAnalysis",
    "DEFAULT_CAMPAIGNS_DATA_DIR",
    "FileSystemCampaignSensorDataRepository",
    "RbwAcquisitionDataset",
    "SensorMeasurementSeries",
    "SensorRankingAnalysisConfig",
    "SensorRankingAnalyzer",
    "align_campaign_sensor_series",
    "align_campaign_sensor_series_with_pruning",
    "analyze_all_campaign_sensor_rankings",
    "analyze_campaign_sensor_ranking",
    "build_campaign_alignment_rows",
    "SensorDistributionDiagnostics",
    "SensorRankingResult",
    "build_dataset_summary_rows",
    "build_distribution_summary_rows",
    "build_score_stability_rows",
    "build_sensor_integrity_rows",
    "build_rbw_overview_rows",
    "build_sensor_ranking_rows",
    "estimate_histogram_noise_floor_db",
    "load_rbw_acquisition_datasets",
    "rank_sensors_by_cumulative_correlation",
    "summarize_psd_distribution",
]
