"""Correlation-based sensor ranking helpers for RBW acquisition notebooks.

The exploratory notebooks under ``check_out/`` rank sensors by how consistently
their PSD shapes agree with the rest of the network after a simple noise-floor
recentering step. This module extracts that workflow into pure, testable
functions so the notebook can stay focused on orchestration and visualization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import csv
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]

_STD_EPSILON = 1.0e-8


@dataclass(frozen=True)
class RbwAcquisitionDataset:
    """Row-aligned acquisition subset for one RBW campaign.

    Parameters
    ----------
    rbw_label:
        Directory name that identifies the RBW subset, for example ``"10K"``.
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


@dataclass(frozen=True)
class SensorRankingResult:
    """Correlation-based ranking output for one RBW subset.

    Parameters
    ----------
    rbw_label:
        Directory name that identifies the processed RBW subset.
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


@dataclass(frozen=True)
class SensorDistributionDiagnostics:
    """Dataset-wide PSD distribution diagnostics for one RBW subset.

    Parameters
    ----------
    rbw_label:
        Directory name that identifies the processed RBW subset.
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


@dataclass(frozen=True)
class _LoadedSensorCsv:
    """Internal parsed representation of one acquisition CSV."""

    sensor_id: str
    frequency_hz: FloatArray
    observations_db: FloatArray
    timestamps_ms: IndexArray


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
        sensor_files = sorted(rbw_dir.glob("Node*.csv"))
        if not sensor_files:
            continue

        loaded_sensors = [_load_sensor_csv(path) for path in sensor_files]
        reference_frequency_hz = loaded_sensors[0].frequency_hz
        reference_record_count = loaded_sensors[0].observations_db.shape[0]

        # The ranking assumes that record index ``k`` refers to the same
        # campaign slice for every sensor in the RBW subset.
        for loaded_sensor in loaded_sensors[1:]:
            if loaded_sensor.observations_db.shape[0] != reference_record_count:
                raise ValueError(
                    "All sensors inside one RBW subset must have the same "
                    f"number of records; {loaded_sensors[0].sensor_id} has "
                    f"{reference_record_count} but {loaded_sensor.sensor_id} has "
                    f"{loaded_sensor.observations_db.shape[0]}"
                )
            if not np.allclose(
                loaded_sensor.frequency_hz,
                reference_frequency_hz,
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "All sensors inside one RBW subset must share the same "
                    f"frequency grid; {loaded_sensor.sensor_id} differs in {rbw_dir}"
                )

        datasets[rbw_dir.name] = RbwAcquisitionDataset(
            rbw_label=rbw_dir.name,
            sensor_ids=tuple(sensor.sensor_id for sensor in loaded_sensors),
            frequency_hz=np.asarray(reference_frequency_hz, dtype=np.float64),
            observations_db=np.stack(
                [sensor.observations_db for sensor in loaded_sensors],
                axis=0,
            ),
            timestamps_ms=np.stack(
                [sensor.timestamps_ms for sensor in loaded_sensors],
                axis=0,
            ),
        )

    if not datasets:
        raise ValueError(f"No RBW acquisition CSV files were found under {root_dir}")
    return datasets


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
    correlation_matrix = np.corrcoef(per_sensor_density)
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


def _load_sensor_csv(
    path: Path,  # Acquisition CSV path
) -> _LoadedSensorCsv:  # Parsed observations for one sensor file
    """Load one RBW acquisition CSV into a numeric tensor.

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
        raise ValueError(f"RBW acquisition CSV does not contain any rows: {path}")

    return _LoadedSensorCsv(
        sensor_id=path.stem,
        frequency_hz=np.asarray(reference_frequency_hz, dtype=np.float64),
        observations_db=np.stack(observations_db, axis=0),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
    )


__all__ = [
    "RbwAcquisitionDataset",
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
