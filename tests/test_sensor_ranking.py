"""Tests for the RBW acquisition sensor-ranking helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from measurement_calibration.sensor_ranking import (
    RbwAcquisitionDataset,
    build_dataset_summary_rows,
    build_distribution_summary_rows,
    build_rbw_overview_rows,
    build_score_stability_rows,
    build_sensor_integrity_rows,
    build_sensor_ranking_rows,
    load_rbw_acquisition_datasets,
    rank_sensors_by_cumulative_correlation,
    summarize_psd_distribution,
)


def test_load_rbw_acquisition_datasets_reads_tree(tmp_path: Path) -> None:
    """The loader should parse one RBW directory into a structured dataset."""

    rbw_dir = tmp_path / "10K"
    rbw_dir.mkdir(parents=True)
    _write_rbw_csv(
        rbw_dir / "Node1.csv",
        timestamps_ms=[1_000, 2_000],
        observations_db=[
            [-60.0, -58.0, -55.0, -52.0],
            [-61.0, -57.5, -54.5, -51.5],
        ],
    )
    _write_rbw_csv(
        rbw_dir / "Node3.csv",
        timestamps_ms=[1_100, 2_100],
        observations_db=[
            [-59.5, -57.8, -55.2, -52.2],
            [-60.8, -57.3, -54.8, -51.8],
        ],
    )

    datasets = load_rbw_acquisition_datasets(tmp_path)

    assert tuple(datasets) == ("10K",)
    dataset = datasets["10K"]
    dataset_rows = build_dataset_summary_rows(datasets)
    sensor_rows = build_sensor_integrity_rows(dataset)
    assert dataset.rbw_label == "10K"
    assert dataset.sensor_ids == ("Node1", "Node3")
    assert dataset.observations_db.shape == (2, 2, 4)
    assert dataset.timestamps_ms.shape == (2, 2)
    assert np.allclose(dataset.frequency_hz, np.linspace(88.0e6, 108.0e6, 4))
    assert dataset_rows == [
        {
            "rbw": "10K",
            "n_sensors": 2,
            "n_records": 2,
            "n_frequencies": 4,
            "frequency_step_hz": pytest.approx(20.0e6 / 3.0),
            "frequency_span_mhz": pytest.approx(20.0),
            "mean_record_time_spread_s": pytest.approx(0.1),
            "max_record_time_spread_s": pytest.approx(0.1),
            "sensor_ids": "Node1, Node3",
        }
    ]
    assert sensor_rows == [
        {
            "sensor_id": "Node1",
            "records": 2,
            "timestamp_start_ms": 1_000,
            "timestamp_end_ms": 2_000,
            "acquisition_span_s": pytest.approx(1.0),
            "mean_psd_db": pytest.approx(-56.1875),
            "std_psd_db": pytest.approx(
                np.std(
                    np.asarray(
                        [[-60.0, -58.0, -55.0, -52.0], [-61.0, -57.5, -54.5, -51.5]]
                    )
                )
            ),
            "min_psd_db": pytest.approx(-61.0),
            "max_psd_db": pytest.approx(-51.5),
        },
        {
            "sensor_id": "Node3",
            "records": 2,
            "timestamp_start_ms": 1_100,
            "timestamp_end_ms": 2_100,
            "acquisition_span_s": pytest.approx(1.0),
            "mean_psd_db": pytest.approx(-56.175),
            "std_psd_db": pytest.approx(
                np.std(
                    np.asarray(
                        [[-59.5, -57.8, -55.2, -52.2], [-60.8, -57.3, -54.8, -51.8]]
                    )
                )
            ),
            "min_psd_db": pytest.approx(-60.8),
            "max_psd_db": pytest.approx(-51.8),
        },
    ]


def test_load_rbw_acquisition_datasets_requires_equal_record_counts(
    tmp_path: Path,
) -> None:
    """Sensors inside one RBW subset must expose the same number of rows."""

    rbw_dir = tmp_path / "30K"
    rbw_dir.mkdir(parents=True)
    _write_rbw_csv(
        rbw_dir / "Node1.csv",
        timestamps_ms=[1_000, 2_000],
        observations_db=[
            [-60.0, -58.0, -55.0, -52.0],
            [-61.0, -57.5, -54.5, -51.5],
        ],
    )
    _write_rbw_csv(
        rbw_dir / "Node5.csv",
        timestamps_ms=[1_100],
        observations_db=[[-59.5, -57.8, -55.2, -52.2]],
    )

    with pytest.raises(ValueError, match="same number of records"):
        load_rbw_acquisition_datasets(tmp_path)


def test_rank_sensors_by_cumulative_correlation_orders_sensors() -> None:
    """Higher-consensus sensors should rank ahead of inconsistent ones."""

    base_records = np.asarray(
        [
            [-61.0, -58.0, -54.0, -50.0, -48.5, -50.5, -55.0, -59.5],
            [-62.0, -60.5, -57.0, -53.0, -49.5, -48.0, -50.0, -54.0],
            [-63.0, -61.0, -58.0, -55.0, -52.0, -53.0, -56.0, -60.0],
        ],
        dtype=np.float64,
    )
    observations_db = np.stack(
        [
            base_records,
            base_records
            + np.asarray(
                [
                    [0.1, 0.0, -0.1, 0.2, 0.0, -0.1, 0.0, 0.1],
                    [0.0, -0.1, 0.1, 0.1, -0.2, 0.1, 0.0, 0.0],
                    [0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.1],
                ],
                dtype=np.float64,
            ),
            np.flip(base_records, axis=1),
        ],
        axis=0,
    )
    dataset = RbwAcquisitionDataset(
        rbw_label="10K",
        sensor_ids=("Node1", "Node3", "Node9"),
        frequency_hz=np.linspace(88.0e6, 108.0e6, observations_db.shape[2]),
        observations_db=observations_db,
        timestamps_ms=np.asarray(
            [
                [1_000, 2_000, 3_000],
                [1_000, 2_000, 3_000],
                [1_000, 2_000, 3_000],
            ],
            dtype=np.int64,
        ),
    )

    result = rank_sensors_by_cumulative_correlation(dataset, histogram_bins=8)
    summary_rows = build_sensor_ranking_rows(result)
    stability_rows = build_score_stability_rows(result)
    overview_rows = build_rbw_overview_rows({"10K": result})

    assert set(result.ranking_sensor_ids[:2]) == {"Node1", "Node3"}
    assert result.ranking_sensor_ids[-1] == "Node9"
    assert result.per_record_score.shape == (3, 3)
    assert result.per_record_correlation.shape == (3, 3, 3)
    assert np.allclose(
        np.diagonal(result.per_record_correlation, axis1=1, axis2=2), 1.0
    )
    assert np.all(result.average_correlation <= 1.0)
    assert np.all(result.average_correlation >= -1.0)
    assert summary_rows[0]["sensor_id"] in {"Node1", "Node3"}
    assert summary_rows[-1]["sensor_id"] == "Node9"
    assert stability_rows[0]["sensor_id"] in {"Node1", "Node3"}
    assert stability_rows[-1]["sensor_id"] == "Node9"
    assert sum(int(row["top_1_count"]) for row in stability_rows) == dataset.n_records
    assert overview_rows == [
        {
            "rbw": "10K",
            "best_sensor_id": result.ranking_sensor_ids[0],
            "best_mean_score": pytest.approx(
                float(result.average_score[np.argmax(result.average_score)])
            ),
            "best_mean_correlation": pytest.approx(
                float(result.average_correlation[np.argmax(result.average_score)])
            ),
            "n_sensors": 3,
            "n_records": 3,
        }
    ]


def test_summarize_psd_distribution_highlights_distribution_outlier() -> None:
    """Distribution diagnostics should flag sensors with mismatched PSD histograms."""

    observations_db = np.asarray(
        [
            [
                [-60.0, -59.5, -58.5, -57.5],
                [-60.5, -59.0, -58.0, -57.0],
            ],
            [
                [-60.2, -59.4, -58.4, -57.4],
                [-60.4, -59.1, -58.1, -57.1],
            ],
            [
                [-73.0, -72.8, -72.6, -72.4],
                [-72.9, -72.7, -72.5, -72.3],
            ],
        ],
        dtype=np.float64,
    )
    dataset = RbwAcquisitionDataset(
        rbw_label="30K",
        sensor_ids=("Node1", "Node3", "Node9"),
        frequency_hz=np.linspace(88.0e6, 108.0e6, observations_db.shape[2]),
        observations_db=observations_db,
        timestamps_ms=np.asarray(
            [
                [1_000, 2_000],
                [1_020, 2_020],
                [1_040, 2_040],
            ],
            dtype=np.int64,
        ),
    )

    diagnostics = summarize_psd_distribution(dataset, histogram_bins=10)
    summary_rows = build_distribution_summary_rows(diagnostics)

    assert diagnostics.ranking_sensor_ids[:2] == ("Node3", "Node1")
    assert diagnostics.ranking_sensor_ids[-1] == "Node9"
    assert diagnostics.per_sensor_density.shape == (3, 10)
    assert diagnostics.correlation_matrix.shape == (3, 3)
    assert np.allclose(np.diag(diagnostics.correlation_matrix), 1.0)
    assert summary_rows[-1]["sensor_id"] == "Node9"
    assert summary_rows[-1]["is_low_similarity_outlier"] is True


def _write_rbw_csv(
    path: Path,  # Output CSV path
    timestamps_ms: list[int],  # Row timestamps [ms]
    observations_db: list[list[float]],  # PSD rows in dB
) -> None:
    """Write a small RBW acquisition CSV for testing."""

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("pxx", "start_freq_hz", "end_freq_hz", "timestamp"),
        )
        writer.writeheader()

        # Keep the synthetic files simple: one shared FM band and one PSD row
        # per provided timestamp.
        for timestamp_ms, power_db in zip(timestamps_ms, observations_db, strict=True):
            writer.writerow(
                {
                    "pxx": json.dumps(power_db),
                    "start_freq_hz": 88.0e6,
                    "end_freq_hz": 108.0e6,
                    "timestamp": timestamp_ms,
                }
            )
