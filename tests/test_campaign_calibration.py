"""Tests for campaign metadata parsing and corpus preparation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from measurement_calibration.artifacts import load_two_level_calibration_artifact
from measurement_calibration.campaign_calibration import (
    build_corpus_calibration_output_dir,
    fit_and_save_calibration_corpus_model,
    load_campaign_configuration,
    prepare_calibration_campaign,
    prepare_calibration_corpus,
)
from measurement_calibration.spectral_calibration import (
    FrequencyBasisConfig,
    PersistentModelConfig,
    TwoLevelFitConfig,
)


def test_load_campaign_configuration_parses_placeholder_metadata(
    tmp_path: Path,
) -> None:
    """The metadata parser should normalize placeholder units into SI values."""

    campaign_dir = tmp_path / "campaigns" / "MeasurementCalibration"
    _write_metadata_csv(
        campaign_dir / "metadata.csv",
        {
            "campaign_label": "MeasurementCalibration",
            "start_date": "03/10/2026",
            "stop_date": "03/10/2026",
            "start_time": "00:00:00",
            "stop_time": "06:00:00",
            "acquisition_freq_minutes": "2",
            "central_freq_MHz": "98",
            "span_MHz": "20",
            "lna_gain_dB": "0",
            "vga_gain_dB": "62",
            "rbw_kHz": "10",
            "antenna_amp": "true",
        },
    )

    configuration = load_campaign_configuration(campaign_dir)

    assert configuration.central_frequency_hz == 98.0e6
    assert configuration.span_hz == 20.0e6
    assert configuration.resolution_bandwidth_hz == 10.0e3
    assert configuration.acquisition_interval_s == 120.0
    assert configuration.antenna_amplifier_enabled is True


def test_load_campaign_configuration_parses_api_style_metadata_aliases(
    tmp_path: Path,
) -> None:
    """The parser should also accept the raw field names emitted by the API client."""

    campaign_dir = tmp_path / "campaigns" / "MeasurementCalibration"
    _write_metadata_csv(
        campaign_dir / "metadata.csv",
        {
            "campaign_label": "MeasurementCalibration",
            "name": "MeasurementCalibration",
            "start_date": "03/10/2026",
            "end_date": "03/10/2026",
            "start_time": "00:00:00",
            "end_time": "06:00:00",
            "interval_seconds": "120",
            "center_freq_hz": "98000000",
            "span": "20",
            "lna_gain": "0",
            "vga_gain": "62",
            "rbw": "10000",
            "antenna_amp": "true",
            "sample_rate_hz": "20000000",
        },
    )

    configuration = load_campaign_configuration(campaign_dir)

    assert configuration.central_frequency_hz == 98.0e6
    assert configuration.span_hz == 20.0e6
    assert configuration.resolution_bandwidth_hz == 10.0e3
    assert configuration.acquisition_interval_s == 120.0
    assert configuration.antenna_amplifier_enabled is True


def test_prepare_calibration_campaign_builds_linear_power_campaign(
    tmp_path: Path,
) -> None:
    """Campaign preparation should ignore metadata.csv and expose the new contract."""

    campaigns_root = tmp_path / "campaigns"
    _write_campaign_fixture(
        campaigns_root / "training-campaign",
        central_freq_mhz=98.0,
        span_mhz=20.0,
        rbw_khz=10.0,
    )

    preparation = prepare_calibration_campaign(
        campaign_label="training-campaign",
        campaigns_root=campaigns_root,
        ranking_histogram_bins=8,
        distribution_histogram_bins=32,
        alignment_tolerance_ms=80,
    )

    assert preparation.metadata_path.name == "metadata.csv"
    assert preparation.campaign.sensor_ids == ("Node1", "Node2", "Node3", "Node9")
    assert preparation.campaign.observations_power.shape == (4, 5, 8)
    assert np.all(preparation.campaign.observations_power > 0.0)
    assert preparation.campaign.configuration.central_frequency_hz == 98.0e6
    assert (
        preparation.reliable_sensor_id
        not in preparation.distribution_outlier_sensor_ids
    )
    assert preparation.distribution_outlier_sensor_ids == ("Node9",)


def test_prepare_calibration_corpus_and_fit_wrapper_write_artifact(
    tmp_path: Path,
) -> None:
    """Corpus preparation and the fit wrapper should produce a reusable artifact."""

    campaigns_root = tmp_path / "campaigns"
    _write_campaign_fixture(
        campaigns_root / "campaign-a",
        central_freq_mhz=98.0,
        span_mhz=20.0,
        rbw_khz=10.0,
    )
    _write_campaign_fixture(
        campaigns_root / "campaign-b",
        central_freq_mhz=104.0,
        span_mhz=18.0,
        rbw_khz=15.0,
        sensor_offsets_db=(0.10, -0.08, 0.15, 9.5),
    )
    preparation = prepare_calibration_corpus(
        campaigns_root=campaigns_root,
        ranking_histogram_bins=8,
        distribution_histogram_bins=32,
        alignment_tolerance_ms=80,
    )
    output_dir = build_corpus_calibration_output_dir(
        "synthetic-corpus",
        models_root=tmp_path / "models",
    )

    fit_result = fit_and_save_calibration_corpus_model(
        preparation=preparation,
        output_dir=output_dir,
        basis_config=FrequencyBasisConfig(
            n_gain_basis=5,
            n_floor_basis=4,
            n_variance_basis=4,
        ),
        model_config=PersistentModelConfig(
            sensor_embedding_dim=2,
            configuration_latent_dim=2,
        ),
        fit_config=TwoLevelFitConfig(
            n_outer_iterations=2,
            n_gradient_steps=4,
            learning_rate=0.02,
            sigma_min=1.0e-8,
            lambda_delta_gain_smooth=0.1,
            lambda_delta_floor_smooth=0.1,
            lambda_delta_variance_smooth=0.1,
            lambda_delta_gain_shrink=0.05,
            lambda_delta_floor_shrink=0.05,
            lambda_delta_variance_shrink=0.05,
            lambda_reliable_sensor_anchor=0.02,
            gradient_clip_norm=2.0,
            random_seed=1,
        ),
    )
    loaded = load_two_level_calibration_artifact(output_dir)

    assert preparation.corpus.sensor_ids == ("Node1", "Node2", "Node3", "Node9")
    assert len(preparation.prepared_campaigns) == 2
    assert fit_result.fit_duration_s >= 0.0
    assert loaded.manifest["training_summary"]["n_campaigns"] == 2
    assert len(loaded.result.campaign_states) == 2


def _write_campaign_fixture(
    campaign_dir: Path,
    central_freq_mhz: float,
    span_mhz: float,
    rbw_khz: float,
    sensor_offsets_db: tuple[float, float, float, float] = (0.08, -0.05, 0.15, 12.0),
) -> None:
    """Create one deterministic campaign directory with metadata and sensor CSVs."""

    campaign_dir.mkdir(parents=True)
    _write_metadata_csv(
        campaign_dir / "metadata.csv",
        {
            "campaign_label": campaign_dir.name,
            "start_date": "03/10/2026",
            "stop_date": "03/10/2026",
            "start_time": "00:00:00",
            "stop_time": "06:00:00",
            "acquisition_freq_minutes": "2",
            "central_freq_MHz": f"{central_freq_mhz}",
            "span_MHz": f"{span_mhz}",
            "lna_gain_dB": "0",
            "vga_gain_dB": "62",
            "rbw_kHz": f"{rbw_khz}",
            "antenna_amp": "true",
        },
    )

    base_records = np.asarray(
        [
            [-63.0, -61.0, -58.0, -54.0, -50.0, -49.0, -52.0, -57.0],
            [-62.5, -60.3, -57.1, -53.4, -49.7, -48.8, -51.6, -56.3],
            [-64.1, -61.7, -58.9, -55.1, -51.0, -50.2, -53.4, -58.0],
            [-63.6, -61.4, -58.3, -54.7, -50.5, -49.6, -52.8, -57.5],
            [-62.8, -60.7, -57.8, -53.8, -49.4, -48.6, -51.9, -56.8],
        ],
        dtype=np.float64,
    )
    sensor_ids = ("Node1", "Node2", "Node3", "Node9")
    timestamp_series = (
        [1_000, 2_000, 3_000, 4_000, 5_000],
        [1_030, 2_030, 3_030, 4_030, 5_030],
        [980, 1_980, 2_980, 3_980, 4_980],
        [1_050, 2_050, 3_050, 4_050, 5_050],
    )
    for sensor_id, timestamps_ms, offset_db in zip(
        sensor_ids,
        timestamp_series,
        sensor_offsets_db,
        strict=True,
    ):
        observations_db = (
            np.flip(base_records, axis=1) + offset_db
            if sensor_id == "Node9"
            else base_records + offset_db
        )
        _write_sensor_csv(
            campaign_dir / f"{sensor_id}.csv",
            timestamps_ms=timestamps_ms,
            observations_db=observations_db,
        )


def _write_metadata_csv(path: Path, row: dict[str, str]) -> None:
    """Write one-row campaign metadata CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_sensor_csv(
    path: Path,
    timestamps_ms: list[int],
    observations_db: np.ndarray,
) -> None:
    """Write one sensor acquisition CSV compatible with the repository schema."""

    rows = []
    for row_index, (timestamp_ms, power_db) in enumerate(
        zip(timestamps_ms, observations_db, strict=True),
        start=1,
    ):
        rows.append(
            {
                "id": row_index,
                "mac": f"00:00:00:00:00:{row_index:02d}",
                "campaign_id": 1,
                "pxx": json.dumps([float(value) for value in power_db]),
                "start_freq_hz": 88_000_000,
                "end_freq_hz": 108_000_000,
                "timestamp": timestamp_ms,
                "lat": 0.0,
                "lng": 0.0,
                "excursion_peak_to_peak_hz": 0.0,
                "excursion_peak_deviation_hz": 0.0,
                "excursion_rms_deviation_hz": 0.0,
                "depth_peak_to_peak": 0.0,
                "depth_peak_deviation": 0.0,
                "depth_rms_deviation": 0.0,
                "created_at": timestamp_ms,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
