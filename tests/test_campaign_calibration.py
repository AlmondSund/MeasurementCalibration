"""Tests for the dynamic campaign-calibration adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from measurement_calibration.artifacts import load_spectral_calibration_artifact
from measurement_calibration.campaign_calibration import (
    build_campaign_calibration_output_dir,
    build_campaign_deployment_inputs,
    fit_and_save_campaign_calibration_model,
    prepare_campaign_calibration_dataset,
)
from measurement_calibration.spectral_calibration import power_db_to_linear


def test_prepare_campaign_calibration_dataset_builds_linear_tensor(
    tmp_path: Path,
) -> None:
    """Campaign preparation should align records and expose the generic fit tensor."""

    campaigns_root = tmp_path / "campaigns"
    _write_synthetic_campaign(campaigns_root / "training-campaign")

    preparation = prepare_campaign_calibration_dataset(
        "training-campaign",
        campaigns_root=campaigns_root,
        ranking_histogram_bins=8,
        distribution_histogram_bins=32,
        alignment_tolerance_ms=80,
    )
    dataset = preparation.calibration_dataset

    assert dataset.sensor_ids == ("Node1", "Node2", "Node3", "Node9")
    assert dataset.observations_power.shape == (4, 5, 8)
    assert np.allclose(
        dataset.observations_power,
        power_db_to_linear(preparation.aligned_dataset.observations_db),
    )
    assert np.allclose(dataset.nominal_gain_power, 1.0)
    assert np.array_equal(
        dataset.experiment_timestamps_ms,
        np.median(preparation.aligned_dataset.timestamps_ms, axis=0).astype(np.int64),
    )
    assert dataset.selected_band_hz == (88.0e6, 108.0e6)
    assert (
        preparation.reliable_sensor_id
        == preparation.ranking_result.ranking_sensor_ids[0]
    )
    assert (
        preparation.reliable_sensor_id
        not in preparation.distribution_outlier_sensor_ids
    )
    assert preparation.distribution_outlier_sensor_ids == ("Node9",)
    assert all(
        dataset.sensor_shifts[sensor_id] == 0 for sensor_id in dataset.sensor_ids
    )
    assert all(
        np.array_equal(
            dataset.source_row_indices[sensor_id], np.arange(5, dtype=np.int64)
        )
        for sensor_id in dataset.sensor_ids
    )


def test_prepare_campaign_calibration_dataset_rejects_unknown_exclusions(
    tmp_path: Path,
) -> None:
    """Preparation should fail fast when the caller excludes an unknown sensor."""

    campaigns_root = tmp_path / "campaigns"
    _write_synthetic_campaign(campaigns_root / "training-campaign")

    with pytest.raises(ValueError, match="unknown sensors"):
        prepare_campaign_calibration_dataset(
            "training-campaign",
            campaigns_root=campaigns_root,
            excluded_sensor_ids=("Node404",),
        )


def test_fit_and_save_campaign_calibration_model_writes_artifact_bundle(
    tmp_path: Path,
) -> None:
    """Campaign fitting should persist a reusable artifact bundle under models/."""

    campaigns_root = tmp_path / "campaigns"
    _write_synthetic_campaign(campaigns_root / "training-campaign")
    preparation = prepare_campaign_calibration_dataset(
        "training-campaign",
        campaigns_root=campaigns_root,
        ranking_histogram_bins=8,
        distribution_histogram_bins=32,
        alignment_tolerance_ms=80,
    )
    output_dir = build_campaign_calibration_output_dir(
        preparation.campaign_label,
        models_root=tmp_path / "models",
    )

    fit_result = fit_and_save_campaign_calibration_model(
        preparation=preparation,
        output_dir=output_dir,
        fit_config={
            "n_iterations": 3,
            "lambda_gain_smooth": 10.0,
            "lambda_noise_smooth": 10.0,
            "lambda_gain_reference": 3.0,
            "lambda_noise_reference": 20.0,
            "lambda_reliable_anchor": 1.0,
            "reliable_weight_boost": 1.05,
            "low_information_threshold_ratio": 0.10,
            "low_information_weight": 0.05,
        },
        test_fraction=0.4,
        split_strategy="random",
        split_random_seed=0,
    )
    loaded = load_spectral_calibration_artifact(output_dir)

    assert loaded.manifest["response_dir"] is None
    assert loaded.manifest["acquisition_dir"] == str(preparation.campaign_dir.resolve())
    assert loaded.manifest["reference_sensor_id"] == (
        preparation.alignment_diagnostics.anchor_sensor_id
    )
    assert loaded.manifest["reliable_sensor_id"] == preparation.reliable_sensor_id
    assert loaded.manifest["excluded_sensor_ids"] == []
    assert loaded.manifest["dataset"]["sensor_ids"] == list(
        preparation.calibration_dataset.sensor_ids
    )
    assert loaded.result.sensor_ids == preparation.calibration_dataset.sensor_ids
    assert loaded.manifest["extra_summary"]["corrected_to_raw_dispersion_ratio"] > 0.0
    assert loaded.manifest["extra_summary"]["distribution_outlier_count"] == 1.0
    assert fit_result.fit_duration_s >= 0.0
    assert len(fit_result.validation.sensor_rows) == len(
        preparation.calibration_dataset.sensor_ids
    )


def test_build_campaign_deployment_inputs_reorders_shared_sensors(
    tmp_path: Path,
) -> None:
    """Deployment inputs should follow the stored model order on shared sensors."""

    campaigns_root = tmp_path / "campaigns"
    _write_synthetic_campaign(campaigns_root / "training-campaign")
    _write_deployment_campaign(campaigns_root / "deployment-campaign")

    training_preparation = prepare_campaign_calibration_dataset(
        "training-campaign",
        campaigns_root=campaigns_root,
        ranking_histogram_bins=8,
        distribution_histogram_bins=32,
        alignment_tolerance_ms=80,
    )
    fit_result = fit_and_save_campaign_calibration_model(
        preparation=training_preparation,
        output_dir=tmp_path / "models" / "training-campaign",
        fit_config={
            "n_iterations": 2,
            "lambda_gain_smooth": 10.0,
            "lambda_noise_smooth": 10.0,
            "lambda_gain_reference": 3.0,
            "lambda_noise_reference": 20.0,
            "lambda_reliable_anchor": 1.0,
            "reliable_weight_boost": 1.05,
            "low_information_threshold_ratio": 0.10,
            "low_information_weight": 0.05,
        },
        test_fraction=0.4,
        split_strategy="tail",
    )
    deployment_preparation = prepare_campaign_calibration_dataset(
        "deployment-campaign",
        campaigns_root=campaigns_root,
        alignment_tolerance_ms=80,
    )

    deployment_inputs = build_campaign_deployment_inputs(
        preparation=deployment_preparation,
        trained_result=load_spectral_calibration_artifact(
            fit_result.artifact.output_dir
        ).result,
    )

    assert deployment_inputs.shared_sensor_ids == ("Node1", "Node3")
    assert deployment_inputs.observations_power.shape == (2, 4, 8)
    assert np.allclose(
        deployment_inputs.observations_power[0],
        deployment_preparation.calibration_dataset.observations_power[0],
    )
    assert np.allclose(
        deployment_inputs.observations_power[1],
        deployment_preparation.calibration_dataset.observations_power[2],
    )


def _write_synthetic_campaign(campaign_dir: Path) -> None:
    """Create a deterministic campaign fixture with one clear histogram outlier."""

    campaign_dir.mkdir(parents=True)
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

    # Keep three sensors close to the latent field and shift one sensor into a
    # clearly incompatible distribution so the ranking and outlier logic are exercised.
    _write_campaign_csv(
        campaign_dir / "Node1.csv",
        timestamps_ms=[1_000, 2_000, 3_000, 4_000, 5_000],
        observations_db=(base_records + 0.08).tolist(),
    )
    _write_campaign_csv(
        campaign_dir / "Node2.csv",
        timestamps_ms=[1_030, 2_030, 3_030, 4_030, 5_030],
        observations_db=(base_records - 0.05).tolist(),
    )
    _write_campaign_csv(
        campaign_dir / "Node3.csv",
        timestamps_ms=[980, 1_980, 2_980, 3_980, 4_980],
        observations_db=(base_records + 0.15).tolist(),
    )
    _write_campaign_csv(
        campaign_dir / "Node9.csv",
        timestamps_ms=[1_050, 2_050, 3_050, 4_050, 5_050],
        observations_db=(np.flip(base_records, axis=1) + 12.0).tolist(),
    )


def _write_deployment_campaign(campaign_dir: Path) -> None:
    """Create a second campaign with partial overlap against the training sensors."""

    campaign_dir.mkdir(parents=True)
    base_records = np.asarray(
        [
            [-62.8, -60.8, -57.8, -53.8, -49.8, -48.9, -51.8, -56.9],
            [-63.2, -61.0, -58.1, -54.1, -50.1, -49.1, -52.1, -57.2],
            [-62.4, -60.5, -57.5, -53.5, -49.6, -48.7, -51.5, -56.6],
            [-63.0, -60.9, -57.9, -53.9, -49.9, -49.0, -51.9, -57.0],
        ],
        dtype=np.float64,
    )

    _write_campaign_csv(
        campaign_dir / "Node1.csv",
        timestamps_ms=[11_000, 12_000, 13_000, 14_000],
        observations_db=(base_records + 0.10).tolist(),
    )
    _write_campaign_csv(
        campaign_dir / "Node3.csv",
        timestamps_ms=[11_020, 12_020, 13_020, 14_020],
        observations_db=(base_records + 0.18).tolist(),
    )
    _write_campaign_csv(
        campaign_dir / "Node10.csv",
        timestamps_ms=[10_980, 11_980, 12_980, 13_980],
        observations_db=(base_records + 4.0).tolist(),
    )


def _write_campaign_csv(
    path: Path,  # Output CSV path
    timestamps_ms: list[int],  # Row timestamps [ms]
    observations_db: list[list[float]],  # PSD rows in dB
) -> None:
    """Write a campaign-style CSV with the API client schema."""

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "id",
                "mac",
                "campaign_id",
                "pxx",
                "start_freq_hz",
                "end_freq_hz",
                "timestamp",
                "created_at",
            ),
        )
        writer.writeheader()

        for row_index, (timestamp_ms, power_db) in enumerate(
            zip(timestamps_ms, observations_db, strict=True),
            start=1,
        ):
            writer.writerow(
                {
                    "id": row_index,
                    "mac": f"mac-{path.stem.lower()}",
                    "campaign_id": 999,
                    "pxx": json.dumps(power_db),
                    "start_freq_hz": 88.0e6,
                    "end_freq_hz": 108.0e6,
                    "timestamp": timestamp_ms,
                    "created_at": "",
                }
            )
