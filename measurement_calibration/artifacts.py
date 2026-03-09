"""Filesystem artifact helpers for fitted spectral calibration models.

This module keeps persistence and report-generation concerns at the project
boundary. The numerical model remains in :mod:`measurement_calibration.
spectral_calibration`, while this module serializes fitted results into a
stable on-disk layout that can be inspected and reloaded later.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .spectral_calibration import (
    CalibrationDataset,
    SpectralCalibrationResult,
    power_linear_to_db,
)


MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SavedCalibrationArtifact:
    """Paths and metadata for one saved calibration artifact.

    Parameters
    ----------
    output_dir:
        Directory that contains every generated artifact file.
    manifest_path:
        JSON manifest path with provenance, fit configuration, and summary
        metrics.
    parameters_path:
        Compressed NumPy archive with the fitted calibration arrays.
    sensor_summary_path:
        CSV path with one human-readable summary row per calibrated sensor.
    manifest:
        Parsed manifest dictionary that was written to disk.
    """

    output_dir: Path
    manifest_path: Path
    parameters_path: Path
    sensor_summary_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedCalibrationArtifact:
    """In-memory representation of a previously saved calibration artifact.

    Parameters
    ----------
    output_dir:
        Directory that contains the artifact files.
    manifest_path:
        JSON manifest path.
    parameters_path:
        Compressed NumPy archive path.
    sensor_summary_path:
        CSV summary path. The file may not exist for legacy artifacts.
    manifest:
        Parsed manifest dictionary loaded from disk.
    result:
        Reconstructed spectral calibration result.
    """

    output_dir: Path
    manifest_path: Path
    parameters_path: Path
    sensor_summary_path: Path
    manifest: dict[str, Any]
    result: SpectralCalibrationResult


def save_spectral_calibration_artifact(
    output_dir: Path,  # Destination directory for the artifact bundle
    result: SpectralCalibrationResult,  # Fitted calibration parameters
    dataset: CalibrationDataset,  # Dataset used to produce the fitted model
    acquisition_dir: Path,  # Source acquisition directory
    response_dir: Path,  # Source nominal-response directory
    reference_sensor_id: str,  # Alignment reference sensor
    reliable_sensor_id: str,  # Softly anchored sensor during fitting
    excluded_sensor_ids: Collection[str],  # Sensors removed before training
    fit_config: Mapping[str, int | float],  # Numerical fitting configuration
    extra_summary: Mapping[str, int | float] | None = None,  # Optional run metrics
) -> SavedCalibrationArtifact:  # Paths and manifest for the stored artifact
    """Persist a fitted calibration result as a self-describing artifact bundle.

    Parameters
    ----------
    output_dir:
        Directory where the artifact bundle will be written. It is created when
        needed, and the standard artifact files are overwritten in place.
    result:
        Fitted calibration result to serialize.
    dataset:
        Calibration dataset that supplied the aligned observations and metadata.
    acquisition_dir:
        Directory containing the raw acquisition CSV files used during loading.
    response_dir:
        Directory containing the nominal response CSV files used during loading.
    reference_sensor_id:
        Sensor that defined the alignment timeline.
    reliable_sensor_id:
        Sensor softly anchored during fitting.
    excluded_sensor_ids:
        Sensors excluded before dataset construction.
    fit_config:
        Numerical fit configuration. Only JSON-serializable scalars are
        accepted so the manifest remains explicit and portable.
    extra_summary:
        Optional scalar run metrics, for example fit duration or holdout
        fraction, that should appear in the manifest summary.

    Returns
    -------
    SavedCalibrationArtifact
        Artifact paths plus the manifest dictionary that was written.

    Side Effects
    ------------
    Creates ``output_dir`` if needed and writes three files:
    ``manifest.json``, ``calibration_parameters.npz``, and
    ``sensor_summary.csv``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters_path = output_dir / "calibration_parameters.npz"
    manifest_path = output_dir / "manifest.json"
    sensor_summary_path = output_dir / "sensor_summary.csv"

    np.savez_compressed(
        parameters_path,
        sensor_ids=np.asarray(result.sensor_ids),
        frequency_hz=result.frequency_hz,
        gain_power=result.gain_power,
        additive_noise_power=result.additive_noise_power,
        residual_variance_power2=result.residual_variance_power2,
        latent_spectra_power=result.latent_spectra_power,
        nominal_gain_power=result.nominal_gain_power,
        correction_gain_power=result.correction_gain_power,
        train_indices=result.train_indices,
        test_indices=result.test_indices,
        objective_history=result.objective_history,
        latent_variation_power2=result.latent_variation_power2,
        frequency_information_weight=result.frequency_information_weight,
        information_weight=result.information_weight,
        frequency_low_information_mask=result.frequency_low_information_mask,
        low_information_mask=result.low_information_mask,
        gain_at_correction_bound_mask=result.gain_at_correction_bound_mask,
        noise_zero_mask=result.noise_zero_mask,
    )

    write_sensor_calibration_summary_csv(sensor_summary_path, result)

    manifest = _build_artifact_manifest(
        result=result,
        dataset=dataset,
        acquisition_dir=acquisition_dir,
        response_dir=response_dir,
        reference_sensor_id=reference_sensor_id,
        reliable_sensor_id=reliable_sensor_id,
        excluded_sensor_ids=excluded_sensor_ids,
        fit_config=fit_config,
        parameters_path=parameters_path.name,
        sensor_summary_path=sensor_summary_path.name,
        extra_summary=extra_summary,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return SavedCalibrationArtifact(
        output_dir=output_dir,
        manifest_path=manifest_path,
        parameters_path=parameters_path,
        sensor_summary_path=sensor_summary_path,
        manifest=manifest,
    )


def load_spectral_calibration_artifact(
    output_dir: Path,  # Directory that contains the artifact bundle
) -> LoadedCalibrationArtifact:  # Reconstructed result and manifest
    """Load a previously saved calibration artifact from disk.

    Parameters
    ----------
    output_dir:
        Directory that contains ``manifest.json`` and the parameter archive.

    Returns
    -------
    LoadedCalibrationArtifact
        Parsed manifest plus the reconstructed spectral calibration result.

    Raises
    ------
    FileNotFoundError
        If the manifest or parameter archive is missing.
    ValueError
        If the artifact schema version is unsupported.
    """

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Artifact manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(manifest["schema_version"])
    if schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported calibration artifact schema version: "
            f"{schema_version}. Expected {MODEL_SCHEMA_VERSION}."
        )

    parameters_path = output_dir / str(manifest["parameters_file"])
    if not parameters_path.exists():
        raise FileNotFoundError(
            f"Artifact parameter archive does not exist: {parameters_path}"
        )

    with np.load(parameters_path, allow_pickle=False) as arrays:
        result = SpectralCalibrationResult(
            sensor_ids=tuple(str(sensor_id) for sensor_id in arrays["sensor_ids"]),
            frequency_hz=np.asarray(arrays["frequency_hz"], dtype=np.float64),
            gain_power=np.asarray(arrays["gain_power"], dtype=np.float64),
            additive_noise_power=np.asarray(
                arrays["additive_noise_power"], dtype=np.float64
            ),
            residual_variance_power2=np.asarray(
                arrays["residual_variance_power2"], dtype=np.float64
            ),
            latent_spectra_power=np.asarray(
                arrays["latent_spectra_power"], dtype=np.float64
            ),
            nominal_gain_power=np.asarray(
                arrays["nominal_gain_power"], dtype=np.float64
            ),
            correction_gain_power=np.asarray(
                arrays["correction_gain_power"], dtype=np.float64
            ),
            train_indices=np.asarray(arrays["train_indices"], dtype=np.int64),
            test_indices=np.asarray(arrays["test_indices"], dtype=np.int64),
            objective_history=np.asarray(arrays["objective_history"], dtype=np.float64),
            latent_variation_power2=np.asarray(
                arrays["latent_variation_power2"], dtype=np.float64
            ),
            frequency_information_weight=np.asarray(
                arrays["frequency_information_weight"], dtype=np.float64
            ),
            information_weight=np.asarray(
                arrays["information_weight"], dtype=np.float64
            ),
            frequency_low_information_mask=np.asarray(
                arrays["frequency_low_information_mask"], dtype=bool
            ),
            low_information_mask=np.asarray(arrays["low_information_mask"], dtype=bool),
            gain_at_correction_bound_mask=np.asarray(
                arrays["gain_at_correction_bound_mask"], dtype=bool
            ),
            noise_zero_mask=np.asarray(arrays["noise_zero_mask"], dtype=bool),
        )

    sensor_summary_path = output_dir / str(manifest["sensor_summary_file"])
    return LoadedCalibrationArtifact(
        output_dir=output_dir,
        manifest_path=manifest_path,
        parameters_path=parameters_path,
        sensor_summary_path=sensor_summary_path,
        manifest=manifest,
        result=result,
    )


def write_sensor_calibration_summary_csv(
    output_path: Path,  # CSV destination path
    result: SpectralCalibrationResult,  # Fitted calibration result to summarize
) -> None:  # No return value
    """Write one human-readable calibration summary row per sensor.

    The CSV is meant for quick inspection. It reports compact per-sensor
    statistics rather than the full frequency-resolved model arrays, which stay
    in the compressed NumPy archive.
    """

    output_path = Path(output_path)
    rows = _build_sensor_summary_rows(result)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_artifact_manifest(
    result: SpectralCalibrationResult,
    dataset: CalibrationDataset,
    acquisition_dir: Path,
    response_dir: Path,
    reference_sensor_id: str,
    reliable_sensor_id: str,
    excluded_sensor_ids: Collection[str],
    fit_config: Mapping[str, int | float],
    parameters_path: str,
    sensor_summary_path: str,
    extra_summary: Mapping[str, int | float] | None,
) -> dict[str, Any]:
    """Build the JSON manifest dictionary for a saved artifact."""

    manifest: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "spectral_calibration_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters_file": parameters_path,
        "sensor_summary_file": sensor_summary_path,
        "acquisition_dir": str(Path(acquisition_dir).resolve()),
        "response_dir": str(Path(response_dir).resolve()),
        "reference_sensor_id": reference_sensor_id,
        "reliable_sensor_id": reliable_sensor_id,
        "excluded_sensor_ids": sorted(
            str(sensor_id) for sensor_id in excluded_sensor_ids
        ),
        "fit_config": _normalize_scalar_mapping(fit_config),
        "dataset": {
            "sensor_ids": list(dataset.sensor_ids),
            "n_sensors": len(dataset.sensor_ids),
            "n_experiments": int(dataset.observations_power.shape[1]),
            "n_frequencies": int(dataset.frequency_hz.size),
            "selected_band_hz": [
                float(dataset.selected_band_hz[0]),
                float(dataset.selected_band_hz[1]),
            ],
            "sensor_shifts": {
                sensor_id: int(dataset.sensor_shifts[sensor_id])
                for sensor_id in dataset.sensor_ids
            },
            "alignment_median_error_ms": {
                sensor_id: float(dataset.alignment_median_error_ms[sensor_id])
                for sensor_id in dataset.sensor_ids
            },
            "experiment_timestamp_range_ms": [
                int(dataset.experiment_timestamps_ms[0]),
                int(dataset.experiment_timestamps_ms[-1]),
            ],
            "selected_rows_per_sensor": {
                sensor_id: int(dataset.source_row_indices[sensor_id].size)
                for sensor_id in dataset.sensor_ids
            },
        },
        "result_summary": _result_summary(result),
    }
    if extra_summary is not None:
        manifest["extra_summary"] = _normalize_scalar_mapping(extra_summary)
    return manifest


def _normalize_scalar_mapping(
    values: Mapping[str, int | float],  # Mapping with NumPy or Python scalars
) -> dict[str, int | float]:
    """Convert a scalar mapping into a JSON-friendly plain-Python dictionary."""

    normalized: dict[str, int | float] = {}
    for key, value in values.items():
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            normalized[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            normalized[key] = float(value)
        else:
            raise TypeError(
                "Artifact manifest supports only integer and floating-point "
                f"scalars, but {key!r} received {type(value).__name__}."
            )
    return normalized


def _result_summary(
    result: SpectralCalibrationResult,  # Fitted calibration result to summarize
) -> dict[str, int | float]:
    """Compute compact artifact-level summary metrics."""

    return {
        "train_experiments": int(result.train_indices.size),
        "test_experiments": int(result.test_indices.size),
        "objective_start": float(result.objective_history[0]),
        "objective_end": float(result.objective_history[-1]),
        "global_low_information_fraction": float(
            np.mean(result.frequency_low_information_mask)
        ),
        "sensor_low_information_fraction": float(np.mean(result.low_information_mask)),
        "gain_cap_fraction": float(np.mean(result.gain_at_correction_bound_mask)),
        "noise_zero_fraction": float(np.mean(result.noise_zero_mask)),
    }


def _build_sensor_summary_rows(
    result: SpectralCalibrationResult,  # Fitted calibration result to summarize
) -> list[dict[str, float | str]]:
    """Build the rows written to the human-readable sensor summary CSV."""

    correction_gain_db = power_linear_to_db(result.correction_gain_power)
    gain_power_db = power_linear_to_db(result.gain_power)
    additive_noise_db = power_linear_to_db(result.additive_noise_power)
    residual_std_db = power_linear_to_db(np.sqrt(result.residual_variance_power2))

    rows: list[dict[str, float | str]] = []
    for sensor_index, sensor_id in enumerate(result.sensor_ids):
        # Summarize only intent-relevant aggregates here so the CSV remains
        # quick to scan while the full curves stay in the NPZ archive.
        rows.append(
            {
                "sensor_id": sensor_id,
                "median_total_gain_db": float(np.median(gain_power_db[sensor_index])),
                "median_correction_gain_db": float(
                    np.median(correction_gain_db[sensor_index])
                ),
                "max_abs_correction_gain_db": float(
                    np.max(np.abs(correction_gain_db[sensor_index]))
                ),
                "median_additive_noise_db": float(
                    np.median(additive_noise_db[sensor_index])
                ),
                "median_residual_std_db": float(
                    np.median(residual_std_db[sensor_index])
                ),
                "low_information_fraction": float(
                    np.mean(result.low_information_mask[sensor_index])
                ),
                "gain_cap_fraction": float(
                    np.mean(result.gain_at_correction_bound_mask[sensor_index])
                ),
                "noise_zero_fraction": float(
                    np.mean(result.noise_zero_mask[sensor_index])
                ),
            }
        )
    return rows
