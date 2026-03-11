"""Artifact persistence for the two-level configuration-conditional model.

This module keeps filesystem side effects at the project boundary. The
numerical model remains in :mod:`measurement_calibration.spectral_calibration`;
this module serializes fitted results into a stable on-disk bundle that can be
reloaded for deployment or inspection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .spectral_calibration import (
    CampaignCalibrationState,
    CampaignConfiguration,
    FrequencyBasisConfig,
    PersistentModelConfig,
    TwoLevelCalibrationResult,
    TwoLevelFitConfig,
    power_linear_to_db,
)


MODEL_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SavedCalibrationArtifact:
    """Paths and metadata for one saved calibration artifact bundle."""

    output_dir: Path
    manifest_path: Path
    parameters_path: Path
    sensor_summary_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LoadedCalibrationArtifact:
    """In-memory representation of a previously saved artifact bundle."""

    output_dir: Path
    manifest_path: Path
    parameters_path: Path
    sensor_summary_path: Path
    manifest: dict[str, Any]
    result: TwoLevelCalibrationResult


def save_two_level_calibration_artifact(
    output_dir: Path,  # Destination directory for the artifact bundle
    result: TwoLevelCalibrationResult,  # Fitted calibration model
    extra_summary: dict[str, int | float] | None = None,
) -> SavedCalibrationArtifact:
    """Persist a fitted two-level calibration model to disk.

    Side Effects
    ------------
    Creates ``output_dir`` when needed and writes:

    - ``manifest.json`` with configuration, provenance, and high-level summary;
    - ``calibration_parameters.npz`` with the trainable arrays and campaign states;
    - ``sensor_summary.csv`` with compact per-sensor training statistics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters_path = output_dir / "calibration_parameters.npz"
    manifest_path = output_dir / "manifest.json"
    sensor_summary_path = output_dir / "sensor_summary.csv"

    np.savez_compressed(
        parameters_path,
        **_build_parameter_archive_payload(result),
    )
    write_sensor_calibration_summary_csv(sensor_summary_path, result)

    manifest = _build_artifact_manifest(
        result=result,
        parameters_file=parameters_path.name,
        sensor_summary_file=sensor_summary_path.name,
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


def load_two_level_calibration_artifact(
    output_dir: Path,  # Directory that contains the saved artifact bundle
) -> LoadedCalibrationArtifact:
    """Load a previously saved two-level calibration artifact from disk."""

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
        campaign_states = tuple(
            _load_campaign_state(
                arrays=arrays,
                campaign_manifest=campaign_manifest,
                campaign_index=campaign_index,
            )
            for campaign_index, campaign_manifest in enumerate(manifest["campaigns"])
        )
        result = TwoLevelCalibrationResult(
            sensor_ids=tuple(str(sensor_id) for sensor_id in arrays["sensor_ids"]),
            sensor_reference_weight=np.asarray(
                arrays["sensor_reference_weight"], dtype=np.float64
            ),
            basis_config=FrequencyBasisConfig(**manifest["basis_config"]),
            model_config=PersistentModelConfig(**manifest["model_config"]),
            fit_config=TwoLevelFitConfig(**manifest["fit_config"]),
            configuration_feature_mean=np.asarray(
                arrays["configuration_feature_mean"], dtype=np.float64
            ),
            configuration_feature_scale=np.asarray(
                arrays["configuration_feature_scale"], dtype=np.float64
            ),
            frequency_min_hz=float(arrays["frequency_min_hz"][0]),
            frequency_max_hz=float(arrays["frequency_max_hz"][0]),
            sensor_embeddings=np.asarray(arrays["sensor_embeddings"], dtype=np.float64),
            configuration_encoder_weight=np.asarray(
                arrays["configuration_encoder_weight"], dtype=np.float64
            ),
            configuration_encoder_bias=np.asarray(
                arrays["configuration_encoder_bias"], dtype=np.float64
            ),
            gain_head_weight=np.asarray(arrays["gain_head_weight"], dtype=np.float64),
            gain_head_bias=np.asarray(arrays["gain_head_bias"], dtype=np.float64),
            floor_head_weight=np.asarray(arrays["floor_head_weight"], dtype=np.float64),
            floor_head_bias=np.asarray(arrays["floor_head_bias"], dtype=np.float64),
            variance_head_weight=np.asarray(
                arrays["variance_head_weight"], dtype=np.float64
            ),
            variance_head_bias=np.asarray(arrays["variance_head_bias"], dtype=np.float64),
            campaign_states=campaign_states,
            objective_history=np.asarray(arrays["objective_history"], dtype=np.float64),
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
    result: TwoLevelCalibrationResult,  # Fitted model to summarize
) -> None:
    """Write one compact training summary row per registered sensor."""

    output_path = Path(output_path)
    rows = _build_sensor_summary_rows(result)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_parameter_archive_payload(
    result: TwoLevelCalibrationResult,
) -> dict[str, Any]:
    """Flatten a fitted result into NPZ-compatible arrays."""

    payload: dict[str, Any] = {
        "sensor_ids": np.asarray(result.sensor_ids),
        "sensor_reference_weight": np.asarray(
            result.sensor_reference_weight, dtype=np.float64
        ),
        "configuration_feature_mean": np.asarray(
            result.configuration_feature_mean, dtype=np.float64
        ),
        "configuration_feature_scale": np.asarray(
            result.configuration_feature_scale, dtype=np.float64
        ),
        "frequency_min_hz": np.asarray([result.frequency_min_hz], dtype=np.float64),
        "frequency_max_hz": np.asarray([result.frequency_max_hz], dtype=np.float64),
        "sensor_embeddings": np.asarray(result.sensor_embeddings, dtype=np.float64),
        "configuration_encoder_weight": np.asarray(
            result.configuration_encoder_weight, dtype=np.float64
        ),
        "configuration_encoder_bias": np.asarray(
            result.configuration_encoder_bias, dtype=np.float64
        ),
        "gain_head_weight": np.asarray(result.gain_head_weight, dtype=np.float64),
        "gain_head_bias": np.asarray(result.gain_head_bias, dtype=np.float64),
        "floor_head_weight": np.asarray(result.floor_head_weight, dtype=np.float64),
        "floor_head_bias": np.asarray(result.floor_head_bias, dtype=np.float64),
        "variance_head_weight": np.asarray(
            result.variance_head_weight, dtype=np.float64
        ),
        "variance_head_bias": np.asarray(result.variance_head_bias, dtype=np.float64),
        "objective_history": np.asarray(result.objective_history, dtype=np.float64),
    }
    for campaign_index, campaign_state in enumerate(result.campaign_states):
        payload[f"campaign_{campaign_index}_sensor_ids"] = np.asarray(
            campaign_state.sensor_ids
        )
        payload[f"campaign_{campaign_index}_frequency_hz"] = np.asarray(
            campaign_state.frequency_hz, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_latent_spectra_power"] = np.asarray(
            campaign_state.latent_spectra_power, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_persistent_log_gain"] = np.asarray(
            campaign_state.persistent_log_gain, dtype=np.float64
        )
        payload[
            f"campaign_{campaign_index}_persistent_floor_parameter"
        ] = np.asarray(campaign_state.persistent_floor_parameter, dtype=np.float64)
        payload[
            f"campaign_{campaign_index}_persistent_variance_parameter"
        ] = np.asarray(campaign_state.persistent_variance_parameter, dtype=np.float64)
        payload[f"campaign_{campaign_index}_deviation_log_gain"] = np.asarray(
            campaign_state.deviation_log_gain, dtype=np.float64
        )
        payload[
            f"campaign_{campaign_index}_deviation_floor_parameter"
        ] = np.asarray(campaign_state.deviation_floor_parameter, dtype=np.float64)
        payload[
            f"campaign_{campaign_index}_deviation_variance_parameter"
        ] = np.asarray(campaign_state.deviation_variance_parameter, dtype=np.float64)
        payload[f"campaign_{campaign_index}_gain_power"] = np.asarray(
            campaign_state.gain_power, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_additive_noise_power"] = np.asarray(
            campaign_state.additive_noise_power, dtype=np.float64
        )
        payload[
            f"campaign_{campaign_index}_residual_variance_power2"
        ] = np.asarray(campaign_state.residual_variance_power2, dtype=np.float64)
        payload[f"campaign_{campaign_index}_objective_value"] = np.asarray(
            [campaign_state.objective_value], dtype=np.float64
        )
    return payload


def _build_artifact_manifest(
    result: TwoLevelCalibrationResult,
    parameters_file: str,
    sensor_summary_file: str,
    extra_summary: dict[str, int | float] | None,
) -> dict[str, Any]:
    """Build the JSON manifest dictionary for one saved artifact."""

    manifest: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "configuration_conditional_calibration_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters_file": parameters_file,
        "sensor_summary_file": sensor_summary_file,
        "basis_config": asdict(result.basis_config),
        "model_config": asdict(result.model_config),
        "fit_config": asdict(result.fit_config),
        "sensor_ids": list(result.sensor_ids),
        "sensor_reference_weight": {
            sensor_id: float(weight)
            for sensor_id, weight in zip(
                result.sensor_ids,
                result.sensor_reference_weight,
                strict=True,
            )
        },
        "training_summary": _training_summary(result),
        "campaigns": [_campaign_manifest_entry(campaign_state) for campaign_state in result.campaign_states],
    }
    if extra_summary is not None:
        manifest["extra_summary"] = _normalize_scalar_mapping(extra_summary)
    return manifest


def _training_summary(
    result: TwoLevelCalibrationResult,
) -> dict[str, int | float]:
    """Compute compact artifact-level summary metrics."""

    return {
        "n_campaigns": len(result.campaign_states),
        "n_sensors": len(result.sensor_ids),
        "objective_start": float(result.objective_history[0]),
        "objective_end": float(result.objective_history[-1]),
        "mean_campaign_objective": float(
            np.mean([campaign_state.objective_value for campaign_state in result.campaign_states])
        ),
    }


def _campaign_manifest_entry(
    campaign_state: CampaignCalibrationState,
) -> dict[str, Any]:
    """Build the manifest entry for one campaign state."""

    return {
        "campaign_label": campaign_state.campaign_label,
        "sensor_ids": list(campaign_state.sensor_ids),
        "n_acquisitions": int(campaign_state.latent_spectra_power.shape[0]),
        "n_frequencies": int(campaign_state.frequency_hz.size),
        "reliable_sensor_id": campaign_state.reliable_sensor_id,
        "configuration": asdict(campaign_state.configuration),
        "objective_value": float(campaign_state.objective_value),
    }


def _load_campaign_state(
    arrays: Any,
    campaign_manifest: Mapping[str, Any],
    campaign_index: int,
) -> CampaignCalibrationState:
    """Reconstruct one campaign state from the NPZ archive and manifest."""

    return CampaignCalibrationState(
        campaign_label=str(campaign_manifest["campaign_label"]),
        sensor_ids=tuple(
            str(sensor_id)
            for sensor_id in arrays[f"campaign_{campaign_index}_sensor_ids"]
        ),
        frequency_hz=np.asarray(
            arrays[f"campaign_{campaign_index}_frequency_hz"], dtype=np.float64
        ),
        configuration=CampaignConfiguration(**campaign_manifest["configuration"]),
        reliable_sensor_id=campaign_manifest["reliable_sensor_id"],
        latent_spectra_power=np.asarray(
            arrays[f"campaign_{campaign_index}_latent_spectra_power"],
            dtype=np.float64,
        ),
        persistent_log_gain=np.asarray(
            arrays[f"campaign_{campaign_index}_persistent_log_gain"],
            dtype=np.float64,
        ),
        persistent_floor_parameter=np.asarray(
            arrays[f"campaign_{campaign_index}_persistent_floor_parameter"],
            dtype=np.float64,
        ),
        persistent_variance_parameter=np.asarray(
            arrays[f"campaign_{campaign_index}_persistent_variance_parameter"],
            dtype=np.float64,
        ),
        deviation_log_gain=np.asarray(
            arrays[f"campaign_{campaign_index}_deviation_log_gain"],
            dtype=np.float64,
        ),
        deviation_floor_parameter=np.asarray(
            arrays[f"campaign_{campaign_index}_deviation_floor_parameter"],
            dtype=np.float64,
        ),
        deviation_variance_parameter=np.asarray(
            arrays[f"campaign_{campaign_index}_deviation_variance_parameter"],
            dtype=np.float64,
        ),
        gain_power=np.asarray(
            arrays[f"campaign_{campaign_index}_gain_power"],
            dtype=np.float64,
        ),
        additive_noise_power=np.asarray(
            arrays[f"campaign_{campaign_index}_additive_noise_power"],
            dtype=np.float64,
        ),
        residual_variance_power2=np.asarray(
            arrays[f"campaign_{campaign_index}_residual_variance_power2"],
            dtype=np.float64,
        ),
        objective_value=float(arrays[f"campaign_{campaign_index}_objective_value"][0]),
    )


def _normalize_scalar_mapping(
    values: Mapping[str, int | float | None],
) -> dict[str, int | float | None]:
    """Convert scalar mappings into a plain-JSON representation."""

    normalized: dict[str, int | float | None] = {}
    for key, value in values.items():
        if value is None:
            normalized[key] = None
        elif isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            normalized[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            normalized[key] = float(value)
        else:
            raise TypeError(
                "Artifact manifest supports only integer, floating-point, and null "
                f"scalars, but {key!r} received {type(value).__name__}."
            )
    return normalized


def _build_sensor_summary_rows(
    result: TwoLevelCalibrationResult,
) -> list[dict[str, float | int | str]]:
    """Build the human-readable per-sensor training summary rows."""

    rows: list[dict[str, float | int | str]] = []
    for sensor_index, sensor_id in enumerate(result.sensor_ids):
        training_gain_samples_db: list[np.ndarray] = []
        training_noise_samples_db: list[np.ndarray] = []
        training_residual_std_samples_db: list[np.ndarray] = []
        campaigns_seen = 0

        for campaign_state in result.campaign_states:
            if sensor_id not in campaign_state.sensor_ids:
                continue
            campaigns_seen += 1
            local_sensor_index = campaign_state.sensor_ids.index(sensor_id)
            training_gain_samples_db.append(
                power_linear_to_db(campaign_state.gain_power[local_sensor_index])
            )
            training_noise_samples_db.append(
                power_linear_to_db(campaign_state.additive_noise_power[local_sensor_index])
            )
            training_residual_std_samples_db.append(
                power_linear_to_db(
                    np.sqrt(campaign_state.residual_variance_power2[local_sensor_index])
                )
            )

        rows.append(
            {
                "sensor_id": sensor_id,
                "reference_weight": float(result.sensor_reference_weight[sensor_index]),
                "embedding_norm": float(
                    np.linalg.norm(result.sensor_embeddings[sensor_index])
                ),
                "campaigns_seen": campaigns_seen,
                "median_training_gain_db": _nanmedian_from_samples(
                    training_gain_samples_db
                ),
                "median_training_additive_noise_db": _nanmedian_from_samples(
                    training_noise_samples_db
                ),
                "median_training_residual_std_db": _nanmedian_from_samples(
                    training_residual_std_samples_db
                ),
            }
        )
    return rows


def _nanmedian_from_samples(samples: list[np.ndarray]) -> float:
    """Return the median over a list of sample arrays, or NaN when empty."""

    if not samples:
        return float("nan")
    return float(np.median(np.concatenate(samples)))


__all__ = [
    "LoadedCalibrationArtifact",
    "MODEL_SCHEMA_VERSION",
    "SavedCalibrationArtifact",
    "load_two_level_calibration_artifact",
    "save_two_level_calibration_artifact",
    "write_sensor_calibration_summary_csv",
]
