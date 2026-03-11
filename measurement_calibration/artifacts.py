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
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import scipy

from .spectral_calibration import (
    CampaignCalibrationState,
    CampaignConfiguration,
    FitConvergenceDiagnostics,
    FrequencyBasisConfig,
    PersistentModelConfig,
    TwoLevelCalibrationResult,
    TwoLevelFitConfig,
    power_linear_to_db,
)


MODEL_SCHEMA_VERSION = 2
DEFAULT_ARTIFACT_PARAMETERS_FILENAME = "calibration_parameters.npz"
DEFAULT_PRODUCTION_ARTIFACT_DIR = Path("models") / "production"
DEFAULT_ARCHIVED_ARTIFACTS_DIR = Path("models") / "archive"
DEFAULT_PRODUCTION_PARAMETERS_FILENAME = "model.npz"


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
    parameters_filename: str = DEFAULT_ARTIFACT_PARAMETERS_FILENAME,
) -> SavedCalibrationArtifact:
    """Persist a fitted two-level calibration model to disk.

    Purpose
    -------
    The saved bundle is the deployment boundary for notebook and application
    workflows. Most callers can rely on the default parameter filename, while
    notebook-facing production workflows may override it with a stable
    deployment name such as ``model.npz``.

    Parameters
    ----------
    output_dir:
        Destination directory for the saved artifact bundle.
    result:
        Fitted calibration model to serialize.
    extra_summary:
        Optional scalar metadata merged into ``manifest.json``.
    parameters_filename:
        Simple ``.npz`` filename used for the serialized parameter archive
        inside ``output_dir``.

    Side Effects
    ------------
    Creates ``output_dir`` when needed and writes:

    - ``manifest.json`` with configuration, provenance, and high-level summary;
    - one ``.npz`` parameter archive with the trainable arrays and campaign
      states;
    - ``sensor_summary.csv`` with compact per-sensor training statistics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters_path = output_dir / _validate_parameters_filename(parameters_filename)
    manifest_path = output_dir / "manifest.json"
    sensor_summary_path = output_dir / "sensor_summary.csv"

    np.savez_compressed(
        parameters_path,
        **_build_parameter_archive_payload(result),
    )
    write_sensor_calibration_summary_csv(sensor_summary_path, result)

    manifest = _build_artifact_manifest(
        result=result,
        parameters_path=parameters_path,
        sensor_summary_path=sensor_summary_path,
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
            configuration_feature_min=(
                None
                if "configuration_feature_min" not in arrays
                else np.asarray(arrays["configuration_feature_min"], dtype=np.float64)
            ),
            configuration_feature_max=(
                None
                if "configuration_feature_max" not in arrays
                else np.asarray(arrays["configuration_feature_max"], dtype=np.float64)
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
            variance_head_bias=np.asarray(
                arrays["variance_head_bias"], dtype=np.float64
            ),
            campaign_states=campaign_states,
            objective_history=np.asarray(arrays["objective_history"], dtype=np.float64),
            effective_variance_floor_power2=(
                None
                if "effective_variance_floor_power2" not in arrays
                else float(arrays["effective_variance_floor_power2"][0])
            ),
            fit_diagnostics=_load_fit_diagnostics(
                manifest=manifest,
                objective_history=np.asarray(
                    arrays["objective_history"], dtype=np.float64
                ),
            ),
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


def archive_artifact_directory(
    output_dir: Path,  # Live artifact directory to archive
    archive_root: Path = DEFAULT_ARCHIVED_ARTIFACTS_DIR,
    archive_label: str | None = None,  # Prefix used for the archived directory
) -> Path | None:
    """Move an existing artifact directory into a timestamped archive location.

    Purpose
    -------
    Notebook workflows keep a single stable production path for deployment.
    Before a new production artifact is written, the previous bundle can be
    moved under ``archive_root`` so the latest model remains easy to load while
    replaced fits stay auditable.

    Parameters
    ----------
    output_dir:
        Existing artifact directory that should stop being the live location.
    archive_root:
        Parent directory that stores timestamped archived bundles.
    archive_label:
        Optional prefix for the archived directory name. When omitted, the stem
        uses ``output_dir.name``.

    Returns
    -------
    Path | None
        New archived directory path, or ``None`` when ``output_dir`` does not
        exist yet or does not contain a saved bundle.

    Side Effects
    ------------
    Creates ``archive_root`` when needed and renames ``output_dir`` into a
    unique timestamped directory beneath it.

    Raises
    ------
    NotADirectoryError
        If ``output_dir`` exists but is not a directory.
    ValueError
        If ``archive_label`` cannot be normalized into a filesystem-safe token.
    """

    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.exists():
        return None
    if not resolved_output_dir.is_dir():
        raise NotADirectoryError(
            f"Artifact output_dir must be a directory: {resolved_output_dir}"
        )
    if not any(resolved_output_dir.iterdir()):
        return None

    resolved_archive_root = Path(archive_root)
    resolved_archive_root.mkdir(parents=True, exist_ok=True)
    label_token = _normalize_archive_label(
        resolved_output_dir.name if archive_label is None else archive_label
    )
    archive_dir = _build_unique_archive_dir(
        archive_root=resolved_archive_root,
        archive_label=label_token,
    )
    resolved_output_dir.rename(archive_dir)
    return archive_dir


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


def _validate_parameters_filename(parameters_filename: str) -> str:
    """Validate the artifact parameter archive filename.

    The artifact format stores a single NPZ file inside ``output_dir``. This
    helper keeps that contract explicit by rejecting nested paths and
    non-``.npz`` filenames early.
    """

    normalized_filename = str(parameters_filename).strip()
    if not normalized_filename:
        raise ValueError("parameters_filename must be a non-empty filename")

    filename_path = Path(normalized_filename)
    if filename_path.name != normalized_filename:
        raise ValueError(
            "parameters_filename must be a simple filename without directory "
            f"components, got {parameters_filename!r}"
        )
    if filename_path.suffix != ".npz":
        raise ValueError(
            f"parameters_filename must end with '.npz', got {parameters_filename!r}"
        )
    return normalized_filename


def _normalize_archive_label(archive_label: str) -> str:
    """Normalize one archive directory label into a filesystem-safe token."""

    normalized_label = "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(archive_label).strip()
    ).strip("-")
    normalized_label = "-".join(token for token in normalized_label.split("-") if token)
    if not normalized_label:
        raise ValueError(
            f"archive_label must contain at least one alphanumeric token: {archive_label!r}"
        )
    return normalized_label


def _build_unique_archive_dir(
    archive_root: Path,
    archive_label: str,
) -> Path:
    """Build a unique timestamped archive directory path."""

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = archive_root / f"{archive_label}__{timestamp_utc}"
    suffix_index = 1
    while candidate.exists():
        candidate = (
            archive_root / f"{archive_label}__{timestamp_utc}_{suffix_index:02d}"
        )
        suffix_index += 1
    return candidate


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
    if result.effective_variance_floor_power2 is not None:
        payload["effective_variance_floor_power2"] = np.asarray(
            [result.effective_variance_floor_power2],
            dtype=np.float64,
        )
    if result.configuration_feature_min is not None:
        payload["configuration_feature_min"] = np.asarray(
            result.configuration_feature_min, dtype=np.float64
        )
    if result.configuration_feature_max is not None:
        payload["configuration_feature_max"] = np.asarray(
            result.configuration_feature_max, dtype=np.float64
        )
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
        payload[f"campaign_{campaign_index}_persistent_floor_parameter"] = np.asarray(
            campaign_state.persistent_floor_parameter, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_persistent_variance_parameter"] = (
            np.asarray(campaign_state.persistent_variance_parameter, dtype=np.float64)
        )
        payload[f"campaign_{campaign_index}_deviation_log_gain"] = np.asarray(
            campaign_state.deviation_log_gain, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_deviation_floor_parameter"] = np.asarray(
            campaign_state.deviation_floor_parameter, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_deviation_variance_parameter"] = np.asarray(
            campaign_state.deviation_variance_parameter, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_gain_power"] = np.asarray(
            campaign_state.gain_power, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_additive_noise_power"] = np.asarray(
            campaign_state.additive_noise_power, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_residual_variance_power2"] = np.asarray(
            campaign_state.residual_variance_power2, dtype=np.float64
        )
        payload[f"campaign_{campaign_index}_objective_value"] = np.asarray(
            [campaign_state.objective_value], dtype=np.float64
        )
    return payload


def _build_artifact_manifest(
    result: TwoLevelCalibrationResult,
    parameters_path: Path,
    sensor_summary_path: Path,
    extra_summary: dict[str, int | float] | None,
) -> dict[str, Any]:
    """Build the JSON manifest dictionary for one saved artifact."""

    manifest: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "configuration_conditional_calibration_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters_file": parameters_path.name,
        "sensor_summary_file": sensor_summary_path.name,
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
        "fit_diagnostics": asdict(result.fit_diagnostics),
        "provenance": _build_provenance_manifest(
            result=result,
            parameters_path=parameters_path,
            sensor_summary_path=sensor_summary_path,
        ),
        "campaigns": [
            _campaign_manifest_entry(campaign_state)
            for campaign_state in result.campaign_states
        ],
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
        "objective_selected": float(result.fit_diagnostics.selected_objective_value),
        "selected_outer_iteration": int(
            result.fit_diagnostics.selected_outer_iteration
        ),
        "terminated_early": int(result.fit_diagnostics.terminated_early),
        "effective_variance_floor_power2": float(
            result.effective_variance_floor_power2
            if result.effective_variance_floor_power2 is not None
            else result.fit_config.sigma_min
        ),
        "mean_campaign_objective": float(
            np.mean(
                [
                    campaign_state.objective_value
                    for campaign_state in result.campaign_states
                ]
            )
        ),
    }


def _build_provenance_manifest(
    result: TwoLevelCalibrationResult,
    parameters_path: Path,
    sensor_summary_path: Path,
) -> dict[str, Any]:
    """Build reproducibility-oriented provenance metadata for the manifest."""

    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "git_commit": _safe_git_command("rev-parse", "HEAD"),
        "git_branch": _safe_git_command("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": _safe_git_dirty(),
        "corpus_fingerprint": _corpus_fingerprint(result),
        "parameters_sha256": _sha256_file(parameters_path),
        "sensor_summary_sha256": _sha256_file(sensor_summary_path),
    }


def _safe_git_command(*arguments: str) -> str | None:
    """Return one git command output when available, otherwise ``None``."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _safe_git_dirty() -> bool | None:
    """Return whether the git worktree is dirty when git metadata is available."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _corpus_fingerprint(
    result: TwoLevelCalibrationResult,
) -> str:
    """Build a stable fingerprint of the retained training-corpus support."""

    fingerprint_payload = {
        "sensor_ids": list(result.sensor_ids),
        "campaigns": [
            {
                "campaign_label": campaign_state.campaign_label,
                "sensor_ids": list(campaign_state.sensor_ids),
                "n_acquisitions": int(campaign_state.latent_spectra_power.shape[0]),
                "frequency_hz": [
                    float(value) for value in np.asarray(campaign_state.frequency_hz)
                ],
                "configuration": asdict(campaign_state.configuration),
            }
            for campaign_state in result.campaign_states
        ],
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file on disk."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_fit_diagnostics(
    manifest: Mapping[str, Any],
    objective_history: np.ndarray,
) -> FitConvergenceDiagnostics:
    """Load fit diagnostics from the manifest or derive a backward-compatible default."""

    fit_diagnostics_payload = manifest.get("fit_diagnostics")
    if isinstance(fit_diagnostics_payload, Mapping):
        return FitConvergenceDiagnostics(
            selected_outer_iteration=int(
                fit_diagnostics_payload["selected_outer_iteration"]
            ),
            n_completed_outer_iterations=int(
                fit_diagnostics_payload["n_completed_outer_iterations"]
            ),
            selected_objective_value=float(
                fit_diagnostics_payload["selected_objective_value"]
            ),
            final_objective_value=float(
                fit_diagnostics_payload["final_objective_value"]
            ),
            terminated_early=bool(fit_diagnostics_payload["terminated_early"]),
            termination_reason=str(fit_diagnostics_payload["termination_reason"]),
            selected_from_best_iterate=bool(
                fit_diagnostics_payload["selected_from_best_iterate"]
            ),
        )

    if objective_history.size == 0:
        return FitConvergenceDiagnostics(
            selected_outer_iteration=0,
            n_completed_outer_iterations=0,
            selected_objective_value=float("nan"),
            final_objective_value=float("nan"),
            terminated_early=False,
            termination_reason="loaded_without_diagnostics",
            selected_from_best_iterate=False,
        )

    best_outer_iteration = int(np.argmin(objective_history))
    return FitConvergenceDiagnostics(
        selected_outer_iteration=best_outer_iteration,
        n_completed_outer_iterations=int(objective_history.size),
        selected_objective_value=float(objective_history[best_outer_iteration]),
        final_objective_value=float(objective_history[-1]),
        terminated_early=False,
        termination_reason="loaded_without_diagnostics",
        selected_from_best_iterate=best_outer_iteration != objective_history.size - 1,
    )


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
                power_linear_to_db(
                    campaign_state.additive_noise_power[local_sensor_index]
                )
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
    "DEFAULT_ARCHIVED_ARTIFACTS_DIR",
    "DEFAULT_ARTIFACT_PARAMETERS_FILENAME",
    "DEFAULT_PRODUCTION_ARTIFACT_DIR",
    "DEFAULT_PRODUCTION_PARAMETERS_FILENAME",
    "LoadedCalibrationArtifact",
    "MODEL_SCHEMA_VERSION",
    "SavedCalibrationArtifact",
    "archive_artifact_directory",
    "load_two_level_calibration_artifact",
    "save_two_level_calibration_artifact",
    "write_sensor_calibration_summary_csv",
]
