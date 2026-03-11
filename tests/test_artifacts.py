"""Tests for two-level calibration artifact serialization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from measurement_calibration.artifacts import (
    DEFAULT_PRODUCTION_PARAMETERS_FILENAME,
    archive_artifact_directory,
    load_two_level_calibration_artifact,
    save_two_level_calibration_artifact,
)
from measurement_calibration.spectral_calibration import fit_two_level_calibration

from tests._synthetic_two_level import build_synthetic_two_level_fixture


def test_save_and_load_two_level_calibration_artifact_round_trip(
    tmp_path: Path,
) -> None:
    """Saved artifacts should reload into the same fitted two-level result."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )

    artifact = save_two_level_calibration_artifact(
        output_dir=tmp_path / "artifact",
        result=result,
        extra_summary={"fit_duration_s": 1.25},
    )
    loaded = load_two_level_calibration_artifact(artifact.output_dir)

    assert loaded.manifest["schema_version"] == 2
    assert (
        loaded.manifest["artifact_type"]
        == "configuration_conditional_calibration_model"
    )
    assert loaded.manifest["sensor_ids"] == list(result.sensor_ids)
    assert loaded.manifest["training_summary"]["n_campaigns"] == len(
        result.campaign_states
    )
    assert loaded.manifest["training_summary"]["objective_selected"] == pytest.approx(
        result.fit_diagnostics.selected_objective_value
    )
    assert loaded.manifest["training_summary"]["effective_variance_floor_power2"] == (
        pytest.approx(result.effective_variance_floor_power2)
    )
    assert loaded.manifest["fit_diagnostics"]["selected_outer_iteration"] == (
        result.fit_diagnostics.selected_outer_iteration
    )
    assert "provenance" in loaded.manifest
    assert "python_version" in loaded.manifest["provenance"]
    assert "numpy_version" in loaded.manifest["provenance"]
    assert "scipy_version" in loaded.manifest["provenance"]
    assert "corpus_fingerprint" in loaded.manifest["provenance"]
    assert loaded.manifest["extra_summary"]["fit_duration_s"] == 1.25
    assert (
        loaded.manifest["basis_config"]["n_gain_basis"]
        == fixture.basis_config.n_gain_basis
    )
    assert (
        loaded.manifest["fit_config"]["n_outer_iterations"]
        == fixture.fit_config.n_outer_iterations
    )

    assert loaded.result.sensor_ids == result.sensor_ids
    assert np.allclose(
        loaded.result.sensor_reference_weight, result.sensor_reference_weight
    )
    assert np.allclose(
        loaded.result.configuration_feature_mean, result.configuration_feature_mean
    )
    assert np.allclose(
        loaded.result.configuration_feature_scale, result.configuration_feature_scale
    )
    assert loaded.result.configuration_feature_min is not None
    assert loaded.result.configuration_feature_max is not None
    assert result.configuration_feature_min is not None
    assert result.configuration_feature_max is not None
    assert loaded.result.effective_variance_floor_power2 is not None
    assert result.effective_variance_floor_power2 is not None
    assert np.allclose(
        loaded.result.configuration_feature_min, result.configuration_feature_min
    )
    assert np.allclose(
        loaded.result.configuration_feature_max, result.configuration_feature_max
    )
    assert loaded.result.effective_variance_floor_power2 == pytest.approx(
        result.effective_variance_floor_power2
    )
    assert np.allclose(loaded.result.sensor_embeddings, result.sensor_embeddings)
    assert np.allclose(
        loaded.result.configuration_encoder_weight,
        result.configuration_encoder_weight,
    )
    assert np.allclose(
        loaded.result.configuration_encoder_bias,
        result.configuration_encoder_bias,
    )
    assert np.allclose(loaded.result.gain_head_weight, result.gain_head_weight)
    assert np.allclose(loaded.result.gain_head_bias, result.gain_head_bias)
    assert np.allclose(loaded.result.floor_head_weight, result.floor_head_weight)
    assert np.allclose(loaded.result.floor_head_bias, result.floor_head_bias)
    assert np.allclose(loaded.result.variance_head_weight, result.variance_head_weight)
    assert np.allclose(loaded.result.variance_head_bias, result.variance_head_bias)
    assert np.allclose(loaded.result.objective_history, result.objective_history)
    assert loaded.result.fit_diagnostics == result.fit_diagnostics
    assert len(loaded.result.campaign_states) == len(result.campaign_states)

    for loaded_state, original_state in zip(
        loaded.result.campaign_states,
        result.campaign_states,
        strict=True,
    ):
        assert loaded_state.campaign_label == original_state.campaign_label
        assert loaded_state.sensor_ids == original_state.sensor_ids
        assert np.allclose(loaded_state.frequency_hz, original_state.frequency_hz)
        assert np.allclose(
            loaded_state.latent_spectra_power,
            original_state.latent_spectra_power,
        )
        assert np.allclose(
            loaded_state.deviation_log_gain,
            original_state.deviation_log_gain,
        )
        assert np.allclose(loaded_state.gain_power, original_state.gain_power)
        assert np.allclose(
            loaded_state.additive_noise_power,
            original_state.additive_noise_power,
        )
        assert np.allclose(
            loaded_state.residual_variance_power2,
            original_state.residual_variance_power2,
        )

    with loaded.sensor_summary_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == len(result.sensor_ids)
    assert rows[0]["sensor_id"] == "Node1"
    assert "reference_weight" in rows[0]
    assert "embedding_norm" in rows[0]
    assert "campaigns_seen" in rows[0]


def test_save_and_load_two_level_calibration_artifact_with_custom_npz_name(
    tmp_path: Path,
) -> None:
    """Artifacts should support an explicit notebook-facing parameter filename."""

    fixture = build_synthetic_two_level_fixture()
    result = fit_two_level_calibration(
        corpus=fixture.corpus,
        basis_config=fixture.basis_config,
        model_config=fixture.model_config,
        fit_config=fixture.fit_config,
    )

    artifact = save_two_level_calibration_artifact(
        output_dir=tmp_path / "artifact",
        result=result,
        parameters_filename=DEFAULT_PRODUCTION_PARAMETERS_FILENAME,
    )
    loaded = load_two_level_calibration_artifact(artifact.output_dir)

    assert artifact.parameters_path.name == DEFAULT_PRODUCTION_PARAMETERS_FILENAME
    assert artifact.parameters_path.exists()
    assert loaded.manifest["parameters_file"] == DEFAULT_PRODUCTION_PARAMETERS_FILENAME
    assert loaded.parameters_path.name == DEFAULT_PRODUCTION_PARAMETERS_FILENAME


def test_archive_artifact_directory_moves_existing_bundle_into_archive_root(
    tmp_path: Path,
) -> None:
    """Archiving should move the live bundle into a unique archive directory."""

    production_dir = tmp_path / "models" / "production"
    production_dir.mkdir(parents=True)
    (production_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (production_dir / "model.npz").write_bytes(b"npz-bytes")

    archive_dir = archive_artifact_directory(
        output_dir=production_dir,
        archive_root=tmp_path / "models" / "archive",
        archive_label="production",
    )

    assert archive_dir is not None
    assert archive_dir.parent == tmp_path / "models" / "archive"
    assert archive_dir.name.startswith("production__")
    assert not production_dir.exists()
    assert (archive_dir / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert (archive_dir / "model.npz").read_bytes() == b"npz-bytes"


def test_archive_artifact_directory_returns_none_for_missing_or_empty_bundle(
    tmp_path: Path,
) -> None:
    """Archiving should ignore production locations without a saved artifact."""

    archive_root = tmp_path / "models" / "archive"

    assert (
        archive_artifact_directory(
            output_dir=tmp_path / "models" / "missing-production",
            archive_root=archive_root,
        )
        is None
    )

    empty_production_dir = tmp_path / "models" / "production"
    empty_production_dir.mkdir(parents=True)

    assert (
        archive_artifact_directory(
            output_dir=empty_production_dir,
            archive_root=archive_root,
        )
        is None
    )
    assert empty_production_dir.exists()
    assert not any(archive_root.glob("*"))
