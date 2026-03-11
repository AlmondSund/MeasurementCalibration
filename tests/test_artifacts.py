"""Tests for two-level calibration artifact serialization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from measurement_calibration.artifacts import (
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
    assert loaded.manifest["artifact_type"] == "configuration_conditional_calibration_model"
    assert loaded.manifest["sensor_ids"] == list(result.sensor_ids)
    assert loaded.manifest["training_summary"]["n_campaigns"] == len(result.campaign_states)
    assert loaded.manifest["extra_summary"]["fit_duration_s"] == 1.25
    assert loaded.manifest["basis_config"]["n_gain_basis"] == fixture.basis_config.n_gain_basis
    assert loaded.manifest["fit_config"]["n_outer_iterations"] == fixture.fit_config.n_outer_iterations

    assert loaded.result.sensor_ids == result.sensor_ids
    assert np.allclose(loaded.result.sensor_reference_weight, result.sensor_reference_weight)
    assert np.allclose(loaded.result.configuration_feature_mean, result.configuration_feature_mean)
    assert np.allclose(loaded.result.configuration_feature_scale, result.configuration_feature_scale)
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
