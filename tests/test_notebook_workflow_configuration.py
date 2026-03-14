"""Tests for the shared notebook workflow configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from measurement_calibration.notebook_workflow_configuration import (
    DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR,
    build_notebook_workflow_model_label,
    fingerprint_notebook_workflow_config,
    load_notebook_workflow_config,
)


def test_load_notebook_workflow_config_parses_comment_friendly_lists(
    tmp_path: Path,
) -> None:
    """The loader should ignore comments and preserve list ordering."""

    config_dir = tmp_path / "notebook_workflow"
    _write_config_file(
        config_dir / "excluded_nodes.txt",
        """
        # Global exclusions
        Node9

        Node10  # Broken node
        """,
    )
    _write_config_file(
        config_dir / "training_campaigns.txt",
        """
        # Training corpus
        LNA16_VGA0
        MeasurementCalibration
        """,
    )
    _write_config_file(
        config_dir / "testing_campaigns.txt",
        """
        # Held-out deployment campaign
        test-calibration
        """,
    )
    _write_config_file(
        config_dir / "excluded_measurements.txt",
        """
        # Drop one startup-corrupted record from every retained sensor.
        1
        """,
    )

    config = load_notebook_workflow_config(config_dir)

    assert config.excluded_sensor_ids == ("Node9", "Node10")
    assert config.excluded_leading_measurements_per_sensor == 1
    assert config.training_campaign_labels == (
        "LNA16_VGA0",
        "MeasurementCalibration",
    )
    assert config.testing_campaign_labels == ("test-calibration",)
    assert config.workflow_campaign_labels == (
        "LNA16_VGA0",
        "MeasurementCalibration",
        "test-calibration",
    )


def test_load_notebook_workflow_config_rejects_train_test_overlap(
    tmp_path: Path,
) -> None:
    """Training and testing campaign lists should remain disjoint."""

    config_dir = tmp_path / "notebook_workflow"
    _write_config_file(config_dir / "excluded_nodes.txt", "")
    _write_config_file(
        config_dir / "training_campaigns.txt",
        "MeasurementCalibration\n",
    )
    _write_config_file(
        config_dir / "testing_campaigns.txt",
        "MeasurementCalibration\n",
    )
    _write_config_file(config_dir / "excluded_measurements.txt", "")

    with pytest.raises(ValueError, match="overlap"):
        load_notebook_workflow_config(config_dir)


def test_load_notebook_workflow_config_rejects_multiple_measurement_counts(
    tmp_path: Path,
) -> None:
    """The scalar exclusion file should reject more than one configured value."""

    config_dir = tmp_path / "notebook_workflow"
    _write_config_file(config_dir / "excluded_nodes.txt", "")
    _write_config_file(config_dir / "training_campaigns.txt", "campaign-a\n")
    _write_config_file(config_dir / "testing_campaigns.txt", "campaign-b\n")
    _write_config_file(
        config_dir / "excluded_measurements.txt",
        """
        1
        2
        """,
    )

    with pytest.raises(ValueError, match="at most one non-comment integer entry"):
        load_notebook_workflow_config(config_dir)


def test_load_notebook_workflow_config_rejects_negative_measurement_count(
    tmp_path: Path,
) -> None:
    """The global exclusion count must be a non-negative integer."""

    config_dir = tmp_path / "notebook_workflow"
    _write_config_file(config_dir / "excluded_nodes.txt", "")
    _write_config_file(config_dir / "training_campaigns.txt", "campaign-a\n")
    _write_config_file(config_dir / "testing_campaigns.txt", "campaign-b\n")
    _write_config_file(config_dir / "excluded_measurements.txt", "-1\n")

    with pytest.raises(ValueError, match="non-negative integer"):
        load_notebook_workflow_config(config_dir)


def test_build_notebook_workflow_model_label_tracks_configured_workflow() -> None:
    """The artifact label should encode both campaigns and exclusions."""

    model_label = build_notebook_workflow_model_label(
        training_campaign_labels=("LNA16_VGA0", "MeasurementCalibration"),
        excluded_sensor_ids=("Node10", "Node9"),
    )

    assert model_label == (
        "configured_corpus__lna16-vga0__measurementcalibration__exclude__node10__node9"
    )


def test_fingerprint_notebook_workflow_config_changes_when_files_change(
    tmp_path: Path,
) -> None:
    """The workflow fingerprint should track the exact config file contents."""

    config_dir = tmp_path / "notebook_workflow"
    _write_config_file(config_dir / "excluded_nodes.txt", "Node9\n")
    _write_config_file(config_dir / "excluded_measurements.txt", "")
    _write_config_file(config_dir / "training_campaigns.txt", "campaign-a\n")
    _write_config_file(config_dir / "testing_campaigns.txt", "campaign-b\n")

    first_fingerprint = fingerprint_notebook_workflow_config(config_dir)
    _write_config_file(config_dir / "testing_campaigns.txt", "campaign-c\n")
    second_fingerprint = fingerprint_notebook_workflow_config(config_dir)

    assert len(first_fingerprint) == 64
    assert len(second_fingerprint) == 64
    assert first_fingerprint != second_fingerprint


def test_repository_notebook_workflow_config_matches_expected_campaign_split() -> None:
    """The checked-in workflow config should match the intended corpus split."""

    repo_root = Path(__file__).resolve().parents[1]
    config = load_notebook_workflow_config(
        repo_root / DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR
    )

    assert config.excluded_sensor_ids == ("Node7", "Node8", "Node9", "Node10")
    assert config.excluded_leading_measurements_per_sensor == 2
    assert config.training_campaign_labels == (
        "fm_ref_fullband_01",
        "fm_ref_fullband_02",
        "fm_sweep_rbw_003k",
        "fm_sweep_rbw_100k",
        "MeasurementCalibration",
        "LNA_sweep_01",
        "LNA_sweep_02",
        "LNA_sweep_03",
        "LNA_sweep_04",
        "LNA_sweep_05",
        "LNA_sweep_06",
        "VGA_sweep_01",
        "VGA_sweep_02",
        "VGA_sweep_03",
        "VGA_sweep_04",
        "VGA_sweep_05",
        "VGA_sweep_06",
        "VGA_sweep_07",
    )
    assert config.testing_campaign_labels == ("test-calibration",)


def _write_config_file(path: Path, contents: str) -> None:
    """Write one temporary notebook workflow config file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.strip() + "\n", encoding="utf-8")
