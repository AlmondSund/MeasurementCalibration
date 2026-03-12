"""Shared notebook workflow configuration loaded from editable text files.

This module keeps notebook-facing workflow choices out of the notebooks
themselves. The calibration notebooks should orchestrate data preparation,
training, and visualization, while this adapter owns the user-editable lists
that define:

- globally excluded sensor ids;
- campaigns used for training;
- campaigns used for testing/deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR = Path("config") / "notebook_workflow"
_EXCLUDED_NODES_FILENAME = "excluded_nodes.txt"
_TRAINING_CAMPAIGNS_FILENAME = "training_campaigns.txt"
_TESTING_CAMPAIGNS_FILENAME = "testing_campaigns.txt"


@dataclass(frozen=True)
class NotebookWorkflowConfig:
    """Notebook workflow configuration loaded from one dedicated directory.

    Parameters
    ----------
    config_dir:
        Directory that contains the plain-text list files.
    excluded_sensor_ids:
        Global node exclusions applied across training and deployment.
    training_campaign_labels:
        Campaign labels that should participate in offline training.
    testing_campaign_labels:
        Campaign labels reserved for deployment/testing workflows.
    """

    config_dir: Path
    excluded_sensor_ids: tuple[str, ...]
    training_campaign_labels: tuple[str, ...]
    testing_campaign_labels: tuple[str, ...]

    @property
    def workflow_campaign_labels(self) -> tuple[str, ...]:
        """Return the ordered union of training and testing campaign labels."""

        ordered_labels = dict.fromkeys(
            (*self.training_campaign_labels, *self.testing_campaign_labels)
        )
        return tuple(ordered_labels)


def load_notebook_workflow_config(
    config_dir: Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR,
) -> NotebookWorkflowConfig:
    """Load the shared notebook workflow configuration from text files.

    Purpose
    -------
    The notebooks should not each define their own campaign and sensor
    selection lists. This loader centralizes that editable state into one
    directory so every notebook reads the same workflow contract.

    Parameters
    ----------
    config_dir:
        Directory that contains ``excluded_nodes.txt``,
        ``training_campaigns.txt``, and ``testing_campaigns.txt``.

    Returns
    -------
    NotebookWorkflowConfig
        Parsed workflow configuration with duplicates rejected and train/test
        overlap validated explicitly.

    Side Effects
    ------------
    Reads the configuration files from disk.

    Raises
    ------
    FileNotFoundError
        If any required configuration file is missing.
    ValueError
        If a required list is empty, duplicates appear, or training/testing
        campaigns overlap.
    """

    resolved_config_dir = Path(config_dir)
    excluded_sensor_ids = _read_configured_list(
        resolved_config_dir / _EXCLUDED_NODES_FILENAME,
        list_label="excluded sensor ids",
        allow_empty=True,
    )
    training_campaign_labels = _read_configured_list(
        resolved_config_dir / _TRAINING_CAMPAIGNS_FILENAME,
        list_label="training campaign labels",
        allow_empty=False,
    )
    testing_campaign_labels = _read_configured_list(
        resolved_config_dir / _TESTING_CAMPAIGNS_FILENAME,
        list_label="testing campaign labels",
        allow_empty=False,
    )

    overlapping_campaign_labels = tuple(
        sorted(set(training_campaign_labels).intersection(testing_campaign_labels))
    )
    if overlapping_campaign_labels:
        raise ValueError(
            "Training and testing campaign lists must be disjoint, found overlap: "
            f"{overlapping_campaign_labels}"
        )

    return NotebookWorkflowConfig(
        config_dir=resolved_config_dir,
        excluded_sensor_ids=excluded_sensor_ids,
        training_campaign_labels=training_campaign_labels,
        testing_campaign_labels=testing_campaign_labels,
    )


def build_notebook_workflow_model_label(
    training_campaign_labels: Sequence[str],  # Campaigns used to fit the artifact
    excluded_sensor_ids: Sequence[str],  # Global exclusions encoded into the label
) -> str:
    """Build a stable model label from the configured training workflow.

    Encoding the training campaigns and global exclusions into the saved model
    label keeps notebook artifacts aligned with the configuration directory. If
    the workflow files change, the artifact directory changes with them instead
    of silently reusing an incompatible previous fit.
    """

    normalized_campaign_tokens = tuple(
        _slugify_token(campaign_label)
        for campaign_label in training_campaign_labels
        if str(campaign_label).strip()
    )
    if not normalized_campaign_tokens:
        raise ValueError("training_campaign_labels must contain at least one label")

    model_label = "configured_corpus__" + "__".join(normalized_campaign_tokens)
    normalized_sensor_tokens = tuple(
        sorted(
            _slugify_token(sensor_id)
            for sensor_id in excluded_sensor_ids
            if str(sensor_id).strip()
        )
    )
    if normalized_sensor_tokens:
        model_label += "__exclude__" + "__".join(normalized_sensor_tokens)
    return model_label


def fingerprint_notebook_workflow_config(
    config_dir: Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR,
) -> str:
    """Return a stable hash of the notebook workflow configuration directory.

    The fingerprint captures both the filenames and the exact file contents so
    a saved artifact can be traced back to the precise workflow lists that
    selected its campaigns and global sensor exclusions.
    """

    resolved_config_dir = Path(config_dir)
    required_paths = (
        resolved_config_dir / _EXCLUDED_NODES_FILENAME,
        resolved_config_dir / _TRAINING_CAMPAIGNS_FILENAME,
        resolved_config_dir / _TESTING_CAMPAIGNS_FILENAME,
    )
    digest = hashlib.sha256()
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing notebook workflow config file: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_configured_list(
    path: Path,
    *,
    list_label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Read one comment-friendly list file while rejecting duplicates."""

    if not path.exists():
        raise FileNotFoundError(f"Missing notebook workflow config file: {path}")

    ordered_items: list[str] = []
    duplicate_items: list[str] = []
    seen_items: set[str] = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        normalized_line = raw_line.partition("#")[0].strip()
        if not normalized_line:
            continue
        if normalized_line in seen_items:
            duplicate_items.append(normalized_line)
            continue
        seen_items.add(normalized_line)
        ordered_items.append(normalized_line)

    if duplicate_items:
        raise ValueError(
            f"Duplicate entries found in {list_label} file {path}: "
            f"{tuple(sorted(set(duplicate_items)))}"
        )
    if not ordered_items and not allow_empty:
        raise ValueError(
            f"Notebook workflow {list_label} file must contain at least one entry: "
            f"{path}"
        )
    return tuple(ordered_items)


def _slugify_token(raw_value: str) -> str:
    """Convert a user-facing label into a filesystem-stable token."""

    normalized_value = re.sub(r"[^a-z0-9]+", "-", raw_value.strip().lower()).strip("-")
    if not normalized_value:
        raise ValueError(f"Could not build a stable token from value: {raw_value!r}")
    return normalized_value


__all__ = [
    "DEFAULT_NOTEBOOK_WORKFLOW_CONFIG_DIR",
    "NotebookWorkflowConfig",
    "build_notebook_workflow_model_label",
    "fingerprint_notebook_workflow_config",
    "load_notebook_workflow_config",
]
