"""Notebook contract and smoke tests without retraining the model.

The project notebooks are user-facing workflow entry points. These tests keep
their API usage aligned with the library while avoiding the expensive training
path that would make local verification fragile.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest


def _repo_root() -> Path:
    """Return the repository root for notebook-relative paths."""

    return Path(__file__).resolve().parents[1]


def _load_notebook_cells(notebook_path: Path) -> list[dict[str, Any]]:
    """Load the raw cell list from one Jupyter notebook."""

    notebook_payload = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    return list(notebook_payload["cells"])


def _code_cell_source(
    notebook_path: Path,
    cell_index: int,
) -> str:
    """Return the concatenated source code of one notebook cell."""

    cells = _load_notebook_cells(notebook_path)
    cell = cells[cell_index]
    if cell.get("cell_type") != "code":
        raise ValueError(f"Cell {cell_index} is not a code cell in {notebook_path}")
    return "".join(cell.get("source", []))


def _find_code_cell_index(
    notebook_path: Path,
    required_snippet: str,
) -> int:
    """Return the first code-cell index that contains ``required_snippet``."""

    for cell_index, cell in enumerate(_load_notebook_cells(notebook_path)):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if required_snippet in source:
            return cell_index
    raise ValueError(
        f"Could not find a code cell containing {required_snippet!r} in {notebook_path}"
    )


def _execute_code_cells(
    notebook_path: Path,
    cell_indices: Iterable[int],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Execute selected notebook code cells in a shared namespace.

    The execution is intentionally partial: it reuses the notebook's actual
    source code for imports and user-facing workflow cells, but skips any cell
    that would retrain the model.
    """

    monkeypatch.chdir(_repo_root())
    monkeypatch.setenv("MPLBACKEND", "Agg")

    module_name = "__notebook_smoke__"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    module.__dict__.update(
        {
            "__name__": module_name,
            "__file__": str(notebook_path),
            "IS_KAGGLE": False,
            "USE_KAGGLE_GPU_FOR_TRAINING": False,
            "KAGGLE_CUPY_PACKAGE": "cupy-cuda12x",
        }
    )
    sys.modules[module_name] = module
    try:
        cells = _load_notebook_cells(notebook_path)
        for cell_index in cell_indices:
            cell = cells[cell_index]
            if cell.get("cell_type") != "code":
                raise ValueError(
                    f"Requested cell {cell_index} is not executable in {notebook_path}"
                )
            exec("".join(cell.get("source", [])), module.__dict__)
        return module.__dict__
    finally:
        sys.modules.pop(module_name, None)


def test_sensor_calibration_notebook_tracks_fit_result_contract() -> None:
    """The training notebook should keep its stored fit-result variable alive."""

    notebook_path = _repo_root() / "notebooks" / "sensor_calibration.ipynb"
    training_cell_source = _code_cell_source(
        notebook_path,
        _find_code_cell_index(
            notebook_path,
            "fit_result = fit_and_save_calibration_corpus_model(",
        ),
    )
    inspection_cell_source = _code_cell_source(
        notebook_path,
        _find_code_cell_index(
            notebook_path,
            "fit_result.result.objective_history",
        ),
    )

    assert "fit_result = fit_and_save_calibration_corpus_model(" in training_cell_source
    assert (
        "parameters_filename=DEFAULT_PRODUCTION_PARAMETERS_FILENAME"
        in training_cell_source
    )
    assert "fit_result.result.objective_history" in inspection_cell_source


def test_sensor_calibration_notebook_selected_cells_execute_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sensor notebook setup and corpus-preparation cells should execute cleanly.

    This keeps the user-facing training notebook aligned with the library
    through real execution of its import, workflow-resolution, and corpus
    preparation path, while still avoiding the expensive fit cell.
    """

    notebook_path = _repo_root() / "notebooks" / "sensor_calibration.ipynb"
    namespace = _execute_code_cells(
        notebook_path=notebook_path,
        cell_indices=(
            _find_code_cell_index(
                notebook_path,
                'NOTEBOOK_WORKFLOW_CONFIG_DIRNAME = "config/notebook_workflow"',
            ),
            _find_code_cell_index(
                notebook_path,
                "from measurement_calibration import (",
            ),
            _find_code_cell_index(
                notebook_path,
                "preparation = prepare_calibration_corpus(",
            ),
        ),
        monkeypatch=monkeypatch,
    )

    preparation = namespace["preparation"]
    training_campaign_labels = namespace["TRAINING_CAMPAIGN_LABELS"]

    assert len(preparation.prepared_campaigns) == len(training_campaign_labels)
    assert preparation.campaigns_root.exists()
    assert preparation.corpus.sensor_ids
    for prepared_campaign in preparation.prepared_campaigns:
        assert prepared_campaign.campaign.sensor_ids
        assert len(prepared_campaign.campaign.sensor_ids) >= 2
        assert set(prepared_campaign.alignment_pruned_sensor_ids).isdisjoint(
            prepared_campaign.campaign.sensor_ids
        )


def test_deployment_notebook_selected_cells_execute_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment notebook should run its non-training workflow cells."""

    notebook_path = _repo_root() / "notebooks" / "examples" / "deployment.ipynb"
    production_manifest_path = _repo_root() / "models" / "production" / "manifest.json"
    if not production_manifest_path.exists():
        pytest.skip("deployment notebook smoke test requires the checked-in artifact")

    namespace = _execute_code_cells(
        notebook_path=notebook_path,
        cell_indices=(
            _find_code_cell_index(
                notebook_path,
                'NOTEBOOK_WORKFLOW_CONFIG_DIRNAME = "config/notebook_workflow"',
            ),
            _find_code_cell_index(
                notebook_path,
                "from measurement_calibration import (",
            ),
            _find_code_cell_index(
                notebook_path,
                "artifact = load_two_level_calibration_artifact(PRODUCTION_MODEL_DIR)",
            ),
            _find_code_cell_index(
                notebook_path,
                "deployment = calibrate_sensor_observations(",
            ),
        ),
        monkeypatch=monkeypatch,
    )

    artifact = namespace["artifact"]
    curves = namespace["curves"]
    deployment = namespace["deployment"]
    raw_power = namespace["raw_power"]

    assert artifact.manifest_path.exists()
    assert artifact.parameters_path.exists()
    assert curves.trust_diagnostics.frequency_extrapolation_detected is False
    assert curves.trust_diagnostics.configuration_out_of_distribution is False
    assert curves.trust_diagnostics.overall_out_of_distribution is False
    assert deployment.uncertainty_scope == "observation_noise_only"
    assert deployment.calibrated_power.shape == raw_power.shape
