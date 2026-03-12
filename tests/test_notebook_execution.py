"""End-to-end notebook execution regressions for checked-in workflows."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pytest

from scripts.run_repo_checks import notebook_execution_supported


def test_deployment_notebook_executes_headlessly(
    tmp_path: Path,
) -> None:
    """The checked-in deployment notebook should execute end-to-end.

    The executed notebook is written to a temporary output path so the tracked
    notebook file remains untouched. The test is intentionally limited to the
    deployment notebook because it reuses a checked-in artifact and avoids the
    expensive training path.
    """

    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "examples" / "deployment.ipynb"
    if not notebook_path.exists():
        pytest.skip("deployment notebook is not present in this checkout")
    if not (repo_root / "models" / "production" / "manifest.json").exists():
        pytest.skip("deployment notebook execution requires the checked-in artifact")

    jupyter_path = repo_root / ".venv" / "bin" / "jupyter"
    if not jupyter_path.exists():
        pytest.skip("deployment notebook execution requires .venv/bin/jupyter")
    notebook_supported, reason = notebook_execution_supported()
    if not notebook_supported:
        pytest.skip(f"notebook execution is unavailable in this environment: {reason}")

    output_path = tmp_path / "deployment-executed.ipynb"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(
        [
            str(jupyter_path),
            "execute",
            str(notebook_path),
            "--timeout=1200",
            f"--output={output_path}",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    assert output_path.exists()
