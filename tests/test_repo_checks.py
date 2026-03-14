"""Tests for the repository-level verification entry point."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.run_repo_checks import build_verification_steps


def test_build_verification_steps_includes_repo_tests_and_notebook_execution() -> None:
    """The repo check entry point should cover tests and notebook execution."""

    repo_root = Path(__file__).resolve().parents[1]
    steps = build_verification_steps(repo_root, include_notebook_execution=True)

    step_by_label = {step.label: step for step in steps}
    assert tuple(step_by_label) == (
        "ruff",
        "mypy",
        "pytest",
        "deployment-notebook",
    )
    assert step_by_label["pytest"].command[-1] == "tests/"
    assert step_by_label["deployment-notebook"].command[1] == "execute"
    assert (
        step_by_label["deployment-notebook"]
        .command[2]
        .endswith("notebooks/examples/deployment.ipynb")
    )


def test_build_verification_steps_can_skip_notebook_execution() -> None:
    """The repo check entry point should allow notebook execution to be skipped."""

    repo_root = Path(__file__).resolve().parents[1]
    steps = build_verification_steps(repo_root, include_notebook_execution=False)

    assert [step.label for step in steps] == ["ruff", "mypy", "pytest"]


def test_repository_does_not_track_python_bytecode_or_cache_directories() -> None:
    """Generated Python cache files should stay out of the tracked repository.

    Bytecode artifacts are interpreter-local build outputs. Tracking them makes
    diffs noisy, hides real source changes, and can mask stale behavior when
    developers switch Python versions.
    """

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip("git metadata is not available for repository hygiene checks")

    tracked_paths = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
        and ("__pycache__/" in line or line.endswith(".pyc") or line.endswith(".pyo"))
    )

    assert tracked_paths == ()
