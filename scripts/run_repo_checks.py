"""Repository verification entry point for the measurement-calibration project.

This script gives the repository a single command that exercises the main
quality boundaries in one place:

- static checks (`ruff` and `mypy`);
- the full test suite under ``tests/``;
- headless execution of the deployment notebook against a temporary output.

The notebook execution step writes its executed copy into a temporary
directory so the tracked notebook file remains untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import socket
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class VerificationStep:
    """One named repository verification step."""

    label: str
    command: tuple[str, ...]


def build_verification_steps(
    repo_root: Path,
    *,
    include_notebook_execution: bool,
) -> list[VerificationStep]:
    """Build the default verification plan for a clean checkout."""

    steps = [
        VerificationStep(
            label="ruff",
            command=(
                str(repo_root / ".venv" / "bin" / "ruff"),
                "check",
                "measurement_calibration",
                "api",
                "tests",
                "scripts",
            ),
        ),
        VerificationStep(
            label="mypy",
            command=(
                str(repo_root / ".venv" / "bin" / "mypy"),
                "measurement_calibration",
                "api",
                "tests",
                "scripts/run_repo_checks.py",
            ),
        ),
        VerificationStep(
            label="pytest",
            command=(
                str(repo_root / ".venv" / "bin" / "pytest"),
                "-q",
                "tests/",
            ),
        ),
    ]
    if include_notebook_execution:
        steps.append(
            VerificationStep(
                label="deployment-notebook",
                command=(
                    str(repo_root / ".venv" / "bin" / "jupyter"),
                    "execute",
                    str(repo_root / "notebooks" / "examples" / "deployment.ipynb"),
                    "--timeout=1200",
                ),
            )
        )
    return steps


def notebook_execution_supported() -> tuple[bool, str | None]:
    """Return whether the current process can launch a local Jupyter kernel."""

    try:
        with socket.socket():
            pass
    except OSError as error:
        return False, str(error)
    return True, None


def run_step(
    step: VerificationStep,
    *,
    repo_root: Path,
    dry_run: bool,
) -> None:
    """Run one verification step in the repository root."""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    if step.label != "deployment-notebook":
        print(f"[run] {step.label}: {' '.join(step.command)}")
        if dry_run:
            return
        subprocess.run(step.command, cwd=repo_root, env=env, check=True)
        return

    if not (repo_root / "models" / "production" / "manifest.json").exists():
        print("[skip] deployment-notebook: missing checked-in production artifact")
        return
    notebook_supported, reason = notebook_execution_supported()
    if not notebook_supported:
        print(f"[skip] deployment-notebook: notebook execution unavailable ({reason})")
        return

    with tempfile.TemporaryDirectory(
        prefix="measurement-calibration-notebooks-"
    ) as tmpdir:
        executed_notebook_path = Path(tmpdir) / "deployment-executed.ipynb"
        notebook_command = [
            *step.command,
            f"--output={executed_notebook_path}",
        ]
        print(f"[run] {step.label}: {' '.join(notebook_command)}")
        if dry_run:
            return
        subprocess.run(notebook_command, cwd=repo_root, env=env, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the repository checks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    parser.add_argument(
        "--skip-notebook-execution",
        action="store_true",
        help="Skip the headless deployment-notebook execution step.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the repository verification plan."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".venv" / "bin" / "pytest").exists():
        raise FileNotFoundError(
            "Expected project virtualenv tools under .venv/bin. "
            "Create the virtualenv before running repository checks."
        )

    for step in build_verification_steps(
        repo_root,
        include_notebook_execution=not args.skip_notebook_execution,
    ):
        run_step(
            step,
            repo_root=repo_root,
            dry_run=bool(args.dry_run),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
