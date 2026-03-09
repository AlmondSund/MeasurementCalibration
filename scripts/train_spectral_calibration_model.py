"""Train and store the real-data spectral calibration model under ``models/``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Import after adjusting ``sys.path`` so the script works from any launch
# directory inside the repository checkout.
from measurement_calibration.artifacts import (  # noqa: E402
    save_spectral_calibration_artifact,
)
from measurement_calibration.spectral_calibration import (  # noqa: E402
    fit_spectral_calibration,
    load_calibration_dataset,
    make_holdout_split,
    resolve_spectral_fit_config,
)


DEFAULT_EXCLUDED_SENSOR_IDS = (
    "Node3-Bogota",
    "Node6-Bogota",
    "Node8-Bogota",
    "Node9-Funza",
)

DEFAULT_FIT_CONFIG = {
    "n_iterations": 8,
    "lambda_gain_smooth": 120.0,
    "lambda_noise_smooth": 180.0,
    "lambda_gain_reference": 4.0,
    "lambda_noise_reference": 60.0,
    "lambda_reliable_anchor": 1.0,
    "reliable_weight_boost": 1.10,
    "low_information_threshold_ratio": 0.10,
    "low_information_weight": 0.05,
}


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the calibration training workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the spectral calibration model on the repository's common-field "
            "dataset and store the artifact bundle under models/."
        )
    )
    parser.add_argument(
        "--acquisition-dir",
        type=Path,
        default=REPO_ROOT / "data" / "acquisitions",
        help="Directory that contains the acquisition CSV files.",
    )
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=REPO_ROOT / "data" / "frequency-responses",
        help="Directory that contains the nominal response CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "spectral_calibration_real_data",
        help="Destination directory for the saved calibration artifact bundle.",
    )
    parser.add_argument(
        "--reference-sensor-id",
        default="Node7-Bogota",
        help="Sensor used to align the acquisition timeline.",
    )
    parser.add_argument(
        "--reliable-sensor-id",
        default="Node7-Bogota",
        help="Sensor softly anchored during fitting.",
    )
    parser.add_argument(
        "--exclude-sensor-id",
        action="append",
        default=list(DEFAULT_EXCLUDED_SENSOR_IDS),
        help=(
            "Sensor to exclude before training. Pass the flag multiple times to "
            "exclude several sensors."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of aligned experiments reserved for holdout diagnostics.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("tail", "random"),
        default="random",
        help="Hold-out split policy. Use random for drift-robust validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed used when --split-strategy=random.",
    )
    return parser


def main() -> int:
    """Run the real-data training workflow and store the artifact bundle."""

    args = build_argument_parser().parse_args()
    excluded_sensor_ids = tuple(sorted(set(args.exclude_sensor_id)))
    resolved_fit_config = resolve_spectral_fit_config(DEFAULT_FIT_CONFIG)

    dataset = load_calibration_dataset(
        acquisition_dir=args.acquisition_dir,
        response_dir=args.response_dir,
        reference_sensor_id=args.reference_sensor_id,
        excluded_sensor_ids=excluded_sensor_ids,
    )
    train_indices, test_indices = make_holdout_split(
        n_experiments=dataset.observations_power.shape[1],
        test_fraction=args.test_fraction,
        strategy=args.split_strategy,
        random_seed=args.split_seed,
    )

    start_time = time.perf_counter()
    result = fit_spectral_calibration(
        observations_power=dataset.observations_power,
        frequency_hz=dataset.frequency_hz,
        sensor_ids=dataset.sensor_ids,
        nominal_gain_power=dataset.nominal_gain_power,
        train_indices=train_indices,
        test_indices=test_indices,
        reliable_sensor_id=args.reliable_sensor_id,
        n_iterations=int(cast(int, resolved_fit_config["n_iterations"])),
        lambda_gain_smooth=float(
            cast(float, resolved_fit_config["lambda_gain_smooth"])
        ),
        lambda_noise_smooth=float(
            cast(float, resolved_fit_config["lambda_noise_smooth"])
        ),
        lambda_gain_reference=float(
            cast(float, resolved_fit_config["lambda_gain_reference"])
        ),
        lambda_noise_reference=float(
            cast(float, resolved_fit_config["lambda_noise_reference"])
        ),
        lambda_reliable_anchor=float(
            cast(float, resolved_fit_config["lambda_reliable_anchor"])
        ),
        reliable_weight_boost=float(
            cast(float, resolved_fit_config["reliable_weight_boost"])
        ),
        max_correction_db=cast(float | None, resolved_fit_config["max_correction_db"]),
        low_information_threshold_ratio=float(
            cast(float, resolved_fit_config["low_information_threshold_ratio"])
        ),
        low_information_weight=float(
            cast(float, resolved_fit_config["low_information_weight"])
        ),
        min_variance=float(cast(float, resolved_fit_config["min_variance"])),
    )
    fit_duration_s = time.perf_counter() - start_time

    artifact = save_spectral_calibration_artifact(
        output_dir=args.output_dir,
        result=result,
        dataset=dataset,
        acquisition_dir=args.acquisition_dir,
        response_dir=args.response_dir,
        reference_sensor_id=args.reference_sensor_id,
        reliable_sensor_id=args.reliable_sensor_id,
        excluded_sensor_ids=excluded_sensor_ids,
        fit_config=resolved_fit_config,
        extra_summary={
            "fit_duration_s": fit_duration_s,
            "test_fraction": args.test_fraction,
        },
    )

    # Keep CLI output compact but explicit so the user can verify the run
    # without opening the manifest immediately.
    print(f"saved_artifact_dir={artifact.output_dir}")
    print(f"parameters_file={artifact.parameters_path.name}")
    print(f"manifest_file={artifact.manifest_path.name}")
    print(f"sensor_summary_file={artifact.sensor_summary_path.name}")
    print(f"n_sensors={len(dataset.sensor_ids)}")
    print(f"n_experiments={dataset.observations_power.shape[1]}")
    print(f"n_frequencies={dataset.frequency_hz.size}")
    print(f"objective_start={result.objective_history[0]:.6e}")
    print(f"objective_end={result.objective_history[-1]:.6e}")
    print(f"fit_duration_s={fit_duration_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
