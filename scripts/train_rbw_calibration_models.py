"""Train and store one spectral calibration artifact per RBW subset.

The RBW acquisitions are already row-aligned by record index, so this script
adapts them through :mod:`measurement_calibration.rbw_calibration` and reuses
the main spectral-calibration fitter and artifact serializer without creating a
parallel model format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from measurement_calibration.rbw_calibration import (  # noqa: E402
    DEFAULT_RBW_EXCLUDED_SENSOR_IDS,
    DEFAULT_RBW_EXCLUDED_SENSOR_IDS_BY_LABEL,
    build_rbw_preparation_rows,
    fit_and_save_rbw_calibration_model,
    load_rbw_calibration_preparations,
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
    """Build the CLI parser for the RBW calibration workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Train one spectral calibration model per RBW subset under "
            "data/RBW_acquisitions and store the artifacts under models/."
        )
    )
    parser.add_argument(
        "--rbw-root-dir",
        type=Path,
        default=REPO_ROOT / "data" / "RBW_acquisitions",
        help="Root directory that contains one RBW subdirectory per campaign.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=REPO_ROOT / "models" / "spectral_calibration_rbw",
        help="Root directory where per-RBW artifact bundles will be written.",
    )
    parser.add_argument(
        "--exclude-sensor-id",
        action="append",
        default=list(DEFAULT_RBW_EXCLUDED_SENSOR_IDS),
        help=(
            "Sensor to exclude before RBW preparation. Pass the flag multiple "
            "times to exclude several sensors."
        ),
    )
    parser.add_argument(
        "--exclude-sensor-id-by-rbw",
        action="append",
        default=[
            f"{rbw_label}:{sensor_id}"
            for rbw_label, sensor_ids in DEFAULT_RBW_EXCLUDED_SENSOR_IDS_BY_LABEL.items()
            for sensor_id in sensor_ids
        ],
        help=(
            "RBW-specific exclusion in the form RBW:SENSOR, for example 10K:Node5. "
            "Pass the flag multiple times to add several per-RBW exclusions."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of RBW records reserved for hold-out diagnostics.",
    )
    parser.add_argument(
        "--ranking-histogram-bins",
        type=int,
        default=50,
        help="Histogram bins used in the record-wise ranking noise-floor estimate.",
    )
    parser.add_argument(
        "--distribution-histogram-bins",
        type=int,
        default=300,
        help="Histogram bins used in the PSD-distribution diagnostic.",
    )
    return parser


def parse_rbw_sensor_exclusions(
    values: list[str],  # CLI values of the form ``RBW:SENSOR``
) -> dict[str, tuple[str, ...]]:  # Parsed stable mapping by RBW label
    """Parse per-RBW sensor exclusions from the CLI."""

    exclusions_by_rbw: dict[str, set[str]] = {}
    for value in values:
        try:
            rbw_label, sensor_id = value.split(":", maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"Per-RBW exclusions must use the form RBW:SENSOR, received {value!r}."
            ) from error
        rbw_label = rbw_label.strip()
        sensor_id = sensor_id.strip()
        if not rbw_label or not sensor_id:
            raise ValueError(
                "Per-RBW exclusions must provide both RBW and sensor id, "
                f"received {value!r}."
            )
        exclusions_by_rbw.setdefault(rbw_label, set()).add(sensor_id)
    return {
        rbw_label: tuple(sorted(sensor_ids))
        for rbw_label, sensor_ids in sorted(exclusions_by_rbw.items())
    }


def main() -> int:
    """Run RBW preparation, fitting, and artifact persistence."""

    args = build_argument_parser().parse_args()
    excluded_sensor_ids = tuple(
        sorted({str(sensor_id) for sensor_id in args.exclude_sensor_id})
    )
    excluded_sensor_ids_by_label = parse_rbw_sensor_exclusions(
        list(args.exclude_sensor_id_by_rbw)
    )

    preparations = load_rbw_calibration_preparations(
        root_dir=args.rbw_root_dir,
        excluded_sensor_ids=excluded_sensor_ids,
        excluded_sensor_ids_by_label=excluded_sensor_ids_by_label,
        ranking_histogram_bins=int(args.ranking_histogram_bins),
        distribution_histogram_bins=int(args.distribution_histogram_bins),
    )
    preparation_rows = build_rbw_preparation_rows(preparations)

    for row in preparation_rows:
        print(
            "prepared_rbw="
            f"{row['rbw']} reliable_sensor={row['reliable_sensor_id']} "
            f"retained_sensors={row['sensor_ids']} "
            f"excluded_sensors={row['excluded_sensor_ids']}"
        )

    for rbw_label in sorted(preparations):
        fit_result = fit_and_save_rbw_calibration_model(
            preparation=preparations[rbw_label],
            output_dir=args.output_root_dir / rbw_label,
            acquisition_dir=args.rbw_root_dir / rbw_label,
            fit_config=DEFAULT_FIT_CONFIG,
            test_fraction=float(args.test_fraction),
        )
        print(
            "saved_rbw="
            f"{rbw_label} output_dir={fit_result.artifact.output_dir} "
            f"reliable_sensor={fit_result.preparation.reliable_sensor_id} "
            f"excluded_sensors={','.join(fit_result.preparation.excluded_sensor_ids)} "
            f"auto_qc_excluded={','.join(fit_result.qc_auto_excluded_sensor_ids)} "
            f"objective_start={fit_result.artifact.manifest['result_summary']['objective_start']:.6e} "
            f"objective_end={fit_result.artifact.manifest['result_summary']['objective_end']:.6e} "
            f"dispersion_ratio={fit_result.corrected_to_raw_dispersion_ratio:.6f} "
            f"fit_duration_s={fit_result.fit_duration_s:.3f} "
            f"fit_passes={fit_result.n_fit_passes}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
