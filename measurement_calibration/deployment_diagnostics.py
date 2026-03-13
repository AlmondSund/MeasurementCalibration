"""Campaign-level deployment diagnostics for notebook overlay visualizations.

This module keeps notebook code focused on orchestration and rendering. The
helpers here calibrate every sensor of one already-prepared same-scene campaign
through an injected single-sensor calibrator, convert the results to dB, and
compute the frame-level pairwise RMSE values used in the animation title.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .spectral_calibration import (
    CalibrationCampaign,
    DeploymentCalibrationResult,
    power_linear_to_db,
)

FloatArray = NDArray[np.float64]


class CampaignSensorCalibrator(Protocol):
    """Callable that calibrates one sensor stream within a fixed campaign."""

    def __call__(
        self,
        sensor_id: str,  # Sensor identifier to calibrate
        observations_power: FloatArray,  # Sensor PSD tensor [linear power]
    ) -> DeploymentCalibrationResult:  # Calibrated deployment output
        """Return the calibrated deployment result for one campaign sensor."""


@dataclass(frozen=True)
class CrossNodeRecordAlignment:
    """Frame-level cross-node RMSE values used in the overlay title.

    Parameters
    ----------
    record_index:
        Zero-based aligned record index inside the campaign.
    mean_pairwise_raw_rmse_db:
        Mean pairwise RMSE across all sensor pairs before calibration [dB].
    mean_pairwise_calibrated_rmse_db:
        Mean pairwise RMSE across all sensor pairs after calibration [dB].
    """

    record_index: int
    mean_pairwise_raw_rmse_db: float
    mean_pairwise_calibrated_rmse_db: float

    @property
    def record_number(self) -> int:
        """Return the human-facing one-based record number."""

        return self.record_index + 1


@dataclass(frozen=True)
class CrossNodeCampaignAnimationData:
    """Validated tensors and metrics consumed by the overlay animation.

    Parameters
    ----------
    campaign_label:
        Campaign identifier shown in the animation title.
    sensor_ids:
        Ordered sensors rendered in both overlay panels.
    frequency_hz:
        Shared campaign frequency grid [Hz].
    raw_power_db:
        Raw PSD tensor with shape ``(n_sensors, n_records, n_frequencies)`` [dB].
    calibrated_power_db:
        Calibrated PSD tensor with the same shape and sensor ordering [dB].
    record_alignments:
        Frame-level RMSE summaries aligned with the record axis.
    """

    campaign_label: str
    sensor_ids: tuple[str, ...]
    frequency_hz: FloatArray
    raw_power_db: FloatArray
    calibrated_power_db: FloatArray
    record_alignments: tuple[CrossNodeRecordAlignment, ...]

    def __post_init__(self) -> None:
        """Validate tensor shapes and the record-level summary contract."""

        raw_power_db = np.asarray(self.raw_power_db, dtype=np.float64)
        calibrated_power_db = np.asarray(self.calibrated_power_db, dtype=np.float64)
        frequency_hz = np.asarray(self.frequency_hz, dtype=np.float64)

        if not self.campaign_label.strip():
            raise ValueError("campaign_label must be a non-empty string")
        if len(self.sensor_ids) < 2:
            raise ValueError("Cross-node overlays require at least two sensors")
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor_ids must be unique")
        if raw_power_db.ndim != 3:
            raise ValueError(
                "raw_power_db must have shape (n_sensors, n_records, n_frequencies)"
            )
        if calibrated_power_db.shape != raw_power_db.shape:
            raise ValueError("calibrated_power_db must match the shape of raw_power_db")
        if raw_power_db.shape[0] != len(self.sensor_ids):
            raise ValueError(
                "sensor_ids length must match the first tensor axis of raw_power_db"
            )
        if raw_power_db.shape[1] != len(self.record_alignments):
            raise ValueError(
                "record_alignments must contain exactly one summary per record"
            )
        if raw_power_db.shape[2] != frequency_hz.size:
            raise ValueError(
                "frequency_hz length must match the frequency axis of the tensors"
            )
        if raw_power_db.shape[1] < 1:
            raise ValueError("At least one aligned record is required")
        if not np.all(np.isfinite(raw_power_db)):
            raise ValueError("raw_power_db contains non-finite values")
        if not np.all(np.isfinite(calibrated_power_db)):
            raise ValueError("calibrated_power_db contains non-finite values")
        if not np.all(np.isfinite(frequency_hz)):
            raise ValueError("frequency_hz contains non-finite values")
        if np.any(np.diff(frequency_hz) <= 0.0):
            raise ValueError("frequency_hz must be strictly increasing")

        for expected_index, alignment in enumerate(self.record_alignments):
            if alignment.record_index != expected_index:
                raise ValueError(
                    "record_alignments must stay in zero-based record order"
                )

    @property
    def n_sensors(self) -> int:
        """Return the number of overlaid sensors."""

        return len(self.sensor_ids)

    @property
    def n_records(self) -> int:
        """Return the number of aligned records in the animation."""

        return int(np.asarray(self.raw_power_db).shape[1])


def build_cross_node_campaign_animation_data(
    campaign: CalibrationCampaign,  # Same-scene campaign to calibrate and visualize
    calibrate_sensor: CampaignSensorCalibrator,  # Campaign-bound calibrator
) -> CrossNodeCampaignAnimationData:
    """Calibrate every campaign sensor and compute frame-wise RMSE summaries.

    The function is intentionally pure from the notebook's perspective:
    filesystem access, model loading, and campaign preparation all happen
    outside this helper. The only external dependency is the injected
    ``calibrate_sensor`` callable, which lets tests replace the deployed model
    with a deterministic stub.
    """

    raw_power_db = power_linear_to_db(campaign.observations_power)
    calibrated_power_db = np.empty_like(raw_power_db)

    # Calibrate each sensor independently while preserving the campaign order
    # used by the animation legend and the pairwise RMSE summaries.
    for sensor_index, sensor_id in enumerate(campaign.sensor_ids):
        sensor_observations_power = np.asarray(
            campaign.observations_power[sensor_index],
            dtype=np.float64,
        )
        deployment = calibrate_sensor(
            sensor_id=sensor_id,
            observations_power=sensor_observations_power,
        )
        calibrated_sensor_power = np.asarray(
            deployment.calibrated_power,
            dtype=np.float64,
        )
        if calibrated_sensor_power.shape != sensor_observations_power.shape:
            raise ValueError(
                "calibrate_sensor must preserve the per-sensor observation shape"
            )
        if not np.all(np.isfinite(calibrated_sensor_power)):
            raise ValueError("calibrate_sensor returned non-finite calibrated power")
        if np.any(calibrated_sensor_power < 0.0):
            raise ValueError("calibrate_sensor returned negative calibrated power")
        calibrated_power_db[sensor_index] = power_linear_to_db(calibrated_sensor_power)

    record_alignments = []

    # Summarize each aligned record separately so the animation title can show
    # how the node-to-node spread evolves over time.
    for record_index in range(campaign.n_acquisitions):
        record_alignments.append(
            CrossNodeRecordAlignment(
                record_index=record_index,
                mean_pairwise_raw_rmse_db=_mean_pairwise_rmse_db(
                    raw_power_db[:, record_index, :]
                ),
                mean_pairwise_calibrated_rmse_db=_mean_pairwise_rmse_db(
                    calibrated_power_db[:, record_index, :]
                ),
            )
        )

    return CrossNodeCampaignAnimationData(
        campaign_label=campaign.campaign_label,
        sensor_ids=campaign.sensor_ids,
        frequency_hz=np.asarray(campaign.frequency_hz, dtype=np.float64),
        raw_power_db=raw_power_db,
        calibrated_power_db=calibrated_power_db,
        record_alignments=tuple(record_alignments),
    )


def format_cross_node_overlay_title(
    animation_data: CrossNodeCampaignAnimationData,  # Prepared animation payload
    record_index: int,  # Zero-based aligned record index
) -> str:
    """Return the notebook title for one cross-node overlay animation frame."""

    if record_index < 0 or record_index >= animation_data.n_records:
        raise IndexError(
            f"record_index must be in [0, {animation_data.n_records}), "
            f"got {record_index}"
        )

    record_alignment = animation_data.record_alignments[record_index]
    return (
        "Cross-Node PSD Overlays | "
        f"{animation_data.campaign_label} | "
        f"record {record_alignment.record_number}/{animation_data.n_records} | "
        f"mean pairwise RMSE raw={record_alignment.mean_pairwise_raw_rmse_db:.2f} dB, "
        f"cal={record_alignment.mean_pairwise_calibrated_rmse_db:.2f} dB"
    )


def resolve_cross_node_overlay_limits_db(
    animation_data: CrossNodeCampaignAnimationData,  # Prepared animation payload
    padding_db: float = 1.0,  # Symmetric vertical padding [dB]
) -> tuple[float, float]:
    """Return shared y-axis limits for the before/after overlay panels."""

    if padding_db < 0.0:
        raise ValueError("padding_db must be non-negative")

    all_animation_values_db = np.concatenate(
        [
            np.asarray(animation_data.raw_power_db, dtype=np.float64).reshape(-1),
            np.asarray(animation_data.calibrated_power_db, dtype=np.float64).reshape(
                -1
            ),
        ]
    )
    return (
        float(np.min(all_animation_values_db) - padding_db),
        float(np.max(all_animation_values_db) + padding_db),
    )


def _mean_pairwise_rmse_db(
    psd_by_sensor_db: FloatArray,  # PSD matrix with shape (n_sensors, n_frequencies)
) -> float:
    """Return the mean pairwise RMSE across sensors for one aligned record."""

    psd_by_sensor_db = np.asarray(psd_by_sensor_db, dtype=np.float64)
    if psd_by_sensor_db.ndim != 2:
        raise ValueError("psd_by_sensor_db must have shape (n_sensors, n_frequencies)")
    if psd_by_sensor_db.shape[0] < 2:
        raise ValueError("At least two sensors are required for pairwise RMSE")
    if psd_by_sensor_db.shape[1] < 1:
        raise ValueError("At least one frequency bin is required for pairwise RMSE")
    if not np.all(np.isfinite(psd_by_sensor_db)):
        raise ValueError("psd_by_sensor_db contains non-finite values")

    pairwise_rmse_values_db: list[np.ndarray] = []

    # Compute every unique sensor-pair RMSE without materializing a large
    # square distance tensor.
    for left_index in range(psd_by_sensor_db.shape[0] - 1):
        difference_db = (
            psd_by_sensor_db[left_index + 1 :] - psd_by_sensor_db[left_index]
        )
        pairwise_rmse_values_db.append(np.sqrt(np.mean(difference_db**2, axis=1)))

    return float(np.mean(np.concatenate(pairwise_rmse_values_db)))
