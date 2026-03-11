"""Shared synthetic fixtures for the two-level calibration tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from measurement_calibration.spectral_calibration import (
    CalibrationCampaign,
    CalibrationCorpus,
    CampaignConfiguration,
    FrequencyBasisConfig,
    PersistentModelConfig,
    TwoLevelFitConfig,
    _build_spline_basis,
)


@dataclass(frozen=True)
class SyntheticTwoLevelFixture:
    """Synthetic corpus plus ground truth for offline and deployment tests."""

    corpus: CalibrationCorpus
    basis_config: FrequencyBasisConfig
    model_config: PersistentModelConfig
    fit_config: TwoLevelFitConfig
    true_gain_by_campaign: dict[str, np.ndarray]
    true_floor_by_campaign: dict[str, np.ndarray]
    true_variance_by_campaign: dict[str, np.ndarray]
    deployment_sensor_id: str
    deployment_configuration: CampaignConfiguration
    deployment_frequency_hz: np.ndarray
    deployment_observations_power: np.ndarray
    deployment_true_latent_power: np.ndarray


def build_synthetic_two_level_fixture() -> SyntheticTwoLevelFixture:
    """Build a deterministic corpus that matches the implemented model class."""

    sensor_ids = ("Node1", "Node2", "Node3", "Node4")
    basis_config = FrequencyBasisConfig(
        n_gain_basis=6,
        n_floor_basis=5,
        n_variance_basis=5,
        spline_degree=3,
    )
    model_config = PersistentModelConfig(
        sensor_embedding_dim=3,
        configuration_latent_dim=2,
    )
    fit_config = TwoLevelFitConfig(
        n_outer_iterations=8,
        n_gradient_steps=20,
        learning_rate=0.03,
        sigma_min=1.0e-8,
        lambda_delta_gain_smooth=0.5,
        lambda_delta_floor_smooth=0.5,
        lambda_delta_variance_smooth=0.5,
        lambda_delta_gain_shrink=0.1,
        lambda_delta_floor_shrink=0.1,
        lambda_delta_variance_shrink=0.1,
        lambda_reliable_sensor_anchor=0.05,
        weight_decay=1.0e-4,
        gradient_clip_norm=5.0,
        random_seed=4,
    )
    training_configurations = (
        CampaignConfiguration(
            central_frequency_hz=98.0e6,
            span_hz=20.0e6,
            resolution_bandwidth_hz=10.0e3,
            lna_gain_db=0.0,
            vga_gain_db=62.0,
            acquisition_interval_s=120.0,
            antenna_amplifier_enabled=True,
        ),
        CampaignConfiguration(
            central_frequency_hz=104.0e6,
            span_hz=18.0e6,
            resolution_bandwidth_hz=20.0e3,
            lna_gain_db=4.0,
            vga_gain_db=58.0,
            acquisition_interval_s=180.0,
            antenna_amplifier_enabled=False,
        ),
        CampaignConfiguration(
            central_frequency_hz=92.0e6,
            span_hz=16.0e6,
            resolution_bandwidth_hz=15.0e3,
            lna_gain_db=2.0,
            vga_gain_db=60.0,
            acquisition_interval_s=150.0,
            antenna_amplifier_enabled=True,
        ),
    )
    raw_feature_matrix = np.stack(
        [configuration.to_feature_vector() for configuration in training_configurations],
        axis=0,
    )
    feature_mean = np.mean(raw_feature_matrix, axis=0)
    feature_scale = np.std(raw_feature_matrix, axis=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    frequency_min_hz = 84.0e6
    frequency_max_hz = 112.0e6

    sensor_embedding_true = np.asarray(
        [
            [-0.35, 0.15, 0.20],
            [-0.10, 0.05, -0.25],
            [0.18, -0.12, 0.28],
            [0.30, 0.20, -0.18],
        ],
        dtype=np.float64,
    )
    configuration_encoder_weight_true = np.asarray(
        [
            [0.35, -0.20, 0.25, 0.10, -0.12, 0.08, 0.30],
            [-0.18, 0.22, 0.15, -0.10, 0.14, -0.06, -0.28],
        ],
        dtype=np.float64,
    )
    configuration_encoder_bias_true = np.asarray([0.10, -0.05], dtype=np.float64)
    gain_head_weight_true = 2.4 * np.asarray(
        [
            [0.35, -0.22, 0.18, 0.14, -0.08],
            [-0.12, 0.16, 0.10, -0.10, 0.06],
            [0.18, 0.08, -0.15, 0.12, 0.05],
            [0.05, -0.18, 0.14, -0.04, -0.08],
            [-0.08, 0.10, -0.10, 0.06, 0.03],
            [0.03, -0.04, 0.06, 0.04, -0.02],
        ],
        dtype=np.float64,
    )
    gain_head_bias_true = 2.4 * np.asarray(
        [0.03, -0.01, 0.02, -0.02, 0.00, 0.01],
        dtype=np.float64,
    )
    floor_head_weight_true = np.asarray(
        [
            [0.08, -0.05, 0.04, 0.03, -0.02],
            [-0.04, 0.06, -0.05, 0.01, 0.03],
            [0.03, -0.02, 0.05, -0.02, -0.01],
            [0.02, 0.04, -0.03, 0.02, 0.00],
            [-0.01, 0.03, 0.02, -0.01, 0.02],
        ],
        dtype=np.float64,
    )
    floor_head_bias_true = np.asarray([-2.7, -2.5, -2.3, -2.6, -2.4], dtype=np.float64)
    variance_head_weight_true = np.asarray(
        [
            [0.10, -0.03, 0.05, 0.03, -0.01],
            [-0.03, 0.05, 0.04, -0.02, 0.02],
            [0.04, -0.02, -0.03, 0.04, -0.01],
            [0.02, 0.03, -0.01, 0.02, 0.00],
            [-0.01, 0.02, 0.03, -0.01, 0.02],
        ],
        dtype=np.float64,
    )
    variance_head_bias_true = np.asarray(
        [-6.2, -6.0, -6.1, -5.9, -6.15],
        dtype=np.float64,
    )

    rng = np.random.default_rng(17)
    campaign_labels = ("cal-A", "cal-B", "cal-C")
    campaign_sensor_sets = (
        ("Node1", "Node2", "Node3"),
        ("Node2", "Node3", "Node4"),
        ("Node1", "Node3", "Node4"),
    )
    frequency_grids = (
        np.linspace(88.0e6, 108.0e6, 24, dtype=np.float64),
        np.linspace(90.0e6, 108.0e6, 26, dtype=np.float64),
        np.linspace(86.0e6, 106.0e6, 22, dtype=np.float64),
    )
    acquisition_counts = (10, 9, 11)
    campaigns: list[CalibrationCampaign] = []
    true_gain_by_campaign: dict[str, np.ndarray] = {}
    true_floor_by_campaign: dict[str, np.ndarray] = {}
    true_variance_by_campaign: dict[str, np.ndarray] = {}

    for campaign_index, (
        campaign_label,
        campaign_sensor_ids,
        configuration,
        frequency_hz,
        n_acquisitions,
    ) in enumerate(
        zip(
            campaign_labels,
            campaign_sensor_sets,
            training_configurations,
            frequency_grids,
            acquisition_counts,
            strict=True,
        )
    ):
        standardized_configuration = (
            configuration.to_feature_vector() - feature_mean
        ) / feature_scale
        persistent_log_gain_all, persistent_floor_all, persistent_variance_all = (
            _evaluate_persistent_truth(
                frequency_hz=frequency_hz,
                standardized_configuration=standardized_configuration,
                basis_config=basis_config,
                frequency_min_hz=frequency_min_hz,
                frequency_max_hz=frequency_max_hz,
                sensor_embedding=sensor_embedding_true,
                configuration_encoder_weight=configuration_encoder_weight_true,
                configuration_encoder_bias=configuration_encoder_bias_true,
                gain_head_weight=gain_head_weight_true,
                gain_head_bias=gain_head_bias_true,
                floor_head_weight=floor_head_weight_true,
                floor_head_bias=floor_head_bias_true,
                variance_head_weight=variance_head_weight_true,
                variance_head_bias=variance_head_bias_true,
            )
        )
        participant_indices = np.asarray(
            [sensor_ids.index(sensor_id) for sensor_id in campaign_sensor_ids],
            dtype=np.int64,
        )
        persistent_log_gain = persistent_log_gain_all[participant_indices]
        persistent_floor = persistent_floor_all[participant_indices]
        persistent_variance = persistent_variance_all[participant_indices]

        normalized_frequency = (frequency_hz - frequency_hz[0]) / (
            frequency_hz[-1] - frequency_hz[0]
        )
        sensor_phase = np.linspace(
            0.0,
            np.pi,
            len(campaign_sensor_ids),
            endpoint=False,
            dtype=np.float64,
        )[:, np.newaxis]
        raw_delta_log_gain = 0.025 * (campaign_index + 1) * np.sin(
            2.0 * np.pi * normalized_frequency[np.newaxis, :] + sensor_phase
        )
        delta_log_gain = raw_delta_log_gain - np.mean(
            raw_delta_log_gain, axis=0, keepdims=True
        )
        delta_floor = 0.12 * np.cos(
            np.pi * normalized_frequency[np.newaxis, :] + 0.5 * sensor_phase
        )
        delta_variance = 0.08 * np.sin(
            0.5 * np.pi * normalized_frequency[np.newaxis, :] + 0.3 * sensor_phase
        )
        total_log_gain = persistent_log_gain + delta_log_gain
        total_floor = persistent_floor + delta_floor
        total_variance = persistent_variance + delta_variance
        gain_power = np.exp(total_log_gain)
        floor_power = np.logaddexp(0.0, total_floor)
        variance_power2 = fit_config.sigma_min + np.logaddexp(0.0, total_variance)

        acquisition_phase = np.linspace(
            0.0,
            np.pi,
            n_acquisitions,
            endpoint=False,
            dtype=np.float64,
        )[:, np.newaxis]
        latent_power = np.clip(
            0.75
            + 0.14 * np.sin(2.0 * np.pi * normalized_frequency[np.newaxis, :] + acquisition_phase)
            + 0.06 * np.cos(
                (campaign_index + 1) * np.pi * normalized_frequency[np.newaxis, :]
            )
            + 0.03 * np.arange(n_acquisitions, dtype=np.float64)[:, np.newaxis],
            0.08,
            None,
        )
        noise = np.sqrt(variance_power2)[:, np.newaxis, :] * rng.normal(
            size=(len(campaign_sensor_ids), n_acquisitions, frequency_hz.size)
        )
        observations_power = np.clip(
            gain_power[:, np.newaxis, :] * latent_power[np.newaxis, :, :]
            + floor_power[:, np.newaxis, :]
            + noise,
            1.0e-8,
            None,
        )
        campaigns.append(
            CalibrationCampaign(
                campaign_label=campaign_label,
                sensor_ids=campaign_sensor_ids,
                frequency_hz=frequency_hz,
                observations_power=observations_power,
                configuration=configuration,
                reliable_sensor_id=campaign_sensor_ids[0],
            )
        )
        true_gain_by_campaign[campaign_label] = gain_power
        true_floor_by_campaign[campaign_label] = floor_power
        true_variance_by_campaign[campaign_label] = variance_power2

    corpus = CalibrationCorpus(sensor_ids=sensor_ids, campaigns=tuple(campaigns))

    deployment_configuration = CampaignConfiguration(
        central_frequency_hz=100.0e6,
        span_hz=19.0e6,
        resolution_bandwidth_hz=12.0e3,
        lna_gain_db=1.5,
        vga_gain_db=61.0,
        acquisition_interval_s=140.0,
        antenna_amplifier_enabled=False,
    )
    deployment_frequency_hz = np.linspace(89.0e6, 107.0e6, 25, dtype=np.float64)
    deployment_sensor_id = "Node2"
    deployment_standardized_configuration = (
        deployment_configuration.to_feature_vector() - feature_mean
    ) / feature_scale
    deployment_log_gain_all, deployment_floor_all, deployment_variance_all = (
        _evaluate_persistent_truth(
            frequency_hz=deployment_frequency_hz,
            standardized_configuration=deployment_standardized_configuration,
            basis_config=basis_config,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
            sensor_embedding=sensor_embedding_true,
            configuration_encoder_weight=configuration_encoder_weight_true,
            configuration_encoder_bias=configuration_encoder_bias_true,
            gain_head_weight=gain_head_weight_true,
            gain_head_bias=gain_head_bias_true,
            floor_head_weight=floor_head_weight_true,
            floor_head_bias=floor_head_bias_true,
            variance_head_weight=variance_head_weight_true,
            variance_head_bias=variance_head_bias_true,
        )
    )
    deployment_sensor_index = sensor_ids.index(deployment_sensor_id)
    deployment_gain_power = np.exp(deployment_log_gain_all[deployment_sensor_index])
    deployment_floor_power = np.logaddexp(0.0, deployment_floor_all[deployment_sensor_index])
    deployment_variance_power2 = fit_config.sigma_min + np.logaddexp(
        0.0, deployment_variance_all[deployment_sensor_index]
    )
    deployment_normalized_frequency = (
        deployment_frequency_hz - deployment_frequency_hz[0]
    ) / (deployment_frequency_hz[-1] - deployment_frequency_hz[0])
    deployment_true_latent_power = np.clip(
        0.68
        + 0.18
        * np.sin(
            2.5 * np.pi * deployment_normalized_frequency[np.newaxis, :]
            + np.linspace(0.0, 1.5, 7, dtype=np.float64)[:, np.newaxis]
        )
        + 0.05 * np.cos(np.pi * deployment_normalized_frequency[np.newaxis, :]),
        0.06,
        None,
    )
    deployment_noise = np.sqrt(deployment_variance_power2)[np.newaxis, :] * rng.normal(
        size=deployment_true_latent_power.shape
    )
    deployment_observations_power = np.clip(
        deployment_gain_power[np.newaxis, :] * deployment_true_latent_power
        + deployment_floor_power[np.newaxis, :]
        + deployment_noise,
        1.0e-8,
        None,
    )

    return SyntheticTwoLevelFixture(
        corpus=corpus,
        basis_config=basis_config,
        model_config=model_config,
        fit_config=fit_config,
        true_gain_by_campaign=true_gain_by_campaign,
        true_floor_by_campaign=true_floor_by_campaign,
        true_variance_by_campaign=true_variance_by_campaign,
        deployment_sensor_id=deployment_sensor_id,
        deployment_configuration=deployment_configuration,
        deployment_frequency_hz=deployment_frequency_hz,
        deployment_observations_power=deployment_observations_power,
        deployment_true_latent_power=deployment_true_latent_power,
    )


def _evaluate_persistent_truth(
    frequency_hz: np.ndarray,
    standardized_configuration: np.ndarray,
    basis_config: FrequencyBasisConfig,
    frequency_min_hz: float,
    frequency_max_hz: float,
    sensor_embedding: np.ndarray,
    configuration_encoder_weight: np.ndarray,
    configuration_encoder_bias: np.ndarray,
    gain_head_weight: np.ndarray,
    gain_head_bias: np.ndarray,
    floor_head_weight: np.ndarray,
    floor_head_bias: np.ndarray,
    variance_head_weight: np.ndarray,
    variance_head_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the synthetic ground-truth persistent laws."""

    gain_basis = _build_spline_basis(
        frequency_hz=frequency_hz,
        n_basis=basis_config.n_gain_basis,
        degree=basis_config.spline_degree,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    floor_basis = _build_spline_basis(
        frequency_hz=frequency_hz,
        n_basis=basis_config.n_floor_basis,
        degree=basis_config.spline_degree,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    variance_basis = _build_spline_basis(
        frequency_hz=frequency_hz,
        n_basis=basis_config.n_variance_basis,
        degree=basis_config.spline_degree,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    configuration_latent = np.tanh(
        configuration_encoder_weight @ standardized_configuration
        + configuration_encoder_bias
    )
    combined_features = np.concatenate(
        [
            sensor_embedding,
            np.broadcast_to(
                configuration_latent,
                (sensor_embedding.shape[0], configuration_latent.size),
            ),
        ],
        axis=1,
    )
    raw_log_gain = combined_features @ gain_head_weight.T + gain_head_bias
    raw_log_gain = raw_log_gain @ gain_basis.T
    centered_log_gain = raw_log_gain - np.mean(raw_log_gain, axis=0, keepdims=True)
    floor_parameter = (
        combined_features @ floor_head_weight.T + floor_head_bias
    ) @ floor_basis.T
    variance_parameter = (
        combined_features @ variance_head_weight.T + variance_head_bias
    ) @ variance_basis.T
    return centered_log_gain, floor_parameter, variance_parameter
