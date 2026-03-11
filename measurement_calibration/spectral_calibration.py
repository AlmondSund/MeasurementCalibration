"""Two-level configuration-conditional spectral calibration model.

This module implements the new theoretical framework documented in
``docs/main.tex``. The design follows the requested architectural split:

- campaign loading, timestamp alignment, and ranking diagnostics remain in
  adapter/orchestration modules;
- this module owns the pure numerical core for offline corpus fitting and
  online single-sensor deployment;
- artifact persistence is handled elsewhere.

The offline model learns persistent sensor/configuration laws over frequency
from a corpus of colocated campaigns while keeping campaign-specific
deviations as training-only nuisance variables. Deployment then evaluates the
persistent laws for one sensor and applies the calibration map without any
same-scene assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline


FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]

_EPSILON = 1.0e-12
_CONFIGURATION_FEATURE_NAMES = (
    "central_frequency_hz",
    "span_hz",
    "resolution_bandwidth_hz",
    "lna_gain_db",
    "vga_gain_db",
    "acquisition_interval_s",
    "antenna_amplifier_enabled",
)


@dataclass(frozen=True)
class CampaignConfiguration:
    """Acquisition configuration attached to one calibration campaign.

    Parameters
    ----------
    central_frequency_hz:
        Campaign center frequency [Hz].
    span_hz:
        Retained acquisition span or equivalent sample-rate metadata [Hz].
    resolution_bandwidth_hz:
        Resolution bandwidth [Hz].
    lna_gain_db:
        Configured LNA gain [dB].
    vga_gain_db:
        Configured VGA gain [dB].
    acquisition_interval_s:
        Time between successive acquisitions [s].
    antenna_amplifier_enabled:
        Whether an external antenna amplifier was enabled.
    """

    central_frequency_hz: float
    span_hz: float
    resolution_bandwidth_hz: float
    lna_gain_db: float
    vga_gain_db: float
    acquisition_interval_s: float
    antenna_amplifier_enabled: bool

    def __post_init__(self) -> None:
        """Validate the physically meaningful configuration ranges."""

        if self.central_frequency_hz <= 0.0:
            raise ValueError("central_frequency_hz must be strictly positive")
        if self.span_hz <= 0.0:
            raise ValueError("span_hz must be strictly positive")
        if self.resolution_bandwidth_hz <= 0.0:
            raise ValueError("resolution_bandwidth_hz must be strictly positive")
        if self.acquisition_interval_s <= 0.0:
            raise ValueError("acquisition_interval_s must be strictly positive")
        numeric_values = (
            self.central_frequency_hz,
            self.span_hz,
            self.resolution_bandwidth_hz,
            self.lna_gain_db,
            self.vga_gain_db,
            self.acquisition_interval_s,
        )
        if not np.all(np.isfinite(np.asarray(numeric_values, dtype=np.float64))):
            raise ValueError("CampaignConfiguration contains non-finite numeric values")

    def to_feature_vector(self) -> FloatArray:
        """Return the deterministic numeric feature vector used by the model."""

        return np.asarray(
            [
                self.central_frequency_hz,
                self.span_hz,
                self.resolution_bandwidth_hz,
                self.lna_gain_db,
                self.vga_gain_db,
                self.acquisition_interval_s,
                1.0 if self.antenna_amplifier_enabled else 0.0,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CalibrationCampaign:
    """One aligned same-scene campaign used in the offline corpus.

    Parameters
    ----------
    campaign_label:
        Human-readable campaign identifier.
    sensor_ids:
        Ordered sensor identifiers present in this campaign.
    frequency_hz:
        Retained frequency grid [Hz].
    observations_power:
        PSD tensor in linear power units with shape
        ``(n_sensors, n_acquisitions, n_frequencies)``.
    configuration:
        Acquisition configuration held constant within this campaign.
    reliable_sensor_id:
        Optional reliable-sensor annotation used only as a soft anchor on the
        campaign-specific log-gain deviation.
    """

    campaign_label: str
    sensor_ids: tuple[str, ...]
    frequency_hz: FloatArray
    observations_power: FloatArray
    configuration: CampaignConfiguration
    reliable_sensor_id: str | None = None

    def __post_init__(self) -> None:
        """Validate shapes, positivity, and sensor references."""

        if not self.campaign_label.strip():
            raise ValueError("campaign_label must be a non-empty string")
        if len(self.sensor_ids) < 2:
            raise ValueError("Each calibration campaign must contain at least two sensors")
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("sensor_ids must be unique within a campaign")

        frequency_hz = np.asarray(self.frequency_hz, dtype=np.float64)
        observations_power = np.asarray(self.observations_power, dtype=np.float64)
        if observations_power.ndim != 3:
            raise ValueError(
                "observations_power must have shape "
                "(n_sensors, n_acquisitions, n_frequencies)"
            )
        if observations_power.shape[0] != len(self.sensor_ids):
            raise ValueError(
                "sensor_ids length must match the first axis of observations_power"
            )
        if observations_power.shape[2] != frequency_hz.size:
            raise ValueError(
                "frequency_hz length must match the last axis of observations_power"
            )
        if observations_power.shape[1] < 1:
            raise ValueError("Each campaign must contain at least one acquisition")
        if np.any(observations_power <= 0.0):
            raise ValueError("observations_power must be strictly positive")
        if not np.all(np.isfinite(observations_power)):
            raise ValueError("observations_power contains non-finite values")
        if not np.all(np.isfinite(frequency_hz)):
            raise ValueError("frequency_hz contains non-finite values")
        if np.any(np.diff(frequency_hz) <= 0.0):
            raise ValueError("frequency_hz must be strictly increasing")
        if self.reliable_sensor_id is not None and self.reliable_sensor_id not in set(
            self.sensor_ids
        ):
            raise ValueError(
                "reliable_sensor_id must belong to sensor_ids when it is provided"
            )

    @property
    def n_sensors(self) -> int:
        """Return the number of participating sensors."""

        return len(self.sensor_ids)

    @property
    def n_acquisitions(self) -> int:
        """Return the number of aligned acquisitions."""

        return int(self.observations_power.shape[1])

    @property
    def n_frequencies(self) -> int:
        """Return the number of retained frequency bins."""

        return int(self.frequency_hz.size)


@dataclass(frozen=True)
class CalibrationCorpus:
    """Offline corpus of same-scene calibration campaigns.

    Parameters
    ----------
    sensor_ids:
        Global sensor registry used by the persistent deployment-scale laws.
    campaigns:
        Campaigns used to fit the persistent model and campaign deviations.
    """

    sensor_ids: tuple[str, ...]
    campaigns: tuple[CalibrationCampaign, ...]

    def __post_init__(self) -> None:
        """Validate the global registry and campaign membership."""

        if not self.campaigns:
            raise ValueError("CalibrationCorpus must contain at least one campaign")
        if len(set(self.sensor_ids)) != len(self.sensor_ids):
            raise ValueError("CalibrationCorpus.sensor_ids must be unique")
        sensor_id_set = set(self.sensor_ids)
        for campaign in self.campaigns:
            unknown_sensor_ids = sorted(set(campaign.sensor_ids).difference(sensor_id_set))
            if unknown_sensor_ids:
                raise ValueError(
                    "Every campaign sensor must belong to the global registry, but "
                    f"{campaign.campaign_label!r} contains {unknown_sensor_ids}"
                )


@dataclass(frozen=True)
class FrequencyBasisConfig:
    """Frequency-basis configuration for the persistent laws.

    Parameters
    ----------
    n_gain_basis:
        Number of spline basis functions used for the persistent log-gain law.
    n_floor_basis:
        Number of spline basis functions used for the additive-floor
        pre-activation law.
    n_variance_basis:
        Number of spline basis functions used for the variance pre-activation
        law.
    spline_degree:
        Spline degree shared by all basis families. Cubic splines correspond to
        ``3``.
    """

    n_gain_basis: int = 8
    n_floor_basis: int = 8
    n_variance_basis: int = 8
    spline_degree: int = 3

    def __post_init__(self) -> None:
        """Validate the spline basis dimensions."""

        if self.spline_degree < 0:
            raise ValueError("spline_degree cannot be negative")
        for name, count in (
            ("n_gain_basis", self.n_gain_basis),
            ("n_floor_basis", self.n_floor_basis),
            ("n_variance_basis", self.n_variance_basis),
        ):
            if count < self.spline_degree + 1:
                raise ValueError(
                    f"{name} must be at least spline_degree + 1, received {count}"
                )


@dataclass(frozen=True)
class PersistentModelConfig:
    """Trainable persistent-model dimensions.

    Parameters
    ----------
    sensor_embedding_dim:
        Dimension of the trainable sensor embedding.
    configuration_latent_dim:
        Dimension of the learned configuration encoding.
    """

    sensor_embedding_dim: int = 4
    configuration_latent_dim: int = 4

    def __post_init__(self) -> None:
        """Validate latent dimensions."""

        if self.sensor_embedding_dim < 1:
            raise ValueError("sensor_embedding_dim must be at least 1")
        if self.configuration_latent_dim < 1:
            raise ValueError("configuration_latent_dim must be at least 1")


@dataclass(frozen=True)
class TwoLevelFitConfig:
    """Numerical configuration for offline two-level optimization.

    Parameters
    ----------
    n_outer_iterations:
        Number of block-alternating iterations between latent-spectrum updates
        and gradient optimization of persistent parameters and deviations.
    n_gradient_steps:
        Gradient steps performed inside each outer iteration.
    learning_rate:
        Adam learning rate shared by all trainable variables.
    sigma_min:
        Minimum residual variance floor added to the softplus variance law.
    lambda_delta_gain_smooth, lambda_delta_floor_smooth, lambda_delta_variance_smooth:
        Frequency-smoothness strengths for the campaign deviations.
    lambda_delta_gain_shrink, lambda_delta_floor_shrink, lambda_delta_variance_shrink:
        Quadratic shrinkage strengths that keep the deviations close to zero.
    lambda_reliable_sensor_anchor:
        Soft anchor weight applied to the reliable sensor log-gain deviation.
    weight_decay:
        Global L2 regularization on the persistent parameters ``theta``.
    gradient_clip_norm:
        Optional global gradient-norm cap. ``None`` disables clipping.
    random_seed:
        Seed for deterministic parameter initialization.
    adam_beta1, adam_beta2, adam_epsilon:
        Adam optimizer hyperparameters.
    """

    n_outer_iterations: int = 10
    n_gradient_steps: int = 30
    learning_rate: float = 0.03
    sigma_min: float = 1.0e-10
    lambda_delta_gain_smooth: float = 1.0
    lambda_delta_floor_smooth: float = 1.0
    lambda_delta_variance_smooth: float = 1.0
    lambda_delta_gain_shrink: float = 0.2
    lambda_delta_floor_shrink: float = 0.2
    lambda_delta_variance_shrink: float = 0.2
    lambda_reliable_sensor_anchor: float = 0.1
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float | None = 10.0
    random_seed: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate optimizer and regularization hyperparameters."""

        if self.n_outer_iterations < 1:
            raise ValueError("n_outer_iterations must be at least 1")
        if self.n_gradient_steps < 1:
            raise ValueError("n_gradient_steps must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive")
        if self.sigma_min <= 0.0:
            raise ValueError("sigma_min must be strictly positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be strictly positive when set")
        for name, value in (
            ("lambda_delta_gain_smooth", self.lambda_delta_gain_smooth),
            ("lambda_delta_floor_smooth", self.lambda_delta_floor_smooth),
            ("lambda_delta_variance_smooth", self.lambda_delta_variance_smooth),
            ("lambda_delta_gain_shrink", self.lambda_delta_gain_shrink),
            ("lambda_delta_floor_shrink", self.lambda_delta_floor_shrink),
            ("lambda_delta_variance_shrink", self.lambda_delta_variance_shrink),
            ("lambda_reliable_sensor_anchor", self.lambda_reliable_sensor_anchor),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if not (0.0 < self.adam_beta1 < 1.0):
            raise ValueError("adam_beta1 must lie in (0, 1)")
        if not (0.0 < self.adam_beta2 < 1.0):
            raise ValueError("adam_beta2 must lie in (0, 1)")
        if self.adam_epsilon <= 0.0:
            raise ValueError("adam_epsilon must be strictly positive")


@dataclass(frozen=True)
class PersistentCalibrationCurves:
    """Deployment-scale persistent calibration curves for one sensor.

    Parameters
    ----------
    sensor_id:
        Sensor identifier used to query the persistent model.
    configuration:
        Deployment acquisition configuration.
    frequency_hz:
        Deployment frequency grid [Hz].
    gain_power:
        Persistent multiplicative gain curve on the deployment reference scale.
    additive_noise_power:
        Persistent additive floor curve in linear power units.
    residual_variance_power2:
        Persistent residual observation variance in squared linear power units.
    """

    sensor_id: str
    configuration: CampaignConfiguration
    frequency_hz: FloatArray
    gain_power: FloatArray
    additive_noise_power: FloatArray
    residual_variance_power2: FloatArray


@dataclass(frozen=True)
class DeploymentCalibrationResult:
    """Deployment-time calibration output for one sensor.

    Parameters
    ----------
    curves:
        Persistent calibration curves evaluated for the deployed sensor.
    calibrated_power:
        Calibrated PSD estimate on the deployment reference scale with the same
        shape as the input observation array.
    propagated_variance_power2:
        First-order propagated observation-noise variance with the same shape
        as ``calibrated_power``.
    """

    curves: PersistentCalibrationCurves
    calibrated_power: FloatArray
    propagated_variance_power2: FloatArray


@dataclass(frozen=True)
class CampaignCalibrationState:
    """Stored offline state for one campaign after fitting.

    Parameters
    ----------
    campaign_label:
        Campaign identifier.
    sensor_ids:
        Ordered sensors present in the campaign.
    frequency_hz:
        Campaign frequency grid [Hz].
    configuration:
        Campaign acquisition configuration.
    reliable_sensor_id:
        Optional reliable-sensor annotation used during fitting.
    latent_spectra_power:
        Estimated same-scene latent spectra with shape
        ``(n_acquisitions, n_frequencies)``.
    persistent_log_gain:
        Persistent deployment-scale log-gain law evaluated on the campaign grid
        for the participating sensors.
    persistent_floor_parameter:
        Persistent additive-floor pre-activation evaluated on the campaign grid.
    persistent_variance_parameter:
        Persistent variance pre-activation evaluated on the campaign grid.
    deviation_log_gain:
        Campaign-specific log-gain deviations with per-frequency mean zero.
    deviation_floor_parameter:
        Campaign-specific additive-floor deviations.
    deviation_variance_parameter:
        Campaign-specific variance deviations.
    gain_power:
        Final campaign gain curves after combining persistent law and
        campaign-specific deviation.
    additive_noise_power:
        Final campaign additive floor curves in linear power units.
    residual_variance_power2:
        Final campaign residual variance curves.
    objective_value:
        Contribution of this campaign to the full offline objective at the end
        of the fit.
    """

    campaign_label: str
    sensor_ids: tuple[str, ...]
    frequency_hz: FloatArray
    configuration: CampaignConfiguration
    reliable_sensor_id: str | None
    latent_spectra_power: FloatArray
    persistent_log_gain: FloatArray
    persistent_floor_parameter: FloatArray
    persistent_variance_parameter: FloatArray
    deviation_log_gain: FloatArray
    deviation_floor_parameter: FloatArray
    deviation_variance_parameter: FloatArray
    gain_power: FloatArray
    additive_noise_power: FloatArray
    residual_variance_power2: FloatArray
    objective_value: float


@dataclass(frozen=True)
class TwoLevelCalibrationResult:
    """Fitted two-level configuration-conditional calibration model.

    Parameters
    ----------
    sensor_ids:
        Global sensor registry used by the persistent laws.
    sensor_reference_weight:
        Positive weights that define the global deployment-scale reference.
    basis_config:
        Frequency-basis configuration used for all three persistent laws.
    model_config:
        Persistent neural-parameterization dimensions.
    fit_config:
        Numerical optimization configuration actually used.
    configuration_feature_mean:
        Mean of the raw configuration feature vectors across the offline corpus.
    configuration_feature_scale:
        Standard deviation of the raw configuration feature vectors. Constant
        features are assigned a scale of one.
    frequency_min_hz, frequency_max_hz:
        Global frequency support used to normalize the spline bases [Hz].
    sensor_embeddings:
        Learned sensor embedding matrix with shape
        ``(n_sensors, sensor_embedding_dim)``.
    configuration_encoder_weight, configuration_encoder_bias:
        Affine map that produces the latent configuration representation before
        the ``tanh`` nonlinearity.
    gain_head_weight, gain_head_bias:
        Linear head that maps sensor/configuration features to the persistent
        log-gain spline coefficients.
    floor_head_weight, floor_head_bias:
        Linear head that maps sensor/configuration features to the additive
        floor pre-activation spline coefficients.
    variance_head_weight, variance_head_bias:
        Linear head that maps sensor/configuration features to the variance
        pre-activation spline coefficients.
    campaign_states:
        Final campaign-level latent states and deviations.
    objective_history:
        Full offline objective after each outer iteration.
    """

    sensor_ids: tuple[str, ...]
    sensor_reference_weight: FloatArray
    basis_config: FrequencyBasisConfig
    model_config: PersistentModelConfig
    fit_config: TwoLevelFitConfig
    configuration_feature_mean: FloatArray
    configuration_feature_scale: FloatArray
    frequency_min_hz: float
    frequency_max_hz: float
    sensor_embeddings: FloatArray
    configuration_encoder_weight: FloatArray
    configuration_encoder_bias: FloatArray
    gain_head_weight: FloatArray
    gain_head_bias: FloatArray
    floor_head_weight: FloatArray
    floor_head_bias: FloatArray
    variance_head_weight: FloatArray
    variance_head_bias: FloatArray
    campaign_states: tuple[CampaignCalibrationState, ...]
    objective_history: FloatArray


@dataclass
class _PersistentModelParameters:
    """Mutable container for the trainable persistent-model parameters."""

    sensor_embeddings: FloatArray
    configuration_encoder_weight: FloatArray
    configuration_encoder_bias: FloatArray
    gain_head_weight: FloatArray
    gain_head_bias: FloatArray
    floor_head_weight: FloatArray
    floor_head_bias: FloatArray
    variance_head_weight: FloatArray
    variance_head_bias: FloatArray

    def as_dict(self) -> dict[str, FloatArray]:
        """Return mutable parameter arrays keyed by stable optimizer names."""

        return {
            "sensor_embeddings": self.sensor_embeddings,
            "configuration_encoder_weight": self.configuration_encoder_weight,
            "configuration_encoder_bias": self.configuration_encoder_bias,
            "gain_head_weight": self.gain_head_weight,
            "gain_head_bias": self.gain_head_bias,
            "floor_head_weight": self.floor_head_weight,
            "floor_head_bias": self.floor_head_bias,
            "variance_head_weight": self.variance_head_weight,
            "variance_head_bias": self.variance_head_bias,
        }


@dataclass
class _CampaignOptimizationState:
    """Mutable training-only state for one campaign during offline fitting."""

    campaign: CalibrationCampaign
    sensor_indices: IndexArray
    standardized_configuration: FloatArray
    gain_basis: FloatArray
    floor_basis: FloatArray
    variance_basis: FloatArray
    second_difference: FloatArray
    reliable_sensor_local_index: int | None
    latent_spectra_power: FloatArray
    delta_log_gain: FloatArray
    delta_floor_parameter: FloatArray
    delta_variance_parameter: FloatArray
    persistent_log_gain: FloatArray
    persistent_floor_parameter: FloatArray
    persistent_variance_parameter: FloatArray
    gain_power: FloatArray
    additive_noise_power: FloatArray
    residual_variance_power2: FloatArray
    objective_value: float = float("nan")


@dataclass(frozen=True)
class _ForwardCache:
    """Cached campaign forward pass used for objective and gradient evaluation."""

    configuration_latent: FloatArray
    combined_features: FloatArray
    raw_log_gain_all: FloatArray
    centered_log_gain_all: FloatArray
    floor_parameter_all: FloatArray
    variance_parameter_all: FloatArray


@dataclass
class _AdamState:
    """Simple in-place Adam optimizer state for NumPy arrays."""

    beta1: float
    beta2: float
    epsilon: float
    first_moment: dict[str, FloatArray]
    second_moment: dict[str, FloatArray]
    step: int = 0

    def update(
        self,
        parameters: Mapping[str, FloatArray],
        gradients: Mapping[str, FloatArray],
        learning_rate: float,
    ) -> None:
        """Apply one Adam step to every parameter array."""

        self.step += 1
        bias_correction1 = 1.0 - self.beta1**self.step
        bias_correction2 = 1.0 - self.beta2**self.step

        for name, parameter in parameters.items():
            gradient = gradients[name]
            first_moment = self.first_moment.setdefault(
                name, np.zeros_like(parameter, dtype=np.float64)
            )
            second_moment = self.second_moment.setdefault(
                name, np.zeros_like(parameter, dtype=np.float64)
            )
            first_moment[:] = self.beta1 * first_moment + (1.0 - self.beta1) * gradient
            second_moment[:] = (
                self.beta2 * second_moment + (1.0 - self.beta2) * (gradient**2)
            )
            unbiased_first = first_moment / bias_correction1
            unbiased_second = second_moment / bias_correction2
            parameter -= learning_rate * unbiased_first / (
                np.sqrt(unbiased_second) + self.epsilon
            )


def power_db_to_linear(
    power_db: FloatArray,  # PSD values expressed in dB-like units
) -> FloatArray:  # Linear power values on the same array shape
    """Convert dB-like power values into linear power."""

    return np.power(10.0, np.asarray(power_db, dtype=np.float64) / 10.0)


def power_linear_to_db(
    power_linear: FloatArray,  # Linear power values
) -> FloatArray:  # dB representation with positivity clipping
    """Convert linear power values to dB with an explicit positive floor."""

    return 10.0 * np.log10(np.clip(np.asarray(power_linear, dtype=np.float64), _EPSILON, None))


def build_calibration_corpus(
    campaigns: Sequence[CalibrationCampaign],  # Campaigns to register in one corpus
) -> CalibrationCorpus:
    """Build a validated corpus while inferring the global sensor registry."""

    if not campaigns:
        raise ValueError("build_calibration_corpus requires at least one campaign")
    sensor_ids = tuple(sorted({sensor_id for campaign in campaigns for sensor_id in campaign.sensor_ids}))
    return CalibrationCorpus(
        sensor_ids=sensor_ids,
        campaigns=tuple(campaigns),
    )


def fit_two_level_calibration(
    corpus: CalibrationCorpus,  # Offline corpus with same-scene campaigns
    basis_config: FrequencyBasisConfig | None = None,
    model_config: PersistentModelConfig | None = None,
    fit_config: TwoLevelFitConfig | None = None,
    sensor_reference_weight_by_id: Mapping[str, float] | None = None,
) -> TwoLevelCalibrationResult:
    """Fit the two-level configuration-conditional calibration model.

    The solver alternates between:

    1. updating the latent same-scene spectra of each campaign by weighted
       least squares;
    2. optimizing the persistent global parameters and campaign-specific
       deviations by gradient descent on the full offline objective.

    The implementation intentionally keeps the persistent law generator light:
    trainable sensor embeddings, an affine-plus-``tanh`` configuration encoder,
    and linear heads that produce cubic-spline coefficients for the gain,
    additive-floor, and variance laws. This matches the structure of the new
    theory while keeping the numerical core dependency-light and fully
    inspectable.
    """

    resolved_basis_config = FrequencyBasisConfig() if basis_config is None else basis_config
    resolved_model_config = (
        PersistentModelConfig() if model_config is None else model_config
    )
    resolved_fit_config = TwoLevelFitConfig() if fit_config is None else fit_config

    sensor_index_by_id = {
        sensor_id: sensor_index for sensor_index, sensor_id in enumerate(corpus.sensor_ids)
    }
    sensor_reference_weight = _resolve_sensor_reference_weight(
        sensor_ids=corpus.sensor_ids,
        sensor_reference_weight_by_id=sensor_reference_weight_by_id,
    )

    raw_configuration_features = np.stack(
        [campaign.configuration.to_feature_vector() for campaign in corpus.campaigns],
        axis=0,
    )
    configuration_feature_mean = np.mean(raw_configuration_features, axis=0)
    configuration_feature_scale = np.std(raw_configuration_features, axis=0)
    configuration_feature_scale = np.where(
        configuration_feature_scale > _EPSILON,
        configuration_feature_scale,
        1.0,
    )

    frequency_min_hz = float(
        min(np.min(campaign.frequency_hz) for campaign in corpus.campaigns)
    )
    frequency_max_hz = float(
        max(np.max(campaign.frequency_hz) for campaign in corpus.campaigns)
    )
    if not math.isfinite(frequency_min_hz) or not math.isfinite(frequency_max_hz):
        raise ValueError("Corpus frequency support contains non-finite values")
    if frequency_max_hz <= frequency_min_hz:
        raise ValueError("Corpus frequency support must span a positive interval")

    parameter_state = _initialize_persistent_parameters(
        corpus=corpus,
        basis_config=resolved_basis_config,
        model_config=resolved_model_config,
        fit_config=resolved_fit_config,
        configuration_feature_mean=configuration_feature_mean,
        configuration_feature_scale=configuration_feature_scale,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )
    campaign_states = _initialize_campaign_states(
        corpus=corpus,
        sensor_index_by_id=sensor_index_by_id,
        basis_config=resolved_basis_config,
        parameter_state=parameter_state,
        sensor_reference_weight=sensor_reference_weight,
        fit_config=resolved_fit_config,
        configuration_feature_mean=configuration_feature_mean,
        configuration_feature_scale=configuration_feature_scale,
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
    )

    optimizer = _AdamState(
        beta1=resolved_fit_config.adam_beta1,
        beta2=resolved_fit_config.adam_beta2,
        epsilon=resolved_fit_config.adam_epsilon,
        first_moment={},
        second_moment={},
    )
    objective_history: list[float] = []

    for _ in range(resolved_fit_config.n_outer_iterations):
        for campaign_state in campaign_states:
            _refresh_campaign_state(
                campaign_state=campaign_state,
                parameter_state=parameter_state,
                sensor_reference_weight=sensor_reference_weight,
                fit_config=resolved_fit_config,
            )
            campaign_state.latent_spectra_power = _update_latent_spectra(
                observations_power=campaign_state.campaign.observations_power,
                gain_power=campaign_state.gain_power,
                additive_noise_power=campaign_state.additive_noise_power,
                residual_variance_power2=campaign_state.residual_variance_power2,
            )

        for _ in range(resolved_fit_config.n_gradient_steps):
            gradients = _zero_gradient_dict(parameter_state=parameter_state, campaign_states=campaign_states)
            objective_value = 0.0

            for campaign_index, campaign_state in enumerate(campaign_states):
                forward_cache = _forward_campaign(
                    parameter_state=parameter_state,
                    campaign_state=campaign_state,
                    sensor_reference_weight=sensor_reference_weight,
                )
                objective_value += _accumulate_campaign_objective_and_gradients(
                    campaign_index=campaign_index,
                    campaign_state=campaign_state,
                    forward_cache=forward_cache,
                    parameter_state=parameter_state,
                    gradients=gradients,
                    model_config=resolved_model_config,
                    fit_config=resolved_fit_config,
                    sensor_reference_weight=sensor_reference_weight,
                )

            if resolved_fit_config.weight_decay > 0.0:
                for name, parameter in parameter_state.as_dict().items():
                    objective_value += resolved_fit_config.weight_decay * float(np.sum(parameter**2))
                    gradients[name] += 2.0 * resolved_fit_config.weight_decay * parameter

            _clip_gradients_in_place(
                gradients=gradients,
                max_norm=resolved_fit_config.gradient_clip_norm,
            )
            optimizer.update(
                parameters=_optimizer_parameter_dict(
                    parameter_state=parameter_state,
                    campaign_states=campaign_states,
                ),
                gradients=gradients,
                learning_rate=resolved_fit_config.learning_rate,
            )

            for campaign_state in campaign_states:
                _project_campaign_log_gain_deviation(campaign_state.delta_log_gain)

        total_objective_value = _refresh_objective_history(
            campaign_states=campaign_states,
            parameter_state=parameter_state,
            sensor_reference_weight=sensor_reference_weight,
            model_config=resolved_model_config,
            fit_config=resolved_fit_config,
            gradients=None,
        )
        objective_history.append(total_objective_value)

    frozen_campaign_states = tuple(
        CampaignCalibrationState(
            campaign_label=campaign_state.campaign.campaign_label,
            sensor_ids=campaign_state.campaign.sensor_ids,
            frequency_hz=np.asarray(campaign_state.campaign.frequency_hz, dtype=np.float64),
            configuration=campaign_state.campaign.configuration,
            reliable_sensor_id=campaign_state.campaign.reliable_sensor_id,
            latent_spectra_power=np.asarray(campaign_state.latent_spectra_power, dtype=np.float64),
            persistent_log_gain=np.asarray(campaign_state.persistent_log_gain, dtype=np.float64),
            persistent_floor_parameter=np.asarray(campaign_state.persistent_floor_parameter, dtype=np.float64),
            persistent_variance_parameter=np.asarray(campaign_state.persistent_variance_parameter, dtype=np.float64),
            deviation_log_gain=np.asarray(campaign_state.delta_log_gain, dtype=np.float64),
            deviation_floor_parameter=np.asarray(campaign_state.delta_floor_parameter, dtype=np.float64),
            deviation_variance_parameter=np.asarray(campaign_state.delta_variance_parameter, dtype=np.float64),
            gain_power=np.asarray(campaign_state.gain_power, dtype=np.float64),
            additive_noise_power=np.asarray(campaign_state.additive_noise_power, dtype=np.float64),
            residual_variance_power2=np.asarray(campaign_state.residual_variance_power2, dtype=np.float64),
            objective_value=float(campaign_state.objective_value),
        )
        for campaign_state in campaign_states
    )

    return TwoLevelCalibrationResult(
        sensor_ids=corpus.sensor_ids,
        sensor_reference_weight=np.asarray(sensor_reference_weight, dtype=np.float64),
        basis_config=resolved_basis_config,
        model_config=resolved_model_config,
        fit_config=resolved_fit_config,
        configuration_feature_mean=np.asarray(configuration_feature_mean, dtype=np.float64),
        configuration_feature_scale=np.asarray(configuration_feature_scale, dtype=np.float64),
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        sensor_embeddings=np.asarray(parameter_state.sensor_embeddings, dtype=np.float64),
        configuration_encoder_weight=np.asarray(
            parameter_state.configuration_encoder_weight, dtype=np.float64
        ),
        configuration_encoder_bias=np.asarray(
            parameter_state.configuration_encoder_bias, dtype=np.float64
        ),
        gain_head_weight=np.asarray(parameter_state.gain_head_weight, dtype=np.float64),
        gain_head_bias=np.asarray(parameter_state.gain_head_bias, dtype=np.float64),
        floor_head_weight=np.asarray(parameter_state.floor_head_weight, dtype=np.float64),
        floor_head_bias=np.asarray(parameter_state.floor_head_bias, dtype=np.float64),
        variance_head_weight=np.asarray(
            parameter_state.variance_head_weight, dtype=np.float64
        ),
        variance_head_bias=np.asarray(parameter_state.variance_head_bias, dtype=np.float64),
        campaign_states=frozen_campaign_states,
        objective_history=np.asarray(objective_history, dtype=np.float64),
    )


def evaluate_persistent_calibration(
    result: TwoLevelCalibrationResult,  # Trained two-level calibration model
    sensor_id: str,  # Sensor identifier to evaluate
    configuration: CampaignConfiguration,  # Deployment configuration
    frequency_hz: FloatArray,  # Deployment frequency grid [Hz]
) -> PersistentCalibrationCurves:
    """Evaluate the persistent deployment laws for one sensor/configuration."""

    sensor_index = _sensor_index(sensor_ids=result.sensor_ids, sensor_id=sensor_id)
    parameter_state = _result_to_parameter_state(result)
    standardized_configuration = _standardize_configuration_vector(
        raw_configuration_vector=configuration.to_feature_vector(),
        feature_mean=result.configuration_feature_mean,
        feature_scale=result.configuration_feature_scale,
    )
    forward_cache = _forward_external_configuration(
        parameter_state=parameter_state,
        standardized_configuration=standardized_configuration,
        sensor_reference_weight=result.sensor_reference_weight,
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        frequency_min_hz=result.frequency_min_hz,
        frequency_max_hz=result.frequency_max_hz,
        basis_config=result.basis_config,
    )

    log_gain = forward_cache.centered_log_gain_all[sensor_index]
    floor_parameter = forward_cache.floor_parameter_all[sensor_index]
    variance_parameter = forward_cache.variance_parameter_all[sensor_index]
    gain_power = np.exp(log_gain)
    additive_noise_power = _softplus(floor_parameter)
    residual_variance_power2 = result.fit_config.sigma_min + _softplus(variance_parameter)
    return PersistentCalibrationCurves(
        sensor_id=sensor_id,
        configuration=configuration,
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        gain_power=np.asarray(gain_power, dtype=np.float64),
        additive_noise_power=np.asarray(additive_noise_power, dtype=np.float64),
        residual_variance_power2=np.asarray(residual_variance_power2, dtype=np.float64),
    )


def apply_deployed_calibration(
    observations_power: FloatArray,  # Observed PSD values with frequency on the last axis
    gain_power: FloatArray,  # Persistent gain curve with shape (n_frequencies,)
    additive_noise_power: FloatArray,  # Persistent additive floor with shape (n_frequencies,)
    enforce_nonnegative: bool = True,  # Whether to apply the [x]_+ truncation
) -> FloatArray:
    """Apply the single-sensor deployment calibration map ``[Y - N]_+ / G``."""

    observations_power = np.asarray(observations_power, dtype=np.float64)
    gain_power = np.asarray(gain_power, dtype=np.float64)
    additive_noise_power = np.asarray(additive_noise_power, dtype=np.float64)

    if gain_power.ndim != 1:
        raise ValueError("gain_power must have shape (n_frequencies,)")
    if additive_noise_power.shape != gain_power.shape:
        raise ValueError(
            "additive_noise_power must have the same shape as gain_power"
        )
    if observations_power.ndim < 1:
        raise ValueError("observations_power must have at least one dimension")
    if observations_power.shape[-1] != gain_power.size:
        raise ValueError(
            "The last observations_power axis must match the gain/floor frequency axis"
        )
    if np.any(gain_power <= 0.0):
        raise ValueError("gain_power must be strictly positive")

    corrected_power = observations_power - additive_noise_power
    if enforce_nonnegative:
        corrected_power = np.clip(corrected_power, 0.0, None)
    return corrected_power / gain_power


def calibrate_sensor_observations(
    result: TwoLevelCalibrationResult,  # Trained two-level calibration model
    sensor_id: str,  # Sensor to calibrate
    configuration: CampaignConfiguration,  # Deployment configuration
    frequency_hz: FloatArray,  # Deployment frequency grid [Hz]
    observations_power: FloatArray,  # Observed PSD values with frequency on the last axis
    enforce_nonnegative: bool = True,  # Whether to apply the [x]_+ truncation
) -> DeploymentCalibrationResult:
    """Evaluate the persistent laws and calibrate one deployed sensor stream."""

    curves = evaluate_persistent_calibration(
        result=result,
        sensor_id=sensor_id,
        configuration=configuration,
        frequency_hz=frequency_hz,
    )
    calibrated_power = apply_deployed_calibration(
        observations_power=observations_power,
        gain_power=curves.gain_power,
        additive_noise_power=curves.additive_noise_power,
        enforce_nonnegative=enforce_nonnegative,
    )
    propagated_variance_power2 = (
        curves.residual_variance_power2 / np.clip(curves.gain_power**2, _EPSILON, None)
    )
    propagated_variance_power2 = np.broadcast_to(
        propagated_variance_power2,
        np.asarray(calibrated_power).shape,
    ).astype(np.float64)
    return DeploymentCalibrationResult(
        curves=curves,
        calibrated_power=np.asarray(calibrated_power, dtype=np.float64),
        propagated_variance_power2=propagated_variance_power2,
    )


def _softplus(
    values: FloatArray,  # Unconstrained real-valued parameter
) -> FloatArray:
    """Map unconstrained values to positive reals with a stable softplus."""

    values = np.asarray(values, dtype=np.float64)
    return np.logaddexp(0.0, values)


def _inverse_softplus(
    positive_values: FloatArray,  # Strictly positive target values
) -> FloatArray:
    """Invert the softplus map for positive inputs."""

    positive_values = np.clip(np.asarray(positive_values, dtype=np.float64), _EPSILON, None)
    large_mask = positive_values > 20.0
    inverse = np.empty_like(positive_values)
    inverse[large_mask] = positive_values[large_mask]
    inverse[~large_mask] = np.log(np.expm1(positive_values[~large_mask]))
    return inverse


def _sigmoid(
    values: FloatArray,  # Unconstrained real-valued parameter
) -> FloatArray:
    """Evaluate the logistic function with overflow-safe branches."""

    values = np.asarray(values, dtype=np.float64)
    positive_mask = values >= 0.0
    negative_mask = ~positive_mask
    result = np.empty_like(values)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(values[negative_mask])
    result[negative_mask] = exp_values / (1.0 + exp_values)
    return result


def _sensor_index(
    sensor_ids: tuple[str, ...],
    sensor_id: str,
) -> int:
    """Resolve one sensor identifier to its integer position."""

    try:
        return sensor_ids.index(sensor_id)
    except ValueError as error:
        raise ValueError(f"Unknown sensor_id {sensor_id!r}") from error


def _resolve_sensor_reference_weight(
    sensor_ids: tuple[str, ...],
    sensor_reference_weight_by_id: Mapping[str, float] | None,
) -> FloatArray:
    """Resolve the global deployment-scale reference weights."""

    if sensor_reference_weight_by_id is None:
        return np.full(len(sensor_ids), 1.0 / len(sensor_ids), dtype=np.float64)

    unknown_sensor_ids = sorted(set(sensor_reference_weight_by_id).difference(sensor_ids))
    if unknown_sensor_ids:
        raise ValueError(
            "sensor_reference_weight_by_id contains unknown sensors: "
            f"{unknown_sensor_ids}"
        )
    weights = np.asarray(
        [
            float(sensor_reference_weight_by_id.get(sensor_id, 0.0))
            for sensor_id in sensor_ids
        ],
        dtype=np.float64,
    )
    if np.any(weights <= 0.0):
        raise ValueError("All sensor reference weights must be strictly positive")
    weight_sum = float(np.sum(weights))
    return weights / weight_sum


def _standardize_configuration_vector(
    raw_configuration_vector: FloatArray,
    feature_mean: FloatArray,
    feature_scale: FloatArray,
) -> FloatArray:
    """Standardize one configuration vector using corpus-level statistics."""

    return (np.asarray(raw_configuration_vector, dtype=np.float64) - feature_mean) / feature_scale


def _build_spline_basis(
    frequency_hz: FloatArray,
    n_basis: int,
    degree: int,
    frequency_min_hz: float,
    frequency_max_hz: float,
) -> FloatArray:
    """Evaluate a normalized clamped B-spline basis on one frequency grid."""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    normalized_frequency = np.clip(
        (frequency_hz - frequency_min_hz) / (frequency_max_hz - frequency_min_hz),
        0.0,
        1.0,
    )
    n_internal_knots = n_basis - degree - 1
    if n_internal_knots > 0:
        internal_knots = np.linspace(0.0, 1.0, n_internal_knots + 2, dtype=np.float64)[1:-1]
    else:
        internal_knots = np.asarray([], dtype=np.float64)
    knots = np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            internal_knots,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )

    basis = np.empty((frequency_hz.size, n_basis), dtype=np.float64)
    for basis_index in range(n_basis):
        coefficients = np.zeros(n_basis, dtype=np.float64)
        coefficients[basis_index] = 1.0
        basis[:, basis_index] = BSpline(
            knots,
            coefficients,
            degree,
            extrapolate=False,
        )(normalized_frequency)

    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = np.sum(basis, axis=1, keepdims=True)
    basis /= np.clip(row_sum, _EPSILON, None)
    return basis


def _second_difference_matrix(
    frequency_hz: FloatArray,
) -> FloatArray:
    """Build the non-uniform second-difference operator used for smoothing."""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    n_frequencies = frequency_hz.size
    if n_frequencies < 3:
        return np.zeros((0, n_frequencies), dtype=np.float64)

    spacing_hz = np.diff(frequency_hz)
    median_spacing_hz = float(np.median(spacing_hz))
    normalized_coordinate = np.concatenate(
        [[0.0], np.cumsum(spacing_hz / max(median_spacing_hz, _EPSILON))]
    )
    second_difference = np.zeros((n_frequencies - 2, n_frequencies), dtype=np.float64)
    for row_index in range(n_frequencies - 2):
        h_minus = normalized_coordinate[row_index + 1] - normalized_coordinate[row_index]
        h_plus = normalized_coordinate[row_index + 2] - normalized_coordinate[row_index + 1]
        second_difference[row_index, row_index] = 2.0 / (h_minus * (h_minus + h_plus))
        second_difference[row_index, row_index + 1] = -2.0 / (h_minus * h_plus)
        second_difference[row_index, row_index + 2] = 2.0 / (h_plus * (h_minus + h_plus))
    return second_difference


def _initialize_persistent_parameters(
    corpus: CalibrationCorpus,
    basis_config: FrequencyBasisConfig,
    model_config: PersistentModelConfig,
    fit_config: TwoLevelFitConfig,
    configuration_feature_mean: FloatArray,
    configuration_feature_scale: FloatArray,
    frequency_min_hz: float,
    frequency_max_hz: float,
) -> _PersistentModelParameters:
    """Initialize trainable persistent parameters with conservative defaults."""

    rng = np.random.default_rng(fit_config.random_seed)
    n_sensors = len(corpus.sensor_ids)
    configuration_dim = len(_CONFIGURATION_FEATURE_NAMES)
    joint_feature_dim = (
        model_config.sensor_embedding_dim + model_config.configuration_latent_dim
    )

    all_observations = np.concatenate(
        [campaign.observations_power.reshape(-1) for campaign in corpus.campaigns]
    )
    global_floor = 0.10 * float(np.quantile(all_observations, 0.05))
    global_floor_parameter = float(_inverse_softplus(np.asarray(global_floor))[()])
    global_variance = max(float(np.var(all_observations)), fit_config.sigma_min * 10.0)
    global_variance_parameter = float(
        _inverse_softplus(np.asarray(global_variance - fit_config.sigma_min))[()]
    )

    return _PersistentModelParameters(
        sensor_embeddings=rng.normal(
            loc=0.0,
            scale=0.05,
            size=(n_sensors, model_config.sensor_embedding_dim),
        ).astype(np.float64),
        configuration_encoder_weight=rng.normal(
            loc=0.0,
            scale=0.05,
            size=(model_config.configuration_latent_dim, configuration_dim),
        ).astype(np.float64),
        configuration_encoder_bias=np.zeros(
            model_config.configuration_latent_dim,
            dtype=np.float64,
        ),
        gain_head_weight=np.zeros(
            (basis_config.n_gain_basis, joint_feature_dim),
            dtype=np.float64,
        ),
        gain_head_bias=np.zeros(basis_config.n_gain_basis, dtype=np.float64),
        floor_head_weight=np.zeros(
            (basis_config.n_floor_basis, joint_feature_dim),
            dtype=np.float64,
        ),
        floor_head_bias=np.full(
            basis_config.n_floor_basis,
            global_floor_parameter,
            dtype=np.float64,
        ),
        variance_head_weight=np.zeros(
            (basis_config.n_variance_basis, joint_feature_dim),
            dtype=np.float64,
        ),
        variance_head_bias=np.full(
            basis_config.n_variance_basis,
            global_variance_parameter,
            dtype=np.float64,
        ),
    )


def _initialize_campaign_states(
    corpus: CalibrationCorpus,
    sensor_index_by_id: Mapping[str, int],
    basis_config: FrequencyBasisConfig,
    parameter_state: _PersistentModelParameters,
    sensor_reference_weight: FloatArray,
    fit_config: TwoLevelFitConfig,
    configuration_feature_mean: FloatArray,
    configuration_feature_scale: FloatArray,
    frequency_min_hz: float,
    frequency_max_hz: float,
) -> list[_CampaignOptimizationState]:
    """Initialize campaign-specific deviations and latent spectra."""

    campaign_states: list[_CampaignOptimizationState] = []

    for campaign in corpus.campaigns:
        sensor_indices = np.asarray(
            [sensor_index_by_id[sensor_id] for sensor_id in campaign.sensor_ids],
            dtype=np.int64,
        )
        standardized_configuration = _standardize_configuration_vector(
            raw_configuration_vector=campaign.configuration.to_feature_vector(),
            feature_mean=configuration_feature_mean,
            feature_scale=configuration_feature_scale,
        )
        gain_basis = _build_spline_basis(
            frequency_hz=campaign.frequency_hz,
            n_basis=basis_config.n_gain_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        )
        floor_basis = _build_spline_basis(
            frequency_hz=campaign.frequency_hz,
            n_basis=basis_config.n_floor_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        )
        variance_basis = _build_spline_basis(
            frequency_hz=campaign.frequency_hz,
            n_basis=basis_config.n_variance_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        )
        second_difference = _second_difference_matrix(campaign.frequency_hz)
        reliable_sensor_local_index = (
            None
            if campaign.reliable_sensor_id is None
            else campaign.sensor_ids.index(campaign.reliable_sensor_id)
        )

        forward_cache = _forward_campaign(
            parameter_state=parameter_state,
            campaign_state=_CampaignOptimizationState(
                campaign=campaign,
                sensor_indices=sensor_indices,
                standardized_configuration=standardized_configuration,
                gain_basis=gain_basis,
                floor_basis=floor_basis,
                variance_basis=variance_basis,
                second_difference=second_difference,
                reliable_sensor_local_index=reliable_sensor_local_index,
                latent_spectra_power=np.empty((campaign.n_acquisitions, campaign.n_frequencies)),
                delta_log_gain=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                delta_floor_parameter=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                delta_variance_parameter=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                persistent_log_gain=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                persistent_floor_parameter=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                persistent_variance_parameter=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                gain_power=np.ones((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                additive_noise_power=np.ones((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                residual_variance_power2=np.ones((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
            ),
            sensor_reference_weight=sensor_reference_weight,
        )

        persistent_log_gain = forward_cache.centered_log_gain_all[sensor_indices]
        persistent_floor_parameter = forward_cache.floor_parameter_all[sensor_indices]
        persistent_variance_parameter = forward_cache.variance_parameter_all[sensor_indices]

        initial_floor_power = 0.10 * np.quantile(campaign.observations_power, 0.05, axis=1)
        delta_floor_parameter = (
            _inverse_softplus(initial_floor_power) - persistent_floor_parameter
        )
        gain_power = np.exp(persistent_log_gain)
        additive_noise_power = _softplus(persistent_floor_parameter + delta_floor_parameter)
        latent_spectra_power = np.clip(
            np.median(
                (campaign.observations_power - additive_noise_power[:, np.newaxis, :])
                / np.clip(gain_power[:, np.newaxis, :], _EPSILON, None),
                axis=0,
            ),
            _EPSILON,
            None,
        )
        residuals = (
            campaign.observations_power
            - gain_power[:, np.newaxis, :] * latent_spectra_power[np.newaxis, :, :]
            - additive_noise_power[:, np.newaxis, :]
        )
        initial_variance_power2 = np.maximum(
            np.mean(residuals**2, axis=1),
            fit_config.sigma_min,
        )
        delta_variance_parameter = (
            _inverse_softplus(initial_variance_power2 - fit_config.sigma_min)
            - persistent_variance_parameter
        )

        campaign_states.append(
            _CampaignOptimizationState(
                campaign=campaign,
                sensor_indices=sensor_indices,
                standardized_configuration=standardized_configuration,
                gain_basis=gain_basis,
                floor_basis=floor_basis,
                variance_basis=variance_basis,
                second_difference=second_difference,
                reliable_sensor_local_index=reliable_sensor_local_index,
                latent_spectra_power=np.asarray(latent_spectra_power, dtype=np.float64),
                delta_log_gain=np.zeros((campaign.n_sensors, campaign.n_frequencies), dtype=np.float64),
                delta_floor_parameter=np.asarray(delta_floor_parameter, dtype=np.float64),
                delta_variance_parameter=np.asarray(delta_variance_parameter, dtype=np.float64),
                persistent_log_gain=np.asarray(persistent_log_gain, dtype=np.float64),
                persistent_floor_parameter=np.asarray(persistent_floor_parameter, dtype=np.float64),
                persistent_variance_parameter=np.asarray(persistent_variance_parameter, dtype=np.float64),
                gain_power=np.asarray(gain_power, dtype=np.float64),
                additive_noise_power=np.asarray(additive_noise_power, dtype=np.float64),
                residual_variance_power2=np.asarray(initial_variance_power2, dtype=np.float64),
            )
        )

    return campaign_states


def _forward_campaign(
    parameter_state: _PersistentModelParameters,
    campaign_state: _CampaignOptimizationState,
    sensor_reference_weight: FloatArray,
) -> _ForwardCache:
    """Evaluate the persistent laws for one campaign configuration."""

    configuration_pre_activation = (
        parameter_state.configuration_encoder_weight @ campaign_state.standardized_configuration
        + parameter_state.configuration_encoder_bias
    )
    configuration_latent = np.tanh(configuration_pre_activation)
    configuration_latent_per_sensor = np.broadcast_to(
        configuration_latent,
        (parameter_state.sensor_embeddings.shape[0], configuration_latent.size),
    )
    combined_features = np.concatenate(
        [parameter_state.sensor_embeddings, configuration_latent_per_sensor],
        axis=1,
    )

    gain_coefficients = combined_features @ parameter_state.gain_head_weight.T + parameter_state.gain_head_bias
    floor_coefficients = combined_features @ parameter_state.floor_head_weight.T + parameter_state.floor_head_bias
    variance_coefficients = (
        combined_features @ parameter_state.variance_head_weight.T
        + parameter_state.variance_head_bias
    )

    raw_log_gain_all = gain_coefficients @ campaign_state.gain_basis.T
    centered_log_gain_all = raw_log_gain_all - np.sum(
        sensor_reference_weight[:, np.newaxis] * raw_log_gain_all,
        axis=0,
        keepdims=True,
    )
    floor_parameter_all = floor_coefficients @ campaign_state.floor_basis.T
    variance_parameter_all = variance_coefficients @ campaign_state.variance_basis.T
    return _ForwardCache(
        configuration_latent=configuration_latent,
        combined_features=combined_features,
        raw_log_gain_all=raw_log_gain_all,
        centered_log_gain_all=centered_log_gain_all,
        floor_parameter_all=floor_parameter_all,
        variance_parameter_all=variance_parameter_all,
    )


def _forward_external_configuration(
    parameter_state: _PersistentModelParameters,
    standardized_configuration: FloatArray,
    sensor_reference_weight: FloatArray,
    frequency_hz: FloatArray,
    frequency_min_hz: float,
    frequency_max_hz: float,
    basis_config: FrequencyBasisConfig,
) -> _ForwardCache:
    """Evaluate the persistent laws on an arbitrary deployment grid."""

    synthetic_campaign_state = _CampaignOptimizationState(
        campaign=CalibrationCampaign(
            campaign_label="deployment",
            sensor_ids=tuple(f"sensor_{index}" for index in range(parameter_state.sensor_embeddings.shape[0])),
            frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
            observations_power=np.ones(
                (
                    parameter_state.sensor_embeddings.shape[0],
                    1,
                    np.asarray(frequency_hz, dtype=np.float64).size,
                ),
                dtype=np.float64,
            ),
            configuration=CampaignConfiguration(
                central_frequency_hz=1.0,
                span_hz=1.0,
                resolution_bandwidth_hz=1.0,
                lna_gain_db=0.0,
                vga_gain_db=0.0,
                acquisition_interval_s=1.0,
                antenna_amplifier_enabled=False,
            ),
        ),
        sensor_indices=np.arange(parameter_state.sensor_embeddings.shape[0], dtype=np.int64),
        standardized_configuration=np.asarray(standardized_configuration, dtype=np.float64),
        gain_basis=_build_spline_basis(
            frequency_hz=frequency_hz,
            n_basis=basis_config.n_gain_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        ),
        floor_basis=_build_spline_basis(
            frequency_hz=frequency_hz,
            n_basis=basis_config.n_floor_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        ),
        variance_basis=_build_spline_basis(
            frequency_hz=frequency_hz,
            n_basis=basis_config.n_variance_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
        ),
        second_difference=np.zeros((0, np.asarray(frequency_hz).size), dtype=np.float64),
        reliable_sensor_local_index=None,
        latent_spectra_power=np.empty((1, np.asarray(frequency_hz).size), dtype=np.float64),
        delta_log_gain=np.zeros((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        delta_floor_parameter=np.zeros((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        delta_variance_parameter=np.zeros((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        persistent_log_gain=np.zeros((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        persistent_floor_parameter=np.zeros((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        persistent_variance_parameter=np.zeros((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        gain_power=np.ones((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        additive_noise_power=np.ones((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
        residual_variance_power2=np.ones((parameter_state.sensor_embeddings.shape[0], np.asarray(frequency_hz).size), dtype=np.float64),
    )
    return _forward_campaign(
        parameter_state=parameter_state,
        campaign_state=synthetic_campaign_state,
        sensor_reference_weight=sensor_reference_weight,
    )


def _update_latent_spectra(
    observations_power: FloatArray,
    gain_power: FloatArray,
    additive_noise_power: FloatArray,
    residual_variance_power2: FloatArray,
) -> FloatArray:
    """Update the same-scene latent spectra by weighted least squares."""

    weights = 1.0 / np.clip(residual_variance_power2, _EPSILON, None)
    numerator = np.sum(
        weights[:, np.newaxis, :]
        * gain_power[:, np.newaxis, :]
        * (observations_power - additive_noise_power[:, np.newaxis, :]),
        axis=0,
    )
    denominator = np.sum(
        weights * gain_power**2,
        axis=0,
        keepdims=True,
    )
    return np.clip(numerator / np.clip(denominator, _EPSILON, None), _EPSILON, None)


def _refresh_campaign_state(
    campaign_state: _CampaignOptimizationState,
    parameter_state: _PersistentModelParameters,
    sensor_reference_weight: FloatArray,
    fit_config: TwoLevelFitConfig,
) -> None:
    """Refresh persistent predictions and physical campaign parameters."""

    forward_cache = _forward_campaign(
        parameter_state=parameter_state,
        campaign_state=campaign_state,
        sensor_reference_weight=sensor_reference_weight,
    )
    campaign_state.persistent_log_gain[:] = forward_cache.centered_log_gain_all[
        campaign_state.sensor_indices
    ]
    campaign_state.persistent_floor_parameter[:] = forward_cache.floor_parameter_all[
        campaign_state.sensor_indices
    ]
    campaign_state.persistent_variance_parameter[:] = forward_cache.variance_parameter_all[
        campaign_state.sensor_indices
    ]
    total_log_gain = campaign_state.persistent_log_gain + campaign_state.delta_log_gain
    total_floor_parameter = (
        campaign_state.persistent_floor_parameter + campaign_state.delta_floor_parameter
    )
    total_variance_parameter = (
        campaign_state.persistent_variance_parameter
        + campaign_state.delta_variance_parameter
    )
    campaign_state.gain_power[:] = np.exp(total_log_gain)
    campaign_state.additive_noise_power[:] = _softplus(total_floor_parameter)
    campaign_state.residual_variance_power2[:] = fit_config.sigma_min + _softplus(
        total_variance_parameter
    )


def _zero_gradient_dict(
    parameter_state: _PersistentModelParameters,
    campaign_states: Sequence[_CampaignOptimizationState],
) -> dict[str, FloatArray]:
    """Allocate a zero-filled gradient dictionary for the full optimizer state."""

    gradients = {
        name: np.zeros_like(parameter, dtype=np.float64)
        for name, parameter in parameter_state.as_dict().items()
    }
    for campaign_index, campaign_state in enumerate(campaign_states):
        gradients[f"campaign_{campaign_index}_delta_log_gain"] = np.zeros_like(
            campaign_state.delta_log_gain,
            dtype=np.float64,
        )
        gradients[f"campaign_{campaign_index}_delta_floor_parameter"] = np.zeros_like(
            campaign_state.delta_floor_parameter,
            dtype=np.float64,
        )
        gradients[
            f"campaign_{campaign_index}_delta_variance_parameter"
        ] = np.zeros_like(campaign_state.delta_variance_parameter, dtype=np.float64)
    return gradients


def _optimizer_parameter_dict(
    parameter_state: _PersistentModelParameters,
    campaign_states: Sequence[_CampaignOptimizationState],
) -> dict[str, FloatArray]:
    """Return the mutable arrays optimized by Adam."""

    parameters = dict(parameter_state.as_dict())
    for campaign_index, campaign_state in enumerate(campaign_states):
        parameters[f"campaign_{campaign_index}_delta_log_gain"] = campaign_state.delta_log_gain
        parameters[
            f"campaign_{campaign_index}_delta_floor_parameter"
        ] = campaign_state.delta_floor_parameter
        parameters[
            f"campaign_{campaign_index}_delta_variance_parameter"
        ] = campaign_state.delta_variance_parameter
    return parameters


def _accumulate_campaign_objective_and_gradients(
    campaign_index: int,
    campaign_state: _CampaignOptimizationState,
    forward_cache: _ForwardCache,
    parameter_state: _PersistentModelParameters,
    gradients: dict[str, FloatArray],
    model_config: PersistentModelConfig,
    fit_config: TwoLevelFitConfig,
    sensor_reference_weight: FloatArray,
) -> float:
    """Accumulate one campaign contribution to the objective and gradients."""

    participant_log_gain = forward_cache.centered_log_gain_all[campaign_state.sensor_indices]
    participant_floor_parameter = forward_cache.floor_parameter_all[campaign_state.sensor_indices]
    participant_variance_parameter = forward_cache.variance_parameter_all[
        campaign_state.sensor_indices
    ]

    total_log_gain = participant_log_gain + campaign_state.delta_log_gain
    total_floor_parameter = participant_floor_parameter + campaign_state.delta_floor_parameter
    total_variance_parameter = (
        participant_variance_parameter + campaign_state.delta_variance_parameter
    )
    gain_power = np.exp(total_log_gain)
    additive_noise_power = _softplus(total_floor_parameter)
    residual_variance_power2 = fit_config.sigma_min + _softplus(total_variance_parameter)
    residuals = (
        campaign_state.campaign.observations_power
        - gain_power[:, np.newaxis, :] * campaign_state.latent_spectra_power[np.newaxis, :, :]
        - additive_noise_power[:, np.newaxis, :]
    )

    nll = 0.5 * float(
        np.sum(
            np.log(residual_variance_power2)[:, np.newaxis, :]
            + residuals**2 / residual_variance_power2[:, np.newaxis, :]
        )
    )
    objective_value = nll

    latent_spectra = campaign_state.latent_spectra_power
    grad_log_gain_total = (
        -gain_power
        * np.sum(
            residuals * latent_spectra[np.newaxis, :, :],
            axis=1,
        )
        / residual_variance_power2
    )
    grad_floor_total = (
        -np.sum(residuals, axis=1) / residual_variance_power2
    ) * _sigmoid(total_floor_parameter)
    grad_variance_total = (
        0.5
        * (
            campaign_state.campaign.n_acquisitions / residual_variance_power2
            - np.sum(residuals**2, axis=1) / residual_variance_power2**2
        )
        * _sigmoid(total_variance_parameter)
    )

    delta_log_gain_key = f"campaign_{campaign_index}_delta_log_gain"
    delta_floor_key = f"campaign_{campaign_index}_delta_floor_parameter"
    delta_variance_key = f"campaign_{campaign_index}_delta_variance_parameter"
    gradients[delta_log_gain_key] += grad_log_gain_total
    gradients[delta_floor_key] += grad_floor_total
    gradients[delta_variance_key] += grad_variance_total

    objective_value += _accumulate_deviation_regularization(
        deviation=campaign_state.delta_log_gain,
        second_difference=campaign_state.second_difference,
        smooth_weight=fit_config.lambda_delta_gain_smooth,
        shrink_weight=fit_config.lambda_delta_gain_shrink,
        gradient=gradients[delta_log_gain_key],
    )
    objective_value += _accumulate_deviation_regularization(
        deviation=campaign_state.delta_floor_parameter,
        second_difference=campaign_state.second_difference,
        smooth_weight=fit_config.lambda_delta_floor_smooth,
        shrink_weight=fit_config.lambda_delta_floor_shrink,
        gradient=gradients[delta_floor_key],
    )
    objective_value += _accumulate_deviation_regularization(
        deviation=campaign_state.delta_variance_parameter,
        second_difference=campaign_state.second_difference,
        smooth_weight=fit_config.lambda_delta_variance_smooth,
        shrink_weight=fit_config.lambda_delta_variance_shrink,
        gradient=gradients[delta_variance_key],
    )

    if (
        campaign_state.reliable_sensor_local_index is not None
        and fit_config.lambda_reliable_sensor_anchor > 0.0
    ):
        reliable_row = campaign_state.delta_log_gain[campaign_state.reliable_sensor_local_index]
        objective_value += fit_config.lambda_reliable_sensor_anchor * float(
            np.sum(reliable_row**2)
        )
        gradients[delta_log_gain_key][campaign_state.reliable_sensor_local_index] += (
            2.0 * fit_config.lambda_reliable_sensor_anchor * reliable_row
        )

    grad_configuration_latent = np.zeros(
        model_config.configuration_latent_dim,
        dtype=np.float64,
    )
    grad_configuration_latent += _accumulate_head_gradients(
        head_weight=parameter_state.gain_head_weight,
        head_bias_name="gain_head_bias",
        head_weight_name="gain_head_weight",
        basis_matrix=campaign_state.gain_basis,
        forward_output_all=forward_cache.raw_log_gain_all,
        combined_features=forward_cache.combined_features,
        output_gradient_all=_expand_centered_log_gain_gradient(
            participant_gradient=grad_log_gain_total,
            participant_indices=campaign_state.sensor_indices,
            sensor_reference_weight=sensor_reference_weight,
            n_sensors=forward_cache.raw_log_gain_all.shape[0],
        ),
        model_config=model_config,
        gradients=gradients,
    )
    grad_configuration_latent += _accumulate_head_gradients(
        head_weight=parameter_state.floor_head_weight,
        head_bias_name="floor_head_bias",
        head_weight_name="floor_head_weight",
        basis_matrix=campaign_state.floor_basis,
        forward_output_all=forward_cache.floor_parameter_all,
        combined_features=forward_cache.combined_features,
        output_gradient_all=_expand_participant_gradient(
            participant_gradient=grad_floor_total,
            participant_indices=campaign_state.sensor_indices,
            n_sensors=forward_cache.floor_parameter_all.shape[0],
        ),
        model_config=model_config,
        gradients=gradients,
    )
    grad_configuration_latent += _accumulate_head_gradients(
        head_weight=parameter_state.variance_head_weight,
        head_bias_name="variance_head_bias",
        head_weight_name="variance_head_weight",
        basis_matrix=campaign_state.variance_basis,
        forward_output_all=forward_cache.variance_parameter_all,
        combined_features=forward_cache.combined_features,
        output_gradient_all=_expand_participant_gradient(
            participant_gradient=grad_variance_total,
            participant_indices=campaign_state.sensor_indices,
            n_sensors=forward_cache.variance_parameter_all.shape[0],
        ),
        model_config=model_config,
        gradients=gradients,
    )
    grad_configuration_pre_activation = (
        grad_configuration_latent * (1.0 - forward_cache.configuration_latent**2)
    )
    gradients["configuration_encoder_weight"] += np.outer(
        grad_configuration_pre_activation,
        campaign_state.standardized_configuration,
    )
    gradients["configuration_encoder_bias"] += grad_configuration_pre_activation
    campaign_state.objective_value = objective_value
    return objective_value


def _accumulate_deviation_regularization(
    deviation: FloatArray,
    second_difference: FloatArray,
    smooth_weight: float,
    shrink_weight: float,
    gradient: FloatArray,
) -> float:
    """Add deviation smoothness and shrinkage to the objective and gradient."""

    objective_value = 0.0
    if smooth_weight > 0.0 and second_difference.size > 0:
        curvature = second_difference @ deviation.T
        objective_value += smooth_weight * float(np.sum(curvature**2))
        gradient += 2.0 * smooth_weight * (second_difference.T @ curvature).T
    if shrink_weight > 0.0:
        objective_value += shrink_weight * float(np.sum(deviation**2))
        gradient += 2.0 * shrink_weight * deviation
    return objective_value


def _expand_centered_log_gain_gradient(
    participant_gradient: FloatArray,
    participant_indices: IndexArray,
    sensor_reference_weight: FloatArray,
    n_sensors: int,
) -> FloatArray:
    """Backpropagate gradients through the global weighted-mean centering."""

    total_gradient = np.sum(participant_gradient, axis=0, keepdims=True)
    expanded_gradient = -sensor_reference_weight[:, np.newaxis] * total_gradient
    expanded_gradient = np.asarray(expanded_gradient, dtype=np.float64)
    expanded_gradient[participant_indices] += participant_gradient
    if expanded_gradient.shape != (n_sensors, participant_gradient.shape[1]):
        raise RuntimeError("Centered log-gain gradient has an unexpected shape")
    return expanded_gradient


def _expand_participant_gradient(
    participant_gradient: FloatArray,
    participant_indices: IndexArray,
    n_sensors: int,
) -> FloatArray:
    """Embed participant-only gradients in the global sensor registry."""

    expanded_gradient = np.zeros((n_sensors, participant_gradient.shape[1]), dtype=np.float64)
    expanded_gradient[participant_indices] = participant_gradient
    return expanded_gradient


def _accumulate_head_gradients(
    head_weight: FloatArray,
    head_bias_name: str,
    head_weight_name: str,
    basis_matrix: FloatArray,
    forward_output_all: FloatArray,
    combined_features: FloatArray,
    output_gradient_all: FloatArray,
    model_config: PersistentModelConfig,
    gradients: dict[str, FloatArray],
) -> FloatArray:
    """Backpropagate one persistent-law head into the shared parameters."""

    del forward_output_all
    coefficient_gradient = output_gradient_all @ basis_matrix
    gradients[head_weight_name] += coefficient_gradient.T @ combined_features
    gradients[head_bias_name] += np.sum(coefficient_gradient, axis=0)

    combined_gradient = coefficient_gradient @ head_weight
    gradients["sensor_embeddings"] += combined_gradient[:, : model_config.sensor_embedding_dim]
    return np.sum(combined_gradient[:, model_config.sensor_embedding_dim :], axis=0)


def _clip_gradients_in_place(
    gradients: dict[str, FloatArray],
    max_norm: float | None,
) -> None:
    """Apply a global gradient-norm cap in place when requested."""

    if max_norm is None:
        return
    squared_norm = 0.0
    for gradient in gradients.values():
        squared_norm += float(np.sum(gradient**2))
    gradient_norm = math.sqrt(squared_norm)
    if gradient_norm <= max_norm or gradient_norm <= _EPSILON:
        return
    scale = max_norm / gradient_norm
    for gradient in gradients.values():
        gradient *= scale


def _project_campaign_log_gain_deviation(
    delta_log_gain: FloatArray,
) -> None:
    """Enforce the per-campaign mean-zero constraint on log-gain deviations."""

    delta_log_gain -= np.mean(delta_log_gain, axis=0, keepdims=True)


def _refresh_objective_history(
    campaign_states: Sequence[_CampaignOptimizationState],
    parameter_state: _PersistentModelParameters,
    sensor_reference_weight: FloatArray,
    model_config: PersistentModelConfig,
    fit_config: TwoLevelFitConfig,
    gradients: dict[str, FloatArray] | None,
) -> float:
    """Recompute the full objective after one outer iteration."""

    objective_value = 0.0
    for campaign_index, campaign_state in enumerate(campaign_states):
        _refresh_campaign_state(
            campaign_state=campaign_state,
            parameter_state=parameter_state,
            sensor_reference_weight=sensor_reference_weight,
            fit_config=fit_config,
        )
        campaign_state.latent_spectra_power = _update_latent_spectra(
            observations_power=campaign_state.campaign.observations_power,
            gain_power=campaign_state.gain_power,
            additive_noise_power=campaign_state.additive_noise_power,
            residual_variance_power2=campaign_state.residual_variance_power2,
        )
        forward_cache = _forward_campaign(
            parameter_state=parameter_state,
            campaign_state=campaign_state,
            sensor_reference_weight=sensor_reference_weight,
        )
        gradient_store = (
            gradients
            if gradients is not None
            else _zero_gradient_dict(parameter_state=parameter_state, campaign_states=campaign_states)
        )
        objective_value += _accumulate_campaign_objective_and_gradients(
            campaign_index=campaign_index,
            campaign_state=campaign_state,
            forward_cache=forward_cache,
            parameter_state=parameter_state,
            gradients=gradient_store,
            model_config=model_config,
            fit_config=fit_config,
            sensor_reference_weight=sensor_reference_weight,
        )
    if fit_config.weight_decay > 0.0:
        for parameter in parameter_state.as_dict().values():
            objective_value += fit_config.weight_decay * float(np.sum(parameter**2))
    return objective_value


def _result_to_parameter_state(
    result: TwoLevelCalibrationResult,
) -> _PersistentModelParameters:
    """Convert an immutable fitted result back into a mutable parameter state."""

    return _PersistentModelParameters(
        sensor_embeddings=np.asarray(result.sensor_embeddings, dtype=np.float64).copy(),
        configuration_encoder_weight=np.asarray(
            result.configuration_encoder_weight, dtype=np.float64
        ).copy(),
        configuration_encoder_bias=np.asarray(
            result.configuration_encoder_bias, dtype=np.float64
        ).copy(),
        gain_head_weight=np.asarray(result.gain_head_weight, dtype=np.float64).copy(),
        gain_head_bias=np.asarray(result.gain_head_bias, dtype=np.float64).copy(),
        floor_head_weight=np.asarray(result.floor_head_weight, dtype=np.float64).copy(),
        floor_head_bias=np.asarray(result.floor_head_bias, dtype=np.float64).copy(),
        variance_head_weight=np.asarray(
            result.variance_head_weight, dtype=np.float64
        ).copy(),
        variance_head_bias=np.asarray(result.variance_head_bias, dtype=np.float64).copy(),
    )


__all__ = [
    "CalibrationCampaign",
    "CalibrationCorpus",
    "CampaignCalibrationState",
    "CampaignConfiguration",
    "DeploymentCalibrationResult",
    "FrequencyBasisConfig",
    "PersistentCalibrationCurves",
    "PersistentModelConfig",
    "TwoLevelCalibrationResult",
    "TwoLevelFitConfig",
    "apply_deployed_calibration",
    "build_calibration_corpus",
    "calibrate_sensor_observations",
    "evaluate_persistent_calibration",
    "fit_two_level_calibration",
    "power_db_to_linear",
    "power_linear_to_db",
]
