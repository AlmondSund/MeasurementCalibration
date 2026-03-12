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
from dataclasses import dataclass, field
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
            raise ValueError(
                "Each calibration campaign must contain at least two sensors"
            )
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
            unknown_sensor_ids = sorted(
                set(campaign.sensor_ids).difference(sensor_id_set)
            )
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
    adaptive_variance_floor_ratio:
        Additional data-scaled floor expressed as a fraction of the corpus
        observation variance. The effective floor becomes
        ``max(sigma_min, adaptive_variance_floor_ratio * var(Y))``.
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
    select_best_outer_iterate:
        Whether to return the best outer iterate by objective value instead of
        the final iterate.
    early_stopping_patience:
        Number of outer iterations without sufficient relative improvement
        before stopping early. ``None`` disables early stopping.
    early_stopping_relative_tolerance:
        Minimum relative objective improvement required to reset the
        early-stopping patience counter.
    divergence_tolerance_ratio:
        Abort the outer loop when the current objective exceeds this multiple
        of the best objective seen so far. ``None`` disables the check.
    refresh_campaign_variance_from_residuals:
        Whether to re-anchor each campaign variance curve from the current
        residuals at the start of every outer iteration. This keeps the
        variance path closer to a generalized-EM update and reduces collapse
        caused by the latent-spectrum/variance coupling.
    variance_refresh_ridge:
        Tikhonov regularization strength used when projecting empirical
        variance curves onto the campaign variance spline basis.
    """

    n_outer_iterations: int = 10
    n_gradient_steps: int = 30
    learning_rate: float = 0.03
    sigma_min: float = 1.0e-10
    adaptive_variance_floor_ratio: float = 1.0e-4
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
    select_best_outer_iterate: bool = True
    early_stopping_patience: int | None = 5
    early_stopping_relative_tolerance: float = 1.0e-4
    divergence_tolerance_ratio: float | None = 10.0
    refresh_campaign_variance_from_residuals: bool = True
    variance_refresh_ridge: float = 1.0e-6

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
        if self.adaptive_variance_floor_ratio < 0.0:
            raise ValueError("adaptive_variance_floor_ratio cannot be negative")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
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
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be at least 1 when it is set"
            )
        if self.early_stopping_relative_tolerance < 0.0:
            raise ValueError("early_stopping_relative_tolerance cannot be negative")
        if (
            self.divergence_tolerance_ratio is not None
            and self.divergence_tolerance_ratio <= 1.0
        ):
            raise ValueError(
                "divergence_tolerance_ratio must be greater than 1 when it is set"
            )
        if self.variance_refresh_ridge < 0.0:
            raise ValueError("variance_refresh_ridge cannot be negative")


@dataclass(frozen=True)
class FitConvergenceDiagnostics:
    """Summary of how the outer optimization loop terminated.

    Parameters
    ----------
    selected_outer_iteration:
        Outer-iteration index whose parameters were selected for the returned
        result.
    n_completed_outer_iterations:
        Number of completed outer iterations stored in ``objective_history``.
    selected_objective_value:
        Objective value at ``selected_outer_iteration``.
    final_objective_value:
        Objective value at the last completed outer iteration.
    terminated_early:
        Whether the outer loop stopped before exhausting
        ``fit_config.n_outer_iterations``.
    termination_reason:
        Concrete stopping reason such as ``completed_all_iterations`` or
        ``early_stopping_no_relative_improvement``.
    selected_from_best_iterate:
        Whether the returned parameters come from an earlier best iterate
        instead of the last iterate.
    max_gradient_norm_by_outer_iteration:
        Largest pre-clipping global gradient norm observed inside each
        completed outer iteration. This is a stability diagnostic, not a
        convergence proof.
    """

    selected_outer_iteration: int
    n_completed_outer_iterations: int
    selected_objective_value: float
    final_objective_value: float
    terminated_early: bool
    termination_reason: str
    selected_from_best_iterate: bool
    max_gradient_norm_by_outer_iteration: tuple[float, ...] = ()


@dataclass(frozen=True)
class DeploymentTrustDiagnostics:
    """Trust-boundary diagnostics for one deployment evaluation.

    Parameters
    ----------
    frequency_support_hz:
        Training frequency envelope ``(min_hz, max_hz)``.
    requested_frequency_hz:
        Requested deployment frequency envelope ``(min_hz, max_hz)``.
    n_frequencies_below_support, n_frequencies_above_support:
        Counts of requested bins outside the training frequency support.
    frequency_extrapolation_detected:
        Whether any requested frequency lies outside the training envelope.
    configuration_support_available:
        Whether exact training min/max envelopes were available for the
        configuration features.
    configuration_out_of_distribution:
        Whether the deployment configuration falls outside the stored training
        envelope on at least one feature.
    out_of_range_feature_names:
        Feature names whose raw values fall outside the training envelope.
    standardized_configuration:
        Full standardized configuration vector consumed by the persistent
        model. This keeps deployment-time OOD inspection auditable instead of
        collapsing the input to one summary statistic.
    max_abs_standardized_feature:
        Largest absolute standardized feature magnitude under the training
        mean/scale normalization.
    """

    frequency_support_hz: tuple[float, float]
    requested_frequency_hz: tuple[float, float]
    n_frequencies_below_support: int
    n_frequencies_above_support: int
    frequency_extrapolation_detected: bool
    configuration_support_available: bool
    configuration_out_of_distribution: bool
    out_of_range_feature_names: tuple[str, ...]
    standardized_configuration: tuple[float, ...]
    max_abs_standardized_feature: float


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
    trust_diagnostics:
        Trust-boundary diagnostics for the requested configuration and
        frequency support.
    """

    sensor_id: str
    configuration: CampaignConfiguration
    frequency_hz: FloatArray
    gain_power: FloatArray
    additive_noise_power: FloatArray
    residual_variance_power2: FloatArray
    trust_diagnostics: DeploymentTrustDiagnostics


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
    uncertainty_scope:
        Explicit scope label for ``propagated_variance_power2``. The current
        implementation propagates only residual observation noise through the
        calibration map; learned parameter uncertainty is not included.
    """

    curves: PersistentCalibrationCurves
    calibrated_power: FloatArray
    propagated_variance_power2: FloatArray
    uncertainty_scope: str = "observation_noise_only"


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
    effective_variance_floor_power2:
        Effective residual-variance floor used during both offline fitting and
        deployment evaluation.
    configuration_feature_min, configuration_feature_max:
        Optional raw-feature envelopes observed during offline training. These
        are used to detect deployment-time configuration OOD conditions.
    fit_diagnostics:
        Summary of best-iterate selection and outer-loop termination.
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
    effective_variance_floor_power2: float | None = None
    configuration_feature_min: FloatArray | None = None
    configuration_feature_max: FloatArray | None = None
    fit_diagnostics: FitConvergenceDiagnostics = field(
        default_factory=lambda: FitConvergenceDiagnostics(
            selected_outer_iteration=0,
            n_completed_outer_iterations=0,
            selected_objective_value=float("nan"),
            final_objective_value=float("nan"),
            terminated_early=False,
            termination_reason="unavailable",
            selected_from_best_iterate=False,
            max_gradient_norm_by_outer_iteration=(),
        )
    )


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
            second_moment[:] = self.beta2 * second_moment + (1.0 - self.beta2) * (
                gradient**2
            )
            unbiased_first = first_moment / bias_correction1
            unbiased_second = second_moment / bias_correction2
            parameter -= (
                learning_rate
                * unbiased_first
                / (np.sqrt(unbiased_second) + self.epsilon)
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

    return 10.0 * np.log10(
        np.clip(np.asarray(power_linear, dtype=np.float64), _EPSILON, None)
    )


def build_calibration_corpus(
    campaigns: Sequence[CalibrationCampaign],  # Campaigns to register in one corpus
) -> CalibrationCorpus:
    """Build a validated corpus while inferring the global sensor registry."""

    if not campaigns:
        raise ValueError("build_calibration_corpus requires at least one campaign")
    sensor_ids = tuple(
        sorted(
            {sensor_id for campaign in campaigns for sensor_id in campaign.sensor_ids}
        )
    )
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

    resolved_basis_config = (
        FrequencyBasisConfig() if basis_config is None else basis_config
    )
    resolved_model_config = (
        PersistentModelConfig() if model_config is None else model_config
    )
    resolved_fit_config = TwoLevelFitConfig() if fit_config is None else fit_config

    sensor_index_by_id = {
        sensor_id: sensor_index
        for sensor_index, sensor_id in enumerate(corpus.sensor_ids)
    }
    sensor_reference_weight = _resolve_sensor_reference_weight(
        sensor_ids=corpus.sensor_ids,
        sensor_reference_weight_by_id=sensor_reference_weight_by_id,
    )

    raw_configuration_features = np.stack(
        [campaign.configuration.to_feature_vector() for campaign in corpus.campaigns],
        axis=0,
    )
    all_observations_power = np.concatenate(
        [campaign.observations_power.reshape(-1) for campaign in corpus.campaigns]
    )
    variance_floor_power2 = _resolve_effective_variance_floor_power2(
        all_observations_power=all_observations_power,
        fit_config=resolved_fit_config,
    )
    configuration_feature_mean = np.mean(raw_configuration_features, axis=0)
    configuration_feature_min = np.min(raw_configuration_features, axis=0)
    configuration_feature_max = np.max(raw_configuration_features, axis=0)
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
        all_observations_power=all_observations_power,
        variance_floor_power2=variance_floor_power2,
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
        variance_floor_power2=variance_floor_power2,
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
    max_gradient_norm_by_outer_iteration: list[float] = []
    best_outer_iteration = -1
    best_objective_value = float("inf")
    best_snapshot: (
        tuple[_PersistentModelParameters, list[_CampaignOptimizationState]] | None
    ) = None
    iterations_without_improvement = 0
    terminated_early = False
    termination_reason = "completed_all_iterations"

    for outer_iteration in range(resolved_fit_config.n_outer_iterations):
        outer_max_gradient_norm = 0.0
        for campaign_state in campaign_states:
            _refresh_campaign_latent_and_variance(
                campaign_state=campaign_state,
                parameter_state=parameter_state,
                sensor_reference_weight=sensor_reference_weight,
                fit_config=resolved_fit_config,
                variance_floor_power2=variance_floor_power2,
            )
            _assert_finite_campaign_state(
                campaign_state=campaign_state,
                context_label=(
                    f"outer_iteration={outer_iteration} latent_and_variance_refresh "
                    f"{campaign_state.campaign.campaign_label}"
                ),
            )

        for gradient_step in range(resolved_fit_config.n_gradient_steps):
            gradients = _zero_gradient_dict(
                parameter_state=parameter_state, campaign_states=campaign_states
            )
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
                    variance_floor_power2=variance_floor_power2,
                    sensor_reference_weight=sensor_reference_weight,
                )
            _assert_finite_scalar(
                objective_value,
                context_label=(
                    f"outer_iteration={outer_iteration} gradient_step={gradient_step} "
                    "objective"
                ),
            )

            if resolved_fit_config.weight_decay > 0.0:
                for name, parameter in parameter_state.as_dict().items():
                    objective_value += resolved_fit_config.weight_decay * float(
                        np.sum(parameter**2)
                    )
                    gradients[name] += (
                        2.0 * resolved_fit_config.weight_decay * parameter
                    )

            _assert_finite_gradients(
                gradients=gradients,
                context_label=(
                    f"outer_iteration={outer_iteration} gradient_step={gradient_step}"
                ),
            )
            outer_max_gradient_norm = max(
                outer_max_gradient_norm,
                _gradient_global_norm(gradients),
            )
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
            _assert_finite_parameter_state(
                parameter_state=parameter_state,
                context_label=(
                    f"outer_iteration={outer_iteration} gradient_step={gradient_step}"
                ),
            )

            for campaign_state in campaign_states:
                _project_campaign_log_gain_deviation(campaign_state.delta_log_gain)
                _assert_finite_array(
                    campaign_state.delta_log_gain,
                    context_label=(
                        f"outer_iteration={outer_iteration} gradient_step={gradient_step} "
                        f"delta_log_gain {campaign_state.campaign.campaign_label}"
                    ),
                )

        total_objective_value = _refresh_objective_history(
            campaign_states=campaign_states,
            parameter_state=parameter_state,
            sensor_reference_weight=sensor_reference_weight,
            model_config=resolved_model_config,
            fit_config=resolved_fit_config,
            variance_floor_power2=variance_floor_power2,
            gradients=None,
        )
        _assert_finite_scalar(
            total_objective_value,
            context_label=f"outer_iteration={outer_iteration} total_objective",
        )
        objective_history.append(total_objective_value)
        max_gradient_norm_by_outer_iteration.append(float(outer_max_gradient_norm))

        improvement_threshold = (
            resolved_fit_config.early_stopping_relative_tolerance
            * max(abs(best_objective_value), 1.0)
        )
        if best_outer_iteration < 0 or (
            total_objective_value < best_objective_value - improvement_threshold
        ):
            best_objective_value = total_objective_value
            best_outer_iteration = outer_iteration
            best_snapshot = _snapshot_training_state(
                parameter_state=parameter_state,
                campaign_states=campaign_states,
            )
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        if (
            resolved_fit_config.divergence_tolerance_ratio is not None
            and best_outer_iteration >= 0
            and total_objective_value
            > resolved_fit_config.divergence_tolerance_ratio
            * max(best_objective_value, 1.0)
        ):
            terminated_early = True
            termination_reason = "diverged_from_best_objective"
            break

        if (
            resolved_fit_config.early_stopping_patience is not None
            and iterations_without_improvement
            >= resolved_fit_config.early_stopping_patience
        ):
            terminated_early = True
            termination_reason = "early_stopping_no_relative_improvement"
            break

    if best_snapshot is None:
        best_outer_iteration = len(objective_history) - 1
        best_objective_value = objective_history[-1]
        best_snapshot = _snapshot_training_state(
            parameter_state=parameter_state,
            campaign_states=campaign_states,
        )

    selected_parameter_state, selected_campaign_states = (
        best_snapshot
        if resolved_fit_config.select_best_outer_iterate
        else _snapshot_training_state(
            parameter_state=parameter_state,
            campaign_states=campaign_states,
        )
    )
    selected_outer_iteration = (
        best_outer_iteration
        if resolved_fit_config.select_best_outer_iterate
        else len(objective_history) - 1
    )
    selected_objective_value = objective_history[selected_outer_iteration]
    fit_diagnostics = FitConvergenceDiagnostics(
        selected_outer_iteration=int(selected_outer_iteration),
        n_completed_outer_iterations=len(objective_history),
        selected_objective_value=float(selected_objective_value),
        final_objective_value=float(objective_history[-1]),
        terminated_early=bool(terminated_early),
        termination_reason=str(termination_reason),
        selected_from_best_iterate=(
            resolved_fit_config.select_best_outer_iterate
            and selected_outer_iteration != len(objective_history) - 1
        ),
        max_gradient_norm_by_outer_iteration=tuple(
            float(value) for value in max_gradient_norm_by_outer_iteration
        ),
    )

    frozen_campaign_states = _freeze_campaign_states(selected_campaign_states)

    return TwoLevelCalibrationResult(
        sensor_ids=corpus.sensor_ids,
        sensor_reference_weight=np.asarray(sensor_reference_weight, dtype=np.float64),
        basis_config=resolved_basis_config,
        model_config=resolved_model_config,
        fit_config=resolved_fit_config,
        configuration_feature_mean=np.asarray(
            configuration_feature_mean, dtype=np.float64
        ),
        configuration_feature_scale=np.asarray(
            configuration_feature_scale, dtype=np.float64
        ),
        configuration_feature_min=np.asarray(
            configuration_feature_min, dtype=np.float64
        ),
        configuration_feature_max=np.asarray(
            configuration_feature_max, dtype=np.float64
        ),
        frequency_min_hz=frequency_min_hz,
        frequency_max_hz=frequency_max_hz,
        sensor_embeddings=np.asarray(
            selected_parameter_state.sensor_embeddings, dtype=np.float64
        ),
        configuration_encoder_weight=np.asarray(
            selected_parameter_state.configuration_encoder_weight, dtype=np.float64
        ),
        configuration_encoder_bias=np.asarray(
            selected_parameter_state.configuration_encoder_bias, dtype=np.float64
        ),
        gain_head_weight=np.asarray(
            selected_parameter_state.gain_head_weight, dtype=np.float64
        ),
        gain_head_bias=np.asarray(
            selected_parameter_state.gain_head_bias, dtype=np.float64
        ),
        floor_head_weight=np.asarray(
            selected_parameter_state.floor_head_weight, dtype=np.float64
        ),
        floor_head_bias=np.asarray(
            selected_parameter_state.floor_head_bias, dtype=np.float64
        ),
        variance_head_weight=np.asarray(
            selected_parameter_state.variance_head_weight, dtype=np.float64
        ),
        variance_head_bias=np.asarray(
            selected_parameter_state.variance_head_bias, dtype=np.float64
        ),
        campaign_states=frozen_campaign_states,
        objective_history=np.asarray(objective_history, dtype=np.float64),
        effective_variance_floor_power2=float(variance_floor_power2),
        fit_diagnostics=fit_diagnostics,
    )


def evaluate_persistent_calibration(
    result: TwoLevelCalibrationResult,  # Trained two-level calibration model
    sensor_id: str,  # Sensor identifier to evaluate
    configuration: CampaignConfiguration,  # Deployment configuration
    frequency_hz: FloatArray,  # Deployment frequency grid [Hz]
    allow_frequency_extrapolation: bool = False,  # Whether clipped extrapolation is allowed
    allow_configuration_ood: bool = True,  # Whether configuration OOD evaluations are allowed
) -> PersistentCalibrationCurves:
    """Evaluate the persistent deployment laws for one sensor/configuration.

    Parameters
    ----------
    allow_frequency_extrapolation:
        Whether deployment frequencies outside the offline support should be
        clipped to the training boundary instead of rejected.
    allow_configuration_ood:
        Whether configurations outside the stored training envelope should be
        evaluated. When ``False``, the function raises on OOD inputs.

    Returns
    -------
    PersistentCalibrationCurves
        Persistent gain, floor, and residual-variance curves evaluated on the
        requested deployment grid, together with trust diagnostics that make
        extrapolation and configuration OOD conditions explicit.
    """

    sensor_index = _sensor_index(sensor_ids=result.sensor_ids, sensor_id=sensor_id)
    raw_configuration_vector = configuration.to_feature_vector()
    parameter_state = _result_to_parameter_state(result)
    standardized_configuration = _standardize_configuration_vector(
        raw_configuration_vector=raw_configuration_vector,
        feature_mean=result.configuration_feature_mean,
        feature_scale=result.configuration_feature_scale,
    )
    trust_diagnostics = _deployment_trust_diagnostics(
        result=result,
        raw_configuration_vector=raw_configuration_vector,
        standardized_configuration=standardized_configuration,
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
    )
    if (
        trust_diagnostics.frequency_extrapolation_detected
        and not allow_frequency_extrapolation
    ):
        raise ValueError(
            "Requested frequency grid lies outside the offline training support. "
            "Pass allow_frequency_extrapolation=True to evaluate with explicit "
            "boundary clamping."
        )
    if (
        trust_diagnostics.configuration_out_of_distribution
        and not allow_configuration_ood
    ):
        raise ValueError(
            "Deployment configuration lies outside the stored training envelope for "
            f"features {trust_diagnostics.out_of_range_feature_names}."
        )

    forward_cache = _forward_external_configuration(
        parameter_state=parameter_state,
        standardized_configuration=standardized_configuration,
        sensor_reference_weight=result.sensor_reference_weight,
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        frequency_min_hz=result.frequency_min_hz,
        frequency_max_hz=result.frequency_max_hz,
        basis_config=result.basis_config,
        clip_to_support=allow_frequency_extrapolation,
    )

    log_gain = forward_cache.centered_log_gain_all[sensor_index]
    floor_parameter = forward_cache.floor_parameter_all[sensor_index]
    variance_parameter = forward_cache.variance_parameter_all[sensor_index]
    gain_power = np.exp(log_gain)
    additive_noise_power = _softplus(floor_parameter)
    residual_variance_power2 = _result_variance_floor_power2(result) + _softplus(
        variance_parameter
    )
    return PersistentCalibrationCurves(
        sensor_id=sensor_id,
        configuration=configuration,
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        gain_power=np.asarray(gain_power, dtype=np.float64),
        additive_noise_power=np.asarray(additive_noise_power, dtype=np.float64),
        residual_variance_power2=np.asarray(residual_variance_power2, dtype=np.float64),
        trust_diagnostics=trust_diagnostics,
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
        raise ValueError("additive_noise_power must have the same shape as gain_power")
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
    allow_frequency_extrapolation: bool = False,  # Whether clipped extrapolation is allowed
    allow_configuration_ood: bool = True,  # Whether configuration OOD evaluations are allowed
) -> DeploymentCalibrationResult:
    """Evaluate the persistent laws and calibrate one deployed sensor stream.

    The returned variance is a first-order observation-noise propagation only.
    Learned parameter uncertainty is intentionally not modeled here, so
    callers should treat ``propagated_variance_power2`` as a partial
    uncertainty estimate.
    """

    curves = evaluate_persistent_calibration(
        result=result,
        sensor_id=sensor_id,
        configuration=configuration,
        frequency_hz=frequency_hz,
        allow_frequency_extrapolation=allow_frequency_extrapolation,
        allow_configuration_ood=allow_configuration_ood,
    )
    calibrated_power = apply_deployed_calibration(
        observations_power=observations_power,
        gain_power=curves.gain_power,
        additive_noise_power=curves.additive_noise_power,
        enforce_nonnegative=enforce_nonnegative,
    )
    propagated_variance_power2 = curves.residual_variance_power2 / np.clip(
        curves.gain_power**2, _EPSILON, None
    )
    propagated_variance_power2 = np.broadcast_to(
        propagated_variance_power2,
        np.asarray(calibrated_power).shape,
    ).astype(np.float64)
    return DeploymentCalibrationResult(
        curves=curves,
        calibrated_power=np.asarray(calibrated_power, dtype=np.float64),
        propagated_variance_power2=propagated_variance_power2,
        uncertainty_scope="observation_noise_only",
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

    positive_values = np.clip(
        np.asarray(positive_values, dtype=np.float64), _EPSILON, None
    )
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

    unknown_sensor_ids = sorted(
        set(sensor_reference_weight_by_id).difference(sensor_ids)
    )
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

    return (
        np.asarray(raw_configuration_vector, dtype=np.float64) - feature_mean
    ) / feature_scale


def _resolve_effective_variance_floor_power2(
    all_observations_power: FloatArray,
    fit_config: TwoLevelFitConfig,
) -> float:
    """Resolve the residual-variance floor used by the offline solver.

    The absolute ``sigma_min`` prevents singular log-likelihood evaluations.
    The adaptive floor keeps the variance branch tied to the observed power
    scale so the alternating latent/variance updates cannot collapse onto an
    unrealistically tiny noise level.
    """

    all_observations_power = np.asarray(all_observations_power, dtype=np.float64)
    if all_observations_power.size == 0:
        raise ValueError("Calibration corpus must contain at least one observation")
    if not np.all(np.isfinite(all_observations_power)):
        raise ValueError("Calibration corpus observations contain non-finite values")

    global_observation_variance = float(np.var(all_observations_power))
    adaptive_floor_power2 = (
        fit_config.adaptive_variance_floor_ratio * global_observation_variance
    )
    return max(fit_config.sigma_min, adaptive_floor_power2)


def _result_variance_floor_power2(
    result: TwoLevelCalibrationResult,
) -> float:
    """Return the residual-variance floor encoded by one fitted result."""

    if result.effective_variance_floor_power2 is not None:
        return float(result.effective_variance_floor_power2)
    return float(result.fit_config.sigma_min)


def _deployment_trust_diagnostics(
    result: TwoLevelCalibrationResult,
    raw_configuration_vector: FloatArray,
    standardized_configuration: FloatArray,
    frequency_hz: FloatArray,
) -> DeploymentTrustDiagnostics:
    """Build deployment-time trust diagnostics from stored training support."""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    raw_configuration_vector = np.asarray(raw_configuration_vector, dtype=np.float64)
    standardized_configuration = np.asarray(
        standardized_configuration, dtype=np.float64
    )
    if frequency_hz.size < 1:
        raise ValueError("frequency_hz must contain at least one deployment bin")

    n_frequencies_below_support = int(np.sum(frequency_hz < result.frequency_min_hz))
    n_frequencies_above_support = int(np.sum(frequency_hz > result.frequency_max_hz))
    frequency_extrapolation_detected = (
        n_frequencies_below_support > 0 or n_frequencies_above_support > 0
    )

    out_of_range_feature_names: list[str] = []
    configuration_support_available = (
        result.configuration_feature_min is not None
        and result.configuration_feature_max is not None
    )
    configuration_out_of_distribution = False
    if configuration_support_available:
        feature_min = np.asarray(result.configuration_feature_min, dtype=np.float64)
        feature_max = np.asarray(result.configuration_feature_max, dtype=np.float64)
        out_of_range_mask = np.logical_or(
            raw_configuration_vector < feature_min,
            raw_configuration_vector > feature_max,
        )
        out_of_range_feature_names = [
            feature_name
            for feature_name, is_out_of_range in zip(
                _CONFIGURATION_FEATURE_NAMES,
                out_of_range_mask,
                strict=True,
            )
            if bool(is_out_of_range)
        ]
        configuration_out_of_distribution = bool(np.any(out_of_range_mask))

    return DeploymentTrustDiagnostics(
        frequency_support_hz=(
            float(result.frequency_min_hz),
            float(result.frequency_max_hz),
        ),
        requested_frequency_hz=(
            float(np.min(frequency_hz)),
            float(np.max(frequency_hz)),
        ),
        n_frequencies_below_support=n_frequencies_below_support,
        n_frequencies_above_support=n_frequencies_above_support,
        frequency_extrapolation_detected=frequency_extrapolation_detected,
        configuration_support_available=configuration_support_available,
        configuration_out_of_distribution=configuration_out_of_distribution,
        out_of_range_feature_names=tuple(out_of_range_feature_names),
        standardized_configuration=tuple(
            float(value) for value in standardized_configuration
        ),
        max_abs_standardized_feature=float(np.max(np.abs(standardized_configuration))),
    )


def _build_spline_basis(
    frequency_hz: FloatArray,
    n_basis: int,
    degree: int,
    frequency_min_hz: float,
    frequency_max_hz: float,
    clip_to_support: bool = True,
) -> FloatArray:
    """Evaluate a normalized clamped B-spline basis on one frequency grid."""

    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(frequency_hz < frequency_min_hz) or np.any(
        frequency_hz > frequency_max_hz
    ):
        if not clip_to_support:
            raise ValueError(
                "frequency_hz lies outside the spline support and clip_to_support "
                "is disabled"
            )
    normalized_frequency = np.clip(
        (frequency_hz - frequency_min_hz) / (frequency_max_hz - frequency_min_hz),
        0.0,
        1.0,
    )
    n_internal_knots = n_basis - degree - 1
    if n_internal_knots > 0:
        internal_knots = np.linspace(0.0, 1.0, n_internal_knots + 2, dtype=np.float64)[
            1:-1
        ]
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
        h_minus = (
            normalized_coordinate[row_index + 1] - normalized_coordinate[row_index]
        )
        h_plus = (
            normalized_coordinate[row_index + 2] - normalized_coordinate[row_index + 1]
        )
        second_difference[row_index, row_index] = 2.0 / (h_minus * (h_minus + h_plus))
        second_difference[row_index, row_index + 1] = -2.0 / (h_minus * h_plus)
        second_difference[row_index, row_index + 2] = 2.0 / (
            h_plus * (h_minus + h_plus)
        )
    return second_difference


def _initialize_persistent_parameters(
    corpus: CalibrationCorpus,
    basis_config: FrequencyBasisConfig,
    model_config: PersistentModelConfig,
    fit_config: TwoLevelFitConfig,
    all_observations_power: FloatArray,
    variance_floor_power2: float,
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

    global_floor = 0.10 * float(np.quantile(all_observations_power, 0.05))
    global_floor_parameter = float(_inverse_softplus(np.asarray(global_floor))[()])
    global_variance = max(
        float(np.var(all_observations_power)),
        variance_floor_power2 * 10.0,
    )
    global_variance_parameter = float(
        _inverse_softplus(np.asarray(global_variance - variance_floor_power2))[()]
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
    variance_floor_power2: float,
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
                latent_spectra_power=np.empty(
                    (campaign.n_acquisitions, campaign.n_frequencies)
                ),
                delta_log_gain=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                delta_floor_parameter=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                delta_variance_parameter=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                persistent_log_gain=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                persistent_floor_parameter=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                persistent_variance_parameter=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                gain_power=np.ones(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                additive_noise_power=np.ones(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                residual_variance_power2=np.ones(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
            ),
            sensor_reference_weight=sensor_reference_weight,
        )

        persistent_log_gain = forward_cache.centered_log_gain_all[sensor_indices]
        persistent_floor_parameter = forward_cache.floor_parameter_all[sensor_indices]
        persistent_variance_parameter = forward_cache.variance_parameter_all[
            sensor_indices
        ]

        initial_floor_power = 0.10 * np.quantile(
            campaign.observations_power, 0.05, axis=1
        )
        delta_floor_parameter = (
            _inverse_softplus(initial_floor_power) - persistent_floor_parameter
        )
        gain_power = np.exp(persistent_log_gain)
        additive_noise_power = _softplus(
            persistent_floor_parameter + delta_floor_parameter
        )
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
            variance_floor_power2,
        )
        initial_variance_parameter = _inverse_softplus(
            np.maximum(initial_variance_power2 - variance_floor_power2, _EPSILON)
        )
        if fit_config.refresh_campaign_variance_from_residuals:
            initial_variance_parameter = _project_curves_onto_basis(
                curves=initial_variance_parameter,
                basis=variance_basis,
                ridge=fit_config.variance_refresh_ridge,
            )
            initial_variance_power2 = variance_floor_power2 + _softplus(
                initial_variance_parameter
            )
        delta_variance_parameter = (
            initial_variance_parameter - persistent_variance_parameter
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
                delta_log_gain=np.zeros(
                    (campaign.n_sensors, campaign.n_frequencies), dtype=np.float64
                ),
                delta_floor_parameter=np.asarray(
                    delta_floor_parameter, dtype=np.float64
                ),
                delta_variance_parameter=np.asarray(
                    delta_variance_parameter, dtype=np.float64
                ),
                persistent_log_gain=np.asarray(persistent_log_gain, dtype=np.float64),
                persistent_floor_parameter=np.asarray(
                    persistent_floor_parameter, dtype=np.float64
                ),
                persistent_variance_parameter=np.asarray(
                    persistent_variance_parameter, dtype=np.float64
                ),
                gain_power=np.asarray(gain_power, dtype=np.float64),
                additive_noise_power=np.asarray(additive_noise_power, dtype=np.float64),
                residual_variance_power2=np.asarray(
                    initial_variance_power2, dtype=np.float64
                ),
            )
        )

    return campaign_states


def _forward_campaign(
    parameter_state: _PersistentModelParameters,
    campaign_state: _CampaignOptimizationState,
    sensor_reference_weight: FloatArray,
) -> _ForwardCache:
    """Evaluate the persistent laws for one campaign configuration."""

    return _forward_persistent_laws(
        parameter_state=parameter_state,
        standardized_configuration=campaign_state.standardized_configuration,
        sensor_reference_weight=sensor_reference_weight,
        gain_basis=campaign_state.gain_basis,
        floor_basis=campaign_state.floor_basis,
        variance_basis=campaign_state.variance_basis,
    )


def _forward_persistent_laws(
    parameter_state: _PersistentModelParameters,
    standardized_configuration: FloatArray,
    sensor_reference_weight: FloatArray,
    gain_basis: FloatArray,
    floor_basis: FloatArray,
    variance_basis: FloatArray,
) -> _ForwardCache:
    """Evaluate persistent laws on arbitrary basis matrices without campaign glue."""

    configuration_pre_activation = (
        parameter_state.configuration_encoder_weight @ standardized_configuration
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

    gain_coefficients = (
        combined_features @ parameter_state.gain_head_weight.T
        + parameter_state.gain_head_bias
    )
    floor_coefficients = (
        combined_features @ parameter_state.floor_head_weight.T
        + parameter_state.floor_head_bias
    )
    variance_coefficients = (
        combined_features @ parameter_state.variance_head_weight.T
        + parameter_state.variance_head_bias
    )

    raw_log_gain_all = gain_coefficients @ gain_basis.T
    centered_log_gain_all = raw_log_gain_all - np.sum(
        sensor_reference_weight[:, np.newaxis] * raw_log_gain_all,
        axis=0,
        keepdims=True,
    )
    floor_parameter_all = floor_coefficients @ floor_basis.T
    variance_parameter_all = variance_coefficients @ variance_basis.T
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
    clip_to_support: bool = True,
) -> _ForwardCache:
    """Evaluate the persistent laws on an arbitrary deployment grid.

    This helper is the deployment-side counterpart to ``_forward_campaign``:
    it builds only the spline bases required by the requested frequency grid,
    then reuses the shared persistent-law evaluator without inventing a dummy
    campaign object.
    """

    return _forward_persistent_laws(
        parameter_state=parameter_state,
        standardized_configuration=np.asarray(
            standardized_configuration, dtype=np.float64
        ),
        sensor_reference_weight=sensor_reference_weight,
        gain_basis=_build_spline_basis(
            frequency_hz=frequency_hz,
            n_basis=basis_config.n_gain_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
            clip_to_support=clip_to_support,
        ),
        floor_basis=_build_spline_basis(
            frequency_hz=frequency_hz,
            n_basis=basis_config.n_floor_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
            clip_to_support=clip_to_support,
        ),
        variance_basis=_build_spline_basis(
            frequency_hz=frequency_hz,
            n_basis=basis_config.n_variance_basis,
            degree=basis_config.spline_degree,
            frequency_min_hz=frequency_min_hz,
            frequency_max_hz=frequency_max_hz,
            clip_to_support=clip_to_support,
        ),
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


def _project_curves_onto_basis(
    curves: FloatArray,
    basis: FloatArray,
    ridge: float,
) -> FloatArray:
    """Project one or more frequency curves onto a spline basis.

    Parameters
    ----------
    curves:
        Array with shape ``(n_rows, n_frequencies)`` containing the values to
        smooth and compress onto the supplied basis.
    basis:
        Basis matrix with shape ``(n_frequencies, n_basis)``.
    ridge:
        Non-negative Tikhonov regularization applied in coefficient space.

    Returns
    -------
    np.ndarray
        Basis-projected curves with the same shape as ``curves``.
    """

    curves = np.asarray(curves, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)

    gram_matrix = basis.T @ basis
    if ridge > 0.0:
        gram_matrix = gram_matrix + ridge * np.eye(
            gram_matrix.shape[0], dtype=np.float64
        )
    right_hand_side = basis.T @ curves.T
    try:
        coefficient_matrix = np.linalg.solve(gram_matrix, right_hand_side)
    except np.linalg.LinAlgError:
        coefficient_matrix = np.linalg.pinv(gram_matrix) @ right_hand_side
    return np.asarray((basis @ coefficient_matrix).T, dtype=np.float64)


def _campaign_residuals(
    campaign_state: _CampaignOptimizationState,
) -> FloatArray:
    """Return campaign observation residuals for the current latent/state pair."""

    return (
        campaign_state.campaign.observations_power
        - campaign_state.gain_power[:, np.newaxis, :]
        * campaign_state.latent_spectra_power[np.newaxis, :, :]
        - campaign_state.additive_noise_power[:, np.newaxis, :]
    )


def _refresh_campaign_variance_from_residuals(
    campaign_state: _CampaignOptimizationState,
    fit_config: TwoLevelFitConfig,
    variance_floor_power2: float,
) -> None:
    """Re-estimate campaign variance curves from residual energy.

    The latent-spectrum update can otherwise drive some variance channels
    unrealistically low because the same residual variance is also used as the
    weighting measure for the latent least-squares solve. Re-estimating the
    variance from the residuals and projecting it back to the variance spline
    basis keeps the campaign-specific variance path smooth and bounded without
    adding another heavy optimizer block.
    """

    empirical_variance_power2 = np.maximum(
        np.mean(_campaign_residuals(campaign_state) ** 2, axis=1),
        variance_floor_power2,
    )
    empirical_variance_parameter = _inverse_softplus(
        np.maximum(empirical_variance_power2 - variance_floor_power2, _EPSILON)
    )
    projected_variance_parameter = _project_curves_onto_basis(
        curves=empirical_variance_parameter,
        basis=campaign_state.variance_basis,
        ridge=fit_config.variance_refresh_ridge,
    )

    campaign_state.delta_variance_parameter[:] = (
        projected_variance_parameter - campaign_state.persistent_variance_parameter
    )
    campaign_state.residual_variance_power2[:] = variance_floor_power2 + _softplus(
        projected_variance_parameter
    )


def _refresh_campaign_latent_and_variance(
    campaign_state: _CampaignOptimizationState,
    parameter_state: _PersistentModelParameters,
    sensor_reference_weight: FloatArray,
    fit_config: TwoLevelFitConfig,
    variance_floor_power2: float,
) -> None:
    """Refresh the physical campaign state before one optimizer block.

    This helper keeps the block-alternating schedule explicit: first update the
    persistent campaign view, then refresh the same-scene latent spectra, and
    finally re-anchor the campaign variance curve from the resulting residuals
    when that safeguard is enabled.
    """

    _refresh_campaign_state(
        campaign_state=campaign_state,
        parameter_state=parameter_state,
        sensor_reference_weight=sensor_reference_weight,
        fit_config=fit_config,
        variance_floor_power2=variance_floor_power2,
    )
    campaign_state.latent_spectra_power = _update_latent_spectra(
        observations_power=campaign_state.campaign.observations_power,
        gain_power=campaign_state.gain_power,
        additive_noise_power=campaign_state.additive_noise_power,
        residual_variance_power2=campaign_state.residual_variance_power2,
    )
    if fit_config.refresh_campaign_variance_from_residuals:
        _refresh_campaign_variance_from_residuals(
            campaign_state=campaign_state,
            fit_config=fit_config,
            variance_floor_power2=variance_floor_power2,
        )


def _refresh_campaign_state(
    campaign_state: _CampaignOptimizationState,
    parameter_state: _PersistentModelParameters,
    sensor_reference_weight: FloatArray,
    fit_config: TwoLevelFitConfig,
    variance_floor_power2: float,
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
    campaign_state.persistent_variance_parameter[:] = (
        forward_cache.variance_parameter_all[campaign_state.sensor_indices]
    )
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
    campaign_state.residual_variance_power2[:] = variance_floor_power2 + _softplus(
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
        gradients[f"campaign_{campaign_index}_delta_variance_parameter"] = (
            np.zeros_like(campaign_state.delta_variance_parameter, dtype=np.float64)
        )
    return gradients


def _optimizer_parameter_dict(
    parameter_state: _PersistentModelParameters,
    campaign_states: Sequence[_CampaignOptimizationState],
) -> dict[str, FloatArray]:
    """Return the mutable arrays optimized by Adam."""

    parameters = dict(parameter_state.as_dict())
    for campaign_index, campaign_state in enumerate(campaign_states):
        parameters[f"campaign_{campaign_index}_delta_log_gain"] = (
            campaign_state.delta_log_gain
        )
        parameters[f"campaign_{campaign_index}_delta_floor_parameter"] = (
            campaign_state.delta_floor_parameter
        )
        parameters[f"campaign_{campaign_index}_delta_variance_parameter"] = (
            campaign_state.delta_variance_parameter
        )
    return parameters


def _accumulate_campaign_objective_and_gradients(
    campaign_index: int,
    campaign_state: _CampaignOptimizationState,
    forward_cache: _ForwardCache,
    parameter_state: _PersistentModelParameters,
    gradients: dict[str, FloatArray],
    model_config: PersistentModelConfig,
    fit_config: TwoLevelFitConfig,
    variance_floor_power2: float,
    sensor_reference_weight: FloatArray,
) -> float:
    """Accumulate one campaign contribution to the objective and gradients."""

    participant_log_gain = forward_cache.centered_log_gain_all[
        campaign_state.sensor_indices
    ]
    participant_floor_parameter = forward_cache.floor_parameter_all[
        campaign_state.sensor_indices
    ]
    participant_variance_parameter = forward_cache.variance_parameter_all[
        campaign_state.sensor_indices
    ]

    total_log_gain = participant_log_gain + campaign_state.delta_log_gain
    total_floor_parameter = (
        participant_floor_parameter + campaign_state.delta_floor_parameter
    )
    total_variance_parameter = (
        participant_variance_parameter + campaign_state.delta_variance_parameter
    )
    gain_power = np.exp(total_log_gain)
    additive_noise_power = _softplus(total_floor_parameter)
    residual_variance_power2 = variance_floor_power2 + _softplus(
        total_variance_parameter
    )
    residuals = (
        campaign_state.campaign.observations_power
        - gain_power[:, np.newaxis, :]
        * campaign_state.latent_spectra_power[np.newaxis, :, :]
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
        reliable_row = campaign_state.delta_log_gain[
            campaign_state.reliable_sensor_local_index
        ]
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
    grad_configuration_pre_activation = grad_configuration_latent * (
        1.0 - forward_cache.configuration_latent**2
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

    expanded_gradient = np.zeros(
        (n_sensors, participant_gradient.shape[1]), dtype=np.float64
    )
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
    gradients["sensor_embeddings"] += combined_gradient[
        :, : model_config.sensor_embedding_dim
    ]
    return np.sum(combined_gradient[:, model_config.sensor_embedding_dim :], axis=0)


def _clip_gradients_in_place(
    gradients: dict[str, FloatArray],
    max_norm: float | None,
) -> None:
    """Apply a global gradient-norm cap in place when requested."""

    if max_norm is None:
        return
    gradient_norm = _gradient_global_norm(gradients)
    if gradient_norm <= max_norm or gradient_norm <= _EPSILON:
        return
    scale = max_norm / gradient_norm
    for gradient in gradients.values():
        gradient *= scale


def _gradient_global_norm(
    gradients: Mapping[str, FloatArray],
) -> float:
    """Return the global Euclidean norm of the current gradient dictionary."""

    squared_norm = 0.0
    for gradient in gradients.values():
        squared_norm += float(np.sum(np.asarray(gradient, dtype=np.float64) ** 2))
    gradient_norm = math.sqrt(squared_norm)
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("Global gradient norm became non-finite")
    return float(gradient_norm)


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
    variance_floor_power2: float,
    gradients: dict[str, FloatArray] | None,
) -> float:
    """Recompute the full objective after one outer iteration."""

    objective_value = 0.0
    for campaign_index, campaign_state in enumerate(campaign_states):
        _refresh_campaign_latent_and_variance(
            campaign_state=campaign_state,
            parameter_state=parameter_state,
            sensor_reference_weight=sensor_reference_weight,
            fit_config=fit_config,
            variance_floor_power2=variance_floor_power2,
        )
        forward_cache = _forward_campaign(
            parameter_state=parameter_state,
            campaign_state=campaign_state,
            sensor_reference_weight=sensor_reference_weight,
        )
        gradient_store = (
            gradients
            if gradients is not None
            else _zero_gradient_dict(
                parameter_state=parameter_state, campaign_states=campaign_states
            )
        )
        objective_value += _accumulate_campaign_objective_and_gradients(
            campaign_index=campaign_index,
            campaign_state=campaign_state,
            forward_cache=forward_cache,
            parameter_state=parameter_state,
            gradients=gradient_store,
            model_config=model_config,
            fit_config=fit_config,
            variance_floor_power2=variance_floor_power2,
            sensor_reference_weight=sensor_reference_weight,
        )
    if fit_config.weight_decay > 0.0:
        for parameter in parameter_state.as_dict().values():
            objective_value += fit_config.weight_decay * float(np.sum(parameter**2))
    return objective_value


def _freeze_campaign_states(
    campaign_states: Sequence[_CampaignOptimizationState],
) -> tuple[CampaignCalibrationState, ...]:
    """Freeze mutable campaign states into the immutable result payload."""

    return tuple(
        CampaignCalibrationState(
            campaign_label=campaign_state.campaign.campaign_label,
            sensor_ids=campaign_state.campaign.sensor_ids,
            frequency_hz=np.asarray(
                campaign_state.campaign.frequency_hz, dtype=np.float64
            ),
            configuration=campaign_state.campaign.configuration,
            reliable_sensor_id=campaign_state.campaign.reliable_sensor_id,
            latent_spectra_power=np.asarray(
                campaign_state.latent_spectra_power, dtype=np.float64
            ),
            persistent_log_gain=np.asarray(
                campaign_state.persistent_log_gain, dtype=np.float64
            ),
            persistent_floor_parameter=np.asarray(
                campaign_state.persistent_floor_parameter, dtype=np.float64
            ),
            persistent_variance_parameter=np.asarray(
                campaign_state.persistent_variance_parameter, dtype=np.float64
            ),
            deviation_log_gain=np.asarray(
                campaign_state.delta_log_gain, dtype=np.float64
            ),
            deviation_floor_parameter=np.asarray(
                campaign_state.delta_floor_parameter, dtype=np.float64
            ),
            deviation_variance_parameter=np.asarray(
                campaign_state.delta_variance_parameter, dtype=np.float64
            ),
            gain_power=np.asarray(campaign_state.gain_power, dtype=np.float64),
            additive_noise_power=np.asarray(
                campaign_state.additive_noise_power, dtype=np.float64
            ),
            residual_variance_power2=np.asarray(
                campaign_state.residual_variance_power2, dtype=np.float64
            ),
            objective_value=float(campaign_state.objective_value),
        )
        for campaign_state in campaign_states
    )


def _copy_parameter_state(
    parameter_state: _PersistentModelParameters,
) -> _PersistentModelParameters:
    """Deep-copy the mutable persistent parameter state for checkpointing."""

    return _PersistentModelParameters(
        sensor_embeddings=np.asarray(
            parameter_state.sensor_embeddings, dtype=np.float64
        ).copy(),
        configuration_encoder_weight=np.asarray(
            parameter_state.configuration_encoder_weight, dtype=np.float64
        ).copy(),
        configuration_encoder_bias=np.asarray(
            parameter_state.configuration_encoder_bias, dtype=np.float64
        ).copy(),
        gain_head_weight=np.asarray(
            parameter_state.gain_head_weight, dtype=np.float64
        ).copy(),
        gain_head_bias=np.asarray(
            parameter_state.gain_head_bias, dtype=np.float64
        ).copy(),
        floor_head_weight=np.asarray(
            parameter_state.floor_head_weight, dtype=np.float64
        ).copy(),
        floor_head_bias=np.asarray(
            parameter_state.floor_head_bias, dtype=np.float64
        ).copy(),
        variance_head_weight=np.asarray(
            parameter_state.variance_head_weight, dtype=np.float64
        ).copy(),
        variance_head_bias=np.asarray(
            parameter_state.variance_head_bias, dtype=np.float64
        ).copy(),
    )


def _copy_campaign_state(
    campaign_state: _CampaignOptimizationState,
) -> _CampaignOptimizationState:
    """Deep-copy one mutable campaign optimization state for checkpointing."""

    return _CampaignOptimizationState(
        campaign=campaign_state.campaign,
        sensor_indices=np.asarray(campaign_state.sensor_indices, dtype=np.int64).copy(),
        standardized_configuration=np.asarray(
            campaign_state.standardized_configuration, dtype=np.float64
        ).copy(),
        gain_basis=np.asarray(campaign_state.gain_basis, dtype=np.float64).copy(),
        floor_basis=np.asarray(campaign_state.floor_basis, dtype=np.float64).copy(),
        variance_basis=np.asarray(
            campaign_state.variance_basis, dtype=np.float64
        ).copy(),
        second_difference=np.asarray(
            campaign_state.second_difference, dtype=np.float64
        ).copy(),
        reliable_sensor_local_index=campaign_state.reliable_sensor_local_index,
        latent_spectra_power=np.asarray(
            campaign_state.latent_spectra_power, dtype=np.float64
        ).copy(),
        delta_log_gain=np.asarray(
            campaign_state.delta_log_gain, dtype=np.float64
        ).copy(),
        delta_floor_parameter=np.asarray(
            campaign_state.delta_floor_parameter, dtype=np.float64
        ).copy(),
        delta_variance_parameter=np.asarray(
            campaign_state.delta_variance_parameter, dtype=np.float64
        ).copy(),
        persistent_log_gain=np.asarray(
            campaign_state.persistent_log_gain, dtype=np.float64
        ).copy(),
        persistent_floor_parameter=np.asarray(
            campaign_state.persistent_floor_parameter, dtype=np.float64
        ).copy(),
        persistent_variance_parameter=np.asarray(
            campaign_state.persistent_variance_parameter, dtype=np.float64
        ).copy(),
        gain_power=np.asarray(campaign_state.gain_power, dtype=np.float64).copy(),
        additive_noise_power=np.asarray(
            campaign_state.additive_noise_power, dtype=np.float64
        ).copy(),
        residual_variance_power2=np.asarray(
            campaign_state.residual_variance_power2, dtype=np.float64
        ).copy(),
        objective_value=float(campaign_state.objective_value),
    )


def _snapshot_training_state(
    parameter_state: _PersistentModelParameters,
    campaign_states: Sequence[_CampaignOptimizationState],
) -> tuple[_PersistentModelParameters, list[_CampaignOptimizationState]]:
    """Deep-copy the full mutable optimizer state for best-iterate selection."""

    return _copy_parameter_state(parameter_state), [
        _copy_campaign_state(campaign_state) for campaign_state in campaign_states
    ]


def _assert_finite_scalar(value: float, context_label: str) -> None:
    """Fail fast when a scalar objective or diagnostic becomes non-finite."""

    if not math.isfinite(float(value)):
        raise FloatingPointError(
            f"Non-finite scalar encountered during {context_label}"
        )


def _assert_finite_array(array: FloatArray, context_label: str) -> None:
    """Fail fast when an internal state array becomes non-finite."""

    if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
        raise FloatingPointError(f"Non-finite array encountered during {context_label}")


def _assert_finite_gradients(
    gradients: Mapping[str, FloatArray],
    context_label: str,
) -> None:
    """Validate that every gradient array is finite before optimizer updates."""

    for name, gradient in gradients.items():
        _assert_finite_array(gradient, f"{context_label} gradient {name}")


def _assert_finite_parameter_state(
    parameter_state: _PersistentModelParameters,
    context_label: str,
) -> None:
    """Validate that every persistent parameter array remains finite."""

    for name, parameter in parameter_state.as_dict().items():
        _assert_finite_array(parameter, f"{context_label} parameter {name}")


def _assert_finite_campaign_state(
    campaign_state: _CampaignOptimizationState,
    context_label: str,
) -> None:
    """Validate the campaign state arrays that are critical to optimization."""

    for name, value in (
        ("latent_spectra_power", campaign_state.latent_spectra_power),
        ("delta_log_gain", campaign_state.delta_log_gain),
        ("delta_floor_parameter", campaign_state.delta_floor_parameter),
        ("delta_variance_parameter", campaign_state.delta_variance_parameter),
        ("persistent_log_gain", campaign_state.persistent_log_gain),
        ("persistent_floor_parameter", campaign_state.persistent_floor_parameter),
        ("persistent_variance_parameter", campaign_state.persistent_variance_parameter),
        ("gain_power", campaign_state.gain_power),
        ("additive_noise_power", campaign_state.additive_noise_power),
        ("residual_variance_power2", campaign_state.residual_variance_power2),
    ):
        _assert_finite_array(value, f"{context_label} {name}")


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
        variance_head_bias=np.asarray(
            result.variance_head_bias, dtype=np.float64
        ).copy(),
    )


__all__ = [
    "CalibrationCampaign",
    "CalibrationCorpus",
    "CampaignCalibrationState",
    "CampaignConfiguration",
    "DeploymentCalibrationResult",
    "DeploymentTrustDiagnostics",
    "FitConvergenceDiagnostics",
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
