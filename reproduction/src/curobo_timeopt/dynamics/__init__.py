"""Backend-neutral robot dynamics used by fixed-path retimers."""

from .base import CallableDynamics, DiagonalDynamics, InverseDynamicsModel
from .coefficients import (
    AffineTorqueCoefficients,
    PathDynamicsTerms,
    discretize_affine_torque,
    extract_path_dynamics,
    reconstruct_left_states,
    validate_affine_torque,
)
from .validation import (
    InverseDynamicsJVP,
    TorqueRateJVPValidation,
    TorqueRateValidation,
    inverse_dynamics_jvp,
    inverse_dynamics_reverse_mode,
    torque_rate_jvp,
    torque_rate_reverse_hybrid,
    torque_rate_reverse_mode,
    torque_rate_rnea_fd5,
    validate_sampled_against_jvp,
    validate_sampled_torque_rate,
)

__all__ = [
    "AffineTorqueCoefficients",
    "CallableDynamics",
    "DiagonalDynamics",
    "InverseDynamicsJVP",
    "InverseDynamicsModel",
    "PathDynamicsTerms",
    "TorqueRateJVPValidation",
    "TorqueRateValidation",
    "discretize_affine_torque",
    "extract_path_dynamics",
    "inverse_dynamics_jvp",
    "inverse_dynamics_reverse_mode",
    "reconstruct_left_states",
    "torque_rate_jvp",
    "torque_rate_reverse_hybrid",
    "torque_rate_reverse_mode",
    "torque_rate_rnea_fd5",
    "validate_affine_torque",
    "validate_sampled_against_jvp",
    "validate_sampled_torque_rate",
]
