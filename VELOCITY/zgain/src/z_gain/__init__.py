"""z_gain: compute z-gains from normative-model z-scores for any longitudinal
dataset.
"""
from .config import ZGainConfig
from .core import calculate_z_gain, compute_all_z_gains
from .data import (
    ValidationReport,
    build_zgain_frame,
    example_frame,
    example_velocity_matrix,
    validate_zgain_frame,
)
from .pipeline import load_input_data, load_velocity_matrix, run_zgain_analysis

__version__ = "0.1.0"

__all__ = [
    "ZGainConfig",
    "ValidationReport",
    "build_zgain_frame",
    "example_frame",
    "example_velocity_matrix",
    "validate_zgain_frame",
    "calculate_z_gain",
    "compute_all_z_gains",
    "load_input_data",
    "load_velocity_matrix",
    "run_zgain_analysis",
]
