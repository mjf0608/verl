"""Two-stage nonlinear acoustic echo cancellation baseline.

Stage-1: linear NLMS adaptive filter.
Stage-2: dual-input complex-domain neural residual suppressor.
"""

from .linear_aec import nlms_echo_estimate
from .model import ComplexCRNAEC
from .pipeline import TwoStageAEC

__all__ = ["nlms_echo_estimate", "ComplexCRNAEC", "TwoStageAEC"]
