from __future__ import annotations

import torch
from torch import nn

from .linear_aec import nlms_echo_estimate
from .model import ComplexCRNAEC


class TwoStageAEC(nn.Module):
    """Linear NLMS front-end + neural residual suppressor back-end."""

    def __init__(self, filter_len: int = 256, step_size: float = 0.2, **model_kwargs):
        super().__init__()
        self.filter_len = filter_len
        self.step_size = step_size
        self.neural = ComplexCRNAEC(**model_kwargs)

    @torch.no_grad()
    def linear_cancel(self, mic: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return nlms_echo_estimate(ref, mic, filter_len=self.filter_len, step_size=self.step_size)

    def forward(self, mic: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        echo_hat, residual = nlms_echo_estimate(ref, mic, filter_len=self.filter_len, step_size=self.step_size)
        enhanced = self.neural(residual, ref)
        return {"echo_hat_linear": echo_hat, "residual_linear": residual, "enhanced": enhanced}
