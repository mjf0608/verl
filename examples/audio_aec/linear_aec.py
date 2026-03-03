from __future__ import annotations

import torch


def nlms_echo_estimate(
    ref: torch.Tensor,
    mic: torch.Tensor,
    *,
    filter_len: int = 256,
    step_size: float = 0.2,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate linear echo by normalized LMS.

    Args:
        ref: [B, T] reference dry playback signal.
        mic: [B, T] microphone mixed signal.
        filter_len: adaptive filter length.
        step_size: NLMS adaptation step size.
        eps: denominator stabilization.

    Returns:
        echo_hat: [B, T] estimated linear echo.
        residual: [B, T] mic - echo_hat.
    """
    if ref.dim() != 2 or mic.dim() != 2:
        raise ValueError("ref and mic must be 2D tensors of shape [B, T].")
    if ref.shape != mic.shape:
        raise ValueError("ref and mic must have same shape.")

    batch, total = ref.shape
    device = ref.device
    dtype = ref.dtype

    taps = torch.zeros(batch, filter_len, device=device, dtype=dtype)
    echo_hat = torch.zeros_like(mic)

    ref_padded = torch.nn.functional.pad(ref, (filter_len - 1, 0))
    for t in range(total):
        x_t = ref_padded[:, t : t + filter_len].flip(-1)
        y_t = (taps * x_t).sum(dim=-1)
        err = mic[:, t] - y_t

        norm = (x_t.pow(2).sum(dim=-1, keepdim=True) + eps)
        taps = taps + step_size * (err.unsqueeze(-1) * x_t) / norm
        echo_hat[:, t] = y_t

    residual = mic - echo_hat
    return echo_hat, residual
