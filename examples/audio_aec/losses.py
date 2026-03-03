from __future__ import annotations

import torch


def si_snr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_zm = pred - pred.mean(dim=-1, keepdim=True)
    tgt_zm = target - target.mean(dim=-1, keepdim=True)

    proj = (torch.sum(pred_zm * tgt_zm, dim=-1, keepdim=True) * tgt_zm) / (
        torch.sum(tgt_zm * tgt_zm, dim=-1, keepdim=True) + eps
    )
    noise = pred_zm - proj
    ratio = (torch.sum(proj * proj, dim=-1) + eps) / (torch.sum(noise * noise, dim=-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor, fft_sizes: tuple[int, ...] = (256, 512, 1024)) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred.device)
        p = torch.stft(pred, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        t = torch.stft(target, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        loss = loss + (p.abs() - t.abs()).abs().mean()
    return loss / len(fft_sizes)


def echo_residual_loss(enhanced: torch.Tensor, ref: torch.Tensor, echo_only_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Penalize correlation between output and reference on echo-only frames."""
    corr = (enhanced * ref).mean(dim=-1).abs()
    if echo_only_mask is not None:
        return (corr * echo_only_mask).sum() / (echo_only_mask.sum() + 1e-8)
    return corr.mean()


def nearend_protect_loss(enhanced: torch.Tensor, nearend: torch.Tensor, nearend_mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (enhanced - nearend).abs().mean(dim=-1)
    if nearend_mask is not None:
        return (err * nearend_mask).sum() / (nearend_mask.sum() + 1e-8)
    return err.mean()


def total_aec_loss(
    enhanced: torch.Tensor,
    nearend: torch.Tensor,
    ref: torch.Tensor,
    echo_only_mask: torch.Tensor | None = None,
    nearend_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return (
        0.4 * si_snr_loss(enhanced, nearend)
        + 0.4 * mrstft_loss(enhanced, nearend)
        + 0.1 * echo_residual_loss(enhanced, ref, echo_only_mask)
        + 0.1 * nearend_protect_loss(enhanced, nearend, nearend_mask)
    )
