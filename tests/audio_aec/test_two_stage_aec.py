import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from examples.audio_aec import TwoStageAEC, nlms_echo_estimate
from examples.audio_aec.losses import total_aec_loss


def test_nlms_shapes():
    ref = torch.randn(2, 1024)
    mic = torch.randn(2, 1024)
    echo, residual = nlms_echo_estimate(ref, mic, filter_len=64)
    assert echo.shape == mic.shape
    assert residual.shape == mic.shape


def test_two_stage_forward_and_loss():
    model = TwoStageAEC(filter_len=64, n_fft=256, hop_length=64, hidden_size=64, channels=(8, 16))
    mic = torch.randn(2, 2048)
    ref = torch.randn(2, 2048)
    near = torch.randn(2, 2048)

    out = model(mic, ref)
    assert set(out.keys()) == {"echo_hat_linear", "residual_linear", "enhanced"}
    assert out["enhanced"].shape == mic.shape

    loss = total_aec_loss(out["enhanced"], near, ref)
    assert torch.isfinite(loss)
