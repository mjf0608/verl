from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride_f: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=(stride_f, 1), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride_f: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=(stride_f, 1),
                padding=1,
                output_padding=(stride_f - 1, 0),
            ),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexCRNAEC(nn.Module):
    """Dual-input complex-domain CRN-like echo residual suppressor.

    Input: mic residual + reference in time domain [B, T].
    Output: enhanced near-end estimate [B, T].
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        hidden_size: int = 256,
        channels: tuple[int, ...] = (16, 32, 64),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win = torch.hann_window(n_fft)

        in_ch = 4  # mic(real,imag) + ref(real,imag)
        self.mic_ref_encoder = nn.ModuleList()
        prev = in_ch
        for c in channels:
            self.mic_ref_encoder.append(ConvBlock(prev, c, stride_f=2))
            prev = c

        self.rnn = nn.GRU(
            input_size=channels[-1],
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_proj = nn.Linear(hidden_size * 2, channels[-1])

        dec_channels = list(channels[::-1])
        self.decoder = nn.ModuleList()
        prev = dec_channels[0]
        for c in dec_channels[1:]:
            self.decoder.append(DeconvBlock(prev + c, c, stride_f=2))
            prev = c

        self.mask_head = nn.Conv2d(prev, 2, kernel_size=1)

    def _stft(self, wav: torch.Tensor) -> torch.Tensor:
        window = self.win.to(wav.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        return torch.view_as_real(spec).permute(0, 3, 1, 2)  # [B,2,F,N]

    def _istft(self, spec_ri: torch.Tensor, length: int) -> torch.Tensor:
        window = self.win.to(spec_ri.device)
        spec = torch.view_as_complex(spec_ri.permute(0, 2, 3, 1).contiguous())
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=length,
        )

    def forward(self, mic_residual: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        spec_m = self._stft(mic_residual)
        spec_r = self._stft(ref)
        x = torch.cat([spec_m, spec_r], dim=1)

        skips = []
        h = x
        for enc in self.mic_ref_encoder:
            h = enc(h)
            skips.append(h)

        b, c, f, n = h.shape
        seq = h.mean(dim=2).transpose(1, 2)  # [B,N,C]
        seq, _ = self.rnn(seq)
        seq = self.rnn_proj(seq).transpose(1, 2).unsqueeze(2).expand(-1, -1, f, -1)
        h = h + seq

        for i, dec in enumerate(self.decoder):
            skip = skips[-2 - i]
            if h.shape[-2] != skip.shape[-2] or h.shape[-1] != skip.shape[-1]:
                h = torch.nn.functional.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        if h.shape[-2:] != spec_m.shape[-2:]:
            h = torch.nn.functional.interpolate(h, size=spec_m.shape[-2:], mode="bilinear", align_corners=False)

        mask = torch.tanh(self.mask_head(h))
        est_spec = spec_m * mask
        enhanced = self._istft(est_spec, length=mic_residual.shape[-1])
        return enhanced
