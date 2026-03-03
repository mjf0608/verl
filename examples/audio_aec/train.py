from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader, Dataset

from .losses import total_aec_loss
from .pipeline import TwoStageAEC


class SyntheticAECDataset(Dataset):
    def __init__(self, n_samples: int = 256, sample_rate: int = 16000, seconds: float = 2.0):
        self.n_samples = n_samples
        self.length = int(sample_rate * seconds)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(idx + 2025)
        t = torch.linspace(0, 1, self.length)

        ref = 0.3 * torch.sin(2 * math.pi * torch.randint(200, 1200, (1,), generator=g).item() * t)
        ref = ref + 0.1 * torch.randn(self.length, generator=g)

        rir_len = 128
        rir = torch.exp(-torch.linspace(0, 6, rir_len))
        rir = rir * (torch.rand(rir_len, generator=g) * 2 - 1)
        echo_lin = torch.nn.functional.conv1d(
            ref.view(1, 1, -1), rir.view(1, 1, -1), padding=rir_len - 1
        ).view(-1)[: self.length]

        echo_nl = torch.tanh(1.6 * echo_lin)
        near = 0.2 * torch.sin(2 * math.pi * torch.randint(100, 300, (1,), generator=g).item() * t + 0.5)
        noise = 0.03 * torch.randn(self.length, generator=g)
        mic = near + echo_nl + noise

        return {"mic": mic.float(), "ref": ref.float(), "near": near.float()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    ds = SyntheticAECDataset(n_samples=args.n_samples, seconds=args.seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = TwoStageAEC().to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(args.epochs):
        for batch in dl:
            mic = batch["mic"].to(args.device)
            ref = batch["ref"].to(args.device)
            near = batch["near"].to(args.device)

            out = model(mic, ref)
            loss = total_aec_loss(out["enhanced"], near, ref)

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"epoch={epoch} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
