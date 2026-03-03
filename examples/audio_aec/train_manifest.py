from __future__ import annotations

import argparse
import json
import math
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import total_aec_loss
from .pipeline import TwoStageAEC


@dataclass
class SampleItem:
    mic: str
    ref: str
    near: str


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported, got sample_width={sw} for {path}")

    audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)
    mono = audio.mean(axis=1).astype(np.float32) / 32768.0
    return mono, sr


def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] == length:
        return x
    if x.shape[0] > length:
        return x[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


class ManifestAECDataset(Dataset):
    """JSONL manifest dataset.

    Each line example:
    {"mic": ".../mic.wav", "ref": ".../ref.wav", "near": ".../near.wav"}
    """

    def __init__(self, manifest_path: str, sample_rate: int = 16000, seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * seconds)
        self.items: list[SampleItem] = []

        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(SampleItem(mic=obj["mic"], ref=obj["ref"], near=obj["near"]))

        if not self.items:
            raise ValueError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        mic, sr_m = _read_wav_mono(item.mic)
        ref, sr_r = _read_wav_mono(item.ref)
        near, sr_n = _read_wav_mono(item.near)

        if sr_m != self.sample_rate or sr_r != self.sample_rate or sr_n != self.sample_rate:
            raise ValueError(
                f"Sample-rate mismatch for idx={idx}; expected {self.sample_rate}, got mic={sr_m}, ref={sr_r}, near={sr_n}"
            )

        mic = _fix_length(mic, self.target_len)
        ref = _fix_length(ref, self.target_len)
        near = _fix_length(near, self.target_len)

        return {
            "mic": torch.from_numpy(mic),
            "ref": torch.from_numpy(ref),
            "near": torch.from_numpy(near),
        }


def erle_db(mic: torch.Tensor, enhanced: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Approximate ERLE using signal power reduction from mic to enhanced output."""
    p_m = mic.pow(2).mean(dim=-1)
    p_e = enhanced.pow(2).mean(dim=-1)
    return 10.0 * torch.log10((p_m + eps) / (p_e + eps))


def run_epoch(
    model: TwoStageAEC,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    total_erle = 0.0
    count = 0

    for batch in loader:
        mic = batch["mic"].to(device)
        ref = batch["ref"].to(device)
        near = batch["near"].to(device)

        out = model(mic, ref)
        loss = total_aec_loss(out["enhanced"], near, ref)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        with torch.no_grad():
            erle = erle_db(mic, out["enhanced"]).mean().item()

        bsz = mic.shape[0]
        total_loss += loss.item() * bsz
        total_erle += erle * bsz
        count += bsz

    return total_loss / max(count, 1), total_erle / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--valid-manifest", type=str, default="")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints/audio_aec")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    train_ds = ManifestAECDataset(args.train_manifest, sample_rate=args.sample_rate, seconds=args.seconds)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    valid_dl = None
    if args.valid_manifest:
        valid_ds = ManifestAECDataset(args.valid_manifest, sample_rate=args.sample_rate, seconds=args.seconds)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TwoStageAEC().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_valid = math.inf

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_erle = run_epoch(model, train_dl, device, optim)

        msg = f"epoch={epoch:03d} train_loss={tr_loss:.4f} train_erle={tr_erle:.2f}dB"

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_erle": tr_erle,
        }

        if valid_dl is not None:
            with torch.no_grad():
                va_loss, va_erle = run_epoch(model, valid_dl, device, optimizer=None)
            msg += f" valid_loss={va_loss:.4f} valid_erle={va_erle:.2f}dB"
            ckpt["valid_loss"] = va_loss
            ckpt["valid_erle"] = va_erle

            if va_loss < best_valid:
                best_valid = va_loss
                torch.save(ckpt, Path(args.save_dir) / "best.pt")

        torch.save(ckpt, Path(args.save_dir) / "last.pt")
        print(msg)


if __name__ == "__main__":
    main()
