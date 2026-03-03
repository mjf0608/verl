import sys
from pathlib import Path

import json
import wave

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from examples.audio_aec.train_manifest import ManifestAECDataset, erle_db


def _write_wav16(path: Path, sr: int, data: np.ndarray) -> None:
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_manifest_dataset_and_erle(tmp_path: Path):
    sr = 16000
    n = sr
    t = np.linspace(0, 1, n, endpoint=False)

    ref = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    near = 0.1 * np.sin(2 * np.pi * 220 * t + 0.3).astype(np.float32)
    mic = np.tanh(ref * 1.2) + near

    mic_p = tmp_path / "mic.wav"
    ref_p = tmp_path / "ref.wav"
    near_p = tmp_path / "near.wav"
    _write_wav16(mic_p, sr, mic)
    _write_wav16(ref_p, sr, ref)
    _write_wav16(near_p, sr, near)

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"mic": str(mic_p), "ref": str(ref_p), "near": str(near_p)}) + "\n")

    ds = ManifestAECDataset(str(manifest), sample_rate=sr, seconds=1.0)
    item = ds[0]
    assert item["mic"].shape[0] == n
    assert item["ref"].shape[0] == n
    assert item["near"].shape[0] == n

    mic_t = item["mic"].unsqueeze(0)
    enh_t = item["near"].unsqueeze(0)
    erle = erle_db(mic_t, enh_t)
    assert torch.isfinite(erle).all()
