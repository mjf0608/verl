# Audio AEC 训练指南

本目录提供两级 AEC 基线：

1. 线性 NLMS 前端（快速去除线性回声）
2. 神经网络残差后端（处理非线性失真与残余回声）

## 1) 准备数据

建议将每个训练样本准备成 3 条同采样率 wav（16kHz, 16-bit PCM）：

- `mic.wav`: 麦克风混合（近端语音 + 回声 + 噪声）
- `ref.wav`: 电视播放参考干声
- `near.wav`: 近端纯净目标语音（监督训练标签）

然后制作 `train.jsonl`（可选 `valid.jsonl`），每行一个 JSON：

```json
{"mic":"/path/a_mic.wav", "ref":"/path/a_ref.wav", "near":"/path/a_near.wav"}
{"mic":"/path/b_mic.wav", "ref":"/path/b_ref.wav", "near":"/path/b_near.wav"}
```

## 2) 启动训练

```bash
python -m examples.audio_aec.train_manifest \
  --train-manifest /path/train.jsonl \
  --valid-manifest /path/valid.jsonl \
  --sample-rate 16000 \
  --seconds 4.0 \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-3 \
  --device cuda \
  --save-dir checkpoints/audio_aec
```

如果没有 GPU，把 `--device cuda` 改为 `--device cpu`。

## 3) 输出内容

每个 epoch 输出：

- `train_loss`, `valid_loss`（若提供验证集）
- `train_erle`, `valid_erle`（越高一般表示回声抑制越强）

并在 `save-dir` 下保存：

- `last.pt`：最新 checkpoint
- `best.pt`：验证损失最佳 checkpoint（有验证集时）

## 4) 快速烟雾测试（合成数据）

```bash
python -m examples.audio_aec.train --epochs 1 --batch-size 2 --device cpu --n-samples 4 --seconds 0.2
```
