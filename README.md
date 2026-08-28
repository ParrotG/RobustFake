# AIGC Recognizer

面向真实传播变换的轻量 AIGC 图片检测训练工程。模型使用冻结的 OpenAI CLIP ViT-B/16 视觉编码器，只训练双视图聚合检测头，并通过 clean/transformed 成对一致性学习提升对 JPEG、模糊、缩放、噪声、颜色调整和裁剪的鲁棒性。

完整设计、数据策略和参数说明见 [docs/PROJECT.md](docs/PROJECT.md)。比赛原始题目见 [docs/QUESTION.MD](docs/QUESTION.MD)。

## 快速开始

项目要求 Python 3.10+。推荐使用 `uv`：

```bash
uv sync --extra dev
```

所有可调参数均位于 `configs/default.yaml`。首先流式抽取约 20,000 张训练/验证图片：

```bash
uv run aigc-prepare --config configs/default.yaml
```

然后训练冻结 CLIP 主干上的检测头：

```bash
uv run aigc-train --config configs/default.yaml
```

临时覆盖参数时仍通过统一配置接口：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.batch_size=2 \
  --set training.gradient_accumulation_steps=16
```

运行测试：

```bash
uv run pytest
```

## 默认输出

- `data/processed/community_forensics_20k/manifest.jsonl`：可复现数据清单。
- `data/processed/community_forensics_20k/audit.json`：配额、生成器和过滤审计。
- `artifacts/runs/clip_b16_multiview/best.pt`：最佳可训练头 checkpoint。
- `artifacts/runs/clip_b16_multiview/last.pt`：最后一次训练状态。
- `artifacts/runs/clip_b16_multiview/metrics.jsonl`：带时间戳的训练指标。

数据与训练产物不会提交到 Git。
