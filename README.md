# AIGC Recognizer

A lightweight training pipeline for detecting AI-generated images under realistic redistribution artifacts. The detector uses a frozen OpenAI CLIP ViT-B/16 visual encoder and trains only a small, view-order-invariant classification head. Paired clean and degraded views improve robustness to compression, blur, resizing, noise, color changes, and cropping.

The current release implements dataset preparation, model training, validation, checkpointing, and resume support. It does not yet include the final directory-to-JSON inference tool required for a competition submission.

For the full design rationale and data policy, see [docs/PROJECT.md](docs/PROJECT.md). The original challenge specification is available in [docs/QUESTION.MD](docs/QUESTION.MD).

## Technical Overview

### Model

The detector uses `ViT-B-16/openai` from `open_clip` as its visual backbone. Only the visual encoder is retained; the text encoder is not part of the training model. All CLIP parameters are frozen and the encoder remains in evaluation mode throughout training.

Each image is represented by two spatial views:

- A global view that preserves the full composition with aspect-ratio-aware resize and padding.
- A local square crop covering 50%–90% of the shorter image dimension.

Both views share the same frozen encoder. Their L2-normalized CLIP embeddings are aggregated using the view-wise mean and standard deviation:

```text
global image ─┐
              ├─ frozen CLIP ViT-B/16 ─ embeddings ─ mean/std ─ feature head ─ binary logit
local crop ───┘                                              └─ projection head
```

Mean/std aggregation is invariant to view order. The mean captures evidence shared across regions, while the standard deviation exposes disagreement between global and local evidence. The aggregated 1,024-dimensional representation is processed by:

```text
LayerNorm → Linear(1024, 256) → GELU → Dropout
```

The resulting feature feeds a binary classifier and a training-only contrastive projection head. With the default dimensions, approximately 0.36M parameters are trainable. The implementation also enforces total parameters below 2B and trainable parameters below 5M at runtime.

### Training Dataset

Training uses the gated [Shanmuk4622/ai-image-detection-dataset](https://huggingface.co/datasets/Shanmuk4622/ai-image-detection-dataset) at a pinned commit. It contains 10,000 real images, each linked by `source_real_id` to six generated images made from the same image-grounded caption. The dataset inherits non-commercial research restrictions from ImageNet and the individual generator licenses; review the upstream card before use or redistribution.

The preparer retains every real parent and deterministically selects one of its six generated partners. It preserves the upstream pair-level split:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 7,056 | 7,056 | 14,112 |
| Validation | 1,446 | 1,446 | 2,892 |
| Test | 1,498 | 1,498 | 2,996 |

Within each split, generator assignment is seeded and balanced across SD 1.5, SDXL, FLUX Schnell, Kandinsky 2.2, PixArt Sigma, and Würstchen. Counts differ by at most one. Test records remain in the manifest but are not used by `aigc-train` or checkpoint selection.

All source images already share a canonical 512×512 RGB PNG pipeline. Acquisition verifies the declared dimensions, format, pipeline version, and SHA-256. Exact and perceptual duplicates are removed at the parent-pair level. Because the challenge forbids training on COCO val2017, preparation also compares COCO real images against the locally prepared official WildFake real subset using high-resolution perceptual hashes and removes the complete parent pair on a confirmed match.

#### Bounded and resumable shard acquisition

The image Parquet payload is approximately 24.5GB. Preparation first downloads the small metadata index, fixes the exact 20,000 selected IDs, and then scans pinned image shards with a bounded two-file prefetch queue. Only selected image bytes are retained long-term, typically around 7GB.

`preparation_state.json`, `manifest.jsonl`, and `audit.json` are atomically checkpointed. The state records the config fingerprint, exact expected IDs, completed shards, and extracted IDs. Re-running resumes the first unfinished shard without relying on a mutable row offset. Finished files are removed only from the project-owned payload cache; the global Hugging Face cache is never deleted. Hub transport failures use exponential backoff.

The dataset is gated. Accept its conditions on Hugging Face and run `hf auth login` before preparation. The token is read from the active user environment and is never printed or stored in project artifacts.

#### Label-independent standardization and nuisance audit

Before spatial views are generated, both classes pass through the same optional random standardizer. It conservatively samples a resize round trip, a JPEG/WebP re-encode, or both. The standardized base image is shared by clean and degraded branches. Training draws are stochastic; validation, test, and official evaluation use a seed derived from the record ID.

After successful preparation, a lightweight `HistGradientBoostingClassifier` attempts to predict the label from fixed pixel statistics such as colour moments, entropy, sharpness, blockiness, edges, and frequency-band energy. `nuisance_report.json` compares raw canonical images with deterministic standardized images and reports validation/test AUROC, AP, balanced accuracy, F1, confusion matrices, per-generator results, and feature importance. This report is informational and never blocks training; high low-level separability can represent either an unwanted shortcut or a genuine generator fingerprint.

### Paired Robust Training

For every source image, the data pipeline returns four tensors:

- Clean global view.
- Clean local view.
- Degraded global view.
- Degraded local view.

The clean and degraded branches reuse the same global/local geometry. This prevents the consistency objective from confusing a content-region change with a redistribution artifact.

The degraded branch samples zero, one, or two distinct operations with default probabilities 25%, 50%, and 25%. Available operations are:

- JPEG compression, including optional low-probability double JPEG.
- Gaussian blur.
- Downscale/upscale with randomized interpolation.
- Gaussian noise.
- Brightness, contrast, and saturation jitter.
- Center crop followed by resize.
- Optional low-probability WebP recompression.

EXIF orientation, RGB conversion, alpha compositing, aspect-ratio-aware resize/padding, and CLIP normalization are applied consistently to both classes.

The total objective is:

```text
L = 1.0 × mean(BCE(clean), BCE(degraded))
  + 0.5 × SmoothL1(logit_degraded, stop_gradient(logit_clean))
  + 0.1 × supervised_contrastive_loss
```

The two BCE terms teach classification on both clean and transformed inputs. Logit consistency treats the clean prediction as a stable target for its degraded counterpart. The supervised contrastive term groups clean and degraded projections by real/fake label within each batch.

Default optimization uses AdamW, cosine decay after 10% linear warmup, FP16 automatic mixed precision, gradient clipping, and early stopping. A micro-batch of 32 gives an effective batch size of 32. Clean and transformed views are combined into one encoder invocation to improve GPU occupancy while preserving the loss formulation.

### Validation and Evaluation

Validation transforms are seeded from the project seed and record ID, making metrics comparable across runs. Parent pairs never cross train, validation, or test splits.

Every epoch reports metrics separately for clean and degraded validation inputs:

- AUROC.
- Average Precision.
- Balanced Accuracy.
- F1 score at the configured probability threshold.
- Validation loss.

The checkpoint selection score is the arithmetic mean of clean and degraded AUROC. This avoids choosing a model that performs well only on pristine images or only on the sampled degradation distribution.

The external evaluation command uses the challenge-prescribed WildFake subset: 4,998 COCO val2017 real images and 8,843 DALL-E Advanced fake images. It reports the clean result and every severity listed in the challenge statement.

### Visible AI watermark detection

The provenance scanner checks the four image corners for visible AI-generation marks. It recognizes common Chinese vendors such as Doubao/Seedream, Jimeng/Dreamina, Tongyi Wanxiang, and Tencent Hunyuan, plus overseas labels from DALL·E, Google Imagen/Gemini, Midjourney, Adobe Firefly, Stable Diffusion, FLUX, Ideogram, Leonardo, Microsoft Designer, Meta AI, and Canva. OCR is attempted when Tesseract is installed; a conservative pixel fallback also detects the compact white `豆包AI生成` corner mark when OCR is unavailable.

Run a watermark-only scan:

```bash
uv sync --extra dev --extra cv
uv run aigc-watermark --config configs/default.yaml /path/to/image-or-directory
```

The provenance command includes the same result under each record's `watermark` field. Visible marks are provenance hints rather than signatures: they can be cropped, copied, or added after generation, so their absence does not prove an image is real.

## Installation

Requirements:

- Python 3.10 or later.
- PyTorch 2.2 or later.
- An NVIDIA GPU with approximately 8–12GB VRAM is recommended for training.

Install the project and development dependencies with `uv`:

```bash
uv sync --extra dev
```

All runtime parameters are centralized in [configs/default.yaml](configs/default.yaml). Public commands accept only `--config` and optional repeated `--set section.key=value` overrides.

## Dataset Preparation

First prepare the prescribed WildFake subset used by the leakage audit, then accept the gated training dataset terms and authenticate:

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
hf auth login
```

Prepare or resume the default subset:

```bash
uv run aigc-prepare \
  --config configs/default.yaml
```

`data.hf_auth` defaults to `required`. Tokens are read from the active user environment and are never stored in the project configuration or logs.

A successful run produces `complete: true`. The manifest contains at most 20,000 records because any confirmed COCO val2017 overlap is removed as a complete real/fake pair:

```bash
uv run python -c "import json; a=json.load(open('data/processed/ai_image_detection_20k/audit.json')); assert a['complete']; print(a['selected'], 'safe images')"

wc -l data/processed/ai_image_detection_20k/manifest.jsonl
```

If disk space is constrained, lower download concurrency so fewer complete source shards coexist in the project cache:

```bash
uv run aigc-prepare \
  --config configs/default.yaml \
  --set data.download_workers=1
```

## Training

Start training with the centralized defaults:

```bash
uv run aigc-train --config configs/default.yaml
```

For a smaller micro-batch while preserving the effective batch size:

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.batch_size=8 \
  --set training.gradient_accumulation_steps=4
```

Resume a training run from its last checkpoint:

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.resume_from=artifacts/runs/clip_b16_multiview/last.pt
```

Resume validation rejects a checkpoint if its backbone identity or dataset revision differs from the active configuration and manifest.

## Official WildFake Evaluation

Prepare only the prescribed subset without downloading the 1.2TB WildFake repository or its complete 25.6GB DALL-E archive:

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
```

The preparer validates the pinned archive SHA-256 values, reads the official metadata, and uses HTTP Range requests to extract only COCO val2017 and DALL-E Advanced. Approximately 2.91GB of selected image payload is required. Preparation is isolated under `data/evaluation/` and never modifies the training manifest.

Evaluate `best.pt` on clean inputs and the exact challenge severities:

```bash
uv run aigc-evaluate-official --config configs/default.yaml
```

The default matrix contains JPEG quality 90/70/50/30, Gaussian blur sigma 0.5/1.0/2.0, resize 0.5/0.25, Gaussian noise sigma 0.02/0.05/0.10, color jitter within 20%, and center crop 80%. Results and per-image predictions are written under `artifacts/evaluations/wildfake_official/`.

## Outputs

```text
data/processed/ai_image_detection_20k/
├── images/                 Canonical selected image files
├── manifest.jsonl          Idempotent paired manifest
├── preparation_state.json  Exact resumable shard and ID state
├── nuisance_report.json    Informational low-level bias probe
└── audit.json              Revision, split counts, exclusions, and completion

artifacts/runs/clip_b16_multiview/
├── best.pt                 Best trainable-head checkpoint
├── last.pt                 Latest optimizer/training state
├── metrics.jsonl           Timestamped epoch metrics
└── resolved_config.yaml    Exact resolved run configuration

data/evaluation/wildfake_official/
├── images/                 Isolated official evaluation images
├── manifest.jsonl          Exact 13,841-image evaluation manifest
└── audit.json              Source archive identities and counts

artifacts/evaluations/wildfake_official/
├── results.json            Clean and severity-matrix metrics
└── predictions.jsonl       Per-image confidence scores
```

The checkpoints store the trainable detector heads, optimizer, scheduler, AMP scaler, epoch, global step, random state, complete configuration, dataset revision, backbone identity, and parameter counts. Frozen CLIP weights are not duplicated in the checkpoint.

## Tests

Run the offline unit and CPU smoke tests:

```bash
uv run pytest
```

The suite covers strict configuration validation, paired generator selection, leakage exclusion, resumable checkpoints, nuisance probing, deterministic standardization, view-order invariance, frozen-backbone gradients, finite losses, metrics, checkpoint creation, and training resume.

## Known Limitations

- The current model uses only semantic CLIP features; no residual, frequency-domain, or camera-pipeline branch is implemented.
- The paired set covers six text-to-image generators but not image-to-image, editing, or the unseen DALL-E family used by the external benchmark.
- The nuisance probe is diagnostic; it cannot by itself distinguish a harmful acquisition shortcut from a real synthesis fingerprint.
- A binary detector is not proof of image authenticity. Operational use requires calibration, uncertainty handling, and explicit control of false positives on real images.
- The project does not yet provide the final directory inference and JSON submission command.

Dataset files, caches, and training artifacts are excluded from Git.
