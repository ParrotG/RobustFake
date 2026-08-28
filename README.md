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

The first version uses [OwensLab/CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small). Review the upstream dataset card and license before redistribution or commercial use.

The default prepared subset contains 20,000 original images:

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 8,000 | 8,000 | 16,000 |
| Validation | 2,000 | 2,000 | 4,000 |

Fake samples are stratified by generator architecture:

| Architecture group | Target share |
|---|---:|
| Latent diffusion (`LatDiff`) | 60% |
| GAN | 15% |
| Pixel-space diffusion (`PixDiff`) | 10% |
| Other | 15% |

Systematic generators are assigned to train or validation by a stable hash of `model_name` and are capped at six images per model. Manual and Commercial architectures contain too few generator identities to satisfy all per-split architecture quotas with generator-disjoint splitting, so they use a stable image-level split and a higher cap. Real sources are capped to prevent a single source from dominating a split.

The preparation pipeline excludes:

- NSFW records.
- DALL-E/OpenAI generator sources reserved by the challenge policy.
- Real sources containing `COCO`, as a conservative leakage guard.
- Invalid labels, unsafe or corrupted images, exact SHA-256 duplicates, and identical perceptual hashes.

Images keep their original encoding and resolution. The 224×224 conversion happens online during training, not during acquisition.

#### Bounded and resumable shard acquisition

The repository contains 186 original Parquet shards totaling approximately 241.9GiB. Its Dataset Viewer conversion is marked partial and exposes only four converted shards, which are insufficient for the configured architecture quotas. The default `local_shards` mode therefore bypasses the partial conversion and pins nine original shards selected using metadata-only architecture/source scans.

The selected source shards total approximately 9.86GiB. Acquisition remains bounded by `max_shard_cache_gb`, `max_scanned`, and `max_download_gb`:

- `hf_hub_download` caches complete source shards and resumes interrupted file downloads.
- PyArrow reads each completed shard locally without further image range requests.
- Manifest and audit files are atomically checkpointed every 1,000 scanned rows.
- Common Hub transport failures are retried with exponential backoff.
- Re-running the same command reuses completed shards and resumes incomplete ones.
- SHA-256 deduplication makes a replayed boundary row idempotent.
- SIGTERM, SIGHUP, exceptions, and Ctrl-C preserve the latest recoverable state. SIGKILL and power loss fall back to the last periodic checkpoint.

The final dataset is accepted only after every class and architecture quota is satisfied.

`manifest.jsonl` is the source of truth for training. The number of files under `images/` may be larger after an interrupted run because unreferenced files are deliberately not deleted automatically.

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

Default optimization uses AdamW, cosine decay after 10% linear warmup, FP16 automatic mixed precision, gradient clipping, and early stopping. A micro-batch of 16 with two-step gradient accumulation gives an effective batch size of 32. Clean and transformed views are combined into one encoder invocation to improve GPU occupancy while preserving the loss formulation.

### Validation and Evaluation

Validation transforms are seeded from the project seed and record ID, making metrics comparable across runs. The validation fake generators are disjoint from training generators.

Every epoch reports metrics separately for clean and degraded validation inputs:

- AUROC.
- Average Precision.
- Balanced Accuracy.
- F1 score at the configured probability threshold.
- Validation loss.

The checkpoint selection score is the arithmetic mean of clean and degraded AUROC. This avoids choosing a model that performs well only on pristine images or only on the sampled degradation distribution.

For a final robustness report, evaluate `best.pt` on an external generator-disjoint dataset and on a severity matrix for each transformation. That external test suite and the final competition inference CLI are intentionally outside the current implementation.

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

Authenticate once if a Hugging Face token is available:

```bash
hf auth login
```

Prepare or resume the default subset:

```bash
uv run aigc-prepare \
  --config configs/default.yaml \
  --set data.hf_auth=required
```

`data.hf_auth` supports `auto`, `required`, and `disabled`. Tokens are read from the active user environment and are never stored in the project configuration or logs.

A successful run must produce an audit with `complete: true` and a 20,000-line manifest. Verify both before training:

```bash
uv run python -c "import json; a=json.load(open('data/processed/community_forensics_20k/audit.json')); assert a['complete']; assert a['selected'] == 20000; print('Dataset is ready')"

wc -l data/processed/community_forensics_20k/manifest.jsonl
```

If the configured scan limit is reached before all architecture quotas are filled, preserve the existing files and increase only the bound:

```bash
uv run aigc-prepare \
  --config configs/default.yaml \
  --set data.hf_auth=required \
  --set data.max_scanned=250000
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

## Outputs

```text
data/processed/community_forensics_20k/
├── images/                 Original selected image files
├── manifest.jsonl          Idempotent training manifest
└── audit.json              Quotas, filters, revision, and preparation checkpoint

artifacts/runs/clip_b16_multiview/
├── best.pt                 Best trainable-head checkpoint
├── last.pt                 Latest optimizer/training state
├── metrics.jsonl           Timestamped epoch metrics
└── resolved_config.yaml    Exact resolved run configuration
```

The checkpoints store the trainable detector heads, optimizer, scheduler, AMP scaler, epoch, global step, random state, complete configuration, dataset revision, backbone identity, and parameter counts. Frozen CLIP weights are not duplicated in the checkpoint.

## Tests

Run the offline unit and CPU smoke tests:

```bash
uv run pytest
```

The suite covers strict configuration validation, sampling quotas and exclusions, interruption recovery, deterministic transforms, view-order invariance, frozen-backbone gradients, finite losses, metrics, checkpoint creation, and training resume.

## Known Limitations

- The current model uses only semantic CLIP features; no residual, frequency-domain, or camera-pipeline branch is implemented.
- CommunityForensics contains many related Stable Diffusion derivatives, so generator-level separation is stronger than random image splitting but weaker than evaluation on an independent generator family.
- Perceptual deduplication removes identical pHashes but does not perform expensive global near-neighbor search.
- A binary detector is not proof of image authenticity. Operational use requires calibration, uncertainty handling, and explicit control of false positives on real images.
- The project does not yet provide the final directory inference and JSON submission command.

Dataset files, caches, and training artifacts are excluded from Git.
