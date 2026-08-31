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

Training uses an 80,000-image pool with 40,000 real and 40,000 fake records. Every source is pinned to an immutable commit:

| Source | Train real/fake | ID val real/fake | DG val real/fake | Total |
|---|---:|---:|---:|---:|
| Shanmuk paired set | 4k / 4k | 1k / 1k | 0 / 0 | 10k |
| WildFake train | 9k / 9k | 0 / 0 | 4k / 2k | 24k |
| Community Forensics-Small | 13k / 13k | 1k / 2k | 0 / 1k | 30k |
| Tiny-GenImage | 6k / 6k | 2k / 1k | 0 / 1k | 16k |

The 64k training split is strictly class-balanced. The 8k `val_id` split measures in-distribution performance, while the 8k `val_dg` split completely holds out AFHQ/Church real domains, DDPM/GALIP/MAGE/Wukong generators, and stable-hashed Community Forensics model IDs. Best-checkpoint selection weights the clean/degraded AUROC mean of both validation roles equally.

The pinned Tiny-GenImage schema declares eight fake generator classes, but its 35,000 physical rows contain no SD14 examples: the other seven generators contain exactly 2,500 images each. Preparation accepts only this pinned empty-class anomaly, records it in `audit.json`, keeps Wukong as the 1k DG holdout, and distributes the 6k train plus 1k ID fake quota nearly equally across the other six observed generators. It never relabels SD15 as SD14 or fabricates an empty class.

Selection uses fixed source quotas, then square-root balancing within real sources, fake families, architectures, and models. This limits domination by large domains without over-amplifying tiny buckets. Format, resolution, aspect ratio, and encoded density are audited and used only as soft priorities; they are never forced to be label-balanced.

#### Bounded, resumable acquisition and global deduplication

Hugging Face Parquet sources use authenticated pinned downloads; WildFake reads only `total_split/train_metadata.csv` and Range-extracts selected ZIP members in parallel. Each source has an atomic candidate checkpoint containing its completed shards/archives and staged object identities. Re-running skips completed acquisition units. Temporary payloads live only in the project cache, while retained images use content-addressed paths and reuse existing local files by hard link when possible.

Before a candidate enters the manifest, preparation checks upstream provenance, encoded SHA-256, canonical RGB pixel SHA-256, 256-bit pHash+dHash, and crop-resistant hashes. Same-label duplicates are replaced from deterministic reserves; a conflicting-label duplicate fails preparation. External WildFake official/broad and SID-Set manifests are mandatory deny lists. Split priority is external test, DG validation, ID validation, then training. Shanmuk pairs are admitted or rejected together.

On the first mixed preparation run, external manifests that do not already contain perceptual hashes require a local deny-index pass before source payload download begins. This pass displays `Hash external denylist`, uses bounded parallel workers, and atomically checkpoints `data/cache/mixed_aigc_80k/external_deny_index.json` every 500 completed images. It is reused on subsequent runs.

Review every upstream license before use. In particular, Tiny-GenImage is a third-party GenImage derivative distributed under CC BY-NC-SA 4.0, and the combined pool is intended for non-commercial research. Accept gated Hugging Face terms and run `hf auth login`; credentials are read from the user environment and are never saved in project artifacts.

#### Label-independent standardization and nuisance audit

Before spatial views are generated, both classes pass through the same optional random standardizer. It conservatively samples a resize round trip, a JPEG/WebP re-encode, or both. The standardized base image is shared by clean and degraded branches. Training draws are stochastic; validation, test, and official evaluation use a seed derived from the record ID.

After successful preparation, a lightweight `HistGradientBoostingClassifier` attempts to predict the label from fixed pixel statistics such as colour moments, entropy, sharpness, blockiness, edges, and frequency-band energy. `nuisance_report.json` compares raw canonical images with deterministic standardized images and reports both validation roles, per-source and per-generator results, and feature importance. This report is informational and never blocks training; high low-level separability can represent either an unwanted shortcut or a genuine generator fingerprint.

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

Validation transforms are seeded from the project seed and record ID, making metrics comparable across runs. Parent groups and held-out domains never cross training and validation roles.

Every epoch reports metrics separately for clean and degraded validation inputs:

- AUROC.
- Average Precision.
- Balanced Accuracy.
- F1 score at the configured probability threshold.
- Validation loss.

The checkpoint selection score gives 50% to the ID clean/degraded AUROC mean and 50% to the DG clean/degraded AUROC mean. This avoids selecting only for familiar sources or only for the held-out domain.

All external datasets use one manifest-backed evaluator. Dataset acquisition remains isolated so source-specific download and sampling logic cannot affect inference. In addition to the challenge-prescribed WildFake subset, the project supports a complementary 6,000-image WildFake hierarchy sample and a 4,000-image SID-Set validation sample containing only real and fully synthetic images.

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

First prepare every external deny-list evaluation set, then accept the gated source terms and authenticate:

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
uv run aigc-prepare-wildfake-eval --config configs/default.yaml
uv run aigc-prepare-sid-eval --config configs/default.yaml
hf auth login
```

Prepare or resume the mixed pool:

```bash
uv run aigc-prepare \
  --config configs/default.yaml
```

All four revisions and the WildFake train metadata checksum are validated before formal selection. Mutable names such as `main` or `master` are rejected for mixed-data sources.

A successful run produces exactly 80,000 records and `complete: true`:

```bash
uv run python -c "import json; a=json.load(open('data/processed/mixed_aigc_80k/audit.json')); assert a['complete'] and a['selected']==80000; print(a['class_counts'], a['split_counts'])"

wc -l data/processed/mixed_aigc_80k/manifest.jsonl
```

The default network ceiling is 80GiB and expected permanent storage is approximately 25–45GiB. If disk or bandwidth pressure is high, reduce acquisition concurrency:

```bash
uv run aigc-prepare \
  --config configs/default.yaml \
  --set mixed_data.download_workers=1
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

## External Evaluation

Prepare only the prescribed subset without downloading the 1.2TB WildFake repository or its complete 25.6GB DALL-E archive:

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
```

The preparer validates the pinned archive SHA-256 values, reads the official metadata, and uses HTTP Range requests to extract only COCO val2017 and DALL-E Advanced. Approximately 2.91GB of selected image payload is required. Preparation is isolated under `data/evaluation/` and never modifies the training manifest.

Evaluate `best.pt` on clean inputs and the exact challenge severities:

```bash
uv run aigc-evaluate-official --config configs/default.yaml
```

The default matrix contains JPEG quality 90/70/50/30, Gaussian blur sigma 0.5/1.0/2.0, resize 0.5/0.25, Gaussian noise sigma 0.02/0.05/0.10, color jitter within 20%, and center crop 80%. It also enables six ordered composed scenarios by default: social resize/JPEG, double-compressed repost, crop/resize/JPEG, blur/resize/JPEG, edit/noise/JPEG, and one severe but still interpretable crop/blur/resize/JPEG chain. Disable only the composed scenarios with:

```bash
uv run aigc-evaluate-official --config configs/default.yaml \
  --set evaluation.enable_composed_scenarios=false
```

### Broad WildFake sample

The complementary sample is drawn only from WildFake's official held-out `total_split` metadata. It contains 3,000 real and 3,000 fake images. Fake allocation is equal across GAN, diffusion, and other generator families and then equal across supported architectures. Real allocation is equal across AFHQ, CelebA-HQ, LSUN Church, FFHQ, ImageNet, and LAION-5B. DALL-E and COCO are excluded because they are already represented by the prescribed benchmark; multipart SD and Midjourney archives are excluded to keep selective acquisition bounded.

Preparation performs archive-level concurrent HTTP Range extraction. Each worker reads members in ZIP offset order while other archives download in parallel, and the manifest is atomically checkpointed every 100 completed images:

```bash
uv run aigc-prepare-wildfake-eval --config configs/default.yaml
uv run aigc-evaluate-wildfake --config configs/default.yaml
```

The 6,000 selected images are limited to 12GiB by configuration. Re-running the preparation command resumes only missing manifest members.

The upstream GigaGAN archive contains multiple zero-filled members that have `.png` names but cannot be decoded. Before sampling, the preparer finds image entries with an extreme ZIP compression ratio, range-reads only those tiny candidates, and excludes them only after decode validation fails. The result is cached by archive SHA-256 in `archive_integrity.json`; the current pinned archive contains 89 verified corrupt members. Deterministic sampling replaces them within the same GigaGAN stratum, so the GAN family and exact quotas remain intact. `wildfake_evaluation.excluded_source_paths` remains available for manually audited corruption that does not have this compression signature.

### SID-Set sample

SID-Set preparation scans the pinned 30,000-image validation split, ignores label 2 (tampered), and retains 2,000 real plus 2,000 full-synthetic images. Quotas are distributed nearly equally across all 34 Parquet shards and records are selected by a stable seed-based hash within each shard. Three shards are prefetched while the preceding shard is processed, so network and local Parquet decoding overlap. The scan transfers approximately 16.8GB, while only the selected image bytes remain under `data/evaluation/`:

```bash
uv run aigc-prepare-sid-eval --config configs/default.yaml
uv run aigc-evaluate-sid --config configs/default.yaml
```

Preparation state is atomically committed after every shard. Re-running resumes at the first unfinished shard and never deletes the global Hugging Face cache. All three evaluation commands use `evaluation.checkpoint_path`, loader settings, single scenarios, and composed scenarios from the same centralized configuration section. Every scenario also reports source-group counts, mean fake probability, predicted-fake rate, and group accuracy in addition to the overall binary metrics.

## Outputs

```text
data/processed/mixed_aigc_80k/
├── objects/                Content-addressed native image bytes
├── manifest.jsonl          Deterministic unified training manifest
├── selection_state.json    Atomic selection/completion state
├── audit.json              Exact quotas, revisions, licenses, and completion
├── dedup_report.json       Exact/near duplicate and deny-list events
├── distribution_report.json Hard and nuisance bucket distributions
└── nuisance_report.json    Informational raw/standardized low-level probe

artifacts/runs/clip_b16_multiview/
├── best.pt                 Best trainable-head checkpoint
├── last.pt                 Latest optimizer/training state
├── metrics.jsonl           Timestamped epoch metrics
└── resolved_config.yaml    Exact resolved run configuration

data/evaluation/wildfake_official/
├── images/                 Isolated official evaluation images
├── manifest.jsonl          Exact 13,841-image evaluation manifest
└── audit.json              Source archive identities and counts

data/evaluation/wildfake_broad_6k/
├── images/                 Balanced hierarchical WildFake sample
├── manifest.jsonl          Shared evaluation schema
├── preparation_state.json  Resumable archive extraction state
└── audit.json              Exact strata and archive identities

data/evaluation/sid_set_4k/
├── images/                 Real/full-synthetic validation sample
├── manifest.jsonl          Shared evaluation schema
├── preparation_state.json  Resumable shard scan state
└── audit.json              Label and per-shard sampling counts

artifacts/evaluations/wildfake_official/
├── results.json            Clean and severity-matrix metrics
└── predictions.jsonl       Per-image confidence scores

artifacts/evaluations/{wildfake_broad_6k,sid_set_4k}/
├── results.json            Single and composed scenario metrics
└── predictions.jsonl       Per-image confidence scores
```

The checkpoints store the trainable detector heads, optimizer, scheduler, AMP scaler, epoch, global step, random state, complete configuration, dataset revision, backbone identity, and parameter counts. Frozen CLIP weights are not duplicated in the checkpoint.

## Tests

Run the offline unit and CPU smoke tests:

```bash
uv run pytest
```

The suite covers strict mixed quotas, deterministic square-root selection, pair integrity, leakage exclusion, duplicate conflicts and reserve replacement, resumable checkpoints, dual validation, nuisance probing, deterministic standardization, model invariance, finite losses, checkpoint creation, and training resume.

## Known Limitations

- The current model uses only semantic CLIP features; no residual, frequency-domain, or camera-pipeline branch is implemented.
- The source mixture is broader but is still a curated dataset distribution rather than a representative sample of all network uploads.
- The nuisance probe is diagnostic; it cannot by itself distinguish a harmful acquisition shortcut from a real synthesis fingerprint.
- A binary detector is not proof of image authenticity. Operational use requires calibration, uncertainty handling, and explicit control of false positives on real images.
- The project does not yet provide the final directory inference and JSON submission command.

Dataset files, caches, and training artifacts are excluded from Git.
