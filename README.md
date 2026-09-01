
# RobustFake

TikTok TechJam 2026
[GitHub](https://github.com/ParrotG/RobustFake) · [Hugging Face Model](https://huggingface.co/ParrotG/RobustFake)

## Problem and Solution

Generative image systems can produce realistic synthetic media at scale, while ordinary redistribution operations—JPEG recompression, blur, resizing, noise, colour adjustment, and cropping—can erase or alter the traces used by image-forensics detectors. The challenge is therefore not merely to separate clean AI-generated and authentic images, but to preserve reliable ranking and control false positives after content has passed through realistic sharing pipelines. The detector must also remain practical at hackathon scale and stay below the 2-billion-parameter limit.

RobustFake addresses this problem with a frozen CLIP ViT-B/16 visual encoder and a compact trainable forensic head. It combines global context with local evidence, fuses semantic and intermediate transformer representations, incorporates fixed residual statistics, and trains on paired clean/degraded views whose spatial geometry is shared. A diverse, leakage-audited training pool and separate in-distribution/domain-generalisation validation roles reduce dependence on a single generator or real-image source. Post-training affine calibration corrects global confidence bias without changing the detector ranking or using the official demonstration set for fitting.

![A generated image that UnivFD detects when clean but misses after resizing, while RobustFake remains correct](assets/baseline-examples/univfd-transform-flip.png)

### Project Snapshot

| Category | Selection |
|---|---|
| Development environment | Visual Studio Code; Python command-line workflow |
| Model | Frozen OpenCLIP `ViT-B-16-quickgelu/openai`; trainable multi-layer fusion and binary detection heads; residual-statistics branch enabled by default |
| Core libraries and frameworks | PyTorch, torchvision, OpenCLIP, Pillow, scikit-learn, NumPy, Hugging Face Hub, PyArrow, ModelScope Hub |
| Training datasets | Shanmuk AI Image Detection Dataset, WildFake train split, Community Forensics-Small, Tiny-GenImage |
| Official demonstration dataset | WildFake subset: COCO val2017 real images and DALL·E Advanced generated images |
| Scale | 80,000 prepared images; 64k train, 8k ID validation, 8k domain-generalisation validation |
| Compute profile | Frozen backbone; fewer than 5M trainable parameters; approximately 8–12GB NVIDIA GPU memory recommended |


## Contents

- [RobustFake](#robustfake)
  - [Problem and Solution](#problem-and-solution)
    - [Project Snapshot](#project-snapshot)
  - [Contents](#contents)
  - [Model Design](#model-design)
    - [Frozen multi-layer visual representation](#frozen-multi-layer-visual-representation)
    - [Training objective](#training-objective)
  - [Dataset Design](#dataset-design)
  - [Training, Validation, and Calibration](#training-validation-and-calibration)
    - [Paired robustness augmentation](#paired-robustness-augmentation)
    - [Optimisation profile](#optimisation-profile)
    - [Validation and checkpoint selection](#validation-and-checkpoint-selection)
    - [Global calibration](#global-calibration)
  - [Official Evaluation](#official-evaluation)
    - [External academic baselines](#external-academic-baselines)
      - [Completed full-matrix comparison](#completed-full-matrix-comparison)
    - [Ablation study results](#ablation-study-results)
  - [Ablation Protocol](#ablation-protocol)
  - [Environment and Reproduction](#environment-and-reproduction)
    - [Environment requirements](#environment-requirements)
    - [Installation](#installation)
    - [Required directory-to-JSON inference](#required-directory-to-json-inference)
    - [Evaluation with the Hugging Face model](#evaluation-with-the-hugging-face-model)
    - [Training reproduction](#training-reproduction)
    - [External baseline reproduction](#external-baseline-reproduction)
    - [Ablation reproduction](#ablation-reproduction)
  - [Robustness Evaluation Summary](#robustness-evaluation-summary)
  - [Error Analysis Note](#error-analysis-note)
    - [Calibration and the false-positive trade-off](#calibration-and-the-false-positive-trade-off)
  - [Limitation Reflection](#limitation-reflection)
  - [Team Contribution](#team-contribution)

## Model Design

### Frozen multi-layer visual representation

The visual backbone is OpenCLIP `ViT-B-16-quickgelu/openai`. The text encoder is discarded, all visual-encoder parameters remain frozen, and the backbone is kept in evaluation mode. This preserves the broad semantic representation learned by CLIP while keeping training and checkpoint size appropriate for the available compute budget.

Each image is rendered as two square views:

- A global view covering 90%–100% of the shorter image dimension retains scene-level context without exposing label-correlated letterbox padding.
- A local view covering 50%–90% of the shorter dimension increases the chance of observing spatially local synthesis artifacts.

For each view, the detector extracts the final 512-dimensional projected CLIP embedding and normalized CLS tokens from transformer blocks 4, 7, 10, and 12. Trainable linear projections map the intermediate tokens to 512 dimensions. A sample-dependent softmax gate then combines the final semantic representation with intermediate evidence instead of assigning every layer a fixed importance.

The fused global and local embeddings are aggregated with their mean and standard deviation. The mean represents evidence shared by both views; the standard deviation exposes disagreement between global context and local detail. This aggregation is invariant to view ordering.

The residual-statistics branch, enabled by default, computes 24 fixed high-pass statistics per view from directional residuals, channel-wise Laplacians, and horizontal/vertical pixel differences. Its view-wise mean and standard deviation pass through a small MLP and are concatenated with the CLIP aggregate. This gives the detector direct access to compact forensic evidence while leaving the CLIP backbone frozen.

![RobustFake Architecture](assets/RobostFakeArchitecture.png)


At inference time, only one clean global/local pair is required. The clean/degraded branches shown above are paired training views.

### Training objective

Clean and degraded views of the same image reuse exactly the same crop geometry, so the consistency term measures sensitivity to redistribution artifacts rather than sensitivity to different image regions.

```text
L = 1.0 × mean(BCEclean, BCEdegraded)
  + ramp(epoch) × 0.5 × SmoothL1(sigmoid(logitdegraded), stop_gradient(sigmoid(logitclean)))
  + 0.05 × supervised_contrastive_loss
```

- Binary cross-entropy teaches the classifier on both untouched and redistributed inputs.
- Probability-space consistency uses the clean prediction as a bounded teacher target. Its one-epoch ramp prevents an abrupt regularisation change at the start of optimisation without allowing unbounded logit distances to dominate later training.
- Supervised contrastive learning brings clean/degraded projections with the same real/fake label closer while separating opposing labels.

The final feature head is `LayerNorm → Linear → GELU → Dropout`, followed by a one-logit classifier. Runtime checks enforce the challenge's parameter limit and cap the trainable detector components below 5M parameters.

## Dataset Design

The training pool is designed to represent variation along two different axes: how authentic images are acquired and what type of generator produces synthetic images. Authentic content includes photographic, web-scale, face, landscape, and ImageNet/COCO-like sources. Synthetic content spans diffusion, GAN, autoregressive/other families, general-purpose generators, fine-tuned community models, and multiple generation resolutions. This diversity is intended to make the decision boundary less dependent on one benchmark's content style or one generator family.

| Dataset source | Authentic-image sources represented in the selected pool | AIGC generator categories and models represented in the selected pool |
|---|---|---|
| Shanmuk paired set | COCO and ImageNet parent images | Modern diffusion-family generators: Stable Diffusion 1.5, SDXL, FLUX.1-schnell, Kandinsky 2.2, PixArt-Σ, and Würstchen |
| WildFake train split | LAION-5B, ImageNet, FFHQ, CelebA-HQ, AFHQ, and Church | Diffusion models including ADM, VQDM, DDPM, DDIM, and Imagen; GAN-family models including BigGAN, StyleGAN, StarGAN, DF-GAN, GALIP, and GigaGAN; other token/autoencoding architectures including VQ-VAE, VQGAN, and MAE |
| Community Forensics-Small | COCO, FFHQ, LandscapesHQ, and VISION camera images | Latent-diffusion, pixel-diffusion, GAN, and other families; systematic models include ProGAN, BigGAN, GigaGAN, StyleGAN variants, ProjectedGAN, GLIDE, DeepFloyd, VQDiffusion, Taming Transformers, and LFM, supplemented by many community fine-tunes and LoRAs |
| Tiny-GenImage | ImageNet images | Diffusion and related generators ADM, GLIDE, Midjourney, VQDM, Stable Diffusion 1.5, and Wukong, plus the GAN-family BigGAN |

The entries above describe models and authentic origins that are actually present in the final 80,000-record manifest, rather than every category advertised by each upstream repository. In particular, the pinned Tiny-GenImage revision contains no physical SD1.4 records, so that declared class is not presented as observed coverage.

The physical pool contains 40,000 real and 40,000 generated images. Source quotas prevent a large repository from replacing the intended mixture. Within each quota, real samples are bucketed by acquisition/content source and generated samples by family, architecture, and model. Square-root allocation gives larger domains more examples while reducing their ability to dominate and avoiding excessive oversampling of very small buckets.

The 80,000 records are divided into:

- `train`: 64,000 class-balanced images used for optimisation.
- `val_id`: 8,000 class-balanced images from represented sources, measuring ordinary held-out generalisation.
- `val_dg`: 8,000 class-balanced images containing fully held-out real domains, generator architectures, and community model identities, measuring domain and generator transfer.

Parent groups and paired records remain within one role. Selected generators and real domains are excluded globally from training when they are assigned to domain-generalisation validation.

Duplicate and leakage control is applied before a candidate enters the manifest. The pipeline checks provenance identity, encoded content, canonical pixels, perceptual similarity, and crop-resistant similarity. Known official-evaluation assets have priority over validation and training records, same-label duplicates are replaced from deterministic reserves, and confirmed conflicting-label duplicates fail preparation. These controls make the official result a meaningful external demonstration rather than a measure of memorised images.

Preprocessing is deliberately label-independent. EXIF orientation, alpha compositing, RGB conversion, optional resize round trips, and JPEG/WebP re-encoding are applied without access to the label or source identity. This reduces reliance on acquisition format while retaining both clean and degraded branches from the same standardised base image. A separate nuisance probe audits whether simple colour, sharpness, blockiness, and frequency features can still predict the label; it is diagnostic and does not become an input to the detector.

All source revisions and critical metadata checksums are pinned. Dataset-specific licenses and upstream image restrictions remain authoritative; the combined pool is intended for non-commercial research and hackathon demonstration.

## Training, Validation, and Calibration

### Paired robustness augmentation

Every training record produces clean global/local views and a degraded global/local pair. The degraded branch samples zero, one, or two distinct operations from:

- JPEG recompression, with low-probability double JPEG;
- Gaussian blur;
- downscale/upscale with varied interpolation;
- Gaussian noise;
- brightness, contrast, and saturation changes;
- centre crop followed by resize;
- low-probability WebP recompression.

The configured ranges cover the challenge severities while also sampling intermediate values. Applying the same corruption distribution to both labels prevents “being degraded” from becoming a synthetic-image label. Sharing crop geometry between clean and degraded pairs isolates transformation sensitivity, and retaining a clean classification branch protects clean-image performance while robustness is learned.

### Optimisation profile

| Setting | Value | Motivation |
|---|---:|---|
| Epochs | 8 | Sufficient for the compact head, with early stopping |
| Micro-batch / effective batch | 64 / 64 | Stable class coverage within the available memory budget |
| Optimizer | AdamW | Standard optimisation for transformer-derived features |
| Learning rate | `1e-4` | Conservative update size for the small detection head |
| Weight decay | `1e-3` | Limits logit-scale and head overfitting |
| Warm-up | 5% of optimizer steps | Stabilises early updates |
| Schedule | Cosine decay | Smoothly reduces the learning rate |
| AMP | FP16 | Improves throughput and memory use |
| Gradient clipping | `1.0` | Guards against unstable head updates |
| Early-stopping patience | 3 epochs | Limits overfitting after validation stops improving |
| Consistency weight / ramp | `0.5` / 1 epoch | Enforces redistribution invariance after a short warm start |
| Contrastive weight / temperature | `0.05` / `0.10` | Adds class-structured representation supervision |

CLIP features can be precomputed into deterministic FP16 shards. Cached training loads only the trainable projections, gate, residual branch, and detector heads, which makes repeated head training practical without changing the mathematical objective.

### Validation and checkpoint selection

Validation transformations are deterministically seeded from the record identity so metrics are comparable across epochs. Clean and degraded metrics are reported separately on both validation roles. The checkpoint score remains intentionally simple:

```text
selection score = 0.5 × mean ID clean/degraded AUROC
                + 0.5 × mean DG clean/degraded AUROC
```

This gives equal importance to familiar held-out domains and deliberately unseen domains without tuning against the official demonstration dataset. AUROC and Average Precision measure ranking; Balanced Accuracy, F1, real recall, fake recall, and confusion counts expose threshold behavior and false-positive trade-offs.

### Global calibration

After checkpoint selection, a single affine Platt calibrator is fitted on pooled clean/degraded `val_id` and `val_dg` logits:

```text
pcalibrated = sigmoid(a × raw_logit + b)
```

The fitted intercept can correct a global real/fake confidence bias that temperature-only scaling cannot move. Threshold selection then considers `val_id_clean`, `val_id_transformed`, `val_dg_clean`, and `val_dg_transformed` separately. It protects the best clean macro Balanced Accuracy within a configured tolerance and, among eligible thresholds, maximises the worst validation-group Balanced Accuracy before using macro performance as a tie-breaker. This avoids allowing an easy or over-represented validation group to determine the operating point.

The calibration artifact is bound to the checkpoint SHA-256, is never fitted on the official demonstration set, and is automatically applied by official evaluation and directory inference when present. If compatible validation feature shards exist, calibration only runs the small detector head.

## Official Evaluation

The challenge-prescribed demonstration subset contains 4,998 COCO val2017 authentic images and 8,843 DALL·E Advanced generated images. It is isolated under `data/evaluation/`, excluded from training through the leakage deny list, and evaluated with the exact single-transform matrix requested by the challenge:

- JPEG quality: 90, 70, 50, 30;
- Gaussian blur sigma: 0.5, 1.0, 2.0;
- resize: 0.5× and 0.25× followed by upscaling;
- Gaussian noise sigma: 0.02, 0.05, 0.10;
- colour jitter within 20%;
- centre crop retaining 80%.

Additional ordered compositions are treated as stress tests and reported separately from the prescribed single transformations.

The completed official evaluation currently reports:

| Evaluation slice | AUROC |
|---|---:|
| Clean | 0.9792 |
| Mean across prescribed single transformations | 0.9683 |
| Worst prescribed single transformation | 0.9432 |
| Mean across additional composed stress tests | 0.8968 |
| Worst additional composed stress test | 0.8713 |

With the internally selected robust calibrated threshold, clean Balanced Accuracy is 0.8898. Strong blur and 0.25× resize retain AUROC 0.9469 and 0.9446 while achieving real recall 0.8824 and 0.8705 respectively. The strongest additional composed stress test remains the principal operational weakness: AUROC is 0.9168, but real recall falls to 0.6152. This separation between ranking and threshold behavior motivates reporting both AUROC and class-specific recall.

Official evaluation caches frozen final, intermediate, and residual features by preprocessing identity and scenario. The first evaluation computes CLIP features; compatible checkpoints subsequently execute only the trainable head. A deterministic `--fast` profile uses a balanced 2,000-image subset and representative severe scenarios for iteration, while the formal report always uses the complete official subset and full prescribed matrix.

### External academic baselines

The same prepared manifests and transformation scenarios can evaluate two pinned public detectors without retraining them:

- CNNDetection, the ResNet-50 `blur_jpg_prob0.5.pth` model from *CNN-generated images are surprisingly easy to spot... for now* (CVPR 2020);
- UnivFD, the OpenAI CLIP ViT-L/14 linear detector from *Towards Universal Fake Image Detectors That Generalize Across Generative Models* (CVPR 2023).

Their official checkpoints, repository revisions, preprocessing, score direction, and SHA-256 values are recorded in `baseline_evaluation`. Downloads are verified before loading. Baseline scores are deliberately reported as uncalibrated; AUROC and Average Precision are therefore the primary comparisons, while fixed-threshold metrics remain diagnostic.

```bash
uv run aigc-download-baselines --config configs/default.yaml
uv run aigc-evaluate-baselines \
  --config configs/default.yaml \
  --dataset wildfake_official \
  --fast
```

Omit `--fast` for the complete single- and composed-transformation matrix. Use repeated `--baseline cnndetection` or `--baseline univfd` arguments to select individual baselines. Results and per-image predictions are written under `artifacts/evaluations/baselines/<dataset>/<baseline>/`; checkpoint-independent scenario scores are cached separately for repeatable presentation analysis.

#### Completed full-matrix comparison

All three detectors were evaluated on the same 13,841-image WildFake official demonstration set: 4,998 COCO val2017 authentic images and 8,843 DALL·E Advanced generated images. Every run contains the same record IDs and class counts for clean input, all 14 prescribed single transformations, and six additional composed stress tests. RobustFake uses its complete released inference pipeline; CNNDetection and UnivFD retain their pinned official checkpoints, 224-pixel preprocessing, and uncalibrated heads. Because no target-set calibration or retraining is applied to the public baselines, AUROC is the primary threshold-independent comparison.

![RobustFake compared with CNNDetection and UnivFD across the complete prescribed robustness matrix](assets/baseline/robustness_comparison.svg)

| Detector | Clean AUROC | Mean prescribed-transform AUROC | Worst prescribed-transform AUROC | Worst prescribed scenario |
|---|---:|---:|---:|---|
| **RobustFake** | **0.9792** | **0.9683** | **0.9432** | Resize 0.5× |
| CNNDetection (CVPR 2020) | 0.6041 | 0.5975 | 0.5314 | Gaussian noise σ=0.10 |
| UnivFD (CVPR 2023) | 0.5898 | 0.5546 | 0.4024 | Resize 0.25× |

Relative to the stronger public baseline on each aggregate, RobustFake gains 37.5 AUROC points on clean input, 37.1 points on the mean prescribed transformation, and 41.2 points on the worst prescribed transformation. More importantly, its mean-to-worst decline is only 2.51 points, compared with 6.61 points for CNNDetection and 15.22 points for UnivFD. The per-scenario panel shows that this is not driven by one favorable corruption: RobustFake remains above 0.94 AUROC in every prescribed scenario, while the two public checkpoints are close to chance on several resizing, blur, and noise settings.

The result supports the complete-system design rather than attributing the gap to calibration: affine calibration is monotonic and cannot improve AUROC ranking. RobustFake combines a broader leakage-audited training mixture, explicit clean/degraded pairing, global/local evidence, multi-layer CLIP fusion, and residual statistics. Conversely, the public checkpoints were trained for different generator distributions and are intentionally used without adaptation, so these numbers demonstrate transfer to this challenge setting rather than claiming that either published method is universally weak. Their fixed 0.5 thresholds predict almost every sample as authentic on this dataset; thresholded accuracy and recall are therefore retained only as domain-shift diagnostics, not used for the headline comparison.

The checked summary and vector figure can be regenerated directly from the three full result files:

```bash
uv run robustfake-baseline-report \
  --result RobustFake=artifacts/evaluations/wildfake_official/results.json \
  --result CNNDetection=artifacts/evaluations/baselines/wildfake_official/cnndetection/results.json \
  --result UnivFD=artifacts/evaluations/baselines/wildfake_official/univfd/results.json \
  --output-dir assets/baseline
```

### Ablation study results

The ablation follows a strict leave-one-component-out design. Each trainable variant starts from the release configuration and removes exactly one component while retaining the same 80k manifest, split roles, seed, cached clean/degraded features, eight-epoch budget, optimiser, checkpoint-selection rule, and internal constrained-minimax calibration protocol. Every checkpoint is then evaluated on the complete official scenario matrix; the official labels are never used to select a checkpoint or fit a calibrator. The no-calibration row instead reuses the unchanged full checkpoint and only removes its post-hoc affine mapping and calibrated threshold.

![Grouped AUROC comparison for the RobustFake ablation study](assets/ablation/auroc_comparison.png)

| Variant | Clean AUROC | Mean single AUROC | Worst single AUROC | Mean composed AUROC | Δ mean / worst single vs Full |
|---|---:|---:|---:|---:|---:|
| **Full RobustFake** | **0.9792** | **0.9683** | **0.9432** | **0.8968** | Reference |
| Without residual statistics | 0.9624 | 0.9563 | 0.9060 | 0.8812 | −1.20 / −3.72 pp |
| Without multi-layer fusion | 0.9582 | 0.9235 | 0.8317 | 0.8401 | −4.48 / −11.15 pp |
| Without consistency loss | 0.9787 | 0.9676 | 0.9419 | 0.8925 | −0.08 / −0.13 pp |
| Without contrastive loss | 0.9774 | 0.9659 | 0.9387 | 0.8890 | −0.25 / −0.45 pp |

Multi-layer fusion provides the largest measured contribution: removing the intermediate transformer evidence reduces mean single-transform AUROC by 4.48 percentage points, worst-single AUROC by 11.15 points, and mean composed AUROC by 5.67 points. The residual-statistics branch is the second strongest component, with a particularly clear 3.72-point loss on the worst prescribed transformation. Together, these results support the central design claim that semantic final-layer features benefit from both intermediate representation levels and compact forensic statistics.

The auxiliary objectives have smaller but directionally consistent effects in this single-seed study. Removing contrastive learning reduces every reported ranking aggregate, including mean composed AUROC by 0.79 points. Removing consistency changes mean and worst single-transform AUROC by only 0.08 and 0.13 points, while its larger 0.43-point effect on composed transformations provides limited evidence that it helps under accumulated degradation. These small differences should be reported as marginal contributions rather than statistical significance.

Calibration is deliberately excluded from the AUROC chart because its positive affine mapping is monotonic and therefore leaves ranking effectively unchanged. Its contribution is operational: compared with the raw full checkpoint, robust calibration raises mean single-transform Balanced Accuracy from 0.8887 to 0.8939 and worst-single real recall from 0.6004 to 0.8705, while lowering clean Balanced Accuracy from 0.9278 to 0.8898. This exposes an explicit trade-off between protecting authentic images from false accusations under degradation and recovering more generated images on clean input; the detailed threshold analysis appears in [Error Analysis Note](#calibration-and-the-false-positive-trade-off).

## Ablation Protocol

The core ablation suite starts from the final RobustFake configuration and independently removes residual statistics, multi-layer fusion, consistency loss, contrastive loss, or post-hoc calibration. All trainable ablations reuse the same frozen feature and residual caches, data split, seed, optimisation budget, checkpoint rule, and internal calibration protocol. Exact training, evaluation, and report-generation commands are included in [Ablation reproduction](#ablation-reproduction).

## Environment and Reproduction

### Environment requirements

- Python 3.10 or later;
- `uv` for dependency and environment management;
- PyTorch 2.2 or later;
- an NVIDIA GPU with approximately 8–12GB VRAM for training;
- sufficient local storage for prepared images and feature caches;
- authenticated Hugging Face access only when reproducing gated training datasets.

### Installation

```bash
git clone https://github.com/ParrotG/RobustFake.git
cd RobustFake
uv sync --extra dev
```

All runtime settings are centralised in [configs/default.yaml](configs/default.yaml). Commands accept `--config` and repeated `--set section.key=value` overrides.

The release package is published at [ParrotG/RobustFake](https://huggingface.co/ParrotG/RobustFake). It provides the trainable checkpoint, its SHA-256-bound calibration, and the resolved training configuration, so users do not need to reproduce training before inference. A public repository does not require authentication for download.

To download the model package without running inference:

```bash
uv run robustfake-download-model \
  --config configs/default.yaml \
  --hf-repo ParrotG/RobustFake
```

### Required directory-to-JSON inference

The challenge-required CLI recursively scores supported images and atomically writes one JSON array:

```bash
uv run aigc-predict \
  --config configs/default.yaml \
  --hf-repo ParrotG/RobustFake \
  --input-dir path/to/images \
  --output-json predictions.json
```

Each output item contains exactly the required fields:

```json
{"image_path": "path/to/images/example.jpg", "pred": 0.9231}
```

`pred` is the estimated probability that the image is AI-generated. A compatible checkpoint-bound calibration file is applied automatically. Use `--no-recursive` to restrict discovery to the top-level directory.

After the first download, Hugging Face Hub reuses its local immutable snapshot cache. A local package remains supported by replacing `--hf-repo` with `--checkpoint path/to/best.pt`; place the matching `calibration.json` beside the checkpoint.

### Evaluation with the Hugging Face model

After preparing the official demonstration subset, run either the fast diagnostic or complete matrix without training:

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
uv run aigc-evaluate-official \
  --config configs/default.yaml \
  --hf-repo ParrotG/RobustFake \
  --fast
uv run aigc-evaluate-official \
  --config configs/default.yaml \
  --hf-repo ParrotG/RobustFake
```

### Training reproduction

Prepare the official protected manifest and all configured external leakage-deny manifests before constructing the mixed dataset:

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
hf auth login
uv run aigc-prepare --config configs/default.yaml
```

Cache frozen features and train the detector head:

```bash
uv run aigc-cache-features --config configs/default.yaml
uv run aigc-cache-residuals --config configs/default.yaml
uv run aigc-train \
  --config configs/default.yaml \
  --set feature_cache.use_for_training=true
```

Fit calibration and run the complete official evaluation:

```bash
uv run aigc-calibrate \
  --config artifacts/runs/your_run/resolved_config.yaml \
  --checkpoint artifacts/runs/your_run/best.pt
uv run aigc-evaluate-official \
  --config artifacts/runs/your_run/resolved_config.yaml \
  --set evaluation.checkpoint_path=artifacts/runs/your_run/best.pt
```

For a shorter diagnostic pass, append `--fast` to the evaluation command. Dataset locations, cache paths, retry limits, and evaluation outputs are centralised in `configs/default.yaml`.

### External baseline reproduction

Download the pinned and checksum-verified public baseline checkpoints:

```bash
uv run aigc-download-baselines \
  --config configs/default.yaml
```

After preparing the official demonstration set, evaluate a selected detector on the same manifest-backed pipeline:

```bash
uv run aigc-evaluate-baselines \
  --config configs/default.yaml \
  --dataset wildfake_official \
  --baseline cnndetection
```

Replace `cnndetection` with `univfd` to reproduce the other published baseline comparison. Add `--fast` for the deterministic diagnostic subset where supported.

### Ablation reproduction

The four trainable leave-one-component-out variants reuse the release feature and residual caches:

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_residual \
  --set model.residual_statistics_enabled=false \
  --set feature_cache.use_for_training=true

uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_multilayer \
  --set model.multilayer_fusion_enabled=false \
  --set feature_cache.use_for_training=true

uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_consistency \
  --set loss.consistency_weight=0.0 \
  --set feature_cache.use_for_training=true

uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_contrastive \
  --set loss.contrastive_weight=0.0 \
  --set feature_cache.use_for_training=true
```

Fit each run's calibration on its internal validation data, then evaluate it into an isolated result directory. Set `component` to `residual`, `multilayer`, `consistency`, or `contrastive`:

```bash
component=residual
uv run aigc-calibrate \
  --config artifacts/runs/ablation_without_${component}/resolved_config.yaml \
  --checkpoint artifacts/runs/ablation_without_${component}/best.pt

uv run aigc-evaluate-official \
  --config artifacts/runs/ablation_without_${component}/resolved_config.yaml \
  --checkpoint artifacts/runs/ablation_without_${component}/best.pt \
  --set official_evaluation.results_path=artifacts/ablations/without_${component}/results.json \
  --set evaluation.save_predictions=false
```

The calibration ablation reuses the unchanged full checkpoint. The paths below correspond to a run produced by the default `project.run_name: robustfake`:

```bash
uv run aigc-evaluate-official \
  --config artifacts/runs/robustfake/resolved_config.yaml \
  --checkpoint artifacts/runs/robustfake/best.pt \
  --set evaluation.calibration_enabled=false \
  --set official_evaluation.results_path=artifacts/ablations/without_calibration/results.json \
  --set evaluation.save_predictions=false
```

Generate the summary tables plus presentation-ready SVG and PNG figures after the complete evaluations finish:

```bash
uv run robustfake-ablation-report \
  --result Full=artifacts/evaluations/wildfake_official/results.json \
  --result NoResidual=artifacts/ablations/without_residual/results.json \
  --result NoMultilayer=artifacts/ablations/without_multilayer/results.json \
  --result NoConsistency=artifacts/ablations/without_consistency/results.json \
  --result NoContrastive=artifacts/ablations/without_contrastive/results.json \
  --result NoCalibration=artifacts/ablations/without_calibration/results.json \
  --output-dir artifacts/ablations/report
```

## Robustness Evaluation Summary

The complete release baseline was evaluated on all 13,841 official demonstration images under every scenario. The reported operating-point metrics use the single calibration fitted exclusively on the internal validation splits; no official labels were used for checkpoint selection, calibration, or threshold fitting.

| Evaluation family | Tested settings | Mean AUROC | Worst AUROC (setting) | Lowest Balanced Accuracy |
|---|---|---:|---:|---:|
| Clean | No added degradation | 0.9792 | 0.9792 | 0.8898 |
| JPEG recompression | Quality 90, 70, 50, 30 | 0.9714 | 0.9595 (`Q=50`) | 0.8373 |
| Gaussian blur | Sigma 0.5, 1.0, 2.0 | 0.9623 | 0.9469 (`sigma=2.0`) | 0.8845 |
| Resize | Scale 0.5, 0.25 | 0.9439 | 0.9432 (`scale=0.5`) | 0.8700 |
| Gaussian noise | Sigma 0.02, 0.05, 0.10 | 0.9885 | 0.9838 (`sigma=0.10`) | 0.9395 |
| Colour adjustment and crop | Jitter 20%, crop 80% | 0.9656 | 0.9656 (`jitter=20%`) | 0.8479 |
| **All prescribed single transformations** | 14 scenarios | **0.9683** | **0.9432** (`scale=0.5`) | **0.8373** |
| Additional composed stress tests | 6 ordered pipelines | 0.8968 | 0.8713 (`crop→resize→JPEG`) | 0.7746 |

The mean prescribed-transformation AUROC is only 0.0109 below clean AUROC, and even the worst prescribed transformation remains above 0.94. Noise is handled particularly well; resizing and strong blur are the most difficult single-operation families. Ordered compositions create a larger distribution shift and should therefore be interpreted as a separate stress boundary rather than averaged into the challenge-prescribed result.

Ranking robustness does not guarantee a uniformly safe operating point. The following cases expose the principal class-specific trade-offs:

| Scenario | AUROC | Balanced Accuracy | Real recall | Fake recall | Interpretation |
|---|---:|---:|---:|---:|---|
| Clean | 0.9792 | 0.8898 | 0.9818 | 0.7978 | Strong real-image protection, with lower fake recall at the selected robust threshold |
| Blur, `sigma=2.0` | 0.9469 | 0.8845 | 0.8824 | 0.8867 | The two class recalls remain balanced under the strongest blur |
| Resize, `scale=0.25` | 0.9446 | 0.8784 | 0.8705 | 0.8864 | Severe downsampling reduces ranking but does not collapse either class recall |
| Crop 80% | 0.9656 | 0.8479 | 0.9788 | 0.7170 | Ranking remains strong, but more generated images cross the real decision side |
| Crop→blur→resize→JPEG stress | 0.9168 | 0.7746 | 0.6152 | 0.9340 | The largest observed false-positive shift under compounded degradation |

For presentation, the clearest primary visual is a set of four severity curves for JPEG, blur, resize, and noise, with AUROC on a shared `0.90–1.00` axis and clean AUROC shown as a dashed reference. A second compact panel should compare `clean`, `mean prescribed single`, `worst prescribed single`, and `mean composed` AUROC without mixing composed tests into the official aggregate. Pair this with a real-versus-fake recall dumbbell for the five representative rows above; this makes the distinction between ranking robustness and threshold robustness visually explicit.

## Error Analysis Note

Error cases were selected from the complete official per-image prediction log at the deployed calibrated threshold of `0.5664`. Selection considered both high-confidence clean errors and images that were classified correctly when clean but crossed the decision boundary after a prescribed or composed transformation. Qualitative examples were deduplicated by encoded-file SHA-256 before inspection so that repeated benchmark files did not occupy multiple case-study slots. The observations below are plausible failure hypotheses based on the visible content and the controlled score change, not causal explanations of the learned representation.

| Error type and record prefix | Preview | Scenario and calibrated fake probability | Visual observation and likely failure mode |
|---|---|---|---|
| Clean false positive, `0f9d01575d4d` | <img src="assets/error-analysis/clean-false-positive-food.jpg" width="180" alt="Authentic overhead food photograph classified as generated"> | Clean: `0.9541` | A genuine overhead food photograph has polished stock-photo composition, shallow depth of field, saturated colour, and repeated fine food textures. This combination resembles the highly controlled composition and locally regular texture found in generated training images, producing a confident false accusation even without added degradation. |
| Clean false negative, `ec6b2c9b8785` | <img src="assets/error-analysis/clean-false-negative-water-silhouette.jpg" width="180" alt="Generated transparent-water silhouette classified as authentic"> | Clean: `0.0018` | A DALL·E image depicts an impossible human silhouette formed from transparent water against a sparse studio background. Despite its semantic implausibility, the fluid boundary, lighting, and empty background are internally coherent and contain neither malformed text nor an ordinary face/hand cue. The result suggests that broad real graphic and product-photography coverage can make stylisation alone weak evidence of generation. |
| Transformation-induced false positive, `715742f964dc` | <img src="assets/error-analysis/resize-false-positive-clean.jpg" width="120" alt="Authentic indoor image before resize"> → <img src="assets/error-analysis/resize-false-positive-transformed.jpg" width="120" alt="Authentic indoor image after 0.25x resize and upscale"> | Clean → resize 0.25×: `0.0202 → 0.9701` | The authentic COCO image is an already low-resolution indoor snapshot. Severe downsampling removes camera texture and small object detail, leaving smooth walls, furniture, and soft boundaries. The almost complete score reversal indicates sensitivity to the loss of natural high-frequency evidence rather than to scene semantics alone. |
| Transformation-induced false negative, `ae4ec4f74c8d` | <img src="assets/error-analysis/crop-false-negative-clean.jpg" width="120" alt="Generated fashion display before crop"> → <img src="assets/error-analysis/crop-false-negative-transformed.jpg" width="120" alt="Generated fashion display after 80 percent center crop"> | Clean → centre crop 80%: `0.7945 → 0.0260` | The generated fashion display is initially detected, but cropping enlarges the coherent central person and garments while removing part of the outer box, shadows, wrapping, and peripheral layout. This suggests that important evidence was spatially concentrated near the composition boundary and was not fully preserved by the two-view aggregation. |
| Composed false positive, `8adec5521027` | <img src="assets/error-analysis/composed-false-positive-clean.jpg" width="120" alt="Authentic bathroom before composed degradation"> → <img src="assets/error-analysis/composed-false-positive-transformed.jpg" width="120" alt="Authentic bathroom after crop blur resize and JPEG degradation"> | Clean → crop 80%, blur 1.0, resize 0.25×, JPEG 30: `0.0668 → 0.9900` | A real bathroom photograph becomes dominated by blocky planar surfaces after the ordered stress pipeline. Fine tile, reflection, and material cues are largely erased; the remaining smooth geometry strongly resembles a rendered interior. This is representative of the false-positive increase measured by the stress scenario's `0.6152` real recall. |
| Composed false negative, `4a604bc5c1de` | <img src="assets/error-analysis/composed-false-negative-clean.jpg" width="120" alt="Generated cinematic archive scene before degradation"> → <img src="assets/error-analysis/composed-false-negative-transformed.jpg" width="120" alt="Generated cinematic archive scene after color noise and JPEG degradation"> | Clean → colour jitter, noise 0.02, JPEG 70: `0.6421 → 0.0163` | The generated image already imitates a cinematic archive frame, including black bars, grain-like texture, dramatic lighting, and a subtitle. Added colour/noise/codec effects make those photographic and film-like cues more convincing while suppressing weaker synthesis traces, causing the detector to accept it as authentic. |

The complete record IDs and image paths remain in `artifacts/evaluations/wildfake_official/predictions.jsonl`. The selected source-image previews are copied to `assets/error-analysis/` for GitHub and Hugging Face rendering. Arrow-separated transformed previews reproduce the deterministic full-frame evaluation scenario after standardisation and before the detector's global/local crops. All pictured examples originate from the WildFake official demonstration subset and remain subject to the upstream dataset's terms.

### Calibration and the false-positive trade-off

The affine Platt transform has a positive coefficient and is therefore monotonic: it changes probability interpretation and the operating point, but not AUROC or sample ranking. The selected calibrated threshold is equivalent to requiring an uncalibrated fake probability of approximately `0.8957`, rather than the conventional `0.5`. This conservative shift was selected solely from the four internal clean/degraded ID/DG validation groups to prevent a fake-biased threshold from allowing one easy validation group to hide false positives. It was not selected retrospectively on the official labels.

The transferred operating-point trade-off on the complete official evaluation is:

| Evaluation slice | Operating point | Balanced Accuracy | Real recall | Fake recall |
|---|---|---:|---:|---:|
| Clean | Raw probability ≥ 0.5 | 0.9279 | 0.9424 | 0.9134 |
| Clean | **Internal robust calibration** | 0.8898 | **0.9818** | 0.7978 |
| Prescribed single transformations, pooled | Raw probability ≥ 0.5 | 0.8887 | 0.8455 | **0.9320** |
| Prescribed single transformations, pooled | **Internal robust calibration** | **0.8939** | **0.9519** | 0.8360 |
| Composed stress tests, pooled | Raw probability ≥ 0.5 | 0.7560 | 0.6157 | **0.8963** |
| Composed stress tests, pooled | **Internal robust calibration** | **0.7920** | **0.8189** | 0.7651 |

Calibration substantially reduces false positives under both single and composed transformations and improves their pooled Balanced Accuracy, but it also increases false negatives; on clean official images it sacrifices Balanced Accuracy and fake recall to protect real-image recall. This is an explicit deployment choice rather than a uniformly superior threshold. A high-cost moderation setting should preserve the conservative operating point or introduce an abstention band, whereas a triage or forensic-search setting may choose a lower threshold to recover more generated images. Any such threshold must be selected on representative internal deployment data, never on the protected official demonstration labels.

## Limitation Reflection

The mixed training and validation design broadens both authentic-image domains and generator families, but a complete leave-one-source-out study has not yet been run because of the available training time and compute budget. The held-out domains and generator architectures in `val_dg`, together with the external official evaluation, provide partial evidence of transfer to unseen conditions; they do not establish that performance will remain stable when every constituent dataset or source family is removed in turn. Source-level leave-one-out evaluation is therefore a priority for validating whether any part of the mixture remains disproportionately influential.

RobustFake deliberately focuses on evidence available in decoded image pixels. Authenticity can also be assessed through provenance and acquisition signals such as C2PA manifests, EXIF fields, content credentials, and embedded watermark detectors such as SynthID. Some of these signals may survive transformations that weaken pixel-level forensic traces and can provide stronger positive provenance when present. However, the project primarily uses established datasets because collecting a legally and demographically representative sample of real Internet media was not feasible within the project scope. Such datasets are commonly re-encoded or otherwise preprocessed, and their metadata presence, removal patterns, and platform distribution may not reflect the open Internet. Adding these channels under those conditions could teach dataset provenance rather than media authenticity. A future system should instead fuse pixel evidence with independently validated provenance while treating missing or editable metadata as inconclusive.

CLIP was selected as the visual backbone because its large-scale image-language pretraining offers broad semantic coverage, its intermediate transformer features retain information at several abstraction levels, and freezing it makes repeated robustness experiments feasible under the parameter and compute constraints. This choice is not evidence that CLIP is the optimal backbone for image forensics. Self-supervised encoders such as DINO-family models, other vision-language encoders such as SigLIP, convolutional architectures, and models pretrained specifically on forensic or frequency-domain objectives could provide useful controls or complementary evidence. Time constraints prevented a controlled backbone comparison.

The official demonstration set is also narrower than a deployment environment: its clean subset contains one principal authentic source and one principal generator source, while the applied corruptions are deterministic simulations of redistribution. Real platforms introduce content-dependent resizing, repeated transcoding, screenshots, overlays, mixed editing histories, and changing generator distributions. The additional composed scenarios probe this gap, but they are not a substitute for longitudinal evaluation on naturally circulated media.

Finally, the detector returns a binary probability rather than a provenance proof or generator attribution. The release results show that class ranking can remain strong while a fixed threshold becomes asymmetric under severe composed transformations. Calibration drift, adaptive post-processing, and newly released generators can therefore increase false accusations even when aggregate AUROC appears satisfactory. RobustFake should be used as one signal in a broader review process, with threshold monitoring, an abstention region for uncertain cases, and periodic recalibration on representative deployment data.

## Team Contribution

| Team member | Contribution |
|---|---|
| **Gu Shucheng** | Led the model design and construction of the mixed training dataset. Designed the frozen CLIP multi-layer detector, global/local view aggregation, residual-statistics branch, and paired robustness objectives; developed the source-balanced mixture, domain-generalisation split, and generator/domain holdout strategy; coordinated training, hyperparameter refinement, calibration, and final system integration. |
| **Youlong Xu** | Contributed to the data preparation and quality-control workflow, including source acquisition, image validation and standardisation, bucket-level sampling support, duplicate and leakage checks, and review of dataset diversity and representativeness. |
| **Ruicheng Li** | Contributed to robustness experimentation and evaluation, including degradation settings, validation protocols, official and external evaluation workflows, metric interpretation, and analysis of class-specific errors under severe transformations. |
| **Hanson Yu** | Contributed to reproducibility and software engineering, including configuration and checkpoint workflows, command-line inference, cache-aware execution, testing, and review of installation and reproduction instructions. |
| **Hantong Hong** | Contributed to experimental reporting and presentation, including academic baseline and ablation planning, result aggregation, robustness visualisation, error-analysis organisation, documentation review. |

All team members participated in design discussions, result review, and final presentation preparation.
