"""Train the frozen-CLIP multi-view AIGC detector."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import signal
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.dataset import AIGCManifestDataset, validate_preparation
from aigc_recognizer.losses import robust_detection_loss
from aigc_recognizer.metrics import binary_metrics
from aigc_recognizer.model import (
    EncodedViews,
    FrozenClipDetector,
    create_cached_detector,
    create_detector,
)
from aigc_recognizer.utils import (
    append_metric,
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    seed_worker,
    write_yaml,
)

LOGGER = logging.getLogger(__name__)


def resolve_device(config: AppConfig) -> torch.device:
    """Resolve the configured device with explicit availability checks."""
    requested = config.training.device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def _validate_training_distribution(dataset: AIGCManifestDataset) -> None:
    """Require the prepared training manifest to contain both balanced labels."""
    label_counts: dict[int, int] = {}
    for record in dataset.records:
        label = int(record["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    if set(label_counts) != {0, 1}:
        raise RuntimeError("Training requires both real and fake samples.")
    if label_counts[0] != label_counts[1]:
        raise RuntimeError(
            "Training manifest labels must be balanced before full-coverage sampling."
        )


def make_loaders(config: AppConfig) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build a full-coverage train loader and separate ID/DG validation loaders."""
    if config.feature_cache.use_for_training:
        from aigc_recognizer.feature_cache import CachedFeatureDataset

        train_dataset = CachedFeatureDataset(config, "train")
        val_id_dataset = CachedFeatureDataset(config, "val_id")
        val_dg_dataset = CachedFeatureDataset(config, "val_dg")
        labels = train_dataset.tensors["label"][0]
        real_count = int((labels == 0).sum())
        fake_count = int((labels == 1).sum())
        if real_count != fake_count or real_count + fake_count != len(train_dataset):
            raise RuntimeError("Cached training features must contain balanced binary labels.")
    else:
        train_dataset = AIGCManifestDataset(config, "train")
        try:
            val_id_dataset = AIGCManifestDataset(config, "val_id")
            val_dg_dataset = AIGCManifestDataset(config, "val_dg")
        except ValueError as exc:
            if "contains no records" not in str(exc):
                raise
            # Retain compatibility with small legacy smoke manifests.
            val_id_dataset = AIGCManifestDataset(config, "val")
            val_dg_dataset = val_id_dataset
        _validate_training_distribution(train_dataset)
    generator = torch.Generator().manual_seed(config.project.seed)
    worker_count = 0 if config.feature_cache.use_for_training else config.training.num_workers
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": worker_count,
        "pin_memory": config.training.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if worker_count > 0:
        common["persistent_workers"] = config.training.persistent_workers
        common["prefetch_factor"] = config.training.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_id_loader = DataLoader(val_id_dataset, shuffle=False, drop_last=False, **common)
    val_dg_loader = DataLoader(val_dg_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_id_loader, val_dg_loader


def _shutdown_loader_workers(loader: DataLoader[Any]) -> None:
    """Stop persistent workers immediately instead of waiting for garbage collection."""
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None


def make_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_fraction: float
) -> LambdaLR:
    """Create a linear warmup followed by cosine decay."""
    warmup_steps = round(total_steps * warmup_fraction)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, multiplier)


def _autocast_context(config: AppConfig, device: torch.device) -> Any:
    enabled = config.training.amp and device.type == "cuda"
    if not enabled:
        return nullcontext()
    dtype = torch.float16 if config.training.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor | EncodedViews, torch.Tensor | EncodedViews, torch.Tensor]:
    labels = batch["label"].to(device, non_blocking=True)
    if "clean_final" not in batch:
        return (
            batch["clean_views"].to(device, non_blocking=True),
            batch["transformed_views"].to(device, non_blocking=True),
            labels,
        )

    def encoded(prefix: str) -> EncodedViews:
        intermediate = batch[f"{prefix}_intermediate"]
        if intermediate.shape[-2] == 0:
            intermediate = None
        elif intermediate is not None:
            intermediate = intermediate.to(device, non_blocking=True)
        return EncodedViews(
            final=batch[f"{prefix}_final"].to(device, non_blocking=True),
            intermediate=intermediate,
        )

    return encoded("clean"), encoded("transformed"), labels


def _forward_pair(
    model: FrozenClipDetector,
    clean: torch.Tensor | EncodedViews,
    transformed: torch.Tensor | EncodedViews,
) -> tuple[Any, Any]:
    if isinstance(clean, EncodedViews) and isinstance(transformed, EncodedViews):
        return model.forward_pair_encoded(clean, transformed)
    if isinstance(clean, torch.Tensor) and isinstance(transformed, torch.Tensor):
        return model.forward_pair(clean, transformed)
    raise TypeError("Clean and transformed batch representations must have matching types.")


def _group_metrics(
    labels: list[float], probabilities: list[float], threshold: float
) -> dict[str, Any]:
    """Report useful threshold metrics for both mixed- and single-class groups."""
    targets = [int(value) for value in labels]
    predictions = [int(value >= threshold) for value in probabilities]
    real_count = sum(value == 0 for value in targets)
    fake_count = sum(value == 1 for value in targets)
    result: dict[str, Any] = {
        "real_count": real_count,
        "fake_count": fake_count,
        "mean_probability": sum(probabilities) / max(1, len(probabilities)),
        "predicted_fake_rate": sum(predictions) / max(1, len(predictions)),
        "accuracy": sum(
            prediction == target
            for prediction, target in zip(predictions, targets)
        )
        / max(1, len(targets)),
    }
    if fake_count:
        result["fake_recall"] = sum(
            prediction == 1 and target == 1
            for prediction, target in zip(predictions, targets)
        ) / fake_count
    if real_count:
        result["real_recall"] = sum(
            prediction == 0 and target == 0
            for prediction, target in zip(predictions, targets)
        ) / real_count
        result["false_positive_rate"] = 1.0 - result["real_recall"]
    if real_count and fake_count:
        result.update(binary_metrics(labels, probabilities, threshold))
    return result


def _consistency_scale(
    epoch: int, batch_index: int, batch_count: int, rampup_epochs: int
) -> float:
    """Linearly introduce consistency over the configured opening epochs."""
    if rampup_epochs == 0:
        return 1.0
    progress = epoch + (batch_index + 1) / max(1, batch_count)
    return min(1.0, progress / rampup_epochs)


def train_one_epoch(
    model: FrozenClipDetector,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: AppConfig,
    device: torch.device,
    epoch: int,
) -> tuple[dict[str, float], int]:
    """Train one epoch and return aggregate losses and optimizer step count."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = config.training.gradient_accumulation_steps
    totals: dict[str, float] = {}
    example_count = 0
    optimizer_steps = 0
    progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=False)
    for batch_index, batch in enumerate(progress):
        clean_views, transformed_views, labels = _move_batch(batch, device)
        consistency_scale = _consistency_scale(
            epoch,
            batch_index,
            len(loader),
            config.loss.consistency_rampup_epochs,
        )
        with _autocast_context(config, device):
            clean, transformed = _forward_pair(model, clean_views, transformed_views)
            losses = robust_detection_loss(
                clean,
                transformed,
                labels,
                config.loss,
                consistency_scale=consistency_scale,
            )
            scaled_loss = losses["total"] / accumulation
        scaler.scale(scaled_loss).backward()
        examples = labels.numel()
        example_count += examples
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * examples

        last_batch = batch_index + 1 == len(loader)
        if (batch_index + 1) % accumulation == 0 or last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.training.gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
        progress.set_postfix(loss=f"{float(losses['total'].detach()):.4f}")
    return {name: value / max(1, example_count) for name, value in totals.items()}, optimizer_steps


@torch.no_grad()
def evaluate(
    model: FrozenClipDetector,
    loader: DataLoader[Any],
    config: AppConfig,
    device: torch.device,
    *,
    consistency_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate clean and deterministically transformed validation views."""
    model.eval()
    labels_all: list[float] = []
    clean_probabilities: list[float] = []
    transformed_probabilities: list[float] = []
    group_values: dict[str, list[str]] = {
        "source_dataset": [],
        "real_source": [],
        "generator_family": [],
        "architecture": [],
    }
    loss_total = 0.0
    example_count = 0
    for batch in tqdm(loader, desc="Validate", leave=False):
        clean_views, transformed_views, labels = _move_batch(batch, device)
        with _autocast_context(config, device):
            clean, transformed = _forward_pair(model, clean_views, transformed_views)
            losses = robust_detection_loss(
                clean,
                transformed,
                labels,
                config.loss,
                consistency_scale=consistency_scale,
            )
        examples = labels.numel()
        example_count += examples
        loss_total += float(losses["total"]) * examples
        labels_all.extend(labels.cpu().tolist())
        clean_probabilities.extend(torch.sigmoid(clean.logits).float().cpu().tolist())
        transformed_probabilities.extend(
            torch.sigmoid(transformed.logits).float().cpu().tolist()
        )
        for field in group_values:
            group_values[field].extend(str(value) for value in batch[field])
    clean_metrics = binary_metrics(labels_all, clean_probabilities, config.training.threshold)
    transformed_metrics = binary_metrics(
        labels_all, transformed_probabilities, config.training.threshold
    )
    result = {"loss": loss_total / max(1, example_count)}
    result.update({f"clean_{name}": value for name, value in clean_metrics.items()})
    result.update(
        {f"transformed_{name}": value for name, value in transformed_metrics.items()}
    )
    result["monitor_auroc"] = 0.5 * (
        result["clean_auroc"] + result["transformed_auroc"]
    )
    result["groups"] = {}
    for field, values in group_values.items():
        result["groups"][field] = {}
        names = sorted({value for value in values if value})
        for name in names:
            if field == "source_dataset":
                indices = [index for index, value in enumerate(values) if value == name]
            elif field == "real_source":
                indices = [
                    index for index, value in enumerate(values)
                    if labels_all[index] == 1 or value == name
                ]
            else:
                indices = [
                    index for index, value in enumerate(values)
                    if labels_all[index] == 0 or value == name
                ]
            selected_labels = [labels_all[index] for index in indices]
            clean_group = _group_metrics(
                selected_labels,
                [clean_probabilities[index] for index in indices],
                config.training.threshold,
            )
            transformed_group = _group_metrics(
                selected_labels,
                [transformed_probabilities[index] for index in indices],
                config.training.threshold,
            )
            result["groups"][field][name] = {
                "count": len(indices),
                **{f"clean_{key}": value for key, value in clean_group.items()},
                **{f"transformed_{key}": value for key, value in transformed_group.items()},
            }
    return result


def _source_revision(config: AppConfig) -> str:
    revisions: dict[str, str] = {}
    with Path(config.data.manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            source = str(record.get("source_dataset", "legacy"))
            revision = str(record["source_revision"])
            if source in revisions and revisions[source] != revision:
                raise RuntimeError(f"Manifest source {source} contains multiple revisions.")
            revisions[source] = revision
    if len(revisions) == 1:
        return next(iter(revisions.values()))
    digest = hashlib.sha256(json.dumps(revisions, sort_keys=True).encode()).hexdigest()
    return f"mixed:{digest}"


def _checkpoint_payload(
    model: FrozenClipDetector,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: AppConfig,
    epoch: int,
    global_step: int,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "trainable_model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config.to_dict(),
        "source_revision": _source_revision(config),
        "parameter_counts": model.parameter_counts(),
        "backbone": {
            "name": config.model.backbone_name,
            "pretrained": config.model.pretrained,
        },
    }


def _resume(
    path: str,
    model: FrozenClipDetector,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: AppConfig,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected_backbone = {
        "name": config.model.backbone_name,
        "pretrained": config.model.pretrained,
    }
    if checkpoint.get("backbone") != expected_backbone:
        raise RuntimeError("Checkpoint backbone does not match the active configuration.")
    if checkpoint.get("source_revision") != _source_revision(config):
        raise RuntimeError("Checkpoint dataset revision does not match the active manifest.")
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint["best_metric"]),
    )


def _run_training_loop(
    config: AppConfig,
    model: FrozenClipDetector | None,
    device: torch.device,
    train_loader: DataLoader[Any],
    val_id_loader: DataLoader[Any],
    val_dg_loader: DataLoader[Any],
) -> Path:
    """Run the model and optimization lifecycle using initialized loaders."""
    detector = model if model is not None else (
        create_cached_detector(config.model)
        if config.feature_cache.use_for_training
        else create_detector(config.model)
    )
    detector.to(device)
    counts = detector.parameter_counts()
    if counts["total"] >= 2_000_000_000:
        raise RuntimeError("The configured model violates the 2B parameter constraint.")
    if counts["trainable"] > 5_000_000:
        raise RuntimeError("The trainable detector head exceeds the 5M parameter budget.")
    LOGGER.info("Model parameters: total=%d trainable=%d", counts["total"], counts["trainable"])

    trainable_parameters = [
        parameter for parameter in detector.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / config.training.gradient_accumulation_steps
    )
    scheduler = make_scheduler(
        optimizer,
        steps_per_epoch * config.training.epochs,
        config.training.warmup_fraction,
    )
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_dir = Path(config.output.root_dir) / config.project.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(config.to_dict(), run_dir / "resolved_config.yaml")
    metric_path = run_dir / "metrics.jsonl"
    session_id = uuid.uuid4().hex
    start_epoch, global_step, best_metric = 0, 0, -math.inf
    if config.training.resume_from:
        start_epoch, global_step, best_metric = _resume(
            config.training.resume_from,
            detector,
            optimizer,
            scheduler,
            scaler,
            config,
            device,
        )
        LOGGER.info("Resumed training from epoch %d", start_epoch)

    epochs_without_improvement = 0
    best_path = run_dir / "best.pt"
    for epoch in range(start_epoch, config.training.epochs):
        train_metrics, optimizer_steps = train_one_epoch(
            detector,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            config,
            device,
            epoch,
        )
        global_step += optimizer_steps
        validation_consistency_scale = _consistency_scale(
            epoch,
            len(train_loader) - 1,
            len(train_loader),
            config.loss.consistency_rampup_epochs,
        )
        val_id_metrics = evaluate(
            detector,
            val_id_loader,
            config,
            device,
            consistency_scale=validation_consistency_scale,
        )
        val_dg_metrics = evaluate(
            detector,
            val_dg_loader,
            config,
            device,
            consistency_scale=validation_consistency_scale,
        )
        monitored = 0.5 * (
            val_id_metrics["monitor_auroc"] + val_dg_metrics["monitor_auroc"]
        )
        improved = math.isfinite(monitored) and monitored > best_metric
        if improved:
            best_metric = monitored
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            detector,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch,
            global_step,
            best_metric,
        )
        if improved:
            atomic_torch_save(payload, best_path)
        if config.output.save_last:
            atomic_torch_save(payload, run_dir / "last.pt")
        append_metric(
            metric_path,
            {
                "run_name": config.project.run_name,
                "session_id": session_id,
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation_id": val_id_metrics,
                "validation_dg": val_dg_metrics,
                "validation": val_id_metrics,
                "best_metric": best_metric,
            },
        )
        LOGGER.info(
            "Epoch %d: train_loss=%.4f id_loss=%.4f dg_loss=%.4f monitor_auroc=%.4f",
            epoch + 1,
            train_metrics["total"],
            val_id_metrics["loss"],
            val_dg_metrics["loss"],
            monitored,
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            LOGGER.info("Early stopping after %d unimproved epochs.", epochs_without_improvement)
            break
    if not best_path.is_file():
        raise RuntimeError("Training finished without a valid two-class validation AUROC.")
    return best_path


def run_training(config: AppConfig, model: FrozenClipDetector | None = None) -> Path:
    """Run end-to-end training and always release loader and CUDA resources."""
    validate_preparation(config)
    seed_everything(config.project.seed)
    device = resolve_device(config)
    LOGGER.info("Training on device %s", device)
    train_loader, val_id_loader, val_dg_loader = make_loaders(config)
    try:
        return _run_training_loop(
            config, model, device, train_loader, val_id_loader, val_dg_loader
        )
    finally:
        _shutdown_loader_workers(train_loader)
        _shutdown_loader_workers(val_id_loader)
        if val_dg_loader is not val_id_loader:
            _shutdown_loader_workers(val_dg_loader)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    """Run the public training command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Train a robust multi-view AIGC detector.")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    previous_signal_handlers: dict[signal.Signals, Any] = {}

    def interrupt_training(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        raise KeyboardInterrupt(f"Training received {signal_name}.")

    for signal_name in ("SIGHUP", "SIGTERM"):
        selected_signal = getattr(signal, signal_name, None)
        if selected_signal is not None:
            previous_signal_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, interrupt_training)
    try:
        best_path = run_training(config)
    except KeyboardInterrupt as exc:
        LOGGER.warning("Training interrupted; data loader workers are shutting down. %s", exc)
        raise SystemExit(130) from exc
    finally:
        for selected_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(selected_signal, previous_handler)
    LOGGER.info("Training completed. Best checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
