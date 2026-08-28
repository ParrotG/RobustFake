"""Train the frozen-CLIP multi-view AIGC detector."""

from __future__ import annotations

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
from aigc_recognizer.model import FrozenClipDetector, create_detector
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


def make_loaders(config: AppConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build reproducible train and validation loaders."""
    train_dataset = AIGCManifestDataset(config, "train")
    val_dataset = AIGCManifestDataset(config, "val")
    generator = torch.Generator().manual_seed(config.project.seed)
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if config.training.num_workers > 0:
        common["persistent_workers"] = config.training.persistent_workers
        common["prefetch_factor"] = config.training.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


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


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["clean_views"].to(device, non_blocking=True),
        batch["transformed_views"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


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
        with _autocast_context(config, device):
            clean, transformed = model.forward_pair(clean_views, transformed_views)
            losses = robust_detection_loss(clean, transformed, labels, config.loss)
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
) -> dict[str, float]:
    """Evaluate clean and deterministically transformed validation views."""
    model.eval()
    labels_all: list[float] = []
    clean_probabilities: list[float] = []
    transformed_probabilities: list[float] = []
    loss_total = 0.0
    example_count = 0
    for batch in tqdm(loader, desc="Validate", leave=False):
        clean_views, transformed_views, labels = _move_batch(batch, device)
        with _autocast_context(config, device):
            clean, transformed = model.forward_pair(clean_views, transformed_views)
            losses = robust_detection_loss(clean, transformed, labels, config.loss)
        examples = labels.numel()
        example_count += examples
        loss_total += float(losses["total"]) * examples
        labels_all.extend(labels.cpu().tolist())
        clean_probabilities.extend(torch.sigmoid(clean.logits).float().cpu().tolist())
        transformed_probabilities.extend(
            torch.sigmoid(transformed.logits).float().cpu().tolist()
        )
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
    return result


def _source_revision(config: AppConfig) -> str:
    with Path(config.data.manifest_path).open("r", encoding="utf-8") as handle:
        first_record = json.loads(next(line for line in handle if line.strip()))
    return str(first_record["source_revision"])


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
    val_loader: DataLoader[Any],
) -> Path:
    """Run the model and optimization lifecycle using initialized loaders."""
    detector = model if model is not None else create_detector(config.model)
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
        val_metrics = evaluate(detector, val_loader, config, device)
        monitored = val_metrics["monitor_auroc"]
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
                "validation": val_metrics,
                "best_metric": best_metric,
            },
        )
        LOGGER.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f monitor_auroc=%.4f",
            epoch + 1,
            train_metrics["total"],
            val_metrics["loss"],
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
    train_loader, val_loader = make_loaders(config)
    try:
        return _run_training_loop(config, model, device, train_loader, val_loader)
    finally:
        _shutdown_loader_workers(train_loader)
        _shutdown_loader_workers(val_loader)
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
