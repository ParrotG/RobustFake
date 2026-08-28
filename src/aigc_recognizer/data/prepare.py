"""Stream and stratify a bounded Community Forensics training subset."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import signal
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps, UnidentifiedImageError

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config

LOGGER = logging.getLogger(__name__)
GIB = 1024**3


class PreparationError(RuntimeError):
    """Raised when safe sampling cannot satisfy the configured quotas."""


@dataclass(frozen=True)
class DecodedImage:
    """Validated image information used during acquisition."""

    width: int
    height: int
    image_format: str
    perceptual_hash: str


def stable_bucket(value: str, buckets: int = 100) -> int:
    """Map a string to a stable bucket independently of Python hash randomization."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def normalize_architecture(value: object) -> str:
    """Normalize the dataset architecture value to a configured training group."""
    architecture = str(value or "other")
    return architecture if architecture in {"LatDiff", "GAN", "PixDiff"} else "other"


def effective_real_source(row: Mapping[str, Any]) -> str:
    """Resolve the real source from the field actually populated by this dataset."""
    real_source = str(row.get("real_source") or "").strip()
    if real_source.casefold() in {"", "n/a", "na", "none", "unknown"}:
        return str(row.get("model_name") or "unknown").strip()
    return real_source


def architecture_targets(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Convert fractional architecture targets to integers with an exact total."""
    raw = {name: total * ratio for name, ratio in ratios.items()}
    targets = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(raw, key=lambda name: raw[name] - targets[name], reverse=True)
    for name in order[:remainder]:
        targets[name] += 1
    return targets


def row_split(row: Mapping[str, Any], train_generator_percent: int) -> str:
    """Assign fake images by generator and real images by stable source identity."""
    label = int(row.get("label", -1))
    if label == 1:
        model_name = str(row.get("model_name") or "unknown-generator")
        if str(row.get("subset") or "").casefold() == "systematic":
            identity = model_name
        else:
            identity = f"{model_name}|{row.get('image_name') or row.get('id') or 'unknown-image'}"
    else:
        identity = "|".join(
            [
                str(row.get("image_name") or row.get("id") or "unknown-image"),
                effective_real_source(row),
            ]
        )
    return "train" if stable_bucket(identity) < train_generator_percent else "val"


def is_forbidden(row: Mapping[str, Any], config: AppConfig) -> bool:
    """Apply challenge leakage guards and content safety filters."""
    label = int(row.get("label", -1))
    if label not in {0, 1}:
        return True
    if config.data.exclude_nsfw and bool(row.get("nsfw_flag", False)):
        return True
    if label == 1:
        model_name = str(row.get("model_name") or "").casefold()
        return any(token.casefold() in model_name for token in config.data.excluded_generator_tokens)
    real_source = effective_real_source(row).casefold()
    return any(token.casefold() in real_source for token in config.data.excluded_real_source_tokens)


def extract_image_bytes(value: Any) -> bytes:
    """Extract bytes from the representations emitted by Parquet readers."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, Mapping) and value.get("bytes") is not None:
        return bytes(value["bytes"])
    raise PreparationError("The image_data field does not contain in-memory image bytes.")


def validate_and_describe_image(image_bytes: bytes, config: AppConfig) -> DecodedImage:
    """Safely decode an image and compute its perceptual identity."""
    import imagehash

    Image.MAX_IMAGE_PIXELS = config.data.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                image = ImageOps.exif_transpose(image).convert("RGB")
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise PreparationError("Decoded image has an invalid size.")
                image_format = (image.format or "").lower()
                perceptual_hash = str(
                    imagehash.phash(image, hash_size=config.data.perceptual_hash_size)
                )
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PreparationError("Image exceeds the configured safe pixel limit.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreparationError("Image decoding failed.") from exc
    return DecodedImage(width, height, image_format, perceptual_hash)


def _extension(format_hint: object, decoded_format: str) -> str:
    value = str(format_hint or decoded_format or "bin").casefold()
    aliases = {"jpeg": "jpg", "tif": "tiff"}
    value = aliases.get(value, value)
    return value if value in {"jpg", "png", "webp", "bmp", "tiff"} else "bin"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write image bytes without exposing a partially written destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _config_fingerprint(config: AppConfig, revision: str) -> str:
    data_config = config.to_dict()["data"]
    for operational_key in (
        "audit_path",
        "checkpoint_every_scanned",
        "hf_auth",
        "manifest_path",
        "max_download_gb",
        "max_scanned",
        "network_max_retries",
        "network_retry_base_seconds",
        "output_dir",
    ):
        data_config.pop(operational_key, None)
    relevant = {
        "data": data_config,
        "seed": config.project.seed,
        "revision": revision,
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _targets(config: AppConfig) -> tuple[dict[tuple[str, int], int], dict[str, dict[str, int]]]:
    class_targets = {
        ("train", 0): config.data.train_per_class,
        ("train", 1): config.data.train_per_class,
        ("val", 0): config.data.val_per_class,
        ("val", 1): config.data.val_per_class,
    }
    fake_targets = {
        "train": architecture_targets(
            config.data.train_per_class, config.data.architecture_ratios
        ),
        "val": architecture_targets(config.data.val_per_class, config.data.architecture_ratios),
    }
    return class_targets, fake_targets


def _quotas_complete(
    class_counts: Counter[tuple[str, int]],
    architecture_counts: Counter[tuple[str, str]],
    class_targets: Mapping[tuple[str, int], int],
    fake_targets: Mapping[str, Mapping[str, int]],
) -> bool:
    return all(class_counts[key] >= value for key, value in class_targets.items()) and all(
        architecture_counts[(split, architecture)] >= target
        for split, targets in fake_targets.items()
        for architecture, target in targets.items()
    )


def sample_rows(
    rows: Iterable[Mapping[str, Any]],
    config: AppConfig,
    revision: str,
) -> dict[str, Any]:
    """Consume a bounded iterable and atomically produce a stratified manifest."""
    output_dir = Path(config.data.output_dir)
    manifest_path = Path(config.data.manifest_path)
    audit_path = Path(config.data.audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_targets, fake_targets = _targets(config)
    fingerprint = _config_fingerprint(config, revision)
    class_counts: Counter[tuple[str, int]] = Counter()
    architecture_counts: Counter[tuple[str, str]] = Counter()
    model_counts: Counter[tuple[str, str]] = Counter()
    real_source_counts: Counter[tuple[str, str]] = Counter()
    skipped: Counter[str] = Counter()
    seen_sha256: set[str] = set()
    seen_perceptual_hashes: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    selected_bytes = 0
    scanned = 0
    resume_scanned = 0
    stop_reason = "maximum rows scanned"

    existing_audit: dict[str, Any] | None = None
    if manifest_path.is_file() and audit_path.is_file():
        try:
            existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise PreparationError(f"Existing audit cannot be read: {audit_path}") from exc
        fingerprint_matches = existing_audit.get("sampling_config_sha256") == fingerprint
        if fingerprint_matches:
            if bool(existing_audit.get("complete")):
                LOGGER.info("The prepared dataset is already complete; no download is needed.")
                return existing_audit
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                content_id = str(record["id"])
                image_path = output_dir / str(record["path"])
                if not image_path.is_file():
                    raise PreparationError(f"Resume image is missing: {image_path}")
                records[content_id] = record
                seen_sha256.add(content_id)
                perceptual_hash = record.get("perceptual_hash")
                if perceptual_hash:
                    seen_perceptual_hashes.add(str(perceptual_hash))
                selected_bytes += image_path.stat().st_size
                split, label = str(record["split"]), int(record["label"])
                class_counts[(split, label)] += 1
                if label == 1:
                    architecture_counts[(split, str(record["architecture"]))] += 1
                    model_counts[(split, str(record["model_name"]))] += 1
                else:
                    real_source_counts[(split, str(record["real_source"]))] += 1
            skipped.update(existing_audit.get("skipped", {}))
            resume_scanned = int(existing_audit.get("scanned", 0))
            scanned = resume_scanned
            LOGGER.info(
                "Resuming preparation from %d selected images after %d scanned rows.",
                len(records),
                resume_scanned,
            )
        else:
            LOGGER.warning(
                "Existing partial data uses a different sampling configuration; "
                "the manifest will be rebuilt while matching image files are reused."
            )

    def make_audit(complete: bool, reason: str) -> dict[str, Any]:
        audit = {
            "complete": complete,
            "source_mode": "pinned_local_shards",
            "stop_reason": reason,
            "source_revision": revision,
            "sampling_config_sha256": fingerprint,
            "scanned": scanned,
            "selected": len(records),
            "selected_bytes": selected_bytes,
            "class_targets": {f"{split}:{label}": value for (split, label), value in class_targets.items()},
            "class_counts": {f"{split}:{label}": class_counts[(split, label)] for split, label in class_targets},
            "fake_architecture_targets": fake_targets,
            "fake_architecture_counts": {
                split: {
                    architecture: architecture_counts[(split, architecture)]
                    for architecture in targets
                }
                for split, targets in fake_targets.items()
            },
            "unique_fake_generators": {
                split: len({model for (record_split, model), count in model_counts.items() if record_split == split and count})
                for split in ("train", "val")
            },
            "skipped": dict(skipped),
        }
        return audit

    def checkpoint(complete: bool, reason: str) -> dict[str, Any]:
        manifest_lines = [
            json.dumps(record, sort_keys=True, ensure_ascii=False)
            for record in sorted(
                records.values(), key=lambda item: (item["split"], item["label"], item["id"])
            )
        ]
        _atomic_write_text(
            manifest_path, "\n".join(manifest_lines) + ("\n" if manifest_lines else "")
        )
        audit = make_audit(complete, reason)
        _atomic_write_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
        return audit

    try:
        for row_position, row in enumerate(rows, start=1):
            if row_position <= resume_scanned:
                continue
            if row_position > config.data.max_scanned:
                scanned = config.data.max_scanned
                break
            if row_position % config.data.checkpoint_every_scanned == 0:
                scanned = row_position - 1
                checkpoint(False, "periodic checkpoint")
                LOGGER.info(
                    "Preparation checkpoint: scanned=%d selected=%d stored=%.2f GiB.",
                    scanned,
                    len(records),
                    selected_bytes / GIB,
                )
            scanned = row_position
            if is_forbidden(row, config):
                skipped["filtered"] += 1
                continue

            label = int(row["label"])
            split = row_split(row, config.data.train_generator_percent)
            if class_counts[(split, label)] >= class_targets[(split, label)]:
                skipped["class quota full"] += 1
                continue

            architecture = "real" if label == 0 else normalize_architecture(row.get("architecture"))
            model_name = str(row.get("model_name") or ("real" if label == 0 else "unknown"))
            subset = str(row.get("subset") or "unknown")
            real_source = effective_real_source(row)

            if label == 1:
                if architecture_counts[(split, architecture)] >= fake_targets[split][architecture]:
                    skipped["architecture quota full"] += 1
                    continue
                model_cap = (
                    config.data.systematic_per_model_cap
                    if subset.casefold() == "systematic"
                    else config.data.non_systematic_per_model_cap
                )
                if model_counts[(split, model_name)] >= model_cap:
                    skipped["model quota full"] += 1
                    continue
            else:
                source_cap = max(
                    1,
                    math.ceil(class_targets[(split, 0)] * config.data.max_real_source_fraction),
                )
                if real_source_counts[(split, real_source)] >= source_cap:
                    skipped["real source quota full"] += 1
                    continue

            try:
                image_bytes = extract_image_bytes(row.get("image_data"))
            except PreparationError:
                skipped["missing image bytes"] += 1
                continue
            content_id = hashlib.sha256(image_bytes).hexdigest()
            if config.data.exact_deduplication and content_id in seen_sha256:
                skipped["exact duplicate"] += 1
                continue
            try:
                decoded = validate_and_describe_image(image_bytes, config)
            except PreparationError:
                skipped["decode failure"] += 1
                continue
            if (
                config.data.perceptual_deduplication
                and decoded.perceptual_hash in seen_perceptual_hashes
            ):
                skipped["perceptual duplicate"] += 1
                continue
            if selected_bytes + len(image_bytes) > config.data.max_download_gb * GIB:
                stop_reason = "configured byte budget reached"
                break

            extension = _extension(row.get("format"), decoded.image_format)
            relative_path = Path("images") / split / ("fake" if label else "real") / f"{content_id}.{extension}"
            absolute_path = output_dir / relative_path
            if not absolute_path.exists():
                _atomic_write_bytes(absolute_path, image_bytes)

            record = {
                "id": content_id,
                "image_name": str(row.get("image_name") or "unknown-image"),
                "path": str(relative_path),
                "label": label,
                "split": split,
                "model_name": model_name,
                "architecture": architecture,
                "subset": subset,
                "real_source": real_source,
                "format": extension,
                "width": decoded.width,
                "height": decoded.height,
                "source_revision": revision,
                "perceptual_hash": decoded.perceptual_hash,
            }
            records[content_id] = record
            seen_sha256.add(content_id)
            seen_perceptual_hashes.add(decoded.perceptual_hash)
            selected_bytes += len(image_bytes)
            class_counts[(split, label)] += 1
            if label == 1:
                architecture_counts[(split, architecture)] += 1
                model_counts[(split, model_name)] += 1
            else:
                real_source_counts[(split, real_source)] += 1

            if _quotas_complete(class_counts, architecture_counts, class_targets, fake_targets):
                stop_reason = "all quotas satisfied"
                break
        else:
            stop_reason = "source rows exhausted"
    except BaseException as exc:
        checkpoint(False, f"interrupted by {type(exc).__name__}")
        raise

    complete = _quotas_complete(class_counts, architecture_counts, class_targets, fake_targets)
    audit = checkpoint(complete, stop_reason)
    if not complete:
        raise PreparationError(
            f"Sampling stopped before all quotas were satisfied: {stop_reason}. "
            f"Inspect {audit_path} and adjust only the centralized configuration."
        )
    return audit


def _resolve_revision(config: AppConfig, token: str | bool | None) -> str:
    if config.data.revision:
        return config.data.revision
    existing_audit = Path(config.data.audit_path)
    if existing_audit.is_file():
        try:
            revision = json.loads(existing_audit.read_text(encoding="utf-8")).get("source_revision")
            if revision:
                return str(revision)
        except (json.JSONDecodeError, OSError):
            pass
    from huggingface_hub import HfApi

    return str(HfApi(token=token).dataset_info(config.data.repo_id).sha)


def _resolve_hf_token(config: AppConfig) -> str | bool | None:
    """Resolve a saved Hugging Face credential without logging its value."""
    if config.data.hf_auth == "disabled":
        LOGGER.info("Hugging Face authentication is disabled by configuration.")
        return False
    from huggingface_hub import get_token

    token = get_token()
    if token:
        LOGGER.info("Using the saved Hugging Face credential for Hub requests.")
        return token
    if config.data.hf_auth == "required":
        raise PreparationError(
            "Hugging Face authentication is required, but no saved token was found. "
            "Run 'hf auth login' in the same user environment."
        )
    LOGGER.warning(
        "No saved Hugging Face credential was found; Hub requests will be unauthenticated."
    )
    return None


def _is_retryable_network_error(error: BaseException) -> bool:
    """Recognize common Hub transport failures through wrapped exception chains."""
    current: BaseException | None = error
    visited: set[int] = set()
    network_modules = (
        "aiohttp",
        "fsspec",
        "httpcore",
        "httpx",
        "huggingface_hub",
        "requests",
        "urllib3",
    )
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError, OSError)):
            return True
        if type(current).__module__.startswith(network_modules):
            return True
        current = current.__cause__ or current.__context__
    return False


def _shard_filename(index: int) -> str:
    return f"data/HFCF_small_{index}.parquet"


def _preflight_local_shards(
    config: AppConfig, revision: str, token: str | bool | None
) -> list[str]:
    """Resolve selected shard sizes without downloading their payloads."""
    from huggingface_hub import hf_hub_download

    filenames = [_shard_filename(index) for index in config.data.shard_indices]
    total_size = 0
    for filename in filenames:
        info = hf_hub_download(
            repo_id=config.data.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=config.data.shard_cache_dir,
            token=token,
            dry_run=True,
        )
        total_size += int(info.file_size)
    total_gib = total_size / GIB
    if total_gib > config.data.max_shard_cache_gb:
        raise PreparationError(
            f"Selected source shards require {total_gib:.2f} GiB, exceeding "
            f"data.max_shard_cache_gb={config.data.max_shard_cache_gb:.2f}."
        )
    LOGGER.info(
        "Selected %d original Parquet shards (%.2f GiB maximum local cache).",
        len(filenames),
        total_gib,
    )
    return filenames


def _iter_local_shards(
    config: AppConfig,
    revision: str,
    token: str | bool | None,
    filenames: Iterable[str],
) -> Iterable[Mapping[str, Any]]:
    """Download resumable source files one at a time and iterate them offline."""
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    for position, filename in enumerate(filenames, start=1):
        LOGGER.info(
            "Ensuring source shard %d/%d is cached: %s",
            position,
            len(config.data.shard_indices),
            filename,
        )
        local_path = hf_hub_download(
            repo_id=config.data.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=config.data.shard_cache_dir,
            token=token,
        )
        LOGGER.info("Reading cached source shard: %s", local_path)
        parquet_file = parquet.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(batch_size=32):
            yield from batch.to_pylist()


def prepare_dataset(config: AppConfig) -> dict[str, Any]:
    """Download pinned source shards and run bounded stratified sampling."""
    token = _resolve_hf_token(config)
    for attempt in range(config.data.network_max_retries + 1):
        try:
            revision = _resolve_revision(config, token)
            LOGGER.info("Using dataset revision %s", revision)
            filenames = _preflight_local_shards(config, revision, token)
            return sample_rows(
                _iter_local_shards(config, revision, token, filenames),
                config,
                revision,
            )
        except PreparationError:
            raise
        except Exception as exc:
            if not _is_retryable_network_error(exc) or attempt >= config.data.network_max_retries:
                raise
            delay = min(
                60.0,
                config.data.network_retry_base_seconds * (2**attempt),
            )
            LOGGER.warning(
                "Hub shard acquisition failed with %s. Retrying from the latest checkpoint in %.1f "
                "seconds (%d/%d).",
                type(exc).__name__,
                delay,
                attempt + 1,
                config.data.network_max_retries,
            )
            time.sleep(delay)
    raise AssertionError("The retry loop terminated unexpectedly.")


def main() -> None:
    """Run the public dataset preparation command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Prepare a bounded Community Forensics subset.")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    previous_signal_handlers: dict[signal.Signals, Any] = {}

    def interrupt_with_checkpoint(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        raise PreparationError(f"Dataset preparation received {signal_name}.")

    for signal_name in ("SIGHUP", "SIGTERM"):
        selected_signal = getattr(signal, signal_name, None)
        if selected_signal is not None:
            previous_signal_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, interrupt_with_checkpoint)
    try:
        audit = prepare_dataset(config)
    except PreparationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    finally:
        for selected_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(selected_signal, previous_handler)
    LOGGER.info(
        "Dataset preparation completed with %d images after scanning %d rows.",
        audit["selected"],
        audit["scanned"],
    )


if __name__ == "__main__":
    main()
