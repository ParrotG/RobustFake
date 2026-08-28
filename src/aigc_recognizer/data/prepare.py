"""Stream and stratify a bounded Community Forensics training subset."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import tempfile
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
        identity = str(row.get("model_name") or "unknown-generator")
    else:
        identity = "|".join(
            [
                str(row.get("image_name") or "unknown-image"),
                str(row.get("real_source") or "unknown-source"),
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
    real_source = str(row.get("real_source") or "").casefold()
    return any(token.casefold() in real_source for token in config.data.excluded_real_source_tokens)


def extract_image_bytes(value: Any) -> bytes:
    """Extract bytes from the representations emitted by Hugging Face datasets."""
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


def _config_fingerprint(config: AppConfig, revision: str) -> str:
    relevant = {
        "data": config.to_dict()["data"],
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


def sample_stream(
    rows: Iterable[Mapping[str, Any]], config: AppConfig, revision: str
) -> dict[str, Any]:
    """Consume a bounded iterable and atomically produce a stratified manifest."""
    output_dir = Path(config.data.output_dir)
    manifest_path = Path(config.data.manifest_path)
    audit_path = Path(config.data.audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_targets, fake_targets = _targets(config)
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
    stop_reason = "maximum rows scanned"

    for scanned, row in enumerate(rows, start=1):
        if scanned > config.data.max_scanned:
            break
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
        real_source = str(row.get("real_source") or "unknown")

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
                math.ceil(
                    class_targets[(split, 0)] * config.data.max_real_source_fraction
                ),
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
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        if not absolute_path.exists():
            absolute_path.write_bytes(image_bytes)

        record = {
            "id": content_id,
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

    complete = _quotas_complete(class_counts, architecture_counts, class_targets, fake_targets)
    manifest_lines = [
        json.dumps(record, sort_keys=True, ensure_ascii=False)
        for record in sorted(records.values(), key=lambda item: (item["split"], item["label"], item["id"]))
    ]
    _atomic_write_text(manifest_path, "\n".join(manifest_lines) + ("\n" if manifest_lines else ""))

    audit = {
        "complete": complete,
        "stop_reason": stop_reason,
        "source_revision": revision,
        "sampling_config_sha256": _config_fingerprint(config, revision),
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
    _atomic_write_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not complete:
        raise PreparationError(
            f"Sampling stopped before all quotas were satisfied: {stop_reason}. "
            f"Inspect {audit_path} and adjust only the centralized configuration."
        )
    return audit


def _resolve_revision(config: AppConfig) -> str:
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

    return str(HfApi().dataset_info(config.data.repo_id).sha)


def prepare_dataset(config: AppConfig) -> dict[str, Any]:
    """Open the remote streaming dataset and run bounded stratified sampling."""
    from datasets import load_dataset

    revision = _resolve_revision(config)
    LOGGER.info("Using dataset revision %s", revision)
    stream = load_dataset(
        config.data.repo_id,
        split="train",
        streaming=True,
        revision=revision,
        cache_dir=config.data.cache_dir,
    )
    stream = stream.shuffle(seed=config.project.seed, buffer_size=config.data.shuffle_buffer)
    return sample_stream(stream, config, revision)


def main() -> None:
    """Run the public dataset preparation command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Prepare a bounded Community Forensics subset.")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    try:
        audit = prepare_dataset(config)
    except PreparationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    LOGGER.info(
        "Dataset preparation completed with %d images after scanning %d rows.",
        audit["selected"],
        audit["scanned"],
    )


if __name__ == "__main__":
    main()
