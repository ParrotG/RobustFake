"""Prepare a deterministic, resumable, multi-source AIGC training pool."""

from __future__ import annotations

import csv
import copy
import functools
import hashlib
import io
import json
import logging
import math
import os
import random
import shutil
import sqlite3
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import quote

import imagehash
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.prepare import PreparationError, atomic_write_text

LOGGER = logging.getLogger(__name__)
GIB = 1024**3
MIXED_SPLITS = ("train", "val_id", "val_dg")
SOURCE_ORDER = {"shanmuk": 0, "wildfake": 1, "community_forensics": 2, "tiny_genimage": 3}
SPLIT_PRIORITY = {"train": 0, "val_id": 1, "val_dg": 2, "external_test": 3}
DESCRIPTION_ALGORITHM_VERSION = 3


def stable_rank(seed: int, *parts: object) -> str:
    """Return a stable hexadecimal priority independent of input ordering."""
    identity = "\0".join(str(part) for part in parts)
    return hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()


def canonical_generator(value: object) -> str:
    """Normalize generator aliases for cross-dataset holdout enforcement."""
    text = "".join(character for character in str(value).casefold() if character.isalnum())
    aliases = {
        "stablediffusionv14": "sd14",
        "stablediffusion14": "sd14",
        "stable_diffusion_v_1_4": "sd14",
        "stablediffusionv15": "sd15",
        "stablediffusion15": "sd15",
        "stable_diffusion_v_1_5": "sd15",
        "biggan": "biggan",
        "wukong": "wukong",
        "vqdiffusion": "vqdm",
    }
    return aliases.get(text, text)


def _matches_generator_alias(value: object, alias: str) -> bool:
    normalized = canonical_generator(value)
    normalized_alias = canonical_generator(alias)
    if normalized == normalized_alias:
        return True
    suffix = normalized.removeprefix(normalized_alias)
    return normalized.startswith(normalized_alias) and bool(suffix) and (
        suffix[0].isdigit() or suffix.startswith(("v", "base", "large", "small", "xl"))
    )


def nuisance_buckets(width: int, height: int, image_format: str, encoded_bytes: int) -> dict[str, str]:
    """Map technical metadata to fixed, label-independent audit buckets."""
    short = min(width, height)
    if short < 192:
        resolution = "lt192"
    elif short < 256:
        resolution = "192_255"
    elif short < 384:
        resolution = "256_383"
    elif short < 512:
        resolution = "384_511"
    elif short < 768:
        resolution = "512_767"
    else:
        resolution = "ge768"
    ratio = width / max(1, height)
    aspect = "portrait" if ratio < 0.8 else "landscape" if ratio > 1.25 else "squareish"
    normalized_format = image_format.casefold()
    if normalized_format in {"jpg", "jpeg"}:
        normalized_format = "jpeg"
    elif normalized_format not in {"png", "webp"}:
        normalized_format = "other"
    density = encoded_bytes / max(1, width * height)
    density_bucket = "low" if density < 0.5 else "medium" if density < 1.5 else "high"
    return {
        "resolution_bucket": resolution,
        "aspect_bucket": aspect,
        "format_bucket": normalized_format,
        "encoding_density_bucket": density_bucket,
    }


def _quota(config: AppConfig, source: str, split: str, label: int) -> int:
    return int(config.mixed_data.source_quotas[source][split][str(label)])


def _stable_sorted(records: Iterable[Mapping[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda item: stable_rank(
            config.project.seed,
            item.get("source_dataset"),
            item.get("source_revision"),
            item.get("source_id"),
        ),
    )


def square_root_allocation(total: int, counts: Mapping[str, int], cap: int | None = None) -> dict[str, int]:
    """Allocate a total proportionally to sqrt(availability) with deterministic water filling."""
    available = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    if total < 0 or total > sum(available.values()):
        raise PreparationError(f"Cannot allocate {total} samples from {sum(available.values())} candidates.")
    result = {key: 0 for key in available}
    remaining = total
    while remaining:
        eligible = {
            key: value
            for key, value in available.items()
            if result[key] < value and (cap is None or result[key] < cap)
        }
        if not eligible:
            raise PreparationError("A per-domain cap prevents the configured quota from being met.")
        denominator = sum(math.sqrt(value) for value in eligible.values())
        proposals = {
            key: min(
                value - result[key],
                cap - result[key] if cap is not None else value,
                max(1, int(remaining * math.sqrt(value) / denominator)),
            )
            for key, value in eligible.items()
        }
        progressed = 0
        for key in sorted(proposals):
            addition = min(proposals[key], remaining)
            result[key] += addition
            remaining -= addition
            progressed += addition
            if not remaining:
                break
        if not progressed:
            raise PreparationError("Square-root allocation made no progress.")
    return result


def select_by_groups(
    records: Iterable[Mapping[str, Any]],
    total: int,
    config: AppConfig,
    group: Callable[[Mapping[str, Any]], str],
    *,
    cap: int | None = None,
) -> list[dict[str, Any]]:
    """Select a stable square-root-balanced subset across independent groups."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(group(record))].append(dict(record))
    allocation = square_root_allocation(total, {key: len(value) for key, value in grouped.items()}, cap)
    selected: list[dict[str, Any]] = []
    for name in sorted(grouped):
        selected.extend(_stable_sorted(grouped[name], config)[: allocation[name]])
    return _stable_sorted(selected, config)


def _read_manifest(path: Path, root: Path | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreparationError(f"Required source manifest does not exist: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if root is not None:
        for record in records:
            source_path = root / str(record["path"])
            if not source_path.is_file():
                raise PreparationError(f"Source manifest image is missing: {source_path}")
            record["local_path"] = str(source_path)
    return records


def _extension(image_format: str) -> str:
    value = image_format.casefold()
    return "jpg" if value in {"jpg", "jpeg"} else value if value in {"png", "webp"} else "bin"


def describe_path(
    path: Path, config: AppConfig, *, include_crop_resistant_hash: bool = True
) -> dict[str, Any]:
    """Compute byte, pixel, and perceptual identities for one safely decoded image."""
    content = path.read_bytes()
    Image.MAX_IMAGE_PIXELS = config.mixed_data.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                source.load()
                image_format = str(source.format or "").casefold()
                image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PreparationError(f"A mixed-data image cannot be decoded safely: {path}") from exc
    width, height = image.size
    # Hash normalized pixels directly. Re-encoding every image as PNG is much
    # slower and ties a semantic pixel identity to an encoder implementation.
    pixel_identity = (
        width.to_bytes(8, byteorder="big")
        + height.to_bytes(8, byteorder="big")
        + image.tobytes()
    )
    description = {
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "pixel_sha256": hashlib.sha256(pixel_identity).hexdigest(),
        "perceptual_hash": str(imagehash.phash(image, hash_size=config.mixed_data.phash_size)),
        "difference_hash": str(imagehash.dhash(image, hash_size=config.mixed_data.phash_size)),
        "format": image_format,
        "width": width,
        "height": height,
        "bytes": len(content),
        "encoding_density": len(content) / max(1, width * height),
        "description_algorithm_version": DESCRIPTION_ALGORITHM_VERSION,
    }
    if include_crop_resistant_hash:
        description["crop_resistant_hash"] = str(imagehash.crop_resistant_hash(image))
    description.update(nuisance_buckets(width, height, image_format, len(content)))
    return description


def crop_resistant_hash_path(path: Path, config: AppConfig) -> str:
    """Compute the expensive crop-resistant hash only for a plausible near match."""
    content = path.read_bytes()
    Image.MAX_IMAGE_PIXELS = config.mixed_data.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PreparationError(f"A mixed-data image cannot be decoded safely: {path}") from exc
    return str(imagehash.crop_resistant_hash(image))


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class DedupIndex:
    """Track exact and conservative near duplicates across sources and splits."""

    def __init__(self, config: AppConfig, deny_records: Iterable[Mapping[str, Any]]) -> None:
        self.config = config
        self.content: dict[str, dict[str, Any]] = {}
        self.pixels: dict[str, dict[str, Any]] = {}
        self.hashes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.hash_tree: tuple[int, dict[int, Any]] | None = None
        self.events: list[dict[str, Any]] = []
        for record in deny_records:
            normalized = dict(record)
            if normalized.get("content_sha256"):
                self.content[str(normalized["content_sha256"])] = normalized
            if normalized.get("pixel_sha256"):
                self.pixels[str(normalized["pixel_sha256"])] = normalized
            if normalized.get("perceptual_hash") and normalized.get("difference_hash"):
                self._add_hash(normalized)

    def _add_hash(self, record: dict[str, Any]) -> None:
        value = int(str(record["perceptual_hash"]), 16)
        self.hashes[value].append(record)
        # A BK-tree stores one node per unique value. Adding duplicate values as
        # zero-distance child nodes creates a long chain and quadratic behavior.
        if len(self.hashes[value]) > 1:
            return
        if self.hash_tree is None:
            self.hash_tree = (value, {})
            return
        node = self.hash_tree
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def _near_hashes(self, value: int, radius: int) -> Iterable[dict[str, Any]]:
        if self.hash_tree is None:
            return
        pending = [self.hash_tree]
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield from self.hashes[node[0]]
            low, high = distance - radius, distance + radius
            pending.extend(child for edge, child in node[1].items() if low <= edge <= high)

    def _crop_hash(self, record: Mapping[str, Any]) -> str:
        existing = str(record.get("crop_resistant_hash") or "")
        if existing:
            return existing
        local_path = record.get("local_path")
        if not local_path:
            return ""
        value = crop_resistant_hash_path(Path(str(local_path)), self.config)
        if isinstance(record, dict):
            record["crop_resistant_hash"] = value
        return value

    def collision(self, record: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
        for kind, index, field in (
            ("content_sha256", self.content, "content_sha256"),
            ("pixel_sha256", self.pixels, "pixel_sha256"),
        ):
            value = str(record.get(field) or "")
            if value and value in index:
                return kind, index[value]
        candidate_phash = str(record.get("perceptual_hash") or "")
        candidate_dhash = str(record.get("difference_hash") or "")
        if candidate_phash and candidate_dhash:
            candidate_value = int(candidate_phash, 16)
            for existing in self._near_hashes(
                candidate_value,
                self.config.mixed_data.crop_phash_distance,
            ):
                phash_distance = _hamming(candidate_phash, str(existing["perceptual_hash"]))
                strict = (
                    phash_distance <= self.config.mixed_data.phash_distance
                    and _hamming(candidate_dhash, str(existing["difference_hash"]))
                    <= self.config.mixed_data.dhash_distance
                )
                crop_match = False
                if not strict and phash_distance <= self.config.mixed_data.crop_phash_distance:
                    candidate_crop = self._crop_hash(record)
                    existing_crop = self._crop_hash(existing)
                    if candidate_crop and existing_crop:
                        crop_match = imagehash.hex_to_multihash(candidate_crop).matches(
                            imagehash.hex_to_multihash(existing_crop),
                            hamming_cutoff=self.config.mixed_data.dhash_distance,
                        )
                if strict or crop_match:
                    return "perceptual" if strict else "crop_resistant", existing
        return None

    def add(self, record: dict[str, Any]) -> None:
        collision = self.collision(record)
        if collision is not None:
            kind, existing = collision
            if int(existing.get("label", record["label"])) != int(record["label"]):
                raise PreparationError(
                    f"Conflicting labels share a confirmed {kind} duplicate: "
                    f"{existing.get('id', existing.get('source_id'))} and {record.get('source_id')}"
                )
            self.events.append(
                {
                    "kind": kind,
                    "kept": existing.get("id", existing.get("source_id")),
                    "discarded": record.get("source_id"),
                    "label": int(record["label"]),
                }
            )
            if existing.get("split") != "external_test" and record.get("provenance_sources"):
                provenance = existing.setdefault("provenance_sources", [])
                for source in record["provenance_sources"]:
                    if source not in provenance:
                        provenance.append(source)
            raise DuplicateCandidate
        self.content[str(record["content_sha256"])] = record
        self.pixels[str(record["pixel_sha256"])] = record
        self._add_hash(record)


class DuplicateCandidate(Exception):
    """Internal control flow for a same-label duplicate candidate."""


def _describe_candidates_resumable(
    candidates: Sequence[Mapping[str, Any]], config: AppConfig
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Hash candidate images concurrently and persist completed work transactionally."""
    cache_path = Path(config.mixed_data.cache_dir) / "candidate_hash_cache.sqlite3"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS image_descriptions (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            algorithm_version INTEGER NOT NULL,
            phash_size INTEGER NOT NULL,
            description_json TEXT NOT NULL
        )
        """
    )
    connection.commit()

    path_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_candidates: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        path = Path(str(candidate["local_path"])).resolve()
        if not path.is_file():
            invalid_candidates.append(candidate)
            continue
        path_candidates[str(path)].append(candidate)

    descriptions_by_path: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, int, int]] = []
    algorithm_version = DESCRIPTION_ALGORITHM_VERSION
    for path_text in sorted(path_candidates):
        stat = Path(path_text).stat()
        cached = connection.execute(
            """
            SELECT description_json FROM image_descriptions
            WHERE path = ? AND size = ? AND modified_ns = ?
              AND algorithm_version = ? AND phash_size = ?
            """,
            (path_text, stat.st_size, stat.st_mtime_ns, algorithm_version, config.mixed_data.phash_size),
        ).fetchone()
        if cached is not None:
            try:
                descriptions_by_path[path_text] = json.loads(str(cached[0]))
                continue
            except (json.JSONDecodeError, TypeError):
                pass
        pending.append((path_text, stat.st_size, stat.st_mtime_ns))

    LOGGER.info(
        "Candidate image hashes: %d cached, %d pending, %d unique files total.",
        len(descriptions_by_path),
        len(pending),
        len(path_candidates),
    )

    def compute(task: tuple[str, int, int]) -> tuple[str, int, int, dict[str, Any] | None]:
        path_text, size, modified_ns = task
        try:
            return path_text, size, modified_ns, describe_path(
                Path(path_text), config, include_crop_resistant_hash=False
            )
        except PreparationError:
            return path_text, size, modified_ns, None

    completed_since_commit = 0
    executor = ThreadPoolExecutor(max_workers=config.mixed_data.hash_workers)
    iterator = iter(pending)
    futures: dict[Any, None] = {}
    progress = tqdm(total=len(pending), desc="Hash mixed candidates")
    try:
        for _ in range(min(len(pending), config.mixed_data.hash_workers * 2)):
            futures[executor.submit(compute, next(iterator))] = None
        while futures:
            done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                path_text, size, modified_ns, description = future.result()
                if description is None:
                    invalid_candidates.extend(path_candidates.pop(path_text))
                    connection.execute("DELETE FROM image_descriptions WHERE path = ?", (path_text,))
                else:
                    descriptions_by_path[path_text] = description
                    connection.execute(
                        """
                        INSERT INTO image_descriptions
                            (path, size, modified_ns, algorithm_version, phash_size, description_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            size = excluded.size,
                            modified_ns = excluded.modified_ns,
                            algorithm_version = excluded.algorithm_version,
                            phash_size = excluded.phash_size,
                            description_json = excluded.description_json
                        """,
                        (
                            path_text,
                            size,
                            modified_ns,
                            algorithm_version,
                            config.mixed_data.phash_size,
                            json.dumps(description, sort_keys=True),
                        ),
                    )
                completed_since_commit += 1
                progress.update(1)
                if completed_since_commit >= config.mixed_data.hash_checkpoint_every:
                    connection.commit()
                    completed_since_commit = 0
                try:
                    task = next(iterator)
                except StopIteration:
                    continue
                futures[executor.submit(compute, task)] = None
    finally:
        progress.close()
        connection.commit()
        connection.close()
        executor.shutdown(wait=True, cancel_futures=True)

    descriptions: dict[tuple[str, str], dict[str, Any]] = {}
    valid_candidates: list[dict[str, Any]] = []
    for path_text, grouped_candidates in path_candidates.items():
        description = descriptions_by_path[path_text]
        for candidate in grouped_candidates:
            identity = (str(candidate["source_dataset"]), str(candidate["source_id"]))
            descriptions[identity] = description
            valid_candidates.append(candidate)
    return descriptions, valid_candidates, invalid_candidates


def load_external_denylist(config: AppConfig) -> list[dict[str, Any]]:
    """Load and resumably enrich every mandatory external evaluation denylist."""
    cache_path = Path(config.mixed_data.cache_dir) / "external_deny_index.json"
    cached: dict[str, Any] = {
        "schema_version": DESCRIPTION_ALGORITHM_VERSION,
        "manifest_sha256": {},
        "records": {},
    }
    if cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if int(loaded.get("schema_version", 0)) == DESCRIPTION_ALGORITHM_VERSION:
                cached = loaded
        except (json.JSONDecodeError, OSError):
            LOGGER.warning("Ignoring an unreadable external deny-index cache at %s.", cache_path)
    pending: list[tuple[str, dict[str, Any], Path, str]] = []
    active_keys: set[str] = set()
    for configured_path in config.mixed_data.external_deny_manifests:
        path = Path(configured_path)
        if not path.is_file():
            raise PreparationError(
                f"External leakage manifest is missing: {path}. Prepare all evaluation sets first."
            )
        manifest_sha = _cached_file_sha256(path)
        previous_sha = str(cached.get("manifest_sha256", {}).get(configured_path, ""))
        if previous_sha and previous_sha != manifest_sha:
            cached["records"] = {
                key: value
                for key, value in cached.get("records", {}).items()
                if not key.startswith(f"{configured_path}\0")
            }
        cached.setdefault("manifest_sha256", {})[configured_path] = manifest_sha
        root = path.parent
        for index, record in enumerate(_read_manifest(path)):
            item = dict(record)
            image_path = root / str(item["path"])
            if not image_path.is_file():
                raise PreparationError(f"External leakage image is missing: {image_path}")
            identity = str(item.get("id") or item.get("source_id") or index)
            cache_key = f"{configured_path}\0{identity}"
            active_keys.add(cache_key)
            cached_item = cached.get("records", {}).get(cache_key)
            required = (
                "content_sha256",
                "pixel_sha256",
                "perceptual_hash",
                "difference_hash",
                "crop_resistant_hash",
            )
            if (
                cached_item
                and int(cached_item.get("description_algorithm_version", 0))
                == DESCRIPTION_ALGORITHM_VERSION
                and all(cached_item.get(field) for field in required)
            ):
                continue
            if (
                int(item.get("description_algorithm_version", 0))
                == DESCRIPTION_ALGORITHM_VERSION
                and all(item.get(field) for field in required)
            ):
                item["split"] = "external_test"
                cached.setdefault("records", {})[cache_key] = item
                continue
            pending.append((cache_key, item, image_path, configured_path))

    cached["records"] = {
        key: value for key, value in cached.get("records", {}).items() if key in active_keys
    }
    if pending:
        LOGGER.info(
            "Building the external near-duplicate deny index for %d images; this is cached and resumable.",
            len(pending),
        )

        def enrich(task: tuple[str, dict[str, Any], Path, str]) -> tuple[str, dict[str, Any]]:
            cache_key, item, image_path, _manifest = task
            enriched = {**item, **describe_path(image_path, config), "split": "external_test"}
            return cache_key, enriched

        completed = 0
        worker_count = max(1, config.mixed_data.download_workers)
        executor = ThreadPoolExecutor(max_workers=worker_count)
        iterator = iter(pending)
        futures: dict[Any, None] = {}
        progress = tqdm(total=len(pending), desc="Hash external denylist")
        try:
            for _ in range(min(len(pending), worker_count * 2)):
                futures[executor.submit(enrich, next(iterator))] = None
            while futures:
                done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    cache_key, item = future.result()
                    cached.setdefault("records", {})[cache_key] = item
                    completed += 1
                    progress.update(1)
                    if completed % max(500, config.mixed_data.checkpoint_every) == 0:
                        atomic_write_text(
                            cache_path, json.dumps(cached, indent=2, sort_keys=True) + "\n"
                        )
                    try:
                        task = next(iterator)
                    except StopIteration:
                        continue
                    futures[executor.submit(enrich, task)] = None
        finally:
            progress.close()
            executor.shutdown(wait=True, cancel_futures=True)
            if completed:
                atomic_write_text(
                    cache_path, json.dumps(cached, indent=2, sort_keys=True) + "\n"
                )
        atomic_write_text(cache_path, json.dumps(cached, indent=2, sort_keys=True) + "\n")
    elif not cache_path.is_file():
        atomic_write_text(cache_path, json.dumps(cached, indent=2, sort_keys=True) + "\n")
    denied = [dict(cached["records"][key]) for key in sorted(active_keys)]
    LOGGER.info("External leakage deny index is ready with %d images.", len(denied))
    return denied


@functools.lru_cache(maxsize=32)
def _file_sha256_cached(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return hashlib.sha256(Path(path_text).read_bytes()).hexdigest()


def _cached_file_sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _preflight_sources(config: AppConfig) -> None:
    """Validate credentials and every pinned remote identity before payload acquisition."""
    from huggingface_hub import HfApi, get_token
    from modelscope_hub import HubApi

    mixed = config.mixed_data
    token = get_token()
    if not token:
        raise PreparationError(
            "Mixed dataset preparation requires a saved Hugging Face token; run 'hf auth login'."
        )
    api = HfApi(token=token)
    for name, repo_id, revision in (
        ("Shanmuk", mixed.shanmuk_repo_id, mixed.shanmuk_revision),
        ("Community Forensics", mixed.community_repo_id, mixed.community_revision),
        ("Tiny-GenImage", mixed.tiny_genimage_repo_id, mixed.tiny_genimage_revision),
    ):
        info = _retry(
            config,
            f"{name} revision preflight",
            lambda repo_id=repo_id, revision=revision: api.dataset_info(
                repo_id, revision=revision, files_metadata=False
            ),
        )
        if str(info.sha) != revision:
            raise PreparationError(f"{name} revision identity changed.")
    wildfake_files = _retry(
        config,
        "WildFake revision preflight",
        lambda: HubApi().list_repo_files(
            mixed.wildfake_repo_id, "dataset", revision=mixed.wildfake_revision
        ),
    )
    if mixed.wildfake_train_metadata_file not in {item.path for item in wildfake_files}:
        raise PreparationError("Pinned WildFake revision does not contain train metadata.")


def _normalize_local_record(
    record: Mapping[str, Any], source: str, revision: str, local_path: str
) -> dict[str, Any]:
    generator = str(record.get("generator") or record.get("model_name") or "real")
    return {
        **dict(record),
        "source_dataset": source,
        "source_revision": revision,
        "source_id": str(record.get("id") or record.get("image_name")),
        "upstream_split": str(record.get("source_split") or record.get("split") or "train"),
        "real_source": str(record.get("real_source") or record.get("source_dataset") or "unknown"),
        "generator": generator,
        "generator_family": str(record.get("generator_family") or record.get("architecture") or "unknown"),
        "architecture": str(record.get("architecture") or generator),
        "model_id": str(record.get("model_name") or generator),
        "subset": str(record.get("subset") or "paired"),
        "content_group": str(record.get("source_real_id") or record.get("category") or "unknown"),
        "local_path": local_path,
    }


def select_shanmuk(config: AppConfig) -> list[dict[str, Any]]:
    """Select 5,000 intact real/fake parent pairs from the prepared paired source."""
    root = Path(config.mixed_data.shanmuk_root)
    if not (root / "manifest.jsonl").is_file():
        from aigc_recognizer.data.prepare import prepare_dataset

        source_config = copy.deepcopy(config)
        source_config.data.output_dir = str(root)
        source_config.data.manifest_path = str(root / "manifest.jsonl")
        source_config.data.audit_path = str(root / "audit.json")
        source_config.data.state_path = str(root / "preparation_state.json")
        source_config.data.nuisance_report_path = str(root / "nuisance_report.json")
        prepare_dataset(source_config)
    rows = _read_manifest(root / "manifest.jsonl", root)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source_split") == "test" or row.get("split") == "test":
            continue
        normalized = _normalize_local_record(
            row, "shanmuk", config.mixed_data.shanmuk_revision, row["local_path"]
        )
        groups[str(row.get("source_real_id"))].append(normalized)
    valid = {
        parent: values
        for parent, values in groups.items()
        if Counter(int(item["label"]) for item in values) == {0: 1, 1: 1}
    }
    reserve_pairs = math.ceil(5_000 * config.mixed_data.reserve_multiplier)
    if len(valid) < reserve_pairs:
        raise PreparationError("The prepared Shanmuk source has fewer than 5,000 intact pairs.")
    available = set(valid)

    def choose_balanced(amount: int, role: str) -> list[str]:
        if amount == 0:
            return []
        generator_groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for parent in available:
            fake = next(item for item in valid[parent] if int(item["label"]) == 1)
            real = next(item for item in valid[parent] if int(item["label"]) == 0)
            generator_groups[str(fake["generator"])][str(real["real_source"])].append(parent)
        generators = sorted(
            generator_groups,
            key=lambda name: stable_rank(config.project.seed, "shanmuk-generator", role, name),
        )
        if set(generators) != set(config.data.generators):
            raise PreparationError("Shanmuk source does not contain the configured six generators.")
        generator_allocation = square_root_allocation(
            amount,
            {
                generator: sum(len(parents) for parents in generator_groups[generator].values())
                for generator in generators
            },
        )
        chosen: list[str] = []
        for generator in generators:
            target = generator_allocation[generator]
            source_groups = generator_groups[generator]
            allocation = square_root_allocation(
                target, {source: len(parents) for source, parents in source_groups.items()}
            )
            for source in sorted(source_groups):
                ordered = sorted(
                    source_groups[source],
                    key=lambda parent: stable_rank(
                        config.project.seed, "shanmuk", role, generator, source, parent
                    ),
                )
                chosen.extend(ordered[: allocation[source]])
        for parent in chosen:
            available.remove(parent)
        return chosen

    reserve_extra = reserve_pairs - 5_000
    reserve_train = round(reserve_extra * 0.8)
    assigned = [
        ("train", 0, choose_balanced(4_000, "train-primary")),
        ("val_id", 0, choose_balanced(1_000, "id-primary")),
        ("train", 1, choose_balanced(reserve_train, "train-reserve")),
        ("val_id", 1, choose_balanced(reserve_extra - reserve_train, "id-reserve")),
    ]
    selected: list[dict[str, Any]] = []
    for split, tier, parents in assigned:
        for parent in parents:
            for record in valid[parent]:
                record["split"] = split
                record["selection_tier"] = tier
                record["pair_id"] = parent
                selected.append(record)
    return selected


def _heldout_alias(config: AppConfig, record: Mapping[str, Any]) -> bool:
    aliases = set(config.mixed_data.global_heldout_generator_aliases)
    values = (
        record.get("generator"), record.get("architecture"), record.get("model_id")
    )
    return any(_matches_generator_alias(value, alias) for alias in aliases for value in values)


def select_community_local(config: AppConfig) -> list[dict[str, Any]]:
    """Select all reusable Community Forensics candidates before remote backfill."""
    root = Path(config.mixed_data.community_root)
    if not (root / "manifest.jsonl").is_file():
        return []
    rows = _read_manifest(root / "manifest.jsonl", root)
    candidates = []
    for row in rows:
        model = str(row.get("model_name") or "")
        if "dall" in canonical_generator(model) or "openai" in model.casefold():
            continue
        normalized = _normalize_local_record(
            row, "community_forensics", config.mixed_data.community_revision, row["local_path"]
        )
        if int(normalized["label"]) == 1 and _heldout_alias(config, normalized):
            continue
        candidates.append(normalized)
    return candidates


def _assign_community(candidates: Sequence[Mapping[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    """Assign Community Forensics candidates to train, ID, and model-held-out DG splits."""
    real = [dict(item) for item in candidates if int(item["label"]) == 0]
    fake = [dict(item) for item in candidates if int(item["label"]) == 1]
    systematic_models: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in fake:
        if str(item.get("subset")).casefold() == "systematic":
            systematic_models[str(item["model_id"])].append(item)
    dg_models: list[str] = []
    dg_count = 0
    dg_pool_target = math.ceil(1_000 * config.mixed_data.reserve_multiplier)
    for model in sorted(systematic_models, key=lambda name: stable_rank(config.project.seed, "cf-dg", name)):
        available = min(config.mixed_data.community_systematic_model_cap, len(systematic_models[model]))
        if available:
            dg_models.append(model)
            dg_count += available
        if dg_count >= dg_pool_target:
            break
    if dg_count < dg_pool_target:
        raise PreparationError(
            f"Community Forensics cannot reserve {dg_pool_target} held-out Systematic samples."
        )
    dg_pool = []
    for model in dg_models:
        remaining = dg_pool_target - len(dg_pool)
        dg_pool.extend(_stable_sorted(systematic_models[model], config)[: min(remaining, 8)])
        if len(dg_pool) == dg_pool_target:
            break
    dg = dg_pool[:1_000]
    dg_ids = {str(item["source_id"]) for item in dg}
    remaining_fake = [item for item in fake if str(item["model_id"]) not in set(dg_models)]
    cap_by_model = {
        str(item["model_id"]): (
            config.mixed_data.community_systematic_model_cap
            if str(item.get("subset")).casefold() == "systematic"
            else config.mixed_data.community_other_model_cap
        )
        for item in remaining_fake
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in remaining_fake:
        grouped[str(item.get("architecture") or "other")].append(item)
    architecture_targets = _weighted_targets(
        15_000, config.mixed_data.community_architecture_weights, {key: len(value) for key, value in grouped.items()}
    )
    subset_availability = Counter(str(item.get("subset") or "unknown") for item in remaining_fake)
    subset_targets = _weighted_targets(
        15_000, config.mixed_data.community_subset_weights, subset_availability
    )
    chosen_fake: list[dict[str, Any]] = []
    model_counts: Counter[str] = Counter()
    architecture_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    used: set[str] = set()

    def admit(item: dict[str, Any]) -> bool:
        model = str(item["model_id"])
        identity = str(item["source_id"])
        if identity in used or model_counts[model] >= cap_by_model[model]:
            return False
        chosen_fake.append(item)
        used.add(identity)
        model_counts[model] += 1
        architecture_counts[str(item.get("architecture") or "other")] += 1
        subset_counts[str(item.get("subset") or "unknown")] += 1
        return True

    ordered_remaining = _stable_sorted(remaining_fake, config)
    # First satisfy both independent marginals, then preserve architecture, then water-fill.
    for pass_index in range(3):
        for item in ordered_remaining:
            architecture = str(item.get("architecture") or "other")
            subset = str(item.get("subset") or "unknown")
            if pass_index == 0 and not (
                architecture_counts[architecture] < architecture_targets.get(architecture, 0)
                and subset_counts[subset] < subset_targets.get(subset, 0)
            ):
                continue
            if pass_index == 1 and not (
                architecture_counts[architecture] < architecture_targets.get(architecture, 0)
            ):
                continue
            admit(item)
            if len(chosen_fake) == 15_000:
                break
        if len(chosen_fake) == 15_000:
            break
    if len(chosen_fake) < 15_000:
        for item in ordered_remaining:
            model = str(item["model_id"])
            admit(item)
            if len(chosen_fake) == 15_000:
                break
    if len(chosen_fake) < 15_000 or len(real) < 14_000:
        raise PreparationError("Community Forensics candidates cannot meet the 30k source quota.")
    chosen_real = select_by_groups(real, 14_000, config, lambda item: str(item["real_source"]))
    real_order = _stable_sorted(chosen_real, config)
    fake_order = _stable_sorted(chosen_fake, config)
    for item in real_order[:13_000]:
        item["split"] = "train"
    for item in real_order[13_000:14_000]:
        item["split"] = "val_id"
    for item in fake_order[:13_000]:
        item["split"] = "train"
    for item in fake_order[13_000:15_000]:
        item["split"] = "val_id"
    for item in dg:
        item["split"] = "val_dg"
    assert not dg_ids & {str(item["source_id"]) for item in fake_order}
    selected = real_order + fake_order + dg
    selected_ids = {str(item["source_id"]) for item in selected}
    extra_real = [item for item in _stable_sorted(real, config) if str(item["source_id"]) not in selected_ids]
    extra_fake = [
        item
        for item in _stable_sorted(remaining_fake, config)
        if str(item["source_id"]) not in selected_ids
    ]
    for index, item in enumerate(extra_real[:3_500]):
        item["split"] = "val_id" if index % 14 == 0 else "train"
        item["selection_tier"] = 1
        selected.append(item)
    for index, item in enumerate(extra_fake[:3_750]):
        item["split"] = "val_id" if index % 8 == 0 else "train"
        item["selection_tier"] = 1
        selected.append(item)
    extra_dg = [item for item in dg_pool if str(item["source_id"]) not in dg_ids]
    for item in extra_dg[:250]:
        item["split"] = "val_dg"
        item["selection_tier"] = 1
        selected.append(item)
    return selected


def _weighted_targets(total: int, weights: Mapping[str, float], availability: Mapping[str, int]) -> dict[str, int]:
    """Allocate configured marginal weights with deterministic deficit redistribution."""
    known = {key: max(0.0, float(weights.get(key, weights.get("other", 0.0)))) for key in availability}
    if not any(known.values()):
        return square_root_allocation(total, availability)
    targets = {key: min(int(availability[key]), int(total * known[key] / sum(known.values()))) for key in known}
    remaining = total - sum(targets.values())
    for key in sorted(known, key=lambda name: (-known[name], name)):
        addition = min(remaining, int(availability[key]) - targets[key])
        targets[key] += addition
        remaining -= addition
        if not remaining:
            break
    if remaining:
        raise PreparationError("Weighted target buckets cannot meet their total quota.")
    return targets


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, Mapping):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise PreparationError("A source row does not contain embedded image bytes.")
    return value


def _stage_bytes(content: bytes, cache_root: Path) -> Path:
    digest = hashlib.sha256(content).hexdigest()
    destination = cache_root / "staged" / digest[:2] / digest
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{digest}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def _source_cache_path(cache_root: Path) -> Path:
    return cache_root / "candidate_state.json"


def _load_source_cache(cache_root: Path, revision: str) -> dict[str, Any]:
    """Load a pinned per-source acquisition checkpoint and discard stale entries."""
    path = _source_cache_path(cache_root)
    if not path.is_file():
        return {"revision": revision, "completed_units": [], "candidates": [], "row_count": 0}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("revision") != revision:
        raise PreparationError(f"Source cache revision changed at {path}.")
    cached_candidates = list(state.get("candidates", []))
    if int(state.get("bounded_reserve_version", 0)) == 1 and cached_candidates:
        stride = max(1, len(cached_candidates) // 64)
        probe = cached_candidates[::stride]
        if all(Path(str(item.get("local_path", ""))).is_file() for item in probe):
            state["candidates"] = cached_candidates
            return state
    state["candidates"] = [
        item
        for item in cached_candidates
        if Path(str(item.get("local_path", ""))).is_file()
    ]
    if len(state["candidates"]) != len(cached_candidates):
        # A manually cleaned staged object invalidates unit completion. Re-scanning is safe.
        state["completed_units"] = []
        state["row_count"] = 0
    return state


def _save_source_cache(cache_root: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist an idempotent source acquisition checkpoint."""
    candidates = {
        str(item["source_id"]): dict(item) for item in state.get("candidates", [])
    }
    payload = {
        **dict(state),
        "completed_units": sorted(set(str(value) for value in state.get("completed_units", []))),
        "candidates": [candidates[key] for key in sorted(candidates)],
    }
    atomic_write_text(
        _source_cache_path(cache_root), json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _retry(config: AppConfig, description: str, operation: Callable[[], Any]) -> Any:
    """Retry transient source operations with bounded exponential backoff."""
    attempts = max(1, int(config.mixed_data.network_max_retries) + 1)
    for attempt in range(attempts):
        try:
            return operation()
        except (requests.RequestException, OSError, TimeoutError) as exc:
            if attempt + 1 == attempts:
                raise PreparationError(f"{description} failed after {attempts} attempts.") from exc
            delay = min(
                30.0, float(config.mixed_data.network_retry_base_seconds) * 2.0**attempt
            )
            LOGGER.warning("%s failed; retrying in %.1f seconds.", description, delay)
            time.sleep(delay)
    raise AssertionError("Unreachable retry state.")


def _bounded_prefetch(
    items: Sequence[Any], workers: int, download: Callable[[Any], Path]
) -> Iterable[tuple[Any, Path]]:
    """Overlap bounded remote downloads with ordered local shard decoding."""
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    pending: dict[int, Any] = {}
    try:
        initial = min(len(items), max(1, workers))
        for index in range(initial):
            pending[index] = executor.submit(download, items[index])
        for index, item in enumerate(items):
            future = pending.pop(index)
            following = index + initial
            if following < len(items):
                pending[following] = executor.submit(download, items[following])
            yield item, Path(future.result())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _community_row_candidate(
    row: Mapping[str, Any], config: AppConfig, cache_root: Path
) -> dict[str, Any] | None:
    label = int(row.get("label", -1))
    if label not in {0, 1} or str(row.get("split") or "train").casefold() != "train":
        return None
    if bool(row.get("nsfw_flag")):
        return None
    model = str(row.get("model_name") or ("real" if label == 0 else "unknown"))
    normalized_model = canonical_generator(model)
    if "dall" in normalized_model or "openai" in model.casefold():
        return None
    subset = str(row.get("subset") or ("Real" if label == 0 else "unknown"))
    architecture = str(row.get("architecture") or ("real" if label == 0 else "other"))
    provisional = {
        "generator": model,
        "architecture": architecture,
        "model_id": model,
    }
    if label == 1 and _heldout_alias(config, provisional):
        return None
    content = _image_bytes(row.get("image_data"))
    digest = hashlib.sha256(content).hexdigest()
    source_id = hashlib.sha256(
        f"community_forensics\0{model}\0{row.get('image_name')}\0{digest}".encode()
    ).hexdigest()
    local_path = _stage_bytes(content, cache_root)
    return {
        "source_dataset": "community_forensics",
        "source_revision": config.mixed_data.community_revision,
        "source_id": source_id,
        "upstream_split": "train",
        "label": label,
        "real_source": str(
            row.get("real_source") or model
            if label == 0
            else row.get("real_source") or "unknown"
        ),
        "generator": model,
        "generator_family": architecture,
        "architecture": architecture,
        "model_id": model,
        "subset": subset,
        "content_group": str(row.get("real_source") or "unknown"),
        "prompt": str(row.get("prompt") or ""),
        "local_path": str(local_path),
        "declared_format": str(row.get("format") or ""),
    }


def _prune_community_candidates(
    candidates: Sequence[Mapping[str, Any]], config: AppConfig, cache_root: Path
) -> list[dict[str, Any]]:
    """Bound staged Community candidates while preserving every final cap and reserve role."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        record = dict(item)
        if int(record["label"]) == 0:
            key = ("real", str(record.get("real_source") or "unknown"))
        else:
            key = ("fake", str(record.get("model_id") or "unknown"))
        grouped[key].append(record)
    retained: list[dict[str, Any]] = []
    for (kind, _name), records in grouped.items():
        if kind == "real":
            limit = math.ceil(14_000 * config.mixed_data.reserve_multiplier)
        else:
            systematic = str(records[0].get("subset")).casefold() == "systematic"
            final_cap = (
                config.mixed_data.community_systematic_model_cap
                if systematic
                else config.mixed_data.community_other_model_cap
            )
            limit = math.ceil(final_cap * config.mixed_data.reserve_multiplier)
        retained.extend(_stable_sorted(records, config)[:limit])
    retained_ids = {
        (str(item["source_dataset"]), str(item["source_id"])) for item in retained
    }
    staged_root = (cache_root / "staged").resolve()
    retained_paths = {str(Path(str(item["local_path"])).resolve()) for item in retained}
    discarded = [
        item
        for item in candidates
        if (str(item["source_dataset"]), str(item["source_id"])) not in retained_ids
    ]
    for item in tqdm(
        discarded,
        desc="Prune Community cache",
        disable=len(discarded) < 5_000,
    ):
        path = Path(str(item.get("local_path", "")))
        try:
            resolved = path.resolve()
            resolved.relative_to(staged_root)
        except (OSError, ValueError):
            continue
        if str(resolved) not in retained_paths:
            resolved.unlink(missing_ok=True)
    if staged_root.is_dir():
        for directory in sorted(staged_root.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
    return retained


def _cleanup_project_payload(payload_dir: Path) -> None:
    """Remove only completed project-owned payload files, never the global hub cache."""
    if not payload_dir.is_dir():
        return
    for path in payload_dir.rglob("*"):
        if path.is_file():
            path.unlink(missing_ok=True)
    for path in sorted(payload_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _compact_source_candidate_cache(
    cache_root: Path,
    revision: str,
    retained: Sequence[Mapping[str, Any]],
) -> None:
    """Keep only a finalized deterministic reserve pool in a project-owned staged cache."""
    state = _load_source_cache(cache_root, revision)
    retained_records = [dict(item) for item in retained]
    retained_paths = {
        str(Path(str(item["local_path"])).resolve()) for item in retained_records
    }
    staged_root = (cache_root / "staged").resolve()
    discarded = [
        item
        for item in state.get("candidates", [])
        if str(Path(str(item.get("local_path", ""))).resolve()) not in retained_paths
    ]
    for item in tqdm(
        discarded,
        desc=f"Compact {cache_root.name} cache",
        disable=len(discarded) < 5_000,
    ):
        path = Path(str(item.get("local_path", ""))).resolve()
        try:
            path.relative_to(staged_root)
        except ValueError:
            continue
        path.unlink(missing_ok=True)
    state.update(candidates=retained_records, bounded_reserve_version=1)
    _save_source_cache(cache_root, state)


def acquire_community_candidates(config: AppConfig) -> list[dict[str, Any]]:
    """Reuse local Community Forensics data and backfill from pinned Parquet shards."""
    cache_root = Path(config.mixed_data.cache_dir) / "community_forensics"
    source_state = _load_source_cache(cache_root, config.mixed_data.community_revision)
    if source_state["candidates"]:
        combined = {str(item["source_id"]): dict(item) for item in source_state["candidates"]}
    else:
        combined = {
            str(item["source_id"]): item for item in select_community_local(config)
        }
    needs_bounded_migration = int(source_state.get("bounded_reserve_version", 0)) != 1
    if needs_bounded_migration:
        LOGGER.info(
            "Pruning %d cached Community Forensics candidates to bounded per-domain reserves.",
            len(combined),
        )
        candidates = _prune_community_candidates(list(combined.values()), config, cache_root)
        LOGGER.info("Retained %d bounded Community Forensics candidates.", len(candidates))
        source_state["bounded_reserve_version"] = 1
    else:
        candidates = list(combined.values())
    seen = {str(item["source_id"]) for item in candidates}
    completed_units = set(str(value) for value in source_state["completed_units"])
    cached_shards = sorted(Path("data/cache/community_forensics_shards").glob("**/*.parquet"))
    processed_paths: set[Path] = set()

    def sufficient() -> bool:
        real_count = sum(int(item["label"]) == 0 for item in candidates)
        fake_candidates = [item for item in candidates if int(item["label"]) == 1]
        fake_count = len(fake_candidates)
        systematic_models = {
            str(item["model_id"])
            for item in candidates
            if int(item["label"]) == 1 and str(item.get("subset")).casefold() == "systematic"
        }
        model_counts = Counter(str(item["model_id"]) for item in fake_candidates)
        systematic_by_model = {
            str(item["model_id"])
            for item in fake_candidates
            if str(item.get("subset")).casefold() == "systematic"
        }
        usable_capacity = sum(
            min(
                count,
                config.mixed_data.community_systematic_model_cap
                if model in systematic_by_model
                else config.mixed_data.community_other_model_cap,
            )
            for model, count in model_counts.items()
        )
        return (
            real_count >= math.ceil(14_000 * config.mixed_data.reserve_multiplier)
            and fake_count >= 17_000
            and usable_capacity >= 17_000
            and len(systematic_models)
            >= math.ceil(1_000 * config.mixed_data.reserve_multiplier / config.mixed_data.community_systematic_model_cap)
        )

    def consume(path: Path) -> None:
        import pyarrow.parquet as parquet

        for batch in parquet.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                candidate = _community_row_candidate(row, config, cache_root)
                if candidate is None or str(candidate["source_id"]) in seen:
                    continue
                seen.add(str(candidate["source_id"]))
                candidates.append(candidate)

    for path in cached_shards:
        consume(path)
        candidates = _prune_community_candidates(candidates, config, cache_root)
        processed_paths.add(path.resolve())
        if sufficient():
            source_state.update(candidates=candidates)
            _save_source_cache(cache_root, source_state)
            _cleanup_project_payload(cache_root / "payload")
            return candidates

    candidates = _prune_community_candidates(candidates, config, cache_root)
    source_state.update(candidates=candidates)
    if needs_bounded_migration:
        _save_source_cache(cache_root, source_state)
    if sufficient():
        _cleanup_project_payload(cache_root / "payload")
        return candidates

    from huggingface_hub import HfApi, get_token, hf_hub_download

    token = get_token()
    info = _retry(
        config,
        "Community Forensics metadata request",
        lambda: HfApi(token=token).dataset_info(
            config.mixed_data.community_repo_id,
            revision=config.mixed_data.community_revision,
            files_metadata=True,
        ),
    )
    if str(info.sha) != config.mixed_data.community_revision:
        raise PreparationError("Community Forensics revision identity changed.")
    shards = [
        item for item in info.siblings if str(item.rfilename).endswith(".parquet")
    ]
    shards.sort(key=lambda item: stable_rank(config.project.seed, "cf-shard", item.rfilename))
    payload_dir = cache_root / "payload"
    pending_shards = []
    planned_bytes = 0
    for sibling in shards:
        if str(sibling.rfilename) in completed_units:
            continue
        size = int(sibling.size or 0)
        if planned_bytes + size > config.mixed_data.max_network_gb * GIB:
            break
        pending_shards.append(sibling)
        planned_bytes += size

    def download_shard(sibling: Any) -> Path:
        return Path(_retry(
            config,
            f"Community Forensics shard {sibling.rfilename}",
            lambda: hf_hub_download(
                repo_id=config.mixed_data.community_repo_id,
                repo_type="dataset",
                revision=config.mixed_data.community_revision,
                filename=str(sibling.rfilename),
                token=token,
                local_dir=payload_dir,
            ),
        ))

    for sibling, path in _bounded_prefetch(
        pending_shards, config.mixed_data.download_workers, download_shard
    ):
        if sufficient():
            break
        if path.resolve() not in processed_paths:
            consume(path)
            candidates = _prune_community_candidates(candidates, config, cache_root)
            processed_paths.add(path.resolve())
        completed_units.add(str(sibling.rfilename))
        source_state.update(completed_units=sorted(completed_units), candidates=candidates)
        _save_source_cache(cache_root, source_state)
        try:
            path.relative_to(payload_dir).exists()
            path.unlink(missing_ok=True)
        except ValueError:
            pass
    if not sufficient():
        raise PreparationError("Pinned Community Forensics shards cannot meet mixed-data quotas.")
    _cleanup_project_payload(payload_dir)
    return candidates


def _tiny_generator(value: Any, config: AppConfig) -> str:
    if isinstance(value, str) and not value.isdigit():
        name = value
    else:
        index = int(value)
        names = ["Real", *config.mixed_data.tiny_generators]
        if not 0 <= index < len(names):
            raise PreparationError(f"Tiny-GenImage has an unknown generator label: {value}")
        name = names[index]
    normalized = canonical_generator(name)
    for configured in config.mixed_data.tiny_generators:
        if canonical_generator(configured) == normalized:
            return configured
    if normalized == "real":
        return "Real"
    raise PreparationError(f"Tiny-GenImage has an unexpected generator: {name}")


def acquire_tiny_candidates(config: AppConfig) -> list[dict[str, Any]]:
    """Download and validate every small GenImage Parquet shard at a pinned revision."""
    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, get_token, hf_hub_download

    mixed = config.mixed_data
    token = get_token()
    info = _retry(
        config,
        "Tiny-GenImage metadata request",
        lambda: HfApi(token=token).dataset_info(
            mixed.tiny_genimage_repo_id,
            revision=mixed.tiny_genimage_revision,
            files_metadata=True,
        ),
    )
    if str(info.sha) != mixed.tiny_genimage_revision:
        raise PreparationError("Tiny-GenImage revision identity changed.")
    card_data = getattr(info, "card_data", None)
    declared_license = (
        card_data.get("license") if isinstance(card_data, Mapping) else getattr(card_data, "license", None)
    )
    declared_licenses = {str(value).casefold() for value in (
        declared_license if isinstance(declared_license, list) else [declared_license]
    )}
    if mixed.tiny_expected_license.casefold() not in declared_licenses:
        raise PreparationError(
            f"Tiny-GenImage license changed from {mixed.tiny_expected_license}: {declared_license}"
        )
    shards = sorted(
        (item for item in info.siblings if str(item.rfilename).endswith(".parquet")),
        key=lambda item: str(item.rfilename),
    )
    if not shards:
        raise PreparationError("Tiny-GenImage contains no Parquet shards.")
    declared_bytes = sum(int(item.size or 0) for item in shards)
    if declared_bytes > mixed.max_network_gb * GIB:
        raise PreparationError("Tiny-GenImage exceeds the mixed-data network budget.")
    cache_root = Path(mixed.cache_dir) / "tiny_genimage"
    source_state = _load_source_cache(cache_root, mixed.tiny_genimage_revision)
    candidates = [dict(item) for item in source_state["candidates"]]
    row_count = int(source_state.get("row_count", 0))
    completed_units = set(str(value) for value in source_state["completed_units"])
    seen_content = {str(item.get("content_sha256")) for item in candidates}
    seen_ids = {str(item["source_id"]) for item in candidates}
    pending_shards = [
        sibling for sibling in shards if str(sibling.rfilename) not in completed_units
    ]

    def download_shard(sibling: Any) -> Path:
        return Path(_retry(
            config,
            f"Tiny-GenImage shard {sibling.rfilename}",
            lambda: hf_hub_download(
                repo_id=mixed.tiny_genimage_repo_id,
                repo_type="dataset",
                revision=mixed.tiny_genimage_revision,
                filename=str(sibling.rfilename),
                token=token,
                local_dir=cache_root / "payload",
            ),
        ))

    for sibling, path in _bounded_prefetch(
        pending_shards, mixed.download_workers, download_shard
    ):
        upstream_split = "validation" if "validation" in str(sibling.rfilename) else "train"
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=32):
            for index, row in enumerate(batch.to_pylist()):
                row_count += 1
                label = int(row.get("label", -1))
                if label not in {0, 1}:
                    raise PreparationError("Tiny-GenImage contains an invalid binary label.")
                generator = _tiny_generator(row.get("generator"), config)
                if (label == 0) != (generator == "Real"):
                    raise PreparationError("Tiny-GenImage label and generator disagree.")
                content = _image_bytes(row.get("image"))
                digest = hashlib.sha256(content).hexdigest()
                if label == 0 and digest in seen_content:
                    continue
                seen_content.add(digest)
                local_path = _stage_bytes(content, cache_root)
                source_id = hashlib.sha256(
                    f"tiny_genimage\0{upstream_split}\0{generator}\0{digest}".encode()
                ).hexdigest()
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                candidates.append(
                    {
                        "source_dataset": "tiny_genimage",
                        "source_revision": mixed.tiny_genimage_revision,
                        "source_id": source_id,
                        "upstream_split": upstream_split,
                        "label": label,
                        "real_source": "ImageNet" if label == 0 else "",
                        "generator": generator if label else "real",
                        "generator_family": "GAN" if generator == "BigGAN" else "diffusion",
                        "architecture": generator if label else "real",
                        "model_id": generator if label else "ImageNet",
                        "subset": "Tiny-GenImage",
                        "content_group": "unknown",
                        "local_path": str(local_path),
                        "content_sha256": digest,
                    }
                )
        completed_units.add(str(sibling.rfilename))
        source_state.update(
            completed_units=sorted(completed_units), candidates=candidates, row_count=row_count
        )
        _save_source_cache(cache_root, source_state)
        path.unlink(missing_ok=True)
    if row_count != mixed.tiny_expected_rows:
        raise PreparationError(
            f"Tiny-GenImage contains {row_count} rows, expected {mixed.tiny_expected_rows}."
        )
    if {str(item["upstream_split"]) for item in candidates} != {"train", "validation"}:
        raise PreparationError("Tiny-GenImage must contain both train and validation shards.")
    return candidates


def assign_tiny(candidates: Sequence[Mapping[str, Any]], config: AppConfig) -> list[dict[str, Any]]:
    """Assign 8k real and 8k fake while recording the pinned source's empty SD14 class."""
    real = _stable_sorted((item for item in candidates if int(item["label"]) == 0), config)
    if len(real) < 8_000:
        raise PreparationError("Tiny-GenImage has fewer than 8,000 unique real images.")
    selected: list[dict[str, Any]] = []
    for index, item in enumerate(real[:10_000]):
        if index < 6_000:
            item["split"] = "train"
        elif index < 8_000:
            item["split"] = "val_id"
        else:
            item["split"] = "val_id" if (index - 8_000) % 4 == 0 else "train"
            item["selection_tier"] = 1
        selected.append(item)
    fake_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        if int(item["label"]) == 1:
            fake_groups[str(item["generator"])].append(dict(item))
    declared = set(config.mixed_data.tiny_generators)
    observed = set(fake_groups)
    missing = declared - observed
    allowed_missing = set(config.mixed_data.tiny_allowed_empty_generators)
    if observed - declared or missing != allowed_missing:
        raise PreparationError(
            "Tiny-GenImage generator rows differ from the pinned empty-class exception: "
            f"observed={sorted(observed)}, missing={sorted(missing)}."
        )
    active = [name for name in config.mixed_data.tiny_generators if name in observed]
    train_generators = [name for name in active if name != "Wukong"]
    if len(train_generators) != 6 or "Wukong" not in active:
        raise PreparationError("Tiny-GenImage requires six train/ID generators and held-out Wukong.")
    id_targets = square_root_allocation(
        1_000, {generator: len(fake_groups[generator]) for generator in train_generators}
    )
    for generator in active:
        ordered = _stable_sorted(fake_groups[generator], config)
        if generator == "Wukong":
            reserve_count = math.ceil(1_000 * config.mixed_data.reserve_multiplier)
            if len(ordered) < reserve_count:
                raise PreparationError("Tiny-GenImage Wukong cannot provide its DG reserve.")
            chosen = ordered[:reserve_count]
            for index, item in enumerate(chosen):
                item["split"] = "val_dg"
                item["selection_tier"] = int(index >= 1_000)
            selected.extend(chosen)
            continue
        id_count = id_targets[generator]
        train_pool = math.ceil(1_000 * config.mixed_data.reserve_multiplier)
        id_pool = math.ceil(id_count * config.mixed_data.reserve_multiplier)
        if len(ordered) < train_pool + id_pool:
            raise PreparationError(
                f"Tiny-GenImage generator {generator} cannot provide its train/ID reserves."
            )
        train_candidates = ordered[:train_pool]
        id_candidates = ordered[train_pool : train_pool + id_pool]
        for index, item in enumerate(train_candidates):
            item["split"] = "train"
            item["selection_tier"] = int(index >= 1_000)
            selected.append(item)
        for index, item in enumerate(id_candidates):
            item["split"] = "val_id"
            item["selection_tier"] = int(index >= id_count)
            selected.append(item)
    for item in selected:
        if int(item["label"]) == 1:
            item["quota_bucket"] = f"tiny_genimage\0{item['split']}\0{item['generator']}"
    return selected


class _BottomK:
    """Keep the smallest deterministic priorities without materializing a full CSV group."""

    def __init__(self, limit: int) -> None:
        import heapq

        self.limit = limit
        self.heap: list[tuple[int, str, dict[str, str]]] = []
        self.identities: set[str] = set()
        self._heapq = heapq

    def add(self, rank: str, row: Mapping[str, str]) -> None:
        value = int(rank, 16)
        identity = str(row.get("Image_path") or "")
        if identity in self.identities:
            return
        entry = (-value, identity, dict(row))
        if len(self.heap) < self.limit:
            self._heapq.heappush(self.heap, entry)
            self.identities.add(identity)
        elif entry > self.heap[0]:
            removed = self._heapq.heapreplace(self.heap, entry)
            self.identities.discard(removed[1])
            self.identities.add(identity)

    def rows(self) -> list[dict[str, str]]:
        return [item[2] for item in sorted(self.heap, key=lambda entry: (-entry[0], entry[1]))]


def _safe_wildfake_path(row: Mapping[str, str]) -> str:
    value = str(row.get("Image_path") or "").removeprefix("./")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise PreparationError(f"Unsafe WildFake source path: {value}")
    return str(path)


def _wildfake_archive(row: Mapping[str, str]) -> tuple[str, str]:
    source_path = _safe_wildfake_path(row)
    family = str(row.get("Generator") or "")
    architecture = str(row.get("Architecture") or "")
    if family == "Real":
        return f"Images/Real/{architecture}.zip", source_path.removeprefix("Real/")
    if family == "GAN_based":
        return "Images/GAN_based.zip", source_path
    if family == "Other_based":
        return "Images/Other_based.zip", source_path
    if family == "Diffusion_based":
        return f"Images/Diffusion_based/{architecture}.zip", source_path.removeprefix("Diffusion_based/")
    raise PreparationError(f"Unsupported WildFake family: {family}")


def _wildfake_deny_paths(config: AppConfig) -> set[str]:
    denied: set[str] = set()
    for configured in config.mixed_data.external_deny_manifests:
        path = Path(configured)
        if not path.is_file():
            continue
        for record in _read_manifest(path):
            if record.get("source_path"):
                denied.add(str(record["source_path"]).removeprefix("./"))
    return denied


def _modelscope_metadata(config: AppConfig) -> Path:
    from modelscope_hub import HubApi

    mixed = config.mixed_data
    path = Path(_retry(
        config,
        "WildFake train metadata request",
        lambda: HubApi().download_file(
            mixed.wildfake_repo_id,
            "dataset",
            mixed.wildfake_train_metadata_file,
            revision=mixed.wildfake_revision,
            local_dir=Path(mixed.cache_dir) / "wildfake" / "metadata",
        ),
    ))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != mixed.wildfake_train_metadata_sha256:
        raise PreparationError("WildFake train metadata identity changed.")
    return path


def _wildfake_allocations(
    counts: Mapping[tuple[str, str, str], int], config: AppConfig
) -> dict[tuple[str, str, str], int]:
    mixed = config.mixed_data
    allocations: dict[tuple[str, str, str], int] = {}
    for source in mixed.wildfake_train_real_sources:
        key = ("train", "real", source)
        if counts.get(key, 0) <= 0:
            raise PreparationError(f"WildFake train metadata is missing real source {source}.")
    real_train_counts = {
        source: counts[("train", "real", source)] for source in mixed.wildfake_train_real_sources
    }
    for source, amount in square_root_allocation(9_000, real_train_counts).items():
        allocations[("train", "real", source)] = amount
    for source in mixed.wildfake_dg_real_sources:
        key = ("val_dg", "real", source)
        if counts.get(key, 0) < 2_000:
            raise PreparationError(f"WildFake DG real source {source} cannot provide 2,000 images.")
        allocations[key] = 2_000
    for architecture in mixed.wildfake_dg_fake_architectures:
        key = ("val_dg", "fake", architecture)
        if counts.get(key, 0) <= 0:
            raise PreparationError(f"WildFake DG architecture {architecture} is missing.")
    for index, architecture in enumerate(mixed.wildfake_dg_fake_architectures):
        allocations[("val_dg", "fake", architecture)] = 667 if index < 2 else 666
    for family in ("Diffusion_based", "GAN_based", "Other_based"):
        family_counts = {
            architecture: amount
            for (split, group, architecture), amount in counts.items()
            if split == "train" and group == family
        }
        if not family_counts:
            raise PreparationError(f"WildFake train fake family {family} has no eligible architecture.")
        for architecture, amount in square_root_allocation(3_000, family_counts, cap=1_000).items():
            allocations[("train", family, architecture)] = amount
    return allocations


def acquire_wildfake(config: AppConfig) -> list[dict[str, Any]]:
    """Select from official train metadata and range-extract only selected ZIP members."""
    from modelscope_hub import HubApi
    from remotezip import RemoteZip

    mixed = config.mixed_data
    metadata = _modelscope_metadata(config)
    denied = _wildfake_deny_paths(config)
    train_real = set(mixed.wildfake_train_real_sources)
    dg_real = set(mixed.wildfake_dg_real_sources)
    dg_fake = set(mixed.wildfake_dg_fake_architectures)
    heldout = set(mixed.global_heldout_generator_aliases)
    counts: Counter[tuple[str, str, str]] = Counter()

    def bucket(row: Mapping[str, str]) -> tuple[str, str, str] | None:
        source_path = _safe_wildfake_path(row)
        if source_path in denied:
            return None
        label = int(row.get("IsFake", -1))
        family = str(row.get("Generator") or "")
        architecture = str(row.get("Architecture") or "")
        if label == 0 and architecture in train_real:
            return "train", "real", architecture
        if label == 0 and architecture in dg_real:
            return "val_dg", "real", architecture
        if label == 1 and architecture in dg_fake:
            return "val_dg", "fake", architecture
        if label == 1 and family in {"Diffusion_based", "GAN_based", "Other_based"}:
            if any(_matches_generator_alias(architecture, alias) for alias in heldout):
                return None
            if architecture in {"SD", "Midjourney", "DALLE"}:
                return None
            return "train", family, architecture
        return None

    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Generator", "Architecture", "IsFake", "Image_path"}
        if not required <= set(reader.fieldnames or []):
            raise PreparationError("WildFake train metadata is missing required columns.")
        for row in reader:
            selected_bucket = bucket(row)
            if selected_bucket is not None:
                counts[selected_bucket] += 1
    allocations = _wildfake_allocations(counts, config)
    reservoirs = {
        key: _BottomK(math.ceil(amount * mixed.reserve_multiplier))
        for key, amount in allocations.items()
    }
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            selected_bucket = bucket(row)
            if selected_bucket in reservoirs:
                reservoirs[selected_bucket].add(
                    stable_rank(config.project.seed, "wildfake", row["Image_path"]), row
                )
    descriptors: list[dict[str, Any]] = []
    bucket_needs = dict(allocations)
    for selected_bucket, reservoir in reservoirs.items():
        for row in reservoir.rows():
            archive, member = _wildfake_archive(row)
            descriptors.append(
                {
                    "bucket": selected_bucket,
                    "archive": archive,
                    "member": member,
                    "source_path": _safe_wildfake_path(row),
                    "row": row,
                }
            )
    files = {
        item.path: item
        for item in HubApi().list_repo_files(
            mixed.wildfake_repo_id, "dataset", revision=mixed.wildfake_revision
        )
    }
    missing = {item["archive"] for item in descriptors} - set(files)
    if missing:
        raise PreparationError(f"WildFake archives are missing: {sorted(missing)}")
    archive_bytes = sum(int(files[name].size or 0) for name in {item["archive"] for item in descriptors})
    # Range extraction transfers selected members, but the archive sum is retained for provenance only.
    LOGGER.info("WildFake selected members span %.2f GiB of logical archives.", archive_bytes / GIB)
    by_archive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in descriptors:
        by_archive[str(item["archive"])].append(item)
    cache_root = Path(mixed.cache_dir) / "wildfake"
    source_state = _load_source_cache(cache_root, mixed.wildfake_revision)
    extracted = [dict(item) for item in source_state["candidates"]]
    completed_units = set(str(value) for value in source_state["completed_units"])

    def signed_url(archive: str, session: requests.Session) -> str:
        url = (
            f"https://modelscope.cn/api/v1/datasets/{mixed.wildfake_repo_id}/repo"
            f"?Revision={quote(mixed.wildfake_revision, safe='')}"
            f"&FilePath={quote(archive, safe='')}"
        )
        response = session.get(url, allow_redirects=False, timeout=mixed.request_timeout_seconds)
        response.raise_for_status()
        location = response.headers.get("Location")
        if not location:
            raise PreparationError("ModelScope did not return a WildFake archive redirect.")
        return location

    def extract_archive(archive: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        session = requests.Session()
        retries = Retry(
            total=mixed.network_max_retries,
            connect=mixed.network_max_retries,
            read=mixed.network_max_retries,
            backoff_factor=mixed.network_retry_base_seconds,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        results: list[dict[str, Any]] = []
        try:
            with RemoteZip(
                signed_url(archive, session),
                session=session,
                timeout=mixed.request_timeout_seconds,
            ) as remote:
                names = set(remote.namelist())
                for item in sorted(items, key=lambda value: value["member"]):
                    if item["member"] not in names:
                        continue
                    try:
                        content = remote.read(item["member"])
                        local_path = _stage_bytes(content, cache_root)
                        # Decode now so corrupt upstream entries are replaced from the reserve.
                        describe_path(local_path, config)
                    except (OSError, ValueError, PreparationError):
                        LOGGER.warning("Skipping corrupt WildFake member %s.", item["source_path"])
                        continue
                    row = item["row"]
                    split, _group, _name = item["bucket"]
                    label = int(row["IsFake"])
                    architecture = str(row["Architecture"])
                    results.append(
                        {
                            "source_dataset": "wildfake",
                            "source_revision": mixed.wildfake_revision,
                            "source_id": hashlib.sha256(
                                f"wildfake\0{mixed.wildfake_revision}\0{item['source_path']}".encode()
                            ).hexdigest(),
                            "upstream_split": "train",
                            "split": split,
                            "label": label,
                            "real_source": architecture if label == 0 else "",
                            "generator": architecture if label else "real",
                            "generator_family": str(row["Generator"]),
                            "architecture": architecture if label else "real",
                            "model_id": architecture if label else architecture,
                            "subset": "WildFake",
                            "content_group": str(row.get("Category") or "unknown"),
                            "source_path": item["source_path"],
                            "archive_path": archive,
                            "archive_sha256": str(files[archive].sha256),
                            "local_path": str(local_path),
                            "quota_bucket": "\0".join(item["bucket"]),
                        }
                    )
        finally:
            session.close()
        return results

    with ThreadPoolExecutor(max_workers=mixed.download_workers) as executor:
        futures = {
            executor.submit(extract_archive, archive, items): archive
            for archive, items in sorted(by_archive.items())
            if archive not in completed_units
        }
        for future in as_completed(futures):
            extracted.extend(future.result())
            completed_units.add(futures[future])
            source_state.update(completed_units=sorted(completed_units), candidates=extracted)
            _save_source_cache(cache_root, source_state)
    final: list[dict[str, Any]] = []
    grouped_extracted: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in extracted:
        parts = str(item["quota_bucket"]).split("\0")
        grouped_extracted[(parts[0], parts[1], parts[2])].append(item)
    for selected_bucket, amount in allocations.items():
        unique = {
            str(item["source_id"]): item for item in grouped_extracted[selected_bucket]
        }
        ordered = _stable_sorted(unique.values(), config)
        if len(ordered) < amount:
            raise PreparationError(
                f"WildFake bucket {selected_bucket} has {len(ordered)} decoded images, needs {amount}."
            )
        for index, item in enumerate(ordered):
            item["selection_tier"] = int(index >= amount)
            final.append(item)
    return final


def _candidate_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(record["source_dataset"]), str(record["split"]), int(record["label"])


def _expected_quotas(config: AppConfig) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = {}
    for source, splits in config.mixed_data.source_quotas.items():
        for split, labels in splits.items():
            for label, amount in labels.items():
                result[(source, split, int(label))] = int(amount)
    return result


def _source_revision_set(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for record in records:
        source = str(record["source_dataset"])
        revision = str(record["source_revision"])
        previous = revisions.setdefault(source, revision)
        if previous != revision:
            raise PreparationError(f"Source {source} contains multiple revisions.")
    return dict(sorted(revisions.items()))


def _write_manifest(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    split_order = {name: index for index, name in enumerate(MIXED_SPLITS)}
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda item: (
            split_order[str(item["split"])],
            int(item["label"]),
            str(item["source_dataset"]),
            str(item["id"]),
        ),
    )
    atomic_write_text(
        path,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered),
    )


def _state_fingerprint(config: AppConfig) -> str:
    deny_identities = {}
    for configured_path in config.mixed_data.external_deny_manifests:
        path = Path(configured_path)
        deny_identities[configured_path] = (
            _cached_file_sha256(path) if path.is_file() else "missing"
        )
    payload = {
        "seed": config.project.seed,
        "quotas": config.mixed_data.source_quotas,
        "revisions": {
            "shanmuk": config.mixed_data.shanmuk_revision,
            "wildfake": config.mixed_data.wildfake_revision,
            "community_forensics": config.mixed_data.community_revision,
            "tiny_genimage": config.mixed_data.tiny_genimage_revision,
        },
        "heldout": config.mixed_data.global_heldout_generator_aliases,
        "tiny_declared_generators": config.mixed_data.tiny_generators,
        "tiny_allowed_empty_generators": config.mixed_data.tiny_allowed_empty_generators,
        "deny_manifests": deny_identities,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _write_state(
    config: AppConfig,
    records: Sequence[Mapping[str, Any]],
    complete: bool,
    reason: str,
    *,
    include_ids: bool = True,
) -> None:
    counts = Counter(
        f"{record['source_dataset']}:{record['split']}:{record['label']}" for record in records
    )
    state = {
        "schema_version": 2,
        "complete": complete,
        "sampling_config_sha256": _state_fingerprint(config),
        "selected": len(records),
        "counts": dict(sorted(counts.items())),
        "stop_reason": reason,
    }
    if include_ids:
        state["selected_ids"] = sorted(str(record["id"]) for record in records)
    atomic_write_text(Path(config.mixed_data.state_path), json.dumps(state, indent=2, sort_keys=True) + "\n")


def _distribution_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "source_dataset", "split", "label", "real_source", "generator_family",
        "architecture", "model_id", "subset", "resolution_bucket", "aspect_bucket",
        "format_bucket", "encoding_density_bucket",
    )
    distributions = {
        field: dict(sorted(Counter(str(record.get(field, "unknown")) for record in records).items()))
        for field in fields
    }
    by_label_nuisance: dict[str, Any] = {}
    for field in ("resolution_bucket", "aspect_bucket", "format_bucket", "encoding_density_bucket"):
        by_label_nuisance[field] = {
            str(label): dict(
                sorted(Counter(str(record[field]) for record in records if int(record["label"]) == label).items())
            )
            for label in (0, 1)
        }
    return {
        "schema_version": 1,
        "total": len(records),
        "distributions": distributions,
        "nuisance_by_label": by_label_nuisance,
        "encoding_density_quantile_thresholds": _encoding_density_thresholds(records),
    }


def _encoding_density_thresholds(records: Sequence[Mapping[str, Any]]) -> list[float]:
    values = sorted(float(record["encoding_density"]) for record in records)
    if not values:
        return []
    return [values[min(len(values) - 1, int(len(values) * fraction))] for fraction in (0.25, 0.50, 0.75)]


def _apply_encoding_density_quantiles(records: Sequence[dict[str, Any]]) -> None:
    thresholds = _encoding_density_thresholds(records)
    for record in records:
        density = float(record["encoding_density"])
        index = sum(density > threshold for threshold in thresholds)
        record["encoding_density_bucket"] = f"q{index + 1}"


def _audit(config: AppConfig, records: Sequence[Mapping[str, Any]], dedup: DedupIndex) -> dict[str, Any]:
    expected = _expected_quotas(config)
    actual = Counter(_candidate_key(record) for record in records)
    community_fake = [
        record
        for record in records
        if record["source_dataset"] == "community_forensics"
        and int(record["label"]) == 1
        and record["split"] in {"train", "val_id"}
    ]
    desired_subset = {
        key: round(15_000 * float(weight))
        for key, weight in config.mixed_data.community_subset_weights.items()
    }
    desired_architecture = {
        key: round(15_000 * float(weight))
        for key, weight in config.mixed_data.community_architecture_weights.items()
    }
    actual_subset = Counter(str(record.get("subset", "unknown")) for record in community_fake)
    actual_architecture = Counter(str(record.get("architecture", "other")) for record in community_fake)
    tiny_observed = sorted(
        {
            str(record.get("generator"))
            for record in records
            if record["source_dataset"] == "tiny_genimage" and int(record["label"]) == 1
        }
    )
    acquisition: dict[str, Any] = {}
    for source in ("community_forensics", "tiny_genimage", "wildfake"):
        state_path = _source_cache_path(Path(config.mixed_data.cache_dir) / source)
        if not state_path.is_file():
            continue
        source_state = json.loads(state_path.read_text(encoding="utf-8"))
        unique_paths = {
            str(item.get("local_path")) for item in source_state.get("candidates", [])
            if Path(str(item.get("local_path", ""))).is_file()
        }
        acquisition[source] = {
            "completed_units": len(set(source_state.get("completed_units", []))),
            "candidate_count": len(source_state.get("candidates", [])),
            "staged_bytes": sum(Path(path).stat().st_size for path in unique_paths),
        }
    return {
        "schema_version": 2,
        "complete": all(actual[key] == amount for key, amount in expected.items()),
        "selected": len(records),
        "target_total": config.mixed_data.target_total,
        "class_counts": dict(sorted(Counter(str(record["label"]) for record in records).items())),
        "split_counts": dict(sorted(Counter(str(record["split"]) for record in records).items())),
        "source_split_label_counts": {
            f"{source}:{split}:{label}": actual[(source, split, label)]
            for source, split, label in sorted(expected)
        },
        "source_revisions": _source_revision_set(records),
        "duplicate_count": sum(event.get("kind") != "unsafe_decode" for event in dedup.events),
        "unsafe_decode_count": sum(event.get("kind") == "unsafe_decode" for event in dedup.events),
        "heldout_generators": config.mixed_data.global_heldout_generator_aliases,
        "licenses": {
            "shanmuk": "Upstream component licenses; non-commercial research restrictions apply.",
            "wildfake": "Review the pinned upstream dataset terms before redistribution.",
            "community_forensics": "Per-image and generator licenses remain authoritative.",
            "tiny_genimage": "CC BY-NC-SA 4.0; third-party GenImage derivative.",
        },
        "network_budget_gib": config.mixed_data.max_network_gb,
        "acquisition": acquisition,
        "community_marginals": {
            "subset_target": desired_subset,
            "subset_actual": dict(sorted(actual_subset.items())),
            "architecture_target": desired_architecture,
            "architecture_actual": dict(sorted(actual_architecture.items())),
            "fallback_delta": {
                "subset": {
                    key: actual_subset[key] - amount for key, amount in desired_subset.items()
                },
                "architecture": {
                    key: actual_architecture[key] - amount
                    for key, amount in desired_architecture.items()
                },
            },
        },
        "tiny_genimage_schema_anomaly": {
            "declared_generators": config.mixed_data.tiny_generators,
            "observed_generators": tiny_observed,
            "declared_but_empty": config.mixed_data.tiny_allowed_empty_generators,
            "policy": "The empty SD14 class is not synthesized or silently replaced; its quota is distributed across the six active train/ID generators.",
        },
        "nuisance_report": config.data.nuisance_report_path,
    }


def commit_candidates(
    candidates: Sequence[Mapping[str, Any]], config: AppConfig, deny_records: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], DedupIndex]:
    """Validate, globally deduplicate, and content-address exactly the configured quotas."""
    expected = _expected_quotas(config)
    counts: Counter[tuple[str, str, int]] = Counter()
    auxiliary_counts: Counter[str] = Counter()
    community_model_counts: Counter[str] = Counter()
    materialized_deny_records = [dict(record) for record in deny_records]
    LOGGER.info(
        "Building the external perceptual search index for %d deny records.",
        len(materialized_deny_records),
    )
    dedup = DedupIndex(config, materialized_deny_records)
    LOGGER.info(
        "External perceptual search index is ready with %d unique pHash values.",
        len(dedup.hashes),
    )
    output_root = Path(config.mixed_data.output_dir)
    records: list[dict[str, Any]] = []
    heldout = set(config.mixed_data.global_heldout_generator_aliases)
    materialized_candidates = [dict(item) for item in candidates]
    auxiliary_expected = Counter(
        str(item["quota_bucket"])
        for item in materialized_candidates
        if item.get("quota_bucket") and int(item.get("selection_tier", 0)) == 0
    )
    descriptions, materialized_candidates, invalid_candidates = _describe_candidates_resumable(
        materialized_candidates, config
    )
    for candidate in invalid_candidates:
        LOGGER.warning("Skipping an unsafe or missing mixed-data reserve candidate: %s", candidate["source_id"])
        dedup.events.append(
            {"kind": "unsafe_decode", "discarded": candidate["source_id"], "label": int(candidate["label"])}
        )
    LOGGER.info(
        "Candidate hashing is complete: %d valid candidates and %d rejected candidates.",
        len(materialized_candidates),
        len(invalid_candidates),
    )
    nuisance_counts = Counter(
        (
            _candidate_key(candidate),
            descriptions[(str(candidate["source_dataset"]), str(candidate["source_id"]))]["resolution_bucket"],
            descriptions[(str(candidate["source_dataset"]), str(candidate["source_id"]))]["aspect_bucket"],
            descriptions[(str(candidate["source_dataset"]), str(candidate["source_id"]))]["format_bucket"],
            descriptions[(str(candidate["source_dataset"]), str(candidate["source_id"]))]["encoding_density_bucket"],
        )
        for candidate in materialized_candidates
    )

    def nuisance_priority(candidate: Mapping[str, Any]) -> float:
        description = descriptions[(str(candidate["source_dataset"]), str(candidate["source_id"]))]
        bucket = (
            _candidate_key(candidate),
            description["resolution_bucket"],
            description["aspect_bucket"],
            description["format_bucket"],
            description["encoding_density_bucket"],
        )
        rank = int(
            stable_rank(config.project.seed, "nuisance", candidate["source_dataset"], candidate["source_id"]),
            16,
        )
        uniform = (rank + 1) / (2**256 + 1)
        # Per-item weight is 1/sqrt(bucket size); lower exponential-race keys win.
        return -math.log(uniform) * math.sqrt(nuisance_counts[bucket])

    def materialize(candidate: dict[str, Any]) -> dict[str, Any]:
        local_path = Path(str(candidate["local_path"]))
        description = descriptions[(str(candidate["source_dataset"]), str(candidate["source_id"]))]
        record = {**candidate, **description}
        record_id = description["content_sha256"]
        record["id"] = record_id
        record["provenance_sources"] = [
            {
                "source_dataset": candidate["source_dataset"],
                "source_revision": candidate["source_revision"],
                "source_id": candidate["source_id"],
            }
        ]
        return record

    def publish(record: dict[str, Any]) -> None:
        local_path = Path(str(record["local_path"]))
        suffix = _extension(str(record["format"]))
        relative = Path("objects") / str(record["id"])[:2] / f"{record['id']}.{suffix}"
        _link_or_copy(local_path, output_root / relative)
        record["path"] = str(relative)
        auxiliary_bucket = str(record.pop("quota_bucket", ""))
        records.append(record)
        counts[_candidate_key(record)] += 1
        if record["source_dataset"] == "community_forensics" and int(record["label"]) == 1:
            community_model_counts[str(record.get("model_id") or "unknown")] += 1
        if auxiliary_bucket:
            auxiliary_counts[auxiliary_bucket] += 1
        if len(records) % config.mixed_data.checkpoint_every == 0:
            _write_state(
                config,
                records,
                False,
                "periodic content checkpoint",
                include_ids=False,
            )

    # Paired Shanmuk records are admitted or rejected as an indivisible parent group.
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in materialized_candidates:
        if candidate["source_dataset"] == "shanmuk":
            pairs[str(candidate["pair_id"])].append(candidate)
    units: list[tuple[str, str, str, int, float, list[dict[str, Any]]]] = []
    for pair_id, pair in pairs.items():
        if (
            len(pair) != 2
            or {int(item["label"]) for item in pair} != {0, 1}
            or len({str(item["split"]) for item in pair}) != 1
        ):
            raise PreparationError(f"Shanmuk reserve parent {pair_id} is not an intact split-local pair.")
        units.append(
            (
                str(pair[0]["split"]),
                "shanmuk",
                pair_id,
                int(pair[0].get("selection_tier", 0)),
                sum(nuisance_priority(item) for item in pair) / 2.0,
                pair,
            )
        )
    for candidate in materialized_candidates:
        if candidate["source_dataset"] != "shanmuk":
            units.append(
                (
                    str(candidate["split"]),
                    str(candidate["source_dataset"]),
                    str(candidate["source_id"]),
                    int(candidate.get("selection_tier", 0)),
                    nuisance_priority(candidate),
                    [candidate],
                )
            )
    units.sort(
        key=lambda unit: (
            -SPLIT_PRIORITY[unit[0]],
            SOURCE_ORDER[unit[1]],
            unit[3],
            unit[4],
            stable_rank(config.project.seed, unit[1], unit[2]),
        )
    )

    for _split, source, unit_id, _tier, _priority, unit in tqdm(
        units, desc="Deduplicate mixed candidates"
    ):
        if source == "shanmuk":
            keys = [_candidate_key(item) for item in unit]
            if any(counts[key] >= expected[key] for key in keys):
                continue
            prepared = [materialize(item) for item in unit]
            collisions = [dedup.collision(item) for item in prepared]
            if any(collisions):
                for item, collision in zip(prepared, collisions):
                    if collision is None:
                        continue
                    kind, existing = collision
                    if int(existing.get("label", item["label"])) != int(item["label"]):
                        raise PreparationError(
                            f"Conflicting labels share a confirmed {kind} duplicate in pair {unit_id}."
                        )
                    dedup.events.append(
                        {
                            "kind": kind,
                            "kept": existing.get("id", existing.get("source_id")),
                            "discarded": item["source_id"],
                            "label": int(item["label"]),
                            "pair_rejected": unit_id,
                        }
                    )
                continue
            for item in prepared:
                dedup.add(item)
                publish(item)
            continue

        candidate = unit[0]
        key = _candidate_key(candidate)
        if key not in expected or counts[key] >= expected[key]:
            continue
        auxiliary_bucket = str(candidate.get("quota_bucket", ""))
        if auxiliary_bucket and auxiliary_counts[auxiliary_bucket] >= auxiliary_expected[auxiliary_bucket]:
            continue
        if candidate["source_dataset"] == "community_forensics" and int(candidate["label"]) == 1:
            model = str(candidate.get("model_id") or "unknown")
            model_cap = (
                config.mixed_data.community_systematic_model_cap
                if str(candidate.get("subset")).casefold() == "systematic"
                else config.mixed_data.community_other_model_cap
            )
            if community_model_counts[model] >= model_cap:
                continue
        if int(candidate["label"]) == 1 and candidate["split"] != "val_dg":
            normalized_values = [
                canonical_generator(candidate.get(field, ""))
                for field in ("generator", "architecture", "model_id")
            ]
            if any(_matches_generator_alias(value, alias) for alias in heldout for value in normalized_values):
                continue
        record = materialize(candidate)
        try:
            dedup.add(record)
        except DuplicateCandidate:
            if candidate.get("pair_id"):
                raise PreparationError(
                    f"A Shanmuk parent pair collided with another selected or external image: {candidate['pair_id']}"
                )
            continue
        publish(record)
    missing = {key: amount - counts[key] for key, amount in expected.items() if counts[key] != amount}
    if missing:
        raise PreparationError(f"Mixed-data reserves cannot meet quotas after deduplication: {missing}")
    if len(records) != config.mixed_data.target_total:
        raise PreparationError("Mixed-data records do not match target_total after quota validation.")
    auxiliary_missing = {
        bucket: amount - auxiliary_counts[bucket]
        for bucket, amount in auxiliary_expected.items()
        if auxiliary_counts[bucket] != amount
    }
    if auxiliary_missing:
        raise PreparationError(f"Mixed-data sub-bucket reserves cannot meet quotas: {auxiliary_missing}")
    _apply_encoding_density_quantiles(records)
    for record in records:
        record.pop("local_path", None)
    return records, dedup


def prepare_mixed_dataset(config: AppConfig) -> dict[str, Any]:
    """Build the complete four-source pool and atomically publish its manifests and audits."""
    mixed = config.mixed_data
    if not mixed.enabled:
        raise PreparationError("mixed_data.enabled must be true for aigc-prepare.")
    output_parent = Path(mixed.output_dir).resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_parent).free / GIB
    if free_gib < mixed.storage_warning_min_gb:
        raise PreparationError(
            f"Only {free_gib:.1f} GiB is free; mixed preparation requires at least "
            f"{mixed.storage_warning_min_gb:.1f} GiB of working headroom."
        )
    if free_gib < mixed.storage_warning_max_gb:
        LOGGER.warning(
            "Only %.1f GiB is free; mixed preparation may require up to %.1f GiB.",
            free_gib,
            mixed.storage_warning_max_gb,
        )
    if (
        Path(mixed.state_path).is_file()
        and Path(mixed.audit_path).is_file()
        and Path(mixed.manifest_path).is_file()
    ):
        state = json.loads(Path(mixed.state_path).read_text(encoding="utf-8"))
        if state.get("sampling_config_sha256") != _state_fingerprint(config):
            raise PreparationError("Existing mixed-data state uses a different configuration.")
        if bool(state.get("complete")):
            audit = json.loads(Path(mixed.audit_path).read_text(encoding="utf-8"))
            manifest_records = _read_manifest(Path(mixed.manifest_path))
            missing_objects = [
                record["path"]
                for record in manifest_records
                if not (Path(mixed.output_dir) / str(record["path"])).is_file()
            ]
            if (
                not bool(audit.get("complete"))
                or len(manifest_records) != mixed.target_total
                or missing_objects
            ):
                raise PreparationError(
                    "Completed mixed-data state is inconsistent with its manifest or object store."
                )
            if config.nuisance_audit.enabled and not Path(config.data.nuisance_report_path).is_file():
                LOGGER.info("Mixed dataset is complete; running the missing nuisance audit.")
                try:
                    from aigc_recognizer.data.nuisance import run_nuisance_audit

                    run_nuisance_audit(config)
                except Exception:
                    LOGGER.exception("Nuisance audit failed; the complete mixed dataset remains usable.")
            return audit
    Path(mixed.output_dir).mkdir(parents=True, exist_ok=True)
    deny_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    dedup: DedupIndex | None = None
    try:
        LOGGER.info("Validating credentials and pinned source revisions.")
        _preflight_sources(config)
        LOGGER.info("Loading external evaluation leakage deny lists.")
        deny_records = load_external_denylist(config)
        LOGGER.info("Selecting reusable Shanmuk pairs.")
        shanmuk = select_shanmuk(config)
        LOGGER.info("Selecting WildFake train domains.")
        wildfake = acquire_wildfake(config)
        LOGGER.info("Collecting Community Forensics candidates.")
        community = _assign_community(acquire_community_candidates(config), config)
        LOGGER.info("Collecting Tiny-GenImage candidates.")
        tiny = assign_tiny(acquire_tiny_candidates(config), config)
        _compact_source_candidate_cache(
            Path(mixed.cache_dir) / "tiny_genimage",
            mixed.tiny_genimage_revision,
            tiny,
        )
        records, dedup = commit_candidates(shanmuk + wildfake + community + tiny, config, deny_records)
        _write_manifest(Path(mixed.manifest_path), records)
        distribution = _distribution_report(records)
        atomic_write_text(
            Path(mixed.distribution_report_path),
            json.dumps(distribution, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            Path(mixed.dedup_report_path),
            json.dumps(
                {"schema_version": 1, "events": dedup.events, "count": len(dedup.events)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        audit = _audit(config, records, dedup)
        atomic_write_text(Path(mixed.audit_path), json.dumps(audit, indent=2, sort_keys=True) + "\n")
        _write_state(config, records, True, "all source quotas satisfied")
    except BaseException as exc:
        _write_state(config, records, False, f"interrupted by {type(exc).__name__}")
        partial = {
            "schema_version": 2,
            "complete": False,
            "selected": len(records),
            "stop_reason": f"interrupted by {type(exc).__name__}: {exc}",
        }
        atomic_write_text(Path(mixed.audit_path), json.dumps(partial, indent=2, sort_keys=True) + "\n")
        raise
    if config.nuisance_audit.enabled:
        try:
            from aigc_recognizer.data.nuisance import run_nuisance_audit

            LOGGER.info("Mixed dataset is complete; starting nuisance audit.")
            run_nuisance_audit(config)
        except Exception:
            LOGGER.exception("Nuisance audit failed; the complete mixed dataset remains usable.")
    return audit


def main() -> None:
    """Run the public mixed-data preparation command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Prepare the pinned 80k multi-source AIGC training pool.")
    args = parser.parse_args()
    try:
        audit = prepare_mixed_dataset(load_config(args.config, args.set))
    except (PreparationError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    LOGGER.info("Mixed dataset preparation completed with %d images.", audit["selected"])


if __name__ == "__main__":
    main()
