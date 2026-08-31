"""Prepare broad WildFake and SID-Set samples for the shared evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import requests
from modelscope_hub import HubApi
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.prepare import (
    PreparationError,
    _atomic_write_bytes,
    atomic_write_text,
    validate_and_describe_image,
)

LOGGER = logging.getLogger(__name__)
GIB = 1024**3


def _stable_rank(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()


def _even_quotas(total: int, names: Iterable[str]) -> dict[str, int]:
    """Allocate an exact total across sorted strata with at most one count difference."""
    ordered = sorted(set(names))
    if not ordered:
        raise PreparationError("Cannot allocate a sample across zero strata.")
    base, remainder = divmod(total, len(ordered))
    return {name: base + int(index < remainder) for index, name in enumerate(ordered)}


def select_wildfake_rows(
    rows: Iterable[Mapping[str, str]],
    config: AppConfig,
    additional_excluded_paths: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build a class-balanced, hierarchy-uniform sample from the official test split."""
    broad = config.wildfake_evaluation
    real_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    fake_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    allowed_families = set(broad.fake_families)
    allowed_architectures = set(broad.fake_architectures)
    allowed_real = set(broad.real_sources)
    excluded_paths = {
        str(path).removeprefix("./") for path in broad.excluded_source_paths
    }
    excluded_paths.update(str(path).removeprefix("./") for path in additional_excluded_paths)
    for source in rows:
        row = dict(source)
        source_path = str(row.get("Image_path") or "")
        path = PurePosixPath(source_path.removeprefix("./"))
        if not source_path or path.is_absolute() or ".." in path.parts:
            raise PreparationError(f"Unsafe WildFake source path: {source_path}")
        if str(path) in excluded_paths:
            continue
        label = int(row.get("IsFake", -1))
        family = str(row.get("Generator") or "")
        architecture = str(row.get("Architecture") or "")
        if label == 0 and architecture in allowed_real:
            real_groups[architecture].append(row)
        elif (
            label == 1
            and family in allowed_families
            and architecture in allowed_architectures
        ):
            fake_groups[(family, architecture)].append(row)

    missing_real = allowed_real - set(real_groups)
    if missing_real:
        raise PreparationError(f"WildFake metadata is missing real strata: {sorted(missing_real)}")
    present_fake_architectures = {architecture for _family, architecture in fake_groups}
    missing_fake = allowed_architectures - present_fake_architectures
    if missing_fake:
        raise PreparationError(
            f"WildFake metadata is missing fake strata: {sorted(missing_fake)}"
        )
    real_quotas = _even_quotas(broad.target_real, allowed_real)
    selected: list[dict[str, Any]] = []
    for source_name, candidates in real_groups.items():
        quota = real_quotas[source_name]
        ranked = sorted(candidates, key=lambda item: _stable_rank(config.project.seed, item["Image_path"]))
        if len(ranked) < quota:
            raise PreparationError(f"WildFake real stratum {source_name} cannot meet quota {quota}.")
        selected.extend(_wildfake_descriptor(item) for item in ranked[:quota])

    family_quotas = _even_quotas(broad.target_fake, allowed_families)
    for family in sorted(allowed_families):
        architectures = sorted(
            architecture for group, architecture in fake_groups if group == family
        )
        configured = sorted(
            architecture
            for architecture in allowed_architectures
            if (family, architecture) in fake_groups
        )
        if architectures != configured or not configured:
            raise PreparationError(f"WildFake fake family {family} has no configured strata.")
        architecture_quotas = _even_quotas(family_quotas[family], configured)
        for architecture in configured:
            candidates = fake_groups[(family, architecture)]
            quota = architecture_quotas[architecture]
            ranked = sorted(
                candidates,
                key=lambda item: _stable_rank(config.project.seed, item["Image_path"]),
            )
            if len(ranked) < quota:
                raise PreparationError(
                    f"WildFake fake stratum {family}/{architecture} cannot meet quota {quota}."
                )
            selected.extend(_wildfake_descriptor(item) for item in ranked[:quota])
    selected.sort(key=lambda item: (item["label"], item["family"], item["architecture"], item["source_member"]))
    return selected


def _wildfake_descriptor(row: Mapping[str, str]) -> dict[str, Any]:
    source_path = str(row["Image_path"]).removeprefix("./")
    family = str(row["Generator"])
    architecture = str(row["Architecture"])
    if family == "Real":
        archive = f"Images/Real/{architecture}.zip"
        member = source_path.removeprefix("Real/")
    elif family == "GAN_based":
        archive = "Images/GAN_based.zip"
        member = source_path
    elif family == "Other_based":
        archive = "Images/Other_based.zip"
        member = source_path
    elif family == "Diffusion_based":
        archive = f"Images/Diffusion_based/{architecture}.zip"
        member = source_path.removeprefix("Diffusion_based/")
    else:
        raise PreparationError(f"Unsupported WildFake family: {family}")
    return {
        "label": int(row["IsFake"]),
        "family": family,
        "architecture": architecture,
        "weight": str(row.get("Weight") or ""),
        "category": str(row.get("Category") or ""),
        "is_advanced": int(row.get("IsAdvanced", 0)),
        "source_path": source_path,
        "source_member": member,
        "archive_path": archive,
    }


def _modelscope_session(config: AppConfig) -> requests.Session:
    broad = config.wildfake_evaluation
    retry = Retry(
        total=broad.network_max_retries,
        connect=broad.network_max_retries,
        read=broad.network_max_retries,
        backoff_factor=broad.network_retry_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _signed_modelscope_url(
    config: AppConfig, archive_path: str, session: requests.Session
) -> str:
    broad = config.wildfake_evaluation
    url = (
        f"https://modelscope.cn/api/v1/datasets/{broad.repo_id}/repo"
        f"?Revision={quote(broad.revision, safe='')}"
        f"&FilePath={quote(archive_path, safe='')}"
    )
    response = session.get(url, allow_redirects=False, timeout=broad.request_timeout_seconds)
    response.raise_for_status()
    location = response.headers.get("Location")
    if not location:
        raise PreparationError("ModelScope did not return a CDN redirect for an archive.")
    return location


def _detect_corrupt_gan_members(
    config: AppConfig, archive_sha256: str
) -> set[str]:
    """Detect zero-filled image payloads using ZIP ratios, then confirm by decoding."""
    broad = config.wildfake_evaluation
    if not broad.detect_extreme_zip_compression:
        return set()
    cache_path = Path(broad.integrity_cache_path)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOGGER.warning("Ignoring an unreadable WildFake archive integrity cache.")
            cached = {}
        if (
            cached.get("archive_path") == "Images/GAN_based.zip"
            and cached.get("archive_sha256") == archive_sha256
            and float(cached.get("compression_ratio", -1.0))
            == broad.extreme_zip_compression_ratio
        ):
            rejected = {str(path) for path in cached.get("corrupt_source_paths", [])}
            LOGGER.info(
                "Loaded %d verified corrupt GAN members from the integrity cache.",
                len(rejected),
            )
            return rejected

    archive_path = "Images/GAN_based.zip"
    session = _modelscope_session(config)
    rejected: set[str] = set()
    suspicious_count = 0
    try:
        signed_url = _signed_modelscope_url(config, archive_path, session)
        with RemoteZip(
            signed_url, session=session, timeout=broad.request_timeout_seconds
        ) as archive:
            suspicious = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and Path(item.filename).suffix.casefold()
                in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                and item.file_size >= 1024
                and item.compress_size / item.file_size
                < broad.extreme_zip_compression_ratio
            ]
            suspicious_count = len(suspicious)
            for item in tqdm(suspicious, desc="Validate suspicious GAN members"):
                content = archive.read(item.filename)
                try:
                    validate_and_describe_image(content, config)
                except PreparationError:
                    rejected.add(item.filename)
    finally:
        session.close()
    report = {
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "compression_ratio": broad.extreme_zip_compression_ratio,
        "suspicious_members_checked": suspicious_count,
        "corrupt_source_paths": sorted(rejected),
    }
    atomic_write_text(cache_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    LOGGER.warning(
        "Verified and excluded %d corrupt GAN archive members after checking %d "
        "extreme-compression candidates.",
        len(rejected),
        suspicious_count,
    )
    return rejected


def _wildfake_id(config: AppConfig, source_path: str) -> str:
    broad = config.wildfake_evaluation
    return hashlib.sha256(
        f"{broad.repo_id}\0{broad.revision}\0{source_path}".encode("utf-8")
    ).hexdigest()


def _load_records(
    manifest_path: Path, root: Path, *, key_field: str = "id"
) -> dict[str, dict[str, Any]]:
    if not manifest_path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (root / record["path"]).is_file():
            records[str(record[key_field])] = record
    return records


def _wildfake_checkpoint(
    config: AppConfig,
    records: Mapping[str, Mapping[str, Any]],
    expected_ids: set[str],
    archive_hashes: Mapping[str, str],
    fingerprint: str,
    complete: bool,
    reason: str,
) -> dict[str, Any]:
    broad = config.wildfake_evaluation
    excluded_corrupt_paths = set(broad.excluded_source_paths)
    integrity_cache = Path(broad.integrity_cache_path)
    if integrity_cache.is_file():
        cached_integrity = json.loads(integrity_cache.read_text(encoding="utf-8"))
        excluded_corrupt_paths.update(
            str(path) for path in cached_integrity.get("corrupt_source_paths", [])
        )
    ordered = sorted(records.values(), key=lambda item: (item["label"], item["id"]))
    lines = "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered)
    counts = {
        "real": sum(int(item["label"]) == 0 for item in ordered),
        "fake": sum(int(item["label"]) == 1 for item in ordered),
    }
    state = {
        "complete": complete,
        "sampling_config_sha256": fingerprint,
        "expected_ids": sorted(expected_ids),
        "stored_ids": sorted(records),
        "archive_sha256": dict(sorted(archive_hashes.items())),
        "stop_reason": reason,
    }
    strata = Counter(
        f"{item['generator_family']}/{item['architecture']}" for item in ordered
    )
    audit = {
        "complete": complete,
        "stop_reason": reason,
        "repo_id": broad.repo_id,
        "revision": broad.revision,
        "counts": counts,
        "expected_counts": {"real": broad.target_real, "fake": broad.target_fake},
        "selected_bytes": sum(int(item["bytes"]) for item in ordered),
        "sampling": {
            "source_split": "official total_split/test_metadata.csv",
            "class_balance": "equal",
            "fake_allocation": "equal by family, then equal by architecture",
            "real_allocation": "equal by source",
            "excluded_default_architectures": {
                "DALLE": "already covered by the challenge-prescribed evaluation",
                "coco": "already covered by the challenge-prescribed evaluation",
                "SD": "very large multipart selective archives",
                "Midjourney": "very large multipart selective archives",
            },
            "excluded_corrupt_source_paths": sorted(excluded_corrupt_paths),
            "strata": dict(sorted(strata.items())),
        },
        "archives": dict(sorted(archive_hashes.items())),
    }
    # Build every serialized object before committing any file. The state file is
    # written last and therefore acts as the checkpoint commit marker.
    atomic_write_text(Path(broad.manifest_path), lines)
    atomic_write_text(Path(broad.audit_path), json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_write_text(Path(broad.state_path), json.dumps(state, indent=2, sort_keys=True) + "\n")
    return audit


def _extract_wildfake_archive(
    config: AppConfig,
    archive_path: str,
    selected: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    expected_ids: set[str],
    archive_hashes: Mapping[str, str],
    fingerprint: str,
    lock: threading.Lock,
) -> None:
    broad = config.wildfake_evaluation
    output_root = Path(broad.output_dir)
    attempt = 0
    while True:
        with lock:
            pending = [
                item
                for item in selected
                if _wildfake_id(config, item["source_path"]) not in records
            ]
        if not pending:
            return
        session = _modelscope_session(config)
        try:
            signed_url = _signed_modelscope_url(config, archive_path, session)
            with RemoteZip(
                signed_url, session=session, timeout=broad.request_timeout_seconds
            ) as archive:
                available = {item.filename: item for item in archive.infolist()}
                missing = [item["source_member"] for item in pending if item["source_member"] not in available]
                if missing:
                    raise PreparationError(
                        f"WildFake archive {archive_path} is missing selected member {missing[0]}."
                    )
                pending.sort(key=lambda item: available[item["source_member"]].header_offset)
                for item in tqdm(pending, desc=f"Extract {Path(archive_path).name}"):
                    content = archive.read(item["source_member"])
                    logical_id = _wildfake_id(config, item["source_path"])
                    try:
                        description = validate_and_describe_image(content, config)
                    except PreparationError as exc:
                        raise PreparationError(
                            "WildFake contains an undecodable selected image: "
                            f"{item['source_path']} in {archive_path}. Add the verified corrupt "
                            "path to wildfake_evaluation.excluded_source_paths and rerun."
                        ) from exc
                    extension = Path(item["source_member"]).suffix.lower() or ".img"
                    relative = Path("images") / ("fake" if item["label"] else "real") / f"{logical_id}{extension}"
                    destination = output_root / relative
                    if not destination.is_file():
                        _atomic_write_bytes(destination, content)
                    record = {
                        "id": logical_id,
                        "path": str(relative),
                        "label": int(item["label"]),
                        "split": "external_test",
                        "source_name": str(item["architecture"]),
                        "generator_family": str(item["family"]),
                        "architecture": str(item["architecture"]),
                        "weight": str(item["weight"]),
                        "category": str(item["category"]),
                        "is_advanced": int(item["is_advanced"]),
                        "source_member": str(item["source_member"]),
                        "source_path": str(item["source_path"]),
                        "source_repo": broad.repo_id,
                        "source_revision": broad.revision,
                        "archive_path": archive_path,
                        "archive_sha256": archive_hashes[archive_path],
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                        "format": description.image_format,
                        "width": description.width,
                        "height": description.height,
                    }
                    with lock:
                        records[logical_id] = record
                        stored_bytes = sum(int(value["bytes"]) for value in records.values())
                        if stored_bytes > broad.max_download_gb * GIB:
                            raise PreparationError(
                                "Broad WildFake selected payload exceeded max_download_gb."
                            )
                        if len(records) % broad.checkpoint_every == 0:
                            _wildfake_checkpoint(
                                config,
                                records,
                                expected_ids,
                                archive_hashes,
                                fingerprint,
                                False,
                                "periodic archive checkpoint",
                            )
            return
        except PreparationError:
            raise
        except Exception as exc:
            attempt += 1
            if attempt > broad.network_max_retries:
                raise PreparationError(
                    f"WildFake archive extraction failed after {attempt} attempts: {archive_path}"
                ) from exc
            delay = min(60.0, broad.network_retry_backoff * 2 ** (attempt - 1))
            LOGGER.warning(
                "Archive %s failed with %s; refreshing its signed URL in %.1f seconds.",
                archive_path,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
        finally:
            session.close()


def prepare_wildfake(config: AppConfig) -> dict[str, Any]:
    """Prepare the complementary hierarchy-uniform WildFake test sample."""
    broad = config.wildfake_evaluation
    output_root = Path(broad.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    files = {
        item.path: item
        for item in api.list_repo_files(broad.repo_id, "dataset", revision=broad.revision)
    }
    gan_archive = files.get("Images/GAN_based.zip")
    if gan_archive is None or not gan_archive.sha256:
        raise PreparationError("WildFake GAN archive identity metadata is incomplete.")
    detected_corrupt_paths = _detect_corrupt_gan_members(
        config, str(gan_archive.sha256)
    )
    metadata_path = Path(
        api.download_file(
            broad.repo_id,
            "dataset",
            broad.metadata_file,
            revision=broad.revision,
            local_dir=Path(broad.metadata_dir),
        )
    )
    metadata_digest = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if metadata_digest != broad.metadata_sha256:
        raise PreparationError("WildFake broad-test metadata identity changed.")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Generator", "Architecture", "Weight", "Category", "IsAdvanced", "IsFake", "Image_path"
        }
        if not required <= set(reader.fieldnames or []):
            raise PreparationError("WildFake broad-test metadata is missing required columns.")
        selected = select_wildfake_rows(reader, config, detected_corrupt_paths)
    by_archive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_archive[item["archive_path"]].append(item)
    missing_archives = set(by_archive) - set(files)
    if missing_archives:
        raise PreparationError(f"WildFake archives are missing: {sorted(missing_archives)}")
    archive_hashes = {path: str(files[path].sha256) for path in by_archive}
    if any(not digest or digest == "None" for digest in archive_hashes.values()):
        raise PreparationError("WildFake archive identity metadata is incomplete.")
    expected_ids = {_wildfake_id(config, item["source_path"]) for item in selected}
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "seed": config.project.seed,
                "metadata": broad.metadata_sha256,
                "expected_ids": sorted(expected_ids),
                "archives": archive_hashes,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    state_path = Path(broad.state_path)
    records = _load_records(Path(broad.manifest_path), output_root)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("sampling_config_sha256") != fingerprint:
            incompatible_records = set(records) - expected_ids
            if bool(state.get("complete")) or incompatible_records:
                raise PreparationError(
                    "Existing broad WildFake state does not match the active sample configuration."
                )
            LOGGER.warning(
                "Migrating an incomplete broad WildFake checkpoint after safe source exclusion; "
                "%d validated images will be reused.",
                len(records),
            )
        if bool(state.get("complete")):
            if set(records) != expected_ids:
                raise PreparationError(
                    "Completed broad WildFake state has missing manifest files."
                )
            return json.loads(Path(broad.audit_path).read_text(encoding="utf-8"))
    if set(records) - expected_ids:
        raise PreparationError("Broad WildFake manifest contains records outside the sample.")
    lock = threading.Lock()
    try:
        with ThreadPoolExecutor(max_workers=broad.download_workers) as executor:
            futures = [
                executor.submit(
                    _extract_wildfake_archive,
                    config,
                    archive,
                    items,
                    records,
                    expected_ids,
                    archive_hashes,
                    fingerprint,
                    lock,
                )
                for archive, items in sorted(by_archive.items())
            ]
            for future in as_completed(futures):
                future.result()
    except BaseException as exc:
        with lock:
            _wildfake_checkpoint(
                config,
                records,
                expected_ids,
                archive_hashes,
                fingerprint,
                False,
                f"interrupted by {type(exc).__name__}",
            )
        raise
    if set(records) != expected_ids:
        raise PreparationError("Broad WildFake preparation ended with missing selected images.")
    return _wildfake_checkpoint(
        config,
        records,
        expected_ids,
        archive_hashes,
        fingerprint,
        True,
        "all selected archives completed",
    )


def _sid_token(config: AppConfig) -> str | bool | None:
    from huggingface_hub import get_token

    mode = config.sid_evaluation.hf_auth
    if mode == "disabled":
        return False
    token = get_token()
    if token:
        LOGGER.info("Using the saved Hugging Face credential for SID-Set requests.")
        return token
    if mode == "required":
        raise PreparationError("SID-Set requires a saved Hugging Face token; run 'hf auth login'.")
    LOGGER.warning("No saved Hugging Face token was found; SID-Set requests are unauthenticated.")
    return None


def _sid_retry(config: AppConfig, operation: Any, description: str) -> Any:
    sid = config.sid_evaluation
    for attempt in range(sid.network_max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= sid.network_max_retries:
                raise
            delay = min(60.0, sid.network_retry_base_seconds * 2**attempt)
            LOGGER.warning(
                "%s failed with %s; retrying in %.1f seconds (%d/%d).",
                description,
                type(exc).__name__,
                delay,
                attempt + 1,
                sid.network_max_retries,
            )
            time.sleep(delay)
    raise AssertionError("SID retry loop terminated unexpectedly.")


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Mapping) and value.get("bytes") is not None:
        return bytes(value["bytes"])
    raise PreparationError("SID-Set image column did not contain embedded bytes.")


def _sid_shard_quotas(total: int, shards: list[str]) -> dict[str, int]:
    quotas = _even_quotas(total, shards)
    return {name: quotas[name] for name in shards}


def _select_sid_shard(
    path: Path, shard_name: str, real_quota: int, fake_quota: int, seed: int
) -> dict[int, set[str]]:
    import pyarrow.parquet as parquet

    candidates: dict[int, list[str]] = {0: [], 1: []}
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=2048, columns=["img_id", "label"]
    ):
        for row in batch.to_pylist():
            label = int(row.get("label", -1))
            if label in candidates:
                candidates[label].append(str(row.get("img_id") or ""))
    result: dict[int, set[str]] = {}
    for label, quota in ((0, real_quota), (1, fake_quota)):
        ranked = sorted(
            (item for item in candidates[label] if item),
            key=lambda item: _stable_rank(seed, f"{shard_name}\0{item}"),
        )
        if len(ranked) < quota:
            raise PreparationError(
                f"SID shard {shard_name} has {len(ranked)} label-{label} rows, below quota {quota}."
            )
        result[label] = set(ranked[:quota])
    return result


def _extract_sid_shard(
    path: Path,
    shard_name: str,
    selected: Mapping[int, set[str]],
    config: AppConfig,
    records: dict[str, dict[str, Any]],
) -> None:
    import pyarrow.parquet as parquet

    sid = config.sid_evaluation
    output_root = Path(sid.output_dir)
    wanted = selected[0] | selected[1]
    found: set[str] = set()
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=16, columns=["img_id", "image", "width", "height", "label"]
    ):
        for row in batch.to_pylist():
            image_id = str(row.get("img_id") or "")
            if image_id not in wanted or image_id in records:
                continue
            label = int(row["label"])
            content = _image_bytes(row["image"])
            description = validate_and_describe_image(content, config)
            logical_id = hashlib.sha256(
                f"{sid.repo_id}\0{sid.revision}\0{image_id}".encode("utf-8")
            ).hexdigest()
            extension = "." + description.image_format if description.image_format else ".img"
            relative = Path("images") / ("fake" if label else "real") / f"{logical_id}{extension}"
            destination = output_root / relative
            if not destination.is_file():
                _atomic_write_bytes(destination, content)
            records[image_id] = {
                "id": logical_id,
                "source_id": image_id,
                "path": str(relative),
                "label": label,
                "split": "external_test",
                "source_name": "full_synthetic" if label else "OpenImages V7 real",
                "source_repo": sid.repo_id,
                "source_revision": sid.revision,
                "source_split": sid.split,
                "source_shard": shard_name,
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "format": description.image_format,
                "width": description.width,
                "height": description.height,
                "declared_width": int(row.get("width") or 0),
                "declared_height": int(row.get("height") or 0),
            }
            found.add(image_id)
    missing = wanted - found - set(records)
    if missing:
        raise PreparationError(f"SID shard extraction missed {len(missing)} selected rows.")


def _sid_checkpoint(
    config: AppConfig,
    records: Mapping[str, Mapping[str, Any]],
    completed: set[str],
    shards: list[str],
    fingerprint: str,
    complete: bool,
    reason: str,
) -> dict[str, Any]:
    sid = config.sid_evaluation
    ordered = sorted(records.values(), key=lambda item: (item["label"], item["id"]))
    atomic_write_text(
        Path(sid.manifest_path),
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered),
    )
    counts = {
        "real": sum(int(item["label"]) == 0 for item in ordered),
        "fake": sum(int(item["label"]) == 1 for item in ordered),
    }
    state = {
        "complete": complete,
        "sampling_config_sha256": fingerprint,
        "completed_shards": sorted(completed),
        "source_shards": shards,
        "stop_reason": reason,
    }
    atomic_write_text(Path(sid.state_path), json.dumps(state, indent=2, sort_keys=True) + "\n")
    shard_counts = Counter(str(item["source_shard"]) for item in ordered)
    audit = {
        "complete": complete,
        "stop_reason": reason,
        "repo_id": sid.repo_id,
        "revision": sid.revision,
        "counts": counts,
        "expected_counts": {"real": sid.target_real, "fake": sid.target_fake},
        "selected_bytes": sum(int(item["bytes"]) for item in ordered),
        "sampling": {
            "source_split": sid.split,
            "included_labels": {"0": "real", "1": "full_synthetic"},
            "excluded_labels": {"2": "tampered"},
            "method": "equal per class and near-equal per source shard, stable hash within shard",
            "shards": dict(sorted(shard_counts.items())),
        },
    }
    atomic_write_text(Path(sid.audit_path), json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def prepare_sid(config: AppConfig) -> dict[str, Any]:
    """Scan the pinned SID validation split and retain only a balanced 4k sample."""
    from huggingface_hub import HfApi, hf_hub_download

    sid = config.sid_evaluation
    token = _sid_token(config)
    try:
        info = _sid_retry(
            config,
            lambda: HfApi(token=token).dataset_info(
                sid.repo_id, revision=sid.revision, files_metadata=True
            ),
            "SID-Set metadata request",
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            raise PreparationError(
                "Hugging Face denied SID-Set access. Run 'hf auth login' and verify access."
            ) from exc
        raise
    if str(info.sha) != sid.revision:
        raise PreparationError(
            f"SID-Set revision resolved to {info.sha}, expected {sid.revision}."
        )
    siblings = {
        item.rfilename: item
        for item in info.siblings
        if item.rfilename.startswith(sid.shard_prefix) and item.rfilename.endswith(".parquet")
    }
    shards = sorted(siblings)
    if not shards:
        raise PreparationError("The pinned SID-Set revision has no validation Parquet shards.")
    total_download = sum(int(getattr(siblings[name], "size", 0) or 0) for name in shards)
    if total_download > sid.max_download_gb * GIB:
        raise PreparationError(
            f"SID validation shards require {total_download / GIB:.2f} GiB, above max_download_gb."
        )
    largest = sorted(
        (int(getattr(siblings[name], "size", 0) or 0) for name in shards), reverse=True
    )[: sid.download_workers]
    if sum(largest) > sid.max_shard_cache_gb * GIB:
        raise PreparationError("SID prefetch queue exceeds max_shard_cache_gb.")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "repo": sid.repo_id,
                "revision": sid.revision,
                "seed": config.project.seed,
                "targets": [sid.target_real, sid.target_fake],
                "shards": [
                    [name, int(getattr(siblings[name], "size", 0) or 0)] for name in shards
                ],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    output_root = Path(sid.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(sid.manifest_path), output_root, key_field="source_id")
    completed: set[str] = set()
    state_path = Path(sid.state_path)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("sampling_config_sha256") != fingerprint:
            raise PreparationError("Existing SID state does not match the active configuration.")
        completed = set(state.get("completed_shards", []))
        if bool(state.get("complete")):
            complete_counts = Counter(int(item["label"]) for item in records.values())
            if complete_counts != Counter({0: sid.target_real, 1: sid.target_fake}):
                raise PreparationError("Completed SID state has missing manifest files.")
            return json.loads(Path(sid.audit_path).read_text(encoding="utf-8"))
    real_quotas = _sid_shard_quotas(sid.target_real, shards)
    fake_quotas = _sid_shard_quotas(sid.target_fake, shards)
    pending = [name for name in shards if name not in completed]
    cache_root = Path(sid.shard_cache_dir) / "payload"
    cache_root.mkdir(parents=True, exist_ok=True)

    def download(name: str) -> Path:
        return Path(
            _sid_retry(
                config,
                lambda: hf_hub_download(
                    repo_id=sid.repo_id,
                    filename=name,
                    repo_type="dataset",
                    revision=sid.revision,
                    token=token,
                    local_dir=cache_root,
                ),
                f"Download of {name}",
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=sid.download_workers) as executor:
            futures: dict[str, Future[Path]] = {}
            next_index = 0

            def fill_queue() -> None:
                nonlocal next_index
                while next_index < len(pending) and len(futures) < sid.download_workers:
                    name = pending[next_index]
                    futures[name] = executor.submit(download, name)
                    next_index += 1

            fill_queue()
            since_checkpoint = 0
            for position, name in enumerate(pending, start=1):
                path = futures.pop(name).result()
                fill_queue()
                LOGGER.info("Processing SID shard %d/%d: %s", position, len(pending), name)
                selected = _select_sid_shard(
                    path,
                    name,
                    real_quotas[name],
                    fake_quotas[name],
                    config.project.seed,
                )
                _extract_sid_shard(path, name, selected, config, records)
                completed.add(name)
                try:
                    path.resolve().relative_to(cache_root.resolve())
                    path.unlink(missing_ok=True)
                except ValueError as exc:
                    raise PreparationError("Refusing to remove a SID shard outside its cache.") from exc
                since_checkpoint += 1
                if since_checkpoint >= sid.checkpoint_every_shards:
                    _sid_checkpoint(
                        config,
                        records,
                        completed,
                        shards,
                        fingerprint,
                        False,
                        "periodic shard checkpoint",
                    )
                    since_checkpoint = 0
    except BaseException as exc:
        _sid_checkpoint(
            config,
            records,
            completed,
            shards,
            fingerprint,
            False,
            f"interrupted by {type(exc).__name__}",
        )
        raise
    counts = Counter(int(item["label"]) for item in records.values())
    if counts != Counter({0: sid.target_real, 1: sid.target_fake}):
        raise PreparationError(f"SID preparation count mismatch: {dict(counts)}")
    return _sid_checkpoint(
        config,
        records,
        completed,
        shards,
        fingerprint,
        True,
        "all validation shards processed",
    )


def _main(kind: str, description: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(description)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    try:
        audit = prepare_wildfake(config) if kind == "wildfake" else prepare_sid(config)
    except PreparationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    LOGGER.info(
        "External dataset preparation completed: real=%d fake=%d.",
        audit["counts"]["real"],
        audit["counts"]["fake"],
    )


def main_wildfake() -> None:
    _main("wildfake", "Prepare the broad stratified WildFake evaluation sample.")


def main_sid() -> None:
    _main("sid", "Prepare the SID-Set real/full-synthetic evaluation sample.")
