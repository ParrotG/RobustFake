"""Prepare the challenge-prescribed WildFake evaluation subset."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import requests
from modelscope_hub import HubApi
from PIL import Image
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.prepare import PreparationError, validate_and_describe_image

LOGGER = logging.getLogger(__name__)
GIB = 1024**3


def _atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace a file with complete bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _session(config: AppConfig) -> requests.Session:
    """Create a retrying HTTP session for CDN range requests."""
    retry = Retry(
        total=config.official_evaluation.network_max_retries,
        connect=config.official_evaluation.network_max_retries,
        read=config.official_evaluation.network_max_retries,
        backoff_factor=config.official_evaluation.network_retry_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _metadata_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate the columns used by the official subset definition."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"IsAdvanced", "IsFake", "Image_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise PreparationError(
                f"WildFake metadata is missing columns: {', '.join(sorted(missing))}"
            )
        return [dict(row) for row in reader]


def select_official_rows(
    dalle_rows: Iterable[Mapping[str, str]],
    coco_rows: Iterable[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Select DALL-E Advanced and COCO val2017 exactly as specified."""
    selected: list[dict[str, Any]] = []
    for row in dalle_rows:
        if row.get("IsAdvanced") == "1" and row.get("IsFake") == "1":
            source_path = str(row["Image_path"]).removeprefix("./Diffusion_based/")
            selected.append(
                {"label": 1, "source_name": "DALL-E Advanced", "member": source_path}
            )
    for row in coco_rows:
        source_path = str(row["Image_path"])
        if row.get("IsFake") == "0" and "/val2017/" in source_path:
            selected.append(
                {
                    "label": 0,
                    "source_name": "COCO val2017",
                    "member": source_path.removeprefix("./Real/"),
                }
            )
    for item in selected:
        path = PurePosixPath(item["member"])
        if path.is_absolute() or ".." in path.parts:
            raise PreparationError(f"Unsafe archive member in metadata: {item['member']}")
    return selected


def _download_url(config: AppConfig, archive_path: str) -> str:
    official = config.official_evaluation
    return (
        f"https://modelscope.cn/api/v1/datasets/{official.repo_id}/repo"
        f"?Revision={quote(official.revision, safe='')}"
        f"&FilePath={quote(archive_path, safe='')}"
    )


def _signed_url(config: AppConfig, archive_path: str, session: requests.Session) -> str:
    """Resolve a short-lived ModelScope CDN URL without downloading its payload."""
    response = session.get(
        _download_url(config, archive_path),
        allow_redirects=False,
        timeout=config.official_evaluation.request_timeout_seconds,
    )
    response.raise_for_status()
    location = response.headers.get("Location")
    if not location:
        raise PreparationError("ModelScope did not return a CDN redirect for the archive.")
    return location


def _record_id(config: AppConfig, member: str) -> str:
    official = config.official_evaluation
    identity = f"{official.repo_id}\0{official.revision}\0{member}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _load_existing(config: AppConfig) -> dict[str, dict[str, Any]]:
    """Load only committed records whose atomically written image still exists."""
    manifest_path = Path(config.official_evaluation.manifest_path)
    output_dir = Path(config.official_evaluation.output_dir)
    if not manifest_path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (output_dir / record["path"]).is_file():
            records[str(record["source_member"])] = record
    return records


def _checkpoint(
    config: AppConfig,
    records: Mapping[str, Mapping[str, Any]],
    *,
    complete: bool,
    archive_metadata: Mapping[str, Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Atomically commit an idempotent manifest and preparation audit."""
    official = config.official_evaluation
    ordered = sorted(records.values(), key=lambda item: (item["label"], item["id"]))
    manifest = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in ordered
    )
    _atomic_write(Path(official.manifest_path), manifest.encode("utf-8"))
    counts = {
        "real": sum(int(record["label"]) == 0 for record in ordered),
        "fake": sum(int(record["label"]) == 1 for record in ordered),
    }
    audit = {
        "complete": complete,
        "stop_reason": reason,
        "repo_id": official.repo_id,
        "revision": official.revision,
        "counts": counts,
        "expected_counts": {
            "real": official.expected_real_count,
            "fake": official.expected_fake_count,
        },
        "selected_bytes": sum(int(record["bytes"]) for record in ordered),
        "archives": archive_metadata,
    }
    _atomic_write(
        Path(official.audit_path),
        (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return audit


def _extract_archive(
    config: AppConfig,
    archive_path: str,
    expected_archive_sha256: str,
    selected: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    archive_metadata: dict[str, dict[str, Any]],
) -> None:
    """Range-read selected ZIP members with periodic resumable checkpoints."""
    official = config.official_evaluation
    output_dir = Path(official.output_dir)
    pending = [item for item in selected if item["member"] not in records]
    if not pending:
        archive_metadata[archive_path] = {
            "sha256": expected_archive_sha256,
            "selected_files": len(selected),
            "selected_uncompressed_bytes": sum(
                int(records[item["member"]]["bytes"]) for item in selected
            ),
        }
        return
    attempt = 0
    while pending:
        session = _session(config)
        try:
            signed_url = _signed_url(config, archive_path, session)
            with RemoteZip(
                signed_url,
                session=session,
                timeout=official.request_timeout_seconds,
            ) as archive:
                available = {info.filename: info for info in archive.infolist()}
                missing = [item["member"] for item in pending if item["member"] not in available]
                if missing:
                    raise PreparationError(
                        f"Official metadata references missing archive members; first: {missing[0]}"
                    )
                pending.sort(key=lambda item: available[item["member"]].header_offset)
                projected = sum(available[item["member"]].file_size for item in selected)
                existing_bytes = sum(int(record["bytes"]) for record in records.values())
                pending_bytes = sum(available[item["member"]].file_size for item in pending)
                if existing_bytes + pending_bytes > official.max_download_gb * GIB:
                    raise PreparationError(
                        "Selected official files exceed official_evaluation.max_download_gb."
                    )
                archive_metadata[archive_path] = {
                    "sha256": expected_archive_sha256,
                    "selected_files": len(selected),
                    "selected_uncompressed_bytes": projected,
                }
                for item in tqdm(pending, desc=f"Extract {Path(archive_path).name}"):
                    content = archive.read(item["member"])
                    logical_id = _record_id(config, item["member"])
                    extension = Path(item["member"]).suffix.lower() or ".img"
                    relative_path = Path("images") / ("fake" if item["label"] else "real") / (
                        logical_id + extension
                    )
                    decoded = validate_and_describe_image(content, config)
                    destination = output_dir / relative_path
                    if not destination.is_file():
                        _atomic_write(destination, content)
                    records[item["member"]] = {
                        "id": logical_id,
                        "path": str(relative_path),
                        "label": int(item["label"]),
                        "split": "official_validation",
                        "source_name": item["source_name"],
                        "source_member": item["member"],
                        "source_repo": official.repo_id,
                        "source_revision": official.revision,
                        "archive_path": archive_path,
                        "archive_sha256": expected_archive_sha256,
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                        "format": decoded.image_format,
                        "width": decoded.width,
                        "height": decoded.height,
                    }
                    if len(records) % official.checkpoint_every == 0:
                        _checkpoint(
                            config,
                            records,
                            complete=False,
                            archive_metadata=archive_metadata,
                            reason="periodic checkpoint",
                        )
            return
        except PreparationError:
            raise
        except Exception as exc:
            attempt += 1
            _checkpoint(
                config,
                records,
                complete=False,
                archive_metadata=archive_metadata,
                reason=f"interrupted by {type(exc).__name__}",
            )
            if attempt > official.network_max_retries:
                raise PreparationError(
                    f"Selective archive extraction failed after {attempt} attempts."
                ) from exc
            delay = min(60.0, official.network_retry_backoff * (2 ** (attempt - 1)))
            LOGGER.warning(
                "Archive range request failed with %s; refreshing the signed URL in %.1f seconds.",
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
            pending = [item for item in selected if item["member"] not in records]
        finally:
            session.close()


def prepare_official_evaluation(config: AppConfig) -> dict[str, Any]:
    """Prepare only the 13,841 images reserved by the challenge statement."""
    official = config.official_evaluation
    output_dir = Path(official.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(official.audit_path)
    existing_audit: dict[str, Any] = {}
    if audit_path.is_file():
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if bool(existing_audit.get("complete")):
            expected_counts = {
                "real": official.expected_real_count,
                "fake": official.expected_fake_count,
            }
            expected_archives = {
                official.dalle_archive_file: official.dalle_archive_sha256,
                official.coco_archive_file: official.coco_archive_sha256,
            }
            recorded_archives = {
                path: metadata.get("sha256")
                for path, metadata in existing_audit.get("archives", {}).items()
            }
            if (
                existing_audit.get("repo_id") != official.repo_id
                or existing_audit.get("revision") != official.revision
                or existing_audit.get("expected_counts") != expected_counts
                or recorded_archives != expected_archives
            ):
                raise PreparationError(
                    "Completed official evaluation data does not match the active configuration."
                )
            LOGGER.info("The official evaluation subset is already complete.")
            return existing_audit

    api = HubApi()
    metadata_dir = Path(official.metadata_dir)
    dalle_path = api.download_file(
        official.repo_id,
        "dataset",
        official.dalle_metadata_file,
        revision=official.revision,
        local_dir=metadata_dir,
    )
    coco_path = api.download_file(
        official.repo_id,
        "dataset",
        official.coco_metadata_file,
        revision=official.revision,
        local_dir=metadata_dir,
    )
    selected = select_official_rows(_metadata_rows(dalle_path), _metadata_rows(coco_path))
    fake_count = sum(item["label"] == 1 for item in selected)
    real_count = sum(item["label"] == 0 for item in selected)
    if (real_count, fake_count) != (
        official.expected_real_count,
        official.expected_fake_count,
    ):
        raise PreparationError(
            "Official subset metadata count mismatch: "
            f"real={real_count}, fake={fake_count}."
        )

    files = {
        file.path: file
        for file in api.list_repo_files(
            official.repo_id, "dataset", revision=official.revision
        )
    }
    archives = {
        official.dalle_archive_file: official.dalle_archive_sha256,
        official.coco_archive_file: official.coco_archive_sha256,
    }
    for path, expected_sha256 in archives.items():
        file_info = files.get(path)
        if file_info is None or file_info.sha256 != expected_sha256:
            raise PreparationError(f"WildFake archive identity changed: {path}")

    records = _load_existing(config)
    archive_metadata: dict[str, dict[str, Any]] = dict(existing_audit.get("archives", {}))
    try:
        for archive_path, expected_sha256 in archives.items():
            archive_rows = [
                item
                for item in selected
                if (item["label"] == 1) == (archive_path == official.dalle_archive_file)
            ]
            _extract_archive(
                config,
                archive_path,
                expected_sha256,
                archive_rows,
                records,
                archive_metadata,
            )
            stored_bytes = sum(int(record["bytes"]) for record in records.values())
            if stored_bytes > official.max_download_gb * GIB:
                raise PreparationError("Official evaluation exceeded its configured byte budget.")
    except BaseException as exc:
        _checkpoint(
            config,
            records,
            complete=False,
            archive_metadata=archive_metadata,
            reason=f"interrupted by {type(exc).__name__}",
        )
        raise

    complete = len(records) == official.expected_real_count + official.expected_fake_count
    audit = _checkpoint(
        config,
        records,
        complete=complete,
        archive_metadata=archive_metadata,
        reason="all official files extracted" if complete else "count mismatch",
    )
    if not complete:
        raise PreparationError("Official evaluation preparation did not reach its exact quota.")
    return audit


def main() -> None:
    """Run official evaluation subset preparation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Prepare the challenge-prescribed WildFake subset.")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    try:
        audit = prepare_official_evaluation(config)
    except PreparationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    LOGGER.info(
        "Official evaluation preparation completed: real=%d fake=%d.",
        audit["counts"]["real"],
        audit["counts"]["fake"],
    )


if __name__ == "__main__":
    main()
