"""Offline EXIF and C2PA provenance inspection for image files.

This module deliberately reports provenance evidence rather than trying to turn
metadata into a universal real/fake classifier. C2PA assertions are useful only
when their signatures validate; EXIF is mutable and is therefore always a hint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, UnidentifiedImageError

from .config import AppConfig, ProvenanceConfig, config_argument_parser, load_config
from .watermark import inspect_watermark


IMAGE_SUFFIXES = frozenset({".avif", ".heic", ".heics", ".heif", ".heifs", ".hif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
HEIF_SUFFIXES = frozenset({".heic", ".heics", ".heif", ".heifs", ".hif"})
AVIF_SUFFIXES = frozenset({".avif"})
EXIF_FIELDS = frozenset({"DateTime", "DateTimeDigitized", "DateTimeOriginal", "HostComputer", "ImageDescription", "Make", "Model", "ProcessingSoftware", "Software"})
EXIF_DETAIL_FIELDS = frozenset({"DateTime", "DateTimeDigitized", "DateTimeOriginal", "HostComputer", "Make", "Model", "Software", "ProcessingSoftware"})
AI_MARKERS = (
    "adobe firefly",
    "comfyui",
    "dall-e",
    "dalle",
    "flux",
    "gemini",
    "google imagen",
    "ideogram",
    "leonardo",
    "midjourney",
    "openai",
    "stable diffusion",
)
TRAINED_MEDIA_MARKERS = ("trainedalgorithmicmedia", "trained_algorithmic_media")
CAMERA_CAPTURE_MARKERS = ("digitalcapture", "digital_capture")
UNTRUSTED_CREDENTIAL_CODE = "signingCredential.untrusted"
TRUSTED_CREDENTIAL_CODE = "signingCredential.trusted"


def _compact_value(value: Any, limit: int = 512) -> str:
    """Render metadata without making the report unbounded."""
    if isinstance(value, bytes):
        rendered = value.decode("utf-8", errors="replace")
    else:
        rendered = str(value)
    return rendered[:limit]


def _marker_hits(values: Iterable[str], markers: Iterable[str] = AI_MARKERS) -> list[str]:
    haystack = "\n".join(values).casefold()
    return [marker for marker in markers if marker in haystack]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _register_heif_decoder(path: Path) -> tuple[bool, str | None]:
    """Register pillow-heif before Pillow opens HEIC/AVIF files."""
    suffix = path.suffix.casefold()
    if suffix not in HEIF_SUFFIXES and suffix not in AVIF_SUFFIXES:
        return True, None
    try:
        import pillow_heif
    except ImportError:
        return False, "Install HEIC support with: uv sync --extra provenance"
    try:
        if suffix in HEIF_SUFFIXES:
            pillow_heif.register_heif_opener(thumbnails=False)
        else:
            pillow_heif.register_avif_opener(thumbnails=False)
    except (AttributeError, OSError, RuntimeError, ValueError) as error:
        return False, f"HEIF decoder registration failed: {type(error).__name__}"
    return True, None


def inspect_exif(path: Path, max_image_pixels: int) -> dict[str, Any]:
    """Read EXIF and Pillow-level image metadata without altering the asset."""
    decoder_ready, decoder_error = _register_heif_decoder(path)
    if not decoder_ready:
        return {
            "available": False,
            "status": "heif_decoder_missing" if decoder_error and decoder_error.startswith("Install") else "heif_decoder_error",
            "error": decoder_error,
        }
    try:
        with Image.open(path) as image:
            pixel_count = image.width * image.height
            if pixel_count > max_image_pixels:
                return {
                    "available": False,
                    "status": "image_too_large",
                    "pixel_count": pixel_count,
                    "max_image_pixels": max_image_pixels,
                }
            exif = image.getexif()
            exif_items = dict(exif.items())
            try:
                exif_items.update(exif.get_ifd(ExifTags.IFD.Exif))
            except (AttributeError, KeyError, TypeError, ValueError):
                # Some formats do not expose a separate EXIF sub-IFD.
                pass
            fields = {
                ExifTags.TAGS.get(tag, str(tag)): _compact_value(value)
                for tag, value in exif_items.items()
                if ExifTags.TAGS.get(tag, str(tag)) in EXIF_FIELDS
            }
            for key in ("Software", "Comment", "Description", "XML:com.adobe.xmp"):
                value = image.info.get(key)
                if value is not None:
                    fields.setdefault(key, _compact_value(value))
            values = list(fields.values())
            return {
                "available": True,
                "status": "ok",
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "pixel_count": pixel_count,
                "has_exif": bool(exif),
                "fields": fields,
                "camera": {
                    "make": fields.get("Make"),
                    "model": fields.get("Model"),
                    "capture_time": fields.get("DateTimeOriginal") or fields.get("DateTime"),
                },
                "software": fields.get("Software") or fields.get("ProcessingSoftware"),
                "ai_software_markers": _marker_hits(values),
                "warning": "EXIF can be removed or edited and is not proof of image origin.",
            }
    except (OSError, UnidentifiedImageError) as error:
        return {"available": False, "status": "unreadable", "error": type(error).__name__}


def _plain_value(value: Any) -> Any:
    """Keep C2PA SDK results JSON-safe and bounded."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_plain_value(item) for item in value[:100]]
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in list(value.items())[:100]}
    return _compact_value(value)


def _all_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _actions(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    """Extract standard C2PA actions while tolerating older assertion layouts."""
    result: list[dict[str, str]] = []
    assertions = manifest.get("assertions", [])
    if not isinstance(assertions, list):
        return result
    for assertion in assertions:
        if not isinstance(assertion, Mapping) or not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        data = assertion.get("data", {})
        actions = data.get("actions", []) if isinstance(data, Mapping) else []
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            result.append(
                {
                    "action": _compact_value(action.get("action", "")),
                    "digital_source_type": _compact_value(action.get("digitalSourceType", "")),
                    "software_agent": _compact_value(action.get("softwareAgent", "")),
                }
            )
    return result


def _normalized_validation_state(state: Any) -> str:
    """Normalize SDK enum and string representations of C2PA validation state."""
    return str(state).rsplit(".", maxsplit=1)[-1].casefold()


def _status_codes(value: Any) -> list[str]:
    """Collect validation codes from flat and C2PA 2.x structured results."""
    codes: list[str] = []
    if isinstance(value, Mapping):
        code = value.get("code")
        if isinstance(code, str):
            codes.append(code)
        for item in value.values():
            codes.extend(_status_codes(item))
    elif isinstance(value, list):
        for item in value:
            codes.extend(_status_codes(item))
    return codes


def _failure_codes(value: Any, *, flat_statuses_are_failures: bool = False) -> list[str]:
    """Collect only failure codes from structured or legacy validation output."""
    codes: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in {"failure", "validation_status"}:
                codes.extend(_status_codes(item))
            else:
                codes.extend(_failure_codes(item))
    elif isinstance(value, list):
        if flat_statuses_are_failures:
            codes.extend(_status_codes(value))
        else:
            for item in value:
                codes.extend(_failure_codes(item))
    return codes


def _integrity_valid(
    state: Any,
    results: Any,
    *,
    results_were_reported: bool,
    flat_statuses_are_failures: bool = False,
) -> bool | None:
    """Map C2PA Invalid/Valid/Trusted semantics to cryptographic integrity."""
    normalized = _normalized_validation_state(state)
    if normalized in {"valid", "trusted"}:
        return True
    if normalized == "invalid":
        return False
    if not results_were_reported:
        return None
    tolerated_prefixes = ("cawg.x509.",)
    hard_failures = [
        code
        for code in _failure_codes(
            results,
            flat_statuses_are_failures=flat_statuses_are_failures,
        )
        if code != UNTRUSTED_CREDENTIAL_CODE
        and not code.startswith(tolerated_prefixes)
    ]
    return not hard_failures


def _credential_trusted(state: Any, results: Any, *, trust_checked: bool) -> bool | None:
    """Report signer trust separately from manifest cryptographic integrity."""
    normalized = _normalized_validation_state(state)
    codes = _status_codes(results)
    if normalized == "trusted" or TRUSTED_CREDENTIAL_CODE in codes:
        return True
    if UNTRUSTED_CREDENTIAL_CODE in codes:
        return False
    if trust_checked and normalized == "valid":
        return False
    return None


def _parse_json_output(output: str) -> Mapping[str, Any] | None:
    """Parse c2patool JSON, tolerating a diagnostic line before the object."""
    text = output.strip()
    if not text:
        return None
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _report_from_store(
    store: Mapping[str, Any],
    *,
    source: str,
    embedded: bool,
    validation_state: Any = None,
    validation_results: Any = None,
    signature_valid: bool | None = None,
    credential_trusted: bool | None = None,
    trust_checked: bool = False,
    tool_exit_code: int | None = None,
) -> dict[str, Any]:
    """Normalize c2pa-python and c2patool manifest JSON to one report shape."""
    manifests = store.get("manifests", {})
    manifests = manifests if isinstance(manifests, Mapping) else {}
    active_label = store.get("active_manifest")
    active = manifests.get(active_label, {}) if active_label is not None else {}
    active = active if isinstance(active, Mapping) else {}
    actions = _actions(active)
    action_strings = list(_all_strings(actions))
    manifest_strings = list(_all_strings({"claim_generator": active.get("claim_generator", ""), "actions": actions}))
    trained_media = _marker_hits(action_strings, TRAINED_MEDIA_MARKERS)
    camera_capture = _marker_hits(action_strings, CAMERA_CAPTURE_MARKERS)
    ai_tool_markers = _marker_hits(manifest_strings)
    report: dict[str, Any] = {
        "available": True,
        "status": "ok",
        "manifest_present": bool(manifests),
        "embedded": bool(embedded),
        "source": source,
        "active_manifest": active_label,
        "claim_generator": _compact_value(active.get("claim_generator", "")),
        "actions": actions,
        "validation_state": _plain_value(validation_state),
        "validation_results": _plain_value(validation_results),
        "signature_valid": signature_valid,
        "integrity_valid": signature_valid,
        "credential_trusted": credential_trusted,
        "trust_checked": trust_checked,
        "trained_algorithmic_media_markers": trained_media,
        "camera_capture_markers": camera_capture,
        "ai_tool_markers": ai_tool_markers,
    }
    if tool_exit_code is not None:
        report["tool_exit_code"] = tool_exit_code
    # c2patool calls validation failures `validation_status`; preserve the
    # complete value because individual status codes are useful for triage.
    if "validation_status" in store:
        report["validation_status"] = _plain_value(store.get("validation_status"))
    return report


def _c2pa_tool_path(settings: ProvenanceConfig) -> str | None:
    configured = (settings.c2pa_tool_path or "").strip()
    if configured:
        return configured
    return shutil.which("c2patool")


def _inspect_c2pa_tool(path: Path, settings: ProvenanceConfig) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Run the official Rust c2patool CLI and parse its default JSON output."""
    tool = _c2pa_tool_path(settings)
    if not tool:
        return None, None
    try:
        completed = subprocess.run(
            [tool, str(path)],
            capture_output=True,
            text=True,
            timeout=settings.c2pa_tool_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout", "tool": tool}
    except (OSError, ValueError) as error:
        return None, {"status": "execution_error", "tool": tool, "error": type(error).__name__}
    store = _parse_json_output(completed.stdout)
    if completed.returncode != 0 or store is None:
        failure: dict[str, Any] = {"status": "no_manifest", "tool": tool, "exit_code": completed.returncode}
        if completed.stderr:
            failure["stderr"] = _compact_value(completed.stderr)
        return None, failure
    validation_status = store.get("validation_status")
    results_were_reported = "validation_status" in store
    signature_valid = _integrity_valid(
        None,
        validation_status,
        results_were_reported=results_were_reported,
        flat_statuses_are_failures=True,
    )
    credential_trusted = _credential_trusted(
        None,
        validation_status,
        trust_checked=False,
    )
    return (
        _report_from_store(
            store,
            source="c2patool",
            embedded=True,
            validation_state="Valid" if signature_valid else "Invalid" if signature_valid is False else "Unknown",
            validation_results=validation_status or [],
            signature_valid=signature_valid,
            credential_trusted=credential_trusted,
            trust_checked=False,
            tool_exit_code=completed.returncode,
        ),
        None,
    )


def _inspect_c2pa_sdk(path: Path, settings: ProvenanceConfig) -> dict[str, Any]:
    """Read C2PA through c2pa-python when c2patool is unavailable."""
    try:
        from c2pa import Context, Reader
    except ImportError:
        return {
            "available": False,
            "status": "dependency_missing",
            "install": "Install the provenance extra: uv sync --extra provenance",
        }

    context_options = {
        "verify": {
            "remote_manifest_fetch": settings.c2pa_remote_manifest_fetch,
            "ocsp_fetch": settings.c2pa_ocsp_fetch,
            "verify_trust": settings.c2pa_verify_trust,
            "verify_timestamp_trust": settings.c2pa_verify_trust,
        }
    }
    try:
        with Context.from_dict(context_options) as context:
            with Reader(str(path), context=context) as reader:
                store = json.loads(reader.json())
                state = reader.get_validation_state()
                results = reader.get_validation_results()
                embedded = reader.is_embedded()
        if not isinstance(store, Mapping):
            raise ValueError("C2PA SDK returned a non-object JSON value")
        return _report_from_store(
            store,
            source="c2pa-python",
            embedded=bool(embedded),
            validation_state=state,
            validation_results=results,
            signature_valid=_integrity_valid(state, results, results_were_reported=True),
            credential_trusted=_credential_trusted(
                state,
                results,
                trust_checked=settings.c2pa_verify_trust,
            ),
            trust_checked=settings.c2pa_verify_trust,
        )
    except Exception as error:  # SDK errors are data-dependent and must become a report, not a crash.
        return {
            "available": True,
            "status": "not_found_or_unreadable",
            "manifest_present": False,
            "error": type(error).__name__,
        }


def inspect_c2pa(path: Path, settings: ProvenanceConfig) -> dict[str, Any]:
    """Prefer official c2patool, then fall back to the Python SDK."""
    tool_report, tool_failure = _inspect_c2pa_tool(path, settings)
    if tool_report is not None:
        return tool_report
    sdk_report = _inspect_c2pa_sdk(path, settings)
    if tool_failure is not None:
        sdk_report["c2pa_tool_fallback"] = tool_failure
    return sdk_report


def classify_provenance(
    exif: Mapping[str, Any],
    c2pa: Mapping[str, Any],
    watermark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce an intentionally conservative, evidence-scoped conclusion."""
    c2pa_valid = bool(c2pa.get("integrity_valid", c2pa.get("signature_valid")))
    credential_trusted = c2pa.get("credential_trusted") is True
    if c2pa_valid and credential_trusted and c2pa.get("trained_algorithmic_media_markers"):
        return {
            "classification": "ai_declared_by_trusted_c2pa",
            "confidence": "high",
            "confidence_score": 0.99,
            "reason": "A trusted, cryptographically valid C2PA action declares trained algorithmic media.",
        }
    if c2pa_valid and c2pa.get("trained_algorithmic_media_markers"):
        reason = (
            "A cryptographically valid C2PA action declares trained algorithmic media, but the signing credential was not trusted."
            if c2pa.get("credential_trusted") is False
            else "A cryptographically valid C2PA action declares trained algorithmic media, but signer trust was not established."
        )
        return {
            "classification": "ai_declared_by_valid_c2pa",
            "confidence": "medium",
            "confidence_score": 0.85,
            "reason": reason,
        }
    if c2pa_valid and credential_trusted and c2pa.get("camera_capture_markers"):
        return {
            "classification": "camera_capture_declared_by_trusted_c2pa",
            "confidence": "high",
            "confidence_score": 0.02,
            "reason": "A trusted, cryptographically valid C2PA action declares digital capture; later edits may still exist.",
        }
    if c2pa_valid and c2pa.get("camera_capture_markers"):
        reason = (
            "A cryptographically valid C2PA action declares digital capture, but the signing credential was not trusted; later edits may still exist."
            if c2pa.get("credential_trusted") is False
            else "A cryptographically valid C2PA action declares digital capture, but signer trust was not established; later edits may still exist."
        )
        return {
            "classification": "camera_capture_declared_by_valid_c2pa",
            "confidence": "medium",
            "confidence_score": 0.10,
            "reason": reason,
        }
    if watermark and watermark.get("detected"):
        vendors = watermark.get("vendors", [])
        vendor_text = ", ".join(str(item) for item in vendors) or "known AI vendor"
        return {
            "classification": "visible_ai_watermark",
            "confidence": "medium",
            "confidence_score": 0.75,
            "reason": f"A visible watermark associated with {vendor_text} was detected; visible marks can be removed or added.",
        }
    if c2pa.get("manifest_present"):
        return {
            "classification": "c2pa_present_but_not_decisive",
            "confidence": "low",
            "confidence_score": 0.50,
            "reason": "The manifest is unverified or does not contain a decisive source-type assertion.",
        }
    if exif.get("ai_software_markers"):
        return {
            "classification": "ai_software_hint_from_exif",
            "confidence": "low",
            "confidence_score": 0.65,
            "reason": "EXIF names AI software but is editable and therefore not proof.",
        }
    return {
        "classification": "inconclusive",
        "confidence": "none",
        "confidence_score": 0.50,
        "reason": "No verified provenance assertion identifies AI generation or camera capture.",
    }


def _exif_evidence(exif: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize how much useful EXIF is present without treating it as proof."""
    structural_fields = [
        field
        for field in ("format", "width", "height", "pixel_count")
        if exif.get(field) is not None
    ]
    fields = exif.get("fields", {})
    fields = fields if isinstance(fields, Mapping) else {}
    populated_fields = sorted(
        str(name)
        for name, value in fields.items()
        if value is not None and str(value).strip()
    )
    detail_count = sum(name in EXIF_DETAIL_FIELDS for name in populated_fields)
    camera = exif.get("camera", {})
    camera = camera if isinstance(camera, Mapping) else {}
    camera_field_count = sum(
        bool(value is not None and str(value).strip())
        for value in (camera.get("make"), camera.get("model"), camera.get("capture_time"))
    )
    if detail_count >= 4 and camera_field_count >= 2:
        level = "detailed"
        label = "详细 EXIF"
        support = "strong_metadata_support"
        metadata_confidence = "high"
    elif detail_count >= 2 or camera_field_count >= 1:
        level = "partial"
        label = "部分 EXIF"
        support = "limited_metadata_support"
        metadata_confidence = "medium"
    else:
        level = "missing_or_minimal"
        label = "缺少或极少 EXIF"
        support = "no_meaningful_metadata_support"
        metadata_confidence = "none"
    return {
        "level": level,
        "label_zh": label,
        "populated_fields": populated_fields,
        "detail_field_count": detail_count,
        "camera_field_count": camera_field_count,
        "support": support,
        "metadata_confidence": metadata_confidence,
        "structural_fields": structural_fields,
        "interpretation": "详细 EXIF 只能提高元数据可信度，不能单独证明图片真实；EXIF 可以被编辑或移除。",
    }


def semantic_assessment(
    exif: Mapping[str, Any],
    c2pa: Mapping[str, Any],
    decision: Mapping[str, Any] | None = None,
    watermark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a clear real/fake/unknown summary from C2PA and EXIF evidence."""
    exif_evidence = _exif_evidence(exif)
    c2pa_valid = bool(c2pa.get("integrity_valid", c2pa.get("signature_valid")))
    credential_trusted = c2pa.get("credential_trusted") is True
    has_trained_media = bool(c2pa.get("trained_algorithmic_media_markers"))
    has_camera_capture = bool(c2pa.get("camera_capture_markers"))
    ai_exif_hint = bool(exif.get("ai_software_markers"))
    watermark = watermark if isinstance(watermark, Mapping) else {}
    base_classification = str((decision or classify_provenance(exif, c2pa, watermark)).get("classification", "inconclusive"))

    def result(
        *,
        assessment: str,
        synthetic_likelihood: str,
        capture_provenance: str,
        confidence: str,
        primary_evidence: str,
        verdict: str,
        verdict_label_zh: str,
        c2pa_conclusion: str,
        reason: str,
        limitations: list[str],
        ai_confidence: float,
    ) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "verdict_label_zh": verdict_label_zh,
            "assessment": assessment,
            "synthetic_likelihood": synthetic_likelihood,
            "capture_provenance": capture_provenance,
            "confidence": confidence,
            "ai_confidence": round(max(0.0, min(1.0, ai_confidence)), 4),
            "primary_evidence": primary_evidence,
            "basis": primary_evidence,
            "c2pa_conclusion": c2pa_conclusion,
            "exif_conclusion": exif_evidence,
            "decision_classification": base_classification,
            "reason": reason,
            "limitations": limitations,
            "watermark_evidence": watermark,
        }

    if c2pa_valid and has_trained_media:
        if credential_trusted:
            return result(
                assessment="likely_ai_generated",
                synthetic_likelihood="likely",
                capture_provenance="not_verified",
                confidence="high",
                primary_evidence="trusted_c2pa",
                verdict="fake",
                verdict_label_zh="虚假（C2PA 明确声明 AI 生成）",
                c2pa_conclusion="签名有效且签名凭证受信任，C2PA 声明内容属于训练算法生成媒体。",
                reason="以经过验证的 C2PA AI 生成声明为主要依据，判定为虚假/AI 生成内容。",
                limitations=["该结论表示内容来源为 AI 生成，不是法律意义上的欺诈认定。", "EXIF 只能作为辅助信息。"],
                ai_confidence=0.99,
            )
        return result(
            assessment="likely_ai_generated",
            synthetic_likelihood="likely",
            capture_provenance="not_verified",
            confidence="medium",
            primary_evidence="valid_c2pa",
            verdict="fake",
            verdict_label_zh="疑似虚假（C2PA 声明 AI 生成）",
            c2pa_conclusion="C2PA 完整性有效并声明训练算法生成媒体，但签名凭证未被信任或未建立信任。",
            reason="C2PA 对 AI 生成有明确声明，但由于签名信任不足，结论降为中等置信度。",
            limitations=["签名凭证未建立信任，因此不能达到最高证据等级。", "EXIF 可以被编辑或移除。"],
            ai_confidence=0.85,
        )

    if c2pa_valid and has_camera_capture:
        if credential_trusted:
            return result(
                assessment="likely_camera_captured",
                synthetic_likelihood="unlikely",
                capture_provenance="verified",
                confidence="high",
                primary_evidence="trusted_c2pa",
                verdict="real",
                verdict_label_zh="真实来源（C2PA 声明数字采集）",
                c2pa_conclusion="签名有效且签名凭证受信任，C2PA 声明内容来自数字采集。",
                reason="以经过验证的 C2PA 数字采集声明为主要依据，判定为真实来源；这不排除后续编辑。",
                limitations=["C2PA 数字采集声明不能排除签名之后的图像编辑。", "EXIF 可以被编辑或移除。"],
                ai_confidence=0.02,
            )
        return result(
            assessment="likely_camera_captured",
            synthetic_likelihood="unlikely",
            capture_provenance="declared",
            confidence="medium",
            primary_evidence="valid_c2pa",
            verdict="real",
            verdict_label_zh="疑似真实来源（C2PA 声明数字采集）",
            c2pa_conclusion="C2PA 完整性有效并声明数字采集，但签名凭证未被信任或未建立信任。",
            reason="C2PA 支持数字采集来源，但由于签名信任不足，结论降为中等置信度。",
            limitations=["签名凭证未建立信任。", "C2PA 数字采集声明不能排除后续编辑。"],
            ai_confidence=0.10,
        )

    if watermark.get("detected"):
        vendors = watermark.get("vendors", [])
        vendor_text = ", ".join(str(item) for item in vendors) or "已知 AI 厂商"
        return result(
            assessment="visible_ai_watermark",
            synthetic_likelihood="likely",
            capture_provenance="not_verified",
            confidence="medium",
            primary_evidence="visible_watermark",
            verdict="fake",
            verdict_label_zh="疑似虚假（检测到 AI 水印）",
            c2pa_conclusion=(
                "未发现可用于判断的 C2PA 声明。"
                if not c2pa.get("manifest_present")
                else "存在 C2PA，但未形成可用于判断真实来源的有效结论。"
            ),
            reason=f"在图像角落检测到与 {vendor_text} 相关的可见水印；这支持 AI 生成判断，但水印可能被后期添加或移除。",
            limitations=["可见水印不是加密签名，不能单独证明图片来源。", "裁剪、压缩或 OCR 失败可能导致漏检。"],
            ai_confidence=0.75,
        )

    if ai_exif_hint:
        return result(
            assessment="ai_software_hint_only",
            synthetic_likelihood="possible",
            capture_provenance="not_verified",
            confidence="low",
            primary_evidence="exif_hint",
            verdict="fake",
            verdict_label_zh="疑似虚假（EXIF 出现 AI 软件）",
            c2pa_conclusion="没有经过验证的 C2PA AI 生成结论。",
            reason="EXIF 软件字段出现 AI 软件标记，但 EXIF 可编辑，因此只能作为低置信度提示。",
            limitations=["EXIF 软件字段可以伪造、复制、删除或被后处理软件改写。", "没有有效 C2PA 时不能高置信度判定。"],
            ai_confidence=0.65,
        )

    return result(
        assessment="metadata_only" if exif_evidence["level"] == "detailed" else "unknown",
        synthetic_likelihood="unknown",
        capture_provenance="not_verified",
        confidence="low" if exif_evidence["level"] == "detailed" else "none",
        primary_evidence="exif_hint" if exif_evidence["level"] == "detailed" else "none",
        verdict="unknown",
        verdict_label_zh="无法确定",
        c2pa_conclusion=(
            "存在 C2PA，但未形成可用于判断真实或 AI 生成的有效结论。"
            if c2pa.get("manifest_present")
            else "未发现可用于判断真实或 AI 生成的 C2PA 声明。"
        ),
        reason="详细 EXIF 可以增强元数据可信度，但没有经过验证的 C2PA 来源声明时，不能据此确定图片真实。",
        limitations=["详细 EXIF 只能说明元数据较完整，不能单独证明图片真实。", "EXIF 可以被编辑或移除。"],
        ai_confidence=0.50,
    )


def inspect_image(path: str | Path, config: AppConfig) -> dict[str, Any]:
    """Inspect one image and return a serializable provenance report."""
    source = Path(path)
    exif = inspect_exif(source, config.data.max_image_pixels)
    c2pa = inspect_c2pa(source, config.provenance)
    watermark = inspect_watermark(source, config)
    decision = classify_provenance(exif, c2pa, watermark)
    return {
        "schema_version": 1,
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _hash_file(source),
        "exif": exif,
        "c2pa": c2pa,
        "watermark": watermark,
        "decision": decision,
        "authenticity_summary": semantic_assessment(exif, c2pa, decision, watermark),
    }


def iter_image_paths(source: str | Path, max_files: int) -> Iterator[Path]:
    """Yield supported images from one asset or a deterministic directory walk."""
    path = Path(source)
    candidates = [path] if path.is_file() else sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    selected = 0
    for candidate in candidates:
        if candidate.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        if selected >= max_files:
            raise ValueError(f"Input exceeds provenance.max_files={max_files}.")
        selected += 1
        yield candidate
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file() and path.suffix.casefold() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.suffix or '<no suffix>'}")


def inspect_path(source: str | Path, config: AppConfig) -> list[dict[str, Any]]:
    """Inspect a file or directory, retaining deterministic output ordering."""
    return [inspect_image(path, config) for path in iter_image_paths(source, config.provenance.max_files)]


def _write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _semantic_output_path(output_path: Path) -> Path:
    """Derive a separate, human-readable summary path from the detailed report."""
    if output_path.suffix.casefold() == ".json":
        return output_path.with_name(f"{output_path.stem}-semantic.json")
    return output_path.with_name(f"{output_path.name}-semantic.json")


def _semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    records = records if isinstance(records, list) else []
    semantic_records = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        assessment = record.get("authenticity_summary", record.get("semantic_assessment", {}))
        assessment = assessment if isinstance(assessment, Mapping) else {}
        semantic_records.append(
            {
                "path": record.get("path"),
                "sha256": record.get("sha256"),
                "verdict": assessment.get("verdict", "unknown"),
                "verdict_label_zh": assessment.get("verdict_label_zh", "无法确定"),
                "confidence": assessment.get("confidence", "none"),
                "ai_confidence": round(float(assessment.get("ai_confidence", 0.5)), 4),
                "basis": assessment.get("basis", "no_decisive_provenance"),
                "reason": assessment.get("reason", "没有决定性来源证据。"),
                "c2pa_conclusion": assessment.get("c2pa_conclusion", ""),
                "exif_conclusion": assessment.get("exif_conclusion", {}),
                "watermark_evidence": assessment.get("watermark_evidence", {}),
            }
        )
    counts = {verdict: sum(record["verdict"] == verdict for record in semantic_records) for verdict in ("real", "fake", "unknown")}
    scores = [record["ai_confidence"] for record in semantic_records]
    return {
        "schema_version": 1,
        "report_type": "semantic_provenance_assessment",
        "generated_at": payload.get("generated_at"),
        "input": payload.get("input"),
        "summary": {
            "total": len(semantic_records),
            "real_count": counts["real"],
            "fake_count": counts["fake"],
            "unknown_count": counts["unknown"],
            "ai_confidence_mean": round(sum(scores) / len(scores), 4) if scores else 0.5,
            "ai_confidence_min": min(scores) if scores else 0.5,
            "ai_confidence_max": max(scores) if scores else 0.5,
            "ai_confidence_ge_0_5_count": sum(score >= 0.5 for score in scores),
            "ai_confidence_ge_0_75_count": sum(score >= 0.75 for score in scores),
            "interpretation": "C2PA 是主要判断依据；EXIF 详细程度只能作为辅助证据，不能单独证明真实。",
        },
        "records": semantic_records,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = config_argument_parser("Inspect image provenance through EXIF and C2PA.")
    parser.add_argument("--input", required=True, help="Image file or directory to inspect.")
    parser.add_argument("--output", required=True, help="Destination JSON report path.")
    parser.add_argument(
        "--semantic-output",
        help="Optional semantic summary JSON path; defaults to <output>-semantic.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    records = inspect_path(args.input, config)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "records": records,
    }
    output_path = Path(args.output)
    _write_json(payload, output_path)
    _write_json(_semantic_payload(payload), Path(args.semantic_output) if args.semantic_output else _semantic_output_path(output_path))


if __name__ == "__main__":
    main()
