import subprocess
from pathlib import Path

from PIL import Image

from aigc_recognizer.config import load_config
from aigc_recognizer.provenance import classify_provenance, semantic_assessment
from aigc_recognizer.watermark import detect_watermark_text, inspect_watermark


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_detect_watermark_text_recognizes_doubao_and_other_vendors() -> None:
    matches = detect_watermark_text("豆 包 · Seedream 生成 | Adobe Firefly")
    assert {match["vendor"] for match in matches} == {"doubao", "adobe-firefly"}


def test_generic_ai_watermark_is_reported() -> None:
    matches = detect_watermark_text("AI-generated image")
    assert matches[0]["vendor"] == "generic-ai"


def test_inspect_watermark_degrades_without_tesseract(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    config = load_config(DEFAULT_CONFIG)
    config.watermark.tesseract_path = "/missing/tesseract"
    result = inspect_watermark(image_path, config)
    assert result["detected"] is False
    assert result["status"] == "tesseract_not_installed"
    assert len(result["regions"]) == 4


def test_inspect_watermark_reads_corner_ocr(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "doubao.png"
    Image.new("RGB", (80, 80), "white").save(image_path)
    config = load_config(DEFAULT_CONFIG)
    tool_path = tmp_path / "tesseract"
    tool_path.touch()
    config.watermark.tesseract_path = str(tool_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, b"Doubao", b"")

    monkeypatch.setattr("aigc_recognizer.watermark.subprocess.run", fake_run)
    result = inspect_watermark(image_path, config)
    assert result["detected"] is True
    assert result["vendors"] == ["doubao"]


def test_inspect_watermark_uses_pixel_fallback_without_ocr(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "doubao.png"
    Image.new("RGB", (80, 80), "black").save(image_path)
    config = load_config(DEFAULT_CONFIG)
    config.watermark.tesseract_path = "/missing/tesseract"
    config.watermark.pixel_fallback_enabled = True
    monkeypatch.setattr("aigc_recognizer.watermark._looks_like_doubao_corner_mark", lambda *_args: True)
    result = inspect_watermark(image_path, config)
    assert result["detected"] is True
    assert result["vendors"] == ["doubao"]
    assert result["matches"][0]["source"] == "pixel:bottom_right"
    assert result["confidence"] == "low"


def test_pixel_fallback_is_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "bright-corner.png"
    Image.new("RGB", (80, 80), "white").save(image_path)
    config = load_config(DEFAULT_CONFIG)
    config.watermark.tesseract_path = "/missing/tesseract"
    monkeypatch.setattr("aigc_recognizer.watermark._looks_like_doubao_corner_mark", lambda *_args: True)
    result = inspect_watermark(image_path, config)
    assert result["detected"] is False
    assert result["matches"] == []


def test_watermark_is_medium_confidence_but_trusted_c2pa_wins() -> None:
    exif = {"ai_software_markers": []}
    watermark = {"detected": True, "vendors": ["doubao"]}
    c2pa = {"integrity_valid": False, "manifest_present": False}
    decision = classify_provenance(exif, c2pa, watermark)
    assert decision["classification"] == "visible_ai_watermark"
    assert semantic_assessment(exif, c2pa, decision, watermark)["confidence"] == "medium"

    trusted_camera = {
        "integrity_valid": True,
        "credential_trusted": True,
        "camera_capture_markers": ["digitalcapture"],
        "trained_algorithmic_media_markers": [],
        "manifest_present": True,
    }
    result = semantic_assessment(exif, trusted_camera, None, watermark)
    assert result["verdict"] == "real"
    assert result["basis"] == "trusted_c2pa"
