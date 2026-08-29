import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from aigc_recognizer.config import load_config
from aigc_recognizer.provenance import _exif_evidence, inspect_image, inspect_path, main, semantic_assessment


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def write_jpeg(path: Path, software: str | None = None) -> None:
    image = Image.new("RGB", (24, 18), (12, 34, 56))
    exif = Image.Exif()
    exif[271] = "Example Camera"
    exif[272] = "Example Model"
    if software:
        exif[305] = software
    image.save(path, exif=exif)


def test_exif_ai_software_is_reported_as_a_hint(tmp_path: Path) -> None:
    image_path = tmp_path / "imagen.jpg"
    write_jpeg(image_path, "Google Imagen")
    config = load_config(DEFAULT_CONFIG)

    result = inspect_image(image_path, config)

    assert result["exif"]["camera"]["make"] == "Example Camera"
    assert result["exif"]["ai_software_markers"] == ["google imagen"]
    assert result["decision"]["classification"] == "ai_software_hint_from_exif"
    assert result["decision"]["confidence"] == "low"
    assert result["authenticity_summary"]["verdict"] == "fake"
    assert result["authenticity_summary"]["confidence"] == "low"


def test_detailed_exif_without_c2pa_remains_unknown(tmp_path: Path) -> None:
    image_path = tmp_path / "camera.jpg"
    write_jpeg(image_path)

    result = inspect_image(image_path, load_config(DEFAULT_CONFIG))

    assert result["authenticity_summary"]["verdict"] == "unknown"
    assert result["authenticity_summary"]["exif_conclusion"]["level"] == "partial"
    assert "不能据此确定" in result["authenticity_summary"]["reason"]


def test_image_dimensions_are_not_detailed_exif() -> None:
    evidence = _exif_evidence(
        {
            "available": True,
            "status": "ok",
            "format": "JPEG",
            "width": 1920,
            "height": 1080,
            "pixel_count": 2073600,
            "has_exif": False,
            "fields": {},
            "camera": {"make": None, "model": None, "capture_time": None},
            "ai_software_markers": [],
        }
    )

    assert evidence["structural_fields"] == ["format", "width", "height", "pixel_count"]
    assert evidence["level"] == "missing_or_minimal"
    assert evidence["metadata_confidence"] == "none"


def test_c2pa_semantic_assessment_prioritizes_verified_source_type() -> None:
    exif = {"fields": {"Make": "Example Camera", "Model": "Example Model"}, "camera": {}, "ai_software_markers": []}
    c2pa = {
        "integrity_valid": True,
        "credential_trusted": True,
        "trained_algorithmic_media_markers": [],
        "camera_capture_markers": ["digitalcapture"],
        "manifest_present": True,
    }

    result = semantic_assessment(exif, c2pa)

    assert result["verdict"] == "real"
    assert result["confidence"] == "high"
    assert result["basis"] == "trusted_c2pa"


def test_heic_registers_pillow_heif_before_open(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "asset.heic"
    image_path.write_bytes(b"not a decoded image")
    calls = []
    fake_pillow_heif = SimpleNamespace(
        register_heif_opener=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "pillow_heif", fake_pillow_heif)

    result = inspect_image(image_path, load_config(DEFAULT_CONFIG))

    assert calls == [{"thumbnails": False}]
    assert result["exif"]["status"] == "unreadable"


def test_valid_untrusted_c2pa_trained_media_takes_precedence(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "asset.jpg"
    write_jpeg(image_path)
    # Force this test down the Python SDK fallback path even when c2patool is
    # installed on the developer machine.
    monkeypatch.setattr("aigc_recognizer.provenance.shutil.which", lambda _name: None)

    class Context:
        @classmethod
        def from_dict(cls, options):
            assert options["verify"]["remote_manifest_fetch"] is False
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Reader:
        def __init__(self, path, context):
            assert Path(path) == image_path
            assert isinstance(context, Context)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def json(self):
            return json.dumps(
                {
                    "active_manifest": "manifest-1",
                    "manifests": {
                        "manifest-1": {
                            "claim_generator": "Example Generator",
                            "assertions": [
                                {
                                    "label": "c2pa.actions.v2",
                                    "data": {
                                        "actions": [
                                            {
                                                "action": "c2pa.created",
                                                "digitalSourceType": "https://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    },
                }
            )

        def get_validation_state(self):
            return "Valid"

        def get_validation_results(self):
            return {
                "activeManifest": {
                    "success": [{"code": "claimSignature.validated"}],
                    "informational": [],
                    "failure": [{"code": "signingCredential.untrusted"}],
                }
            }

        def is_embedded(self):
            return True

    monkeypatch.setitem(sys.modules, "c2pa", SimpleNamespace(Context=Context, Reader=Reader))
    result = inspect_image(image_path, load_config(DEFAULT_CONFIG))

    assert result["c2pa"]["signature_valid"] is True
    assert result["c2pa"]["integrity_valid"] is True
    assert result["c2pa"]["credential_trusted"] is False
    assert result["c2pa"]["trained_algorithmic_media_markers"] == ["trainedalgorithmicmedia"]
    assert result["decision"]["classification"] == "ai_declared_by_valid_c2pa"
    assert result["decision"]["confidence"] == "medium"
    assert "not trusted" in result["decision"]["reason"]


def test_trusted_c2pa_trained_media_is_high_confidence(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "asset.jpg"
    write_jpeg(image_path)
    monkeypatch.setattr("aigc_recognizer.provenance.shutil.which", lambda _name: None)

    class Context:
        @classmethod
        def from_dict(cls, _options):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Reader:
        def __init__(self, _path, context):
            assert isinstance(context, Context)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def json(self):
            return json.dumps(
                {
                    "active_manifest": "manifest-1",
                    "manifests": {
                        "manifest-1": {
                            "claim_generator": "Trusted Generator",
                            "assertions": [
                                {
                                    "label": "c2pa.actions.v2",
                                    "data": {
                                        "actions": [
                                            {
                                                "action": "c2pa.created",
                                                "digitalSourceType": "trainedAlgorithmicMedia",
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    },
                }
            )

        def get_validation_state(self):
            return "Trusted"

        def get_validation_results(self):
            return {"activeManifest": {"success": [{"code": "signingCredential.trusted"}], "informational": [], "failure": []}}

        def is_embedded(self):
            return True

    monkeypatch.setitem(sys.modules, "c2pa", SimpleNamespace(Context=Context, Reader=Reader))
    result = inspect_image(image_path, load_config(DEFAULT_CONFIG))

    assert result["c2pa"]["integrity_valid"] is True
    assert result["c2pa"]["credential_trusted"] is True
    assert result["decision"]["classification"] == "ai_declared_by_trusted_c2pa"
    assert result["decision"]["confidence"] == "high"


def test_c2patool_json_is_used_when_available(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "asset.jpg"
    write_jpeg(image_path)
    manifest = {
        "active_manifest": "manifest-1",
        "validation_status": [],
        "manifests": {
            "manifest-1": {
                "claim_generator": "c2patool test",
                "assertions": [
                    {
                        "label": "c2pa.actions.v2",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": "trainedAlgorithmicMedia",
                                }
                            ]
                        },
                    }
                ],
            }
        },
    }

    monkeypatch.setattr("aigc_recognizer.provenance.shutil.which", lambda _name: "/usr/local/bin/c2patool")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return __import__("subprocess").CompletedProcess(command, 0, json.dumps(manifest), "")

    monkeypatch.setattr("aigc_recognizer.provenance.subprocess.run", fake_run)
    result = inspect_image(image_path, load_config(DEFAULT_CONFIG))

    assert result["c2pa"]["source"] == "c2patool"
    assert result["c2pa"]["signature_valid"] is True
    assert result["c2pa"]["credential_trusted"] is None
    assert result["decision"]["classification"] == "ai_declared_by_valid_c2pa"
    assert result["decision"]["confidence"] == "medium"
    assert "not established" in result["decision"]["reason"]
    assert calls[0][0] == ["/usr/local/bin/c2patool", str(image_path)]
    assert calls[0][1]["timeout"] == 30.0


def test_directory_scan_filters_extensions_and_is_sorted(tmp_path: Path) -> None:
    write_jpeg(tmp_path / "b.jpg")
    write_jpeg(tmp_path / "a.jpg")
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")
    config = load_config(DEFAULT_CONFIG)

    results = inspect_path(tmp_path, config)

    assert [Path(result["path"]).name for result in results] == ["a.jpg", "b.jpg"]


def test_cli_writes_a_json_report(tmp_path: Path) -> None:
    image_path = tmp_path / "asset.jpg"
    report_path = tmp_path / "report.json"
    write_jpeg(image_path)

    main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--input",
            str(image_path),
            "--output",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["input"] == str(image_path)
    assert report["records"][0]["path"] == str(image_path)
    semantic_path = tmp_path / "report-semantic.json"
    semantic_report = json.loads(semantic_path.read_text(encoding="utf-8"))
    assert semantic_report["report_type"] == "semantic_provenance_assessment"
    assert semantic_report["records"][0]["verdict"] == "unknown"
