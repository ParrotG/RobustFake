from pathlib import Path

import pytest

from aigc_recognizer.config import load_config
from aigc_recognizer.hub import download_model_artifacts, resolve_inference_checkpoint


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_local_checkpoint_takes_priority_over_hub(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = load_config(DEFAULT_CONFIG)

    assert resolve_inference_checkpoint(config, checkpoint) == checkpoint


def test_explicit_missing_checkpoint_does_not_silently_fall_back(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_inference_checkpoint(config, tmp_path / "missing.pt")


def test_hub_download_resolves_checkpoint_and_calibration_together(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
        assert repo_id == "owner/RobustFake"
        assert revision == "commit"
        path = snapshot / filename
        path.write_bytes(b"artifact")
        return str(path)

    monkeypatch.setattr("aigc_recognizer.hub.hf_hub_download", fake_download)
    config = load_config(DEFAULT_CONFIG)
    artifacts = download_model_artifacts(
        config, repo_id="owner/RobustFake", revision="commit"
    )

    assert artifacts.checkpoint == snapshot / "best.pt"
    assert artifacts.calibration == snapshot / "calibration.json"


def test_explicit_hub_repository_overrides_an_existing_local_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_checkpoint = tmp_path / "local.pt"
    local_checkpoint.write_bytes(b"local")
    downloaded_checkpoint = tmp_path / "downloaded.pt"
    downloaded_checkpoint.write_bytes(b"downloaded")
    config = load_config(DEFAULT_CONFIG)
    config.evaluation.checkpoint_path = str(local_checkpoint)

    monkeypatch.setattr(
        "aigc_recognizer.hub.download_model_artifacts",
        lambda *args, **kwargs: type(
            "Artifacts",
            (),
            {"checkpoint": downloaded_checkpoint},
        )(),
    )

    assert resolve_inference_checkpoint(
        config, hf_repo_id="owner/RobustFake"
    ) == downloaded_checkpoint
