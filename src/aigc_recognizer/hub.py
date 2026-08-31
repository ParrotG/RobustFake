"""Download the public RobustFake inference artifacts from Hugging Face Hub."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubModelArtifacts:
    """Local paths for one checkpoint-bound Hugging Face model package."""

    checkpoint: Path
    calibration: Path
    repo_id: str
    revision: str


def download_model_artifacts(
    config: AppConfig,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> HubModelArtifacts:
    """Download the checkpoint and matching calibration into one Hub snapshot."""
    selected_repo = repo_id or config.evaluation.hf_model_repo_id
    if not selected_repo:
        raise ValueError(
            "No Hugging Face model repository is configured; pass --hf-repo or set "
            "evaluation.hf_model_repo_id."
        )
    selected_revision = revision or config.evaluation.hf_model_revision
    checkpoint = Path(
        hf_hub_download(
            repo_id=selected_repo,
            filename=config.evaluation.hf_checkpoint_filename,
            revision=selected_revision,
        )
    )
    calibration = Path(
        hf_hub_download(
            repo_id=selected_repo,
            filename=config.evaluation.hf_calibration_filename,
            revision=selected_revision,
        )
    )
    if checkpoint.parent != calibration.parent:
        raise RuntimeError("Downloaded model artifacts do not share one immutable Hub snapshot.")
    LOGGER.info(
        "Hugging Face model ready: repo=%s revision=%s checkpoint=%s",
        selected_repo,
        selected_revision,
        checkpoint,
    )
    return HubModelArtifacts(checkpoint, calibration, selected_repo, selected_revision)


def resolve_inference_checkpoint(
    config: AppConfig,
    checkpoint_path: str | Path | None = None,
    *,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
) -> Path:
    """Resolve a local checkpoint, falling back to Hugging Face when requested."""
    if hf_repo_id is not None:
        return download_model_artifacts(
            config,
            repo_id=hf_repo_id,
            revision=hf_revision,
        ).checkpoint
    selected = Path(checkpoint_path or config.evaluation.checkpoint_path)
    if selected.is_file():
        return selected
    if checkpoint_path is not None:
        raise FileNotFoundError(f"Detector checkpoint does not exist: {selected}")
    if config.evaluation.hf_model_repo_id is None:
        raise FileNotFoundError(
            f"Detector checkpoint does not exist and no Hugging Face fallback is configured: {selected}"
        )
    return download_model_artifacts(
        config,
        repo_id=hf_repo_id,
        revision=hf_revision,
    ).checkpoint


def main() -> None:
    """Download the configured public model without running inference."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Download RobustFake inference artifacts from Hugging Face.")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face model repository override.")
    parser.add_argument("--hf-revision", default=None, help="Immutable revision or branch override.")
    arguments = parser.parse_args()
    artifacts = download_model_artifacts(
        load_config(arguments.config, arguments.set),
        repo_id=arguments.hf_repo,
        revision=arguments.hf_revision,
    )
    LOGGER.info("Downloaded checkpoint: %s", artifacts.checkpoint)


if __name__ == "__main__":
    main()
