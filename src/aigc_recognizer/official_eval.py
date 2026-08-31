"""Compatibility entry point for the challenge-prescribed WildFake evaluation."""

from __future__ import annotations

from aigc_recognizer.config import AppConfig
from aigc_recognizer.external_eval import (
    _extended_metrics,
    _scenario_image,
    evaluate_external,
)
from aigc_recognizer.external_eval import _main as _shared_main

__all__ = ["_extended_metrics", "_scenario_image", "evaluate_official", "main"]


def evaluate_official(config: AppConfig) -> dict[str, object]:
    """Evaluate through the same manifest-backed pipeline as other datasets."""
    return evaluate_external(config, "wildfake_official")


def main() -> None:
    """Run the challenge-prescribed external evaluation."""
    _shared_main(
        "wildfake_official",
        "Evaluate a detector on the challenge-prescribed WildFake subset.",
    )


if __name__ == "__main__":
    main()
