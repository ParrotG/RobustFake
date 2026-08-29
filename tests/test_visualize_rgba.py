import json
from pathlib import Path

import numpy as np
from PIL import Image

from aigc_recognizer.visualize_rgba import analyze_channel, load_rgba, main


def test_rgba_channel_analysis_marks_alpha_and_returns_stable_shapes() -> None:
    image = np.full((24, 32), 255, dtype=np.uint8)
    image[8:16, 10:22] = 80
    analysis = analyze_channel(image, "A")

    assert analysis.channel.shape == (24, 32)
    assert analysis.residual.shape == (24, 32)
    assert analysis.suspicious.shape == (24, 32)
    assert analysis.suspicious[10, 12]


def test_rgba_cli_writes_panel_chart_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.png"
    output_dir = tmp_path / "outputs"
    image = Image.new("RGBA", (32, 24), (30, 80, 140, 255))
    image.putpixel((12, 10), (255, 0, 0, 100))
    image.save(input_path)

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "aigc-visualize-rgba",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
        main()
    finally:
        sys.argv = old_argv

    assert (output_dir / "rgba_channel_analysis.png").is_file()
    assert (output_dir / "rgba_residual_chart.png").is_file()
    report = json.loads((output_dir / "rgba_channel_report.json").read_text())
    assert report["had_alpha_channel"] is True
    assert set(report["channels"]) == {"R", "G", "B", "A"}


def test_rgb_input_synthesizes_opaque_alpha(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jpg"
    Image.new("RGB", (10, 10), (20, 30, 40)).save(input_path)

    image, had_alpha = load_rgba(input_path)

    assert image.mode == "RGBA"
    assert had_alpha is False
    assert image.getchannel("A").getextrema() == (255, 255)
