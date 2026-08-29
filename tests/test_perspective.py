from pathlib import Path

import numpy as np
import pytest

from aigc_recognizer.config import load_config
from aigc_recognizer.config import PerspectiveConfig
from aigc_recognizer.perspective import (
    IntersectionCluster,
    LineSegment,
    _contour_curvature,
    _parallel_line_groups,
    _resolve_output_paths,
    _relationship,
    analyze_perspective,
    main,
)


cv2 = pytest.importorskip("cv2")
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def draw_lines(lines: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    for start, end in lines:
        cv2.line(image, start, end, (255, 255, 255), 4)
    return image


def segment_toward_vanishing_point(
    vanishing_point: tuple[float, float],
    anchor: tuple[float, float],
    first_x: int,
    second_x: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    vx, vy = vanishing_point
    ax, ay = anchor

    def point_at(x: int) -> tuple[int, int]:
        ratio = (x - vx) / (ax - vx)
        return x, round(vy + ratio * (ay - vy))

    return point_at(first_x), point_at(second_x)


def test_single_point_perspective_from_concurrent_lines() -> None:
    image = draw_lines([
        ((50, 50), (400, 300)),
        ((50, 550), (400, 300)),
        ((750, 50), (400, 300)),
        ((750, 550), (400, 300)),
    ])
    result = analyze_perspective(image, PerspectiveConfig(strict_color_length_selection=False))
    assert result["detection"]["similar_length_line_count"] >= 3
    assert result["intersections"]["count"] == 1
    assert result["perspective"]["relationship"] == "single_point_perspective"


def test_two_point_perspective_from_two_vanishing_points() -> None:
    left = (-500.0, 300.0)
    right = (1300.0, 300.0)
    image = draw_lines(
        [segment_toward_vanishing_point(left, (300, y), 50, 350) for y in (160, 300, 440)]
        + [segment_toward_vanishing_point(right, (500, y), 450, 750) for y in (160, 300, 440)]
    )
    result = analyze_perspective(image, PerspectiveConfig(strict_color_length_selection=False))
    assert result["intersections"]["count"] == 2
    assert result["perspective"]["relationship"] == "two_point_perspective"


def test_three_point_perspective_from_three_line_families() -> None:
    left = (-500.0, 300.0)
    right = (1300.0, 300.0)
    top = (400.0, -900.0)
    horizontal_families = (
        [segment_toward_vanishing_point(left, (300, y), 50, 350) for y in (170, 300, 430)]
        + [segment_toward_vanishing_point(right, (500, y), 450, 750) for y in (170, 300, 430)]
    )
    vertical_family = []
    for x in (270, 400, 530):
        first_y, second_y = 330, 590
        first_ratio = (first_y - top[1]) / (350 - top[1])
        second_ratio = (second_y - top[1]) / (350 - top[1])
        first_x = round(top[0] + first_ratio * (x - top[0]))
        second_x = round(top[0] + second_ratio * (x - top[0]))
        vertical_family.append(((first_x, first_y), (second_x, second_y)))
    image = draw_lines(horizontal_families + vertical_family)
    result = analyze_perspective(image, PerspectiveConfig(strict_color_length_selection=False))
    assert result["intersections"]["count"] == 3
    assert result["perspective"]["relationship"] == "three_point_perspective"


def test_noisy_image_reports_insufficient_evidence() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    result = analyze_perspective(image, PerspectiveConfig(strict_color_length_selection=False))
    assert result["perspective"]["relationship"] in {
        "insufficient_evidence",
        "no_stable_perspective",
        "multiple_or_ambiguous_points",
    }


def test_single_dominant_point_with_outlier_is_reported_separately() -> None:
    relationship, reason = _relationship(
        [IntersectionCluster(100.0, 100.0, 4, (0, 1, 2, 3))],
        line_count=5,
        curvature={"score": 0.0},
        settings=PerspectiveConfig(),
    )
    assert relationship == "single_point_perspective_with_outliers"
    assert "4 of 5" in reason


def test_parallel_groups_are_marked_as_camera_parallel() -> None:
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    lines = [
        LineSegment(80, 100, 280, 118, 200.809, 5.132),
        LineSegment(80, 220, 280, 238, 200.809, 5.132),
        LineSegment(320, 120, 120, 102, 200.809, 174.868),
        LineSegment(320, 240, 120, 222, 200.809, 174.868),
    ]
    groups = _parallel_line_groups(lines, image, PerspectiveConfig())
    assert len(groups) == 2
    assert all(group["relationship"] == "parallel_to_camera" for group in groups)
    assert sorted(group["support"] for group in groups) == [2, 2]


def test_fisheye_requires_consistent_curved_long_contours() -> None:
    image = np.zeros((600, 1000), dtype=np.uint8)
    for baseline in (170, 430):
        points = np.array(
            [[x, int(baseline + 100 * ((x - 500) / 500) ** 2)] for x in np.linspace(30, 970, 300)],
            dtype=np.int32,
        )
        cv2.polylines(image, [points], False, 255, 5)
    edges = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 0), 50, 150)
    curvature = _contour_curvature(edges, PerspectiveConfig())
    assert curvature["fisheye_evidence"] is True
    assert curvature["curved_contour_count"] >= 2


def test_cli_generates_json_and_overlay_by_default(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(input_path), np.zeros((80, 100, 3), dtype=np.uint8))
    monkeypatch.chdir(tmp_path)

    main(["--config", str(DEFAULT_CONFIG), "--input", str(input_path)])

    assert (tmp_path / "output/perspective_show/sample-perspective-report.json").is_file()
    assert (tmp_path / "output/perspective_show/sample-perspective-overlay.jpg").is_file()


def test_visual_output_directory_gets_a_supported_image_name(tmp_path) -> None:
    input_path = tmp_path / "sample.jpg"
    report_path, visual_path = _resolve_output_paths(input_path, tmp_path / "report.json", tmp_path / "overlays")

    assert report_path == tmp_path / "report.json"
    assert visual_path == tmp_path / "overlays/sample-perspective-overlay.jpg"


def test_output_directory_generates_both_paths(tmp_path) -> None:
    report_path, visual_path = _resolve_output_paths(tmp_path / "3.jpg", tmp_path / "perspective_show", None)

    assert report_path == tmp_path / "perspective_show/3-perspective-report.json"
    assert visual_path == tmp_path / "perspective_show/3-perspective-overlay.jpg"
