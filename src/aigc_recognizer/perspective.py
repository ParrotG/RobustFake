"""Traditional-CV line convergence and perspective relationship analysis."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig, PerspectiveConfig, config_argument_parser, load_config


DEFAULT_PERSPECTIVE_OUTPUT_DIR = Path("output/perspective_show")
SUPPORTED_IMAGE_OUTPUT_SUFFIXES = frozenset({
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
})


@dataclass(frozen=True)
class LineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_degrees: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class IntersectionCluster:
    x: float
    y: float
    support: int
    line_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "support": self.support,
            "line_indices": list(self.line_indices),
        }


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install traditional CV support with: uv sync --extra cv") from error
    return cv2


def _load_image(source: str | Path | np.ndarray, max_dimension_limit: int) -> tuple[np.ndarray, float]:
    cv2 = _cv2()
    if isinstance(source, np.ndarray):
        image = source.copy()
    else:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    max_dimension = max(height, width)
    scale = 1.0
    if max_dimension > max_dimension_limit:
        scale = max_dimension_limit / max_dimension
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


def _line_from_array(values: np.ndarray) -> LineSegment:
    x1, y1, x2, y2 = (float(value) for value in values)
    length = math.hypot(x2 - x1, y2 - y1)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return LineSegment(x1, y1, x2, y2, length, angle)


def _angle_distance(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _point_line_distance(point: tuple[float, float], line: LineSegment) -> float:
    numerator = abs(
        (line.y2 - line.y1) * point[0]
        - (line.x2 - line.x1) * point[1]
        + line.x2 * line.y1
        - line.y2 * line.x1
    )
    return numerator / max(line.length, 1e-9)


def _similar_length_lines(lines: list[LineSegment], settings: PerspectiveConfig, diagonal: float) -> list[LineSegment]:
    long_threshold = max(20.0, diagonal * settings.min_long_line_length_ratio)
    long_lines = [line for line in lines if line.length >= long_threshold]
    if not long_lines:
        return []
    best_group: list[LineSegment] = []
    best_score = (-1, -1.0)
    for reference in long_lines:
        group = [
            line
            for line in long_lines
            if abs(line.length - reference.length) / max(reference.length, 1e-9)
            <= settings.similar_length_tolerance
        ]
        score = (len(group), sum(line.length for line in group))
        if score > best_score:
            best_group, best_score = group, score
    return best_group


def _line_color_signature(image: np.ndarray, line: LineSegment) -> np.ndarray:
    """Estimate a robust CIE-Lab color signature around a line segment."""
    cv2 = _cv2()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    dx, dy = line.x2 - line.x1, line.y2 - line.y1
    normal_x, normal_y = -dy / max(line.length, 1e-9), dx / max(line.length, 1e-9)
    samples: list[np.ndarray] = []
    for fraction in np.linspace(0.12, 0.88, 48):
        x = line.x1 + fraction * dx
        y = line.y1 + fraction * dy
        for offset in (-6.0, -3.0, 0.0, 3.0, 6.0):
            sample_x = round(x + normal_x * offset)
            sample_y = round(y + normal_y * offset)
            if 0 <= sample_x < image.shape[1] and 0 <= sample_y < image.shape[0]:
                samples.append(lab[sample_y, sample_x].astype(np.float64))
    if not samples:
        return np.zeros(3, dtype=np.float64)
    return np.median(np.asarray(samples), axis=0)


def _line_anchor(line: LineSegment, width: int) -> tuple[float, float]:
    """Return the endpoint closest to the horizontal image center."""
    center_x = width / 2
    first_distance = abs(line.x1 - center_x)
    second_distance = abs(line.x2 - center_x)
    return (line.x1, line.y1) if first_distance <= second_distance else (line.x2, line.y2)


def _strict_color_length_lines(
    lines: list[LineSegment],
    image: np.ndarray,
    settings: PerspectiveConfig,
    diagonal: float,
) -> list[LineSegment]:
    """Select a strict two-sided color/length line structure.

    This intentionally rejects isolated equal-length edges. A valid group must
    contain two lines from each of two opposing shallow-angle families, with
    central anchors, similar Lab colors, and low overall length variation.
    """
    long_threshold = max(20.0, diagonal * settings.min_long_line_length_ratio)
    width, height = image.shape[1], image.shape[0]
    center_x = width / 2
    anchor_x_limit = width * settings.strict_anchor_x_ratio
    anchor_y_limit = height * settings.strict_anchor_y_ratio
    candidates: list[tuple[LineSegment, np.ndarray]] = []
    for line in lines:
        if line.length < long_threshold:
            continue
        anchor_x, anchor_y = _line_anchor(line, width)
        if abs(anchor_x - center_x) > anchor_x_limit or anchor_y > anchor_y_limit:
            continue
        candidates.append((line, _line_color_signature(image, line)))
    angle_band = settings.strict_angle_band_degrees
    low_family = [item for item in candidates if item[0].angle_degrees <= angle_band]
    high_family = [item for item in candidates if item[0].angle_degrees >= 180.0 - angle_band]

    def best_pair(family: list[tuple[LineSegment, np.ndarray]]) -> tuple[LineSegment, LineSegment] | None:
        possible: list[tuple[float, LineSegment, LineSegment]] = []
        for first_index, (first, first_color) in enumerate(family):
            for second, second_color in family[first_index + 1 :]:
                length_difference = abs(first.length - second.length) / max(first.length, second.length, 1e-9)
                color_distance = float(np.linalg.norm(first_color[1:] - second_color[1:]))
                if length_difference > settings.strict_length_tolerance or color_distance > settings.strict_color_distance:
                    continue
                score = length_difference + color_distance / 255.0
                possible.append((score, first, second))
        if not possible:
            return None
        possible.sort(key=lambda item: item[0])
        return possible[0][1], possible[0][2]

    low_pair = best_pair(low_family)
    high_pair = best_pair(high_family)
    if low_pair is None or high_pair is None:
        return []
    selected = [*low_pair, *high_pair]
    lengths = np.asarray([line.length for line in selected], dtype=np.float64)
    if float(np.std(lengths) / max(np.mean(lengths), 1e-9)) > settings.strict_max_length_cv:
        return []
    return sorted(selected, key=lambda item: item.length, reverse=True)


def _parallel_line_groups(
    lines: list[LineSegment],
    image: np.ndarray,
    settings: PerspectiveConfig,
) -> list[dict[str, Any]]:
    """Group selected lines that are 2D-parallel and color/length compatible."""
    if len(lines) < 2:
        return []
    signatures = [_line_color_signature(image, line) for line in lines]
    relations: dict[int, set[int]] = {index: set() for index in range(len(lines))}
    pair_metrics: dict[tuple[int, int], tuple[float, float]] = {}
    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            second = lines[second_index]
            angle_difference = _angle_distance(first.angle_degrees, second.angle_degrees)
            length_difference = abs(first.length - second.length) / max(first.length, second.length, 1e-9)
            color_distance = float(np.linalg.norm(signatures[first_index][1:] - signatures[second_index][1:]))
            if angle_difference > settings.parallel_angle_tolerance_degrees:
                continue
            if length_difference > settings.strict_length_tolerance:
                continue
            if color_distance > settings.strict_color_distance:
                continue
            relations[first_index].add(second_index)
            relations[second_index].add(first_index)
            pair_metrics[(first_index, second_index)] = (angle_difference, color_distance)

    groups: list[dict[str, Any]] = []
    visited: set[int] = set()
    for start in range(len(lines)):
        if start in visited or not relations[start]:
            continue
        component: set[int] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(relations[current] - component)
        visited.update(component)
        if len(component) < 2:
            continue
        indices = tuple(sorted(component))
        metrics = [
            pair_metrics[tuple(sorted((first, second)))]
            for position, first in enumerate(indices)
            for second in indices[position + 1 :]
            if tuple(sorted((first, second))) in pair_metrics
        ]
        lengths = np.asarray([lines[index].length for index in indices], dtype=np.float64)
        groups.append(
            {
                "line_indices": list(indices),
                "support": len(indices),
                "angle_degrees": round(float(np.mean([lines[index].angle_degrees for index in indices])), 3),
                "angle_spread_degrees": round(float(max(item[0] for item in metrics)), 3),
                "mean_length": round(float(np.mean(lengths)), 3),
                "length_cv": round(float(np.std(lengths) / max(np.mean(lengths), 1e-9)), 6),
                "max_color_distance": round(float(max(item[1] for item in metrics)), 3),
                "relationship": "parallel_to_camera",
            }
        )
    for group_index, group in enumerate(sorted(groups, key=lambda item: item["support"], reverse=True), start=1):
        group["group_id"] = f"P{group_index}"
    return groups


def _deduplicate_lines(lines: list[LineSegment], settings: PerspectiveConfig, diagonal: float) -> list[LineSegment]:
    retained: list[LineSegment] = []
    line_tolerance = max(4.0, diagonal * 0.012)
    for line in sorted(lines, key=lambda item: item.length, reverse=True):
        midpoint = ((line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2)
        duplicate = False
        for previous in retained:
            previous_midpoint = ((previous.x1 + previous.x2) / 2, (previous.y1 + previous.y2) / 2)
            if _angle_distance(line.angle_degrees, previous.angle_degrees) > settings.angle_deduplication_degrees:
                continue
            if _point_line_distance(midpoint, previous) > line_tolerance:
                continue
            direction = np.array(
                [(previous.x2 - previous.x1) / previous.length, (previous.y2 - previous.y1) / previous.length]
            )
            previous_interval = sorted(
                [
                    float(np.dot(np.array([previous.x1, previous.y1]), direction)),
                    float(np.dot(np.array([previous.x2, previous.y2]), direction)),
                ]
            )
            current_interval = sorted(
                [
                    float(np.dot(np.array([line.x1, line.y1]), direction)),
                    float(np.dot(np.array([line.x2, line.y2]), direction)),
                ]
            )
            overlap = min(previous_interval[1], current_interval[1]) - max(previous_interval[0], current_interval[0])
            gap = max(0.0, -overlap)
            if overlap >= min(previous.length, line.length) * 0.25 or gap <= settings.hough_max_line_gap * 2:
                duplicate = True
                break
        if not duplicate:
            retained.append(line)
    return retained


def _line_intersection(first: LineSegment, second: LineSegment) -> tuple[float, float] | None:
    first_x = first.x2 - first.x1
    first_y = first.y2 - first.y1
    second_x = second.x2 - second.x1
    second_y = second.y2 - second.y1
    denominator = first_x * second_y - first_y * second_x
    if abs(denominator) < 1e-9:
        return None
    offset_x = second.x1 - first.x1
    offset_y = second.y1 - first.y1
    first_factor = (offset_x * second_y - offset_y * second_x) / denominator
    return first.x1 + first_factor * first_x, first.y1 + first_factor * first_y


def _cluster_intersections(
    lines: list[LineSegment],
    width: int,
    height: int,
    settings: PerspectiveConfig,
) -> list[IntersectionCluster]:
    diagonal = math.hypot(width, height)
    inlier_distance = max(6.0, diagonal * settings.intersection_cluster_ratio)
    max_distance = diagonal * settings.max_vanishing_distance_ratio
    image_center = (width / 2, height / 2)
    remaining = set(range(len(lines)))
    result: list[IntersectionCluster] = []
    while len(remaining) >= settings.min_vanishing_point_support and len(result) < 4:
        best: tuple[tuple[int, ...], tuple[float, float], float] | None = None
        remaining_indices = sorted(remaining)
        for position, first_index in enumerate(remaining_indices):
            first = lines[first_index]
            for second_index in remaining_indices[position + 1 :]:
                second = lines[second_index]
                if _angle_distance(first.angle_degrees, second.angle_degrees) < settings.min_intersection_angle_degrees:
                    continue
                point = _line_intersection(first, second)
                if point is None or math.dist(point, image_center) > max_distance:
                    continue
                support = tuple(
                    index
                    for index in remaining_indices
                    if _point_line_distance(point, lines[index]) <= inlier_distance
                )
                if len(support) < settings.min_vanishing_point_support:
                    continue
                fitted = _least_squares_intersection([lines[index] for index in support])
                if fitted is None or math.dist(fitted, image_center) > max_distance:
                    continue
                residual = float(np.mean([_point_line_distance(fitted, lines[index]) for index in support]))
                if best is None or (len(support), -residual) > (len(best[0]), -best[2]):
                    best = support, fitted, residual
        if best is None:
            break
        support, point, _ = best
        result.append(
            IntersectionCluster(
                x=float(point[0]),
                y=float(point[1]),
                support=len(support),
                line_indices=support,
            )
        )
        remaining.difference_update(support)
    return result


def _least_squares_intersection(lines: list[LineSegment]) -> tuple[float, float] | None:
    coefficients: list[tuple[float, float, float]] = []
    for line in lines:
        a = line.y2 - line.y1
        b = line.x1 - line.x2
        norm = math.hypot(a, b)
        if norm <= 1e-9:
            continue
        coefficients.append((a / norm, b / norm, (line.x2 * line.y1 - line.x1 * line.y2) / norm))
    if len(coefficients) < 2:
        return None
    matrix = np.array([[a, b] for a, b, _ in coefficients], dtype=np.float64)
    vector = np.array([-c for _, _, c in coefficients], dtype=np.float64)
    if np.linalg.matrix_rank(matrix) < 2:
        return None
    point, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    return float(point[0]), float(point[1])


def _contour_curvature(edges: np.ndarray, settings: PerspectiveConfig) -> dict[str, Any]:
    cv2 = _cv2()
    height, width = edges.shape[:2]
    diagonal = math.hypot(width, height)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_scores: list[float] = []
    contour_details: list[dict[str, float]] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64)
        arc_length = float(cv2.arcLength(contour, False))
        if len(points) < 40 or arc_length < diagonal * settings.fisheye_min_contour_length_ratio:
            continue
        mean_point = np.mean(points, axis=0)
        centered = points - mean_point
        covariance = np.cov(centered, rowvar=False)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            continue
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float64)
        if direction.shape != (2,) or not np.isfinite(direction).all():
            continue
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        offsets = points - mean_point
        longitudinal = offsets[:, 0] * direction[0] + offsets[:, 1] * direction[1]
        transverse = offsets[:, 0] * normal[0] + offsets[:, 1] * normal[1]
        span = float(np.percentile(longitudinal, 97) - np.percentile(longitudinal, 3))
        width_span = float(np.percentile(transverse, 97) - np.percentile(transverse, 3))
        elongation = span / max(width_span, 1e-9)
        if (
            span <= diagonal * settings.fisheye_min_span_ratio
            or elongation < settings.fisheye_min_contour_elongation
        ):
            continue
        bin_edges = np.linspace(np.percentile(longitudinal, 3), np.percentile(longitudinal, 97), 40)
        centerline: list[np.ndarray] = []
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            selected = points[(longitudinal >= lower) & (longitudinal <= upper)]
            if len(selected) >= 2:
                centerline.append(np.median(selected, axis=0))
        sampled = np.asarray(centerline, dtype=np.float64)
        if len(sampled) < 8:
            continue
        chord = float(np.linalg.norm(sampled[-1] - sampled[0]))
        if chord <= 1e-6:
            continue
        centerline_arc = float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum())
        chord_excess = max(0.0, centerline_arc / chord - 1.0)
        line = cv2.fitLine(sampled, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        direction = np.array([line[0], line[1]], dtype=np.float64)
        origin = np.array([line[2], line[3]], dtype=np.float64)
        offsets = sampled - origin
        distances = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
        line_residual_ratio = float(np.percentile(distances, 90) / max(span, 1e-9))
        tangent_vectors = np.diff(sampled, axis=0)
        tangent_angles = np.unwrap(np.arctan2(tangent_vectors[:, 1], tangent_vectors[:, 0]))
        tangent_change_degrees = float(np.degrees(np.percentile(tangent_angles, 90) - np.percentile(tangent_angles, 10)))
        chord_component = min(chord_excess / max(settings.fisheye_min_chord_excess * 3.0, 1e-9), 1.0)
        residual_component = min(line_residual_ratio / max(settings.fisheye_min_line_residual_ratio * 3.0, 1e-9), 1.0)
        tangent_component = min(tangent_change_degrees / max(settings.fisheye_min_angle_change_degrees * 2.0, 1e-9), 1.0)
        score = 0.35 * chord_component + 0.35 * residual_component + 0.30 * tangent_component
        contour_scores.append(score)
        contour_details.append(
            {
                "arc_length": round(arc_length, 3),
                "span": round(span, 3),
                "elongation": round(elongation, 3),
                "chord_excess": round(chord_excess, 6),
                "line_residual_ratio": round(line_residual_ratio, 6),
                "tangent_change_degrees": round(tangent_change_degrees, 3),
                "score": round(score, 6),
            }
        )
    if not contour_scores:
        return {
            "score": 0.0,
            "curved_contour_fraction": 0.0,
            "long_contour_count": 0,
            "curved_contour_count": 0,
            "fisheye_evidence": False,
            "contours": [],
        }
    curved = [
        detail
        for detail in contour_details
        if detail["score"] >= settings.fisheye_score_threshold
        and (
            detail["tangent_change_degrees"] >= settings.fisheye_min_angle_change_degrees
            or detail["chord_excess"] >= settings.fisheye_min_chord_excess
            or detail["line_residual_ratio"] >= settings.fisheye_min_line_residual_ratio
        )
    ]
    fisheye_evidence = (
        len(curved) >= settings.fisheye_min_support
        and len(curved) <= settings.fisheye_max_curved_contour_count
        and len(curved) / len(contour_scores) >= 0.25
    )
    return {
        "score": round(float(np.percentile(contour_scores, 75)), 6),
        "curved_contour_fraction": round(len(curved) / len(contour_scores), 6),
        "long_contour_count": len(contour_scores),
        "curved_contour_count": len(curved),
        "fisheye_evidence": fisheye_evidence,
        "contours": sorted(contour_details, key=lambda item: item["score"], reverse=True)[:20],
    }


def _relationship(intersections: list[IntersectionCluster], line_count: int, curvature: Mapping[str, Any], settings: PerspectiveConfig) -> tuple[str, str]:
    fisheye_evidence = bool(curvature.get("fisheye_evidence"))
    if fisheye_evidence and not intersections:
        return "fisheye_perspective", "Multiple long contours show consistent non-linear curvature without a stable straight-line model."
    if line_count < 3:
        return "insufficient_evidence", "At least three similar-length long lines are required."
    count = len(intersections)
    if count == 1:
        support = intersections[0].support
        if support < line_count:
            return (
                "single_point_perspective_with_outliers",
                f"One dominant intersection is supported by {support} of {line_count} lines; the remaining lines do not pass through that point.",
            )
        return "single_point_perspective", "The detected lines converge to one dominant intersection cluster."
    if count == 2:
        return "two_point_perspective", "The detected lines form two dominant intersection clusters."
    if count == 3:
        return "three_point_perspective", "The detected lines form three dominant intersection clusters."
    if count > 3:
        return "multiple_or_ambiguous_points", "More than three intersection clusters were found; the geometry is ambiguous."
    if fisheye_evidence:
        return "fisheye_perspective", "Long contours show curvature, but the straight-line intersections are not a reliable model."
    return "no_stable_perspective", "Long lines were detected, but their intersections do not form a stable relationship."


def analyze_perspective(source: str | Path | np.ndarray, settings: PerspectiveConfig | None = None) -> dict[str, Any]:
    """Detect similar-length long lines and estimate their intersection relationship."""
    settings = settings or PerspectiveConfig()
    cv2 = _cv2()
    image, scale = _load_image(source, settings.max_dimension)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, settings.canny_low_threshold, settings.canny_high_threshold)
    height, width = gray.shape[:2]
    diagonal = math.hypot(width, height)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=settings.hough_threshold,
        minLineLength=max(20, int(diagonal * settings.hough_min_line_length_ratio)),
        maxLineGap=settings.hough_max_line_gap,
    )
    all_lines = [_line_from_array(item[0]) for item in raw] if raw is not None else []
    unique_lines = _deduplicate_lines(all_lines, settings, diagonal)
    if settings.strict_color_length_selection:
        similar_lines = _strict_color_length_lines(unique_lines, image, settings, diagonal)
    else:
        similar_lines = _similar_length_lines(unique_lines, settings, diagonal)
    similar_lines = sorted(similar_lines, key=lambda item: item.length, reverse=True)[: settings.max_lines]
    parallel_groups = _parallel_line_groups(similar_lines, image, settings)
    intersections = _cluster_intersections(similar_lines, width, height, settings)
    curvature = _contour_curvature(edges, settings)
    relationship, reason = _relationship(intersections, len(similar_lines), curvature, settings)
    lengths = [line.length for line in similar_lines]
    return {
        "schema_version": 1,
        "image": {"width": width, "height": height, "scale": round(scale, 6)},
        "detection": {
            "raw_line_count": len(all_lines),
            "deduplicated_line_count": len(unique_lines),
            "similar_length_line_count": len(similar_lines),
            "mean_length": round(float(np.mean(lengths)), 3) if lengths else 0.0,
            "length_std": round(float(np.std(lengths)), 3) if lengths else 0.0,
            "length_cv": round(float(np.std(lengths) / max(np.mean(lengths), 1e-9)), 6) if lengths else None,
            "lines": [line.as_dict() for line in similar_lines],
            "parallel_group_count": len(parallel_groups),
            "parallel_groups": parallel_groups,
        },
        "intersections": {
            "count": len(intersections),
            "points": [item.as_dict() for item in intersections],
        },
        "curvature": curvature,
        "perspective": {"relationship": relationship, "reason": reason},
    }


def render_perspective_overlay(
    source: str | Path | np.ndarray,
    result: Mapping[str, Any],
    settings: PerspectiveConfig | None = None,
) -> np.ndarray:
    """Render detected long lines, intersection clusters, and the relationship label."""
    settings = settings or PerspectiveConfig()
    cv2 = _cv2()
    image, _ = _load_image(source, settings.max_dimension)
    canvas = image.copy()

    def clipped_line(line: Mapping[str, Any]) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Clip the infinite supporting line to the visible image rectangle."""
        x1, y1 = float(line["x1"]), float(line["y1"])
        x2, y2 = float(line["x2"]), float(line["y2"])
        dx, dy = x2 - x1, y2 - y1
        height, width = canvas.shape[:2]
        candidates: list[tuple[float, float, float]] = []
        if abs(dx) > 1e-9:
            for x in (0.0, float(width - 1)):
                parameter = (x - x1) / dx
                y = y1 + parameter * dy
                if 0.0 <= y <= height - 1:
                    candidates.append((parameter, x, y))
        if abs(dy) > 1e-9:
            for y in (0.0, float(height - 1)):
                parameter = (y - y1) / dy
                x = x1 + parameter * dx
                if 0.0 <= x <= width - 1:
                    candidates.append((parameter, x, y))
        if len(candidates) < 2:
            return None
        candidates.sort(key=lambda item: item[0])
        first, second = candidates[0], candidates[-1]
        return (round(first[1]), round(first[2])), (round(second[1]), round(second[2]))

    def draw_dashed(start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        """Draw a dashed line so extension portions remain visually distinct."""
        start_point = np.array(start, dtype=np.float64)
        end_point = np.array(end, dtype=np.float64)
        vector = end_point - start_point
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-9:
            return
        direction = vector / distance
        dash_length, gap_length = 18.0, 10.0
        position = 0.0
        while position < distance:
            dash_end = min(position + dash_length, distance)
            dash_start_point = tuple(np.round(start_point + direction * position).astype(int))
            dash_end_point = tuple(np.round(start_point + direction * dash_end).astype(int))
            cv2.line(canvas, dash_start_point, dash_end_point, color, 3, cv2.LINE_AA)
            position += dash_length + gap_length

    lines = result.get("detection", {}).get("lines", [])
    palette = (
        (0, 80, 255),
        (0, 190, 255),
        (0, 220, 120),
        (255, 180, 0),
        (220, 80, 255),
        (255, 80, 80),
    )
    for index, line in enumerate(lines):
        start = (round(float(line["x1"])), round(float(line["y1"])))
        end = (round(float(line["x2"])), round(float(line["y2"])))
        color = palette[index % len(palette)]
        visible_line = clipped_line(line)
        if visible_line is not None:
            visible_start, visible_end = visible_line
            start_parameter = math.dist(visible_start, start)
            end_parameter = math.dist(visible_end, end)
            if start_parameter > 8:
                draw_dashed(visible_start, start, color)
            if end_parameter > 8:
                draw_dashed(end, visible_end, color)
        cv2.line(canvas, start, end, color, 5, cv2.LINE_AA)
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(
            canvas,
            f"L{index + 1}",
            (midpoint[0] + 6, midpoint[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    parallel_groups = result.get("detection", {}).get("parallel_groups", [])
    for group in parallel_groups:
        group_indices = [int(index) for index in group.get("line_indices", [])]
        group_lines = [lines[index] for index in group_indices if 0 <= index < len(lines)]
        if not group_lines:
            continue
        image_center_x = canvas.shape[1] / 2
        outer_points = []
        for line in group_lines:
            first = (float(line["x1"]), float(line["y1"]))
            second = (float(line["x2"]), float(line["y2"]))
            outer_points.append(first if abs(first[0] - image_center_x) >= abs(second[0] - image_center_x) else second)
        label_x = round(float(np.mean([point[0] for point in outer_points])))
        label_y = round(float(np.mean([point[1] for point in outer_points])))
        label = f"{group.get('group_id', 'P?')} || lens"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0]
        origin = (label_x - text_size[0] // 2, label_y - 12)
        cv2.rectangle(
            canvas,
            (origin[0] - 5, origin[1] - text_size[1] - 5),
            (origin[0] + text_size[0] + 5, origin[1] + 5),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    points = result.get("intersections", {}).get("points", [])
    for index, point in enumerate(points, start=1):
        center = (round(float(point["x"])), round(float(point["y"])))
        cv2.drawMarker(canvas, center, (0, 0, 255), cv2.MARKER_CROSS, 36, 4, cv2.LINE_AA)
        cv2.circle(canvas, center, 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"V{index} ({point['x']:.0f}, {point['y']:.0f})",
            (center[0] + 14, center[1] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    pair_intersections: list[tuple[float, float]] = []
    segment_objects = [
        _line_from_array(np.array([line["x1"], line["y1"], line["x2"], line["y2"]], dtype=np.float64))
        for line in lines
    ]
    height, width = canvas.shape[:2]
    dominant_points = [(float(point["x"]), float(point["y"])) for point in points]
    for first_index, first in enumerate(segment_objects):
        for second in segment_objects[first_index + 1 :]:
            if _angle_distance(first.angle_degrees, second.angle_degrees) < settings.min_intersection_angle_degrees:
                continue
            point = _line_intersection(first, second)
            if point is None or not (0 <= point[0] < width and 0 <= point[1] < height):
                continue
            if any(math.dist(point, dominant) < 24 for dominant in dominant_points):
                continue
            if any(math.dist(point, previous) < 18 for previous in pair_intersections):
                continue
            pair_intersections.append(point)
            center = (round(point[0]), round(point[1]))
            cv2.circle(canvas, center, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 7, (30, 30, 30), 2, cv2.LINE_AA)
    relationship = result.get("perspective", {}).get("relationship", "unknown")
    dominant_support = max((int(point.get("support", 0)) for point in points), default=0)
    banner_primary = relationship
    banner_secondary = (
        f"solid = detected segment | dashed = extension | lines = {len(lines)} | "
        f"parallel groups = {len(parallel_groups)} | dominant support = {dominant_support}/{len(lines)} | "
        f"other intersections = {len(pair_intersections)}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 68), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        banner_primary,
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        banner_secondary,
        (14, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def _default_output_stem(input_source: str | Path) -> str:
    input_path = Path(input_source)
    return input_path.stem or "image"


def _is_directory_argument(path: Path) -> bool:
    if path.exists():
        return path.is_dir()
    return not path.suffix


def _default_report_path(input_source: str | Path, output_directory: Path = DEFAULT_PERSPECTIVE_OUTPUT_DIR) -> Path:
    return output_directory / f"{_default_output_stem(input_source)}-perspective-report.json"


def _default_visual_path(input_source: str | Path, output_directory: Path = DEFAULT_PERSPECTIVE_OUTPUT_DIR) -> Path:
    return output_directory / f"{_default_output_stem(input_source)}-perspective-overlay.jpg"


def _visual_path_for_report(report_path: Path) -> Path:
    report_stem = report_path.stem
    suffix = "-perspective-report"
    if report_stem.endswith(suffix):
        visual_stem = report_stem[: -len(suffix)] + "-perspective-overlay"
    else:
        visual_stem = report_stem + "-overlay"
    return report_path.with_name(visual_stem + ".jpg")


def _resolve_output_paths(
    input_source: str | Path,
    output: str | Path | None,
    visual_output: str | Path | None,
) -> tuple[Path, Path]:
    """Resolve report and overlay destinations, accepting files or directories."""
    if output is None:
        report_path = _default_report_path(input_source)
    else:
        output_path = Path(output)
        if _is_directory_argument(output_path):
            report_path = _default_report_path(input_source, output_path)
        elif output_path.suffix.lower() == ".json":
            report_path = output_path
        else:
            raise ValueError(
                f"JSON output must use a .json extension or be a directory: {output_path}"
            )

    if visual_output is None:
        visual_path = _visual_path_for_report(report_path)
    else:
        visual_output_path = Path(visual_output)
        if _is_directory_argument(visual_output_path):
            visual_path = visual_output_path / _default_visual_path(input_source).name
        elif visual_output_path.suffix.lower() in SUPPORTED_IMAGE_OUTPUT_SUFFIXES:
            visual_path = visual_output_path
        else:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_OUTPUT_SUFFIXES))
            raise ValueError(
                f"Annotated image output must use a supported image extension ({supported}) "
                f"or be a directory: {visual_output_path}"
            )
    return report_path, visual_path


def _write_image(image: np.ndarray, destination: Path) -> None:
    """Write an image atomically, ensuring OpenCV receives a valid extension."""
    cv2 = _cv2()
    suffix = destination.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_OUTPUT_SUFFIXES))
        raise ValueError(f"Unsupported image output extension {destination.suffix!r}; use {supported}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"Could not write annotated image: {destination}")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def argument_parser() -> argparse.ArgumentParser:
    parser = config_argument_parser("Analyze long-line intersections and perspective relationships.")
    parser.add_argument("--input", required=True, help="Image path to analyze.")
    parser.add_argument(
        "--output",
        help=(
            "JSON report path or output directory. Defaults to "
            f"{DEFAULT_PERSPECTIVE_OUTPUT_DIR}/<input>-perspective-report.json."
        ),
    )
    parser.add_argument(
        "--visual-output",
        help=(
            "Optional annotated image path or directory. When omitted, an overlay is "
            "generated beside the JSON report."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    config: AppConfig = load_config(args.config, args.set)
    result = analyze_perspective(args.input, config.perspective)
    report_path, visual_path = _resolve_output_paths(args.input, args.output, args.visual_output)
    _write_json(result, report_path)
    annotated = render_perspective_overlay(args.input, result, config.perspective)
    _write_image(annotated, visual_path)


if __name__ == "__main__":
    main()
