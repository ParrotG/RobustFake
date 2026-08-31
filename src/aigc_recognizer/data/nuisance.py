"""Low-level nuisance probing for a prepared paired image dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

from aigc_recognizer.config import AppConfig
from aigc_recognizer.data.transforms import RobustPairTransform, canonical_rgb

LOGGER = logging.getLogger(__name__)


def _seed(project_seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{project_seed}:{record_id}:nuisance".encode()).hexdigest()
    return int(digest[:16], 16)


def _entropy(values: np.ndarray) -> float:
    histogram = np.histogram(values, bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    probabilities = histogram / max(1.0, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _feature_names() -> list[str]:
    names: list[str] = []
    for channel in ("red", "green", "blue", "luma"):
        names.extend(f"{channel}_{stat}" for stat in ("mean", "std", "q10", "q50", "q90"))
        names.extend(f"{channel}_hist_{index}" for index in range(8))
    names.extend(
        [
            "luma_entropy",
            "saturation_mean",
            "saturation_std",
            "gradient_mean",
            "gradient_std",
            "edge_density",
            "laplacian_variance",
            "blockiness",
            "fft_band_0",
            "fft_band_1",
            "fft_band_2",
            "fft_band_3",
            "fft_high_low_ratio",
        ]
    )
    return names


FEATURE_NAMES = _feature_names()


def extract_nuisance_features(image: Image.Image, size: int = 128) -> np.ndarray:
    """Extract fixed image-visible statistics without provenance or semantic embeddings."""
    image = image.resize((size, size), Image.Resampling.BILINEAR).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    features: list[float] = []
    for channel in (rgb[..., 0], rgb[..., 1], rgb[..., 2], luma):
        features.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                *np.quantile(channel, [0.1, 0.5, 0.9]).astype(float).tolist(),
            ]
        )
        features.extend(
            (np.histogram(channel, bins=8, range=(0.0, 1.0))[0] / channel.size)
            .astype(float)
            .tolist()
        )
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 1e-6,
    )
    gradient_x = np.diff(luma, axis=1)
    gradient_y = np.diff(luma, axis=0)
    gradient = np.hypot(gradient_x[:-1], gradient_y[:, :-1])
    edge_threshold = float(gradient.mean() + gradient.std())
    laplacian = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    block_boundaries = np.arange(8, size, 8)
    if block_boundaries.size:
        boundary_vertical = np.abs(
            luma[:, block_boundaries] - luma[:, block_boundaries - 1]
        ).mean()
        boundary_horizontal = np.abs(
            luma[block_boundaries, :] - luma[block_boundaries - 1, :]
        ).mean()
    else:
        boundary_vertical = boundary_horizontal = 0.0
    regular_vertical = np.abs(np.diff(luma, axis=1)).mean()
    regular_horizontal = np.abs(np.diff(luma, axis=0)).mean()
    blockiness = float(
        (boundary_vertical + boundary_horizontal)
        / max(1e-8, regular_vertical + regular_horizontal)
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(luma - luma.mean()))) ** 2
    yy, xx = np.indices(spectrum.shape)
    radius = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    radius /= max(1.0, radius.max())
    bands = []
    total_energy = float(spectrum.sum()) + 1e-12
    for low, high in ((0.0, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.01)):
        bands.append(float(spectrum[(radius >= low) & (radius < high)].sum() / total_energy))
    features.extend(
        [
            _entropy(luma),
            float(saturation.mean()),
            float(saturation.std()),
            float(gradient.mean()),
            float(gradient.std()),
            float(np.mean(gradient > edge_threshold)),
            float(laplacian.var()),
            blockiness,
            *bands,
            float((bands[2] + bands[3]) / max(1e-12, bands[0] + bands[1])),
        ]
    )
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Nuisance feature extraction produced invalid values.")
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "count": int(labels.size),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _encoded_byte_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    materialized = list(records)
    for label in (0, 1):
        values = np.asarray(
            [
                float(record["bytes"])
                for record in materialized
                if record["label"] == label and "bytes" in record
            ],
            dtype=np.float64,
        )
        if values.size:
            summaries[str(label)] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "q10": float(np.quantile(values, 0.1)),
                "q50": float(np.quantile(values, 0.5)),
                "q90": float(np.quantile(values, 0.9)),
            }
    return summaries


def _sample_records(
    records: list[dict[str, Any]], limit: int, project_seed: int, split: str
) -> list[dict[str, Any]]:
    """Select a deterministic label/source-stratified diagnostic sample."""
    if len(records) <= limit:
        return records
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["label"]), str(record.get("source_dataset", "unknown")))
        grouped.setdefault(key, []).append(record)
    selected: list[dict[str, Any]] = []
    remaining = limit
    # Keep every non-empty label/source cell represented before allocating the
    # remaining budget proportionally. This audit is diagnostic, not training.
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: _seed(project_seed, str(item["id"])))
        selected.append(group[0])
        grouped[key] = group[1:]
        remaining -= 1
    if remaining < 0:
        return selected[:limit]
    pool = [record for group in grouped.values() for record in group]
    pool.sort(key=lambda item: _seed(project_seed, f"{split}:{item['id']}"))
    selected.extend(pool[:remaining])
    return selected


def _condition_features(
    records: Iterable[dict[str, Any]], config: AppConfig, standardized: bool, split: str = "unknown"
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    root = Path(config.data.output_dir)
    transform = RobustPairTransform(config)

    def extract(record: dict[str, Any]) -> np.ndarray:
        with Image.open(root / record["path"]) as source:
            image = canonical_rgb(source.copy(), config.views.padding_color)
        if standardized:
            image = transform.standardize(
                image,
                random.Random(_seed(config.project.seed, str(record["id"]))),
            )
        return extract_nuisance_features(image, config.nuisance_audit.feature_size)

    retained = list(records)
    condition_name = "standardized" if standardized else "raw"
    with ThreadPoolExecutor(max_workers=config.nuisance_audit.feature_workers) as executor:
        vectors = list(
            tqdm(
                executor.map(extract, retained),
                total=len(retained),
                desc=f"Nuisance features ({condition_name}, {split})",
            )
        )
    labels = [int(record["label"]) for record in retained]
    return np.stack(vectors), np.asarray(labels, dtype=np.int64), retained


def _probe_condition(
    all_records: list[dict[str, Any]], config: AppConfig, standardized: bool
) -> dict[str, Any]:
    available = {str(record["split"]) for record in all_records}
    evaluation_splits = (
        ["val_id", "val_dg"] if {"val_id", "val_dg"} <= available else ["val", "test"]
    )
    selected_splits = ["train", *evaluation_splits]
    by_split = {}
    for split in selected_splits:
        records = [record for record in all_records if record["split"] == split]
        limit = (
            config.nuisance_audit.max_train_samples
            if split == "train"
            else config.nuisance_audit.max_evaluation_samples
        )
        by_split[split] = _sample_records(records, limit, config.project.seed, split)
        LOGGER.info(
            "Nuisance audit selected %d of %d %s records.",
            len(by_split[split]),
            len(records),
            split,
        )
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    ordered: dict[str, list[dict[str, Any]]] = {}
    for split, records in by_split.items():
        matrices[split], labels[split], ordered[split] = _condition_features(
            records, config, standardized, split
        )
    audit = config.nuisance_audit
    classifier = HistGradientBoostingClassifier(
        learning_rate=audit.learning_rate,
        max_iter=audit.max_iter,
        max_leaf_nodes=audit.max_leaf_nodes,
        min_samples_leaf=audit.min_samples_leaf,
        random_state=audit.random_state,
        early_stopping=False,
    )
    classifier.fit(matrices["train"], labels["train"])
    result: dict[str, Any] = {"splits": {}, "per_generator": {}}
    probabilities: dict[str, np.ndarray] = {}
    for split in evaluation_splits:
        probabilities[split] = classifier.predict_proba(matrices[split])[:, 1]
        result["splits"][split] = _metrics(labels[split], probabilities[split])
    importance = permutation_importance(
        classifier,
        matrices[evaluation_splits[0]],
        labels[evaluation_splits[0]],
        scoring="roc_auc",
        n_repeats=audit.permutation_repeats,
        random_state=audit.random_state,
    )
    result["permutation_importance"] = sorted(
        (
            {
                "feature": name,
                "mean": float(mean),
                "std": float(std),
            }
            for name, mean, std in zip(
                FEATURE_NAMES, importance.importances_mean, importance.importances_std
            )
        ),
        key=lambda item: item["mean"],
        reverse=True,
    )
    generators = sorted(
        {
            str(record.get("generator", record.get("model_id", "unknown")))
            for record in all_records
            if int(record["label"]) == 1
        }
    )
    for generator in generators:
        result["per_generator"][generator] = {}
        for split in evaluation_splits:
            selected = np.asarray(
                [
                    record["label"] == 0 or record.get("generator") == generator
                    for record in ordered[split]
                ],
                dtype=bool,
            )
            selected_labels = labels[split][selected]
            if selected.sum() and len(set(selected_labels.tolist())) == 2:
                result["per_generator"][generator][split] = _metrics(
                    selected_labels, probabilities[split][selected]
                )
            else:
                result["per_generator"][generator][split] = {
                    "count": int(selected.sum()),
                    "skipped": "The selected split does not contain both labels.",
                }
    result["per_source_dataset"] = {}
    for source in sorted({str(record.get("source_dataset", "unknown")) for record in all_records}):
        result["per_source_dataset"][source] = {}
        for split in evaluation_splits:
            selected = np.asarray(
                [str(record.get("source_dataset", "unknown")) == source for record in ordered[split]],
                dtype=bool,
            )
            source_labels = labels[split][selected]
            if selected.sum() and len(set(source_labels.tolist())) == 2:
                result["per_source_dataset"][source][split] = _metrics(
                    source_labels, probabilities[split][selected]
                )
    return result


def run_nuisance_audit(config: AppConfig) -> dict[str, Any]:
    """Fit an informational low-level probe and atomically write its report."""
    manifest = Path(config.data.manifest_path)
    records = _records(manifest)
    report: dict[str, Any] = {
        "schema_version": 1,
        "informational_only": True,
        "interpretation": {
            "weak_shortcut": "AUROC <= 0.60",
            "investigate": "0.60 < AUROC <= 0.70",
            "strong_signal": "AUROC > 0.70",
            "caveat": "Low-level generator fingerprints may be task signal rather than dataset nuisance.",
        },
        "metadata": {
            "counts_by_split_label": dict(
                sorted(Counter(f"{r['split']}:{r['label']}" for r in records).items())
            ),
            "fake_counts_by_generator": dict(
                sorted(Counter(r["generator"] for r in records if r["label"] == 1).items())
            ),
            "real_counts_by_source": dict(
                sorted(Counter(r.get("real_source", "unknown") for r in records if r["label"] == 0).items())
            ),
            "real_counts_by_dataset": dict(
                sorted(Counter(r.get("source_dataset", "unknown") for r in records if r["label"] == 0).items())
            ),
            "formats": dict(sorted(Counter(r["format"] for r in records).items())),
            "dimensions": dict(
                sorted(Counter(f"{r['width']}x{r['height']}" for r in records).items())
            ),
            "encoded_bytes_by_label": _encoded_byte_summary(records),
        },
        "raw_canonical": _probe_condition(records, config, standardized=False),
        "standardized": _probe_condition(records, config, standardized=True),
    }
    from aigc_recognizer.data.prepare import atomic_write_text

    path = Path(config.data.nuisance_report_path)
    atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Nuisance audit report written to %s.", path)
    return report
