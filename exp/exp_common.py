"""Shared, read-only utilities for the revision experiments."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = {"RV": 1, "MYO": 2, "LV": 3}
FILENAME_RE = re.compile(
    r"patient(?P<patient>\d+)_frame(?P<frame>\d+)_z(?P<slice>\d+)\.png$"
)


def parse_key(path: Path) -> tuple[int, int, int]:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected ACDC filename: {path.name}")
    return tuple(int(match[name]) for name in ("patient", "slice", "frame"))


def index_pngs(directory: Path) -> dict[tuple[int, int, int], Path]:
    return {parse_key(path): path for path in directory.glob("patient*_frame*_z*.png")}


def load_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def scores_to_labels(rgb: np.ndarray, threshold: float) -> np.ndarray:
    scores = rgb.astype(np.float32) / 255.0
    labels = np.argmax(scores, axis=-1).astype(np.uint8) + 1
    labels[np.max(scores, axis=-1) <= threshold] = 0
    return labels


def labels_to_one_hot(labels: np.ndarray) -> np.ndarray:
    return np.stack([labels == value for value in CLASSES.values()], axis=-1)


def one_hot_to_labels(one_hot: np.ndarray) -> np.ndarray:
    labels = np.argmax(one_hot, axis=-1).astype(np.uint8) + 1
    labels[~np.any(one_hot, axis=-1)] = 0
    return labels


def dice(first: np.ndarray, second: np.ndarray, empty_value: float = 1.0) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = int(first.sum() + second.sum())
    if denominator == 0:
        return empty_value
    return 2.0 * float(np.logical_and(first, second).sum()) / denominator


def class_dice(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    result = {name: dice(gt == label, pred == label) for name, label in CLASSES.items()}
    result["foreground"] = dice(gt > 0, pred > 0)
    return result


def evaluate_predictions(
    predictions: dict[tuple[int, int, int], np.ndarray],
    mask_dir: Path,
    dataset: str,
    method: str,
) -> tuple[list[dict], list[dict]]:
    """Return image-level and patient-level Dice rows."""
    mask_paths = index_pngs(mask_dir)
    image_rows = []
    patient_values: dict[tuple[int, str], list[float]] = defaultdict(list)

    for key, gt_path in sorted(mask_paths.items()):
        if key not in predictions:
            continue
        patient, slice_id, frame_id = key
        scores = class_dice(load_label(gt_path), predictions[key])
        for class_name, value in scores.items():
            image_rows.append({
                "dataset": dataset,
                "method": method,
                "patient": patient,
                "slice": slice_id,
                "frame": frame_id,
                "class": class_name,
                "dice": value,
            })
            patient_values[(patient, class_name)].append(value)

    patient_rows = [
        {
            "dataset": dataset,
            "method": method,
            "patient": patient,
            "class": class_name,
            "dice": float(np.mean(values)),
            "annotated_images": len(values),
        }
        for (patient, class_name), values in sorted(patient_values.items())
    ]
    return image_rows, patient_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction_set(
    output_dir: Path,
    predictions: dict[tuple[int, int, int], np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for (patient, slice_id, frame_id), mask in predictions.items():
        filename = f"patient{patient:03d}_frame{frame_id:02d}_z{slice_id:03d}.png"
        Image.fromarray(mask.astype(np.uint8), mode="L").save(output_dir / filename)

