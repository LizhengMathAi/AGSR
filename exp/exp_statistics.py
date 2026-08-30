"""Patient-level Dice, DT/DS/DN, bootstrap CIs, and paired tests.

Predictions are supplied as DATASET:METHOD=MASK_DIRECTORY. Patients—not slices
or frames—are the statistical units. Presence-aware neighbor Dice excludes
empty/empty pairs, retains one-empty pairs as zero, and therefore cannot be
inflated by repeated omissions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp_common import CLASSES, class_dice, dice, index_pngs, load_label, write_csv


def parse_method(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use NAME=MASK_DIRECTORY")
    name, directory = text.split("=", 1)
    return name.strip(), Path(directory)


def parse_prediction(text: str) -> tuple[str, str, Path]:
    if ":" not in text or "=" not in text:
        raise argparse.ArgumentTypeError("Use DATASET:METHOD=MASK_DIRECTORY")
    dataset, remainder = text.split(":", 1)
    method, directory = remainder.split("=", 1)
    return dataset.strip(), method.strip(), Path(directory)


def patient_metrics(mask_paths, prediction_paths, dataset, method):
    predictions = {key: load_label(path) for key, path in prediction_paths.items()}
    patients = sorted({key[0] for key in predictions})
    rows = []
    for patient in patients:
        patient_keys = {key for key in predictions if key[0] == patient}
        for class_name, label in {**CLASSES, "foreground": -1}.items():
            selector = (lambda mask: mask > 0) if label == -1 else (lambda mask, value=label: mask == value)
            gt_values = []
            for key, path in mask_paths.items():
                if key[0] == patient and key in predictions:
                    gt_values.append(dice(selector(load_label(path)), selector(predictions[key])))

            temporal, spatial, temporal_pa, spatial_pa = [], [], [], []
            for _, slice_id, frame_id in sorted(patient_keys):
                center = predictions[(patient, slice_id, frame_id)]
                temporal_keys = (
                    (patient, slice_id, frame_id - 1),
                    (patient, slice_id, frame_id + 1),
                )
                spatial_keys = (
                    (patient, slice_id - 1, frame_id),
                    (patient, slice_id + 1, frame_id),
                )

                def local_scores(neighbor_keys):
                    ordinary, aware = [], []
                    for neighbor in neighbor_keys:
                        neighbor_mask = predictions[neighbor]
                        first, second = selector(center), selector(neighbor_mask)
                        ordinary.append(dice(first, second))
                        aware.append(np.nan if not first.any() and not second.any() else dice(first, second))
                    return float(np.mean(ordinary)), (float(np.nanmean(aware)) if np.isfinite(aware).any() else np.nan)

                if all(key in predictions for key in temporal_keys):
                    ordinary, aware = local_scores(temporal_keys)
                    temporal.append(ordinary)
                    temporal_pa.append(aware)
                if all(key in predictions for key in spatial_keys):
                    ordinary, aware = local_scores(spatial_keys)
                    spatial.append(ordinary)
                    spatial_pa.append(aware)

            def safe_mean(values):
                values = np.asarray(values, dtype=float)
                return float(np.nanmean(values)) if values.size and np.isfinite(values).any() else np.nan

            dt, ds = safe_mean(temporal), safe_mean(spatial)
            dt_pa, ds_pa = safe_mean(temporal_pa), safe_mean(spatial_pa)
            rows.append({
                "dataset": dataset, "method": method, "patient": patient, "class": class_name,
                "dice": safe_mean(gt_values), "DT": dt, "DS": ds,
                "DN": np.nanmean([dt, ds]), "DT_presence_aware": dt_pa,
                "DS_presence_aware": ds_pa, "DN_presence_aware": np.nanmean([dt_pa, ds_pa]),
            })
    return rows


def bootstrap_ci(values, rng, resamples):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975]).tolist()


def paired_sign_flip(first, second, rng, permutations):
    delta = np.asarray(first) - np.asarray(second)
    delta = delta[np.isfinite(delta)]
    if not len(delta):
        return np.nan, np.nan, 0
    observed = float(delta.mean())
    extreme = 0
    for _ in range(permutations):
        statistic = np.mean(delta * rng.choice((-1, 1), len(delta)))
        extreme += abs(statistic) >= abs(observed)
    return observed, (extreme + 1) / (permutations + 1), len(delta)


def add_holm_adjustment(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["dataset"], row["class"], row["metric"])].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: rows[index]["p_value_two_sided"])
        running_max = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * rows[index]["p_value_two_sided"])
            running_max = max(running_max, adjusted)
            rows[index]["p_value_holm"] = running_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=ROOT")
    parser.add_argument("--prediction", action="append", type=parse_prediction, required=True, metavar="DATASET:METHOD=MASK_DIR")
    parser.add_argument("--reference", required=True, help="Method used for paired comparisons")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, default=Path("exp_results/statistics"))
    args = parser.parse_args()
    datasets = [parse_method(item) for item in args.dataset]
    rng = np.random.default_rng(args.seed)

    patient_rows = []
    prediction_specs = defaultdict(list)
    for dataset, method, directory in args.prediction:
        prediction_specs[dataset].append((method, directory))

    for dataset, root in datasets:
        mask_paths = index_pngs(root / "masks")
        if not prediction_specs[dataset]:
            raise ValueError(f"No --prediction entries were supplied for dataset {dataset!r}")
        for method, directory in prediction_specs[dataset]:
            patient_rows.extend(patient_metrics(mask_paths, index_pngs(directory), dataset, method))
    write_csv(args.output_dir / "patient_metrics.csv", patient_rows)

    metrics = ["dice", "DT", "DS", "DN", "DT_presence_aware", "DS_presence_aware", "DN_presence_aware"]
    grouped = defaultdict(list)
    for row in patient_rows:
        for metric in metrics:
            grouped[(row["dataset"], row["method"], row["class"], metric)].append(row[metric])
    summary = []
    for key, values in sorted(grouped.items()):
        mean, low, high = bootstrap_ci(values, rng, args.bootstrap)
        summary.append(dict(zip(("dataset", "method", "class", "metric"), key), mean=mean, ci95_low=low, ci95_high=high, n_patients=np.isfinite(values).sum()))
    write_csv(args.output_dir / "bootstrap_summary.csv", summary)

    lookup = {(r["dataset"], r["method"], r["patient"], r["class"]): r for r in patient_rows}
    tests = []
    for dataset, _ in datasets:
        methods = sorted({row["method"] for row in patient_rows if row["dataset"] == dataset and row["method"] != args.reference})
        classes = sorted({row["class"] for row in patient_rows if row["dataset"] == dataset})
        for method in methods:
            for class_name in classes:
                patients = sorted({row["patient"] for row in patient_rows if row["dataset"] == dataset and row["method"] == method and row["class"] == class_name})
                for metric in metrics:
                    pairs = [(lookup.get((dataset, method, p, class_name)), lookup.get((dataset, args.reference, p, class_name))) for p in patients]
                    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                    effect, p_value, n = paired_sign_flip([a[metric] for a, _ in pairs], [b[metric] for _, b in pairs], rng, args.permutations)
                    tests.append({"dataset": dataset, "method": method, "reference": args.reference, "class": class_name, "metric": metric, "mean_paired_difference": effect, "p_value_two_sided": p_value, "n_patients": n})
    add_holm_adjustment(tests)
    write_csv(args.output_dir / "paired_tests.csv", tests)
    print(f"Wrote patient-level statistical results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
