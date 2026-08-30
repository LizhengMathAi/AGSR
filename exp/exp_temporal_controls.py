"""Matched temporal-voting and optional flow-only controls for ACDC Table II.

Both controls start from the same SAM3 RGB confidence masks and threshold.
Temporal voting uses a centered window. Flow-only propagation selects the
same automatic confidence/area anchor per slice and transports that anchor
bidirectionally with TV-L1 flow computed from adjacent cine images. It does
not apply anatomy/topology correction. Because TV-L1 is expensive, temporal
voting runs by default; pass --include-flow-only for the Table II flow row.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp_common import (
    CLASSES,
    evaluate_predictions,
    index_pngs,
    load_rgb,
    save_prediction_set,
    scores_to_labels,
    write_csv,
)


def temporal_vote(rgb_sequence: np.ndarray, threshold: float, radius: int) -> list[np.ndarray]:
    labels = [scores_to_labels(frame, threshold) for frame in rgb_sequence]
    output = []
    for center in range(len(labels)):
        start, stop = max(0, center - radius), min(len(labels), center + radius + 1)
        window = labels[start:stop]
        votes = np.stack([
            np.sum([mask == label for mask in window], axis=0)
            for label in CLASSES.values()
        ], axis=-1)
        winner = np.argmax(votes, axis=-1).astype(np.uint8) + 1
        winner[np.max(votes, axis=-1) < (len(window) // 2 + 1)] = 0
        output.append(winner)
    return output


def select_anchor(rgb_sequence: np.ndarray, threshold: float) -> int:
    scores = rgb_sequence.astype(np.float32) / 255.0
    active = scores > threshold
    numerator = np.sum(scores * active, axis=(1, 2, 3))
    denominator = np.sum(active, axis=(1, 2, 3))
    quality = numerator / np.maximum(denominator, 1)
    quality[denominator == 0] = -np.inf
    return int(np.argmax(quality))


def warp_mask(reference_image: np.ndarray, target_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Transport a reference-frame label mask into target-frame coordinates."""
    from skimage.registration import optical_flow_tvl1
    from skimage.transform import warp

    flow_y, flow_x = optical_flow_tvl1(target_image, reference_image)
    yy, xx = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing="ij")
    coordinates = np.array([yy + flow_y, xx + flow_x])
    channels = []
    for label in CLASSES.values():
        transported = warp(
            (mask == label).astype(np.float32), coordinates,
            order=0, mode="constant", cval=0, preserve_range=True,
        )
        channels.append(transported > 0.5)
    channels = np.stack(channels, axis=-1)
    return np.where(np.any(channels, axis=-1), np.argmax(channels, axis=-1) + 1, 0).astype(np.uint8)


def flow_only(images: np.ndarray, rgb_sequence: np.ndarray, threshold: float) -> list[np.ndarray]:
    anchor = select_anchor(rgb_sequence, threshold)
    output = [None] * len(images)
    output[anchor] = scores_to_labels(rgb_sequence[anchor], threshold)
    gray = images.astype(np.float32) / 255.0
    for frame in range(anchor + 1, len(images)):
        output[frame] = warp_mask(gray[frame - 1], gray[frame], output[frame - 1])
    for frame in range(anchor - 1, -1, -1):
        output[frame] = warp_mask(gray[frame + 1], gray[frame], output[frame + 1])
    return output


def run_dataset(root: Path, dataset: str, args) -> tuple[list[dict], list[dict]]:
    score_paths = index_pngs(root / "pseudo_soft_masks")
    image_paths = index_pngs(root / "images")
    sequences = defaultdict(list)
    for patient, slice_id, frame_id in score_paths:
        sequences[(patient, slice_id)].append(frame_id)

    predictions_by_method = {"temporal_voting": {}}
    if args.include_flow_only:
        predictions_by_method["flow_only_bidirectional"] = {}
    for (patient, slice_id), frame_ids in sorted(sequences.items()):
        frame_ids.sort()
        keys = [(patient, slice_id, frame_id) for frame_id in frame_ids]
        rgb_sequence = np.stack([load_rgb(score_paths[key]) for key in keys])
        voted = temporal_vote(rgb_sequence, args.threshold, args.vote_radius)
        for key, mask in zip(keys, voted):
            predictions_by_method["temporal_voting"][key] = mask

        if args.include_flow_only:
            from PIL import Image

            missing_images = [key for key in keys if key not in image_paths]
            if missing_images:
                raise FileNotFoundError(f"Missing cine images for {missing_images[:3]}")
            images = np.stack([
                np.asarray(Image.open(image_paths[key]).convert("L"), dtype=np.uint8)
                for key in keys
            ])
            flowed = flow_only(images, rgb_sequence, args.threshold)
            for key, mask in zip(keys, flowed):
                predictions_by_method["flow_only_bidirectional"][key] = mask
        print(f"Finished {dataset} patient {patient:03d}, slice {slice_id:03d}", flush=True)

    image_rows, patient_rows = [], []
    for method, predictions in predictions_by_method.items():
        current_images, current_patients = evaluate_predictions(
            predictions, root / "masks", dataset, method,
        )
        image_rows.extend(current_images)
        patient_rows.extend(current_patients)
        if args.save_masks:
            save_prediction_set(args.output_dir / "masks" / dataset / method, predictions)
    return image_rows, patient_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("acdc_train"))
    parser.add_argument("--test-root", type=Path, default=Path("acdc_test"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--vote-radius", type=int, default=2, help="Radius 2 gives a five-frame window")
    parser.add_argument("--output-dir", type=Path, default=Path("exp_results/temporal_controls"))
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument(
        "--include-flow-only", action="store_true",
        help="also run the slow TV-L1 flow-only bidirectional control",
    )
    args = parser.parse_args()

    image_rows, patient_rows = [], []
    for dataset, root in (("train", args.train_root), ("test", args.test_root)):
        current_images, current_patients = run_dataset(root, dataset, args)
        image_rows.extend(current_images)
        patient_rows.extend(current_patients)

    write_csv(args.output_dir / "image_dice.csv", image_rows)
    write_csv(args.output_dir / "patient_dice.csv", patient_rows)
    summary = []
    groups = defaultdict(list)
    for row in image_rows:
        groups[(row["dataset"], row["method"], row["class"])].append(row["dice"])
    for (dataset, method, class_name), values in sorted(groups.items()):
        summary.append({"dataset": dataset, "method": method, "class": class_name, "mean_dice": np.mean(values), "n_images": len(values)})
    write_csv(args.output_dir / "table_ii_summary.csv", summary)
    print(f"Wrote matched-control results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
