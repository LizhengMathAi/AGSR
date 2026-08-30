"""Materialize matched baseline and no-flow proposed masks for statistics.

The proposed condition uses the manuscript's automatic confidence/area anchor,
anatomy-aware anchor cleaning, and bidirectional autoregressive propagation.
The optional TV-L1 warping is deliberately omitted because the manuscript
permits this efficient form and its missing results do not require flow here.
No annotation is used to choose the anchor.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from denoise import _get_reference_frame, _relabel_by_myo, denoise_sequence_with_anchor
from exp_common import index_pngs, load_rgb, one_hot_to_labels, save_prediction_set, scores_to_labels
from exp_temporal_controls import select_anchor


def generate(root, dataset, args):
    score_paths = index_pngs(root / "pseudo_soft_masks")
    sequences = defaultdict(list)
    for patient, slice_id, frame_id in score_paths:
        sequences[(patient, slice_id)].append(frame_id)

    reference_threshold = 255 * np.array([[[[0.15, 0.4, 0.05]]]])
    relabel_threshold = 255 * np.array([[[0.15, 0.4, 0.05]]])
    for (patient, slice_id), frame_ids in sorted(sequences.items()):
        frame_ids.sort()
        keys = [(patient, slice_id, frame_id) for frame_id in frame_ids]
        rgb = np.stack([load_rgb(score_paths[key]) for key in keys])
        labels = [scores_to_labels(frame, args.threshold) for frame in rgb]
        noisy = np.stack([
            np.stack([(mask == label).astype(np.uint8) for label in range(1, 4)], axis=-1) * 255
            for mask in labels
        ])
        anchor_index = select_anchor(rgb, args.threshold)
        reference = _get_reference_frame(rgb, threshold=reference_threshold)
        clean_anchor = _relabel_by_myo(
            reference, rgb[anchor_index], max_size=args.max_size, threshold=relabel_threshold,
        )
        # rho=1 returns the anatomy-cleaned target without calling TV-L1.
        refined = denoise_sequence_with_anchor(
            noisy, clean_anchor, anchor_index, rho=1.0, max_size=args.max_size,
        )
        baseline, proposed = {}, {}
        for key, base_mask, refined_mask in zip(keys, labels, refined):
            baseline[key] = base_mask
            proposed[key] = one_hot_to_labels(refined_mask > 0)
        # Checkpoint every sequence and keep memory bounded.
        save_prediction_set(args.output_dir / "masks" / dataset / "sam3_matched", baseline)
        save_prediction_set(args.output_dir / "masks" / dataset / "proposed", proposed)
        print(f"Finished {dataset} patient {patient:03d}, slice {slice_id:03d}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("acdc_train"))
    parser.add_argument("--test-root", type=Path, default=Path("acdc_test"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--rho", type=float, default=1.0,
        help="Deprecated compatibility option; no-flow refinement always uses rho=1",
    )
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("exp_results/refinements"))
    args = parser.parse_args()
    if args.rho != 1.0:
        print(f"Ignoring --rho {args.rho:g}; this experiment intentionally omits TV-L1", flush=True)
    for dataset, root in (("train", args.train_root), ("test", args.test_root)):
        generate(root, dataset, args)
    print(f"Wrote matched baseline and proposed masks to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
