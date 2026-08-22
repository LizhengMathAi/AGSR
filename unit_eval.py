from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import numpy as np

from denoise import _rgb2mask, _get_reference_frame, _relabel_by_myo


ACDC_CLASSES = {
    "RV": 1,
    "MYO": 2,
    "LV": 3,
}

def dice_binary(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()

    # NOTE: perfect score if both are empty
    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom


def acdc_mean_dice(gt, pred):
    """
    gt: (H, W) uint8, values in {0, 1, 2, 3}
    pred: (H, W) uint8, values in {0, 1, 2, 3}
    """
    scores = {}

    for name, label in ACDC_CLASSES.items():
        scores[name] = dice_binary(gt == label, pred == label)

    if gt.sum() > 0:
        scores["foreground"] = dice_binary(gt > 0, pred > 0)
    else:
        scores["foreground"] = 1.0

    return scores


def compute_dice_scores(args):
    gt_dir = Path(args.root_dir) / "masks"
    pred_dir = Path(args.root_dir) / "pseudo_soft_masks"

    total_scores = {
        "RV": [],
        "MYO": [],
        "LV": [],
        "foreground": [],
    }
    for gt_file in sorted(gt_dir.glob("*.png")):

        pred_file = pred_dir / gt_file.name

        if not pred_file.exists():
            print(f"Missing prediction: {gt_file.name}")
            continue

        gt = np.array(Image.open(gt_file)).astype(np.uint8)
        gt_rgb = np.stack([(gt == label).astype(np.uint8) for label in range(1, 4)], axis=-1) * 255

        if args.mode == "baseline":
            pred_rgb = np.array(Image.open(pred_file)).astype(np.uint8)
            pred = _rgb2mask(pred_rgb, thres=np.array([args.thres]*3))
        elif args.mode == "self_derived_prompts":
            patient_id = int(gt_file.name.split("_")[0][7:])
            frame_id = int(gt_file.name.split("_")[1][5:])
            slice_id = int(gt_file.name.split("_")[2][1:-4])
            ref_paths, stride = sorted(pred_dir.glob(f"patient{patient_id:03d}_frame*_z{slice_id:03d}.png")), 5
            ref_paths = ref_paths[max(frame_id-1-stride, 0):min(frame_id+stride, len(ref_paths))]
            ref_seq = np.stack([np.array(Image.open(p)).astype(np.uint8) for p in ref_paths])  # [T, H, W, 3]
            ref_rgb = _get_reference_frame(ref_seq, threshold=255*np.array([[[[0.15, 0.4, 0.05]]]]))  # [H, W, 3]

            pred_rgb = np.array(Image.open(pred_file)).astype(np.uint8)
            pred_rgb = _relabel_by_myo(ref_rgb, pred_rgb, max_size=8, threshold=255*np.array([[[0.15, 0.4, 0.05]]]))
            pred = _rgb2mask(pred_rgb, thres=np.array([0, 0, 0]))
        elif args.mode == "annotation_derived_prompts":
            pred_rgb = np.array(Image.open(pred_file)).astype(np.uint8)
            pred_rgb = _relabel_by_myo(gt_rgb, pred_rgb, max_size=8, threshold=255*np.array([[[0.05, 0.97, 0.5]]]))
            pred = _rgb2mask(pred_rgb, thres=np.array([0, 0, 0]))

        scores = acdc_mean_dice(gt, pred)
        for key in total_scores:
            total_scores[key].append(scores[key])
    score = {key: np.mean(values) for key, values in total_scores.items()}
    print(f"Overall Mean Dice Scores: Foreground = {score['foreground']:.4f}, RV = {score['RV']:.4f}, MYO = {score['MYO']:.4f}, LV = {score['LV']:.4f}")

    # write the all scores to a text file, if file may already existed, continue write it.
    out_file = Path("./dice_scores.txt").resolve()
    print(f"Saving dice scores to: {out_file}, RV Count = {len(total_scores['RV'])}, MYO Count = {len(total_scores['MYO'])}, LV Count = {len(total_scores['LV'])}")

    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"{args}\n")

        line = f"Overall Mean Dice Scores: Foreground = {score['foreground']:.4f}, RV = {score['RV']:.4f}, MYO = {score['MYO']:.4f}, LV = {score['LV']:.4f}\n"
        f.write(line)

        for key, values in total_scores.items():
            values = [f"{v:.2f}" for v in values if v > 0]
            line = f"{key} = [{', '.join(values)}]\n"
            f.write(line)


def compute_areas(args, patient_id):
    gt_dir = Path(args.root_dir) / "masks"
    pred_dir = Path(args.root_dir) / "pseudo_soft_masks"

    pred_paths = sorted(pred_dir.glob(f"patient{patient_id:03d}_frame*_z*.png"))
    slice_ids = sorted(set([int(p.name.split("_")[2][1:-4]) for p in pred_paths]))

    total_areas = {
        "RV": [],
        "MYO": [],
        "LV": []
    }
    for slice_id in slice_ids:
        pred_paths = sorted(pred_dir.glob(f"patient{patient_id:03d}_frame*_z{slice_id:03d}.png"))
        pred_rgb_seq = np.stack([np.array(Image.open(pred_file)).astype(np.uint8) for pred_file in pred_paths])  # [T, H, W, 3]

        if args.mode == "baseline":
            pred_seq = [_rgb2mask(pred_rgb, thres=np.array([args.thres]*3)) for pred_rgb in pred_rgb_seq]
        elif args.mode == "self_derived_prompts":
            pred_seq, stride = [], 5

            for frame_id, pred_rgb in enumerate(pred_rgb_seq):
                ref_seq = pred_rgb_seq[max(frame_id-1-stride, 0):min(frame_id+stride, len(pred_paths))]
                ref_rgb = _get_reference_frame(ref_seq, threshold=255*np.array([[[[0.15, 0.4, 0.05]]]]))  # [H, W, 3]

                pred_rgb = _relabel_by_myo(ref_rgb, pred_rgb, max_size=8, threshold=255*np.array([[[0.15, 0.4, 0.05]]]))
                pred_seq.append(_rgb2mask(pred_rgb, thres=np.array([0, 0, 0])))
        elif args.mode == "annotation_derived_prompts":
            gt_path = sorted(gt_dir.glob(f"patient{patient_id:03d}_frame*_z{slice_id:03d}.png"))[0]
            gt = np.array(Image.open(gt_path)).astype(np.uint8)
            gt_rgb = np.stack([(gt == label).astype(np.uint8) for label in range(1, 4)], axis=-1) * 255

            pred_seq = []
            for pred_rgb in pred_rgb_seq:
                pred_rgb = _relabel_by_myo(gt_rgb, pred_rgb, max_size=8, threshold=255*np.array([[[0.05, 0.97, 0.5]]]))
                pred_seq.append(_rgb2mask(pred_rgb, thres=np.array([0, 0, 0])))


        total_areas["RV"].append([(p == 1).sum().item() for p in pred_seq])
        total_areas["MYO"].append([(p == 2).sum().item() for p in pred_seq])
        total_areas["LV"].append([(p == 3).sum().item() for p in pred_seq])

    out_file = Path("./areas.txt").resolve()
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"{args}, patient_id={patient_id}\n")

        for key in ["RV", "MYO", "LV"]:
            array = total_areas[key]
            for vec in array:
                values = [f"{v}" for v in vec]
                line = f"{key} = [{', '.join(values)}]\n"
                f.write(line)


def compute_neighbor_dices(args, patient_id):
    """Compute four-neighbor spatial-temporal Dice for interior predictions.

    For a prediction at (z, t), average its Dice against the masks at
    (z - 1, t), (z + 1, t), (z, t - 1), and (z, t + 1).
    """
    gt_dir = Path(args.root_dir) / "masks"
    pred_dir = Path(args.root_dir) / "pseudo_soft_masks"

    all_pred_paths = sorted(pred_dir.glob(f"patient{patient_id:03d}_frame*_z*.png"))
    if not all_pred_paths:
        raise FileNotFoundError(f"No predictions found for patient {patient_id}")

    def path_key(path):
        parts = path.stem.split("_")
        return int(parts[2][1:]), int(parts[1][5:])  # (slice_id, frame_id)

    pred_paths_by_key = {path_key(path): path for path in all_pred_paths}
    slice_ids = sorted({key[0] for key in pred_paths_by_key})
    pred_masks = {}

    for slice_id in slice_ids:
        pred_paths = sorted(
            (path for key, path in pred_paths_by_key.items() if key[0] == slice_id),
            key=lambda path: path_key(path)[1],
        )
        frame_ids = [path_key(path)[1] for path in pred_paths]
        pred_rgb_seq = np.stack([
            np.array(Image.open(pred_file)).astype(np.uint8)
            for pred_file in pred_paths
        ])

        if args.mode == "baseline":
            pred_seq = [
                _rgb2mask(pred_rgb, thres=np.array([args.thres] * 3))
                for pred_rgb in pred_rgb_seq
            ]
        elif args.mode == "self_derived_prompts":
            pred_seq, stride = [], 5
            for frame_id, pred_rgb in enumerate(pred_rgb_seq):
                ref_seq = pred_rgb_seq[
                    max(frame_id - 1 - stride, 0):min(frame_id + stride, len(pred_paths))
                ]
                ref_rgb = _get_reference_frame(
                    ref_seq,
                    threshold=255 * np.array([[[[0.15, 0.4, 0.05]]]]),
                )
                pred_rgb = _relabel_by_myo(
                    ref_rgb,
                    pred_rgb,
                    max_size=8,
                    threshold=255 * np.array([[[0.15, 0.4, 0.05]]]),
                )
                pred_seq.append(_rgb2mask(pred_rgb, thres=np.array([0, 0, 0])))
        elif args.mode == "annotation_derived_prompts":
            gt_paths = sorted(gt_dir.glob(f"patient{patient_id:03d}_frame*_z{slice_id:03d}.png"))
            if not gt_paths:
                print(f"Missing annotation for patient {patient_id}, slice {slice_id}; skipping")
                continue

            gt = np.array(Image.open(gt_paths[0])).astype(np.uint8)
            gt_rgb = np.stack([
                (gt == label).astype(np.uint8) for label in range(1, 4)
            ], axis=-1) * 255

            pred_seq = []
            for pred_rgb in pred_rgb_seq:
                pred_rgb = _relabel_by_myo(
                    gt_rgb,
                    pred_rgb,
                    max_size=8,
                    threshold=255 * np.array([[[0.05, 0.97, 0.5]]]),
                )
                pred_seq.append(_rgb2mask(pred_rgb, thres=np.array([0, 0, 0])))

        pred_masks.update({(slice_id, frame_id): mask for frame_id, mask in zip(frame_ids, pred_seq)})

    class_specs = {
        "RV": lambda mask: mask == 1,
        "MYO": lambda mask: mask == 2,
        "LV": lambda mask: mask == 3,
        "foreground": lambda mask: mask > 0,
    }
    results_by_slice = {}

    for slice_id in range(1, 9):
        slice_dices = {name: [] for name in class_specs}

        for frame_id in range(2, 30):
            current_key = (slice_id, frame_id)
            neighbor_keys = [
                (slice_id - 1, frame_id),
                (slice_id + 1, frame_id),
                (slice_id, frame_id - 1),
                (slice_id, frame_id + 1),
            ]
            required_keys = [current_key, *neighbor_keys]
            missing_keys = [key for key in required_keys if key not in pred_masks]
            if missing_keys:
                raise FileNotFoundError(
                    f"Missing predictions required for center {current_key}: {missing_keys}"
                )

            current = pred_masks[current_key]
            for name, select in class_specs.items():
                neighbor_scores = [
                    dice_binary(select(current), select(pred_masks[neighbor_key]))
                    for neighbor_key in neighbor_keys
                ]
                slice_dices[name].append(float(np.mean(neighbor_scores)))

        results_by_slice[slice_id] = slice_dices

    blocks = []
    for slice_id, results in results_by_slice.items():
        rows = [f"z{slice_id:03d}:"]
        for key in ["RV", "MYO", "LV", "foreground"]:
            values = ", ".join(f"{value:.4f}" for value in results[key])
            rows.append(f"{key}=[{values}]")
        blocks.append("\n".join(rows))

    report = "\n".join(blocks) + "\n"
    print(report, end="")

    out_file = Path("./neighbor_dices.txt").resolve()
    print(f"Saving neighbor Dice to: {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "self_derived_prompts", "annotation_derived_prompts"], help="Evaluation mode")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold for binarizing predictions")
    parser.add_argument("--operation", choices=["dice", "areas", "neighbor_dice"], default="dice", help="Computation to run")
    parser.add_argument("--patient-id", type=int, default=1, help="Patient used for area or neighbor-Dice computation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.operation == "dice":
        compute_dice_scores(args)
    elif args.operation == "neighbor_dice":
        compute_neighbor_dices(args, patient_id=args.patient_id)
    else:
        compute_areas(args, patient_id=args.patient_id)

    # NOTE: compute the dice, area, and neighbor_dice respectively.
    # python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0
    # python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0 --operation areas --patient-id 1
    # python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0 --operation neighbor_dice --patient-id 1
