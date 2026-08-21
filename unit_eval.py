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


def iou_binary(first, second):
    first = first.astype(bool)
    second = second.astype(bool)

    union = np.logical_or(first, second).sum()

    # Treat two empty masks as a perfect temporal match.
    if union == 0:
        return 1.0

    intersection = np.logical_and(first, second).sum()
    return intersection / union


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


def compute_neighbor_ious(args, patient_id):
    """Compute class-wise IoU between consecutive predicted frames per slice."""
    gt_dir = Path(args.root_dir) / "masks"
    pred_dir = Path(args.root_dir) / "pseudo_soft_masks"

    all_pred_paths = sorted(pred_dir.glob(f"patient{patient_id:03d}_frame*_z*.png"))
    slice_ids = sorted(set(int(p.name.split("_")[2][1:-4]) for p in all_pred_paths))

    total_ious = {
        "RV": [],
        "MYO": [],
        "LV": [],
        "foreground": [],
    }
    processed_slice_ids = []

    for slice_id in slice_ids:
        pred_paths = sorted(pred_dir.glob(f"patient{patient_id:03d}_frame*_z{slice_id:03d}.png"))
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

        slice_ious = {key: [] for key in total_ious}
        for first, second in zip(pred_seq[:-1], pred_seq[1:]):
            slice_ious["RV"].append(iou_binary(first == 1, second == 1))
            slice_ious["MYO"].append(iou_binary(first == 2, second == 2))
            slice_ious["LV"].append(iou_binary(first == 3, second == 3))
            slice_ious["foreground"].append(iou_binary(first > 0, second > 0))

        for key in total_ious:
            total_ious[key].append(slice_ious[key])
        processed_slice_ids.append(slice_id)

    flat_ious = {
        key: [value for slice_values in values for value in slice_values]
        for key, values in total_ious.items()
    }
    mean_ious = {
        key: np.mean(values) if values else np.nan
        for key, values in flat_ious.items()
    }
    print(
        "Overall Mean Neighbor IoUs: "
        f"Foreground = {mean_ious['foreground']:.4f}, "
        f"RV = {mean_ious['RV']:.4f}, "
        f"MYO = {mean_ious['MYO']:.4f}, "
        f"LV = {mean_ious['LV']:.4f}"
    )

    out_file = Path("./neighbor_ious.txt").resolve()
    print(f"Saving neighbor IoUs to: {out_file}")
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"{args}, patient_id={patient_id}\n")
        f.write(
            "Overall Mean Neighbor IoUs: "
            f"Foreground = {mean_ious['foreground']:.4f}, "
            f"RV = {mean_ious['RV']:.4f}, "
            f"MYO = {mean_ious['MYO']:.4f}, "
            f"LV = {mean_ious['LV']:.4f}\n"
        )
        for slice_index, slice_id in enumerate(processed_slice_ids):
            f.write(f"slice={slice_id}\n")
            for key in ["RV", "MYO", "LV", "foreground"]:
                values = ", ".join(f"{value:.4f}" for value in total_ious[key][slice_index])
                f.write(f"{key} = [{values}]\n")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "self_derived_prompts", "annotation_derived_prompts"], help="Evaluation mode")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold for binarizing predictions")
    parser.add_argument("--operation", choices=["dice", "areas", "neighbor_iou"], default="dice", help="Computation to run")
    parser.add_argument("--patient-id", type=int, default=1, help="Patient used for area or neighbor-IoU computation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.operation == "dice":
        compute_dice_scores(args)
    elif args.operation == "neighbor_iou":
        compute_neighbor_ious(args, patient_id=args.patient_id)
    else:
        compute_areas(args, patient_id=args.patient_id)

    # NOTE: compute the dice, area, and neighbor_iou respectivily.
    # python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0
    # python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0 --operation areas --patient-id 1
    # python unit_eval.py --root-dir ./acdc_train --mode baseline --thres 0.0 --operation neighbor_iou --patient-id 1
