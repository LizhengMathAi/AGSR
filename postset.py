from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

import denoise


# IMAGE_DIR = Path("./acdc_train/images")
# MASK_DIR = Path("./acdc_train/masks")
# PSEUDO_SOFT_DIR = Path("./acdc_train/pseudo_soft_masks")
# PSEUDO_OPT_DIR = Path("./acdc_train/pseudo_opt_masks")


def load(path, mode=None, dtype=None):
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    arr = np.array(img)
    return arr.astype(dtype) if dtype else arr


def one_hot(mask):
    return np.stack([mask == c for c in range(1, 4)], axis=-1).astype(np.float32)


def one_hot_to_label(mask):
    labels = np.argmax(mask, axis=-1).astype(np.uint8) + 1
    labels[~np.any(mask > 0, axis=-1)] = 0
    return labels


def generate_optimized_pseudo_masks(root_dir, patient_id, slice_id, rho, max_size):
    pattern = f"patient{patient_id:03d}_frame*_z{slice_id:03d}.png"

    image_paths = sorted((Path(root_dir) / "images").glob(pattern))
    mask_paths = sorted((Path(root_dir) / "masks").glob(pattern))
    opt_dir = Path(root_dir) / "pseudo_opt_masks" / f"rho_{rho:.2f}" / f"max_size_{int(max_size)}"

    for anchor_idx in range(len(mask_paths)):
        anchor = one_hot(load(mask_paths[anchor_idx]))
        if anchor.sum() != 0:
            break
    if anchor.sum() == 0:
        print(f"No non-empty anchor mask found for pattern: {pattern}")
        return

    soft_seq = np.stack([
        load(Path(root_dir) / "pseudo_soft_masks" / p.name, dtype=np.float32)
        for p in image_paths
    ])

    labels = np.argmax(soft_seq, axis=-1)
    fg = np.any(soft_seq > 0, axis=-1)

    noisy = np.zeros_like(soft_seq)
    noisy[fg, labels[fg]] = 1.0

    clean_seq = denoise.denoise_sequence_with_anchor(
        noisy_seq=noisy,
        clean_anchor_frame=anchor,
        anchor_idx=anchor_idx,
        rho=rho,
        max_size=max_size,
    )

    opt_dir.mkdir(parents=True, exist_ok=True)

    for frame, path in zip(clean_seq, image_paths):
        Image.fromarray(one_hot_to_label(frame)).save(opt_dir / path.name)


def prepare_optimized_pseudo_masks(root_dir, rho, max_size):
    patient_ids = [int(p.name.split("_")[0][7:]) for p in (Path(root_dir) / "images").glob("patient*_frame*_z*.png")]
    patient_ids = sorted(set(patient_ids))

    for patient_id in patient_ids:
        slice_ids = [int(p.name.split("_")[2][1:-4]) for p in (Path(root_dir) / "images").glob(f"patient{patient_id:03d}_frame*_z*.png")]
        slice_ids = sorted(set(slice_ids))

        for slice_id in slice_ids:
            generate_optimized_pseudo_masks(
                root_dir=root_dir,
                patient_id=patient_id,
                slice_id=slice_id,
                rho=rho,
                max_size=max_size,
            )
        print(f"Finished patient {patient_id:03d}")
    


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--max-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_optimized_pseudo_masks(
        root_dir=args.root_dir,
        rho=args.rho,
        max_size=args.max_size,
    )
    #  Example usage:
    #  python postset.py --root-dir ./acdc_train --rho 0.5 --max-size 8
    #  python postset.py --root-dir ./acdc_test --rho 0.5 --max-size 8