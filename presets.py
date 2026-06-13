"""
ACDC flat PNG exporter.

Creates this output layout:

acdc_train/
├── images/
│   └── patientxxx_framexx_zxxx.png
├── masks/
│   └── patientxxx_framexx_zxxx.png
└── classification_images.csv

By default, images are exported for every temporal frame from:
  patientxxx_4d.nii.gz

Masks are exported only when a matching annotation exists:
  patientxxx_frameXX_gt.nii.gz

If the 4D cine volume is not present, the script falls back to any available
patientxxx_frameXX.nii.gz files.

Mask labels are preserved as integer class IDs:
  0 background, 1 right ventricle, 2 myocardium, 3 left ventricle

Example:
  python presets.py --raw_dir /home/cs/Documents/datasets/ACDC/training --out_dir acdc_train --image_size 224
  python presets.py --raw_dir /home/cs/Documents/datasets/ACDC/testing --out_dir acdc_test --image_size 224
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms as T
from transformers import Sam3Processor, Sam3Model


ACDC_LABELS = ["NOR", "MINF", "DCM", "HCM", "RV"]
LABEL_ALIASES = {
    "NORMAL": "NOR",
    "N": "NOR",
    "NOR": "NOR",
    "MINF": "MINF",
    "MI": "MINF",
    "DCM": "DCM",
    "HCM": "HCM",
    "RV": "RV",
    "ARV": "RV",
}


def read_acdc_info(info_path: Path) -> Dict[str, str]:
    """Read ACDC Info.cfg as key/value strings."""
    info: Dict[str, str] = {}
    if not info_path.exists():
        return info

    for line in info_path.read_text(errors="ignore").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def normalize_label(label: object) -> str:
    text = "" if label is None else str(label).strip().upper()
    return LABEL_ALIASES.get(text, text)


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def frame_number(path: Path) -> int:
    match = re.search(r"frame(\d+)", path.name)
    return int(match.group(1)) if match else -1


def patient_dirs(raw_dir: Path) -> List[Path]:
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith("patient")
    )


def read_volume(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).get_fdata())


def squeeze_volume(arr: np.ndarray) -> np.ndarray:
    return np.squeeze(np.asarray(arr))


def iter_slices(volume: np.ndarray) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (z_index, 2D slice) from a 2D/3D NIfTI volume."""
    arr = squeeze_volume(volume)
    if arr.ndim == 2:
        yield 0, arr
    elif arr.ndim == 3:
        for z in range(arr.shape[2]):
            yield z, arr[:, :, z]
    else:
        raise ValueError(f"Expected a 2D or 3D volume, got shape {arr.shape}")


def get_mask_slice(mask_volume: np.ndarray, z: int) -> np.ndarray:
    mask = squeeze_volume(mask_volume)
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return mask[:, :, z]
    raise ValueError(f"Expected a 2D or 3D mask, got shape {mask.shape}")


def iter_frame_volumes(pdir: Path, export_all_frames: bool = True) -> Iterator[Tuple[int, Path, np.ndarray]]:
    """Yield (1-based frame_number, source_path, 2D/3D image volume).

    When export_all_frames=True and patientxxx_4d.nii.gz exists, every temporal
    frame from the 4D cine volume is yielded. Otherwise, this falls back to the
    individual patientxxx_frameXX.nii.gz files that are present on disk.
    """
    patient_id = pdir.name
    cine_4d = pdir / f"{patient_id}_4d.nii.gz"

    if export_all_frames and cine_4d.exists():
        cine = squeeze_volume(read_volume(cine_4d))
        if cine.ndim == 3:
            # Rare fallback: a single 3D frame stored under the 4D name.
            yield 1, cine_4d, cine
            return
        if cine.ndim != 4:
            raise ValueError(f"Expected a 4D cine volume at {cine_4d}, got shape {cine.shape}")

        for t in range(cine.shape[3]):
            yield t + 1, cine_4d, cine[:, :, :, t]
        return

    image_files = sorted(
        [f for f in pdir.glob("*_frame*.nii.gz") if "_gt" not in f.name],
        key=frame_number,
    )
    for image_file in image_files:
        yield frame_number(image_file), image_file, read_volume(image_file)


def normalize01(x: np.ndarray, percentile_clip: Tuple[float, float] = (1, 99)) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(x[finite], percentile_clip)
    x = np.clip(x, lo, hi)
    denom = float(x.max() - x.min())
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x.min()) / (denom + 1e-8)


def save_image_png(image_slice: np.ndarray, out_path: Path, image_size: int) -> None:
    arr = normalize01(image_slice)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img.save(out_path)
    return img


def save_mask_png(mask_slice: np.ndarray, out_path: Path, image_size: int) -> None:
    mask = np.asarray(mask_slice, dtype=np.uint8)
    img = Image.fromarray(mask, mode="L")
    img = img.resize((image_size, image_size), Image.NEAREST)
    img.save(out_path)


def save_pseudo_mask_png(img_path: Path, out_path: Path, model, processor, threshold: float, mask_threshold: float = 0.25) -> None:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    mask = []

    for i, text_prompt in enumerate(["right ventricle", "myocardium", "left ventricle"]):
        input = processor(images=img, text=text_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**input)
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=input.get("original_sizes").tolist()
        )[0]

        if result["scores"].numel() != 0:
            scores = torch.mean(result["scores"][:, None, None] * result["masks"], dim=0)
            counts = torch.clamp(torch.sum(result["masks"], dim=0), min=1.0)
            mask.append(scores / counts)
        else:
            mask.append(torch.zeros((h, w), dtype=torch.float32, device=model.device))

    mask = torch.stack(mask, dim=0).permute(1, 2, 0)  # (H, W, C)
    mask = (255 * mask.cpu().numpy()).astype(np.uint8)
    Image.fromarray(mask).save(out_path)


def prepare_acdc_flat(
    raw_dir: Path,
    out_dir: Path,
    image_size: int = 224,
    keep_empty_masks: bool = True,
    overwrite: bool = False,
    export_all_frames: bool = True,
) -> None:
    """Export ACDC 2D slices to a flat layout.

    Images are written for all frames when patientxxx_4d.nii.gz is available.
    Masks are written only for frames that have patientxxx_frameXX_gt.nii.gz.
    """
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    pseudo_soft_dir = out_dir / "pseudo_soft_masks"
    pseudo_hard_dir = out_dir / "pseudo_hard_masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    pseudo_soft_dir.mkdir(parents=True, exist_ok=True)
    pseudo_hard_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path("~/.cache/huggingface/hub/models--facebook--sam3/snapshots").expanduser()
    if model_path.exists():
        model_path = next(model_path.iterdir())  # Get the first (and only) subdirectory
        model = Sam3Model.from_pretrained(model_path).to(device)
        processor = Sam3Processor.from_pretrained(model_path)
    else:
        model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")

    rows = []

    for pdir in tqdm(patient_dirs(raw_dir), desc="Exporting ACDC slices"):
        info = read_acdc_info(pdir / "Info.cfg")
        patient_id = pdir.name
        group = normalize_label(info.get("Group", "UNKNOWN"))
        label_id = ACDC_LABELS.index(group) if group in ACDC_LABELS else -1
        height = safe_float(info.get("Height"))
        weight = safe_float(info.get("Weight"))
        ed_frame = safe_int(info.get("ED"))
        es_frame = safe_int(info.get("ES"))

        gt_files = {frame_number(f): f for f in pdir.glob("*_frame*_gt.nii.gz")}
        mask_volumes: Dict[int, np.ndarray] = {}

        for frame, image_file, image_volume in iter_frame_volumes(pdir, export_all_frames=export_all_frames):
            mask_file = gt_files.get(frame)
            if mask_file is not None and frame not in mask_volumes:
                mask_volumes[frame] = read_volume(mask_file).astype(np.uint8)
            mask_volume = mask_volumes.get(frame)

            for z, image_slice in iter_slices(image_volume):
                filename = f"{patient_id}_frame{frame:02d}_z{z:03d}.png"
                image_path = images_dir / filename
                mask_path = masks_dir / filename
                pseudo_soft_path = pseudo_soft_dir / filename
                pseudo_hard_path = pseudo_hard_dir / filename

                save_image_png(image_slice, image_path, image_size)
                save_pseudo_mask_png(image_path, pseudo_soft_path, model, processor, threshold=0.25, mask_threshold=0.25)
                save_pseudo_mask_png(image_path, pseudo_hard_path, model, processor, threshold=0.60, mask_threshold=0.25)

                mask_available = mask_volume is not None
                has_foreground = False
                mask_path_text = ""

                if mask_available:
                    mask_slice = get_mask_slice(mask_volume, z)
                    has_foreground = bool(np.max(mask_slice) > 0)
                    if keep_empty_masks or has_foreground:
                        save_mask_png(mask_slice, mask_path, image_size)
                        mask_path_text = str(mask_path)

                rows.append(
                    {
                        "patient_id": patient_id,
                        "group": group,
                        "label_id": label_id,
                        "height": height,
                        "weight": weight,
                        "ed_frame": ed_frame,
                        "es_frame": es_frame,
                        "frame": frame,
                        "slice": z,
                        "filename": filename,
                        "image_path": str(image_path),
                        "mask_path": mask_path_text,
                        "mask_available": mask_available,
                        "has_foreground": has_foreground,
                    }
                )

    csv_path = out_dir / "classification_images.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    mask_count = int(pd.DataFrame(rows)["mask_path"].astype(bool).sum()) if rows else 0
    print(f"Saved {len(rows)} image slices")
    print(f"Saved {mask_count} mask slices")
    print(f"Images: {images_dir}")
    print(f"Masks:  {masks_dir}")
    print(f"CSV:    {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ACDC labeled slices to acdc_train/images, acdc_train/masks, and classification_images.csv."
    )
    parser.add_argument("--raw_dir", type=Path, required=True, help="Path to raw ACDC training folder.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output folder, for example acdc_train.")
    parser.add_argument("--image_size", type=int, default=224, help="Output PNG width/height.")
    parser.add_argument("--drop_empty_masks", action="store_true", help="Do not write mask PNGs where the annotation is all background. Image PNGs are still written.")
    parser.add_argument("--annotated_frames_only", action="store_true", help="Only export patientxxx_frameXX.nii.gz files instead of every frame from patientxxx_4d.nii.gz.")
    parser.add_argument("--overwrite", action="store_true", help="Delete out_dir before exporting.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_acdc_flat(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        image_size=args.image_size,
        keep_empty_masks=not args.drop_empty_masks,
        overwrite=args.overwrite,
        export_all_frames=not args.annotated_frames_only,
    )


def unit_test() -> None:
    img_path = "/home/cs/Documents/BHI/acdc_train/images/patient001_frame01_z001.png"
    mask_path = "/home/cs/Documents/BHI/acdc_train/pseudo_masks/patient001_frame01_z001.png"

    image = Image.open(img_path).convert("RGBA")
    mask = np.array(Image.open(mask_path))
    print("Unique values in mask:", np.unique(mask))

    colors = {
        1: (255, 0, 0),    # RV
        2: (0, 255, 0),    # MYO
        3: (0, 0, 255),    # LV
    }

    for cls, color in colors.items():
        binary_mask = (mask == cls).astype(np.uint8) * 255

        binary_mask = Image.fromarray(binary_mask)

        overlay = Image.new("RGBA", image.size, color + (0,))

        # 30% opacity
        alpha = binary_mask.point(lambda v: int(v * 0.3))

        overlay.putalpha(alpha)

        image = Image.alpha_composite(image, overlay)

    image.show()



if __name__ == "__main__":
    main()