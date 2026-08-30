"""Run the native SAM3 video model as the Table II video-propagation control.

This is intentionally separate from the saved framewise-mask controls because
native propagation requires the original cine images and Sam3VideoModel. It
uses the same three class prompts and detection threshold for every sequence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from exp_common import evaluate_predictions, index_pngs, save_prediction_set, write_csv


PROMPTS = ["right ventricle", "myocardium", "left ventricle"]
DEFAULT_MODEL_ID = "facebook/sam3"


def require_sam3_video():
    try:
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "This control requires a Transformers release containing "
            "Sam3VideoModel and Sam3VideoProcessor. Upgrade Transformers, then rerun."
        ) from error
    return torch, Sam3VideoModel, Sam3VideoProcessor


def resolve_model_source(model: str) -> tuple[Path | str, bool]:
    """Prefer a complete local Hugging Face snapshot and avoid Hub requests."""
    requested = Path(model).expanduser()
    if requested.exists():
        return requested.resolve(), True

    if model == DEFAULT_MODEL_ID:
        snapshot_root = (
            Path("~/.cache/huggingface/hub/models--facebook--sam3/snapshots")
            .expanduser()
        )
        if snapshot_root.exists():
            snapshots = sorted(
                (path for path in snapshot_root.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return snapshots[0].resolve(), True

    return model, False


def merge_outputs(outputs, height, width):
    object_ids = outputs["object_ids"].detach().cpu().tolist()
    scores = outputs["scores"].detach().float().cpu().tolist()
    masks = outputs["masks"].detach().cpu().numpy().astype(bool)
    id_to_index = {int(object_id): index for index, object_id in enumerate(object_ids)}
    channels = np.zeros((height, width, len(PROMPTS)), dtype=np.float32)

    for class_index, prompt in enumerate(PROMPTS):
        prompt_ids = outputs["prompt_to_obj_ids"].get(prompt, [])
        if hasattr(prompt_ids, "detach"):
            prompt_ids = prompt_ids.detach().cpu().tolist()
        for object_id in prompt_ids:
            index = id_to_index.get(int(object_id))
            if index is not None:
                channels[..., class_index] = np.maximum(
                    channels[..., class_index], masks[index] * float(scores[index])
                )
    labels = np.argmax(channels, axis=-1).astype(np.uint8) + 1
    labels[np.max(channels, axis=-1) == 0] = 0
    return labels


def run_dataset(root, dataset, model, processor, torch, args):
    image_paths = index_pngs(root / "images")
    sequences = defaultdict(list)
    for patient, slice_id, frame_id in image_paths:
        sequences[(patient, slice_id)].append(frame_id)

    predictions = {}
    for (patient, slice_id), frame_ids in sorted(sequences.items()):
        frame_ids.sort()
        keys = [(patient, slice_id, frame_id) for frame_id in frame_ids]
        frames = [Image.open(image_paths[key]).convert("RGB") for key in keys]
        height, width = np.asarray(frames[0]).shape[:2]
        session = processor.init_video_session(
            video=frames,
            inference_device=args.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=getattr(torch, args.dtype),
        )
        session = processor.add_text_prompt(inference_session=session, text=PROMPTS)
        for raw_output in model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(frames) - 1,
            show_progress_bar=False,
        ):
            frame_index = int(raw_output.frame_idx)
            output = processor.postprocess_outputs(session, raw_output)
            predictions[keys[frame_index]] = merge_outputs(output, height, width)
        print(f"Finished {dataset} patient {patient:03d}, slice {slice_id:03d}")

    save_prediction_set(args.output_dir / "masks" / dataset / "sam3_native_video", predictions)
    return evaluate_predictions(predictions, root / "masks", dataset, "sam3_native_video")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("acdc_train"))
    parser.add_argument("--test-root", type=Path, default=Path("acdc_test"))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Local snapshot path or Hugging Face model ID; cached facebook/sam3 is preferred",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=Path("exp_results/sam3_native_video"))
    args = parser.parse_args()

    torch, model_class, processor_class = require_sam3_video()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_source, local_only = resolve_model_source(args.model)
    print(f"Loading SAM3 video model and processor from: {model_source}")
    load_kwargs = {"local_files_only": True} if local_only else {}
    model = model_class.from_pretrained(model_source, **load_kwargs).to(
        args.device, dtype=getattr(torch, args.dtype)
    )
    model.config.score_threshold_detection = args.threshold
    processor = processor_class.from_pretrained(model_source, **load_kwargs)

    image_rows, patient_rows = [], []
    for dataset, root in (("train", args.train_root), ("test", args.test_root)):
        current_images, current_patients = run_dataset(root, dataset, model, processor, torch, args)
        image_rows.extend(current_images)
        patient_rows.extend(current_patients)
    write_csv(args.output_dir / "image_dice.csv", image_rows)
    write_csv(args.output_dir / "patient_dice.csv", patient_rows)
    groups = defaultdict(list)
    for row in image_rows:
        groups[(row["dataset"], row["method"], row["class"])].append(row["dice"])
    summary = [
        {"dataset": dataset, "method": method, "class": class_name,
         "mean_dice": float(np.mean(values)), "n_images": len(values)}
        for (dataset, method, class_name), values in sorted(groups.items())
    ]
    write_csv(args.output_dir / "table_ii_summary.csv", summary)
    print(f"Wrote native SAM3 video results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
