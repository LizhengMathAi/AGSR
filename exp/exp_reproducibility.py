"""Create a reproducibility manifest without decoding any image pixels."""

import argparse
import json
from collections import Counter
from pathlib import Path

from exp_common import index_pngs


def audit(name, root):
    directories = {item.name: item for item in root.iterdir() if item.is_dir()}
    result = {"name": name, "root": str(root.resolve()), "directories": sorted(directories)}
    for folder in ("images", "masks", "pseudo_soft_masks", "pseudo_hard_masks"):
        directory = directories.get(folder)
        if directory is None:
            result[folder] = {"present": False}
            continue
        keys = index_pngs(directory)
        patients = Counter(key[0] for key in keys)
        result[folder] = {
            "present": True, "files": len(keys), "patients": len(patients),
            "patient_ids": sorted(patients), "files_per_patient": dict(sorted(patients.items())),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("acdc_train"))
    parser.add_argument("--test-root", type=Path, default=Path("acdc_test"))
    parser.add_argument("--output", type=Path, default=Path("exp_results/reproducibility_manifest.json"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--vote-radius", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    manifest = {
        "datasets": [audit("train", args.train_root), audit("test", args.test_root)],
        "protocol": {
            "prompts": ["right ventricle", "myocardium", "left ventricle"],
            "matched_threshold": args.threshold,
            "proposed_warping": "omitted; anatomy correction and autoregressive guidance only",
            "flow_only_control": "TV-L1; opt in with --include-flow-only",
            "small_component_pixels": args.max_size, "temporal_vote_radius": args.vote_radius,
            "bootstrap_resamples": args.bootstrap, "random_seed": args.seed,
            "statistical_unit": "patient", "ci": "two-sided percentile bootstrap 95%",
            "paired_test": "two-sided paired sign-flip permutation test",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote reproducibility manifest to {args.output.resolve()}")


if __name__ == "__main__":
    main()
