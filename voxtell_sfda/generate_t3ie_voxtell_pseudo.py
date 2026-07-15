#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.data import discover_image_paths
from voxtell_sfda.losses import binary_entropy
from voxtell_sfda.modeling import load_voxtell_predictor
from voxtell_sfda.nifti import write_reoriented_nifti
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping
from voxtell_sfda.t3ie import t3ie_views


def safe_name(prompt: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in prompt)


def get_device(device_name: str, gpu: int) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if device_name == "cuda":
        print("CUDA is unavailable; using CPU.", file=sys.stderr)
    return torch.device("cpu")


@torch.no_grad()
def predict_t3ie_probabilities(predictor, image_np: np.ndarray, prompts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data, bbox, orig_shape = predictor.preprocess(image_np)
    embeddings = predictor.embed_text_prompts(prompts)

    probs = []
    for view in t3ie_views(data):
        logits = predictor.predict_sliding_window_return_logits(view, embeddings).to("cpu").float()
        probs.append(torch.sigmoid(logits))

    stacked = torch.stack(probs, dim=0)
    mean_prob = stacked.mean(dim=0)
    consistency = stacked.std(dim=0)
    uncertainty = binary_entropy(mean_prob)

    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    mean_np = mean_prob.numpy().astype(np.float32, copy=False)
    consistency_np = consistency.numpy().astype(np.float32, copy=False)
    uncertainty_np = uncertainty.numpy().astype(np.float32, copy=False)

    prob_full = np.zeros((mean_np.shape[0], *orig_shape), dtype=np.float32)
    consistency_full = np.zeros_like(prob_full, dtype=np.float32)
    uncertainty_full = np.ones_like(prob_full, dtype=np.float32)
    prob_full = insert_crop_into_image(prob_full, mean_np, bbox)
    consistency_full = insert_crop_into_image(consistency_full, consistency_np, bbox)
    uncertainty_full = insert_crop_into_image(uncertainty_full, uncertainty_np, bbox)
    return prob_full, uncertainty_full, consistency_full


@torch.no_grad()
def save_t3ie_debug_outputs(
    predictor,
    image_np: np.ndarray,
    props: dict,
    prompts: list[str],
    pseudo_prob: np.ndarray,
    uncertainty: np.ndarray,
    consistency: np.ndarray,
    output_dir: Path,
) -> None:
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    data, bbox, orig_shape = predictor.preprocess(image_np)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = NibabelIOWithReorient()
    for view_index, view in enumerate(t3ie_views(data)):
        view_np = view.numpy().astype(np.float32, copy=False)
        view_full = np.zeros((view_np.shape[0], *orig_shape), dtype=np.float32)
        view_full = insert_crop_into_image(view_full, view_np, bbox)
        write_reoriented_nifti(view_full[0], output_dir / f"t3ie_view{view_index}.nii.gz", props)

    for prompt_index, prompt in enumerate(prompts):
        name = safe_name(prompt)
        write_reoriented_nifti(pseudo_prob[prompt_index], output_dir / f"{name}_prob.nii.gz", props)
        writer.write_seg((pseudo_prob[prompt_index] > 0.5).astype(np.uint8), str(output_dir / f"{name}_pseudo_label.nii.gz"), props)
        write_reoriented_nifti(uncertainty[prompt_index], output_dir / f"{name}_uncertainty.nii.gz", props)
        write_reoriented_nifti(consistency[prompt_index], output_dir / f"{name}_consistency.nii.gz", props)


def generate(args: argparse.Namespace) -> None:
    device = get_device(args.device, args.gpu)
    predictor = load_voxtell_predictor(Path(args.voxtell_root), Path(args.model_dir), device)
    predictor.network.eval()
    print_label_mapping(args.prompts)

    image_paths = discover_image_paths(Path(args.data_root), args.sequences)
    if args.case_limit is not None:
        image_paths = image_paths[: args.case_limit]
    debug_cases = set()
    if args.debug_save_dir and args.debug_save_count > 0:
        rng = random.Random(args.debug_seed)
        debug_cases = set(rng.sample(image_paths, min(args.debug_save_count, len(image_paths))))

    reader = NibabelIOWithReorient()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        output_file = output_root / sequence / f"{image_path.name}.npz"
        if args.skip_existing and output_file.exists():
            print(f"[{index}/{len(image_paths)}] skip {output_file}")
            continue
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(image_paths)}] source VoxTell + T3IE: {sequence}/{image_path.name}")
        image_np, props = reader.read_images([str(image_path)])
        pseudo_prob, uncertainty, consistency = predict_t3ie_probabilities(predictor, image_np, args.prompts)
        np.savez_compressed(
            output_file,
            image_path=str(image_path),
            prompts=np.asarray(args.prompts),
            pseudo_prob=pseudo_prob,
            uncertainty=uncertainty,
            consistency=consistency,
        )
        if image_path in debug_cases:
            save_t3ie_debug_outputs(
                predictor=predictor,
                image_np=image_np,
                props=props,
                prompts=args.prompts,
                pseudo_prob=pseudo_prob,
                uncertainty=uncertainty,
                consistency=consistency,
                output_dir=Path(args.debug_save_dir) / sequence / image_path.name,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SRPL-SFDA initial pseudo-labels with source VoxTell and T3IE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--voxtell-root", default="/data/zy/VoxTell_from_disk")
    parser.add_argument("--model-dir", default="/data/zy/VoxTell_from_disk/model")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--debug-save-dir", default=None)
    parser.add_argument("--debug-save-count", type=int, default=0)
    parser.add_argument("--debug-seed", type=int, default=2024)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
