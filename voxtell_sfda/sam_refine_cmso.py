#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from scipy.ndimage import binary_dilation, label
from scipy.optimize import minimize

from voxtell_sfda.data import discover_image_paths, select_image_split
from voxtell_sfda.nifti import write_reoriented_nifti
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping


def safe_name(prompt: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in prompt)


def normalize_nonzero_volume(image_3d: np.ndarray) -> np.ndarray:
    image = image_3d.astype(np.float32, copy=False)
    normalized = np.zeros_like(image, dtype=np.float32)
    mask = image != 0
    if not np.any(mask):
        return normalized
    values = image[mask]
    min_value = float(values.min())
    max_value = float(values.max())
    normalized[mask] = (values - min_value) / max(max_value - min_value, 1e-6)
    return normalized


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(image + 1e-10, gamma).astype(np.float32, copy=False)


def adaptive_gamma_volume(image_norm: np.ndarray) -> np.ndarray:
    values = image_norm[image_norm > 0]
    if values.size == 0:
        return image_norm.copy()

    def objective(gamma_array):
        corrected = gamma_correction(values, float(gamma_array[0]))
        return abs((float(corrected.mean()) - 0.5) + (float(corrected.std()) - 0.29))

    result = minimize(objective, np.asarray([1.0], dtype=np.float32), method="Nelder-Mead")
    gamma = float(result.x[0])
    corrected = np.zeros_like(image_norm, dtype=np.float32)
    mask = image_norm > 0
    corrected[mask] = gamma_correction(image_norm[mask], gamma)
    return corrected


def fixed_gamma_volume(image_norm: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    corrected = np.zeros_like(image_norm, dtype=np.float32)
    mask = image_norm > 0
    corrected[mask] = gamma_correction(image_norm[mask], gamma)
    return corrected


def equalize_uint8_slice(slice_norm: np.ndarray) -> np.ndarray:
    slice_uint8 = np.clip(slice_norm * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(slice_uint8.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = cdf > 0
    if not np.any(nonzero):
        return slice_uint8
    cdf_min = cdf[nonzero][0]
    denom = max(int(cdf[-1] - cdf_min), 1)
    lut = np.round((cdf - cdf_min) * 255.0 / denom).clip(0, 255).astype(np.uint8)
    return lut[slice_uint8]


def equalized_volume(image_norm: np.ndarray) -> np.ndarray:
    equalized = np.zeros_like(image_norm, dtype=np.float32)
    for z in range(image_norm.shape[0]):
        equalized[z, :, :] = equalize_uint8_slice(image_norm[z, :, :]).astype(np.float32) / 255.0
    return equalized


def build_t3ie_concat_volumes(image_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the three enhanced volumes used as SAM RGB channels.

    This mirrors the original 2D concat step, but each enhancement is produced
    as a 3D volume first and then sliced/concatenated for SAM.
    """

    image_norm = normalize_nonzero_volume(image_3d)
    adaptive = adaptive_gamma_volume(image_norm)
    fixed = fixed_gamma_volume(image_norm, gamma=0.6)
    equalized = equalized_volume(image_norm)
    return tuple((np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8) for v in (adaptive, fixed, equalized))


def concat_t3ie_slice(enhanced_volumes: tuple[np.ndarray, np.ndarray, np.ndarray], z: int) -> np.ndarray:
    return np.stack([volume[z, :, :] for volume in enhanced_volumes], axis=-1)


def gray_to_rgb_slice(slice_uint8: np.ndarray) -> np.ndarray:
    return np.repeat(slice_uint8[:, :, None], 3, axis=-1)


def bbox_from_mask(mask: np.ndarray, margin: int = 3) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask.shape[1])
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, mask.shape[0])
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return float((2 * inter + eps) / (a.sum() + b.sum() + eps))


def foreground_band(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=False)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask.astype(bool, copy=False), structure=structure)


def clip_mask_to_box(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    clipped = np.zeros_like(mask, dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


def keep_components_overlapping_init(mask: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
    '''
    保留与初始化掩码重叠的组件。
    '''
    components, num_components = label(mask.astype(bool, copy=False)) # 连通域标记
    if num_components == 0:
        return mask.astype(bool, copy=False)
    overlap_ids = np.unique(components[np.logical_and(init_mask, components > 0)]) # 只保留与初始掩码重叠的连通分量
    if overlap_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(components, overlap_ids)


def postprocess_sam_mask(mask: np.ndarray, box: np.ndarray, init_mask: np.ndarray, mode: str) -> np.ndarray:
    processed = mask.astype(bool, copy=False)
    if mode in {"box", "init_overlap_component_box"}:
        processed = clip_mask_to_box(processed, box)
    if mode in {"init_overlap_component", "init_overlap_component_box"}:
        processed = keep_components_overlapping_init(processed, init_mask)
    if mode == "none":
        return processed
    if mode in {"box", "init_overlap_component", "init_overlap_component_box"}:
        return processed
    raise ValueError(f"Unknown --sam-candidate-postprocess: {mode}")


def select_sam_mask(
    sam_predictor,
    image_rgb: np.ndarray,
    box: np.ndarray,
    init_mask: np.ndarray,
    postprocess_mode: str = "none",
) -> tuple[np.ndarray, float, float]:
    with torch.inference_mode():
        sam_predictor.set_image(image_rgb)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )
    masks = np.stack(
        [postprocess_sam_mask(mask, box=box, init_mask=init_mask, mode=postprocess_mode) for mask in masks],
        axis=0,
    )
    scores = np.asarray(scores, dtype=np.float32)
    agreements = np.asarray([dice(mask, init_mask) for mask in masks], dtype=np.float32)
    best_index = int(np.argmax(scores + agreements))
    return masks[best_index], float(scores[best_index]), float(agreements[best_index])


def load_sam_predictor(args: argparse.Namespace):
    if args.skip_sam:
        return None
    from segment_anything import SamPredictor, sam_model_registry

    if not args.sam_checkpoint:
        raise ValueError("--sam-checkpoint is required unless --skip-sam is set")
    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable for SAM/MedSAM; loading checkpoint on CPU.", file=sys.stderr)
        device = "cpu"

    original_torch_load = torch.load

    def torch_load_with_map_location(*load_args, **load_kwargs):
        load_kwargs.setdefault("map_location", torch.device(device))
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_with_map_location
    try:
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    finally:
        torch.load = original_torch_load
    sam = sam.to(device=device)
    return SamPredictor(sam)


def refine_one_case(
    image: np.ndarray,
    pseudo_prob: np.ndarray,
    uncertainty: np.ndarray,
    consistency: np.ndarray,
    sam_predictor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pseudo_binary = pseudo_prob > args.prob_threshold
    base_reliable = (uncertainty <= args.uncertainty_threshold) & (consistency <= args.consistency_threshold)
    if args.base_reliable_region == "all":
        base_reliable = base_reliable.astype(np.float32)
    elif args.base_reliable_region == "foreground":
        base_reliable = np.logical_and(base_reliable, pseudo_binary).astype(np.float32)
    elif args.base_reliable_region == "none":
        base_reliable = np.zeros_like(base_reliable, dtype=np.float32)
    else:
        raise ValueError(f"Unknown --base-reliable-region: {args.base_reliable_region}")
    refined_prob = pseudo_binary.astype(np.float32)
    reliable_mask = base_reliable.copy()

    if sam_predictor is None:
        return refined_prob, reliable_mask

    image_3d = image[0] if image.ndim == 4 else image
    enhanced_volumes = build_t3ie_concat_volumes(image_3d)
    # Keep VoxTell/T3IE as the base pseudo-label by default. SAM is used to find
    # reliable foreground regions through cross-view mask consistency.
    refined_prob = pseudo_binary.astype(np.float32)
    reliable_mask = base_reliable.copy()

    num_classes, depth, _, _ = pseudo_prob.shape
    for z in range(depth):
        view_slices = [gray_to_rgb_slice(volume[z, :, :]) for volume in enhanced_volumes]
        for class_index in range(num_classes):
            init_mask = pseudo_binary[class_index, z, :, :]
            box = bbox_from_mask(init_mask, margin=args.bbox_margin)
            if box is None:
                continue
            candidates = []
            scores = []
            agreements = []
            for image_rgb in view_slices:
                selected, score, agreement = select_sam_mask(
                    sam_predictor=sam_predictor,
                    image_rgb=image_rgb,
                    box=box,
                    init_mask=init_mask,
                    postprocess_mode=args.sam_candidate_postprocess,
                )
                candidates.append(selected)
                scores.append(score)
                agreements.append(agreement)

            pairwise = [
                dice(candidates[i], candidates[j])
                for i in range(len(candidates))
                for j in range(i + 1, len(candidates))
            ]
            cmso_score = float(np.mean(pairwise)) if pairwise else 1.0
            candidate_stack = np.stack(candidates, axis=0)
            consensus = candidate_stack.sum(axis=0) >= args.sam_consensus_votes
            mean_score = float(np.mean(scores))
            mean_agreement = float(np.mean(agreements))
            init_area = int(init_mask.sum())
            added_area = int(np.logical_and(consensus, np.logical_not(init_mask)).sum())
            added_ratio = added_area / max(init_area, 1)
            selected_reliable = (
                cmso_score >= args.cmso_threshold
                and mean_agreement >= args.init_agreement_threshold
                and mean_score >= args.sam_score_threshold
                and added_ratio <= args.max_sam_added_ratio
            )

            if not selected_reliable:
                continue

            if args.sam_update_mode == "replace_reliable":
                updated = consensus
            elif args.sam_update_mode == "union_reliable":
                updated = np.logical_or(init_mask, consensus)
            elif args.sam_update_mode == "keep_source":
                updated = init_mask
            else:
                raise ValueError(f"Unknown --sam-update-mode: {args.sam_update_mode}")

            refined_prob[class_index, z, :, :] = updated.astype(np.float32)
            if args.reliable_region == "sam":
                reliable_region = consensus
            elif args.reliable_region == "sam_init_intersection":
                reliable_region = np.logical_and(consensus, init_mask)
            elif args.reliable_region == "sam_init_band":
                reliable_region = np.logical_and(consensus, foreground_band(init_mask, args.reliable_band_radius))
            else:
                raise ValueError(f"Unknown --reliable-region: {args.reliable_region}")
            reliable_mask[class_index, z, :, :] = np.logical_or(
                reliable_mask[class_index, z, :, :] > 0,
                reliable_region,
            ).astype(np.float32)

    return refined_prob, reliable_mask


def run(args: argparse.Namespace) -> None:
    print_label_mapping(args.prompts)
    sam_predictor = load_sam_predictor(args)
    image_paths = discover_image_paths(Path(args.data_root), args.sequences)
    image_paths = select_image_split(
        image_paths,
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        test_cases_file=Path(args.test_cases_file) if args.test_cases_file else None,
    )
    if args.case_limit is not None:
        image_paths = image_paths[: args.case_limit]
    debug_cases = set()
    if args.debug_save_dir and args.debug_save_count > 0:
        rng = random.Random(args.debug_seed)
        debug_cases = set(rng.sample(image_paths, min(args.debug_save_count, len(image_paths))))

    reader = NibabelIOWithReorient()
    output_root = Path(args.output_dir)
    pseudo_root = Path(args.pseudo_dir)

    for index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        pseudo_file = pseudo_root / sequence / f"{image_path.name}.npz"
        if not pseudo_file.exists():
            raise FileNotFoundError(f"Missing initial pseudo file: {pseudo_file}")

        output_file = output_root / sequence / f"{image_path.name}.npz"
        if args.skip_existing and output_file.exists():
            print(f"[{index}/{len(image_paths)}] skip {output_file}")
            continue
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(image_paths)}] SAM/CMSO refine: {sequence}/{image_path.name}", flush=True)
        image, props = reader.read_images([str(image_path)])
        pseudo = np.load(pseudo_file)
        refined_prob, reliable_mask = refine_one_case(
            image=image,
            pseudo_prob=pseudo["pseudo_prob"],
            uncertainty=pseudo["uncertainty"],
            consistency=pseudo["consistency"],
            sam_predictor=sam_predictor,
            args=args,
        )
        np.savez_compressed(
            output_file,
            image_path=str(image_path),
            prompts=np.asarray(args.prompts),
            pseudo_prob=refined_prob.astype(np.float32),
            reliable_mask=reliable_mask.astype(np.float32),
        )
        if image_path in debug_cases:
            debug_dir = Path(args.debug_save_dir) / sequence / image_path.name
            debug_dir.mkdir(parents=True, exist_ok=True)
            writer = NibabelIOWithReorient()
            for prompt_index, prompt in enumerate(args.prompts):
                name = safe_name(prompt)
                write_reoriented_nifti(refined_prob[prompt_index], debug_dir / f"{name}_sam_refined_prob.nii.gz", props)
                writer.write_seg((refined_prob[prompt_index] > args.prob_threshold).astype(np.uint8), str(debug_dir / f"{name}_sam_refined_label.nii.gz"), props)
                writer.write_seg(reliable_mask[prompt_index].astype(np.uint8), str(debug_dir / f"{name}_reliable_mask.nii.gz"), props)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine VoxTell pseudo-labels with SAM/MedSAM and CMSO reliable selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--pseudo-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_reliable")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--split", choices=["all", "train", "test"], default="train")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--test-cases-file", default=None, help="JSON file with fixed test_cases/test list. Overrides random split.")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam-checkpoint", default="/data/zy/SRPL-SFDA-main/work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.4)
    parser.add_argument("--consistency-threshold", type=float, default=0.15)
    parser.add_argument("--cmso-threshold", type=float, default=0.75)
    parser.add_argument("--init-agreement-threshold", type=float, default=0.5)
    parser.add_argument("--sam-score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--base-reliable-region",
        choices=["all", "foreground", "none"],
        default="all",
        help="Initial reliable region before SAM/CMSO. 'all' keeps the old behavior; 'foreground' avoids marking broad confident background as reliable.",
    )
    parser.add_argument(
        "--max-sam-added-ratio",
        type=float,
        default=0.35,
        help="Reject SAM masks if added foreground area exceeds this fraction of the VoxTell/T3IE foreground on the slice.",
    )
    parser.add_argument(
        "--sam-update-mode",
        choices=["keep_source", "replace_reliable", "union_reliable"],
        default="keep_source",
        help="How reliable SAM masks update the VoxTell/T3IE pseudo-label.",
    )
    parser.add_argument("--sam-consensus-votes", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument(
        "--reliable-region",
        choices=["sam", "sam_init_intersection", "sam_init_band"],
        default="sam_init_intersection",
        help="SAM-guided foreground region added to the reliability mask.",
    )
    parser.add_argument("--reliable-band-radius", type=int, default=3)
    parser.add_argument("--bbox-margin", type=int, default=3)
    parser.add_argument(
        "--sam-candidate-postprocess",
        choices=["none", "box", "init_overlap_component", "init_overlap_component_box"],
        default="init_overlap_component_box",
        help="Post-process SAM candidates before CMSO scoring. The default clips to the prompt box and keeps only connected components touching the source pseudo-label.",
    )
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--debug-save-dir", default=None)
    parser.add_argument("--debug-save-count", type=int, default=0)
    parser.add_argument("--debug-seed", type=int, default=2024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
