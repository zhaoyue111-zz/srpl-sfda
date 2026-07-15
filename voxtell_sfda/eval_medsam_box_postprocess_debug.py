#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.data import discover_image_paths, pseudo_path_for_image, select_image_split
from voxtell_sfda.infer_voxtell_sfda import resolve_label_values
from voxtell_sfda.nifti import write_reoriented_nifti
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping
from voxtell_sfda.sam_refine_cmso import (
    bbox_from_mask,
    build_t3ie_concat_volumes,
    concat_t3ie_slice,
    dice,
    postprocess_sam_mask,
)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return safe_div(float(np.logical_and(a, b).sum(dtype=np.int64)), float(np.logical_or(a, b).sum(dtype=np.int64)))


def load_predictor(args: argparse.Namespace):
    from segment_anything import SamPredictor, sam_model_registry

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable for MedSAM; loading checkpoint on CPU.", file=sys.stderr)
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


def source_mask_for_case(
    args: argparse.Namespace,
    image_path: Path,
    gt_target: np.ndarray,
    prompt_index: int,
) -> np.ndarray:
    if args.source_mask == "gt":
        return gt_target

    pseudo_file = pseudo_path_for_image(Path(args.pseudo_dir), image_path)
    if not pseudo_file.exists():
        raise FileNotFoundError(f"Missing pseudo file: {pseudo_file}")
    pseudo = np.load(pseudo_file)
    pseudo_prob = pseudo["pseudo_prob"]
    pseudo_class_index = args.pseudo_class_index
    if pseudo_class_index is None:
        pseudo_class_index = 0 if pseudo_prob.shape[0] == 1 else prompt_index
    if pseudo_class_index >= pseudo_prob.shape[0]:
        raise IndexError(f"Pseudo class index {pseudo_class_index} is outside pseudo_prob shape {pseudo_prob.shape}")
    return pseudo_prob[pseudo_class_index] > args.prob_threshold


def metric_row(
    prefix: str,
    pred: np.ndarray,
    gt: np.ndarray,
    box_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum(dtype=np.int64))
    pred_area = int(pred.sum(dtype=np.int64))
    gt_area = int(gt.sum(dtype=np.int64))
    row: dict[str, float | int] = {
        f"{prefix}_area": pred_area,
        f"{prefix}_intersection_gt": inter,
        f"{prefix}_dice": dice(pred, gt),
        f"{prefix}_iou": iou(pred, gt),
        f"{prefix}_precision_mask_in_gt": safe_div(inter, pred_area),
        f"{prefix}_gt_coverage": safe_div(inter, gt_area),
    }
    if box_mask is not None:
        box_area = int(box_mask.sum(dtype=np.int64))
        row[f"{prefix}_fraction_inside_box"] = safe_div(int(np.logical_and(pred, box_mask).sum(dtype=np.int64)), pred_area)
        row[f"{prefix}_box_fraction_used"] = safe_div(pred_area, box_area)
    return row


def write_seg(reader: NibabelIOWithReorient, mask: np.ndarray, path: Path, properties: dict, label_value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reader.write_seg((mask.astype(np.uint8) * int(label_value)).astype(np.uint8), str(path), properties)


def run(args: argparse.Namespace) -> None:
    print_label_mapping(args.prompts)
    label_values = resolve_label_values(args.prompts, args.label_values)
    if args.prompt not in args.prompts:
        raise ValueError(f"--prompt {args.prompt!r} is not in --prompts")
    prompt_index = args.prompts.index(args.prompt)
    target_label = label_values[prompt_index]

    predictor = load_predictor(args)
    reader = NibabelIOWithReorient()
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

    output_root = Path(args.output_dir)
    volume_root = output_root / "volumes"
    rows = []
    case_rows = []
    split_manifest = {
        "split": args.split,
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "test_cases_file": args.test_cases_file,
        "source_mask": args.source_mask,
        "bbox_margin": args.bbox_margin,
        "postprocess": args.postprocess,
        "cases": [str(p) for p in image_paths],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    for case_index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
        if not gt_path.exists():
            print(f"missing label: {gt_path}", file=sys.stderr)
            continue

        print(f"[{case_index}/{len(image_paths)}] MedSAM box debug: {sequence}/{image_path.name}", flush=True)
        image, properties = reader.read_images([str(image_path)])
        gt, _ = reader.read_images([str(gt_path)])
        image_3d = image[0] if image.ndim == 4 else image
        label_map = gt[0].astype(np.int16, copy=False)
        gt_target = label_map == target_label
        source_mask = source_mask_for_case(args, image_path, gt_target, prompt_index)  # 从伪标签中获取源掩码
        if source_mask.shape != gt_target.shape:
            raise ValueError(f"Shape mismatch for {image_path}: source={source_mask.shape}, gt={gt_target.shape}")

        enhanced_volumes = build_t3ie_concat_volumes(image_3d)
        box_volume = np.zeros_like(gt_target, dtype=bool)
        raw_volume = np.zeros_like(gt_target, dtype=bool)
        post_volume = np.zeros_like(gt_target, dtype=bool)
        source_positive_slices = 0
        used_slices = 0

        for z in range(source_mask.shape[0]):
            init_mask = source_mask[z]
            gt_slice = gt_target[z]
            if init_mask.sum(dtype=np.int64) == 0:
                continue
            source_positive_slices += 1
            box = bbox_from_mask(init_mask, margin=args.bbox_margin)
            if box is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            box_slice = np.zeros_like(init_mask, dtype=bool)
            box_slice[y0:y1, x0:x1] = True  # 填充为一个立方体
            box_volume[z] = box_slice

            predictor.set_image(concat_t3ie_slice(enhanced_volumes, z))
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            raw_mask = masks[best_index].astype(bool)
            post_mask = postprocess_sam_mask(raw_mask, box=box, init_mask=init_mask, mode=args.postprocess)
            raw_volume[z] = raw_mask
            post_volume[z] = post_mask
            used_slices += 1

            target_in_box = int(np.logical_and(box_slice, gt_slice).sum(dtype=np.int64))
            box_area = int(box_slice.sum(dtype=np.int64))
            gt_area = int(gt_slice.sum(dtype=np.int64))
            init_gt_intersection = int(np.logical_and(init_mask, gt_slice).sum(dtype=np.int64))
            row = {
                "sequence": sequence,
                "case": image_path.name,
                "z": z,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "sam_score": float(scores[best_index]),
                "source_area": int(init_mask.sum(dtype=np.int64)),
                "gt_area": gt_area,
                "source_gt_dice": dice(init_mask, gt_slice),
                "source_precision_mask_in_gt": safe_div(init_gt_intersection, int(init_mask.sum(dtype=np.int64))),
                "source_gt_coverage": safe_div(init_gt_intersection, gt_area),
                "box_area": box_area,
                "box_target_pixels": target_in_box,
                "box_gt_coverage": safe_div(target_in_box, gt_area),
                "box_target_fraction": safe_div(target_in_box, box_area),
            }
            row.update(metric_row("raw", raw_mask, gt_slice, box_slice))
            row.update(metric_row("post", post_mask, gt_slice, box_slice))
            rows.append(row)

        case_dir = volume_root / sequence / image_path.name
        write_seg(reader, gt_target, case_dir / "gt_mask.nii.gz", properties, label_value=target_label)
        write_seg(reader, source_mask, case_dir / "source_mask.nii.gz", properties, label_value=target_label)
        write_seg(reader, box_volume, case_dir / "box_mask.nii.gz", properties, label_value=1)
        write_seg(reader, raw_volume, case_dir / "sam_raw_mask.nii.gz", properties, label_value=target_label)
        write_seg(reader, post_volume, case_dir / "sam_post_mask.nii.gz", properties, label_value=target_label)
        ct_in_box = np.where(box_volume, image_3d, 0).astype(image_3d.dtype, copy=False)
        write_reoriented_nifti(ct_in_box, case_dir / "ct_in_box.nii.gz", properties)

        case_row = {
            "sequence": sequence,
            "case": image_path.name,
            "source_positive_slices": source_positive_slices,
            "used_slices": used_slices,
        }
        case_row.update(metric_row("source", source_mask, gt_target, box_volume))
        case_row.update(metric_row("box", box_volume, gt_target))
        case_row.update(metric_row("raw", raw_volume, gt_target, box_volume))
        case_row.update(metric_row("post", post_volume, gt_target, box_volume))
        case_rows.append(case_row)
        print(
            f"  source Dice={case_row['source_dice']:.4f}, "
            f"raw Dice={case_row['raw_dice']:.4f}, post Dice={case_row['post_dice']:.4f}, "
            f"raw precision={case_row['raw_precision_mask_in_gt']:.4f}, "
            f"post precision={case_row['post_precision_mask_in_gt']:.4f}"
        )

    if rows:
        detail_path = output_root / "slice_metrics.csv"
        with detail_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        case_path = output_root / "case_metrics.csv"
        with case_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(case_rows)

        summary = {}
        for key in case_rows[0].keys():
            if key in {"sequence", "case"}:
                continue
            values = []
            for row in case_rows:
                value = row[key]
                if isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(float(value))
            if values:
                summary[f"mean_{key}"] = float(np.mean(values))
        summary["num_cases"] = len(case_rows)
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved slice metrics: {detail_path}")
        print(f"Saved case metrics: {case_path}")
        print(f"Saved summary: {summary_path}")
        print(
            "Mean case metrics: "
            f"source Dice={summary['mean_source_dice']:.4f}, "
            f"raw Dice={summary['mean_raw_dice']:.4f}, "
            f"post Dice={summary['mean_post_dice']:.4f}, "
            f"raw precision={summary['mean_raw_precision_mask_in_gt']:.4f}, "
            f"post precision={summary['mean_post_precision_mask_in_gt']:.4f}, "
            f"raw GT coverage={summary['mean_raw_gt_coverage']:.4f}, "
            f"post GT coverage={summary['mean_post_gt_coverage']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedSAM with box prompts and save raw/postprocessed masks plus box CT volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--pseudo-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/medsam_box_post_debug")
    parser.add_argument("--sequences", nargs="+", default=["P0"])
    parser.add_argument("--split", choices=["all", "train", "test"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--test-cases-file", default=None, help="JSON file with fixed test_cases/test list. Overrides random split.")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--label-values", nargs="+", type=int, default=None)
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--source-mask", choices=["pseudo", "gt"], default="pseudo")
    parser.add_argument("--pseudo-class-index", type=int, default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--bbox-margin", type=int, default=0)
    parser.add_argument(
        "--postprocess",
        choices=["none", "box", "init_overlap_component", "init_overlap_component_box"],
        default="init_overlap_component_box",
    )
    parser.add_argument("--sam-checkpoint", default="/data/zy/SRPL-SFDA-main/work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
