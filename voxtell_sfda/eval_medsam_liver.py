#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.data import discover_image_paths, select_image_split
from voxtell_sfda.sam_refine_cmso import bbox_from_mask, build_t3ie_concat_volumes, concat_t3ie_slice, dice


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    return float((np.logical_and(pred, gt).sum() + eps) / (np.logical_or(pred, gt).sum() + eps))


def load_predictor(args):
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


def run(args: argparse.Namespace) -> None:
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
    rows = []
    for case_index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
        if not gt_path.exists():
            print(f"missing label: {gt_path}", file=sys.stderr)
            continue

        image, properties = reader.read_images([str(image_path)])
        gt, _ = reader.read_images([str(gt_path)])
        image_3d = image[0] if image.ndim == 4 else image  # (C, X, Y, Z)
        gt_liver = gt[0] == args.label_value
        enhanced_volumes = build_t3ie_concat_volumes(image_3d) # 三路增强
        pred = np.zeros_like(gt_liver, dtype=np.uint8)

        print(f"[{case_index}/{len(image_paths)}] MedSAM liver oracle bbox: {sequence}/{image_path.name}")
        for z in range(gt_liver.shape[0]):
            box = bbox_from_mask(gt_liver[z, :, :], margin=args.bbox_margin)
            if box is None:
                continue
            predictor.set_image(concat_t3ie_slice(enhanced_volumes, z))
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            pred[z, :, :] = masks[best_index].astype(np.uint8)

        dice_score = dice(pred, gt_liver)
        iou_score = iou(pred, gt_liver)
        rows.append({"sequence": sequence, "case": image_path.name, "dice": dice_score, "iou": iou_score})
        output_path = output_root / sequence / image_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reader.write_seg((pred * args.label_value).astype(np.uint8), str(output_path), properties)
        print(f"  Dice={dice_score:.4f}, IoU={iou_score:.4f}, saved={output_path}")

    if rows:
        output_root.mkdir(parents=True, exist_ok=True)
        metrics_path = output_root / "medsam_liver_metrics.csv"
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence", "case", "dice", "iou"])
            writer.writeheader()
            writer.writerows(rows)
        summary_path = output_root / "medsam_liver_summary.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mean_dice", "mean_iou", "num_cases"])
            writer.writeheader()
            writer.writerow(
                {
                    "mean_dice": float(np.mean([r["dice"] for r in rows])),
                    "mean_iou": float(np.mean([r["iou"] for r in rows])),
                    "num_cases": len(rows),
                }
            )
        print(f"Mean Dice={np.mean([r['dice'] for r in rows]):.4f}, mIoU={np.mean([r['iou'] for r in rows]):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate direct MedSAM liver segmentation on P0 using GT bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--sequences", nargs="+", default=["P0"])
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--test-cases-file", default=None, help="JSON file with fixed test_cases/test list. Overrides random split.")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/medsam_liver_p0")
    parser.add_argument("--sam-checkpoint", default="/data/zy/SRPL-SFDA-main/work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label-value", type=int, default=5)
    parser.add_argument("--bbox-margin", type=int, default=3)
    parser.add_argument("--case-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

'''
python voxtell_sfda/eval_medsam_liver.py \
      --sequences P0 \
      --output-dir /data/zy/SRPL-SFDA-main/runs/medsam_liver_p0
'''
