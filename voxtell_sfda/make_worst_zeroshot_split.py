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
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.data import discover_image_paths


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    pred_sum = pred.sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)
    return float((2 * inter + eps) / (pred_sum + gt_sum + eps)), float((inter + eps) / (union + eps))


def baseline_path_for_image(baseline_root: Path, image_path: Path) -> Path:
    sequence_path = baseline_root / image_path.parent.name / image_path.name
    if sequence_path.exists():
        return sequence_path
    return baseline_root / image_path.name


def run(args: argparse.Namespace) -> None:
    reader = NibabelIOWithReorient()
    image_paths = discover_image_paths(Path(args.data_root), args.sequences)
    rows = []
    for image_path in image_paths:
        sequence = image_path.parent.name
        gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
        pred_path = baseline_path_for_image(Path(args.baseline_pred_dir), image_path)
        if not gt_path.exists() or not pred_path.exists():
            print(f"skip missing gt/pred: gt={gt_path.exists()} pred={pred_path.exists()} {image_path}", file=sys.stderr)
            continue
        gt, _ = reader.read_images([str(gt_path)])
        pred, _ = reader.read_images([str(pred_path)])
        gt_mask = gt[0] == args.label_value
        pred_mask = pred[0] == args.label_value
        dice, iou = dice_iou(pred_mask, gt_mask)
        rows.append(
            {
                "sequence": sequence,
                "case": image_path.name,
                "dice": dice,
                "iou": iou,
                "score": (dice + iou) / 2.0,
                "gt_voxels": int(gt_mask.sum(dtype=np.int64)),
                "pred_voxels": int(pred_mask.sum(dtype=np.int64)),
            }
        )
    if not rows:
        raise RuntimeError("No baseline rows computed.")

    rows = sorted(rows, key=lambda row: (float(row["score"]), float(row["dice"]), float(row["iou"]), row["case"]))
    test_count = int(round(len(rows) * (1.0 - args.train_ratio)))
    test_count = min(max(test_count, 1), len(rows) - 1)
    test_rows = rows[:test_count]
    test_cases = [row["case"] for row in test_rows]
    train_cases = [row["case"] for row in rows[test_count:]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "zeroshot_case_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    split = {
        "strategy": "worst_zeroshot_score",
        "score": "(dice+iou)/2 ascending",
        "train_ratio": args.train_ratio,
        "label_value": args.label_value,
        "num_cases": len(rows),
        "num_train_cases": len(train_cases),
        "num_test_cases": len(test_cases),
        "train_cases": train_cases,
        "test_cases": test_cases,
        "test": test_cases,
    }
    split_path = output_dir / "worst_zeroshot_split.json"
    split_path.write_text(json.dumps(split, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved split: {split_path}")
    print("Worst test cases:")
    for row in test_rows:
        print(f"  {row['case']}: dice={float(row['dice']):.4f}, iou={float(row['iou']):.4f}, score={float(row['score']):.4f}")
    print(
        f"Mean test zero-shot Dice={np.mean([float(r['dice']) for r in test_rows]):.4f}, "
        f"mIoU={np.mean([float(r['iou']) for r in test_rows]):.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed split whose test set is the worst zero-shot baseline cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--baseline-pred-dir", default="/data/zy/VoxTell_from_disk/out_multi")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/worst_zeroshot_split_p0")
    parser.add_argument("--sequences", nargs="+", default=["P0"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--label-value", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
