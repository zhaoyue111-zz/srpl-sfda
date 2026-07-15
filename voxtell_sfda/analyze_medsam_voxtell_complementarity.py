#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)
    intersection = np.logical_and(pred, gt).sum(dtype=np.int64)
    pred_sum = pred.sum(dtype=np.int64)
    gt_sum = gt.sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)
    if gt_sum == 0:
        score = 1.0 if pred_sum == 0 else 0.0
        return score, score
    return float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps)), float((intersection + eps) / (union + eps))


def load_label(path: Path, label_value: int) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj) == label_value


def save_mask(mask: np.ndarray, reference_path: Path, output_path: Path) -> None:
    reference = nib.load(str(reference_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    nib.save(image, str(output_path))


def analyze(args: argparse.Namespace) -> None:
    pred_dir = Path(args.voxtell_pred_dir)
    medsam_dir = Path(args.medsam_pred_dir)
    gt_dir = Path(args.gt_dir)
    output_csv = Path(args.output_csv)
    rows = []
    debug_cases = []

    for gt_path in sorted(gt_dir.glob("*.nii.gz")):
        case = gt_path.name
        voxtell_path = pred_dir / case
        medsam_path = medsam_dir / case
        if not voxtell_path.exists() or not medsam_path.exists():
            continue

        gt = load_label(gt_path, args.label_value)
        voxtell = load_label(voxtell_path, args.label_value)
        medsam = load_label(medsam_path, args.label_value)
        if gt.shape != voxtell.shape or gt.shape != medsam.shape:
            raise ValueError(f"Shape mismatch for {case}: gt={gt.shape}, voxtell={voxtell.shape}, medsam={medsam.shape}")

        union = voxtell | medsam
        intersection = voxtell & medsam
        voxtell_dice, voxtell_iou = dice_iou(voxtell, gt)
        medsam_dice, medsam_iou = dice_iou(medsam, gt)
        union_dice, union_iou = dice_iou(union, gt)
        intersection_dice, intersection_iou = dice_iou(intersection, gt)

        voxtell_fn = gt & ~voxtell
        medsam_added = medsam & ~voxtell
        medsam_added_tp = medsam_added & gt
        medsam_added_fp = medsam_added & ~gt
        fn_coverage = float(medsam_added_tp.sum(dtype=np.int64) / max(voxtell_fn.sum(dtype=np.int64), 1))
        added_precision = float(medsam_added_tp.sum(dtype=np.int64) / max(medsam_added.sum(dtype=np.int64), 1))
        row = {
            "case": case,
            "voxtell_dice": voxtell_dice,
            "medsam_dice": medsam_dice,
            "union_dice": union_dice,
            "intersection_dice": intersection_dice,
            "voxtell_iou": voxtell_iou,
            "medsam_iou": medsam_iou,
            "union_iou": union_iou,
            "intersection_iou": intersection_iou,
            "union_delta_dice": union_dice - voxtell_dice,
            "voxtell_gt_fn_voxels": int(voxtell_fn.sum(dtype=np.int64)),
            "medsam_covers_voxtell_fn_voxels": int(medsam_added_tp.sum(dtype=np.int64)),
            "fn_coverage_by_medsam": fn_coverage,
            "medsam_added_voxels_not_voxtell": int(medsam_added.sum(dtype=np.int64)),
            "medsam_added_true_positive": int(medsam_added_tp.sum(dtype=np.int64)),
            "medsam_added_false_positive": int(medsam_added_fp.sum(dtype=np.int64)),
            "medsam_added_precision": added_precision,
            "gt_voxels": int(gt.sum(dtype=np.int64)),
        }
        rows.append(row)
        debug_cases.append((row, gt_path, voxtell_fn, medsam_added_tp, medsam_added_fp, union))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_csv}")
    for key in ("voxtell_dice", "medsam_dice", "union_dice", "fn_coverage_by_medsam", "medsam_added_precision"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        print(f"{key}: mean={values.mean():.6f} median={np.median(values):.6f} min={values.min():.6f} max={values.max():.6f}")

    if args.debug_dir:
        debug_root = Path(args.debug_dir)
        best_gain = sorted(debug_cases, key=lambda item: item[0]["union_delta_dice"], reverse=True)[: args.debug_count]
        worst_gain = sorted(debug_cases, key=lambda item: item[0]["union_delta_dice"])[: args.debug_count]
        high_coverage = sorted(debug_cases, key=lambda item: item[0]["fn_coverage_by_medsam"], reverse=True)[: args.debug_count]
        for group_name, group in (("best_union_gain", best_gain), ("worst_union_drop", worst_gain), ("high_fn_coverage", high_coverage)):
            for row, gt_path, voxtell_fn, medsam_added_tp, medsam_added_fp, union in group:
                case_dir = debug_root / group_name / row["case"]
                save_mask(voxtell_fn, gt_path, case_dir / "voxtell_false_negative_gt.nii.gz")
                save_mask(medsam_added_tp, gt_path, case_dir / "medsam_recovers_voxtell_fn.nii.gz")
                save_mask(medsam_added_fp, gt_path, case_dir / "medsam_added_false_positive.nii.gz")
                save_mask(union, gt_path, case_dir / "voxtell_union_medsam.nii.gz")
        print(f"Saved debug masks: {debug_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether MedSAM covers VoxTell false negatives on P0 liver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voxtell-pred-dir", default="/data/zy/VoxTell_from_disk/out_multi/P0")
    parser.add_argument("--medsam-pred-dir", default="/data/zy/SRPL-SFDA-main/runs/medsam_liver_p0/P0")
    parser.add_argument("--gt-dir", default="/data/zy/CT_MRI_DATA_3D/labels/P0")
    parser.add_argument("--label-value", type=int, default=5)
    parser.add_argument("--output-csv", default="/data/zy/SRPL-SFDA-main/runs/medsam_voxtell_overlap_analysis.csv")
    parser.add_argument("--debug-dir", default="/data/zy/SRPL-SFDA-main/runs/medsam_voxtell_overlap_debug")
    parser.add_argument("--debug-count", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
