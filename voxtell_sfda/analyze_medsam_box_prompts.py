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

from voxtell_sfda.data import discover_image_paths, pseudo_path_for_image, select_image_split
from voxtell_sfda.infer_voxtell_sfda import resolve_label_values
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping
from voxtell_sfda.sam_refine_cmso import bbox_from_mask, dice


def parse_int_list(value: str) -> list[int]:
    values = [int(v) for v in value.replace(",", " ").split()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


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


def other_label_counts(label_map: np.ndarray, target_label: int, box: np.ndarray) -> dict[int, int]:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = label_map[y0:y1, x0:x1]
    labels, counts = np.unique(crop, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels, counts)
        if int(label) not in (0, target_label)
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    summary_rows = []
    for margin in sorted({int(row["margin"]) for row in rows}):
        subset = [row for row in rows if int(row["margin"]) == margin]
        box_area = sum(int(row["box_area"]) for row in subset)
        box_target = sum(int(row["box_target_pixels"]) for row in subset)
        box_other = sum(int(row["box_other_labeled_pixels"]) for row in subset)
        box_labeled = sum(int(row["box_labeled_pixels"]) for row in subset)
        target_area = sum(int(row["target_area"]) for row in subset)
        source_area = sum(int(row["source_area"]) for row in subset)
        source_target_intersection = sum(int(row["source_target_intersection"]) for row in subset)
        rows_with_other = sum(1 for row in subset if int(row["box_other_labeled_pixels"]) > 0)
        summary_rows.append(
            {
                "margin": margin,
                "num_boxes": len(subset),
                "boxes_with_other_labeled_ratio": safe_div(rows_with_other, len(subset)),
                "target_recall_weighted": safe_div(box_target, target_area),
                "box_target_fraction_weighted": safe_div(box_target, box_area),
                "box_other_label_fraction_weighted": safe_div(box_other, box_area),
                "other_fraction_among_labeled_weighted": safe_div(box_other, box_labeled),
                "box_area_over_target_weighted": safe_div(box_area, target_area),
                "source_target_precision_weighted": safe_div(source_target_intersection, source_area),
                "source_target_recall_weighted": safe_div(source_target_intersection, target_area),
                "mean_target_recall": float(np.mean([float(row["target_recall"]) for row in subset])),
                "mean_box_target_fraction": float(np.mean([float(row["box_target_fraction"]) for row in subset])),
                "mean_box_other_label_fraction": float(np.mean([float(row["box_other_label_fraction"]) for row in subset])),
            }
        )
    return summary_rows


def print_recommendation(summary_rows: list[dict], target_recall_min: float, other_fraction_max: float) -> None:
    viable = [
        row
        for row in summary_rows
        if float(row["target_recall_weighted"]) >= target_recall_min
        and float(row["box_other_label_fraction_weighted"]) <= other_fraction_max
    ]
    if viable:
        best = min(viable, key=lambda row: (float(row["box_other_label_fraction_weighted"]), float(row["box_area_over_target_weighted"])))
        print(
            "Recommended margin="
            f"{best['margin']} under recall>={target_recall_min:.2f} and "
            f"other-label fraction<={other_fraction_max:.2f}."
        )
    else:
        best_recall = max(summary_rows, key=lambda row: float(row["target_recall_weighted"]))
        cleanest = min(summary_rows, key=lambda row: float(row["box_other_label_fraction_weighted"]))
        print(
            "No margin satisfies both thresholds. "
            f"Best recall margin={best_recall['margin']} "
            f"(recall={float(best_recall['target_recall_weighted']):.4f}); "
            f"cleanest margin={cleanest['margin']} "
            f"(other_fraction={float(cleanest['box_other_label_fraction_weighted']):.4f})."
        )


def run(args: argparse.Namespace) -> None:
    print_label_mapping(args.prompts)
    label_values = resolve_label_values(args.prompts, args.label_values)
    if args.prompt not in args.prompts:
        raise ValueError(f"--prompt {args.prompt!r} is not in --prompts")
    prompt_index = args.prompts.index(args.prompt)
    target_label = label_values[prompt_index]
    margins = parse_int_list(args.margins)

    image_paths = discover_image_paths(Path(args.data_root), args.sequences)
    image_paths = select_image_split(
        image_paths,
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
    )
    if args.case_limit is not None:
        image_paths = image_paths[: args.case_limit]

    reader = NibabelIOWithReorient()
    rows = []
    skipped = []
    for case_index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
        if not gt_path.exists():
            skipped.append({"case": str(image_path), "reason": "missing_label"})
            continue
        print(f"[{case_index}/{len(image_paths)}] box analysis: {sequence}/{image_path.name}", flush=True)
        gt, _ = reader.read_images([str(gt_path)])
        label_map_3d = gt[0].astype(np.int16, copy=False)
        gt_target = label_map_3d == target_label
        source_mask = source_mask_for_case(args, image_path, gt_target, prompt_index)
        if source_mask.shape != gt_target.shape:
            raise ValueError(f"Shape mismatch for {image_path}: source={source_mask.shape}, gt={gt_target.shape}")

        for z in range(source_mask.shape[0]):
            source_slice = source_mask[z]
            target_slice = gt_target[z]
            source_area = int(source_slice.sum(dtype=np.int64))
            target_area = int(target_slice.sum(dtype=np.int64))
            if source_area == 0 and args.source_mask == "pseudo":
                continue
            if target_area == 0 and args.source_mask == "gt":
                continue
            source_target_intersection = int(np.logical_and(source_slice, target_slice).sum(dtype=np.int64))
            for margin in margins:
                box = bbox_from_mask(source_slice, margin=margin)
                if box is None:
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                box_area = int((x1 - x0) * (y1 - y0))
                label_slice = label_map_3d[z]
                target_in_box = int((label_slice[y0:y1, x0:x1] == target_label).sum(dtype=np.int64))
                other_counts = other_label_counts(label_slice, target_label, box)
                other_labeled = int(sum(other_counts.values()))
                labeled = target_in_box + other_labeled
                background = box_area - labeled
                rows.append(
                    {
                        "sequence": sequence,
                        "case": image_path.name,
                        "z": z,
                        "prompt": args.prompt,
                        "target_label": target_label,
                        "source_mask": args.source_mask,
                        "margin": margin,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "source_area": source_area,
                        "target_area": target_area,
                        "source_target_intersection": source_target_intersection,
                        "source_slice_dice": dice(source_slice, target_slice),
                        "source_target_precision": safe_div(source_target_intersection, source_area),
                        "source_target_recall": safe_div(source_target_intersection, target_area),
                        "box_area": box_area,
                        "box_target_pixels": target_in_box,
                        "box_other_labeled_pixels": other_labeled,
                        "box_background_pixels": background,
                        "box_labeled_pixels": labeled,
                        "target_recall": safe_div(target_in_box, target_area),
                        "box_target_fraction": safe_div(target_in_box, box_area),
                        "box_other_label_fraction": safe_div(other_labeled, box_area),
                        "other_fraction_among_labeled": safe_div(other_labeled, labeled),
                        "box_area_over_target": safe_div(box_area, target_area),
                        "other_label_counts_json": json.dumps(other_counts, sort_keys=True),
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "box_prompt_detail.csv"
    if rows:
        with detail_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary_rows = summarize_rows(rows)
        summary_path = output_dir / "box_prompt_summary.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"Saved detail: {detail_path}")
        print(f"Saved summary: {summary_path}")
        print("\nSummary:")
        for row in summary_rows:
            print(
                f"margin={row['margin']:>2} boxes={row['num_boxes']:>5} "
                f"target_recall={float(row['target_recall_weighted']):.4f} "
                f"target_frac={float(row['box_target_fraction_weighted']):.4f} "
                f"other_frac={float(row['box_other_label_fraction_weighted']):.4f} "
                f"other_among_labeled={float(row['other_fraction_among_labeled_weighted']):.4f} "
                f"area/target={float(row['box_area_over_target_weighted']):.2f} "
                f"source_precision={float(row['source_target_precision_weighted']):.4f}"
            )
        print_recommendation(summary_rows, args.target_recall_min, args.other_fraction_max)
    else:
        print("No boxes were analyzed.")

    if skipped:
        skipped_path = output_dir / "box_prompt_skipped.json"
        skipped_path.write_text(json.dumps(skipped, indent=2))
        print(f"Skipped {len(skipped)} cases. Details: {skipped_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MedSAM box prompt purity before running SAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--pseudo-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/box_prompt_analysis")
    parser.add_argument("--sequences", nargs="+", default=["P0"])
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--label-values", nargs="+", type=int, default=None)
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--source-mask", choices=["pseudo", "gt"], default="pseudo")
    parser.add_argument("--pseudo-class-index", type=int, default=None, help="Pseudo_prob channel to analyze. Default: 0 for single-class files, otherwise prompt index.")
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--margins", default="0 1 2 3 5 8 10")
    parser.add_argument("--target-recall-min", type=float, default=0.98)
    parser.add_argument("--other-fraction-max", type=float, default=0.05)
    parser.add_argument("--case-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
