#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.data import discover_image_paths
from voxtell_sfda.infer_voxtell_sfda import compute_metrics_from_label_map, resolve_label_values, write_metrics
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping


def export(args: argparse.Namespace) -> None:
    print_label_mapping(args.prompts)
    label_values = resolve_label_values(args.prompts, args.label_values)
    reader = NibabelIOWithReorient()
    output_root = Path(args.output_dir)
    pseudo_root = Path(args.pseudo_dir)
    metric_rows = []

    for index, image_path in enumerate(discover_image_paths(Path(args.data_root), args.sequences), start=1):
        sequence = image_path.parent.name
        pseudo_path = pseudo_root / sequence / f"{image_path.name}.npz"
        if not pseudo_path.exists():
            raise FileNotFoundError(f"Missing pseudo file: {pseudo_path}")

        image, properties = reader.read_images([str(image_path)])
        pseudo = np.load(pseudo_path)
        segmentations = (pseudo["pseudo_prob"] > args.prob_threshold).astype(np.uint8)
        combined = np.zeros(segmentations.shape[1:], dtype=np.uint8)
        for class_index, label_value in enumerate(label_values):
            combined[segmentations[class_index] > 0] = label_value

        output_path = output_root / sequence / image_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reader.write_seg(combined, str(output_path), properties)
        print(f"[{index}] saved={output_path}")

        if args.evaluate:
            gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
            gt, _ = reader.read_images([str(gt_path)])
            dice, iou = compute_metrics_from_label_map(
                segmentations,
                gt[0].astype(np.int16, copy=False),
                label_values,
            )
            for prompt_index, prompt in enumerate(args.prompts):
                metric_rows.append(
                    {
                        "sequence": sequence,
                        "case": image_path.name,
                        "label": label_values[prompt_index],
                        "prompt": prompt,
                        "dice": dice[prompt_index],
                        "iou": iou[prompt_index],
                    }
                )

    if args.evaluate and metric_rows:
        write_metrics(metric_rows, output_root, args.prompts, label_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pseudo-label npz files as NIfTI predictions and optionally evaluate them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--label-values", nargs="+", type=int, default=None)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
