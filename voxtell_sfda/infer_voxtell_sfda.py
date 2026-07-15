#!/usr/bin/env python3
from __future__ import annotations
'''
推理时不只预测原图，也预测 T3IE 增强后的图，然后平均概率。这是目前唯一稳定超过 zero-shot 的方法
'''
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.data import discover_image_paths, select_image_split
from voxtell_sfda.generate_t3ie_voxtell_pseudo import predict_t3ie_probabilities
from voxtell_sfda.modeling import (
    apply_lora_to_linear_modules,
    configure_trainable_parameters,
    load_voxtell_predictor,
)
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping


DEFAULT_PROMPT_TO_LABEL = {prompt: index + 1 for index, prompt in enumerate(DEFAULT_PROMPTS)}


def load_adapted_weights(predictor, checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("network_weights", checkpoint)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if checkpoint_args.get("trainable") == "lora":
        if "lora_attention" in checkpoint_args:
            configure_trainable_parameters(
                predictor.network,
                "lora",
                lora_rank=int(checkpoint_args.get("lora_rank", 8)),
                lora_alpha=float(checkpoint_args.get("lora_alpha", 16.0)),
                lora_dropout=float(checkpoint_args.get("lora_dropout", 0.0)),
                lora_target=checkpoint_args.get("lora_target", "prompt_decoder"),
                lora_attention=checkpoint_args.get("lora_attention", "none"),
            )
        else:
            apply_lora_to_linear_modules(
                predictor.network,
                rank=int(checkpoint_args.get("lora_rank", 8)),
                alpha=float(checkpoint_args.get("lora_alpha", 16.0)),
                dropout=float(checkpoint_args.get("lora_dropout", 0.0)),
                target=checkpoint_args.get("lora_target", "prompt_decoder"),
                skip_multihead_attention=False,
            )
    predictor.network.load_state_dict(weights)
    return int(checkpoint.get("iteration", -1)) if isinstance(checkpoint, dict) else -1


def combine_binary_masks(segmentations: np.ndarray, label_values: list[int]) -> np.ndarray:
    combined = np.zeros_like(segmentations[0], dtype=np.uint8)
    for class_index, segmentation in enumerate(segmentations):
        combined[segmentation > 0] = label_values[class_index]
    return combined


def compute_metrics_from_label_map(
    pred: np.ndarray,
    gt_label_map: np.ndarray,
    label_values: list[int],
) -> tuple[list[float], list[float]]:
    if pred.shape[1:] != gt_label_map.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt_label_map.shape}")

    dice = []
    iou = []
    eps = 1e-7
    for class_index in range(pred.shape[0]):
        pred_class = pred[class_index].astype(bool)
        gt_class = gt_label_map == label_values[class_index]
        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()
        pred_sum = pred_class.sum()
        gt_sum = gt_class.sum()
        dice.append(float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps)))
        iou.append(float((intersection + eps) / (union + eps)))
    return dice, iou


def resolve_label_values(prompts: list[str], label_values_arg: list[int] | None) -> list[int]:
    if label_values_arg is not None:
        if len(label_values_arg) != len(prompts):
            raise ValueError(
                f"--label-values length ({len(label_values_arg)}) must match "
                f"--prompts length ({len(prompts)})"
            )
        return [int(v) for v in label_values_arg]

    label_values = []
    for index, prompt in enumerate(prompts):
        label_values.append(DEFAULT_PROMPT_TO_LABEL.get(prompt, index + 1))
    return label_values


def write_metrics(
    metric_rows: list[dict],
    output_root: Path,
    prompts: list[str],
    label_values: list[int],
) -> None:
    metrics_path = output_root / "metrics_per_case.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Saved per-case metrics: {metrics_path}")

    summary_rows = []
    print("\nMetric summary:")
    print(f"{'label':>5s}  {'prompt':22s}  {'mean_dice':>10s}  {'mean_iou':>10s}  {'n':>5s}")
    for label_value, prompt in zip(label_values, prompts):
        rows = [r for r in metric_rows if int(r["label"]) == label_value and r["prompt"] == prompt]
        if not rows:
            continue
        mean_dice = float(np.mean([float(r["dice"]) for r in rows]))
        mean_iou = float(np.mean([float(r["iou"]) for r in rows]))
        summary_rows.append(
            {
                "label": label_value,
                "prompt": prompt,
                "mean_dice": mean_dice,
                "mean_iou": mean_iou,
                "num_cases": len(rows),
            }
        )
        print(f"{label_value:5d}  {prompt:22s}  {mean_dice:10.4f}  {mean_iou:10.4f}  {len(rows):5d}")

    if summary_rows:
        overall_mean_dice = float(np.mean([r["mean_dice"] for r in summary_rows]))
        overall_miou = float(np.mean([r["mean_iou"] for r in summary_rows]))
        summary_rows.append(
            {
                "label": "mean",
                "prompt": "overall",
                "mean_dice": overall_mean_dice,
                "mean_iou": overall_miou,
                "num_cases": "",
            }
        )
        print(f"{'mean':>5s}  {'overall':22s}  {overall_mean_dice:10.4f}  {overall_miou:10.4f}")

    summary_path = output_root / "metrics_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "prompt", "mean_dice", "mean_iou", "num_cases"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary metrics: {summary_path}")


def get_device(device_name: str, gpu: int) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if device_name == "cuda":
        print("CUDA is unavailable; using CPU.", file=sys.stderr)
    return torch.device("cpu")


def save_segmentation(segmentation: np.ndarray, output_path: Path, properties: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    NibabelIOWithReorient().write_seg(segmentation, str(output_path), properties)
    print(f"Saved: {output_path}")


@torch.no_grad()
def predict_segmentations(predictor, image: np.ndarray, prompts: list[str], t3ie_ensemble: bool) -> np.ndarray:
    if not t3ie_ensemble:
        return predictor.predict_single_image(image, prompts, output_type="binary")
    pseudo_prob, _, _ = predict_t3ie_probabilities(predictor, image, prompts)
    return (pseudo_prob > 0.5).astype(np.uint8)


def infer(args: argparse.Namespace) -> None:
    device = get_device(args.device, args.gpu)
    predictor = load_voxtell_predictor(Path(args.voxtell_root), Path(args.model_dir), device)
    if args.checkpoint:
        iteration = load_adapted_weights(predictor, Path(args.checkpoint))
        print(f"Loaded adapted checkpoint: {args.checkpoint} (iteration={iteration})")
    else:
        print("No adapted checkpoint provided; using source VoxTell weights.")

    print_label_mapping(args.prompts)
    label_values = resolve_label_values(args.prompts, args.label_values)
    print("Evaluation/output label values:")
    for prompt, label_value in zip(args.prompts, label_values):
        print(f"  {label_value:2d}: {prompt}")

    image_paths = discover_image_paths(Path(args.data_root), args.sequences)
    image_paths = select_image_split(
        image_paths,
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        test_cases_file=Path(args.test_cases_file) if args.test_cases_file else None,
    )
    print(
        f"Evaluation split={args.split}: {len(image_paths)} images "
        f"(train_ratio={args.train_ratio:.2f}, split_seed={args.split_seed})"
    )
    reader_writer = NibabelIOWithReorient()
    output_root = Path(args.output_dir)
    metric_rows = []

    for image_index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        print(f"[{image_index}/{len(image_paths)}] {sequence}/{image_path.name}")
        image, properties = reader_writer.read_images([str(image_path)])
        segmentations = predict_segmentations(
            predictor=predictor,
            image=image,
            prompts=args.prompts,
            t3ie_ensemble=args.t3ie_ensemble,
        )
        combined = combine_binary_masks(segmentations, label_values)
        output_path = output_root / sequence / image_path.name
        save_segmentation(combined, output_path, properties)

        if args.evaluate:
            gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
            if not gt_path.exists():
                print(f"Missing label, skip metrics: {gt_path}", file=sys.stderr)
                continue
            gt, _ = reader_writer.read_images([str(gt_path)])
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
            print("  " + " ".join(
                f"{prompt}: Dice={d:.3f}, IoU={j:.3f}"
                for prompt, d, j in zip(args.prompts, dice, iou)
            ))

    if args.evaluate and metric_rows:
        write_metrics(metric_rows, output_root, args.prompts, label_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference/evaluation for source or adapted VoxTell-SFDA checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--voxtell-root", default="/data/zy/VoxTell_from_disk")
    parser.add_argument("--model-dir", default="/data/zy/VoxTell_from_disk/model")
    parser.add_argument("--checkpoint", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted_lora_qkv/checkpoint_best_iter_40_dice_0.8431.pth", help="Adapted VoxTell-SFDA checkpoint. If omitted, source VoxTell is used.")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted_lora_qkv/eval_no_t3ie_ensemble")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--test-cases-file", default=None, help="JSON file with fixed test_cases/test list. Overrides random split.")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--label-values",
        nargs="+",
        type=int,
        default=None,
        help="Ground-truth label value for each prompt. Default maps known prompts to the 11-class label ids.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--t3ie-ensemble",
        action="store_true",
        help="Average predictions from original and T3IE intensity-enhanced views before thresholding.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
