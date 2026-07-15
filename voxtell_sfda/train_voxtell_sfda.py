#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from tqdm import tqdm

from voxtell_sfda.data import (
    ReliablePseudoPatchDataset,
    discover_image_paths,
    load_test_cases_file,
    split_image_paths_by_case,
    split_image_paths_by_test_cases,
)
from voxtell_sfda.infer_voxtell_sfda import (
    combine_binary_masks,
    compute_metrics_from_label_map,
    resolve_label_values,
    write_metrics,
)
from voxtell_sfda.losses import srpl_sfda_loss
from voxtell_sfda.modeling import (
    configure_trainable_parameters,
    expand_text_embeddings,
    load_voxtell_predictor,
)
from voxtell_sfda.prompts import DEFAULT_PROMPTS, print_label_mapping


def parse_patch_size(value: str | None, fallback) -> tuple[int, int, int]:
    if value is None:
        return tuple(int(v) for v in fallback)
    parts = [int(v) for v in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError("--patch-size must contain three integers, for example 192,192,192")
    return tuple(parts)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str, gpu: int) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if device_name == "cuda":
        print("CUDA is unavailable; using CPU.")
    return torch.device("cpu")


def save_checkpoint(
    output_dir: Path,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "iteration": iteration,
        "network_weights": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    atomic_torch_save(checkpoint, output_dir / f"checkpoint_iter_{iteration}.pth")
    atomic_torch_save(checkpoint, output_dir / "checkpoint_latest.pth")


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def save_best_checkpoint(
    output_dir: Path,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metric: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "iteration": iteration,
        "best_metric": metric,
        "network_weights": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    atomic_torch_save(checkpoint, output_dir / f"checkpoint_best_iter_{iteration}_dice_{metric:.4f}.pth")
    atomic_torch_save(checkpoint, output_dir / "checkpoint_best.pth")


def baseline_path_for_image(baseline_root: Path, image_path: Path) -> Path:
    sequence_path = baseline_root / image_path.parent.name / image_path.name
    if sequence_path.exists():
        return sequence_path
    return baseline_root / image_path.name


def update_extreme_cases(
    cases: list[dict],
    item: dict,
    keep: int,
    reverse: bool,
) -> list[dict]:
    cases.append(item)
    cases.sort(key=lambda x: x["case_delta"], reverse=reverse)
    return cases[:keep]


def save_extreme_cases(
    cases: list[dict],
    val_root: Path,
    group_name: str,
    reader_writer: NibabelIOWithReorient,
) -> None:
    for item in cases:
        output_path = val_root / group_name / item["sequence"] / item["case"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reader_writer.write_seg(item["combined"], str(output_path), item["properties"])


def foreground_band_from_pseudo(pseudo_prob: torch.Tensor, radius: int) -> torch.Tensor:
    foreground = (pseudo_prob > 0.5).float()
    if radius <= 0:
        return foreground
    radius = int(radius)
    kernel_size = 2 * radius + 1
    return F.max_pool3d(foreground, kernel_size=kernel_size, stride=1, padding=radius)


def apply_reliable_supervision_mode(
    pseudo_prob: torch.Tensor,
    reliable_mask: torch.Tensor,
    mode: str,
    foreground_band_radius: int,
    reliable_min_weight: float,
) -> torch.Tensor:
    if mode == "mask":
        mask = reliable_mask
    else:
        foreground_band = foreground_band_from_pseudo(pseudo_prob, foreground_band_radius)
        if mode == "foreground_band":
            mask = reliable_mask * foreground_band
        elif mode == "positive_only":
            mask = reliable_mask * (pseudo_prob > 0.5).float()
        else:
            raise ValueError(f"Unknown reliable supervision mode: {mode}")
    if reliable_min_weight > 0:
        mask = mask * (mask >= reliable_min_weight).float()
    return mask


@torch.no_grad()
def validate_and_save_masks(
    args: argparse.Namespace,
    predictor,
    image_paths: list[Path],
    prompts: list[str],
    label_values: list[int],
    output_dir: Path,
    iteration: int,
    writer: SummaryWriter,
) -> dict[str, float] | None:
    if args.val_case_limit is not None:
        image_paths = image_paths[: args.val_case_limit]

    predictor.network.eval()
    reader_writer = NibabelIOWithReorient()
    val_root = output_dir / "validation" / f"iter_{iteration:06d}"
    metric_rows = []
    top_improved: list[dict] = []
    top_worsened: list[dict] = []
    baseline_root = Path(args.baseline_pred_dir)
    for image_index, image_path in enumerate(image_paths, start=1):
        sequence = image_path.parent.name
        print(f"[val iter {iteration}] [{image_index}/{len(image_paths)}] {sequence}/{image_path.name}")
        image, properties = reader_writer.read_images([str(image_path)])
        segmentations = predictor.predict_single_image(image, prompts, output_type="binary")
        combined = combine_binary_masks(segmentations, label_values)

        gt_path = Path(args.data_root) / "labels" / sequence / image_path.name
        if not gt_path.exists():
            continue
        gt, _ = reader_writer.read_images([str(gt_path)])
        dice, iou = compute_metrics_from_label_map(
            segmentations,
            gt[0].astype(np.int16, copy=False),
            label_values,
        )
        baseline_dice = [np.nan for _ in prompts]
        baseline_iou = [np.nan for _ in prompts]
        baseline_file = baseline_path_for_image(baseline_root, image_path)
        if baseline_file.exists():
            baseline, _ = reader_writer.read_images([str(baseline_file)])
            baseline_binary = np.stack(
                [(baseline[0] == label_value) for label_value in label_values],   # 按照VoxTell的多类别label map读取，不要求是单独liver二值mask
                axis=0,
            ).astype(np.uint8)
            baseline_dice, baseline_iou = compute_metrics_from_label_map(
                baseline_binary,
                gt[0].astype(np.int16, copy=False),
                label_values,
            )

        case_deltas = []
        for prompt_index, prompt in enumerate(prompts):
            dice_delta = dice[prompt_index] - baseline_dice[prompt_index] if np.isfinite(baseline_dice[prompt_index]) else np.nan
            iou_delta = iou[prompt_index] - baseline_iou[prompt_index] if np.isfinite(baseline_iou[prompt_index]) else np.nan
            if np.isfinite(dice_delta):
                case_deltas.append(float(dice_delta))
            metric_rows.append(
                {
                    "sequence": sequence,
                    "case": image_path.name,
                    "label": label_values[prompt_index],
                    "prompt": prompt,
                    "dice": dice[prompt_index],
                    "iou": iou[prompt_index],
                    "baseline_dice": baseline_dice[prompt_index],
                    "baseline_iou": baseline_iou[prompt_index],
                    "dice_delta": dice_delta,
                    "iou_delta": iou_delta,
                }
            )
        if case_deltas and args.save_extreme_cases > 0:
            case_item = {
                "sequence": sequence,
                "case": image_path.name,
                "case_delta": float(np.mean(case_deltas)),
                "combined": combined,
                "properties": properties,
            }
            top_improved = update_extreme_cases(top_improved, case_item, args.save_extreme_cases, reverse=True)
            top_worsened = update_extreme_cases(top_worsened, case_item, args.save_extreme_cases, reverse=False)

    if metric_rows:
        write_metrics(metric_rows, val_root, prompts, label_values)
        save_extreme_cases(top_improved, val_root, "top_improved_vs_voxtell", reader_writer)
        save_extreme_cases(top_worsened, val_root, "top_worsened_vs_voxtell", reader_writer)
        for label_value, prompt in zip(label_values, prompts):
            rows = [r for r in metric_rows if int(r["label"]) == label_value and r["prompt"] == prompt]
            if rows:
                writer.add_scalar(f"val/{prompt}_dice", float(np.mean([float(r["dice"]) for r in rows])), iteration)
                writer.add_scalar(f"val/{prompt}_iou", float(np.mean([float(r["iou"]) for r in rows])), iteration)
        mean_dice = float(np.mean([float(r["dice"]) for r in metric_rows]))
        mean_iou = float(np.mean([float(r["iou"]) for r in metric_rows]))
        writer.add_scalar("val/mean_dice", mean_dice, iteration)
        writer.add_scalar("val/mean_iou", mean_iou, iteration)
        finite_baseline_dice = [
            float(r["baseline_dice"])
            for r in metric_rows
            if np.isfinite(float(r["baseline_dice"]))
        ]
        result = {"mean_dice": mean_dice, "mean_iou": mean_iou}
        if finite_baseline_dice:
            result["mean_baseline_dice"] = float(np.mean(finite_baseline_dice))
        return result
    return None


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = get_device(args.device, args.gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    print_label_mapping(args.prompts)
    label_values = resolve_label_values(args.prompts, args.label_values)
    print("Validation/output label values:")
    for prompt, label_value in zip(args.prompts, label_values):
        print(f"  {label_value:2d}: {prompt}")
    predictor = load_voxtell_predictor(Path(args.voxtell_root), Path(args.model_dir), device)
    network = predictor.network.to(device)
    text_embeddings_cpu = predictor.embed_text_prompts(args.prompts).clone().detach().float()
    patch_size = parse_patch_size(args.patch_size, predictor.patch_size)

    trainable_params = configure_trainable_parameters(
        network,
        args.trainable,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        lora_attention=args.lora_attention,
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    image_paths = discover_image_paths(Path(args.data_root), args.sequences)
    if args.case_limit is not None:
        image_paths = image_paths[: args.case_limit]
    if args.test_cases_file:
        train_image_paths, test_image_paths = split_image_paths_by_test_cases(  # 按照测试集set划分
            image_paths,
            load_test_cases_file(Path(args.test_cases_file)),
        )
    else:
        train_image_paths, test_image_paths = split_image_paths_by_case(  # 随机划分
            image_paths,
            train_ratio=args.train_ratio,
            seed=args.split_seed,
        )
    split_manifest = {
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "test_cases_file": args.test_cases_file,
        "num_images": len(image_paths),
        "num_train_images": len(train_image_paths),
        "num_test_images": len(test_image_paths),
        "train": [str(p) for p in train_image_paths],
        "test": [str(p) for p in test_image_paths],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))
    print(
        f"Case-level split: train={len(train_image_paths)} images, "
        f"test={len(test_image_paths)} images, ratio={args.train_ratio:.2f}, seed={args.split_seed}"
    )
    dataset = ReliablePseudoPatchDataset(
        image_paths=train_image_paths,
        pseudo_root=Path(args.pseudo_dir),
        preprocess_fn=predictor.preprocess,
        patch_size=patch_size,
        steps_per_epoch=args.steps_per_epoch,
        sampling=args.sampling,
        cache_size=args.cache_size,
        pseudo_volume_min=args.pseudo_volume_min,
        pseudo_volume_max=args.pseudo_volume_max,
        reliable_ratio_min=args.reliable_ratio_min,
        reliable_ratio_max=args.reliable_ratio_max,
        missing_reliable_mode=args.missing_reliable_mode,
        missing_reliable_uncertainty_threshold=args.missing_reliable_uncertainty_threshold,
        missing_reliable_consistency_threshold=args.missing_reliable_consistency_threshold,
        image_view_mode=args.train_image_view_mode,
    )
    if dataset.skipped_cases:
        skipped_path = output_dir / "skipped_pseudo_cases.json"
        skipped_path.write_text(json.dumps(dataset.skipped_cases, indent=2))
        print(f"Skipped {len(dataset.skipped_cases)} pseudo-label cases. Details: {skipped_path}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metadata = {
        "method": "SRPL-SFDA adapted to VoxTell",
        "data_root": args.data_root,
        "pseudo_dir": args.pseudo_dir,
        "model_dir": args.model_dir,
        "voxtell_root": args.voxtell_root,
        "prompts": args.prompts,
        "patch_size": patch_size,
        "sampling": args.sampling,
        "num_images": len(image_paths),
        "num_test_images": len(test_image_paths),
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "test_cases_file": args.test_cases_file,
        "num_train_images": len(dataset.image_paths),
        "num_skipped_pseudo_cases": len(dataset.skipped_cases),
        "pseudo_volume_min": args.pseudo_volume_min,
        "pseudo_volume_max": args.pseudo_volume_max,
        "reliable_ratio_min": args.reliable_ratio_min,
        "reliable_ratio_max": args.reliable_ratio_max,
        "reliable_supervision": args.reliable_supervision,
        "foreground_band_radius": args.foreground_band_radius,
        "missing_reliable_mode": args.missing_reliable_mode,
        "missing_reliable_uncertainty_threshold": args.missing_reliable_uncertainty_threshold,
        "missing_reliable_consistency_threshold": args.missing_reliable_consistency_threshold,
        "reliable_min_weight": args.reliable_min_weight,
        "train_image_view_mode": args.train_image_view_mode,
        "pseudo_target": args.pseudo_target,
        "trainable": args.trainable,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target": args.lora_target,
        "lora_attention": args.lora_attention,
        "num_trainable_params": sum(p.numel() for p in trainable_params),
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    writer.add_text("config/json", json.dumps(metadata, indent=2), 0)

    best_val_dice = float("-inf")
    baseline_patience_count = 0
    no_improve_count = 0
    iteration = 0
    progress = tqdm(total=args.max_iterations, ncols=100)
    while iteration < args.max_iterations:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True).float()
            pseudo_prob = batch["pseudo_prob"].to(device, non_blocking=True).float()
            reliable_mask = batch["reliable_mask"].to(device, non_blocking=True).float()
            reliable_mask = apply_reliable_supervision_mode(
                pseudo_prob=pseudo_prob,
                reliable_mask=reliable_mask,
                mode=args.reliable_supervision,
                foreground_band_radius=args.foreground_band_radius,
                reliable_min_weight=args.reliable_min_weight,
            )
            batch_text_embeddings = expand_text_embeddings(text_embeddings_cpu, image.shape[0], device)

            network.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = network(image, batch_text_embeddings)
            with torch.amp.autocast("cuda", enabled=False):
                loss, stats = srpl_sfda_loss(
                    logits=logits,
                    pseudo_prob=pseudo_prob,
                    reliable_mask=reliable_mask,
                    entropy_weight=args.entropy_weight,
                    pseudo_target=args.pseudo_target,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss encountered. "
                    f"stats={stats}. Try lowering --lr or running with --no-amp."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            iteration += 1
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss_total", stats["loss_total"], iteration)
            writer.add_scalar("train/loss_bce", stats["loss_bce"], iteration)
            writer.add_scalar("train/loss_dice", stats["loss_dice"], iteration)
            writer.add_scalar("train/loss_mse", stats["loss_mse"], iteration)
            writer.add_scalar("train/loss_entropy", stats["loss_entropy"], iteration)
            writer.add_scalar("train/reliable_ratio", stats["reliable_ratio"], iteration)
            writer.add_scalar("train/lr", current_lr, iteration)
            writer.add_scalar("train/amp_scale", scaler.get_scale(), iteration)
            if device.type == "cuda":
                writer.add_scalar("memory/allocated_gb", torch.cuda.memory_allocated(device) / (1024 ** 3), iteration)
                writer.add_scalar("memory/reserved_gb", torch.cuda.memory_reserved(device) / (1024 ** 3), iteration)
            progress.update(1)
            progress.set_postfix(
                loss=f"{stats['loss_total']:.4f}",
                rel=f"{stats['reliable_ratio']:.3f}",
                ent=f"{stats['loss_entropy']:.4f}",
            )

            if iteration % args.log_every == 0:
                print(
                    f"iter={iteration} "
                    f"loss={stats['loss_total']:.6f} "
                    f"bce={stats['loss_bce']:.6f} "
                    f"dice={stats['loss_dice']:.6f} "
                    f"mse={stats['loss_mse']:.6f} "
                    f"entropy={stats['loss_entropy']:.6f} "
                    f"reliable={stats['reliable_ratio']:.4f}"
                )

            if iteration % args.save_every == 0:
                save_checkpoint(output_dir, network, optimizer, iteration, args)
                writer.add_scalar("checkpoint/iteration", iteration, iteration)

            if args.val_every > 0 and iteration % args.val_every == 0:
                val_result = validate_and_save_masks(
                    args=args,
                    predictor=predictor,
                    image_paths=test_image_paths,
                    prompts=args.prompts,
                    label_values=label_values,
                    output_dir=output_dir,
                    iteration=iteration,
                    writer=writer,
                )
                if val_result is not None:
                    val_dice = val_result["mean_dice"]
                    baseline_dice = val_result.get("mean_baseline_dice")
                    if val_dice > best_val_dice + args.early_stop_min_delta:
                        best_val_dice = val_dice
                        no_improve_count = 0
                        save_best_checkpoint(output_dir, network, optimizer, iteration, val_dice, args)
                        writer.add_scalar("checkpoint/best_val_dice", val_dice, iteration)
                    else:
                        no_improve_count += 1

                    if baseline_dice is not None:
                        writer.add_scalar("val/mean_baseline_dice", baseline_dice, iteration)
                        writer.add_scalar("val/dice_minus_baseline", val_dice - baseline_dice, iteration)
                        if args.stop_below_baseline and val_dice < baseline_dice - args.baseline_margin:
                            baseline_patience_count += 1
                        else:
                            baseline_patience_count = 0

                    if args.stop_below_baseline and baseline_patience_count >= args.baseline_patience:
                        print(
                            f"Early stop at iter={iteration}: val_dice={val_dice:.6f} "
                            f"is below baseline={baseline_dice:.6f} "
                            f"for {baseline_patience_count} validation(s)."
                        )
                        iteration = args.max_iterations
                        break

                    if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                        print(
                            f"Early stop at iter={iteration}: no validation improvement "
                            f"for {no_improve_count} validation(s)."
                        )
                        iteration = args.max_iterations
                        break
                network.train()

            if device.type == "cuda" and iteration % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

            if iteration >= args.max_iterations:
                break

    progress.close()
    save_checkpoint(output_dir, network, optimizer, iteration, args)
    writer.close()
    print(f"Training finished. Latest checkpoint: {output_dir / 'checkpoint_latest.pth'}")
    print(f"TensorBoard logs: {tensorboard_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt pretrained VoxTell with SRPL-SFDA reliable pseudo-label supervision.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--pseudo-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_reliable_medsam_t035_fg_zaxis")
    parser.add_argument("--voxtell-root", default="/data/zy/VoxTell_from_disk")
    parser.add_argument("--model-dir", default="/data/zy/VoxTell_from_disk/model")
    parser.add_argument("--output-dir", default="/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted_lora_qkv")
    parser.add_argument("--tensorboard-dir", default=None, help="Default: <output-dir>/tensorboard")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--label-values", nargs="+", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--patch-size", default=None, help="Three integers. Default: VoxTell plans patch_size.")
    parser.add_argument(
        "--sampling",
        choices=["full_resize", "foreground_crop", "random_crop"],
        default="full_resize",
        help="full_resize uses the entire volume resized to patch_size; no random crop.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Case-level train split ratio. Test split is used for validation.")
    parser.add_argument("--split-seed", type=int, default=2024, help="Seed for deterministic case-level train/test split.")
    parser.add_argument("--test-cases-file", default=None, help="JSON file with fixed test_cases/test list. Overrides random split.")
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument(
        "--reliable-supervision",
        choices=["mask", "foreground_band", "positive_only"],
        default="mask",
        help=(
            "mask uses saved reliable_mask unchanged; foreground_band keeps reliable voxels only "
            "near pseudo foreground; positive_only supervises only pseudo foreground."
        ),
    )
    parser.add_argument("--foreground-band-radius", type=int, default=8)
    parser.add_argument("--pseudo-volume-min", type=float, default=None, help="按伪标签前景体素比例过滤case")
    parser.add_argument("--pseudo-volume-max", type=float, default=None)
    parser.add_argument("--reliable-ratio-min", type=float, default=None, help="按reliable mask占比过滤case")
    parser.add_argument("--reliable-ratio-max", type=float, default=None)
    parser.add_argument("--missing-reliable-mode", choices=["ones", "confidence", "foreground", "t3ie_consensus"], default="ones",
                        help="当.npz 伪标签没有reliable_mask时，如何生成可靠区域.ones:全图都可靠; "
                             "confidence:按abs(prob-0.5)*2生成可靠权重。越接近0或1越可靠，接近0.5越不可靠,适合 soft pseudo;"
                             "foreground:只把prob>0.5的前景作为可靠区域; "
                             "t3ie_consensus:用T3IE的uncertainty和consistency判断可靠区域")
    parser.add_argument("--missing-reliable-uncertainty-threshold", type=float, default=0.25, help="t3ie_consensus用。uncertainty小于等于这个值才可靠")
    parser.add_argument("--missing-reliable-consistency-threshold", type=float, default=0.05)
    parser.add_argument("--reliable-min-weight", type=float, default=0.0,help="给reliable mask设置最小权重。0表示不可靠区域不做伪标签监督，只做entropy loss")
    parser.add_argument("--train-image-view-mode", choices=["original", "t3ie_random", "t3ie_cycle"], default="original")
    parser.add_argument("--pseudo-target", choices=["hard", "soft"], default="soft")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=12.0)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--trainable", choices=["lora", "prompt_decoder", "decoder", "all"], default="lora")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target",
        choices=["attention", "prompt_decoder", "decoder", "all"],
        default="attention",
        help="Module group where LoRA modules are inserted when --trainable lora.",
    )
    parser.add_argument(
        "--lora-attention",
        choices=["none", "q", "v", "qv", "qkv", "qkvo"],
        default="qkv",
        help="Attention matrices to adapt in nn.MultiheadAttention. Use with --lora-target attention/decoder/prompt_decoder/all.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--empty-cache-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--val-case-limit", type=int, default=None)
    parser.add_argument("--baseline-pred-dir", default="/data/zy/VoxTell_from_disk/out_multi")
    parser.add_argument("--stop-below-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline-margin", type=float, default=0.0)
    parser.add_argument("--baseline-patience", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--save-extreme-cases", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
