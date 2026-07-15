from __future__ import annotations

import random
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell_sfda.t3ie import t3ie_views


def discover_image_paths(data_root: Path, sequences: Iterable[str] | None) -> list[Path]:
    image_root = data_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    selected_sequences = list(sequences) if sequences else sorted(p.name for p in image_root.iterdir() if p.is_dir())
    image_paths: list[Path] = []
    for sequence in selected_sequences:
        sequence_dir = image_root / sequence
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"Sequence image directory does not exist: {sequence_dir}")
        image_paths.extend(sorted(sequence_dir.glob("*.nii.gz")))
    if not image_paths:
        raise RuntimeError(f"No .nii.gz images found under {image_root}")
    return image_paths


def split_image_paths_by_case(
    image_paths: list[Path],
    train_ratio: float = 0.8,
    seed: int = 2024,
) -> tuple[list[Path], list[Path]]:
    """Split by case filename so phases/sequences of the same case do not leak."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    case_names = sorted({p.name for p in image_paths})
    if len(case_names) < 2:
        raise ValueError("Need at least two unique cases for a train/test split.")

    rng = random.Random(seed)
    shuffled = case_names[:]
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_cases = set(shuffled[:train_count])

    train_paths = [p for p in image_paths if p.name in train_cases]
    test_paths = [p for p in image_paths if p.name not in train_cases]
    return train_paths, test_paths


def split_image_paths_by_test_cases(
    image_paths: list[Path],
    test_case_names: Iterable[str],
) -> tuple[list[Path], list[Path]]:
    test_cases = set(test_case_names)
    if not test_cases:
        raise ValueError("test_case_names is empty.")
    known_cases = {p.name for p in image_paths}
    missing = sorted(test_cases - known_cases)
    if missing:
        raise ValueError(f"Test cases are not present in image paths: {missing[:5]}")
    train_paths = [p for p in image_paths if p.name not in test_cases]
    test_paths = [p for p in image_paths if p.name in test_cases]
    if not train_paths or not test_paths:
        raise ValueError(f"Invalid split from fixed test cases: train={len(train_paths)}, test={len(test_paths)}")
    return train_paths, test_paths


def load_test_cases_file(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        values = data
    elif isinstance(data, dict):
        values = data.get("test_cases", data.get("test", []))
    else:
        raise ValueError(f"Unsupported test cases file format: {path}")
    cases = []
    for value in values:
        case = Path(str(value)).name
        cases.append(case)
    if not cases:
        raise ValueError(f"No test cases found in {path}")
    return cases


def select_image_split(
    image_paths: list[Path],
    split: str,
    train_ratio: float = 0.8,
    seed: int = 2024,
    test_cases_file: Path | None = None,
) -> list[Path]:
    if split == "all":
        return image_paths
    if test_cases_file is not None:
        train_paths, test_paths = split_image_paths_by_test_cases(image_paths, load_test_cases_file(test_cases_file))
    else:
        train_paths, test_paths = split_image_paths_by_case(image_paths, train_ratio=train_ratio, seed=seed)
    if split == "train":
        return train_paths
    if split == "test":
        return test_paths
    raise ValueError(f"Unknown split: {split}")


def case_key_from_image_path(image_path: Path) -> tuple[str, str]:
    return image_path.parent.name, image_path.name


def pseudo_path_for_image(pseudo_root: Path, image_path: Path) -> Path:
    sequence, filename = case_key_from_image_path(image_path)
    return pseudo_root / sequence / f"{filename}.npz"


def crop_to_bbox(array: np.ndarray, bbox) -> np.ndarray:
    slicer = tuple(slice(int(axis_bbox[0]), int(axis_bbox[1])) for axis_bbox in bbox)
    if array.ndim == len(slicer):
        return array[slicer]
    if array.ndim == len(slicer) + 1:
        return array[(slice(None), *slicer)]
    raise ValueError(f"Cannot apply bbox with {len(slicer)} spatial axes to array shape {array.shape}")


class RandomVoxTellPatchDataset(Dataset):
    """Random target-domain 3D patches for VoxTell SFDA.

    The dataset does not read labels. It is retained for ablation/debugging.
    """

    def __init__(
        self,
        image_paths: list[Path],
        preprocess_fn,
        patch_size: tuple[int, int, int],
        steps_per_epoch: int,
        cache_size: int = 4,
    ) -> None:
        self.image_paths = image_paths
        self.preprocess_fn = preprocess_fn
        self.patch_size = tuple(int(v) for v in patch_size)
        self.steps_per_epoch = steps_per_epoch
        self.cache_size = cache_size
        self.cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self.reader = NibabelIOWithReorient()

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _load_volume(self, path: Path) -> torch.Tensor:
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path)
            return cached

        image, _ = self.reader.read_images([str(path)])
        tensor, _, _ = self.preprocess_fn(image)
        tensor = tensor.float()

        self.cache[path] = tensor
        self.cache.move_to_end(path)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return tensor

    def _pad_to_patch_size(self, volume: torch.Tensor) -> torch.Tensor:
        spatial = volume.shape[1:]
        pads = []
        for size, target in zip(reversed(spatial), reversed(self.patch_size)):
            missing = max(target - size, 0)
            pads.extend([missing // 2, missing - missing // 2])
        if any(pads):
            volume = F.pad(volume, pads)
        return volume

    def _random_crop(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self._pad_to_patch_size(volume)
        starts = []
        for size, target in zip(volume.shape[1:], self.patch_size):
            starts.append(random.randint(0, max(size - target, 0)))
        sx, sy, sz = starts
        px, py, pz = self.patch_size
        return volume[:, sx : sx + px, sy : sy + py, sz : sz + pz]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = random.choice(self.image_paths)
        volume = self._load_volume(path)
        patch = self._random_crop(volume)
        return {"image": patch, "case": str(path)}


class ReliablePseudoPatchDataset(Dataset):
    """Target-domain 3D patches with SRPL-SFDA reliable pseudo-labels."""

    def __init__(
        self,
        image_paths: list[Path],
        pseudo_root: Path,
        preprocess_fn,
        patch_size: tuple[int, int, int],
        steps_per_epoch: int,
        sampling: str = "full_resize",
        cache_size: int = 4,
        pseudo_volume_min: float | None = None,
        pseudo_volume_max: float | None = None,
        reliable_ratio_min: float | None = None,
        reliable_ratio_max: float | None = None,
        missing_reliable_mode: str = "ones",
        missing_reliable_uncertainty_threshold: float = 0.25,
        missing_reliable_consistency_threshold: float = 0.05,
        image_view_mode: str = "original",
    ) -> None:
        self.pseudo_root = pseudo_root
        self.preprocess_fn = preprocess_fn
        self.patch_size = tuple(int(v) for v in patch_size)
        self.steps_per_epoch = steps_per_epoch
        self.sampling = sampling
        self.cache_size = cache_size
        self.missing_reliable_mode = missing_reliable_mode
        self.missing_reliable_uncertainty_threshold = float(missing_reliable_uncertainty_threshold)
        self.missing_reliable_consistency_threshold = float(missing_reliable_consistency_threshold)
        self.image_view_mode = image_view_mode
        self.cache: OrderedDict[Path, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.reader = NibabelIOWithReorient()
        if self.sampling not in {"full_resize", "foreground_crop", "random_crop"}:
            raise ValueError(f"Unknown sampling mode: {self.sampling}")
        if self.image_view_mode not in {"original", "t3ie_random", "t3ie_cycle"}:
            raise ValueError(f"Unknown image view mode: {self.image_view_mode}")

        missing = [p for p in image_paths if not pseudo_path_for_image(self.pseudo_root, p).exists()]
        if missing:
            preview = ", ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} pseudo-label files under {self.pseudo_root}. "
                f"First missing: {preview}"
            )
        self.image_paths, self.skipped_cases = self._filter_image_paths(
            image_paths=image_paths,
            pseudo_volume_min=pseudo_volume_min,
            pseudo_volume_max=pseudo_volume_max,
            reliable_ratio_min=reliable_ratio_min,
            reliable_ratio_max=reliable_ratio_max,
        )
        if not self.image_paths:
            raise RuntimeError("No pseudo-label cases remain after quality filtering.")

    def _filter_image_paths(
        self,
        image_paths: list[Path],
        pseudo_volume_min: float | None,
        pseudo_volume_max: float | None,
        reliable_ratio_min: float | None,
        reliable_ratio_max: float | None,
    ) -> tuple[list[Path], list[dict[str, float | str]]]:
        if all(v is None for v in (pseudo_volume_min, pseudo_volume_max, reliable_ratio_min, reliable_ratio_max)):
            return image_paths, []

        kept = []
        skipped: list[dict[str, float | str]] = []
        for image_path in image_paths:
            pseudo_npz = np.load(pseudo_path_for_image(self.pseudo_root, image_path))
            pseudo_volume = float((pseudo_npz["pseudo_prob"] > 0.5).mean())
            reliable_mask = self._read_reliable_mask(pseudo_npz)
            reliable_ratio = float(reliable_mask.mean())
            reason = None
            if pseudo_volume_min is not None and pseudo_volume < pseudo_volume_min:
                reason = "pseudo_volume_min"
            elif pseudo_volume_max is not None and pseudo_volume > pseudo_volume_max:
                reason = "pseudo_volume_max"
            elif reliable_ratio_min is not None and reliable_ratio < reliable_ratio_min:
                reason = "reliable_ratio_min"
            elif reliable_ratio_max is not None and reliable_ratio > reliable_ratio_max:
                reason = "reliable_ratio_max"

            if reason is None:
                kept.append(image_path)
            else:
                skipped.append(
                    {
                        "case": str(image_path),
                        "pseudo_volume": pseudo_volume,
                        "reliable_ratio": reliable_ratio,
                        "reason": reason,
                    }
                )
        return kept, skipped

    def _read_reliable_mask(self, pseudo_npz) -> np.ndarray:
        if "reliable_mask" in pseudo_npz:
            return pseudo_npz["reliable_mask"].astype(np.float32, copy=False)
        pseudo_prob = pseudo_npz["pseudo_prob"].astype(np.float32, copy=False)

        # 没有reliable_mask时，根据不同的模式生成不同的mask
        if self.missing_reliable_mode == "ones": # 全部可靠
            return np.ones_like(pseudo_prob, dtype=np.float32)
        if self.missing_reliable_mode == "confidence": # 置信度
            return np.abs(pseudo_prob - 0.5).astype(np.float32) * 2.0
        if self.missing_reliable_mode == "foreground": # 前景
            return (pseudo_prob > 0.5).astype(np.float32)
        if self.missing_reliable_mode == "t3ie_consensus": # T3IE一致性
            if "uncertainty" not in pseudo_npz or "consistency" not in pseudo_npz:
                raise KeyError("missing_reliable_mode=t3ie_consensus requires uncertainty and consistency in pseudo npz")
            uncertainty = pseudo_npz["uncertainty"].astype(np.float32, copy=False)
            consistency = pseudo_npz["consistency"].astype(np.float32, copy=False)
            return (
                (uncertainty <= self.missing_reliable_uncertainty_threshold) # 高置信度阈值
                & (consistency <= self.missing_reliable_consistency_threshold) # 低一致性阈值
            ).astype(np.float32)
        raise ValueError(f"Unknown missing reliable mask mode: {self.missing_reliable_mode}")

    def _select_image_view(self, image: torch.Tensor, index: int) -> torch.Tensor:
        if self.image_view_mode == "original": # 原始图像
            views = t3ie_views(image.unsqueeze(0))
        if self.image_view_mode == "t3ie_cycle": # 循环选择一个view
            return views[index % len(views)].squeeze(0)
        return random.choice(views).squeeze(0) # 随机选择一个view

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _load_case(self, image_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self.cache.get(image_path)
        if cached is not None:
            self.cache.move_to_end(image_path)
            return cached

        # cache中没有该文件，从原始数据中读取
        image, _ = self.reader.read_images([str(image_path)])
        image_tensor, bbox, _ = self.preprocess_fn(image)

        pseudo_file = pseudo_path_for_image(self.pseudo_root, image_path)
        pseudo_npz = np.load(pseudo_file)
        pseudo_prob = pseudo_npz["pseudo_prob"].astype(np.float32, copy=False)
        reliable_mask = self._read_reliable_mask(pseudo_npz)

        pseudo_prob = crop_to_bbox(pseudo_prob, bbox)
        reliable_mask = crop_to_bbox(reliable_mask, bbox)

        image_tensor = image_tensor.float()
        pseudo_tensor = torch.from_numpy(np.ascontiguousarray(pseudo_prob)).float()
        reliable_tensor = torch.from_numpy(np.ascontiguousarray(reliable_mask)).float()

        if image_tensor.shape[1:] != pseudo_tensor.shape[1:]:
            raise ValueError(
                f"Shape mismatch for {image_path}: image={image_tensor.shape}, "
                f"pseudo={pseudo_tensor.shape}, reliable={reliable_tensor.shape}"
            )

        # 将数据缓存
        value = (image_tensor, pseudo_tensor, reliable_tensor)
        self.cache[image_path] = value
        self.cache.move_to_end(image_path)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    def _pad_to_patch_size(
        self,
        image: torch.Tensor,
        pseudo: torch.Tensor,
        reliable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = image.shape[1:]
        pads = []
        for size, target in zip(reversed(spatial), reversed(self.patch_size)):
            missing = max(target - size, 0)
            pads.extend([missing // 2, missing - missing // 2])
        if any(pads):
            image = F.pad(image, pads)
            pseudo = F.pad(pseudo, pads)
            reliable = F.pad(reliable, pads)
        return image, pseudo, reliable

    def _random_crop(
        self,
        image: torch.Tensor,
        pseudo: torch.Tensor,
        reliable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, pseudo, reliable = self._pad_to_patch_size(image, pseudo, reliable)
        starts = []
        for size, target in zip(image.shape[1:], self.patch_size):
            starts.append(random.randint(0, max(size - target, 0)))
        sx, sy, sz = starts
        px, py, pz = self.patch_size
        spatial_crop = (slice(sx, sx + px), slice(sy, sy + py), slice(sz, sz + pz))
        return (
            image[(slice(None), *spatial_crop)],
            pseudo[(slice(None), *spatial_crop)],
            reliable[(slice(None), *spatial_crop)],
        )

    def _foreground_crop(
        self,
        image: torch.Tensor,
        pseudo: torch.Tensor,
        reliable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, pseudo, reliable = self._pad_to_patch_size(image, pseudo, reliable)
        positive = torch.nonzero((pseudo > 0.5).any(dim=0), as_tuple=False)
        if positive.numel() == 0:
            positive = torch.nonzero((reliable > 0).any(dim=0), as_tuple=False)
        if positive.numel() == 0:
            return self._random_crop(image, pseudo, reliable)

        center = positive[random.randrange(positive.shape[0])].tolist()
        starts = []
        for axis, target in enumerate(self.patch_size):
            size = image.shape[axis + 1]
            low = max(int(center[axis]) - target + 1, 0)
            high = min(int(center[axis]), size - target)
            starts.append(random.randint(low, max(low, high)))

        sx, sy, sz = starts
        px, py, pz = self.patch_size
        spatial_crop = (slice(sx, sx + px), slice(sy, sy + py), slice(sz, sz + pz))
        return (
            image[(slice(None), *spatial_crop)],
            pseudo[(slice(None), *spatial_crop)],
            reliable[(slice(None), *spatial_crop)],
        )

    def _full_resize(
        self,
        image: torch.Tensor,
        pseudo: torch.Tensor,
        reliable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = tuple(int(v) for v in self.patch_size)
        image_out = F.interpolate(
            image.unsqueeze(0),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        pseudo_out = F.interpolate(
            pseudo.unsqueeze(0),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).clamp(0.0, 1.0)
        reliable_out = F.interpolate(
            reliable.unsqueeze(0),
            size=target_size,
            mode="nearest",
        ).squeeze(0)
        return image_out, pseudo_out, reliable_out

    def _sample(
        self,
        image: torch.Tensor,
        pseudo: torch.Tensor,
        reliable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sampling == "full_resize": # 全尺寸重采样
            return self._full_resize(image, pseudo, reliable)
        if self.sampling == "foreground_crop": # 前景裁剪
            return self._foreground_crop(image, pseudo, reliable)
        return self._random_crop(image, pseudo, reliable)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = random.choice(self.image_paths)
        image, pseudo, reliable = self._load_case(image_path)
        image = self._select_image_view(image, index)
        image_patch, pseudo_patch, reliable_patch = self._sample(image, pseudo, reliable)
        return {
            "image": image_patch,
            "pseudo_prob": pseudo_patch,
            "reliable_mask": reliable_patch,
            "case": str(image_path),
        }
