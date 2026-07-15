from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def binary_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.float().clamp(eps, 1.0 - eps)
    return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob)) / math.log(2.0)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()
    pseudo = pseudo.float()
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none")
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def weighted_dice_loss(
    prob: torch.Tensor,
    pseudo: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    prob = prob.float()
    pseudo = pseudo.float()
    weight = weight.float()
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * pseudo * weight).sum(dim=dims)
    pred_sum = (prob * prob * weight).sum(dim=dims)
    pseudo_sum = (pseudo * pseudo * weight).sum(dim=dims)
    dice = (2.0 * intersection + eps) / (pred_sum + pseudo_sum + eps)
    return 1.0 - dice.mean()


def weighted_mse_loss(
    prob: torch.Tensor,
    pseudo: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    prob = prob.float()
    pseudo = pseudo.float()
    weight = weight.float()
    denom = weight.sum().clamp_min(1.0)
    return (((prob - pseudo) ** 2) * weight).sum() / denom


def entropy_minimization_loss(prob: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = prob.float()
    weight = weight.float()
    entropy = binary_entropy(prob)
    denom = weight.sum().clamp_min(1.0)
    return (entropy * weight).sum() / denom


def srpl_sfda_loss(
    logits: torch.Tensor,
    pseudo_prob: torch.Tensor,
    reliable_mask: torch.Tensor,
    entropy_weight: float,
    pseudo_target: str = "hard",
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = logits.float()
    pseudo_prob = pseudo_prob.float()
    reliable_mask = reliable_mask.float()
    if pseudo_target == "hard":
        pseudo = (pseudo_prob > 0.5).float()
    elif pseudo_target == "soft":
        pseudo = pseudo_prob.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown pseudo target mode: {pseudo_target}")
    prob = torch.sigmoid(logits)
    unreliable_mask = (1.0 - reliable_mask).clamp(0.0, 1.0)

    bce = weighted_bce_with_logits(logits, pseudo, reliable_mask)
    dice = weighted_dice_loss(prob, pseudo, reliable_mask)
    mse = weighted_mse_loss(prob, pseudo, reliable_mask)
    reliable_sup = 0.5 * (bce + dice) if pseudo_target == "hard" else (0.25 * bce + 0.25 * dice + 0.5 * mse)
    entropy = entropy_minimization_loss(prob, unreliable_mask)
    total = reliable_sup + entropy_weight * entropy

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_bce": float(bce.detach().cpu()),
        "loss_dice": float(dice.detach().cpu()),
        "loss_mse": float(mse.detach().cpu()),
        "loss_entropy": float(entropy.detach().cpu()),
        "reliable_ratio": float(reliable_mask.mean().detach().cpu()),
    }
    return total, stats
