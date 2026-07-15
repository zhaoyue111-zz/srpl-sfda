from __future__ import annotations

import torch


def _minmax_gamma(image: torch.Tensor, gamma: float) -> torch.Tensor:
    dims = tuple(range(2, image.ndim))
    min_val = image.amin(dim=dims, keepdim=True)
    max_val = image.amax(dim=dims, keepdim=True)
    norm = (image - min_val) / (max_val - min_val).clamp_min(1e-6)
    enhanced = norm.clamp(0.0, 1.0).pow(gamma)
    return enhanced * (max_val - min_val) + min_val


def t3ie_views(image: torch.Tensor) -> list[torch.Tensor]:
    """Test-time tri-branch intensity enhancement views.

    The original SRPL-SFDA implementation uses intensity perturbations before
    source-model pseudo-label generation. VoxTell is 3D, so this keeps the same
    idea directly on volumetric patches.
    """

    return [
        image,
        _minmax_gamma(image, gamma=0.7),
        _minmax_gamma(image, gamma=1.5),
    ]
