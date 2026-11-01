"""Image quality metrics for paired dehazing evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def ssim_tensor(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Batch SSIM for RGB tensors in NCHW format and [0, 1] range."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects matching NCHW tensors.")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError(f"SSIM needs images at least {window_size} pixels wide and high.")

    x, y = prediction.float(), target.float()
    channels = x.shape[1]
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - (window_size - 1) / 2
    gaussian = torch.exp(-(coords**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian).expand(channels, 1, window_size, window_size)

    mu_x = F.conv2d(x, window_2d, groups=channels)
    mu_y = F.conv2d(y, window_2d, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    covariance_norm = (window_size**2) / (window_size**2 - 1)
    sigma_x2 = ((F.conv2d(x * x, window_2d, groups=channels) - mu_x2) * covariance_norm).clamp_min(0)
    sigma_y2 = ((F.conv2d(y * y, window_2d, groups=channels) - mu_y2) * covariance_norm).clamp_min(0)
    sigma_xy = (F.conv2d(x * y, window_2d, groups=channels) - mu_xy) * covariance_norm

    c1, c2 = 0.01**2, 0.03**2
    score_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score_map.mean(dim=(1, 2, 3))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = torch.from_numpy(np.asarray(prediction, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    ref = torch.from_numpy(np.asarray(target, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    return float(ssim_tensor(pred, ref).item())


def image_quality(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    return psnr(prediction, target), ssim(prediction, target)
