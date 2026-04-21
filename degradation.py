"""
degradation.py — Realistic image degradation for WG-DM training.

Instead of simple avg_pool, we apply a randomised combination of:
  - Gaussian blur (random kernel size + sigma)
  - Bicubic downscale + upscale (the standard SR degradation)
  - JPEG compression artifacts (via PIL round-trip)
  - Additive Gaussian noise

This makes the model robust to real-world degradation rather than
only the single synthetic blur pattern it was seeing before.
"""

import io
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def gaussian_blur(x: torch.Tensor, kernel_size: int = None, sigma: float = None) -> torch.Tensor:
    """Apply random Gaussian blur. x in [-1,1], shape (B, C, H, W)."""
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7, 9])
    if sigma is None:
        sigma = random.uniform(0.5, 3.0)

    # Build 1D Gaussian kernel
    k = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    g = torch.exp(-k**2 / (2 * sigma**2))
    g /= g.sum()
    g2d = g[:, None] * g[None, :]
    kernel = g2d.expand(x.shape[1], 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


def jpeg_compress(x: torch.Tensor, quality: int = None) -> torch.Tensor:
    """
    Simulate JPEG compression artifacts via PIL round-trip.
    x: (B, C, H, W) in [-1, 1]. Works per-image in the batch.
    """
    if quality is None:
        quality = random.randint(30, 85)

    x01 = ((x.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
    out = []
    for i in range(x.shape[0]):
        img = TF.to_pil_image(x01[i])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img_c = Image.open(buf).copy()
        t = TF.to_tensor(img_c)   # [0, 1]
        out.append(t * 2 - 1)     # [0, 1] → [-1, 1]
    return torch.stack(out, dim=0).to(x.device)


def downsample_upsample(x: torch.Tensor, scale: int = 4, mode: str = "bicubic") -> torch.Tensor:
    """Standard SR degradation: downsample then upsample back to original size."""
    H, W = x.shape[-2], x.shape[-1]
    lr = F.interpolate(x, size=(H // scale, W // scale), mode=mode, align_corners=False)
    return F.interpolate(lr, size=(H, W), mode=mode, align_corners=False)


def add_noise(x: torch.Tensor, std: float = None) -> torch.Tensor:
    """Add Gaussian noise with random std."""
    if std is None:
        std = random.uniform(0.01, 0.05)
    return (x + torch.randn_like(x) * std).clamp(-1, 1)


def degrade(x: torch.Tensor, mode: str = "random", scale: int = 4) -> torch.Tensor:
    """
    Apply a realistic degradation pipeline to a batch of HR images.

    mode="random"  — randomly pick one of the degradation combos below.
    mode="sr"      — standard super-resolution (downsample + upsample only).
    mode="blur"    — Gaussian blur + downsample.
    mode="jpeg"    — JPEG artifacts + downsample.
    mode="full"    — all degradations combined (hardest for the model).

    Returns degraded image at the SAME resolution as input (lr_up).
    """
    if mode == "random":
        mode = random.choice(["sr", "blur", "jpeg", "full"])

    if mode == "sr":
        return downsample_upsample(x, scale=scale)

    elif mode == "blur":
        blurred = gaussian_blur(x)
        return downsample_upsample(blurred, scale=scale)

    elif mode == "jpeg":
        lr_up = downsample_upsample(x, scale=scale)
        return jpeg_compress(lr_up)

    elif mode == "full":
        out = gaussian_blur(x)
        out = downsample_upsample(out, scale=scale)
        out = jpeg_compress(out, quality=random.randint(40, 75))
        out = add_noise(out, std=random.uniform(0.01, 0.03))
        return out

    else:
        raise ValueError(f"Unknown degradation mode: {mode}")
