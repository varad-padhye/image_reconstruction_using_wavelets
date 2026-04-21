"""
losses.py — Perceptual (VGG), SSIM, and combined loss for WG-DM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
# Perceptual loss (VGG16 feature matching)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self, layers=(6, 13), device="cuda"):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList()

        prev = 0
        for end in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _normalise(self, x):
        x01 = (x.clamp(-1, 1) + 1) / 2
        return (x01 - self.mean) / self.std

    def forward(self, pred, target):
        p = self._normalise(pred)
        t = self._normalise(target)

        loss = 0.0
        for block in self.blocks:
            p = block(p)
            t = block(t)
            loss += F.l1_loss(p, t)

        return loss / len(self.blocks)


# ─────────────────────────────────────────────
# SSIM loss (FIXED)
# ─────────────────────────────────────────────

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()

        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def forward(self, pred, target):
        p = (pred.clamp(-1, 1) + 1) / 2
        t = (target.clamp(-1, 1) + 1) / 2

        C = p.shape[1]
        pad = self.window_size // 2

        # 🔥 device-safe + correct shape
        k = self.kernel.to(p.device).repeat(C, 1, 1, 1)

        mu1 = F.conv2d(p, k, padding=pad, groups=C)
        mu2 = F.conv2d(t, k, padding=pad, groups=C)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12 = mu1 * mu2

        s1 = F.conv2d(p * p, k, padding=pad, groups=C) - mu1_sq
        s2 = F.conv2d(t * t, k, padding=pad, groups=C) - mu2_sq
        s12 = F.conv2d(p * t, k, padding=pad, groups=C) - mu12

        ssim_map = ((2 * mu12 + self.C1) * (2 * s12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2))

        return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────
# Diffusion loss
# ─────────────────────────────────────────────

class DiffusionLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_perceptual=0.1, w_ssim=0.2, device="cuda"):
        super().__init__()

        self.w_mse = w_mse
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim

        self.perceptual = PerceptualLoss().to(device)
        self.ssim = SSIMLoss().to(device)

    def forward(self, eps_pred, noise, x0_pred, x0_target):
        mse  = F.mse_loss(eps_pred, noise)
        perc = self.perceptual(x0_pred, x0_target)
        ssim = self.ssim(x0_pred, x0_target)

        total = self.w_mse * mse + self.w_perceptual * perc + self.w_ssim * ssim

        return total, {
            "mse": mse.item(),
            "perceptual": perc.item(),
            "ssim": ssim.item()
        }


# ─────────────────────────────────────────────
# HF predictor loss (FIXED)
# ─────────────────────────────────────────────

class HFCombinedLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5, device="cuda"):
        super().__init__()

        self.w_l1 = w_l1
        self.w_ssim = w_ssim

        # 🔥 FIX: move SSIM to device
        self.ssim = SSIMLoss().to(device)

    def forward(self, pred, target):
        loss = 0.0

        for key in ("LH", "HL", "HH"):
            l1   = F.l1_loss(pred[key], target[key])
            ssim = self.ssim(pred[key], target[key])

            loss += self.w_l1 * l1 + self.w_ssim * ssim

        return loss / 3.0