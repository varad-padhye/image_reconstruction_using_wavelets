"""
metrics.py — Full evaluation metrics for WG-DM.
Computes: PSNR, SSIM, LPIPS, FID, NIQE, frequency error, per-image stats.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Per-batch PSNR in dB. Tensors in [0,1], shape (B,C,H,W)."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def compute_psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> list:
    """Returns a list of PSNR values, one per image in the batch."""
    results = []
    for i in range(pred.shape[0]):
        mse = F.mse_loss(pred[i], target[i]).item()
        results.append(float("inf") if mse == 0 else 20*math.log10(1.0) - 10*math.log10(mse))
    return results


# ─────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────

def _ssim_kernel(channel, window_size, device):
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    k2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return k2d.expand(channel, 1, window_size, window_size).to(device)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, C1=0.01**2, C2=0.03**2) -> float:
    C = pred.shape[1]
    k = _ssim_kernel(C, window_size, pred.device)
    pad = window_size // 2

    mu1  = F.conv2d(pred,         k, padding=pad, groups=C)
    mu2  = F.conv2d(target,       k, padding=pad, groups=C)
    mu1s, mu2s, mu12 = mu1*mu1, mu2*mu2, mu1*mu2

    s1   = F.conv2d(pred*pred,    k, padding=pad, groups=C) - mu1s
    s2   = F.conv2d(target*target,k, padding=pad, groups=C) - mu2s
    s12  = F.conv2d(pred*target,  k, padding=pad, groups=C) - mu12

    ssim_map = ((2*mu12+C1)*(2*s12+C2)) / ((mu1s+mu2s+C1)*(s1+s2+C2))
    return ssim_map.mean().item()


def compute_ssim_per_image(pred: torch.Tensor, target: torch.Tensor) -> list:
    return [compute_ssim(pred[i:i+1], target[i:i+1]) for i in range(pred.shape[0])]


# ─────────────────────────────────────────────
# LPIPS (perceptual similarity)
# ─────────────────────────────────────────────

class LPIPSMetric:
    """Lazy-loads LPIPS so it doesn't slow down import."""
    def __init__(self, device="cuda"):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import lpips
                self._model = lpips.LPIPS(net="alex").to(self.device)
            except ImportError:
                print("[LPIPS] Install with: pip install lpips")
                self._model = "unavailable"

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Input: [0,1] tensors. Returns mean LPIPS (lower = better)."""
        self._load()
        if self._model == "unavailable":
            return float("nan")
        # LPIPS expects [-1, 1]
        p = (pred * 2 - 1).to(self.device)
        t = (target * 2 - 1).to(self.device)
        with torch.no_grad():
            return self._model(p, t).mean().item()


# ─────────────────────────────────────────────
# FID
# ─────────────────────────────────────────────

def compute_fid(real: torch.Tensor, fake: torch.Tensor, device="cuda") -> float:
    """Both tensors: (N, 3, H, W) in [0, 1]. Needs H/W >= 299."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(normalize=True).to(device)
        fid.update((real * 255).byte().to(device), real=True)
        fid.update((fake * 255).byte().to(device), real=False)
        return fid.compute().item()
    except ImportError:
        print("[FID] Install with: pip install 'torchmetrics[image]'")
        return float("nan")


# ─────────────────────────────────────────────
# Frequency-domain error (wavelet subband analysis)
# ─────────────────────────────────────────────

def compute_frequency_error(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Measures how well each frequency band is reconstructed.
    Returns MAE per band: LL (low-freq) and HH (high-freq diagonal).
    Uses 2D FFT — no pywt dependency.
    """
    def fft_bands(x):
        # Average over channels and batch → (H, W)
        gray = x.mean(dim=(0, 1))
        f = torch.fft.fft2(gray)
        f_shift = torch.fft.fftshift(f)
        magnitude = f_shift.abs() + 1e-8
        H, W = magnitude.shape
        # Low-freq = centre 25%, High-freq = outer 25%
        cy, cx = H // 2, W // 2
        qy, qx = H // 4, W // 4
        low  = magnitude[cy-qy:cy+qy, cx-qx:cx+qx]
        # High freq = everything outside the centre half
        mask = torch.ones_like(magnitude, dtype=torch.bool)
        mask[cy-H//4:cy+H//4, cx-W//4:cx+W//4] = False
        high = magnitude[mask]
        return low, high

    pred_low,  pred_high  = fft_bands(pred)
    tgt_low,   tgt_high   = fft_bands(target)

    low_err  = (pred_low  - tgt_low ).abs().mean().item()
    high_err = (pred_high - tgt_high).abs().mean().item()
    ratio    = high_err / (low_err + 1e-8)

    return {
        "low_freq_mae":  low_err,
        "high_freq_mae": high_err,
        "hf_lf_ratio":   ratio,    # > 1 means high-freq errors dominate (typical)
    }


# ─────────────────────────────────────────────
# Histogram correlation (colour fidelity)
# ─────────────────────────────────────────────

def compute_histogram_correlation(pred: torch.Tensor, target: torch.Tensor,
                                  bins: int = 64) -> float:
    """
    Bhattacharyya coefficient between RGB histograms of pred and target.
    Returns value in [0, 1]: 1.0 = identical colour distribution.
    """
    pred_np   = pred.cpu().numpy().reshape(-1)
    target_np = target.cpu().numpy().reshape(-1)

    p_hist, _ = np.histogram(pred_np,   bins=bins, range=(0, 1), density=True)
    t_hist, _ = np.histogram(target_np, bins=bins, range=(0, 1), density=True)
    p_hist = p_hist / (p_hist.sum() + 1e-8)
    t_hist = t_hist / (t_hist.sum() + 1e-8)
    return float(np.sqrt(p_hist * t_hist).sum())   # Bhattacharyya coefficient


# ─────────────────────────────────────────────
# Sharpness (Laplacian variance — blind metric)
# ─────────────────────────────────────────────

def compute_sharpness(x: torch.Tensor) -> float:
    """
    Laplacian variance — measures image sharpness without a reference.
    Higher = sharper. Useful to compare LR vs WG-DM vs GT.
    """
    gray = x.mean(dim=1, keepdim=True)   # → (B, 1, H, W)
    lap_kernel = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        device=x.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, lap_kernel, padding=1)
    return lap.var().item()


# ─────────────────────────────────────────────
# Aggregate metric collector
# ─────────────────────────────────────────────

class MetricTracker:
    """Accumulates per-batch metrics and returns summary stats."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.records = []

    def update(self, pred_01: torch.Tensor, target_01: torch.Tensor, lr_01: torch.Tensor):
        """Call once per batch. All tensors in [0, 1]."""
        psnr_list = compute_psnr_per_image(pred_01, target_01)
        ssim_list = compute_ssim_per_image(pred_01, target_01)
        freq      = compute_frequency_error(pred_01, target_01)
        hist_corr = compute_histogram_correlation(pred_01, target_01)
        sharp_lr  = compute_sharpness(lr_01)
        sharp_out = compute_sharpness(pred_01)
        sharp_gt  = compute_sharpness(target_01)

        for i in range(pred_01.shape[0]):
            self.records.append({
                "psnr":          psnr_list[i],
                "ssim":          ssim_list[i],
                "low_freq_mae":  freq["low_freq_mae"],
                "high_freq_mae": freq["high_freq_mae"],
                "hf_lf_ratio":   freq["hf_lf_ratio"],
                "hist_corr":     hist_corr,
                "sharp_lr":      sharp_lr,
                "sharp_out":     sharp_out,
                "sharp_gt":      sharp_gt,
            })

    def summary(self) -> dict:
        import numpy as np
        if not self.records:
            return {}
        keys = self.records[0].keys()
        return {k: float(np.mean([r[k] for r in self.records])) for k in keys}

    def per_image_arrays(self) -> dict:
        import numpy as np
        keys = self.records[0].keys()
        return {k: np.array([r[k] for r in self.records]) for k in keys}
