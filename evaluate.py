"""
evaluate.py — Full evaluation script for WG-DM with chart generation.

Produces:
  eval_images/         — comparison grids + per-image outputs
  eval_charts/
    01_psnr_ssim_distribution.png   — violin + box plots per metric
    02_psnr_per_image.png           — per-image PSNR bar chart
    03_ssim_per_image.png           — per-image SSIM bar chart
    04_frequency_analysis.png       — low vs high freq error + ratio
    05_sharpness_comparison.png     — LR vs WG-DM vs GT sharpness
    06_histogram_correlation.png    — colour fidelity per batch
    07_metric_radar.png             — radar chart summary
    08_loss_curve.png               — training loss curve (if log exists)
    eval_report.txt                 — full text summary

Run:
    python evaluate.py --config configs/train.yaml
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from wavelet import dwt2d, idwt2d
from diffusion import make_beta_schedule, UNetDenoiser
from hf_predictor import HFPredictor
from metrics import MetricTracker, compute_fid, LPIPSMetric
from degradation import degrade


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_val_loader(cfg):
    tfm = transforms.Compose([
        transforms.CenterCrop(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = ImageFolder(os.path.join(cfg["data_root"], "val"), transform=tfm)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                      num_workers=4, pin_memory=True, drop_last=False)


def build_diff_model(cfg, device):
    return UNetDenoiser(
        in_ch=cfg["in_ch"]*2, out_ch=cfg["in_ch"],
        base_ch=cfg["unet_base_ch"], time_dim=cfg["time_dim"]
    ).to(device)


def build_hf_model(cfg, device):
    return HFPredictor(cfg["in_ch"], cfg["hf_base_ch"], cfg["hf_res_blocks"]).to(device)


@torch.no_grad()
def ddim_sample(model, shape, schedule, cond, ddim_steps=50, device="cuda"):
    T = len(schedule["betas"])
    step_ids = torch.linspace(0, T-1, ddim_steps, dtype=torch.long)
    x  = torch.randn(shape, device=device)
    ab = schedule["alpha_bar"].to(device)
    for i in reversed(range(ddim_steps)):
        t_now  = step_ids[i].item()
        t_prev = step_ids[i-1].item() if i > 0 else -1
        t_b    = torch.full((shape[0],), t_now, device=device, dtype=torch.long)
        eps    = model(torch.cat([x, cond], dim=1), t_b)
        ab_now  = ab[t_now]
        ab_prev = ab[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0 = ((x - (1-ab_now).sqrt()*eps) / ab_now.sqrt()).clamp(-1, 1)
        x  = ab_prev.sqrt()*x0 + (1-ab_prev).sqrt()*eps
    return x


# ─────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────

CHART_COLOR = {
    "lr":   "#378ADD",
    "wgdm": "#1D9E75",
    "gt":   "#534AB7",
    "bar":  "#1D9E75",
    "ref":  "#D85A30",
}

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# Individual chart functions
# ─────────────────────────────────────────────

def chart_distribution(arrays, out_dir):
    """Violin + box plot for PSNR and SSIM distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Metric distributions across validation set", fontsize=13, fontweight="bold")

    for ax, key, title, unit in [
        (axes[0], "psnr", "PSNR distribution", "dB"),
        (axes[1], "ssim", "SSIM distribution",  ""),
    ]:
        data = arrays[key]
        parts = ax.violinplot(data, positions=[1], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(CHART_COLOR["bar"])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(CHART_COLOR["ref"])
        parts["cmedians"].set_linewidth(2)

        ax.boxplot(data, positions=[1], widths=0.15,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", color=CHART_COLOR["ref"]),
                   medianprops=dict(color=CHART_COLOR["ref"], linewidth=2),
                   whiskerprops=dict(color="#888"),
                   capprops=dict(color="#888"))

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(f"{key.upper()} ({unit})" if unit else key.upper(), fontsize=10)
        ax.set_xticks([])
        mean_val = data.mean()
        ax.axhline(mean_val, color=CHART_COLOR["ref"], linestyle="--", linewidth=1, alpha=0.7)
        ax.text(1.25, mean_val, f"μ={mean_val:.3f}", va="center", fontsize=9,
                color=CHART_COLOR["ref"])
        ax.grid(axis="y", alpha=0.3)

    savefig(fig, os.path.join(out_dir, "01_psnr_ssim_distribution.png"))


def chart_per_image_bar(arrays, out_dir):
    """Bar chart of PSNR and SSIM per image, sorted descending."""
    for key, title, unit, fname in [
        ("psnr", "PSNR per image (sorted)", "dB", "02_psnr_per_image.png"),
        ("ssim", "SSIM per image (sorted)", "",   "03_ssim_per_image.png"),
    ]:
        data   = np.sort(arrays[key])[::-1]
        n      = len(data)
        mean_v = data.mean()

        fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 4))
        colors = [CHART_COLOR["bar"] if v >= mean_v else CHART_COLOR["ref"] for v in data]
        ax.bar(range(n), data, color=colors, alpha=0.85, width=0.8)
        ax.axhline(mean_v, color="#333", linestyle="--", linewidth=1.2, label=f"Mean: {mean_v:.3f}")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Image index", fontsize=10)
        ax.set_ylabel(f"{key.upper()}" + (f" ({unit})" if unit else ""), fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        patch_above = mpatches.Patch(color=CHART_COLOR["bar"], label="≥ mean")
        patch_below = mpatches.Patch(color=CHART_COLOR["ref"], label="< mean")
        ax.legend(handles=[patch_above, patch_below], fontsize=9, loc="upper right")

        savefig(fig, os.path.join(out_dir, fname))


def chart_frequency_analysis(arrays, out_dir):
    """Low-freq MAE, high-freq MAE, and their ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Frequency-domain error analysis", fontsize=13, fontweight="bold")

    metrics = [
        ("low_freq_mae",  "Low-freq MAE\n(structural accuracy)", CHART_COLOR["wgdm"]),
        ("high_freq_mae", "High-freq MAE\n(edge/texture accuracy)", CHART_COLOR["ref"]),
        ("hf_lf_ratio",   "HF / LF error ratio\n(> 1 means edges harder)", "#534AB7"),
    ]

    for ax, (key, title, color) in zip(axes, metrics):
        data = arrays[key]
        ax.hist(data, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():.4f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    savefig(fig, os.path.join(out_dir, "04_frequency_analysis.png"))


def chart_sharpness(arrays, out_dir):
    """Side-by-side sharpness comparison: LR vs WG-DM vs GT."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Sharpness comparison (Laplacian variance)", fontsize=13, fontweight="bold")

    n = len(arrays["sharp_lr"])
    x = np.arange(n)
    w = 0.28

    ax.bar(x - w, arrays["sharp_lr"],  width=w, label="LR input",    color=CHART_COLOR["lr"],   alpha=0.8)
    ax.bar(x,     arrays["sharp_out"], width=w, label="WG-DM output", color=CHART_COLOR["wgdm"], alpha=0.8)
    ax.bar(x + w, arrays["sharp_gt"],  width=w, label="GT HR",        color=CHART_COLOR["gt"],   alpha=0.8)

    ax.set_xlabel("Batch index", fontsize=10)
    ax.set_ylabel("Laplacian variance (higher = sharper)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add mean lines
    for val, color, label in [
        (arrays["sharp_lr"].mean(),  CHART_COLOR["lr"],   "LR mean"),
        (arrays["sharp_out"].mean(), CHART_COLOR["wgdm"], "WG-DM mean"),
        (arrays["sharp_gt"].mean(),  CHART_COLOR["gt"],   "GT mean"),
    ]:
        ax.axhline(val, color=color, linestyle=":", linewidth=1.5, alpha=0.7)

    savefig(fig, os.path.join(out_dir, "05_sharpness_comparison.png"))


def chart_histogram_correlation(arrays, out_dir):
    """Colour fidelity (Bhattacharyya coefficient) per batch."""
    data = arrays["hist_corr"]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Colour fidelity — histogram correlation (1.0 = perfect)",
                 fontsize=13, fontweight="bold")

    ax.plot(data, marker="o", color=CHART_COLOR["wgdm"], linewidth=1.5,
            markersize=5, label="WG-DM vs GT")
    ax.axhline(data.mean(), color=CHART_COLOR["ref"], linestyle="--", linewidth=1.5,
               label=f"Mean: {data.mean():.4f}")
    ax.axhline(1.0, color="#999", linestyle=":", linewidth=1, label="Perfect = 1.0")

    ax.set_ylim(max(0, data.min() - 0.05), 1.05)
    ax.set_xlabel("Batch index", fontsize=10)
    ax.set_ylabel("Bhattacharyya coefficient", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    savefig(fig, os.path.join(out_dir, "06_histogram_correlation.png"))


def chart_radar(summary, out_dir):
    """Radar chart of normalised metrics."""
    # Normalise everything to [0, 1] for display
    metrics = {
        "PSNR\n(norm)":   min(summary["psnr"] / 40.0, 1.0),
        "SSIM":            summary["ssim"],
        "Hist\ncorr":      summary["hist_corr"],
        "Sharpness\n(norm)": min(summary["sharp_out"] / (summary["sharp_gt"] + 1e-8), 1.0),
        "LF\naccuracy":    max(0, 1 - summary["low_freq_mae"] / 10.0),
        "HF\naccuracy":    max(0, 1 - summary["high_freq_mae"] / 10.0),
    }

    labels = list(metrics.keys())
    vals   = list(metrics.values())
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals   += vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)

    ax.plot(angles, vals, color=CHART_COLOR["wgdm"], linewidth=2)
    ax.fill(angles, vals, color=CHART_COLOR["wgdm"], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(alpha=0.4)
    ax.set_title("WG-DM metric radar\n(all metrics normalised to [0, 1])",
                 fontsize=12, fontweight="bold", pad=20)

    savefig(fig, os.path.join(out_dir, "07_metric_radar.png"))


def chart_loss_curve(cfg, out_dir):
    """Plots training loss curve if a loss log JSON exists."""
    log_path = os.path.join(cfg["ckpt_dir"], "loss_log.json")
    if not os.path.exists(log_path):
        print(f"  [loss curve] No loss_log.json found at {log_path} — skipping.")
        return

    with open(log_path) as f:
        log = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training loss curves", fontsize=13, fontweight="bold")

    if "diffusion" in log:
        d = log["diffusion"]
        epochs = range(1, len(d["total"])+1)
        axes[0].plot(epochs, d["total"],       label="Total",       color="#333",          linewidth=2)
        axes[0].plot(epochs, d["mse"],         label="MSE",         color=CHART_COLOR["lr"],   linewidth=1.2, linestyle="--")
        axes[0].plot(epochs, d["perceptual"],  label="Perceptual",  color=CHART_COLOR["ref"],  linewidth=1.2, linestyle="--")
        axes[0].plot(epochs, d["ssim"],        label="SSIM loss",   color=CHART_COLOR["wgdm"], linewidth=1.2, linestyle="--")
        axes[0].set_title("Diffusion model loss", fontsize=11)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    if "hf" in log:
        h = log["hf"]
        axes[1].plot(range(1, len(h)+1), h, color=CHART_COLOR["wgdm"], linewidth=2)
        axes[1].set_title("HF predictor loss", fontsize=11)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
        axes[1].grid(alpha=0.3)

    savefig(fig, os.path.join(out_dir, "08_loss_curve.png"))


def write_report(summary, lpips_score, fid_score, out_dir, cfg):
    """Write a plain-text evaluation report."""
    lines = [
        "=" * 50,
        "WG-DM EVALUATION REPORT",
        "=" * 50,
        "",
        f"Model config       : {cfg.get('unet_base_ch')}ch UNet, {cfg.get('hf_res_blocks')} HF res-blocks",
        f"Scale factor       : {cfg.get('scale', 4)}x SR",
        f"DDIM steps         : {cfg.get('ddim_steps', 50)}",
        "",
        "─── Reference metrics ───────────────────────",
        f"  PSNR             : {summary['psnr']:.2f} dB",
        f"  SSIM             : {summary['ssim']:.4f}",
        f"  LPIPS            : {lpips_score:.4f}  (lower = better)",
        f"  FID              : {fid_score:.2f}    (lower = better)",
        "",
        "─── Frequency analysis ──────────────────────",
        f"  Low-freq MAE     : {summary['low_freq_mae']:.4f}",
        f"  High-freq MAE    : {summary['high_freq_mae']:.4f}",
        f"  HF/LF ratio      : {summary['hf_lf_ratio']:.3f}  (>1 means edges are harder)",
        "",
        "─── Colour & sharpness ──────────────────────",
        f"  Histogram corr   : {summary['hist_corr']:.4f}  (1.0 = perfect colour match)",
        f"  Sharpness — LR   : {summary['sharp_lr']:.4f}",
        f"  Sharpness — WG-DM: {summary['sharp_out']:.4f}",
        f"  Sharpness — GT   : {summary['sharp_gt']:.4f}",
        f"  Sharpness gain   : {summary['sharp_out']/max(summary['sharp_lr'],1e-8):.2f}x vs LR",
        "",
        "─── Interpretation ──────────────────────────",
        f"  PSNR > 30 dB     : excellent | 25–30: good | 20–25: fair | <20: poor",
        f"  SSIM > 0.90      : excellent | 0.80–0.90: good | 0.70–0.80: fair",
        f"  Your PSNR        : {summary['psnr']:.2f} dB  →  " +
            ("excellent" if summary['psnr']>30 else "good" if summary['psnr']>25 else "fair" if summary['psnr']>20 else "poor"),
        f"  Your SSIM        : {summary['ssim']:.4f}  →  " +
            ("excellent" if summary['ssim']>0.90 else "good" if summary['ssim']>0.80 else "fair" if summary['ssim']>0.70 else "poor"),
        "",
        "=" * 50,
    ]
    path = os.path.join(out_dir, "eval_report.txt")

    # ✅ FIX HERE
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n  Report saved → {path}")
# ─────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(cfg, device):
    schedule   = make_beta_schedule(T=cfg["T"])
    loader     = get_val_loader(cfg)
    ddim_steps = cfg.get("ddim_steps", 50)
    n_save     = cfg.get("n_save_images", 8)

    # Load models
    diff_model = build_diff_model(cfg, device)
    diff_model.load_state_dict(torch.load(cfg["diff_ckpt"], map_location=device))
    diff_model.eval()

    hf_ckpt = cfg.get("hf_ckpt", os.path.join(cfg["ckpt_dir"], "hf_finetuned.pt"))
    if not os.path.exists(hf_ckpt):
        hf_ckpt = os.path.join(cfg["ckpt_dir"], "hf_predictor.pt")
    hf_model = build_hf_model(cfg, device)
    hf_model.load_state_dict(torch.load(hf_ckpt, map_location=device))
    hf_model.eval()

    # Output dirs
    img_dir   = os.path.join(cfg["ckpt_dir"], "eval_images")
    chart_dir = os.path.join(cfg["ckpt_dir"], "eval_charts")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    tracker    = MetricTracker()
    lpips_fn   = LPIPSMetric(device=str(device))
    all_real   = []
    all_fake   = []
    lpips_vals = []

    for batch_idx, (batch, _) in enumerate(tqdm(loader, desc="Evaluating")):
        hr    = batch.to(device)
        lr_up = degrade(hr, mode="sr", scale=cfg.get("scale", 4))

        ll_cond = dwt2d(lr_up)["LL"]
        B, C, h, w = ll_cond.shape

        ll_restored = ddim_sample(diff_model, (B,C,h,w), schedule,
                                   cond=ll_cond, ddim_steps=ddim_steps, device=str(device))
        hf_pred  = hf_model(ll_restored)
        x_hat    = idwt2d({"LL": ll_restored, **hf_pred})[..., :hr.shape[-2], :hr.shape[-1]]

        x_hat_01 = (x_hat.clamp(-1,1) + 1) / 2
        hr_01    = (hr + 1) / 2
        lr_01    = (lr_up.clamp(-1,1) + 1) / 2

        tracker.update(x_hat_01, hr_01, lr_01)
        lpips_vals.append(lpips_fn.compute(x_hat_01, hr_01))

        # Collect for FID (upsample to 299 if needed)
        if hr.shape[-1] < 299:
            real_fid = F.interpolate(hr_01, size=(299,299), mode="bilinear", align_corners=False)
            fake_fid = F.interpolate(x_hat_01, size=(299,299), mode="bilinear", align_corners=False)
        else:
            real_fid, fake_fid = hr_01, x_hat_01
        all_real.append(real_fid.cpu())
        all_fake.append(fake_fid.cpu())

        # Save comparison grids for first few batches
        if batch_idx < cfg.get("n_save_batches", 3):
            n    = min(n_save, B)
            grid = make_grid(torch.cat([lr_01[:n], x_hat_01[:n], hr_01[:n]]),
                             nrow=n, padding=4, pad_value=1.0)
            save_image(grid, os.path.join(img_dir, f"grid_batch{batch_idx:03d}.png"))
            for i in range(n):
                save_image(lr_01[i],    os.path.join(img_dir, f"b{batch_idx:03d}_{i:02d}_LR.png"))
                save_image(x_hat_01[i], os.path.join(img_dir, f"b{batch_idx:03d}_{i:02d}_WGDM.png"))
                save_image(hr_01[i],    os.path.join(img_dir, f"b{batch_idx:03d}_{i:02d}_GT.png"))

    # FID
    print("\nComputing FID (this may take a moment)...")
    real_all = torch.cat(all_real, dim=0)
    fake_all = torch.cat(all_fake, dim=0)
    fid_score   = compute_fid(real_all, fake_all, device=str(device))
    lpips_score = float(np.nanmean(lpips_vals))

    summary = tracker.summary()
    arrays  = tracker.per_image_arrays()

    # ── Generate all charts ──
    print("\nGenerating charts...")
    chart_distribution(arrays, chart_dir)
    chart_per_image_bar(arrays, chart_dir)
    chart_frequency_analysis(arrays, chart_dir)
    chart_sharpness(arrays, chart_dir)
    chart_histogram_correlation(arrays, chart_dir)
    chart_radar(summary, chart_dir)
    chart_loss_curve(cfg, chart_dir)

    write_report(summary, lpips_score, fid_score, chart_dir, cfg)

    print(f"\nAll charts saved → {chart_dir}")
    print(f"All images saved → {img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_evaluation(cfg, device)
