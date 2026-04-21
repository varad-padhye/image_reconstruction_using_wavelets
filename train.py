"""
train.py — Improved WG-DM training with:
  - Perceptual + SSIM loss (instead of plain MSE)
  - Richer degradation (blur, JPEG, noise)
  - HF fine-tune stage on real DDIM outputs
  - Image saving during eval

Stages:
  diffusion  — train UNet denoiser on LL band
  hf         — fast HF predictor training on GT LL + sim noise
  hf_finetune— fine-tune HF predictor on real DDIM LL outputs (slower but better)
  eval       — evaluate and save comparison images
  all        — run diffusion → hf → eval
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from wavelet import dwt2d, idwt2d
from diffusion import make_beta_schedule, q_sample, UNetDenoiser
from hf_predictor import HFPredictor
from losses import DiffusionLoss, HFCombinedLoss
from degradation import degrade
from metrics import compute_psnr, compute_ssim


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataloader(cfg: dict, split: str) -> DataLoader:
    tfm = transforms.Compose([
        transforms.RandomCrop(cfg["image_size"]),      # random crop instead of resize — more variety
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ]) if split == "train" else transforms.Compose([
        transforms.CenterCrop(cfg["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = ImageFolder(os.path.join(cfg["data_root"], split), transform=tfm)
    return DataLoader(dataset, batch_size=cfg["batch_size"],
                      shuffle=(split == "train"), num_workers=4, pin_memory=True,
                      drop_last=True)


def build_diff_model(cfg, device):
    m = UNetDenoiser(
        in_ch=cfg["in_ch"] * 2,
        out_ch=cfg["in_ch"],
        base_ch=cfg["unet_base_ch"],
        time_dim=cfg["time_dim"],
    ).to(device)
    return m


def build_hf_model(cfg, device):
    return HFPredictor(cfg["in_ch"], cfg["hf_base_ch"], cfg["hf_res_blocks"]).to(device)


# ─────────────────────────────────────────────
# DDIM fast sampler
# ─────────────────────────────────────────────

@torch.no_grad()
def ddim_sample(model, shape, schedule, cond, ddim_steps=50, device="cuda"):
    T = len(schedule["betas"])
    step_ids = torch.linspace(0, T - 1, ddim_steps, dtype=torch.long)
    x  = torch.randn(shape, device=device)
    ab = schedule["alpha_bar"].to(device)
    for i in reversed(range(ddim_steps)):
        t_now  = step_ids[i].item()
        t_prev = step_ids[i - 1].item() if i > 0 else -1
        t_b    = torch.full((shape[0],), t_now, device=device, dtype=torch.long)
        eps    = model(torch.cat([x, cond], dim=1), t_b)
        ab_now  = ab[t_now]
        ab_prev = ab[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0 = ((x - (1 - ab_now).sqrt() * eps) / ab_now.sqrt()).clamp(-1, 1)
        x  = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x


# ─────────────────────────────────────────────
# Stage 1: Train diffusion model
# ─────────────────────────────────────────────

def train_diffusion(cfg, device):
    schedule = make_beta_schedule(T=cfg["T"])
    model    = build_diff_model(cfg, device)
    loss_fn  = DiffusionLoss(
        w_mse=cfg.get("w_mse", 1.0),
        w_perceptual=cfg.get("w_perceptual", 0.1),
        w_ssim=cfg.get("w_ssim", 0.2),
        device=str(device),
    )

    opt   = AdamW(model.parameters(), lr=cfg["lr"])
    sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loader = get_dataloader(cfg, "train")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # Loss log — saved each epoch so evaluate.py can plot the curve
    log_path = os.path.join(cfg["ckpt_dir"], "loss_log.json")
    loss_log = {"diffusion": {"total": [], "mse": [], "perceptual": [], "ssim": []}, "hf": []}
    if os.path.exists(log_path):
        import json
        with open(log_path) as f:
            loss_log = json.load(f)
        loss_log.setdefault("diffusion", {"total": [], "mse": [], "perceptual": [], "ssim": []})
        loss_log.setdefault("hf", [])

    model.train()
    for epoch in range(cfg["epochs"]):
        totals = {"mse": 0., "perceptual": 0., "ssim": 0., "total": 0.}
        for batch, _ in tqdm(loader, desc=f"Diffusion {epoch+1}/{cfg['epochs']}"):
            hr = batch.to(device)

            # Richer degradation — random combo of blur/JPEG/noise
            lr_up = degrade(hr, mode=cfg.get("degrade_mode", "random"),
                            scale=cfg.get("scale", 4))

            bands_hr = dwt2d(hr)
            ll_hr    = bands_hr["LL"]
            ll_cond  = dwt2d(lr_up)["LL"]

            B = hr.shape[0]
            t = torch.randint(0, cfg["T"], (B,), device=device)
            x_t, noise = q_sample(ll_hr, t, schedule)

            eps_pred = model(torch.cat([x_t, ll_cond], dim=1), t)

            # Reconstruct x0 from predicted noise (needed for perceptual loss)
            ab = schedule["sqrt_alpha_bar"][t].to(device)[:, None, None, None]
            sab = schedule["sqrt_one_minus_alpha_bar"][t].to(device)[:, None, None, None]
            x0_pred = (x_t - sab * eps_pred) / ab

            loss, breakdown = loss_fn(eps_pred, noise, x0_pred, ll_hr)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            totals["total"] += loss.item()
            for k, v in breakdown.items():
                totals[k] += v

        sched.step()
        n = len(loader)
        avg = {k: v/n for k, v in totals.items()}
        print(f"[Diffusion] ep{epoch+1}  total={avg['total']:.4f}  "
              f"mse={avg['mse']:.4f}  perc={avg['perceptual']:.4f}  ssim={avg['ssim']:.4f}")

        # Append to loss log
        for k in ("total", "mse", "perceptual", "ssim"):
            loss_log["diffusion"][k].append(avg[k])
        import json
        with open(log_path, "w") as f:
            json.dump(loss_log, f)

        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg["ckpt_dir"], f"diffusion_ep{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"], "diffusion_final.pt"))
    return model


# ─────────────────────────────────────────────
# Stage 2a: Fast HF training on GT LL + sim noise
# ─────────────────────────────────────────────

def train_hf_predictor(cfg, device, diff_model):
    hf_model = build_hf_model(cfg, device)
    loss_fn  = HFCombinedLoss(w_l1=1.0, w_ssim=0.5)

    opt   = AdamW(hf_model.parameters(), lr=cfg.get("hf_lr", 2e-4))
    sched = CosineAnnealingLR(opt, T_max=cfg["hf_epochs"])
    loader = get_dataloader(cfg, "train")
    noise_std = cfg.get("hf_sim_noise_std", 0.05)

    hf_model.train()
    for epoch in range(cfg["hf_epochs"]):
        total_loss = 0.0
        for batch, _ in tqdm(loader, desc=f"HF {epoch+1}/{cfg['hf_epochs']}"):
            hr = batch.to(device)
            bands_hr = dwt2d(hr)
            ll_gt    = bands_hr["LL"]
            ll_input = ll_gt + noise_std * torch.randn_like(ll_gt)

            hf_pred   = hf_model(ll_input)
            hf_target = {k: bands_hr[k] for k in ("LH", "HL", "HH")}
            loss = loss_fn(hf_pred, hf_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / len(loader)
        print(f"[HF] ep{epoch+1}/{cfg['hf_epochs']}  loss={avg_loss:.4f}")

        # Append to shared loss log
        import json
        log_path = os.path.join(cfg["ckpt_dir"], "loss_log.json")
        loss_log = {"diffusion": {"total":[],"mse":[],"perceptual":[],"ssim":[]}, "hf": []}
        if os.path.exists(log_path):
            with open(log_path) as f:
                loss_log = json.load(f)
        loss_log["hf"].append(avg_loss)
        with open(log_path, "w") as f:
            json.dump(loss_log, f)

        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(hf_model.state_dict(),
                       os.path.join(cfg["ckpt_dir"], f"hf_ep{epoch+1}.pt"))

    torch.save(hf_model.state_dict(), os.path.join(cfg["ckpt_dir"], "hf_predictor.pt"))
    return hf_model


# ─────────────────────────────────────────────
# Stage 2b: Fine-tune HF on real DDIM outputs
# (run after Stage 2a — closes the train/eval gap)
# ─────────────────────────────────────────────

def finetune_hf_predictor(cfg, device, diff_model, hf_model):
    """
    After fast HF training, the model was never exposed to real diffusion outputs.
    This stage runs DDIM once per batch and fine-tunes HF on those outputs.
    Slower (~3 min/epoch) but closes the distribution gap significantly.
    """
    loss_fn  = HFCombinedLoss(w_l1=1.0, w_ssim=0.5)
    opt      = AdamW(hf_model.parameters(), lr=cfg.get("hf_ft_lr", 5e-5))  # lower LR for fine-tune
    sched    = CosineAnnealingLR(opt, T_max=cfg.get("hf_ft_epochs", 10))
    loader   = get_dataloader(cfg, "train")
    schedule = make_beta_schedule(T=cfg["T"])
    ddim_steps = cfg.get("hf_ft_ddim_steps", 20)   # fewer steps → faster fine-tune

    diff_model.eval()
    hf_model.train()

    for epoch in range(cfg.get("hf_ft_epochs", 10)):
        total_loss = 0.0
        for batch, _ in tqdm(loader, desc=f"HF fine-tune {epoch+1}/{cfg.get('hf_ft_epochs',10)}"):
            hr = batch.to(device)
            bands_hr = dwt2d(hr)
            lr_up    = degrade(hr, mode="sr", scale=cfg.get("scale", 4))
            ll_cond  = dwt2d(lr_up)["LL"]
            B, C, h, w = ll_cond.shape

            with torch.no_grad():
                ll_restored = ddim_sample(
                    diff_model, (B, C, h, w), schedule,
                    cond=ll_cond, ddim_steps=ddim_steps, device=str(device)
                )

            hf_pred   = hf_model(ll_restored)
            hf_target = {k: bands_hr[k] for k in ("LH", "HL", "HH")}
            loss = loss_fn(hf_pred, hf_target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        print(f"[HF FT] ep{epoch+1}  loss={total_loss/len(loader):.4f}")

    torch.save(hf_model.state_dict(), os.path.join(cfg["ckpt_dir"], "hf_finetuned.pt"))
    return hf_model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(cfg, device, diff_model, hf_model):
    schedule = make_beta_schedule(T=cfg["T"])
    loader   = get_dataloader(cfg, "val")
    diff_model.eval()
    hf_model.eval()

    out_dir = os.path.join(cfg["ckpt_dir"], "eval_images")
    os.makedirs(out_dir, exist_ok=True)

    psnr_vals, ssim_vals = [], []
    ddim_steps = cfg.get("ddim_steps", 50)
    n_save     = cfg.get("n_save_images", 8)

    for batch_idx, (batch, _) in enumerate(tqdm(loader, desc="Evaluating")):
        hr = batch.to(device)
        lr_up = degrade(hr, mode="sr", scale=cfg.get("scale", 4))

        ll_cond = dwt2d(lr_up)["LL"]
        B, C, h, w = ll_cond.shape

        ll_restored = ddim_sample(
            diff_model, (B, C, h, w), schedule,
            cond=ll_cond, ddim_steps=ddim_steps, device=str(device)
        )
        hf_pred    = hf_model(ll_restored)
        x_hat      = idwt2d({"LL": ll_restored, **hf_pred})[..., :hr.shape[-2], :hr.shape[-1]]

        x_hat_01 = (x_hat.clamp(-1, 1) + 1) / 2
        hr_01    = (hr + 1) / 2
        lr_01    = (lr_up.clamp(-1, 1) + 1) / 2

        psnr_vals.append(compute_psnr(x_hat_01, hr_01))
        ssim_vals.append(compute_ssim(x_hat_01, hr_01))

        if batch_idx < cfg.get("n_save_batches", 3):
            n    = min(n_save, B)
            grid = make_grid(
                torch.cat([lr_01[:n], x_hat_01[:n], hr_01[:n]], dim=0),
                nrow=n, padding=4, pad_value=1.0
            )
            path = os.path.join(out_dir, f"grid_batch{batch_idx:03d}.png")
            save_image(grid, path)
            print(f"  Saved → {path}  (top=LR | mid=WG-DM | bot=GT)")

            for i in range(n):
                save_image(lr_01[i],    os.path.join(out_dir, f"b{batch_idx:03d}_{i:02d}_LR.png"))
                save_image(x_hat_01[i], os.path.join(out_dir, f"b{batch_idx:03d}_{i:02d}_WGDM.png"))
                save_image(hr_01[i],    os.path.join(out_dir, f"b{batch_idx:03d}_{i:02d}_GT.png"))

    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    print(f"\n{'─'*40}")
    print(f"PSNR : {avg_psnr:.2f} dB")
    print(f"SSIM : {avg_ssim:.4f}")
    print(f"Images → {out_dir}")
    print(f"{'─'*40}")
    return avg_psnr, avg_ssim


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--stage",
                        choices=["diffusion", "hf", "hf_finetune", "eval", "all"],
                        default="all")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.stage in ("diffusion", "all"):
        diff_model = train_diffusion(cfg, device)

    if args.stage in ("hf", "all"):
        if args.stage == "hf":
            diff_model = build_diff_model(cfg, device)
            diff_model.load_state_dict(torch.load(cfg["diff_ckpt"], map_location=device))
        hf_model = train_hf_predictor(cfg, device, diff_model)

    if args.stage == "hf_finetune":
        diff_model = build_diff_model(cfg, device)
        diff_model.load_state_dict(torch.load(cfg["diff_ckpt"], map_location=device))
        hf_model = build_hf_model(cfg, device)
        hf_model.load_state_dict(torch.load(cfg.get("hf_ckpt", cfg["ckpt_dir"] + "/hf_predictor.pt"), map_location=device))
        hf_model = finetune_hf_predictor(cfg, device, diff_model, hf_model)

    if args.stage in ("eval", "all"):
        if args.stage == "eval":
            diff_model = build_diff_model(cfg, device)
            diff_model.load_state_dict(torch.load(cfg["diff_ckpt"], map_location=device))
            hf_model = build_hf_model(cfg, device)
            ckpt = cfg.get("hf_ckpt", os.path.join(cfg["ckpt_dir"], "hf_finetuned.pt"))
            if not os.path.exists(ckpt):
                ckpt = os.path.join(cfg["ckpt_dir"], "hf_predictor.pt")
            hf_model.load_state_dict(torch.load(ckpt, map_location=device))
        evaluate(cfg, device, diff_model, hf_model)
