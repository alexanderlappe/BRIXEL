#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset  # NEW
from torch.utils.data import Dataset
from torchvision import transforms as T

from modeling.dinov3.upsampling.eval.segmentation.ade20k_linear_probe import build_backbone, LinearSegProbe


# ---------------------------
# Utils
# ---------------------------

def seed_everything(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def human_time(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60); return f"{h:d}:{m:02d}:{s:02d}"

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def save_lines(lines, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ln in lines: f.write(ln + "\n")

# ---------------------------
# Dataset
# ---------------------------

class NYUv2(Dataset):
    """
    Expects a filelist with 'relative_rgb_path relative_depth_path' per line.
    Depth: 16-bit PNG in millimeters or float PNG in meters. Converted to meters.
    """
    def __init__(self, root, filelist, split, img_size=(480,640), max_depth=10.0, min_depth=0.001, augment=False):
        self.root = Path(root)
        self.items = []
        with open(filelist, 'r') as f:
            for line in f:
                if not line.strip(): continue
                rgb, dep = line.strip().split()
                self.items.append((rgb, dep))
        self.img_size = img_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.augment = (split == 'train' and augment)

        self.resize_img = T.Resize(self.img_size, interpolation=T.InterpolationMode.BICUBIC)
        self.resize_dep = T.Resize(self.img_size, interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()
        self.color_aug = T.ColorJitter(0.2,0.2,0.2,0.05) if self.augment else None

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def _read_depth(self, p):
        with Image.open(p) as d:
            arr = np.array(d)
        if arr.dtype == np.uint16:
            depth_m = arr.astype(np.float32) / 1000.0
        else:
            depth_m = arr.astype(np.float32)
        return depth_m

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rgb_rel, dep_rel = self.items[idx]
        rgb_p = (self.root / rgb_rel).as_posix()
        dep_p = (self.root / dep_rel).as_posix()

        with Image.open(rgb_p) as im:
            im = im.convert('RGB')
        depth_m = self._read_depth(dep_p)
        depth_m = np.clip(depth_m, self.min_depth, self.max_depth)
        valid = (depth_m > self.min_depth).astype(np.float32)

        im = self.resize_img(im)
        # if self.augment:
        #     if np.random.rand() < 0.5:
        #         im = T.functional.hflip(im)
        #         depth_m = np.fliplr(depth_m).copy()
        #         valid = np.fliplr(valid).copy()
        #     if np.random.rand() < 0.8:
        #         im = self.color_aug(im)

        img_t = self.to_tensor(im)
        depth_t = torch.from_numpy(depth_m).float()
        valid_t = torch.from_numpy(valid).float()

        # normalize (ImageNet)
        img_t = (img_t - self.mean) / self.std
        return {'image': img_t, 'depth': depth_t, 'valid': valid_t, 'meta': {'rgb_path': rgb_rel}}

# ---------------------------
# Loss & metrics
# ---------------------------

def silog_loss(pred, target, valid, beta=0.15, eps=1e-8):
    """
    pred, target: (B,1,H,W)
    valid: (B,1,H,W) or (B,H,W) — masks valid pixels (1.0) vs invalid (0.0)
    """
    if valid.dim() == 3:               # (B,H,W) -> (B,1,H,W)
        valid = valid.unsqueeze(1)

    pred = pred.clamp_min(eps)
    target = target.clamp_min(eps)

    g = (pred.log() - target.log()) * valid  # (B,1,H,W)

    # Count valid pixels per-sample (keep dims for safe broadcasting)
    n = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)   # (B,1,1,1)

    mean = g.sum(dim=(-2, -1), keepdim=True) / n               # (B,1,1,1)
    loss = (g.pow(2).sum(dim=(-2, -1), keepdim=True) / n) - beta * (mean.pow(2))  # (B,1,1,1)

    return loss.mean()  # scalar

@torch.no_grad()
def compute_metrics(pred, gt, valid, min_d=1e-3, max_d=10.0):
    if valid.ndim == 3: valid = valid.unsqueeze(1)
    v = valid > 0.5
    pred = pred.clamp(min=min_d, max=max_d)
    gt = gt.clamp(min=min_d, max=max_d)
    pred, gt, v = pred.squeeze(), gt.squeeze(), v.squeeze()
    p = pred[v]; g = gt[v]
    absrel = torch.mean(torch.abs(p-g)/g).item()
    rmse = torch.sqrt(torch.mean((p-g)**2)).item()
    log10 = torch.mean(torch.abs(torch.log10(p) - torch.log10(g))).item()
    max_ratio = torch.maximum(p/g, g/p)
    d1 = (max_ratio < 1.25).float().mean().item()
    d2 = (max_ratio < 1.25**2).float().mean().item()
    d3 = (max_ratio < 1.25**3).float().mean().item()
    return {'AbsRel': absrel, 'RMSE': rmse, 'log10': log10, 'δ1': d1, 'δ2': d2, 'δ3': d3}

# ---------------------------
# Train/Eval
# ---------------------------

def make_loaders(args):
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    ds = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(args.data_root, "data/train-*.parquet"),
            "val": os.path.join(args.data_root, "data/val-*.parquet"),
        }
    )
    ds = ds.with_format("torch")

    def to_model_batch(ex):
        # --- image ---
        x = torch.as_tensor(ex["image"], dtype=torch.float32)

        # HWC->CHW (single) or BHWC->BCHW (batched)
        if x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        elif x.ndim == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)

        # normalize (broadcast-aware)
        x = (x - (IMAGENET_MEAN if x.ndim == 3 else IMAGENET_MEAN[None])) / \
            (IMAGENET_STD if x.ndim == 3 else IMAGENET_STD[None])

        # --- depth ---
        d = torch.as_tensor(ex["depth"], dtype=torch.float32)
        # make it (H,W) or (B,H,W): squeeze any singleton channel
        if d.ndim == 3 and d.shape[0] == 1:  # (1,H,W)
            d = d[0]
        elif d.ndim == 3 and d.shape[-1] == 1:  # (H,W,1)
            d = d[..., 0]
        elif d.ndim == 4 and d.shape[1] == 1:  # (B,1,H,W)
            d = d[:, 0]
        elif d.ndim == 4 and d.shape[-1] == 1:  # (B,H,W,1)
            d = d[..., 0]
        # else: already (H,W) or (B,H,W)

        v = (d > 1e-3).to(torch.float32)  # same shape as d

        return {"image": x, "depth": d, "valid": v}

    ds["train"].set_transform(to_model_batch)
    ds["val"].set_transform(to_model_batch)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def nyu_eval(pred, gt, valid, align="scale_shift", min_d=1e-3, max_d=10.0):
    if valid.ndim==3: valid=valid.unsqueeze(1)
    v = (valid>0.5)
    p = pred.clamp(min=min_d, max=max_d)
    g = gt.clamp(min=min_d, max=max_d)

    if align == "median":
        s = (g[v].median() / p[v].median().clamp_min(1e-6))
        p = p * s
    elif align == "scale_shift":
        # Solve argmin_{a,b} ||a*p + b - g||_2 over valid pixels
        pv, gv = (p*v).flatten(2), (g*v).flatten(2)
        ones = torch.ones_like(pv)
        A11 = (pv*pv).sum(-1); A12 = pv.sum(-1); A22 = ones.sum(-1)
        b1  = (pv*gv).sum(-1); b2  = gv.sum(-1)
        det = (A11*A22 - A12*A12).clamp_min(1e-12)
        a = (A22*b1 - A12*b2) / det
        b = (A11*b2 - A12*b1) / det
        p = p * a.view(-1,1,1,1) + b.view(-1,1,1,1)

    return compute_metrics(p, g, v, min_d, max_d)

def adjust_lr_cos(optimizer, step, total_steps, base_lr, min_lr=1e-6, warmup=0):
    if step < warmup:
        lr = base_lr * step / max(1, warmup)
    else:
        t = (step - warmup) / max(1, total_steps - warmup)
        lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*t))
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def train_and_eval(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_dl, test_dl = make_loaders(args)

    # Backbone & adapter
    backbone, model_class = build_backbone(args.backbone, args.config_path)
    model = LinearSegProbe(backbone, backbone_class=model_class, out_ch=1,
                           depth_params=(args.min_depth, args.max_depth)).to(device)
    model.eval()

    # Only head params by default
    # Freeze backbone (already frozen in adapter); ensure only readout params require grad
    for n, p in model.named_parameters():
        if "readout" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW([p for p in model.readout.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_dl)
    step = 0
    history = {
        'config': vars(args),
        'train': [],            # per-step (optional) and per-epoch losses
        'eval_per_epoch': []    # test metrics per epoch
    }
    per_step_log = []

    start = time.time()
    for epoch in range(1, args.epochs+1):
        model.readout.train()
        running = 0.0
        for it, batch in enumerate(train_dl, 1):
            img = batch['image'].to(device, non_blocking=True)
            dep = batch['depth'].unsqueeze(1).to(device, non_blocking=True)
            val = batch['valid'].unsqueeze(1).to(device, non_blocking=True)
            print(img.shape, dep.shape, val.shape)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(img)
                loss = silog_loss(pred, dep, val, beta=args.silog_beta)

            loss.backward()
            optimizer.step()

            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                eta = (time.time() - start) / max(1, step) * (total_steps - step)
                msg = f"[ep {epoch:02d}] step {step}/{total_steps} loss={running/it:.4f} eta={human_time(eta)}"
                print(msg)
                per_step_log.append({'epoch': epoch, 'step': step, 'loss': float(loss.item())})

        epoch_loss = running / max(1, len(train_dl))
        history['train'].append({'epoch': epoch, 'loss': float(epoch_loss)})

        # -------- evaluate on TEST after each epoch --------
        model.eval()
        eval_metrics = []
        with torch.no_grad():
            for batch in test_dl:
                img = batch['image'].to(device)
                dep = batch['depth'].unsqueeze(1).to(device)
                val = batch['valid'].unsqueeze(1).to(device)
                pred = model(img)
                eval_metrics.append(nyu_eval(pred, dep, val, align="scale_shift", min_d=1e-3, max_d=10.0))
        # aggregate
        keys = eval_metrics[0].keys()
        agg = {k: float(np.mean([m[k] for m in eval_metrics])) for k in keys}
        agg['epoch'] = epoch
        history['eval_per_epoch'].append(agg)
        print(f"[test@ep{epoch}] AbsRel={agg['AbsRel']:.4f} RMSE={agg['RMSE']:.4f} log10={agg['log10']:.4f} "
              f"d1={agg['δ1']:.3f} d2={agg['δ2']:.3f} d3={agg['δ3']:.3f}")

        # persist history every epoch
        save_json(history, Path(args.save_dir) / "history.json")
        # a light CSV for quick plotting
        csv_lines = ["epoch,train_loss,AbsRel,RMSE,log10,delta1,delta2,delta3"]
        for ep_rec, te_rec in zip(history['train'], history['eval_per_epoch']):
            csv_lines.append(f"{ep_rec['epoch']},{ep_rec['loss']:.6f},{te_rec['AbsRel']:.6f},{te_rec['RMSE']:.6f},"
                             f"{te_rec['log10']:.6f},{te_rec['δ1']:.6f},{te_rec['δ2']:.6f},{te_rec['δ3']:.6f}")
        save_lines(csv_lines, Path(args.save_dir) / "history.csv")

        # optional detailed per-step log
        save_json(per_step_log, Path(args.save_dir) / "per_step_log.json")
    # save checkpoints (head-only or full)
    ckpt = {
        'epoch': epoch,
        'model': model.readout.state_dict(),
        'args': vars(args),
        'train_epoch_loss': float(epoch_loss),
        'test_metrics': agg
    }
    torch.save(ckpt, os.path.join(args.save_dir, f"epoch{epoch:03d}.pth"))

# ---------------------------
# CLI
# ---------------------------

def get_args():
    # TODO Remove the defaults and set necessary arguments properly
    ap = argparse.ArgumentParser("NYUv2 Depth")
    ap.add_argument("--data-root", type=str, required=True, help='Path to dir containing <data>, which contains NYU parquets.')
    ap.add_argument("--backbone", type=str, default="dinov3_vitb16")
    ap.add_argument("--max-depth", type=float, default=10.0)
    ap.add_argument("--min_depth", type=float, default=0.001)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--silog-beta", type=float, default=0.15)
    ap.add_argument("--no-aug", action="store_true")


    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-dir", type=str, default="eval/results/NYU")
    return ap.parse_args()

def main():
    args = get_args()
    args.save_dir = os.path.join(args.save_dir, args.backbone)

    train_and_eval(args)

if __name__ == "__main__":
    main()
