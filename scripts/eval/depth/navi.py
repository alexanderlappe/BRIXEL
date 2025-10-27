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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# ---- reuse your existing helpers/backbone exactly like in nyu.py ----
from modeling.dinov3.upsampling.eval.depth.nyu import seed_everything, human_time, save_json, save_lines, silog_loss  # :contentReference[oaicite:1]{index=1}
from modeling.dinov3.upsampling.eval.segmentation.ade20k_linear_probe import build_backbone
from modeling.dinov3.upsampling.eval.depth.probes import DepthModel


# ---------------------------
# NAVI dataset
# ---------------------------

# --- helper: center-crop to square (keep shortest side) ---
def center_crop_square(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    return pil_img.crop((left, top, left + s, top + s))



class NAVIWildSet(Dataset):
    """
    Expects NAVI layout:
      <data_root>/<class>/wild_set/{images,depth,masks,annotations.json}
    Uses hardcoded split lists in SPLIT_FILES to pick samples:
      each line in the list is a RELATIVE path to a depth png, e.g. 'Mug/wild_set/depth/000123.png'
    Transforms:
      - center-crop to square (keep shortest side)
      - resize to (square_size, square_size)
      - ImageNet normalization
    Returns:
      image:  (3,H,W) float
      depth:  (H,W)   float meters
      valid:  (H,W)   float {0,1}
      normal: (3,H,W) float unit vector (NaN outside valid)
      nvalid: (H,W)   float {0,1}
      meta:   dict
    """

    def __init__(self, data_root, split="train", square_size=384,
                 augment=False, bilateral_pre_smooth=False):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.root = Path(data_root)
        self.items = []  # dicts with img/dep/msk + intrinsics
        self.square_size = int(square_size)
        self.augment = (split == "train" and augment)
        self.bilateral_pre_smooth = bilateral_pre_smooth

        SPLIT_FILES = {
            "train": os.path.join(data_root, "navi_train.txt"),
            "val":   os.path.join(data_root, "navi_val.txt")
        }

        # --------- load from hardcoded split file ----------
        split_file = SPLIT_FILES[split]
        rel_depths = []
        with open(split_file, "r") as f:
            for ln in f:
                ln = ln.strip()
                if ln: rel_depths.append(ln)

        for rel in rel_depths:
            dpng = (self.root / rel)
            if not dpng.exists():
                continue
            wild_dir = dpng.parent.parent  # .../<class>/wild_set
            img_dir, msk_dir = wild_dir / "images", wild_dir / "masks"
            stem = dpng.stem
            ip = img_dir / f"{stem}.jpg"
            if not ip.exists():
                ip = img_dir / f"{stem}.png"
            mp = msk_dir / f"{stem}.png"
            if not (ip.exists() and mp.exists()):
                continue

            ann_path = wild_dir / "annotations.json"
            ann = self._load_annotations(ann_path) if ann_path.exists() else {}
            fx, fy, cx, cy = self._get_intrinsics_for(ip.name, ann, fallback=None)

            self.items.append({
                "img": ip, "dep": dpng, "msk": mp,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy
            })

        # --------- transforms (square crop -> resize) ----------
        self.resize_img = T.Resize((self.square_size, self.square_size), interpolation=T.InterpolationMode.BICUBIC)
        self.resize_dep = T.Resize((self.square_size, self.square_size), interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()
        self.color_aug = T.ColorJitter(0.2, 0.2, 0.2, 0.05) if self.augment else None

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # ---------- JSON / intrinsics ----------
    @staticmethod
    def _load_annotations(path):
        with open(path, "r") as f:
            entries = json.load(f)
        by_fn = {}
        for e in entries:
            fn = e["filename"]
            cam = e.get("camera", {})
            fl = cam.get("focal_length", None)
            by_fn[fn] = {"fl": fl}
        return by_fn

    def _get_intrinsics_for(self, fname, ann_map, fallback):
        if fname in ann_map and ann_map[fname]["fl"] is not None:
            fl = float(ann_map[fname]["fl"])
            return fl, fl, None, None
        return None, None, None, None

    # ---------- I/O ----------
    @staticmethod
    def _read_depth_png(p: Path):
        with Image.open(p) as d:
            arr = np.array(d)
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 1000.0
        return arr.astype(np.float32)

    @staticmethod
    def _read_mask_png(p: Path):
        with Image.open(p) as m:
            arr = np.array(m)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 0).astype(np.uint8)

    # ---------- geometry ----------
    @staticmethod
    def _backproject(depth, fx, fy, cx, cy):
        H, W = depth.shape
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        Z = depth
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        return np.stack([X, Y, Z], axis=-1)

    def _normals_from_depth(self, depth, mask, fx, fy, cx, cy):
        depth = depth.copy()
        depth[~mask] = 0.0
        if self.bilateral_pre_smooth:
            import cv2
            v = depth.astype(np.float32)
            v_mask = (v > 0).astype(np.uint8)
            v_sm = cv2.bilateralFilter(v, d=7, sigmaColor=0.1, sigmaSpace=5)
            depth = np.where(v_mask > 0, v_sm, 0.0)

        P = self._backproject(depth, fx, fy, cx, cy)
        Pv = np.zeros_like(P); Pu = np.zeros_like(P)
        Pv[:-1, :, :] = P[1:, :, :] - P[:-1, :, :]
        Pu[:, :-1, :] = P[:, 1:, :] - P[:, :-1, :]

        N = np.cross(Pu, Pv)
        norm = np.linalg.norm(N, axis=-1, keepdims=True)
        N = N / (norm + 1e-8)

        invalid = (depth <= 0) | np.isnan(depth)
        invalid[:, -1] = True
        invalid[-1, :] = True

        flip = N[..., 2] < 0
        N[flip] *= -1.0
        N[invalid] = np.nan
        nvalid = (~invalid).astype(np.float32)
        return N, nvalid

    # ---------- core ----------
    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        ip, dp, mp = it["img"], it["dep"], it["msk"]

        # read raw
        with Image.open(ip) as im:
            im = im.convert("RGB")
        depth = self._read_depth_png(dp)
        mask = self._read_mask_png(mp).astype(bool)

        # ---------- square center-crop (image, depth, mask) ----------
        im = center_crop_square(im)
        d_img = center_crop_square(Image.fromarray(depth))
        m_img = center_crop_square(Image.fromarray((mask.astype(np.uint8) * 255)))

        # ---------- resize to (S, S) ----------
        im = self.resize_img(im)
        d_img = self.resize_dep(d_img)
        m_img = self.resize_dep(m_img)

        # back to numpy/tensor
        depth = np.array(d_img).astype(np.float32)
        mask = (np.array(m_img) > 0)

        # intrinsics after crop+resize:
        H, W = depth.shape  # W == H == square_size
        fx, fy = it["fx"], it["fy"]
        if fx is None or fy is None:
            fx = fy = 1.2 * W  # gentle fallback if focal length absent
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # normals
        normals, nvalid = self._normals_from_depth(depth, mask, fx, fy, cx, cy)

        # valid depth mask
        valid = ((depth > 0.05) & mask).astype(np.float32)

        # image tensor + normalization (+ optional color aug before normalize)
        if self.augment and self.color_aug is not None:
            im = self.color_aug(im)
        img_t = self.to_tensor(im)
        img_t = (img_t - self.mean) / self.std

        depth_t = torch.from_numpy(depth).float()
        valid_t = torch.from_numpy(valid).float()
        normal_t = torch.from_numpy(normals.transpose(2, 0, 1)).float()
        nvalid_t = torch.from_numpy(nvalid).float()

        return {
            "image": img_t,
            "depth": depth_t,
            "valid": valid_t,
            "normal": normal_t,
            "nvalid": nvalid_t,
            "meta": {"img_path": ip.as_posix()}
        }


# ---------------------------
# Losses & metrics
# ---------------------------

def cosine_normal_loss(pred_n, gt_n, valid_mask, eps=1e-8):
    """
    pred_n: (B,3,H,W) — not necessarily unit; we'll normalize
    gt_n:   (B,3,H,W) — unit where valid, NaN or 0 elsewhere
    valid_mask: (B,1,H,W) or (B,H,W) — 1 where normals valid
    Returns mean(1 - cosθ) over valid pixels.
    """
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    # replace NaNs in gt with 0, but mask them out anyway
    gt = torch.nan_to_num(gt_n, nan=0.0)
    # normalize pred
    p = pred_n / (pred_n.norm(dim=1, keepdim=True) + eps)
    # cosine
    cos = (p * gt).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss = (1.0 - cos) * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return loss.sum() / denom

@torch.no_grad()
def normal_metrics(pred_n, gt_n, valid_mask, eps=1e-8):
    """
    Returns dict with mean/median angular error (deg) and % under thresholds.
    """
    if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)
    gt = torch.nan_to_num(gt_n, nan=0.0)
    p = pred_n / (pred_n.norm(dim=1, keepdim=True) + eps)
    g = gt / (gt.norm(dim=1, keepdim=True) + eps)

    cos = (p * g).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # (B,1,H,W)
    ang = torch.acos(cos) * (180.0 / math.pi)                # degrees
    v = (valid_mask > 0.5)
    vals = ang[v]

    if vals.numel() == 0:
        return {"ang_mean": float("nan"), "ang_med": float("nan"),
                "t_11.25": float("nan"), "t_22.5": float("nan"), "t_30": float("nan")}
    m = {
        "ang_mean": float(vals.mean().item()),
        "ang_med":  float(vals.median().item()),
        "t_11.25": float((vals < 11.25).float().mean().item()),
        "t_22.5":  float((vals < 22.5).float().mean().item()),
        "t_30":    float((vals < 30.0).float().mean().item()),
    }
    return m


# ---------------------------
# Data loaders
# ---------------------------

def make_loaders(args):
    train_ds = NAVIWildSet(args.data_root, split="train",
                           square_size=args.img_size,
                           augment=not args.no_aug,
                           bilateral_pre_smooth=args.bilateral_smooth)
    val_ds   = NAVIWildSet(args.data_root, split="val",
                           square_size=args.img_size,
                           augment=False,
                           bilateral_pre_smooth=args.bilateral_smooth)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


# ---------------------------
# Train/Eval
# ---------------------------

@torch.no_grad()
def eval_epoch(model, loader, device, min_d=1e-3, max_d=10.0):
    depth_metrics = []
    normal_metrics_list = []
    for batch in loader:
        img = batch["image"].to(device)
        dep = batch["depth"].to(device).unsqueeze(1)
        val = batch["valid"].to(device).unsqueeze(1)
        nrm = batch["normal"].to(device)
        nval= batch["nvalid"].to(device).unsqueeze(1)

        pred = model(img)  # (B, 4, H, W): [depth, nx, ny, nz]
        pred_d, pred_n = model(img)

        # Depth metrics (like your NYU flow, simple and stable)
        # (reuse your compute_metrics style; here we inline a minimal version)
        pd = pred_d.clamp(min=min_d, max=max_d)
        gd = dep.clamp(min=min_d, max=max_d)
        v = (val > 0.5)
        p, g = pd[v], gd[v]
        if p.numel() == 0:
            dm = {"AbsRel": float("nan"), "RMSE": float("nan"), "log10": float("nan"),
                  "δ1": float("nan"), "δ2": float("nan"), "δ3": float("nan")}
        else:
            absrel = torch.mean(torch.abs(p-g)/g).item()
            rmse = torch.sqrt(torch.mean((p-g)**2)).item()
            log10 = torch.mean(torch.abs(torch.log10(p) - torch.log10(g))).item()
            max_ratio = torch.maximum(p/g, g/p)
            d1 = (max_ratio < 1.25).float().mean().item()
            d2 = (max_ratio < 1.25**2).float().mean().item()
            d3 = (max_ratio < 1.25**3).float().mean().item()
            dm = {'AbsRel': absrel, 'RMSE': rmse, 'log10': log10, 'δ1': d1, 'δ2': d2, 'δ3': d3}
        depth_metrics.append(dm)

        # Normal metrics
        nm = normal_metrics(pred_n, nrm, nval)
        normal_metrics_list.append(nm)

    # aggregate means
    def mean_of(keys, recs):
        return {k: float(np.nanmean([r[k] for r in recs])) for k in keys}

    dkeys = list(depth_metrics[0].keys()) if depth_metrics else []
    nkeys = list(normal_metrics_list[0].keys()) if normal_metrics_list else []
    d_agg = mean_of(dkeys, depth_metrics) if dkeys else {}
    n_agg = mean_of(nkeys, normal_metrics_list) if nkeys else {}
    return d_agg, n_agg


def train_and_eval(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_loaders(args)

    # Backbone & a single head that outputs depth + 3 normals
    backbone, model_class = build_backbone(args.backbone, args.config_path)  # :contentReference[oaicite:3]{index=3}
    model = DepthModel(backbone, model_class, args.img_size, output_surface_normals=True)
    model = model.to(device)
    model.eval()

    # Freeze backbone, train readout only (same pattern as NYU)
    for n, p in model.named_parameters():
        if "readout" in n: p.requires_grad_(True)
        else: p.requires_grad_(False)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_dl)
    step = 0
    history = {'config': vars(args), 'train': [], 'eval_per_epoch': []}
    per_step_log = []
    start = time.time()

    for epoch in range(1, args.epochs+1):
        model.readout.train()
        model.normal_readout.train()
        running = 0.0
        for it, batch in enumerate(train_dl, 1):
            img = batch["image"].to(device, non_blocking=True)
            dep = batch["depth"].to(device, non_blocking=True).unsqueeze(1)
            val = batch["valid"].to(device, non_blocking=True).unsqueeze(1)
            nrm = batch["normal"].to(device, non_blocking=True)
            nval = batch["nvalid"].to(device, non_blocking=True).unsqueeze(1)
            print(img.shape, dep.shape, val.shape)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_d, pred_n = model(img)             # (B,4,H,W)

                # depth loss (reuse your SiLog)
                loss_d = silog_loss(pred_d, dep, val)
                # normals loss
                loss_n = cosine_normal_loss(pred_n, nrm, nval)
                loss = args.lambda_depth * loss_d + args.lambda_normal * loss_n

            loss.backward()
            optim.step()

            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                eta = (time.time() - start) / max(1, step) * (total_steps - step)
                print(f"[ep {epoch:02d}] step {step}/{total_steps} loss={running/it:.4f} "
                      f"ld={loss_d.item():.4f} ln={loss_n.item():.4f} eta={human_time(eta)}")
                per_step_log.append({'epoch': epoch, 'step': step,
                                     'loss': float(loss.item()),
                                     'loss_d': float(loss_d.item()),
                                     'loss_n': float(loss_n.item())})

        epoch_loss = running / max(1, len(train_dl))
        history['train'].append({'epoch': epoch, 'loss': float(epoch_loss)})

        # ---- Eval ----
        model.eval()
        with torch.no_grad():
            d_metrics, n_metrics = eval_epoch(model, val_dl, device, min_d=args.min_depth, max_d=args.max_depth)

        rec = {**d_metrics, **{f"norm_{k}": v for k, v in n_metrics.items()}, 'epoch': epoch}
        history['eval_per_epoch'].append(rec)
        print(f"[val@ep{epoch}] "
              f"AbsRel={d_metrics.get('AbsRel', float('nan')):.4f} "
              f"RMSE={d_metrics.get('RMSE', float('nan')):.4f} "
              f"log10={d_metrics.get('log10', float('nan')):.4f} "
              f"d1={d_metrics.get('δ1', float('nan')):.3f} "
              f"nAngMean={n_metrics.get('ang_mean', float('nan')):.2f} "
              f"n<11.25={n_metrics.get('t_11.25', float('nan')):.3f}")

        # save history each epoch
        save_json(history, Path(args.save_dir) / "history.json")
        # quick CSV
        csv_lines = ["epoch,train_loss,AbsRel,RMSE,log10,delta1,delta2,delta3,ang_mean,ang_med,t_11.25,t_22.5,t_30"]
        for ep_rec, te_rec in zip(history['train'], history['eval_per_epoch']):
            csv_lines.append(
                f"{ep_rec['epoch']},{ep_rec['loss']:.6f},"
                f"{te_rec.get('AbsRel', float('nan')):.6f},"
                f"{te_rec.get('RMSE', float('nan')):.6f},"
                f"{te_rec.get('log10', float('nan')):.6f},"
                f"{te_rec.get('δ1', float('nan')):.6f},"
                f"{te_rec.get('δ2', float('nan')):.6f},"
                f"{te_rec.get('δ3', float('nan')):.6f},"
                f"{te_rec.get('norm_ang_mean', float('nan')):.6f},"
                f"{te_rec.get('norm_ang_med', float('nan')):.6f},"
                f"{te_rec.get('norm_t_11.25', float('nan')):.6f},"
                f"{te_rec.get('norm_t_22.5', float('nan')):.6f},"
                f"{te_rec.get('norm_t_30', float('nan')):.6f}"
            )
        save_lines(csv_lines, Path(args.save_dir) / "history.csv")
        save_json(per_step_log, Path(args.save_dir) / "per_step_log.json")

        # checkpoint head
        ckpt = {
            'epoch': epoch,
            'readout_state_dict': model.readout.state_dict(),
            'normal_readout_state_dict': (
                model.normal_readout.state_dict() if hasattr(model, 'normal_readout') else None
            ),
            'args': vars(args),
            'train_epoch_loss': float(epoch_loss),
            'val_depth_metrics': d_metrics,
            'val_normal_metrics': n_metrics,
        }

        torch.save(ckpt, os.path.join(args.save_dir, "model.pth"))


# ---------------------------
# CLI
# ---------------------------

def get_args():
    ap = argparse.ArgumentParser("NAVI wild_set — Multi-task Depth + Normals (linear readout)")
    ap.add_argument("--data-root", type=str, required=False, default="/media/alex/DataHDD/Alex/DENSE/navi_v1.0/",
                    help="Path to NAVI root that contains <class>/wild_set/{images,depth,masks,annotations.json}")
    ap.add_argument("--config-path", type=str, required=False, default="/home/alex/PycharmProjects/TheSecretOne/modeling/dinov3/upsampling/eval/local_eval_config.json",
                    help="Path to model config file (same one used in NYU script)")
    ap.add_argument("--backbone", type=str, default="dinov3_vits16_adapter")

    ap.add_argument("--img-size", type=int, default=256)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)

    ap.add_argument("--max-depth", type=float, default=5.0)
    ap.add_argument("--min-depth", type=float, default=0.05)
    ap.add_argument("--silog-beta", type=float, default=0.15)

    ap.add_argument("--lambda-depth", type=float, default=1.0)
    ap.add_argument("--lambda-normal", type=float, default=1.0)

    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--bilateral-smooth", dest="bilateral_smooth", action="store_true")
    ap.add_argument("--save-dir", type=str, required=False, default="/media/alex/DataSSD/alex/Dense/eval/results/NAVI/")
    return ap.parse_args()


def main():
    args = get_args()
    args.save_dir = os.path.join(args.save_dir, args.backbone)
    train_and_eval(args)


if __name__ == "__main__":
    main()
