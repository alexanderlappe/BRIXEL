import argparse
import importlib
import math
import os
import random
import time
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import pickle

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import csv
from itertools import zip_longest

from models import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def color_to_tensor(img: Image.Image) -> torch.Tensor:
    # Convert PIL RGB to float tensor [0,1] then normalize
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (t - mean) / std


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    # Expect a single-channel PNG with int labels
    t = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H,W]
    # Map ADE: 0 -> 255 (ignore), 1..150 -> 0..149
    t = t.clone()
    t[t == 0] = 255
    valid = t != 255
    t[valid] = t[valid] - 1
    return t


def resize_pair(img: Image.Image, mask: Image.Image, size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    img_resized = img.resize(size, resample=Image.BILINEAR)
    mask_resized = mask.resize(size, resample=Image.NEAREST)
    return img_resized, mask_resized


def random_scale(img: Image.Image, mask: Image.Image, min_scale=0.5, max_scale=2.0) -> Tuple[Image.Image, Image.Image]:
    s = random.uniform(min_scale, max_scale)
    w, h = img.size
    new_w, new_h = int(w * s), int(h * s)
    return resize_pair(img, mask, (new_w, new_h))


def random_crop(img: Image.Image, mask: Image.Image, crop_size: int) -> Tuple[Image.Image, Image.Image]:
    w, h = img.size
    if w < crop_size or h < crop_size:
        # pad if needed
        pad_w = max(crop_size - w, 0)
        pad_h = max(crop_size - h, 0)
        if pad_w > 0 or pad_h > 0:
            # pad right/bottom with zeros
            new_img = Image.new('RGB', (w + pad_w, h + pad_h))
            new_img.paste(img, (0, 0))
            img = new_img
            new_mask = Image.new('L', (w + pad_w, h + pad_h), color=0)  # background (will be mapped to ignore)
            new_mask.paste(mask, (0, 0))
            mask = new_mask
            w, h = img.size
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    img_c = img.crop((x, y, x + crop_size, y + crop_size))
    mask_c = mask.crop((x, y, x + crop_size, y + crop_size))
    return img_c, mask_c


def center_crop_or_pad(img: Image.Image, mask: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    w, h = img.size
    # if smaller, pad; if larger, center-crop
    pad_w = max(size - w, 0)
    pad_h = max(size - h, 0)
    if pad_w > 0 or pad_h > 0:
        new_img = Image.new('RGB', (w + pad_w, h + pad_h))
        new_img.paste(img, (pad_w // 2, pad_h // 2))
        img = new_img
        new_mask = Image.new('L', (w + pad_w, h + pad_h), color=0)
        new_mask.paste(mask, (pad_w // 2, pad_h // 2))
        mask = new_mask
        w, h = img.size
    # center crop
    x = (w - size) // 2
    y = (h - size) // 2
    return img.crop((x, y, x + size, y + size)), mask.crop((x, y, x + size, y + size))


# -----------------------------
# Dataset
# -----------------------------

class ADE20KDataset(Dataset):
    def __init__(self, root: str, split: str = "training", img_size: int = 512, is_train: bool = True,
                 scale_min: float = 0.5, scale_max: float = 2.0):
        self.root = Path(root)
        assert split in ("training", "validation")
        self.split = split
        self.img_size = img_size
        print('Training on Image size {}'.format(img_size))
        self.is_train = is_train
        self.scale_min = scale_min
        self.scale_max = scale_max

        img_dir = self.root / "images" / split
        ann_dir = self.root / "annotations" / split
        self.images = sorted([p for p in img_dir.glob("*.jpg")])
        self.masks = [ann_dir / (p.stem + ".png") for p in self.images]
        assert len(self.images) == len(self.masks), "Mismatched images and masks"
        if len(self.images) == 0:
            raise FileNotFoundError(f"No JPEGs found under {img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # single-channel labels

        if self.is_train:
            # Random scale -> random horizontal flip -> random crop to square
            img, mask = random_scale(img, mask, self.scale_min, self.scale_max)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            img, mask = random_crop(img, mask, self.img_size)
        else:
            # Validation: resize short side to img_size then center crop/pad
            w, h = img.size
            short = min(w, h)
            scale = self.img_size / short
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img, mask = resize_pair(img, mask, (new_w, new_h))
            img, mask = center_crop_or_pad(img, mask, self.img_size)

        img_t = color_to_tensor(img)   # [3,H,W], normalized
        mask_t = mask_to_tensor(mask)  # [H,W], int64 with 0..149, 255 ignore
        return img_t, mask_t


# -----------------------------
# Backbone adapter and probe
# -----------------------------

def build_backbone(spec: str, config_path):
    """
    spec formats:
      - 'timm:MODEL_NAME' uses timm.create_model(MODEL_NAME, pretrained=True)
      - 'py:module.submodule:factory_name' imports factory and calls it -> nn.Module
    """
    import json
    with open(config_path) as f:
        config = json.load(f)
    config = config[spec]

    if config['model_class'] == "dinov3":
        backbone = torch.hub.load(config["repo_dir"], spec, source='local',
                                  weights=config["weight_path"])

    elif config['model_class'] == "dinov3_adapter":
        if 'adapter' in spec:
            load_name = spec.split("_adapter")[0]   # load name is the model name w.r.t. original DINOv3 implementation
        elif 'finetuned' in spec:
            load_name = spec.split("_finetuned")[0]
        backbone = torch.hub.load(config["repo_dir"], load_name, source='local',
                                  weights=config["weight_path"])
        adapter = DINOv3_Adapter(backbone=backbone, interaction_indexes=config["block_indices"])
        backbone = Upsampler(adapter)
        ckpt = torch.load(config['adapter_weight_path'], map_location="cpu")
        backbone.load_state_dict(ckpt, strict=False)

    elif config['model_class'] == "siglip":
        backbone = SiglipBackbone(config["weight_path"])
    elif config['model_class'] == "siglip_adapter":
        backbone = SiglipBackbone(config["weight_path"])
        adapter = DINOv3_Adapter(backbone=backbone, interaction_indexes=config["block_indices"])
        backbone = Upsampler(adapter)
        ckpt = torch.load(config['adapter_weight_path'], map_location="cpu")
        backbone.load_state_dict(ckpt, strict=False)

    return backbone, config['model_class']
    raise ValueError("Unknown backbone spec. Use 'timm:...' or 'py:module:factory'.")


class DepthReadout(nn.Module):
    def __init__(self, C, out_channels=1, mid=None, use_softplus=False):
        super().__init__()
        print("Using bigger depth readout.")
        mid = mid or max(32, C // 2)
        self.body = nn.Sequential(
            nn.Conv2d(C, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),  # depthwise context
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 1, bias=True)
        )

    def forward(self, x):
        y = self.body(x)
        return y



class LinearSegProbe(nn.Module):
    def __init__(self, backbone, backbone_class, out_ch: int = 150, depth_params=None, output_surface_normals=False):
        super().__init__()
        self.backbone = backbone
        self.backbone_class = backbone_class
        C = backbone.embed_dim if backbone_class == "dinov3" else backbone.adapter.backbone.embed_dim
        if C is None:
            raise ValueError("embed_dim not found; please specify --embed-dim")

        # For depth models
        if depth_params is not None:
            self.readout = DepthReadout(C, out_channels=1)
            self.min_depth = depth_params[0]
            self.max_depth = depth_params[1]
            if output_surface_normals:
                self.normal_readout = DepthReadout(C, out_channels=3)
        else:
            self.min_depth, self.max_depth = None, None
            self.readout = nn.Conv2d(C, out_ch, kernel_size=1, bias=True)

        self.output_surface_normals = output_surface_normals


    def forward(self, x: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        with torch.no_grad():
            if self.backbone_class == "dinov3":   # only outputs CLS token by default. Get patch tokens instead
                feats = self.backbone.get_intermediate_layers(x, n=1, return_class_token=False, reshape=True)[0]
            if self.backbone_class == "dinov3_adapter":
                feats = self.backbone(x)
        logits_lowres = self.readout(feats)             # [B,K,H',W']

        # show_img_and_feat(x[0], feats[0], feats[0], low_res_target=None)
        if out_size is None:
            out_size = x.shape[-2:]

        # logits are not really logits but unnormalized output of linear layer
        logits = F.interpolate(logits_lowres, size=out_size, mode="bilinear", align_corners=False)
        if self.output_surface_normals:
            normal_logits = self.normal_readout(feats)
            normal_logits = F.interpolate(normal_logits, size=out_size, mode="bilinear", align_corners=False)

        # If we output depth, logits are postprocessed
        if self.min_depth is not None:   # doing depth estimation
            logits = F.softplus(logits, beta=1, threshold=20.0) + self.min_depth
            logits = torch.clamp(logits, min=self.min_depth, max=self.max_depth)
            if self.output_surface_normals:   # put them back together
                logits = torch.cat([logits, normal_logits], dim=1)
        return logits


# -----------------------------
# Train / Eval
# -----------------------------

def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    # pred/target: [H,W]
    mask = target != ignore_index
    if mask.sum() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    t = target[mask].view(-1).cpu().numpy().astype(np.int64)
    p = pred[mask].view(-1).cpu().numpy().astype(np.int64)
    hist = np.bincount(num_classes * t + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int = 150, device: str = "cuda"):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)                    # [B,K,H,W]
        preds = logits.argmax(1)                # [B,H,W]
        for p, l in zip(preds, labels):
            hist += compute_confusion_matrix(p, l, num_classes=num_classes, ignore_index=255)

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = float(np.nanmean(iu))
    pix_acc = float(np.diag(hist).sum() / (hist.sum() + 1e-10))
    return miou, pix_acc, iu


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scaler, device: str = "cuda"):
    model.readout.train()  # only readout has grads
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    running_loss = 0.0
    n = 0

    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        running_loss += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
    return running_loss / max(n, 1)


def main():
    # TODO Remove some of the defaults. Only necessary for debugging.
    parser = argparse.ArgumentParser("ADE20K Linear Probe")

    parser.add_argument("--backbone_model_name", type=str, default='dinov3_vitb16',
                        help="Name of the backbone model.")
    parser.add_argument("--backbone_weight_path", type=str,
                        default='data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                        help="Path to pretrained backbone weights.")

    parser.add_argument("--data-root", type=str, required=False, default="data/ADE20k/ADEChallengeData2016/", help="Path to ADEChallengeData2016")
    parser.add_argument("--readout-out-ch", type=int, default=150, help="Number of classes (ADE20K=150)")

    parser.add_argument("--save-dir", type=str, default="data//results/ade20k")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Mixed precision")

    # Do only evaluation
    parser.add_argument("--eval-only", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")

    args = parser.parse_args()
    set_seed(args.seed)

    backbone_name = args.backbone

    os.makedirs(args.save-dir if hasattr(args, "save-dir") else args.save_dir, exist_ok=True)
    save_dir = Path(args.save_dir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # Datasets / Loaders
    train_ds = ADE20KDataset(args.data_root, split="training", img_size=args.img_size, is_train=False)
    val_ds = ADE20KDataset(args.data_root, split="validation", img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=True,
                            num_workers=args.workers, pin_memory=True)

    # Backbone & adapter

    model = LinearSegProbe(backbone, backbone_class=model_class, out_ch=args.readout_out_ch).to(device)

    # Load head if provided
    if args.head_ckpt is not None:
        sd = torch.load(args.head_ckpt, map_location="cpu")
        if "readout_state_dict" in sd:
            sd = sd["readout_state_dict"]
        missing, unexpected = model.readout.load_state_dict(sd, strict=False)
        print(f"Loaded head from {args.head_ckpt}. Missing: {missing}, Unexpected: {unexpected}")

    # Freeze backbone (already frozen in adapter); ensure only readout params require grad
    for n, p in model.named_parameters():
        if "readout" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW([p for p in model.readout.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    miu_history, pix_history = [], []
    start_time = time.time()

    if args.img_size != 256:
        backbone_name = backbone_name + '_' + str(args.img_size)
    print("I will save as {}.".format(backbone_name))

    if args.eval_only:
        miou, pix, _ = evaluate(model, val_loader, num_classes=args.readout_out_ch, device=device.type)
        print(f"[EVAL] mIoU={miou*100:.2f}  pixAcc={pix*100:.2f}")
        return

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device=device.type)
        miou, pix, _ = evaluate(model, val_loader, num_classes=args.readout_out_ch, device=device.type)

        elapsed = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] loss={train_loss:.4f}  mIoU={miou*100:.2f}  pixAcc={pix*100:.2f}  ({elapsed:.1f}s)")

        # Save latest head
        ckpt = {
            "epoch": epoch,
            "readout_state_dict": model.readout.state_dict(),
            "args": vars(args),
            "miou": miou,
        }
        torch.save(ckpt, os.path.join(save_dir, backbone_name + ".pt"))

        miu_history.append(miou)
        pix_history.append(pix)

        # miou_history and pix_history are your per-epoch lists
        result_rows = zip_longest(
            range(1, max(len(miu_history), len(pix_history)) + 1),
            miu_history,
            pix_history,
            fillvalue=""
        )

        csv_path = os.path.join(save_dir, backbone_name + ".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "miou", "pix"])
            writer.writerows(result_rows)

    total = time.time() - start_time
    print(f"Done. Best mIoU={miu_history[-1]*100:.2f}. Total time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
