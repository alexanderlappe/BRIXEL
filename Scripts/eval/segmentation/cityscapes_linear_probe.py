#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cityscapes Linear Probe (Segmentation) — PyTorch
------------------------------------------------
Analogous to ade20k_linear_probe.py, but for the Cityscapes dataset (19 classes, labelId→trainId mapping).

Directory layout expected:
  cityscapes/
    leftImg8bit/
      train/<city>/*_leftImg8bit.png
      val/<city>/*_leftImg8bit.png
    gtFine/
      train/<city>/*_gtFine_labelIds.png
      val/<city>/*_gtFine_labelIds.png

Usage example:
    python cityscapes_linear_probe.py \
        --data-root /path/to/cityscapes \
        --config-path /path/to/eval_config.json \
        --backbone dinov3_vitl16 \
        --readout-out-ch 19 \
        --img-size 512 \
        --epochs 40 \
        --batch-size 16 \
        --amp
"""

import argparse
import csv
import os
from itertools import zip_longest
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from modeling.dinov3.upsampling.eval.segmentation.ade20k_linear_probe import set_seed, LinearSegProbe, \
    compute_confusion_matrix, train_one_epoch, build_backbone, color_to_tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_pair(img: Image.Image, mask: Image.Image, size: Tuple[int, int]):
    return img.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST)

def center_crop_or_pad(img: Image.Image, mask: Image.Image, size: int):
    w, h = img.size
    pad_w, pad_h = max(size - w, 0), max(size - h, 0)
    if pad_w > 0 or pad_h > 0:
        new_img = Image.new('RGB', (w + pad_w, h + pad_h))
        new_img.paste(img, (pad_w // 2, pad_h // 2))
        img = new_img
        new_mask = Image.new('L', (w + pad_w, h + pad_h), color=0)
        new_mask.paste(mask, (pad_w // 2, pad_h // 2))
        mask = new_mask
        w, h = img.size
    x, y = (w - size) // 2, (h - size) // 2
    return img.crop((x, y, x + size, y + size)), mask.crop((x, y, x + size, y + size))

CITYSCAPES_ID_TO_TRAINID: Dict[int, int] = {
    -1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18
}

def labelids_to_trainids(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, 255, dtype=np.uint8)
    for k, v in CITYSCAPES_ID_TO_TRAINID.items():
        out[arr == k] = v
    return out

class CityscapesDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512, is_train: bool = True,
                 scale_min: float = 0.5, scale_max: float = 2.0):
        assert split in ("train", "val")
        self.root, self.split, self.img_size, self.is_train = Path(root), split, img_size, is_train
        self.scale_min, self.scale_max = scale_min, scale_max
        self.samples = []
        print('Training on Image size {}'.format(img_size))
        img_dir, lbl_dir = self.root / "leftImg8bit_trainvaltest" / "leftImg8bit" / split, self.root / "gtFine_trainvaltest" / "gtFine" / split
        for city in sorted(img_dir.iterdir()):
            if city.is_dir():
                for img_path in sorted(city.glob("*_leftImg8bit.png")):
                    stem = img_path.stem.replace("_leftImg8bit", "")
                    lbl_path = lbl_dir / city.name / f"{stem}_gtFine_labelIds.png"
                    if lbl_path.exists():
                        self.samples.append((img_path, lbl_path))
        if not self.samples:
            raise FileNotFoundError(f"No samples found in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img, mask = Image.open(img_path).convert("RGB"), Image.open(lbl_path)

        w, h = img.size
        short = min(w, h)
        scale = self.img_size / short
        img, mask = resize_pair(img, mask, (int(w * scale), int(h * scale)))
        ### leave the images in original aspect ratio
        # img, mask = center_crop_or_pad(img, mask, self.img_size)
        mask_np = labelids_to_trainids(np.array(mask, dtype=np.int32))
        return color_to_tensor(img), torch.from_numpy(mask_np.astype(np.int64))


@torch.no_grad()
def evaluate(model, loader, num_classes=19, device="cuda"):
    model.eval()
    hist = np.zeros((num_classes, num_classes), np.int64)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for p, l in zip(preds, labels):
            hist += compute_confusion_matrix(p, l, num_classes)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    return float(np.nanmean(iu)), float(np.diag(hist).sum() / (hist.sum() + 1e-10)), iu

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='/media/alex/DataSSD/alex/Dense/eval/Cityscapes/')
    p.add_argument('--config-path', type=str, default='/home/alex/PycharmProjects/TheSecretOne/modeling/dinov3/upsampling/eval/eval_config.json')
    p.add_argument('--backbone', type=str, default='dinov3_vits16')
    p.add_argument('--readout-out-ch', type=int, default=19)
    p.add_argument('--save-dir', type=str, default='/media/alex/DataSSD/alex/Dense/eval/results/cityscapes/')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-9)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_ds = CityscapesDataset(args.data_root, 'train', args.img_size, )
    val_ds = CityscapesDataset(args.data_root, 'val', args.img_size, False)
    train_loader = DataLoader(train_ds, args.batch_size, True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, max(1, args.batch_size // 2), False, num_workers=args.workers)
    backbone, model_class = build_backbone(args.backbone, args.config_path)
    model = LinearSegProbe(backbone, model_class, args.readout_out_ch).to(device)
    for n, p in model.named_parameters():
        p.requires_grad_("readout" in n)
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.eval_only:
        m, a, _ = evaluate(model, val_loader, args.readout_out_ch, device)
        print(f"Eval: mIoU={m*100:.2f} pixAcc={a*100:.2f}"); return

    if args.img_size != 256:
        args.backbone = args.backbone + '_' + str(args.img_size)
    print("I will save as {}.".format(args.backbone))

    miou_hist, pix_hist = [], []
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        miou, pix, _ = evaluate(model, val_loader, args.readout_out_ch, device)
        print(f"[{e:03d}] loss={loss:.4f} mIoU={miou*100:.2f} pixAcc={pix*100:.2f}")
        ckpt = {'epoch': e, 'readout_state_dict': model.readout.state_dict(), 'miou': miou}
        torch.save(ckpt, os.path.join(args.save_dir, args.backbone + '.pt'))
        miou_hist.append(miou); pix_hist.append(pix)
        with open(os.path.join(args.save_dir, args.backbone + '.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['epoch', 'miou', 'pix']); [w.writerow(r) for r in zip_longest(range(1, len(miou_hist)+1), miou_hist, pix_hist)]

if __name__ == '__main__':
    main()
