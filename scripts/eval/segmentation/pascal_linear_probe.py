
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASCAL VOC (SBD .mat) Linear Probe (Segmentation) — PyTorch
-----------------------------------------------------------
Analogous to cityscapes_linear_probe.py, but for PASCAL VOC-style semantic segmentation
labels stored in .mat files (e.g., SBD/cls/*.mat with GTcls.Segmentation).

Directory layout (example):
  VOC2012/
    JPEGImages/*.jpg
  SBD/
    cls/*.mat
  splits/
    train.txt
    val.txt

Usage example:
  python pascal_voc_linear_probe.py \
    --img-dir /path/to/VOC2012/JPEGImages \
    --ann-dir /path/to/SBD/cls \
    --train-list /path/to/splits/train.txt \
    --val-list /path/to/splits/val.txt \
    --config-path /path/to/eval_config.json \
    --backbone dinov3_vits16 \
    --readout-out-ch 21 \
    --img-size 512 \
    --epochs 40 \
    --batch-size 16 \
    --amp

Notes
-----
- Classes: 21 (background=0 .. 20), ignore_index=255.
- This script reuses utility functions (set_seed, LinearSegProbe, etc.) from the ADE20K script,
  just like cityscapes_linear_probe.py does in your setup.
"""
import argparse
from PIL import ImageOps

import csv
import os
from itertools import zip_longest
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import scipy.io as sio  # for .mat loading
except Exception as e:
    raise ImportError('This script requires scipy. Please install with: pip install scipy') from e

# Reuse common utils exactly like your Cityscapes script
from modeling.dinov3.upsampling.eval.segmentation.ade20k_linear_probe import (
    set_seed, LinearSegProbe, compute_confusion_matrix, train_one_epoch, build_backbone, color_to_tensor
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def resize_pair(img: Image.Image, mask: Image.Image, size: Tuple[int, int]):
    return img.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST)

class PascalVOCDataset(Dataset):
    """
    Loads JPEG images and SBD/VOC .mat masks via a list of image IDs.
    Assumes each .mat contains GTcls.Segmentation with values {0..20, 255}.
    """
    def __init__(self, data_root, img_dir: str, ann_dir: str, id_list_file: str, img_size: int = 512):
        self.img_dir = Path(data_root) / Path(img_dir)
        self.ann_dir = Path(data_root) /Path(ann_dir)
        self.ids: List[str] = [x.strip() for x in (Path(data_root) / Path(id_list_file)).read_text().splitlines() if x.strip()]
        self.img_size = img_size
        if not self.ids:
            raise FileNotFoundError(f'No IDs in {id_list_file}')
        # quick existence check
        miss_img = [i for i in self.ids if not (self.img_dir / f'{i}.jpg').exists()]
        miss_ann = [i for i in self.ids if not (self.ann_dir / f'{i}.mat').exists()]
        if miss_img:
            raise FileNotFoundError(f'Missing JPEGs for IDs: {miss_img[:3]}... (total {len(miss_img)})')
        if miss_ann:
            raise FileNotFoundError(f'Missing .mat anns for IDs: {miss_ann[:3]}... (total {len(miss_ann)})')

    def __len__(self): return len(self.ids)

    def _load_mask(self, mat_path: Path, fallback_hw=None) -> np.ndarray:
        """
        Returns [H,W] np.int64 with class indices (0..20) and 255 as ignore (for unknown).
        Supports:
          - SBD/VOC:  mat['GTcls'].Segmentation
          - PASCAL-Part: mat['anno'].objects[*].mask + class name
        """
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        # --- Case 1: SBD/VOC style ---
        if 'GTcls' in mat and hasattr(mat['GTcls'], 'Segmentation'):
            seg = np.array(mat['GTcls'].Segmentation, dtype=np.int64)
            seg[seg == 255] = 255
            return seg

        # --- Case 2: PASCAL-Part style ('anno') ---
        if 'anno' not in mat:
            raise KeyError(f"Neither 'GTcls' nor 'anno' found in {mat_path}")

        anno = mat['anno']
        objects = getattr(anno, 'objects', None)
        if objects is None:
            raise KeyError(f"'objects' not found under 'anno' in {mat_path}")

        # normalize to list
        import numpy as _np
        objs = list(objects.flat) if isinstance(objects, _np.ndarray) else [objects]

        # VOC class mapping
        voc_classes = [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        voc_index = {n: i for i, n in enumerate(voc_classes)}

        def _to_str(x):
            if isinstance(x, str): return x
            if isinstance(x, bytes): return x.decode('utf-8', 'ignore')
            if isinstance(x, _np.ndarray) and x.size == 1: return str(x.flat[0])
            return str(x)

        def _get_name(o):
            for k in ('class', 'cls', 'name'):
                if hasattr(o, k):
                    return _to_str(getattr(o, k)).strip().lower()
                if isinstance(o, dict) and k in o:
                    return _to_str(o[k]).strip().lower()
            return ''

        def _get_mask(o):
            if hasattr(o, 'mask'):
                m = getattr(o, 'mask')
            elif isinstance(o, dict) and 'mask' in o:
                m = o['mask']
            else:
                raise KeyError("Object missing 'mask'")
            m = _np.array(m)
            if m.ndim == 3: m = m.squeeze()
            return (m > 0).astype(_np.uint8)

        # shape: try from first mask, else fallback passed from image
        H = W = None
        for o in objs:
            try:
                m = _get_mask(o)
                H, W = m.shape
                break
            except Exception:
                continue
        if H is None and fallback_hw is not None:
            H, W = fallback_hw
        if H is None:
            raise RuntimeError(f"Could not infer mask size for {mat_path}")

        seg = _np.zeros((H, W), dtype=_np.uint8)
        ignore = _np.zeros((H, W), dtype=bool)

        for o in objs:
            cname = _get_name(o).replace('tv/monitor', 'tvmonitor').replace('aero', 'aeroplane')
            idx = voc_index.get(cname, None)
            m = _get_mask(o).astype(bool)
            if idx is None:
                # unknown classes → mark as void=255
                ignore |= m
                continue
            seg[m] = idx

        seg = seg.astype(_np.int64)
        seg[ignore] = 255
        return seg

    def __getitem__(self, idx: int):
        _id = self.ids[idx]
        img = Image.open(self.img_dir / f'{_id}.jpg').convert('RGB')
        seg_np = self._load_mask(self.ann_dir / f'{_id}.mat', fallback_hw=(img.size[1], img.size[0]))
        mask = Image.fromarray(seg_np.astype(np.uint8), mode='L')

        # 1) resize by short side -> preserve aspect ratio
        w, h = img.size
        short = min(w, h)
        scale = self.img_size / short
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img, mask = resize_pair(img, mask, (new_w, new_h))

        # 2) center-crop or pad to exact (img_size, img_size)
        target = (self.img_size, self.img_size)

        def center_crop_or_pad(pil_img, fill):
            W, H = pil_img.size
            # crop if bigger
            left = max(0, (W - target[0]) // 2)
            top = max(0, (H - target[1]) // 2)
            right = min(W, left + target[0])
            bottom = min(H, top + target[1])
            cropped = pil_img.crop((left, top, right, bottom))
            # pad if smaller
            pad_w = target[0] - cropped.size[0]
            pad_h = target[1] - cropped.size[1]
            if pad_w > 0 or pad_h > 0:
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                cropped = ImageOps.expand(cropped, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)
            return cropped

        img = center_crop_or_pad(img, 0)  # RGB padding=0 is fine
        mask = center_crop_or_pad(mask, 255)  # VERY IMPORTANT: pad mask with ignore=255

        mask_np = np.array(mask, dtype=np.int64)
        return color_to_tensor(img), torch.from_numpy(mask_np)
@torch.no_grad()
def evaluate(model, loader, num_classes=21, device='cuda'):
    model.eval()
    hist = np.zeros((num_classes, num_classes), np.int64)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for p, l in zip(preds, labels):
            hist += compute_confusion_matrix(p, l, num_classes, ignore_index=255)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    return float(np.nanmean(iu)), float(np.diag(hist).sum() / (hist.sum() + 1e-10)), iu

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, required=False,
                   default='/media/alex/DataHDD/Alex/DENSE/PASCAL/',
                   help='Path to VOC root (containing JPEGImages/, ...).')

    # Directories for images/splits/masks
    p.add_argument('--img-dir', type=str, default='archive/VOC2012/JPEGImages',
                   help='Relative images directory under data-root')
    p.add_argument('--ann-dir', type=str, default="trainval/Annotations_Part", help='Directory with .mat annotation files (e.g., SBD/cls)')
    p.add_argument('--train-list', type=str, default="trainval/train.txt", help='train.txt with image IDs (no extension)')
    p.add_argument('--val-list', type=str, default="trainval/val.txt", help='val.txt with image IDs (no extension)')
    p.add_argument('--config-path', type=str, default="/home/alex/PycharmProjects/TheSecretOne/modeling/dinov3/upsampling/eval/local_eval_config.json", help='Path to model config JSON (same as other scripts)')
    p.add_argument('--backbone', type=str, default='dinov3_vitb16')
    p.add_argument('--readout-out-ch', type=int, default=21)
    p.add_argument('--save-dir', type=str, default="/media/alex/DataSSD/alex/Dense/eval/results/PASCAL")
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=1)
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

    train_ds = PascalVOCDataset(args.data_root, args.img_dir, args.ann_dir, args.train_list, args.img_size)
    val_ds   = PascalVOCDataset(args.data_root, args.img_dir, args.ann_dir, args.val_list, args.img_size)
    train_loader = DataLoader(train_ds, args.batch_size, True, num_workers=args.workers)
    val_loader   = DataLoader(val_ds, max(1, args.batch_size // 2), False, num_workers=args.workers)

    backbone, model_class = build_backbone(args.backbone, args.config_path)
    model = LinearSegProbe(backbone, model_class, args.readout_out_ch).to(device)
    for n, p in model.named_parameters():
        p.requires_grad_("readout" in n)

    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.eval_only:
        m, a, _ = evaluate(model, val_loader, args.readout_out_ch, device)
        print(f"Eval: mIoU={m*100:.2f} pixAcc={a*100:.2f}")
        return

    save_stub = args.backbone if args.img_size == 256 else f"{args.backbone}_{args.img_size}"
    print(f"I will save as {save_stub}.")

    miou_hist, pix_hist = [], []
    for e in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        miou, pix, _ = evaluate(model, val_loader, args.readout_out_ch, device)
        print(f"[{e:03d}] loss={loss:.4f} mIoU={miou*100:.2f} pixAcc={pix*100:.2f}")
        ckpt = {'epoch': e, 'readout_state_dict': model.readout.state_dict(), 'miou': miou}
        torch.save(ckpt, os.path.join(args.save_dir, save_stub + '.pt'))
        miou_hist.append(miou); pix_hist.append(pix)
        with open(os.path.join(args.save_dir, save_stub + '.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['epoch', 'miou', 'pix']); [w.writerow(r) for r in zip_longest(range(1, len(miou_hist)+1), miou_hist, pix_hist)]

if __name__ == '__main__':
    main()
