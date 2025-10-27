
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an 80/20 train/val split list for PASCAL VOC-style .mat annotations (e.g., SBD).
Writes train.txt and val.txt with image IDs (no extension), which can be fed to the
training script in this folder.

Examples
--------
python voc_split_80_20.py \
  --ann-dir /path/to/annotations/cls \
  --img-dir /path/to/VOC2012/JPEGImages \
  --out-dir /path/to/splits \
  --val-ratio 0.2 --seed 42
"""
import argparse, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann-dir', type=str, default="/media/alex/DataHDD/Alex/DENSE/PASCAL/trainval/Annotations_Part/", help='Directory with .mat annotation files')
    ap.add_argument('--img-dir', type=str, required=False, default=None,
                    help='Optional JPEGImages dir to filter IDs to those with images present')
    ap.add_argument('--out-dir', type=str, default="/media/alex/DataHDD/Alex/DENSE/PASCAL/trainval/", help='Where to write train.txt / val.txt')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir) if args.img_dir else None

    ids = [p.stem for p in sorted(ann_dir.glob('*.mat'))]
    if not ids:
        raise FileNotFoundError(f'No .mat files found under {ann_dir}')

    if img_dir:
        ids = [i for i in ids if (img_dir / f'{i}.jpg').exists()]
        if not ids:
            raise RuntimeError('After filtering with JPEGImages, zero IDs remain. Check paths.')

    random.seed(args.seed); random.shuffle(ids)
    n_val = max(1, int(round(len(ids) * args.val_ratio)))
    val_ids = set(ids[:n_val]); train_ids = [i for i in ids if i not in val_ids]

    (out_dir / 'train.txt').write_text('\n'.join(train_ids) + '\n', encoding='utf-8')
    (out_dir / 'val.txt').write_text('\n'.join(sorted(val_ids)) + '\n', encoding='utf-8')

    print(f'Wrote {len(train_ids)} train and {len(val_ids)} val IDs to {out_dir}')

if __name__ == '__main__':
    main()
