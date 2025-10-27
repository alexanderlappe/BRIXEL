#!/usr/bin/env python3
import os, random, argparse, json
from pathlib import Path

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def save_lines(lines, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ln in lines: f.write(ln + "\n")

def find_items(data_root: Path):
    """
    Yields relative depth paths for samples that have matching image+mask:
      <class>/wild_set/depth/<stem>.png   (always .png in NAVI)
    We verify image exists as .jpg or .png and mask exists as .png.
    """
    items_by_class = {}
    for cls_dir in sorted(data_root.iterdir()):
        wild_dir = cls_dir / "wild_set"
        if not wild_dir.is_dir():
            continue
        dep_dir = wild_dir / "depth"
        img_dir = wild_dir / "images"
        msk_dir = wild_dir / "masks"
        if not (dep_dir.is_dir() and img_dir.is_dir() and msk_dir.is_dir()):
            continue

        bucket = []
        for dp in sorted(dep_dir.glob("*.png")):
            stem = dp.stem
            ip_jpg = img_dir / f"{stem}.jpg"
            ip_png = img_dir / f"{stem}.png"
            mp = msk_dir / f"{stem}.png"
            if mp.exists() and (ip_jpg.exists() or ip_png.exists()):
                rel = dp.relative_to(data_root).as_posix()  # store depth path as the canonical key
                bucket.append(rel)
        if bucket:
            items_by_class[cls_dir.name] = bucket
    return items_by_class

def main():
    ap = argparse.ArgumentParser("Make NAVI wild_set 90/10 split (stratified by class)")
    ap.add_argument("--data-root", default="/media/alex/DataHDD/Alex/DENSE/navi_v1.0/")
    ap.add_argument("--out-dir", default="/media/alex/DataHDD/Alex/DENSE/navi_v1.0/", help="Where to write train.txt / val.txt (defaults to data-root)")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root

    items_by_class = find_items(data_root)

    random.seed(args.seed)
    train, val = [], []
    report = {"per_class": {}, "val_ratio": args.val_ratio, "seed": args.seed}

    for cls, rel_list in items_by_class.items():
        rel_list = rel_list[:]  # copy
        random.shuffle(rel_list)
        n = len(rel_list)
        nv = max(1, int(round(args.val_ratio * n)))
        val_cls = rel_list[:nv]
        trn_cls = rel_list[nv:]
        val += val_cls
        train += trn_cls
        report["per_class"][cls] = {"total": n, "train": len(trn_cls), "val": len(val_cls)}

    random.shuffle(train)
    random.shuffle(val)

    save_lines(train, out_dir / "navi_train.txt")   # one depth-relative path per line, e.g. Cat/wild_set/depth/000123.png  :contentReference[oaicite:2]{index=2}
    save_lines(val,   out_dir / "navi_val.txt")     # same format                                                       :contentReference[oaicite:3]{index=3}
    save_json({"counts": report, "out_dir": out_dir.as_posix()}, out_dir / "split_summary.json")  # :contentReference[oaicite:4]{index=4}

    print(f"[OK] Wrote {len(train)} train and {len(val)} val samples to {out_dir}")

if __name__ == "__main__":
    main()
