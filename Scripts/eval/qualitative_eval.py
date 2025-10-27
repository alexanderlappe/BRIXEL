"""
Creates visualizations of dense feature maps as shown in the paper for the first N images in a directory.
"""

import json
import os
import glob
import argparse
from pathlib import Path

from PIL import Image

import torch
from torchvision import transforms

from models import dinov3_small_brixel, dinov3_base_brixel, dinov3_large_brixel, dinov3_huge_plus_brixel
from utils.Dataset import GetLargeView, DownsampleLargeView
from utls.visualize_features import show_img_and_feat

# ---------- Config ----------
DEFAULT_SIZE = 1920
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
# ----------------------------


def crop_short_side_and_resize(img: Image.Image, size: int):
    """
    Center-crop the image to a square using the shortest side,
    then resize to (size, size), and return a float tensor in [0,1].
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    img = img.crop((left, top, right, bottom))

    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:
        resample = Image.BICUBIC

    to_tensor = transforms.ToTensor()
    return to_tensor(img.resize((size, size), resample))


def list_images(root: str):
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {root} with extensions {VALID_EXTS}.")
    return files


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def tensor_to_pil_unnormalized(x: torch.Tensor):
    """If you ever need to view the transformed tensor again as an image.
       This unnormalizes ImageNet stats and converts back to PIL (clipped)."""
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std  = torch.tensor(IMAGENET_STD)[:, None, None]
    x = (x * std + mean).clamp(0, 1)
    return transforms.ToPILImage()(x)


def parse_args():
    ap = argparse.ArgumentParser(description="Batch DINOv3 feature map export (first N images).")

    parser.add_argument("--backbone_model_name", type=str, default='dinov3_vitb16',
                        help="Name of the backbone model.")
    parser.add_argument("--backbone_weight_path", type=str,
                        default='data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                        help="Path to pretrained backbone weights.")
    ap.add_argument("--adapter-weight-path", default='data/adapter_weights/dinov3_vitb16.pth',
                    help="Path to adapter weights.")

    # I/O & batching
    ap.add_argument("--data_dir", required=True,
                    help="Directory with images")
    ap.add_argument("--save_dir", required=True,
                    help="Base save directory")
    ap.add_argument("--max-images", type=int, default=100,
                    help="Process the first N images (default: 100)")

    # Preprocess
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help="Square size to evaluate at")

    # Model/config
    ap.add_argument("--model_name", default="dinov3_vitb16",
                    help="Model architecture.")
    ap.add_argument("--device", default="cuda", help="Device to run on, e.g. 'cuda' or 'cpu'")

    # Saved-image names
    ap.add_argument("--names", nargs=4,
                    default=["rgb.png", "high_res_target.png", "high_res_pred.png", "low_res_target.png"],
                    help="Filenames for the 4 output images per input")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.model_weight_path is not None:
        model_weight_path = Path(args.model_weight_path)
    else:
        model_weight_path = Path(args.saved_models_dir) / f"{args.model_name}.pth"

    # Load file list (sorted) and cap to first N
    files = list_images(args.data_dir)
    max_n = min(len(files), args.max_images if hasattr(args, "max-images") else args.max_images)  # guard
    files = files[:args.max_images]


    do_normalize = 'dino' in args.backbone_name

    model = build_model(backbone_model_name, dinov3_weight_path=args.backbone_weight_path,
                        adapter_weight_path=args.adapter_weight_path)

    high_res_transform = GetLargeView(patch_size=model.backbone.patch_size, do_normalize=do_normalize)
    low_res_transform = DownsampleLargeView(patch_size=model.adapter.backbone.patch_size)

    device = torch.device(args.device)
    model = model.to(device)
    model.adapter.backbone.to(device)
    model.eval()

    # Process first N images
    for idx, img_path in enumerate(files):
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping '{img_path}': {e}")
            continue

        x_tensor = crop_short_side_and_resize(pil_img, args.size).unsqueeze(0)  # [1, 3, H, W]

        x_high_res = high_res_transform(x_tensor).to(device)
        x_low_res = low_res_transform(x_high_res).to(device)

        with torch.no_grad():
            low_res_target = model.adapter.backbone.get_intermediate_layers(
                x_low_res, n=1, reshape=True
            )[0]
            high_res_pred = model(x_low_res)
            high_res_target = model.adapter.backbone.get_intermediate_layers(
                x_high_res, n=1, reshape=True
            )[0]

            rgb, high_res_target_img, high_res_pred_img, low_res_target_img = show_img_and_feat(
                x_high_res.squeeze(),
                high_res_target.squeeze(),
                high_res_pred.squeeze(),
                low_res_target=low_res_target.squeeze(),
                return_only=True,
                model_name=args.model_name
            )


            pil_list = [rgb, high_res_target_img, high_res_pred_img, low_res_target_img]
            if 'finetune' in args.model_weight_path:
                out_dir = ensure_dir(os.path.join(args.save_dir, args.model_name + '_finetuned', str(idx)))
            else:
                out_dir = ensure_dir(os.path.join(args.save_dir, args.model_name, str(idx)))
            for name, im in zip(args.names, pil_list):
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(os.path.join(out_dir, name))

            print(f"[{idx+1}/{len(files)}] Saved {len(pil_list)} images to: {out_dir}")
            print(f"Source: {img_path}")

    print("Done.")


if __name__ == "__main__":
    main()
