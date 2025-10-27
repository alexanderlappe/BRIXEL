import math
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from typing import Tuple, Union
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

Img = Union[torch.Tensor, "PIL.Image.Image"]


class ImageDataset(Dataset):
    def __init__(self, root_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
        super().__init__()
        super().__init__()
        self.root_dir = Path(root_dir)

        # Recursively collect image files from subdirectories (exclude JSONs)
        self.filenames = sorted(
            [p for p in self.root_dir.rglob('*')
             if p.is_file() and p.suffix.lower() in exts]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        img = Image.open(path).convert("RGB")  # ensures 3 channels
        img = F.to_tensor(img)
        return img


class BatchRandomResizeCollate:
    def __init__(self, min_size=1024, max_size=2048):
        self.min_size, self.max_size = min_size, max_size   # to include the end of interval

    def __call__(self, batch):
        """
        Want to train on various sizes, but focus on 256 and 512 as those are common sizes for downstream tasks.
        """
        # batch: list of (img, label)

        z = torch.rand(1).item()

        if z < 0.5:
            H, W = self.min_size, self.min_size
        elif z < 1:
            H, W = self.max_size, self.max_size
        else:
            H = (random.randint(self.min_size, self.max_size) + 32) & ~63
            W = (random.randint(self.min_size, self.max_size) + 32) & ~63

        imgs, labels = [], []
        for img in batch:
            # img can be PIL or Tensor; F.resize handles both

            # First pad image if it is too small
            _, img_h, img_w = img.shape
            pad_h = max(H - img_h, 0)
            pad_w = max(W - img_w, 0)
            if not (pad_h == 0 and pad_w == 0):

                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                img = F.pad(img, (left, top, right, bottom), fill=0)

            # Now resize and crop to have desired size
            C, h, w = img.shape

            # scale to COVER target (both dims >= H, W), avoiding upscaling unless allowed
            scale = max(H / h, W / w)

            new_h = int(math.ceil(h * scale))
            new_w = int(math.ceil(w * scale))

            x = F.resize(img, size=[new_h, new_w])

            img_r = F.center_crop(x, [H, W])

            imgs.append(F.to_tensor(img_r) if not torch.is_tensor(img_r) else img_r)

        imgs = torch.stack(imgs, dim=0)
        return imgs


class GetLargeView:
    """
    Only applies Imagenet normalization
    """
    def __init__(self, patch_size: int, max_long: int = 1024, pad_fill: int = 0, do_normalize=True):
        assert patch_size > 0
        self.ps = int(patch_size)
        self.max_long = int(max_long)
        self.pad_fill = pad_fill

        self.normalize = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.do_normalize = do_normalize

    def __call__(self, img: Img) -> Img:
        if self.do_normalize:
            img = self.normalize(img)
        return img


class DownsampleLargeView:
    """
    'Small' view: given the LARGE view (already processed),
    resize it so the token grid is exactly 1/4 in each dimension (no extra crop).

    If large image size is (H, W), output size is (H/4, W/4) in pixels.
    """
    def __init__(self, patch_size: int):
        assert patch_size > 0
        self.ps = int(patch_size)

    def __call__(self, large_img: Img) -> Img:
        w, h = F.get_image_size(large_img)
        # H and W are multiples of 4*ps, so H//4 and W//4 are multiples of ps
        target_h, target_w = h // 4, w // 4
        return F.resize(
            large_img,
            (target_h, target_w),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )



if __name__ == '__main__':
    dataset = ImageDataset(r'C:\Users\Alex\Documents\Uni_Data\distillation\00000')
    pass