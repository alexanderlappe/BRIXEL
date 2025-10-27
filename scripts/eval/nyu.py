from datasets import load_dataset
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

ds = load_dataset(
    "parquet",
    data_files={
        "train": "/media/alex/DataSSD/alex/Dense/eval/NYU/data/train-*.parquet",
        "val":   "/media/alex/DataSSD/alex/Dense/eval/NYU/data/val-*.parquet",
    }
    # cache_dir="/data/hf_cache",           # where to store arrow caches (any writable folder)
)
ds = ds.with_format("torch")
# ds = load_dataset("tanganke/nyuv2",
#                   cache_dir="/me/media/alex/DataSSD/alex/Dense/eval/NYU//media/alex/DataSSD/alex/Dense/eval/NYU/dia/alex/DataSSD/alex/Dense/eval/NYU/",     # where to look/store
#                   local_files_only=True)

def to_model_batch(ex):
    x = (ex["image"] - IMAGENET_MEAN) / IMAGENET_STD       # (3,288,384)
    d = ex["depth"]                                        # (1,288,384) meters
    v = (d > 1e-3).float()
    return {"image": x, "depth": d.squeeze(0), "valid": v.squeeze(0)}

ds["train"].set_transform(to_model_batch)
ds["val"].set_transform(to_model_batch)

from torch.utils.data import DataLoader
train_loader = DataLoader(ds["train"], batch_size=16, shuffle=True,  num_workers=8, pin_memory=True, drop_last=True)
test_loader  = DataLoader(ds["val"],   batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

for split in ds:
    print(split, [f["filename"] for f in ds[split].cache_files])
pass



