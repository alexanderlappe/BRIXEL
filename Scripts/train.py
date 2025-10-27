import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
from einops import rearrange
import pickle
import json

from utils.Dataset import ImageDataset, BatchRandomResizeCollate, GetLargeView, DownsampleLargeView
from utils.train_utils import full_loss, set_lr
from utils.visualize_features import show_img_and_feat
import argparse, os, yaml
import sys
import shutil

# make scripts runnable without install
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def train(
    backbone_model_name,
    backbone_weight_path,
    train_img_dir,
    val_img_dir,
    save_dir,
    n_epochs,
    patch_size,
    batch_size,
    train_size
):

    save_path = os.path.join(save_dir, backbone_model_name) + '.pth'

    model = build_model(backbone_model_name, dinov3_weight_path=backbone_weight_path)

    do_normalize = 'dino' in model_name
    high_res_transform = GetLargeView(patch_size=patch_size, do_normalize=do_normalize)
    low_res_transform = DownsampleLargeView(patch_size=patch_size)

    model = model.to('cuda')
    model.adapter.backbone.to('cuda')

    trainset = ImageDataset(train_img_dir)
    valset = ImageDataset(val_img_dir)

    resize_collate = BatchRandomResizeCollate(train_size, train_size)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                        collate_fn=resize_collate)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True,
                            collate_fn=resize_collate)

    params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params, lr=1e-4)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # enables TF32 on matmul

    best_val_loss = 1e9
    train_loss_history, val_loss_history = [], []
    print('Starting training...')
    for epoch in range(n_epochs):
        model, train_loss = train_one_epoch(model, trainloader, optimizer, epoch, n_epochs, high_res_transform, low_res_transform)

        val_loss = validate(model, val_loader, high_res_transform, low_res_transform)
        if val_loss < best_val_loss:  # save if better val error
            best_val_loss = val_loss
            state = model.state_dict()
            # Filter out backbone keys
            filtered_state = {k: v for k, v in state.items() if not k.startswith("adapter.backbone.")}
            # Save only the filtered weights
            torch.save(filtered_state, save_path)

        # Save losses
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        loss_dir = {'train_loss': train_loss, 'val_loss': val_loss}

        with open(os.path.join(save_dir, backbone_model_name), 'wb') as handle:
            pickle.dump(loss_dir, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def train_one_epoch(model, loader, optimizer, epoch, n_epochs, high_res_transform, low_res_transform):
    bf16_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()
    model.adapter.backbone.eval()
    running_loss = 0.0
    set_lr(optimizer, 1e-4 if epoch < 1 else 1e-3)  # set warm-up learning rate


    is_tty = sys.stdout.isatty()  # False under SLURM log files → use newline prints
    total_steps = len(loader)
    iterator = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", dynamic_ncols=True) if is_tty else loader

    for i, imgs in enumerate(iterator, 1):

        x_high_res = high_res_transform(imgs).to('cuda')
        x_low_res = low_res_transform(x_high_res).to('cuda')

        with bf16_ctx:
            high_res_pred = model(x_low_res)
            with torch.inference_mode():
                high_res_target = \
                model.adapter.backbone.get_intermediate_layers(x_high_res, n=1, return_class_token=False, reshape=True)[
                    0]

        # show_img_and_feat(x_high_res[0], high_res_target[0], high_res_pred[0])

        loss = full_loss(high_res_pred, high_res_target)
        loss.backward()

        # no grad accumulation during fixed-size training
        optimizer.step()
        optimizer.zero_grad()

        # update loss
        running_loss += loss.item()
        avg_loss = running_loss / i

        if is_tty:
            # update the tqdm bar line (no newline)
            iterator.set_postfix_str(f"loss={loss.item():.4f}")
        else:
            # emit NEWLINE output so `tail -f` shows progress
            if (i % 1000 == 0) or (i == total_steps):
                avg = running_loss / i
                print(f"[epoch {epoch}/{n_epochs}] step {i}/{total_steps}  "
                      f"loss={loss.item():.4f}  avg_loss={avg:.4f}", flush=True)
    return model, avg_loss


def validate(model, val_loader, high_res_transform, low_res_transform):
    model.eval()
    running_loss = 0.0
    for i, imgs in enumerate(val_loader, 1):
        x_high_res = high_res_transform(imgs).to('cuda')
        x_low_res = low_res_transform(x_high_res).to('cuda')

        with torch.no_grad():
            high_res_pred = model(x_low_res)
            high_res_target = \
                model.adapter.backbone.get_intermediate_layers(x_high_res, n=1, return_class_token=False,
                                                                   reshape=True)[
                    0]
            loss = full_loss(high_res_pred, high_res_target)

        # update loss
        running_loss += loss.item()
        avg_loss = running_loss / i

    return avg_loss

def get_args():
    parser = argparse.ArgumentParser(description="Training script for the model.")

    parser.add_argument("--backbone_model_name", type=str, default='dinov3_vitb16',
                        help="Name of the backbone model.")
    parser.add_argument("--backbone_weight_path", type=str, default='data/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
                        help="Path to pretrained backbone weights.")
    parser.add_argument("--train_img_dir", type=str, default='data/images',
                        help="Directory with training images.")
    parser.add_argument("--val_img_dir", type=str, default='data/images_val',
                        help="Directory with validation images.")
    parser.add_argument("--save_dir", type=str, default='data/saved_models',
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--n_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Size of image patches.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--train_size", type=float, default=1024,
                        help="Image size for the teacher(!) model.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(**vars(args))