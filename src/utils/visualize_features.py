import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

SIGLIP_MEAN = torch.tensor([0, 0, 0])
SIGLIP_STD = torch.tensor([1, 1, 1])

def _denorm_imagenet(img: torch.Tensor, model_name) -> torch.Tensor:
    """
    img: Tensor [C,H,W] in ImageNet norm
    returns: Tensor [C,H,W] in [0,1]
    """
    if img.ndim == 4 and img.size(0) == 1:
        img = img[0]
    assert img.ndim == 3 and img.size(0) in (1,3), "img must be [C,H,W]"
    if 'dinov3' in model_name:
        mean = IMAGENET_MEAN.to(img.device)[:, None, None]
        std = IMAGENET_STD.to(img.device)[:, None, None]
    elif 'siglip' in model_name:
        mean = SIGLIP_MEAN.to(img.device)[:, None, None]
        std = SIGLIP_STD.to(img.device)[:, None, None]
    x = img * std + mean
    return x.clamp(0, 1)

def _pca_rgb(feat_target: torch.Tensor, feat: torch.Tensor, low_res_target=None, out_hw=None, clip_percentile: float = 0.0) -> torch.Tensor:
    """
    feat: Tensor [C,H,W] (or [1,C,H,W])
    out_hw: (H,W) to upsample to; if None keep feature size
    clip_percentile: e.g. 1.0 to clip low/high 1% for contrast
    returns: Tensor [3,H,W] in [0,1]
    """
    if feat_target.ndim == 4 and feat_target.size(0) == 1:
        feat_target = feat_target[0]
    assert feat_target.ndim == 3, "feat_target must be [C,H,W]"
    C, H, W = feat_target.shape

    x, x_hat = feat_target.detach(), feat.detach()
    x_prime = low_res_target.detach() if low_res_target is not None else None
    # if out_hw is not None and (H, W) != tuple(out_hw):
    #     # interpolate expects [N,C,H,W]
    #     x = F.interpolate(x.unsqueeze(0), size=out_hw, mode="bilinear", align_corners=False)[0]
    #     C, H, W = x.shape

    # Flatten spatial -> [N,C]
    X = x.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    X = X - X.mean(dim=0, keepdim=True)
    X = X.detach().cpu()

    X_hat = x_hat.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
    X_hat = X_hat - X_hat.mean(dim=0, keepdim=True)
    X_hat = X_hat.detach().cpu()

    if x_prime is not None:   # also transform the additional lfeature target
        X_prime = x_prime.permute(1, 2, 0).reshape(-1, C)  # [H*W, C]
        X_prime = X_prime - X_prime.mean(dim=0, keepdim=True)
        X_prime = X_prime.detach().cpu()

    # Handle degenerate channel counts
    if C == 1:
        rgb = X.repeat(1, 3)  # [N,3]
    else:
        # SVD for PCA (torch-only)
        # X = U S Vh ; principal axes are columns of V = Vh.T
        # Take first 3 principal components
        Vh = torch.linalg.svd(X).Vh  # [C,C]
        comps = Vh[:3, :].T if Vh.shape[0] >= 3 else Vh.T[:, :3]  # [C,3]
        rgb_target = X @ comps  # [N,3]
        rgb = X_hat @ comps  # [N,3]

    # Reshape to [H,W,3]
    rgb_target = rgb_target.reshape(H, W, 3)
    rgb = rgb.reshape(H, W, 3)

    # Optional percentile clipping per channel for better contrast
    # if clip_percentile and clip_percentile > 0:
    #     qlo = torch.quantile(rgb, clip_percentile / 100.0, dim=(0,1), keepdim=True)
    #     qhi = torch.quantile(rgb, 1.0 - clip_percentile / 100.0, dim=(0,1), keepdim=True)
    #     rgb = rgb.clamp(qlo, qhi)

    # Min-max per channel to [0,1]
    mn = rgb_target.amin(dim=(0, 1), keepdim=True)
    mx = rgb_target.amax(dim=(0, 1), keepdim=True)
    rgb_target = (rgb_target - mn) / (mx - mn + 1e-8)

    mn = rgb.amin(dim=(0,1), keepdim=True)
    mx = rgb.amax(dim=(0,1), keepdim=True)
    rgb = (rgb - mn) / (mx - mn + 1e-8)

    if x_prime is not None:   # do same steps as for the other two
        rgb_low_res = X_prime @ comps  # [N,3]
        rgb_low_res = rgb_low_res.reshape(H // 4, W // 4, 3)
        mn = rgb_low_res.amin(dim=(0, 1), keepdim=True)
        mx = rgb_low_res.amax(dim=(0, 1), keepdim=True)
        rgb_low_res = (rgb_low_res - mn) / (mx - mn + 1e-8)
        rgb_low_res = rgb_low_res.permute(2, 0, 1).contiguous()
    else:
        rgb_low_res = None

    return rgb_target.permute(2, 0, 1).contiguous(), rgb.permute(2, 0, 1).contiguous(), rgb_low_res


def show_img_and_feat(img: torch.Tensor,
                      feat_targets,  # a list of tensors
                      feat: torch.Tensor,
                      low_res_target=None,
                      upsample_feat_to_img=True,
                      clip_percentile: float = 1.0,
                      return_only=False,
                      titles=("Image (denorm)", "Feature map (PCA RGB)"),
                      model_name='dinov3'):
    """
    Displays side-by-side with matplotlib.
    img: [C,H,W] (ImageNet-normalized)
    feat_target: [C',h,w] feature map (or with leading batch dim of 1)
    """
    # Denorm image
    img_denorm = _denorm_imagenet(img, model_name).detach().cpu()
    H, W = img_denorm.shape[-2:]

    # PCA -> RGB for feature map
    out_hw = (H, W) if upsample_feat_to_img else None
    feat_target_rgb, feat_rgb, low_res_rgb = [rgb.detach().cpu() if rgb is not None else None for rgb in _pca_rgb(feat_targets, feat, low_res_target=low_res_target, out_hw=out_hw, clip_percentile=clip_percentile)]

    # To PIL for consistent rendering
    pil_img = to_pil_image(img_denorm)
    pil_feat_target = to_pil_image(feat_target_rgb)
    pil_feat = to_pil_image(feat_rgb)
    if low_res_rgb is not None:
            low_res_rgb = to_pil_image(low_res_rgb)

    if not return_only:
        # Show
        pil_img.show()
        pil_feat_target.show()
        pil_feat.show()
        low_res_rgb.show()

    return pil_img, pil_feat_target, pil_feat, low_res_rgb
