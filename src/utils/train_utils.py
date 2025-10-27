import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def hann2d(H, W, device, dtype):
    h = torch.hann_window(H, periodic=False, device=device, dtype=dtype)
    w = torch.hann_window(W, periodic=False, device=device, dtype=dtype)
    return (h[:,None] * w[None,:])

def radial_profile(mag, bins=32):
    # mag: [B,C,H,W] amplitude; returns [B, bins] avg over channels & annuli
    B, C, H, W = mag.shape
    yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=mag.device),
                            torch.linspace(-1,1,W,device=mag.device), indexing="ij")
    rr = torch.sqrt(yy**2 + xx**2)  # radius in [0, ~sqrt(2)]
    rr = (rr / rr.max()).clamp(0,1)  # normalize to [0,1]
    edges = torch.linspace(0, 1, bins+1, device=mag.device)
    idx = torch.bucketize(rr.flatten(), edges) - 1  # [H*W] in [0..bins-1]
    idx = idx.clamp(0, bins-1)

    prof = []
    for b in range(B):
        m = mag[b].mean(0)  # [H,W] avg over channels
        vals = m.flatten()
        # bin means via scatter
        num = torch.zeros(bins, device=mag.device)
        den = torch.zeros(bins, device=mag.device)
        num.scatter_add_(0, idx, vals)
        den.scatter_add_(0, idx, torch.ones_like(vals))
        prof.append(num/(den+1e-6))
    return torch.stack(prof, 0)  # [B,bins]

def crispness_spectral_loss(S, T, bins=24, hi_start=0.3, eps=1e-8):
    """
    Match log-amplitude radial spectra on high freqs only (r >= hi_start).
    """
    B, C, H, W = S.shape
    win = hann2d(H, W, S.device, S.dtype)[None,None]  # [1,1,H,W]
    # window to reduce boundary leakage
    Sw = S * win; Tw = T * win

    # FFT amplitude
    As = torch.fft.fft2(Sw.float(), norm="ortho").abs()  # [B,C,H,W]
    At = torch.fft.fft2(Tw.float(), norm="ortho").abs()

    ps = radial_profile(As, bins=bins)  # [B,bins]
    pt = radial_profile(At, bins=bins)

    # select high-frequency bins
    start = int(hi_start * bins)
    ps_h = torch.log(ps[:, start:] + eps)
    pt_h = torch.log(pt[:, start:] + eps)

    return (ps_h - pt_h).pow(2).mean()


def l1_loss(pred, target):
    return ((pred - target).abs()).mean()


def sobel_xy(
        x):  # x: [B,C,H,W]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    kx = kx.repeat(x.size(1), 1, 1, 1);
    ky = ky.repeat(x.size(1), 1, 1, 1)
    gx = F.conv2d(x, kx, padding=0, groups=x.size(1))
    gy = F.conv2d(x, ky, padding=0, groups=x.size(1))
    return gx, gy


def fit_pca_lowrank(X, K=8, overshoot=16, niter=2):
    """
    X: [N, C] float32 (tokens, centered per-channel)
    Returns: mu [C], P [C, K]
    """
    mu = X.mean(dim=0)  # [C]
    Xc = X - mu  # center yourself; keep center=False below
    q = min(K + overshoot, X.shape[1])
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False, niter=niter)
    P = V[:, :K].contiguous()  # [C, K] top-K directions

    C, K = P.shape
    conv = nn.Conv2d(C, K, kernel_size=1, bias=True)
    with torch.no_grad():
        conv.weight.copy_(P.t().view(K, C, 1, 1))  # W = P^T
        conv.bias.copy_(-(P.t() @ mu).view(K))  # b = -P^T mu
    for p in conv.parameters():
        p.requires_grad = False  # freeze
    return conv.to(X.device)


def feature_edge_loss(pred,
                      target):  # both [B,C,H,W], channel-aligned
    ### Do a low-rank PCA of the teacher activations to get semantically meaningful edges
    with torch.no_grad():  # the components are found without grads. But reduction is applied with grads.
        teacher_tokens = rearrange(target, 'b c h w -> (b h w) c')
        shuffled_tokens = teacher_tokens[torch.randperm(teacher_tokens.size(0), device=teacher_tokens.device)]
        subsampled_tokens = shuffled_tokens[::3, :]
        pca_1x1_conv = fit_pca_lowrank(subsampled_tokens)
    pred_reduced, target_reduced = pca_1x1_conv(pred), pca_1x1_conv(target)
    pass

    gx_s, gy_s = sobel_xy(pred_reduced)
    gx_t, gy_t = sobel_xy(target_reduced)

    import matplotlib.pyplot as plt
    loss_vis = ((gx_s - gx_t).pow(2) + (gy_s - gy_t).pow(2)).sum(1).detach().cpu().numpy()
    return F.mse_loss(gx_s, gx_t) + F.mse_loss(gy_s, gy_t)

def full_loss(high_res_pred, high_res_target):
    l1 = l1_loss(high_res_pred, high_res_target)
    with torch.amp.autocast("cuda", enabled=False):   # SVD doesn't work with lower precision
        edge = feature_edge_loss(high_res_pred, high_res_target)
    spectral = crispness_spectral_loss(high_res_pred, high_res_target)
    loss = l1 + edge + 0.1 * spectral
    return loss


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr