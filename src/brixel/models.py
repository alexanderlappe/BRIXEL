# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial

from dinov3_main.dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, patch_size):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // patch_size, w // patch_size)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, token_numbers):
        x = self.fc1(x)
        x = self.dwconv(x, H, W, token_numbers)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, token_numbers):
        B, N, C = x.shape
        L1, L2, L3 = token_numbers
        x1 = x[:, 0 : L1, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, L1 : L1 + L2, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, L1 + L2 :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W, token_numbers):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W, token_numbers))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlockWithCls(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        with_cp=False,
    ):
        super().__init__()
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks, token_numbers):
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H_c,
            W=W_c,
            token_numbers=token_numbers
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H_c,
                    W=W_c,
                    token_numbers=token_numbers
                )
        return x, c, cls


def icnr_(weight: torch.Tensor, scale=2, init=nn.init.kaiming_normal_):
    """
    ICNR initialization for sub-pixel (PixelShuffle) layers.
    weight: [out_ch, in_ch, kH, kW] where out_ch = in_ch * scale^2
    """
    out_ch, in_ch, kH, kW = weight.shape
    r2 = scale * scale
    assert out_ch % r2 == 0, "out_channels must be in_ch * scale^2"
    new_out = out_ch // r2
    # make a kernel that when shuffled replicates a 'resize then conv' start
    subkernel = torch.empty(new_out, in_ch, kH, kW, device=weight.device, dtype=weight.dtype)
    init(subkernel)  # e.g., kaiming_normal
    subkernel = subkernel.repeat_interleave(r2, dim=0)
    with torch.no_grad():
        weight.copy_(subkernel)

class Up2xPixelShuffle(nn.Module):
    def __init__(self, C, post_dw=True, post_pw=True):
        super().__init__()
        self.expand = nn.Conv2d(C, C * 4, kernel_size=1, bias=True)  # r^2=4 for r=2
        self.shuffle = nn.PixelShuffle(2)
        post = []
        if post_dw:
            # depthwise 3x3: small local tidy-up, no channel mixing
            post += [nn.Conv2d(C, C, 3, padding=1, groups=C, bias=True)]
        if post_pw:
            # pointwise 1x1: mixes channels to kill any residual phase separation
            post += [nn.Conv2d(C, C, 1, bias=True)]
        self.post = nn.Sequential(*post) if post else nn.Identity()

        # --- init ---
        icnr_(self.expand.weight, scale=2, init=nn.init.kaiming_normal_)
        nn.init.zeros_(self.expand.bias)
        for m in self.post:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, out_size=None):
        y = self.shuffle(self.expand(x))   # 2x up, ICNR-initialized → no checkerboard at init
        y = self.post(y)
        if out_size is not None:
            # exact align to peer tensor (handles odd/rectangular shapes)
            y = F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)
        return y

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, patch_size=16):
        super().__init__()
        self.with_cp = with_cp
        self.patch_size = patch_size

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            # ---- NEW: make sizes match the ViT token grid exactly ----
            B, _, H, W = x.shape
            Th, Tw = H // self.patch_size, W // self.patch_size  # ViT tokens grid

            # Align each pyramid level to exact targets derived from (Th, Tw)
            # P4 (~1/16) -> tokens grid size (works for ps=16 exactly; ps=14 uses interpolation)
            c3 = F.interpolate(c3, size=(Th, Tw), mode="bilinear", align_corners=False)
            # P3 (~1/8)  -> 2x tokens grid
            c2 = F.interpolate(c2, size=(2 * Th, 2 * Tw), mode="bilinear", align_corners=False)
            # P5 (~1/32) -> half tokens grid (at least 1x1)
            c4 = F.interpolate(c4, size=(max(1, Th // 2), max(1, Tw // 2)),
                               mode="bilinear", align_corners=False)
            # (Optional) also align c1 (~1/4) to 4x tokens grid if you need it later
            c1 = F.interpolate(c1, size=(4 * Th, 4 * Tw), mode="bilinear", align_corners=False)
            # ----------------------------------------------------------

            bs, dim, _, _ = c1.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

class DINOv3_Adapter(nn.Module):
    def __init__(
        self,
        backbone,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=512,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=True,
    ):
        super(DINOv3_Adapter, self).__init__()
        self.backbone = backbone
        # Important: we freeze the backbone
        self.backbone.requires_grad_(False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        print("embed dim", embed_dim)
        print("interaction_indexes", self.interaction_indexes)
        print("patch_size", self.patch_size)

        block_fn = InteractionBlockWithCls
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False, patch_size=self.patch_size)
        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up = Up2xPixelShuffle(embed_dim, post_dw=True, post_pw=True)


        # Use GroupNorm instead of SyncBN for small-batch single GPU training
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=embed_dim)

        self.guide_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 48, kernel_size=1, bias=True),  # 32–64 is fine
            nn.ReLU(inplace=True),
        )


        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // self.patch_size, self.pretrain_size[1] // self.patch_size, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        # keep track of the number of tokens for each resolution
        token_numbers = [c2.shape[1], c3.shape[1], c4.shape[1]]

        c = torch.cat([c2, c3, c4], dim=1)

        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, return_class_token=True
                )

        x_for_shape, _ = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        outs = list()
        for i, layer in enumerate(self.interactions):
            x, cls = all_layers[i]
            _, c, _ = layer(
                x,
                c,
                cls,
                deform_inputs1,
                deform_inputs2,
                H_c,
                W_c,
                H_toks,
                W_toks,
                token_numbers
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        def checkerboard_phase_variance(y: torch.Tensor, reduce=True):
            """
            y: [B,C,H,W] (any dtype); returns a scalar score if reduce=True,
               else a dict with per-channel & per-phase stats.
            """
            # four 2x2 phases
            p00 = y[..., 0::2, 0::2].mean(dim=(-2, -1))
            p01 = y[..., 0::2, 1::2].mean(dim=(-2, -1))
            p10 = y[..., 1::2, 0::2].mean(dim=(-2, -1))
            p11 = y[..., 1::2, 1::2].mean(dim=(-2, -1))
            phases = torch.stack([p00, p01, p10, p11], dim=-1)  # [B,C,4]
            var_across_phases = phases.var(dim=-1, unbiased=False)  # [B,C]

            if reduce:
                # average over batch & channels for one score
                return var_across_phases.mean()
            else:
                return {
                    "per_channel_var": var_across_phases,  # [B,C]
                    "phase_means": phases  # [B,C,4]
                }


        c1 = c1.transpose(1, 2).view(bs, dim, H_c * 4, W_c * 4).contiguous()
        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = c1 + F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)

        guide = self.guide_proj(c1)   # used to predict the kernels for pixel-level resolution


        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        x1, x2, x3, x4 = outs

        x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
        x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)

        vit_feats = F.interpolate(x4, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)

        f1, f2, f3, f4 = f1, f2 + x2, f3 + x3, f4 + x4



        return {"1": f1, "2": f2, "3": f3, "4": f4, 'upsampled_vit_feats': vit_feats}


# ----- Norm helpers -----
class LayerNorm2d(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps)
    def forward(self, x):  # x: [B,C,H,W]
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)

def make_norm(kind: str, C: int):
    if kind.lower() in ("ln", "layernorm", "ln2d"):
        return LayerNorm2d(C)
    elif kind.lower() in ("gn", "groupnorm"):
        return nn.GroupNorm(num_groups=32, num_channels=C)
    else:
        raise ValueError("norm must be 'ln' or 'gn'")

# ----- Depthwise-separable residual block -----
class DWResBlock(nn.Module):
    def __init__(self, C, expansion=1.0, norm="ln"):
        super().__init__()
        mid = max(1, int(round(C * expansion)))
        self.block = nn.Sequential(
            make_norm(norm, C),
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=True),  # depthwise
            nn.GELU(),
            nn.Conv2d(C, mid, 1, bias=True),                    # pointwise expand
            nn.GELU(),
            nn.Conv2d(mid, C, 1, bias=True),                    # pointwise project
        )
    def forward(self, x):
        return x + self.block(x)


class ConvHead(nn.Module):
    def __init__(self, embed_dim, width=64, blocks=3, norm="ln", groups=4):
        super().__init__()
        C = width or max(embed_dim, 128)
        # C = embed_dim // 2
        self.align_in = nn.Conv2d(embed_dim, C, 1, bias=True)
        self.trunk = nn.Sequential(*[DWResBlock(C, 1.0, norm=norm) for _ in range(blocks)])
        self.mix = nn.Conv2d(C, C, 3, padding=1, groups=max(1,min(groups,C)), bias=True)
        self.head_norm = make_norm(norm, C)
        self.out_conv = nn.Conv2d(C, embed_dim, 1, bias=True)
    def forward(self, adaper_output):
        p2, vit_feats = adaper_output['1'], adaper_output['upsampled_vit_feats']
        x = self.align_in(p2)
        x = x + self.trunk(x)
        x = self.mix(x)
        x = self.head_norm(x)
        return self.out_conv(x) + vit_feats


class BRIXEL(nn.Module):
    def __init__(self, adapter):
        super().__init__()

        self.adapter = adapter   # DINOv3 model with ViT adapter
        self.head = ConvHead(adapter.backbone.embed_dim)

    def forward(self, x):
        adapter_out = self.adapter(x)
        return self.head(adapter_out)


BLOCK_INDICES = {
    'dinov3_vits16': [3, 6, 9, 11],
    'dinov3_vitb16': [3, 6, 9, 11],
    'dinov3_vitl16': [6, 12, 18, 23],
    'dinov3_vith16plus': [8, 16, 24, 31],
}

def build_model(load_name, dinov3_weight_path, adapter_weight_path, block_indices=None):
    repo_dir = 'dinov3_main'

    backbone = torch.hub.load(repo_dir, load_name, source='local',
                              weights=dinov3_weight_path)

    if block_indices is None:
        block_indices = BLOCK_INDICES[load_name]
    adapter = DINOv3_Adapter(backbone=backbone, interaction_indexes=block_indices)
    model = BRIXEL(adapter)
    if adapter_weight_path is not None:
        ckpt = torch.load(adapter_weight_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    return model


def dinov3_small_brixel(dinov3_weight_path, adapter_weight_path=None):
    return build_model('dinov3_vits16', dinov3_weight_path, adapter_weight_path, block_indices=[3, 6, 9, 11])

def dinov3_base_brixel(dinov3_weight_path, adapter_weight_path=None):
    return build_model('dinov3_vitb16', dinov3_weight_path, adapter_weight_path, block_indices=[3, 6, 9, 11])

def dinov3_large_brixel(dinov3_weight_path, adapter_weight_path=None):
    return build_model('dinov3_vitl16', dinov3_weight_path, adapter_weight_path, block_indices=[6, 12, 18, 23])

def dinov3_huge_plus_brixel(dinov3_weight_path, adapter_weight_path=None):
    return build_model('dinov3_vith16plus', dinov3_weight_path, adapter_weight_path, block_indices=[8, 16, 24, 31])