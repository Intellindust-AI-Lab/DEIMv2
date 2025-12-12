"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)
Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
import warnings
from typing import Literal, Tuple


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        assert head_dim % 4 == 0, "Head dimension must be divisible by 4 for 2D RoPE"
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = head_dim
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else: # min
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)[None, :]
        if self.training and self.jitter_coords is not None:
            jitter = (torch.empty(2, **dd).uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))).exp()
            coords *= jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = (torch.empty(1, **dd).uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))).exp()
            coords *= rescale
        

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype if self.dtype is not None else torch.get_default_dtype()
        if self.base is not None:
            periods = self.base ** (2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2))
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = self.max_period * (base ** (exponents - 1))
        self.periods.data.copy_(periods)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """Applies RoPE to the input tensor."""
    return (x * cos) + (rotate_half(x) * sin)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std); u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1); tensor.erfinv_(); tensor.mul_(std * math.sqrt(2.)); tensor.add_(mean); tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_sincos=None, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope_sincos is not None:
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]

            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)

            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook: attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, rope_sincos=None, register_hook=False):
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos, register_hook=register_hook)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=192, depth=12,
        num_heads=3, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., return_layers=[3, 7, 11], embed_layer=PatchEmbed,
        norm_layer=None, act_layer=None
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self._model = nn.Module()
        self._model.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = patch_size
        self._model.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self._model.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(depth)
        ])

        self._model.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim, num_heads=num_heads, base=100.0,
            normalize_coords="separate", shift_coords=None, jitter_coords=None,
            rescale_coords=None, dtype=None, device=None,
        )
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self._model.cls_token, std=.02)
        self._model.rope_embed._init_weights()
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}
    
    def get_model(self):
        return self._model
    
    def feature_dim(self):
        return self.embed_dim

    def forward_features(self, x, register_hook=False):
        B, C, H, W = x.shape

        x_embed = self._model.patch_embed(x)
        cls_token = self._model.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((cls_token, x_embed), dim=1)

        patch_grid_h = H // self.patch_size
        patch_grid_w = W // self.patch_size
        rope_sincos = self._model.rope_embed(H=patch_grid_h, W=patch_grid_w)

        for blk in self._model.blocks:
            x = blk(x, rope_sincos=rope_sincos, register_hook=register_hook)
        x = x[:, 1:, :]
        return {'features': x.transpose(1, 2).reshape(-1, self.embed_dim, patch_grid_h, patch_grid_w)}
    
    def forward_pool(self, x):
        features = self.forward_features(x)['features']
        pooled_features = features.mean(dim=[2, 3])
        return {'pooled_features': pooled_features}

    def forward(self, x, register_hook=False):
        outs = []
        B, C, H, W = x.shape

        x_embed = self._model.patch_embed(x)
        cls_token = self._model.cls_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((cls_token, x_embed), dim=1)

        patch_grid_h = H // self.patch_size
        patch_grid_w = W // self.patch_size
        rope_sincos = self._model.rope_embed(H=patch_grid_h, W=patch_grid_w)

        for i, blk in enumerate(self._model.blocks):
            x = blk(x, rope_sincos=rope_sincos, register_hook=register_hook)
            if i in self.return_layers:
                outs.append((x[:, 1:], x[:, 0]))
        return outs
