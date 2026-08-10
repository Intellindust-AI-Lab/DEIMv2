"""
Minimal DINOv2 Vision Transformer for use as a DEIMv2 backbone.

Modified from DINOv2 (https://github.com/facebookresearch/dinov2),
Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.

The module layout matches the official checkpoints (e.g. ``dinov2_vits14_pretrain.pth``),
so they can be loaded with ``load_state_dict(strict=True)``.
"""

import math

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

configs = {
    'dinov2_vits14': {
        'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'num_register_tokens': 0,
        'interpolate_antialias': False, 'interpolate_offset': 0.1,
    },
    'dinov2_vits14_reg': {
        'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'num_register_tokens': 4,
        'interpolate_antialias': True, 'interpolate_offset': 0.0,
    },
    'dinov2_vitb14': {
        'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'num_register_tokens': 0,
        'interpolate_antialias': False, 'interpolate_offset': 0.1,
    },
    'dinov2_vitb14_reg': {
        'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'num_register_tokens': 4,
        'interpolate_antialias': True, 'interpolate_offset': 0.0,
    },
}


class PatchEmbed(nn.Module):
    """Image to patch embedding via a strided convolution."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class Attention(nn.Module):
    """Multi-head self-attention with a fused qkv projection."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class LayerScale(nn.Module):
    """Per-channel learnable scaling (https://arxiv.org/abs/2103.17239)."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    """Standard transformer feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoV2VisionTransformer(nn.Module):
    """
    DINOv2 ViT exposing ``get_intermediate_layers`` with the interface expected by the STA adapter.

    :param name: One of the keys of ``configs`` (e.g. ``'dinov2_vits14_reg'``).
    """

    PRETRAIN_IMG_SIZE = 518
    PATCH_SIZE = 14

    def __init__(self, name: str):
        super().__init__()
        cfg = configs[name]
        embed_dim = cfg['embed_dim']
        self.embed_dim = self.num_features = embed_dim
        self.num_heads = cfg['num_heads']
        self.patch_size = self.PATCH_SIZE
        self.num_register_tokens = cfg['num_register_tokens']
        self.interpolate_antialias = cfg['interpolate_antialias']
        self.interpolate_offset = cfg['interpolate_offset']

        num_patches = (self.PRETRAIN_IMG_SIZE // self.PATCH_SIZE) ** 2
        self.patch_embed = PatchEmbed(self.PATCH_SIZE, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, self.num_heads) for _ in range(cfg['depth'])])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        #: Interpolated position embeddings, keyed by token grid, dtype and device. Only filled in
        #: eval mode, where ``pos_embed`` is frozen. Besides saving the resize on every forward, this
        #: turns the embedding into a constant during ONNX tracing, which is required because the
        #: antialiased bicubic resize has no ONNX symbolic in torch.
        self._pos_embed_cache: dict[tuple[int, int, torch.dtype, torch.device], torch.Tensor] = {}

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """
        Interpolate the pretraining position embeddings to the current token grid.

        :param x: Token tensor of shape ``[B, 1 + N, D]`` (cls token prepended).
        :param w: Input image width in pixels.
        :param h: Input image height in pixels.
        :return: Position embeddings of shape ``[1, 1 + N, D]``.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        # int() forces plain Python keys: under ONNX tracing the shape entries are traced tensors,
        # which would never hit the cache.
        cache_key = (int(w), int(h), previous_dtype, x.device)
        if not self.training and cache_key in self._pos_embed_cache:
            return self._pos_embed_cache[cache_key]
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge from the official implementation: use a slightly off scale factor.
            kwargs['scale_factor'] = (float(w0 + self.interpolate_offset) / M, float(h0 + self.interpolate_offset) / M)
        else:
            kwargs['size'] = (w0, h0)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode='bicubic',
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        out = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
        if not self.training:
            out = out.detach()
            self._pos_embed_cache[cache_key] = out
        return out

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify the image and prepend cls (and register) tokens with position embeddings added."""
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.num_register_tokens > 0:
            x = torch.cat((x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]), dim=1)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence[int]] = 1,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple:
        """
        Return patch tokens (and optionally cls tokens) from selected transformer blocks.

        :param x: Input images of shape ``[B, 3, H, W]`` with ``H``/``W`` divisible by 14.
        :param n: Block indices to take, or the number of last blocks if an int.
        :param return_class_token: If ``True``, return ``(patch_tokens, cls_token)`` tuples.
        :param norm: If ``True``, apply the final LayerNorm to the extracted layers.
        :return: One entry per requested block; patch tokens have shape ``[B, N, D]``.
        """
        x = self.prepare_tokens(x)
        blocks_to_take = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                outputs.append(x)
        assert len(outputs) == len(blocks_to_take), f'only {len(outputs)} / {len(blocks_to_take)} blocks found'
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final-layer cls token."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]
