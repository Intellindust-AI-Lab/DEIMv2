"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch

from ..core import register

__all__ = ['GradScaler']


def _default_amp_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch, 'xpu', None) is not None and torch.xpu.is_available():
        return 'xpu'
    return 'cpu'


@register()
class GradScaler(torch.amp.GradScaler):
    """Device-agnostic GradScaler, defaults to the available accelerator (cuda/xpu/cpu)."""

    def __init__(self, device: str = None, **kwargs):
        super().__init__(device=device or _default_amp_device(), **kwargs)
