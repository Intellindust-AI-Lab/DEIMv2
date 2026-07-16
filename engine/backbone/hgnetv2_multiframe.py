"""
HGNetv2 wrapper with configurable input channels for multiframe images.
"""

import os
import logging

import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from ..core import register


def adapt_input_conv_weight(weight: torch.Tensor, input_channels: int) -> torch.Tensor:
    if weight.ndim != 4 or weight.shape[1] == input_channels:
        return weight

    adapted = weight.new_zeros((weight.shape[0], input_channels, weight.shape[2], weight.shape[3]))
    copy_channels = min(weight.shape[1], input_channels)
    adapted[:, :copy_channels] = weight[:, :copy_channels]
    if input_channels > weight.shape[1]:
        mean_channel = weight.mean(dim=1, keepdim=True)
        adapted[:, weight.shape[1]:] = mean_channel.repeat(1, input_channels - weight.shape[1], 1, 1)
    return adapted


@register()
class HGNetv2MultiFrame(HGNetv2):
    def __init__(self, input_channels=4, **kwargs):
        self.input_channels = int(input_channels)
        pretrained = kwargs.pop("pretrained", True)
        local_model_dir = kwargs.get("local_model_dir", "weight/hgnetv2/")
        name = kwargs.get("name")

        super().__init__(pretrained=False, **kwargs)
        self._replace_stem_input_channels()

        if pretrained:
            self._load_pretrained(name, local_model_dir)

    def _replace_stem_input_channels(self):
        conv = self.stem.stem1.conv
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("HGNetv2MultiFrame expects stem.stem1.conv to be nn.Conv2d")
        if conv.in_channels == self.input_channels:
            return

        new_conv = nn.Conv2d(
            self.input_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(adapt_input_conv_weight(conv.weight, self.input_channels))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        self.stem.stem1.conv = new_conv

    def _load_pretrained(self, name, local_model_dir):
        if name in ["Atto", "Femto", "Pico"]:
            model_name = "B0"
        else:
            model_name = name
        model_path = os.path.join(local_model_dir, f"PPHGNetV2_{model_name}_stage1.pth")
        if not os.path.exists(model_path):
            logging.warning("HGNetv2 pretrained file is missing: %s", model_path)
            return

        state = torch.load(model_path, map_location="cpu")
        key = "stem.stem1.conv.weight"
        if key in state:
            state[key] = adapt_input_conv_weight(state[key], self.input_channels)

        missing_keys, unexpected_keys = self.load_state_dict(state, strict=False)
        print("Loaded multiframe HGNetv2 pretrained weights from local file.")
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
