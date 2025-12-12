# ------------------------------------------------------------------------
# Modified from Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

from torch import nn
from ..core import register

__all__ = ['DEIMv2Pose', ]

@register()
class DEIMv2Pose(nn.Module):
    __inject__ = ['backbone', 'encoder', 'transformer', ]

    def __init__(
        self, 
        backbone, 
        encoder, 
        transformer
        ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.transformer = transformer

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        feats = self.backbone(samples)
        feats = self.encoder(feats)
        out = self.transformer(feats, targets, samples if self.training else None)
        return out

