"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

# for register purpose
from . import optim
from . import data
from . import deimv2_pose

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)