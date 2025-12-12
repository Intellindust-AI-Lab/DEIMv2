"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .pose_solver import PoseSolver



from typing import Dict

TASKS :Dict[str, BaseSolver] = {
    'pose': PoseSolver,
}
