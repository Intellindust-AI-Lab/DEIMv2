"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from pathlib import Path

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def init_clearml(args: argparse.Namespace) -> None:
    """Create a ClearML task on the main process.

    ClearML auto-instruments the TensorBoard ``SummaryWriter``, so all scalars logged during
    training are mirrored to the ClearML server without further code changes. Checkpoint
    auto-upload is disabled because the solver saves ``last.pth`` every epoch.

    :param args: Parsed command-line arguments; uses ``clearml_project``, ``clearml_task``
        and ``config``.
    """
    if int(os.environ.get('RANK', 0)) != 0:
        return

    from clearml import Task

    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task or Path(args.config).stem,
        auto_connect_frameworks={'pytorch': False},
        output_uri=False,
    )
    task.connect_configuration(args.config, name='yaml_config')


def main(args, ) -> None:
    """main
    """
    if not args.avoid_clearml:
        init_clearml(args)

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', 'avoid_clearml', 'clearml_project', 'clearml_task'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # experiment tracking (ClearML is on by default)
    parser.add_argument('--avoid-clearml', action='store_true', default=False, help='do not log the run to ClearML')
    parser.add_argument('--clearml-project', type=str, default='DEIMv2', help='ClearML project name')
    parser.add_argument('--clearml-task', type=str, help='ClearML task name (defaults to config file name)')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
