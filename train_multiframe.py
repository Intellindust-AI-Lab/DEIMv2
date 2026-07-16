"""
Training entrypoint for multiframe DEIMv2 experiments.

This keeps the upstream train.py untouched and imports the extra modules needed
for MultiFrameCocoDetection and HGNetv2MultiFrame registration.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import engine.data.dataset.multiframe_coco_dataset  # noqa: F401
import engine.backbone.hgnetv2_multiframe  # noqa: F401
from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.solver.multiframe_solver import MultiFrameDetSolver


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), "Only support from_scratch or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    if args.resume or args.tuning:
        if "HGNetv2MultiFrame" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2MultiFrame"]["pretrained"] = False

    TASKS["detection"] = MultiFrameDetSolver
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="")
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-dir", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    main(parser.parse_args())
