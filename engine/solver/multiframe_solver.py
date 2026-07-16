"""
Detection solver that adapts 3-channel checkpoints to multiframe input stems.
"""

import copy

import torch

import engine.solver.det_solver as det_solver_module
from .det_solver import DetSolver
from ..backbone.hgnetv2_multiframe import adapt_input_conv_weight
from ..misc import dist_utils


def multiframe_stats(cfg):
    try:
        from calflops import calculate_flops

        base_size = cfg.train_dataloader.collate_fn.base_size
        input_channels = int(cfg.yaml_cfg.get("HGNetv2MultiFrame", {}).get("input_channels", 4))
        input_shape = (1, input_channels, base_size, base_size)
        model_for_info = copy.deepcopy(cfg.model).deploy()
        flops, macs, _ = calculate_flops(
            model=model_for_info,
            input_shape=input_shape,
            output_as_string=True,
            output_precision=4,
            print_detailed=False,
        )
        params = sum(p.numel() for p in model_for_info.parameters())
        del model_for_info
        return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
    except Exception as exc:
        params = sum(p.numel() for p in cfg.model.parameters())
        return params, {"Model FLOPs: skipped for multiframe input (%s)   Params:%s" % (exc, params)}


class MultiFrameDetSolver(DetSolver):
    def fit(self):
        original_stats = det_solver_module.stats
        det_solver_module.stats = multiframe_stats
        try:
            return super().fit()
        finally:
            det_solver_module.stats = original_stats

    def _adapt_input_conv_state(self, current_state, pretrained_state):
        adjusted = dict(pretrained_state)
        for key, tensor in list(adjusted.items()):
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key not in current_state:
                continue
            cur_tensor = current_state[clean_key]
            if (
                tensor.ndim == 4
                and cur_tensor.ndim == 4
                and tensor.shape[0] == cur_tensor.shape[0]
                and tensor.shape[2:] == cur_tensor.shape[2:]
                and tensor.shape[1] != cur_tensor.shape[1]
            ):
                adjusted[key] = adapt_input_conv_weight(tensor, cur_tensor.shape[1])
        return adjusted

    def load_tuning_state(self, path: str):
        if path.startswith("http"):
            state = torch.hub.load_state_dict_from_url(path, map_location="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        module = dist_utils.de_parallel(self.model)
        pretrain_state_dict = state["ema"]["module"] if "ema" in state else state["model"]
        pretrain_state_dict = self._adapt_input_conv_state(module.state_dict(), pretrain_state_dict)

        try:
            adjusted_state_dict = self._adjust_head_parameters(module.state_dict(), pretrain_state_dict)
            stat, infos = self._matched_state(module.state_dict(), adjusted_state_dict)
        except Exception:
            stat, infos = self._matched_state(module.state_dict(), pretrain_state_dict)

        module.load_state_dict(stat, strict=False)
        print(f"Load multiframe model.state_dict, {infos}")
