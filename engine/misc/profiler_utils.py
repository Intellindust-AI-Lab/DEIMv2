"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

import torch
from torch.utils.flop_counter import FlopCounterMode


def _sdpa_math_backend():
    return torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.MATH]
    )

def stats(
    cfg,
    input_shape: Tuple = None,
    verbose: bool = True,
) -> Tuple[int, dict]:
    
    if input_shape is None:
        base_size = cfg.yaml_cfg["eval_spatial_size"]
        input_shape = (1, 3, *base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    model_for_info.eval()

    device = next(model_for_info.parameters()).device
    x = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        with _sdpa_math_backend():
            with FlopCounterMode(display=False) as flop_counter:
                model_for_info(x)

    flops = flop_counter.get_total_flops()

    params = sum(p.numel() for p in model_for_info.parameters())

    if verbose:
        title = "FLOPs and Params Info"
        W = 39 + len(title) + 39          
        lw, nw, uw = 18, 10, 6            
        rw = nw + 1 + uw                  
        mid = W - lw - rw                 

        print("-" * 39 + title + "-" * 39)
        print(f"{'Total Params:':>{lw}}{' ' * mid}{params / 1e6:<.2f} M")
        print(f"{'FLOPs:':>{lw}}{' ' * mid}{flops / 1e9:.2f} GFLOPs")



    del model_for_info

    return params, {
        "Model FLOPs (G)": f"{flops / 1e9:.4f}",
        "Model Params (M)": f"{params / 1e6:.4f}",
    }