# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch
from ..misc import logger as utils
from ..misc import dist_utils

# for visualization
import os
import cv2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import numpy as np
from collections import defaultdict
import json

GIGABYTE = 1024 ** 3

def train_one_epoch(self_lr_scheduler,
                    lr_scheduler,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    batch_size:int,
                    grad_accum_steps:int, 
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    writer=None,
                    warmup_scheduler=None,
                    ema=None,
                    args=None):
    scaler = torch.amp.GradScaler(str(device), enabled=True) # FIXME
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # Create meters for all parameter groups
    for pg_idx in range(len(optimizer.param_groups)):
        lr_name = f'lr_pg{pg_idx}' if pg_idx > 0 else 'lr'
        metric_logger.add_meter(lr_name, utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    
    sub_batch_size = batch_size // args.grad_accum_steps

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Batch size/GPU: ", batch_size)
    print("Total batch size: ", batch_size * dist_utils.get_world_size())

    optimizer.zero_grad()

    # === 检查 optimizer 加载的参数 ===
    all_params = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    opt_params = {id(p) for g in optimizer.param_groups for p in g['params']}

    missing_params = [name for pid, name in all_params.items() if pid not in opt_params]
    loaded_params = [name for pid, name in all_params.items() if pid in opt_params]

    print(f"\n[Optimizer Check] Covers {len(opt_params)} / {len(all_params)} parameters")

    # # 打印已加载参数
    # print("\n Optimizer loaded parameters:")
    # for name in loaded_params:
    #     print("  +", name)

    # 打印未加载参数
    if missing_params:
        print(f"\n  Optimizer missing {len(missing_params)} parameters:")
        for name in missing_params:
            print("  -", name)
    else:
        print("\n All parameters are properly covered by optimizer.")

    
    cur_iters = epoch * len(data_loader)
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # === Data 阶段 ===
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i

        for j in range(args.grad_accum_steps):
            start_idx = j * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples = samples[start_idx:final_idx]
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            # === Forward 阶段 ===
            with torch.amp.autocast(str(device), enabled=True):
                outputs = model(new_samples, new_targets)
            
            # === Loss 阶段 ===
            with torch.amp.autocast(str(device), enabled=False):
                loss_dict = criterion(outputs, new_targets)
                losses = sum(loss_dict.values())

            # === Backward 阶段 ===
            if args.use_amp:
                scaler.scale(losses).backward()
            else:
                losses.backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # === Optimizer 阶段 ===
        # amp backward function
        if args.use_amp:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
                    
        # === EMA 阶段 ===
        # ema
        if ema is not None:
            ema.update(model)
            
        # LR scheduling
        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if warmup_scheduler is not None:
                warmup_scheduler.step() 

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        # Update learning rates for all parameter groups
        for pg_idx, param_group in enumerate(optimizer.param_groups):
            lr_name = f'lr_pg{pg_idx}' if pg_idx > 0 else 'lr'
            metric_logger.update(**{lr_name: param_group["lr"]})     


        # === Log 阶段 ===
        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            free, total = torch.cuda.mem_get_info(device)
            mem_used_MB = (total - free) / GIGABYTE
            writer.add_scalar('Info/memory',  mem_used_MB, global_step)

        optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}




@torch.no_grad()
def evaluate(model, postprocessors, coco_evaluator, data_loader, device, writer=None, save_results=False, multi_decoder_eval=False):
    model.eval()
    if multi_decoder_eval:
        model.transformer.eval_aux = True
    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    res_json = [] 

    # --- 如果启用多层评估，创建多个 evaluator 副本 ---
    layer_evaluators = None
    if multi_decoder_eval and coco_evaluator is not None:
        from copy import deepcopy
        layer_evaluators = []  # 每层一个 evaluator
        for i in range(6):  # decoder 有 6 层，可根据模型改
            layer_evaluators.append(deepcopy(coco_evaluator))

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # ======== 额外评估每一层 decoder 输出 ========
        if multi_decoder_eval and 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                if i >= len(layer_evaluators):
                    break  # 防止层数超出预定义数量
                aux_results = postprocessors(aux_out, orig_target_sizes)
                aux_res = {target['image_id'].item(): output for target, output in zip(targets, aux_results)}
                layer_evaluators[i].update(aux_res)

        if save_results:
            for k, v in res.items():
                scores = v['scores']
                labels = v['labels']
                keypoints = v['keypoints']

                for s, l, kpt in zip(scores, labels, keypoints):
                    res_json.append(
                        {
                        "image_id": k,
                        "category_id": l.item(),
                        "keypoints": kpt.round(decimals=4).tolist(),
                        "score": s.item()
                        }
                        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # ======== 汇总 aux 层结果 ========
    layer_stats = []
    if multi_decoder_eval and layer_evaluators is not None:
        for i, evaluator_i in enumerate(layer_evaluators):
            if len(getattr(evaluator_i, "img_ids", [])) == 0:
                print(f"[Warning] Decoder layer {i} has no eval images, skip synchronization.")
                continue
            evaluator_i.synchronize_between_processes()
            evaluator_i.accumulate()
            evaluator_i.summarize()
            stats_i = evaluator_i.coco_eval['keypoints'].stats.tolist()
            layer_stats.append((i, stats_i))

    if save_results:
        return res_json

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        stats['coco_eval_keypoints'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()

    if multi_decoder_eval:
        stats['decoder_layers'] = layer_stats

    return stats, coco_evaluator