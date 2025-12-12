import json
import logging
import datetime
from pathlib import Path
from omegaconf import OmegaConf
from collections import Counter


import time
import atexit
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from ..misc.metrics import BestMetricHolder
from ..misc import dist_utils, stats
from ..core import instantiate
from ..core.workspace import create
from .pose_engine import train_one_epoch, evaluate
from ._solver import BaseSolver
from ..optim.lr_scheduler import FlatCosineLRScheduler


def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class PoseSolver(BaseSolver):
    def train(self,):
        self._setup()
        self.criterion = self.cfg.criterion
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        # Load datasets
        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # Enable self-defined flat-cosine scheduler for pose if requested
        self.self_lr_scheduler = False
        if hasattr(self.cfg, "lrsheduler") and self.cfg.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} (pose) ## ".format(self.cfg.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                self.cfg.lr_gamma,
                iter_per_epoch,
                total_epochs=self.cfg.epoches,
                warmup_iter=self.cfg.warmup_iter,
                flat_epochs=self.cfg.flat_epoch,
                no_aug_epochs=self.cfg.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        self.best_map_holder = BestMetricHolder(use_ema=self.cfg.use_ema)
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

        if hasattr(self.cfg, "pretrain"):
            self.pretrain(self.cfg.pretrain)

    def fit(self,):
        self.train()
        args = self.cfg
        n_parameters, model_stats = stats(self.cfg)
        
        print(model_stats)
        # print("-" * 42 + "Model Structrue" + "-" * 43)
        # print(self.model)
        
        # print("-" * 42 + "Check Shape of feats" + "-" * 43)
        # model = self.model.module if hasattr(self.model, 'module') else self.model
        # device = next(model.parameters()).device  
        # with torch.no_grad():
        #     feats = model.backbone(torch.randn(1, 3, 640, 640).to(device))
        #     for i, f in enumerate(feats):
        #         print(i, f.shape)

        print("-" * 42 + "Start training" + "-" * 43)
        
        
        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.postprocessor,
                self.evaluator,
                self.val_dataloader,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            epoch_start_time = time.time()

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_avail_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.cfg.train_dataloader.batch_size,
                args.grad_accum_steps,
                self.device, 
                epoch,
                args.clip_max_norm, 
                writer=self.writer, 
                warmup_scheduler=self.lr_warmup_scheduler, 
                ema=self.ema,
                args=args
                )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    # weights = {
                    #     'model': self.state_dict(),
                    #     'ema': self.ema.state_dict() if self.ema is not None else None,
                    #     'optimizer': self.optimizer.state_dict(),
                    #     'lr_scheduler': self.lr_scheduler.state_dict(),
                    #     'warmup_scheduler': self.lr_warmup_scheduler.state_dict() if self.lr_warmup_scheduler is not None else None,
                    #     'epoch': epoch,
                    #     'args': args,
                    # }
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)


            
            # eval ema model if exists
            if self.ema is not None:
                test_stats, coco_evaluator = evaluate(
                    self.ema.module, 
                    self.postprocessor, 
                    self.evaluator,
                    self.val_dataloader, 
                    self.device, 
                    self.writer
                )
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/ema_{k}_{i}'.format(k), v, epoch)
                eval_stats = test_stats
            else:
                # eval regular model
                test_stats, coco_evaluator = evaluate(
                    self.model, 
                    self.postprocessor, 
                    self.evaluator,
                    self.val_dataloader, 
                    self.device, 
                    self.writer
                )
                # Log regular model results
                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/regular_{k}_{i}'.format(k), v, epoch)
                eval_stats = test_stats
            
            for k in eval_stats:
                if k in best_stat:
                    best_stat['epoch'] = epoch if eval_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], eval_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = eval_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if eval_stats[k][0] > top1:
                            top1 = eval_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(eval_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in eval_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            
            # Add EMA results to log if available
            if self.ema is not None:
                log_stats.update({f'test_{k}': v for k, v in test_stats.items()})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "keypoints" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["keypoints"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
                module,
                self.postprocessor,
                self.evaluator,
                self.val_dataloader,
                self.device,
                multi_decoder_eval=False
            )

        # if self.output_dir:
        #     dist_utils.save_on_master(coco_evaluator.coco_eval["keypoints"].eval, self.output_dir / "eval.pth")

        return

    def pretrain(self, model_path):
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)

            if "ema" in state:
                print("USING EMA WEIGHTS!!!")
                pretrain_state_dict = state["ema"]["module"]
            else:
                pretrain_state_dict = state["model"]
            
            new_state_dict = {}
            for k in pretrain_state_dict:
                if ("decoder" in k):
                    continue
                new_state_dict[k] = state['model'][k]

            print(f"⚠️  Loading weights for the backbone and decoder from {model_path} ⚠️")
            missing_keys, unexpected_keys = self.model_without_ddp.load_state_dict(new_state_dict, strict=False)

            if len(unexpected_keys) > 0:
                print("Warning. The following RTGroupPose does not have the following parameters:")
                for k in unexpected_keys:
                    print(f"    - {k}")
            else:
                print(f'✅ Successfully initilized the backbone and encoder using {model_name} weights ✅')

            missing_keys, unexpected_keys = self.ema.module.load_state_dict(new_state_dict, strict=False)

        except (Exception, KeyboardInterrupt) as e:
            if safe_get_rank() == 0:
                print(f"{str(e)}")
                logging.error(
                    RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET
                )
                logging.error(
                    GREEN
                    + "Please check your network connection. Or download the model manually from "
                    + RESET
                    + f"{download_url}"
                    + GREEN
                    + " to "
                    + RESET
                    + f"{local_model_dir}."
                    + RESET
                )
            exit()
