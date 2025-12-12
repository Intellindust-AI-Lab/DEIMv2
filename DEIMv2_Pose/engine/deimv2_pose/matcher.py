# ------------------------------------------------------------------------
# Modified from :
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from ..core import register

@register()
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, focal_alpha=0.25,
                 cost_keypoints=1.0, cost_oks=0.01, num_body_points=17):
        super().__init__()
        self.cost_class = cost_class

        self.cost_keypoints = cost_keypoints
        self.cost_oks = cost_oks
        self.focal_alpha = focal_alpha
        self.num_body_points = num_body_points
        
        if num_body_points==17:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89
            ], dtype=np.float32) / 10.0

        elif num_body_points==14:
            self.sigmas = np.array([
                .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
                .79, .79
            ]) / 10.0
        else:
            raise NotImplementedError

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)  # [batch_size * num_queries, 51]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])  # nkp, 51
        tgt_area = torch.cat([v["area"] for v in targets])  # nkp, 51

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # compute the keypoint costs
        Z_pred = out_keypoints[:, 0:(self.num_body_points * 2)]
        Z_gt = tgt_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt: torch.Tensor = tgt_keypoints[:, (self.num_body_points * 2):]
        if Z_pred.sum() > 0:
            sigmas = Z_pred.new_tensor(self.sigmas)
            variances = (sigmas * 2) ** 2
            kpt_preds = Z_pred.reshape(-1, Z_pred.size(-1) // 2, 2)
            kpt_gts = Z_gt.reshape(-1, Z_gt.size(-1) // 2, 2)
            squared_distance = (kpt_preds[:, None, :, 0] - kpt_gts[None, :, :, 0]) ** 2 + \
                               (kpt_preds[:, None, :, 1] - kpt_gts[None, :, :, 1]) ** 2
            squared_distance0 = squared_distance / (tgt_area[:, None] * variances[None, :] * 2)
            squared_distance1 = torch.exp(-squared_distance0)
            squared_distance1 = squared_distance1 * V_gt
            oks = squared_distance1.sum(dim=-1) / (V_gt.sum(dim=-1) + 1e-6)
            oks = oks.clamp(min=1e-6)
            cost_oks = 1 - oks

            cost_keypoints = torch.abs(Z_pred[:, None, :] - Z_gt[None])  # npred, ngt, 34
            cost_keypoints = cost_keypoints * V_gt.repeat_interleave(2, dim=1)[None]
            cost_keypoints = cost_keypoints.sum(-1)
            C = self.cost_class * cost_class + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
            C = C.view(bs, num_queries, -1).cpu()

        else:
            cost_keypoints = cost_oks = 0
            C = self.cost_class * cost_class + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
            C = C.view(bs, num_queries, -1).cpu()

        # Final cost matrix
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        if tgt_ids.shape[0] > 0:
            cost_mean_dict = {
                'class': cost_class.mean(),
                "keypoints": cost_keypoints.mean()
            }
        else:
            # for the cases when no grounding truth boxes
            cost_mean_dict = {
                'class': torch.zeros_like(cost_class.mean()),
                'keypoints': torch.zeros_like(cost_keypoints.mean()),
            }

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                indices]#, cost_mean_dict

def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher'], "Unknown args.matcher_type: {}".format(
        args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, focal_alpha=args.focal_alpha, cost_keypoints=args.set_cost_keypoints, cost_oks=args.set_cost_oks, num_body_points=args.num_body_points)
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))