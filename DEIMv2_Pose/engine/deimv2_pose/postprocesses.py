import torch
from torch import nn
from torchvision.ops.boxes import nms
from ..core import register

__all__ = ['PostProcessor']

@register()
class PostProcessor(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=60, num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_body_points = num_body_points
        self.deploy_mode = False

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        num_select = self.num_select
        out_logits, out_keypoints= outputs['pred_logits'], outputs['pred_keypoints']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values

        # keypoints
        topk_keypoints = (topk_indexes.float() // out_logits.shape[2]).long()
        labels = topk_indexes % out_logits.shape[2]
        
        if self.deploy_mode:
            keypoints = torch.gather(out_keypoints, 1, topk_keypoints[..., None, None].expand(1, num_select, self.num_body_points, 2))
            keypoints = keypoints * target_sizes[:, None, None, :]
            return scores, labels, keypoints

        keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points*2))
        keypoints = keypoints * target_sizes.repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = keypoints.unflatten(-1, (-1, 2))
        keypoints_res = torch.cat(
            [keypoints_res, torch.ones_like(keypoints_res[..., 0:1])], 
            dim=-1).flatten(-2)

        results = [{'scores': s, 'labels': l, 'keypoints': k} for s, l, k in zip(scores, labels, keypoints_res)]
        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
