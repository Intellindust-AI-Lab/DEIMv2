"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register


__all__ = ['AdamW', 'MuonAdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


@register()
class MuonAdamW(optim.Optimizer):
    """Use Muon for hidden matrices and AdamW for every other parameter."""

    def __init__(self, params, lr=5e-4, betas=(0.9, 0.999), weight_decay=1.25e-4):
        super().__init__(params, defaults={})

        muon_groups = []
        adamw_groups = []
        for group in self.param_groups:
            named_params = list(zip(group['param_names'], group['params'], strict=True))
            muon_params = [item for item in named_params if _use_muon(*item)]
            adamw_params = [item for item in named_params if not _use_muon(*item)]
            options = {
                'lr': group.get('lr', lr),
                'weight_decay': group.get('weight_decay', weight_decay),
            }
            if muon_params:
                muon_groups.append({'params': muon_params, **options})
            if adamw_params:
                adamw_groups.append({'params': adamw_params, **options})

        if not muon_groups or not adamw_groups:
            raise ValueError('MuonAdamW requires both Muon and AdamW parameters')

        self.muon = optim.Muon(muon_groups, adjust_lr_fn='match_rms_adamw')
        self.adamw = optim.AdamW(adamw_groups, betas=betas)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon.state_dict(),
            'adamw': self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def _use_muon(name, param):
    adamw_names = ('embed', 'score_head', 'bbox_head', 'query_pos_head')
    return param.ndim == 2 and not any(part in name for part in adamw_names)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)
