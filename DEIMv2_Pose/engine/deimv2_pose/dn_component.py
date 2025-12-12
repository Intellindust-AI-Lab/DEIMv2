
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from .utils import inverse_sigmoid
import torch.nn.functional as F
import numpy as np

def get_sigmas(num_keypoints, device):
    if num_keypoints == 17:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
            1.07, .87, .87, .89, .89
        ], dtype=np.float32) / 10.0
    elif num_keypoints == 14:
        sigmas = np.array([
            .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
            .79, .79
        ]) / 10.0
    elif num_keypoints == 3:
        sigmas = np.array([
            1.07, 1.07, 0.67
        ]) / 10.0
    else:
        raise ValueError(f'Unsupported keypoints number {num_keypoints}')
    sigmas = np.concatenate([[0.1], sigmas]) # for the center of the human
    sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
    return sigmas[None, :, None]


def prepare_for_cdn(dn_args, training, num_queries, num_classes, num_keypoints, hidden_dim, label_enc, pose_enc, img_dim, device):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))
        dn_number = 1 if dn_number == 0 else dn_number

        unmask_bbox = unmask_label = torch.cat(known)
        
        # instance label denoise
        labels = torch.cat([t['labels'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)

        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_labels_expaned = known_labels.clone()

        known_labels_poses_expaned = torch.arange(num_keypoints, dtype=torch.long, device=device)
        known_labels_poses_expaned = known_labels_poses_expaned[None].repeat(len(known_labels), 1)
       
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

            # weights = torch.ones((len(chosen_indice), num_keypoints), device=p.device)
            # new_label_pose = torch.multinomial(weights, num_samples=num_keypoints, replacement=False)
            # known_labels_poses_expaned.scatter_(0, chosen_indice.unsqueeze(-1).repeat(1, num_keypoints), new_label_pose)  

        # keypoint noise
        boxes = torch.cat([t['boxes'] for t in targets])
        xy = (boxes[:, :2] + boxes[:, 2:]) / 2.
        keypoints = torch.cat([t['keypoints'] for t in targets])
        if 'area' in targets[0]:
            areas = torch.cat([t['area'] for t in targets])
        else:
            areas = boxes[:, 2] * boxes[:, 3] * 0.53
        poses = keypoints[:, 0:(num_keypoints * 2)]
        poses = torch.cat([xy, poses], dim=1)
        non_viz = keypoints[:, (num_keypoints * 2):] == 0
        non_viz = torch.cat((torch.ones_like(non_viz[:, 0:1]).bool(), non_viz), dim=1)
        vars = (2 * get_sigmas(num_keypoints, device)) ** 2


        known_poses = poses.repeat(2 * dn_number, 1).reshape(-1, num_keypoints+1, 2)
        known_areas = areas.repeat(2 * dn_number)[..., None, None] # normalized [0, 1]
        known_areas = known_areas * img_dim[0] * img_dim[1] # scaled [0, h*w]
        known_non_viz = non_viz.repeat(2 * dn_number, 1)

        single_pad = int(max(known_num))
        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(poses))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(poses) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(poses)

        eps = np.finfo('float32').eps
        rand_vector = torch.rand_like(known_poses)
        rand_vector = F.normalize(rand_vector, -1) # ||rand_vector|| = 1 
        rand_alpha = torch.zeros_like(known_poses[..., :1]).uniform_(-np.log(1), -np.log(0.5))
        rand_alpha[negative_idx] = rand_alpha[negative_idx].uniform_(-np.log(0.5), -np.log(0.1))
        rand_alpha *= 2 * (known_areas + eps) * vars ## This is distance **2 
        rand_alpha = torch.sqrt(rand_alpha) / max(img_dim) 
        # rand_alpha = rand_alpha ** 1.25 ## This is distance
        rand_alpha[known_non_viz] = 0.

        known_poses_expand = known_poses + rand_alpha * rand_vector

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        # input_label_pose_embed = pose_enc(known_labels_poses_expaned)
        input_label_pose_embed = pose_enc.weight[None].repeat(known_poses_expand.size(0), 1, 1)
        input_label_embed = torch.cat([input_label_embed.unsqueeze(1), input_label_pose_embed], dim=1)
        input_label_embed = input_label_embed.flatten(1)

        input_pose_embed = inverse_sigmoid(known_poses_expand)

        padding_label = torch.zeros(pad_size, hidden_dim * (num_keypoints + 1)).cuda()
        padding_pose = torch.zeros(pad_size, num_keypoints+1).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_pose = padding_pose[...,None].repeat(batch_size, 1, 1, 2)

        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_pose[(known_bid.long(), map_known_indice)] = input_pose_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True
        # import matplotlib.pyplot as plt
        # plt.imshow(~attn_mask.detach().cpu().numpy(), cmap='gray')
        # plt.show()

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label.unflatten(-1, (-1, hidden_dim)), input_query_pose, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_keypoints, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_keypoints = outputs_keypoints[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_keypoints = outputs_keypoints[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_keypoints': output_known_keypoints[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_keypoints)
        dn_meta['output_known_lbs_keypoints'] = out
    return outputs_class, outputs_keypoints
