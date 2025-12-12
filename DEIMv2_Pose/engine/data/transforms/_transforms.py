"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import transforms

import PIL
from PIL import Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
from omegaconf import ListConfig
import random
import numbers
import numpy as np
import os, uuid
import cv2
import csv
import copy
import matplotlib.pyplot as plt
import shutil
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
# RandomZoomOut = register()(T.RandomZoomOut)
# RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
# Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
# Normalize = register()(T.Normalize)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "keypoints"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "keypoints" in target:
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        keypoints = target["keypoints"]
        cropped_keypoints = keypoints[...,:2] - torch.as_tensor([j, i])[None, None]
        cropped_viz = keypoints[..., 2:]

        # keep keypoint if 0<=x<=w and 0<=y<=h else remove
        cropped_viz = torch.where(
            torch.logical_and( # condition to know if keypoint is inside the image
                torch.logical_and(0<=cropped_keypoints[..., 0].unsqueeze(-1), cropped_keypoints[..., 0].unsqueeze(-1)<=w), 
                torch.logical_and(0<=cropped_keypoints[..., 1].unsqueeze(-1), cropped_keypoints[..., 1].unsqueeze(-1)<=h)
                ),
            cropped_viz, # value if condition is True
            0 # value if condition is False
            )

        cropped_keypoints = torch.cat([cropped_keypoints, cropped_viz], dim=-1)
        cropped_keypoints = torch.where(cropped_keypoints[..., -1:]!=0, cropped_keypoints, 0)

        target["keypoints"] = cropped_keypoints

        keep = cropped_viz.sum(dim=(1, 2)) != 0

    # remove elements for which the no keypoint is on the image
    for field in fields:
        target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "keypoints" in target:
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        keypoints = target["keypoints"]
        # keypoints[:,:,0] = w - keypoints[:,:, 0]
        keypoints[:,:,0] = torch.where(keypoints[..., -1]!=0, w - keypoints[:,:, 0]-1, 0)
        for pair in flip_pairs:
            keypoints[:,pair[0], :], keypoints[:,pair[1], :] = keypoints[:,pair[1], :], keypoints[:,pair[0], :].clone()
        target["keypoints"] = keypoints

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple, ListConfig)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, padding)
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], padding)

    if "keypoints" in target:
        keypoints = target["keypoints"]
        padped_keypoints = keypoints.view(-1, 3)[:,:2] + torch.as_tensor(padding[:2])
        padped_keypoints = torch.cat([padped_keypoints, keypoints.view(-1, 3)[:,2].unsqueeze(1)], dim=1)
        padped_keypoints = torch.where(padped_keypoints[..., -1:]!=0, padped_keypoints, 0)
        target["keypoints"] = padped_keypoints.view(target["keypoints"].shape[0], -1, 3)

    if "boxes" in target:
        boxes = target["boxes"]
        padded_boxes = boxes + torch.as_tensor(padding)
        target["boxes"] = padded_boxes


    return padded_image, target

@register()
class RandomZoomOut(object):
    def __init__(self, p=0.5, side_range=[1, 2.5]):
        self.p = p
        self.side_range = side_range

    def __call__(self, img, target):
        if random.random() < self.p:
            ratio = float(np.random.uniform(self.side_range[0], self.side_range[1], 1))
            h, w = target['size']
            pad_w = int((ratio-1) * w)
            pad_h = int((ratio-1) * h)
            padding = [pad_w, pad_h, pad_w, pad_h]
            img, target = pad(img, target, padding)
        return img, target

@register()
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target        

@register()
class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.p = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img, target):
        if random.random() < self.p:
            fn_idx = torch.randperm(4)
            for fn_id in fn_idx:
                if fn_id == 0 and self.brightness is not None:
                    brightness = self.brightness
                    brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                    img = F.adjust_brightness(img, brightness_factor)

                if fn_id == 1 and self.contrast is not None:
                    contrast = self.contrast
                    contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                    img = F.adjust_contrast(img, contrast_factor)

                if fn_id == 2 and self.saturation is not None:
                    saturation = self.saturation
                    saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                    img = F.adjust_saturation(img, saturation_factor)

                if fn_id == 3 and self.hue is not None:
                    hue = self.hue
                    hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                    img = F.adjust_hue(img, hue_factor)
        return img, target

@register()
class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

@register()
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        if "area" in target:
            area = target["area"]
            area = area / (torch.tensor(w, dtype=torch.float32)*torch.tensor(h, dtype=torch.float32))
            target["area"] = area
        else:
            target["area"] = boxes[:, 2] * boxes[:, 3] * 0.53

        if "keypoints" in target:
            keypoints = target["keypoints"]  # (4, 17, 3) (num_person, num_keypoints, 3)
            keypoints = torch.where(keypoints[..., -1:]!=0, keypoints, 0)
            num_body_points = keypoints.size(1)
            V = keypoints[:, :, 2]  # visibility of the keypoints torch.Size([number of persons, 17])
            V[V == 2] = 1
            Z=keypoints[:, :, :2]
            Z = Z.contiguous().view(-1, 2 * num_body_points)
            Z = Z / torch.tensor([w, h] * num_body_points, dtype=torch.float32)
            all_keypoints = torch.cat([Z, V], dim=1)  # torch.Size([number of persons, 2+34+17])
            target["keypoints"] = all_keypoints
        return image, target


@register()
class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, img, target=None):
        return resize(img, target, self.size, self.max_size)


@register()
class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple, ListConfig))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

@register()
class MixUp(object):
    def __init__(self, p=0.5, max_cached_images=50, random_pop=True):
        self.p = p
        self.mixup_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
    
    def load_mixup(self, img, target):
        img = np.array(img)
        if img is None or img.size == 0:
            return None, None

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        img_h, img_w = img.shape[:2]
        base_img = img.copy()
        base_targets = copy.deepcopy(target)

        # 将当前图像加入缓存
        self.mixup_cache.append(dict(img=img.copy(), labels=target))
        if len(self.mixup_cache) > self.max_cached_images:
            idx = 0 if not self.random_pop else random.randint(0, len(self.mixup_cache) - 2)
            self.mixup_cache.pop(idx)
        if len(self.mixup_cache) < 2:
            return Image.fromarray(base_img), base_targets

        # 随机选取另一张图像用于 mixup
        mix_item = random.choice(self.mixup_cache[:-1])
        mix_img = mix_item['img']
        mix_targets = copy.deepcopy(mix_item['labels'])

        # Resize 确保尺寸一致，并同步缩放 mix_targets
        if mix_img.shape[:2] != base_img.shape[:2]:
            h_src, w_src = mix_img.shape[:2]
            h_dst, w_dst = base_img.shape[:2]

            # Resize the image
            mix_img = cv2.resize(mix_img, (w_dst, h_dst))
            ratio_x = w_dst / w_src
            ratio_y = h_dst / h_src

            # 缩放 mix_targets 中的 bbox 和 keypoints
            if 'boxes' in mix_targets:
                # 合并所有 bbox 为一个 tensor
                mix_targets['boxes'] = torch.stack([torch.tensor([x1 * ratio_x, y1 * ratio_y, x2 * ratio_x, y2 * ratio_y], dtype=torch.float32)
                                                    for x1, y1, x2, y2 in mix_targets['boxes']])

            if 'keypoints' in mix_targets:
                for i in range(len(mix_targets['keypoints'])):
                    # 获取当前目标的 keypoints，假设每个目标的 keypoints 是一个 17x3 的二维数组
                    keypoints = mix_targets['keypoints'][i]
                    new_kpts = []

                    # 遍历每个关键点 (17 个关键点，每个关键点由 x, y, v 组成)
                    for kpt in keypoints:
                        xk, yk, v = kpt
                        # 按比例缩放 x 和 y，保持 v 不变
                        new_kpts.append([xk * ratio_x, yk * ratio_y, v])

                    # 将修改后的 keypoints 赋值回去，并保持形状为 [17, 3]
                    mix_targets['keypoints'][i] = torch.tensor(new_kpts, dtype=torch.float32)

        # mixup 比例
        beta = round(random.uniform(0.45, 0.55), 6)
        mixed_img = (base_img * beta + mix_img * (1 - beta)).astype(np.uint8)

        keys = ['boxes', 'keypoints', 'labels', 'iscrowd', 'area']

        # 合并目标信息
        new_targets = {key: [] for key in keys}

        # 处理 base_targets
        for key in keys:
            if key in base_targets:
                new_targets[key].extend(base_targets[key])

        # 处理 mix_targets
        for key in keys:
            if key in mix_targets:
                new_targets[key].extend(mix_targets[key])

        if 'size' in base_targets:
            new_targets['size'] = base_targets['size']

        for key in new_targets:
            if key == 'boxes' or key == 'keypoints':  
                new_targets[key] = torch.stack(new_targets[key])  
            elif key == 'labels': 
                new_targets[key] = torch.tensor(new_targets[key], dtype=torch.int64)
            elif key == 'size':  
                new_targets[key] = torch.tensor(new_targets[key], dtype=torch.int64)
            else:
                new_targets[key] = torch.tensor(new_targets[key], dtype=torch.float32)

        new_targets['mixup'] = torch.tensor([beta] * len(base_targets['boxes']) + [1.0 - beta] * len(mix_targets['boxes']), dtype=torch.float32)
        kpts = new_targets['keypoints'] 
        vis = kpts[..., 2] 

        vis_ratio = torch.ones_like(vis, dtype=torch.float32)
        vis_ratio[vis == 2] = beta
        new_targets['vis_ratio'] = vis_ratio
        
        base_kpts_count = len(base_targets['keypoints'])
        mix_kpts_count = len(mix_targets['keypoints'])
        
        if base_targets['keypoints'] is not None and len(base_targets['keypoints']) > 0:
            num_body_points = base_targets['keypoints'][0].shape[0]
        else:
            num_body_points = 17
        
        if base_targets['keypoints'] is not None and len(base_targets['keypoints']) > 0:
            ref_dtype = base_targets['keypoints'][0].dtype
            ref_device = base_targets['keypoints'][0].device
        else:
            ref_dtype = torch.float32
            ref_device = torch.device('cpu')
            
        base_occlusion_ratio = torch.full((base_kpts_count, num_body_points), 1 - beta, dtype=ref_dtype, device=ref_device)
        mix_occlusion_ratio = torch.full((mix_kpts_count, num_body_points), beta, dtype=ref_dtype, device=ref_device)
        
        new_targets['occlusion_ratio'] = torch.cat([base_occlusion_ratio, mix_occlusion_ratio], dim=0)
        
        return Image.fromarray(mixed_img), new_targets



    def __call__(self, img, target):
        if random.random() < self.p:
            return self.load_mixup(img, target)
        return img, target

@register()
class CopyPaste(object):
    def __init__(self, p=1.0, beta_range=(0.45, 0.55), max_cached_images=50, 
                 experiment_type=None, partial_threshold=0.67, crop_prob=1):

        self.p = p
        self.beta_range = beta_range
        self.max_cached_images = max_cached_images
        self.experiment_type = experiment_type
        self.partial_threshold = partial_threshold
        self.crop_prob = crop_prob
        self.copypaste_cache = []

    def _get_random_patch(self, img, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return img.crop((x1, y1, x2, y2)), (x1, y1, x2 - x1, y2 - y1)
    
    def _analyze_keypoint_visibility(self, keypoints):

        visibilities = [kpt[2] for kpt in keypoints]  # 取每行的第3列
        # print(f"Debug - CopyPaste: visibilities={visibilities}")
        
        visible_count = sum(1 for v in visibilities if v > 0)
        total_count = len(visibilities)
        visibility_ratio = visible_count / total_count if total_count > 0 else 0.0
        
        result = {
            'visible_count': visible_count,
            'total_count': total_count,
            'visibility_ratio': visibility_ratio,
            'is_complete': visibility_ratio > self.partial_threshold,
            'is_partial': visibility_ratio <= self.partial_threshold
        }
        
        # print(f"Debug - CopyPaste: keypoint analysis - visible={visible_count}/{total_count}, ratio={visibility_ratio:.2f}, is_complete={result['is_complete']}, is_partial={result['is_partial']}, threshold={self.partial_threshold}")
        
        return result
    
    def _crop_keypoints(self, keypoints, crop_bbox, img_bbox):
        cx1, cy1, cx2, cy2 = crop_bbox
        ix1, iy1, ix2, iy2 = img_bbox
        
        cx1 = max(cx1, ix1)
        cy1 = max(cy1, iy1)
        cx2 = min(cx2, ix2)
        cy2 = min(cy2, iy2)
        
        cropped_keypoints = []
        for kpt in keypoints:
            x, y, v = kpt[0], kpt[1], kpt[2]
            
            if (cx1 <= x <= cx2 and cy1 <= y <= cy2 and v > 0):
                new_x = x - cx1
                new_y = y - cy1
                cropped_keypoints.append([new_x, new_y, v])
            else:
                cropped_keypoints.append([0, 0, 0])
        
        return cropped_keypoints

    def _adjust_keypoints(self, keypoints, src_bbox, dst_pos):
        adjusted = []
        sx, sy, sw, sh = src_bbox
        dx, dy = dst_pos
        for kpt in keypoints:
            x, y, v = kpt
            if v > 0 and sx <= x < sx + sw and sy <= y < sy + sh:
                new_x = x - sx + dx
                new_y = y - sy + dy
                adjusted.append([new_x, new_y, v])
            else:
                adjusted.append([0, 0, 0])
        return adjusted

    def _check_keypoint_coverage(self, keypoints, paste_region):
        dx, dy, dx_sw, dy_sh = paste_region
        x, y = keypoints[:, :, 0], keypoints[:, :, 1]
        in_region = (x >= dx) & (x <= dx_sw) & (y >= dy) & (y <= dy_sh)
        return in_region

    def _select_random_objects(self, target, num_objects=3):
        if 'boxes' not in target or len(target['boxes']) == 0:
            return []
        
        valid_objects = []
        for i, area in enumerate(target['area']):
            if area > 0:
                keypoints = target['keypoints'][i].tolist()
                vis_info = self._analyze_keypoint_visibility(keypoints)
                
                obj = {
                    'bbox': target['boxes'][i].tolist(),
                    'keypoints': keypoints,
                    'area': area.item(),
                    'labels': target['labels'][i].item(),
                    'iscrowd': target['iscrowd'][i].item(),
                    'visibility_info': vis_info
                }
                valid_objects.append((i, obj))
        
        if not valid_objects:
            return []
        
        # print(f"Debug - CopyPaste: experiment_type={self.experiment_type}, valid_objects_count={len(valid_objects)}")
        
        if self.experiment_type == 'complete':
            complete_objects = [(i, obj) for i, obj in valid_objects if obj['visibility_info']['is_complete']]
            if not complete_objects:
                return []
            selected_objects = random.sample(complete_objects, min(num_objects, len(complete_objects)))
            
        elif self.experiment_type == 'partial':
            selected_objects = random.sample(valid_objects, min(num_objects, len(valid_objects)))
            
        else: 
            selected_objects = random.sample(valid_objects, min(num_objects, len(valid_objects)))
            # print(f"Debug - CopyPaste: selected {len(selected_objects)} random objects")
        
        return [obj for _, obj in selected_objects]

    def __call__(self, img, target, dataset=None):
        if random.random() > self.p or 'boxes' not in target:
            return img, target

        current_cache_item = {
            'img': img.copy(),
            'target': copy.deepcopy(target),
            'objects': self._select_random_objects(target)
        }
        self.copypaste_cache.append(current_cache_item)
        if len(self.copypaste_cache) > self.max_cached_images:
            self.copypaste_cache.pop(0)

        augmented_img = img.copy()
        augmented_target = copy.deepcopy(target)

        all_objects_with_source = []
        for cache_item in self.copypaste_cache:
            for obj in cache_item['objects']:
                all_objects_with_source.append({
                    'obj': obj,
                    'source_img': cache_item['img']
                })

        if not all_objects_with_source:
            return img, target

        selected = random.sample(all_objects_with_source, min(3, len(all_objects_with_source)))

        for item in selected:
            obj = item['obj']
            source_img = item['source_img']
            
            should_crop = False
            if self.experiment_type == 'partial' and obj['visibility_info']['is_complete']:
                should_crop = True  
            
            if should_crop:
                original_bbox = obj['bbox']
                original_keypoints_before_crop = obj['keypoints'].copy()  # 保存crop前的原始关键点
                x1, y1, x2, y2 = original_bbox
                w, h = x2 - x1, y2 - y1
                
                crop_ratio = random.uniform(0.3, 0.8)  # 保留30%-80%的区域
                crop_w = int(w * crop_ratio)
                crop_h = int(h * crop_ratio)
                
                crop_x1 = x1 + random.randint(0, max(0, int(w - crop_w)))
                crop_y1 = y1 + random.randint(0, max(0, int(h - crop_h)))
                crop_x2 = crop_x1 + crop_w
                crop_y2 = crop_y1 + crop_h
                
                obj['bbox'] = [crop_x1, crop_y1, crop_x2, crop_y2]
                obj['keypoints'] = self._crop_keypoints(
                    obj['keypoints'], 
                    (crop_x1, crop_y1, crop_x2, crop_y2),
                    (x1, y1, x2, y2)
                )
                obj['area'] = crop_w * crop_h
                
                obj['visibility_info']['is_complete'] = False
                obj['visibility_info']['is_partial'] = True
                cropped_keypoints = obj['keypoints']
                visibilities = [kpt[2] for kpt in cropped_keypoints]
                visible_count = sum(1 for v in visibilities if v > 0)
                total_count = len(visibilities)
                obj['visibility_info']['visibility_ratio'] = visible_count / total_count if total_count > 0 else 0.0
                obj['visibility_info']['visible_count'] = visible_count
            # else:
                # print(f"Debug - CopyPaste: NOT cropping object. experiment_type={self.experiment_type}, is_complete={obj['visibility_info']['is_complete']}, random_val={random.random():.3f}, crop_prob={self.crop_prob}")

            patch, (sx, sy, sw, sh) = self._get_random_patch(source_img, obj['bbox'])
            if patch.size[0] == 0 or patch.size[1] == 0:
                continue

            dx = random.randint(0, max(0, img.width - sw))
            dy = random.randint(0, max(0, img.height - sh))
            beta = random.uniform(*self.beta_range)

            base_region = augmented_img.crop((dx, dy, dx + sw, dy + sh))
            patch_blended = Image.blend(base_region, patch, beta)
            augmented_img.paste(patch_blended, (dx, dy))

            n_old = augmented_target['keypoints'].shape[0] 
            num_kpts = augmented_target['keypoints'].shape[1]
            paste_region = (dx, dy, dx + sw, dy + sh)

            old_keypoints = augmented_target['keypoints'][:n_old]
            old_vis = old_keypoints[..., 2] > 0
            coverage_mask = self._check_keypoint_coverage(old_keypoints, paste_region) & old_vis

            if 'occlusion_ratio' in augmented_target:
                base_occlusion_ratio = augmented_target['occlusion_ratio'][:n_old].clone()
            else:
                ref_dtype = augmented_target['keypoints'].dtype
                ref_device = augmented_target['keypoints'].device
                base_occlusion_ratio = torch.zeros((n_old, num_kpts), dtype=ref_dtype, device=ref_device)

            if coverage_mask.any():
                prev = base_occlusion_ratio[coverage_mask]
                base_occlusion_ratio[coverage_mask] = 1.0 - (1.0 - prev) * (1.0 - beta)

            adjusted_kpts = self._adjust_keypoints(obj['keypoints'], (sx, sy, sw, sh), (dx, dy))

            augmented_target['boxes'] = torch.cat([
                augmented_target['boxes'],
                torch.tensor([[dx, dy, dx + sw, dy + sh]], dtype=torch.float32)
            ])
            augmented_target['keypoints'] = torch.cat([
                augmented_target['keypoints'],
                torch.tensor([adjusted_kpts], dtype=torch.float32)
            ])
            augmented_target['area'] = torch.cat([
                augmented_target['area'],
                torch.tensor([sw * sh], dtype=torch.float32)
            ])
            augmented_target['labels'] = torch.cat([
                augmented_target['labels'],
                torch.tensor([obj['labels']], dtype=torch.int64)
            ])
            augmented_target['iscrowd'] = torch.cat([
                augmented_target['iscrowd'],
                torch.tensor([obj['iscrowd']], dtype=torch.int64)
            ])
            

            adjusted_kpts_tensor = torch.tensor([adjusted_kpts], dtype=torch.float32)
            new_vis = adjusted_kpts_tensor[..., 2].squeeze(0) > 0
            ref_dtype = augmented_target['keypoints'].dtype
            ref_device = augmented_target['keypoints'].device
            new_occlusion_ratio = torch.zeros((1, num_kpts), dtype=ref_dtype, device=ref_device)
            new_occlusion_ratio[0, new_vis] = 1.0 - beta

            if 'occlusion_ratio' in augmented_target:
                tail = augmented_target['occlusion_ratio'][n_old:]  
                augmented_target['occlusion_ratio'] = torch.cat([
                    base_occlusion_ratio, tail, new_occlusion_ratio
                ], dim=0)
            else:
                augmented_target['occlusion_ratio'] = torch.cat([
                    base_occlusion_ratio, new_occlusion_ratio
                ], dim=0)

        return augmented_img, augmented_target

@register()
class MixUpCopyPaste(object):
    def __init__(self, mixup_prob=0.5, copypaste_prob=1, max_cached_images=50, random_pop=True,
                 experiment_type=None, partial_threshold=0.67, crop_prob=1):
        self.mixup_prob = mixup_prob
        self.copypaste_prob = copypaste_prob
        
        self.mixup = MixUp(p=mixup_prob, max_cached_images=max_cached_images, random_pop=random_pop)
        self.copypaste = CopyPaste(
            p=copypaste_prob, 
            max_cached_images=max_cached_images,
            experiment_type=experiment_type,
            partial_threshold=partial_threshold,
            crop_prob=crop_prob
        )
        
    def __call__(self, img, target):
        if random.random() < self.mixup_prob:
            return self.mixup(img, target)
        
        elif random.random() < self.copypaste_prob:
            return self.copypaste(img, target)
        
        return img, target

@register()
class PoseMosaic(object):
    def __init__(self, output_size=320, max_size=None, probability=1.0, 
        use_cache=False, max_cached_images=50, random_pop=True) -> None:
        super().__init__()
        self.resize = RandomResize(sizes=[output_size], max_size=max_size)
        self.probability = probability

        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            image, target = dataset.load_item(idx)
            image, target = self.resize(image, target)
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        
        to_pil = transforms.ToPILImage()
        pil_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                pil_images.append(to_pil(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        images = pil_images

        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)

        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        offsets_pose = torch.tensor([[0, 0, 0], [max_width, 0, 0], [0, max_height, 0], [max_width, max_height, 0]])
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            elif key == 'keypoints':
                values = [torch.where(target[key][..., -1:]!=0, target[key] + offsets_pose[i], 0) for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values

        return merged_image, merged_target

    def __call__(self, image, target, dataset):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target
        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        return mosaic_image, mosaic_target

