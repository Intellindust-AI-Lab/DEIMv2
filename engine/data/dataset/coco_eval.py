"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import io
import os

import faster_coco_eval.core.mask as mask_util
import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster

from ...core import register
from ...misc import dist_utils

__all__ = ['CocoEvaluator',]


def _ultrafast_tools():
    try:
        from ultrafast_pycocotools import COCO, COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Install the optional evaluator with pip install "ultrafast-pycocotools>=0.1.11,<0.2".'
        ) from exc
    return COCO, COCOeval


class _UltrafastCOCOeval(COCOeval_faster):
    """Native matching/accumulation with the existing faster-coco-eval summaries."""

    def __init__(self, coco_gt, iouType, **kwargs):
        super().__init__(None, iouType=iouType, **kwargs)
        self.cocoGt = coco_gt
        self.params.imgIds = sorted(coco_gt.getImgIds())
        self.params.catIds = sorted(coco_gt.catToImgs if iouType == "keypoints" else coco_gt.getCatIds())

    def evaluate(self):
        self.params.imgIds = list(np.unique(self.params.imgIds))
        if self.params.useCats:
            self.params.catIds = list(np.unique(self.params.catIds))
        self.params.maxDets = sorted(self.params.maxDets)
        _, evaluator = _ultrafast_tools()
        self._native = evaluator(
            self.cocoGt,
            self.cocoDt,
            self.params.iouType,
            lvis_style=self.lvis_style,
            lvis_protocol="coco",
            print_function=self.print_function,
        )
        # Do not assign Faster Params wholesale: its deprecated useSegm property
        # would reinterpret keypoints as bbox in the reference-compatible API.
        for name in (
            "imgIds",
            "catIds",
            "iouThrs",
            "recThrs",
            "maxDets",
            "areaRng",
            "areaRngLbl",
            "useCats",
            "kpt_oks_sigmas",
        ):
            if hasattr(self.params, name):
                setattr(self._native.params, name, copy.deepcopy(getattr(self.params, name)))
        self._native.evaluate()
        self._paramsEval = copy.deepcopy(self.params)
        if self.lvis_style:
            self.freq_groups = self._prepare_freq_group()

    def accumulate(self):
        self._native.accumulate()
        self.eval = dict(self._native.eval)
        self.eval["params"] = self.params


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, backend="faster_coco_eval"):
        if backend not in ("faster_coco_eval", "ultrafast"):
            raise ValueError(f"Unknown COCO backend {backend!r}")
        self.backend = backend
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self._eval_class = COCOeval_faster
        if backend == "ultrafast":
            coco_class, _ = _ultrafast_tools()
            coco_gt = coco_class(coco_gt.dataset, verbose=False)
            self._eval_class = _UltrafastCOCOeval
        self.coco_gt = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = self._eval_class(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self._predictions = {k: {} for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = self._eval_class(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self._predictions = {k: {} for k in self.iou_types}


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        if not img_ids:
            return
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            if self.backend == "ultrafast":
                per_image = {image_id: [] for image_id in img_ids}
                for result in results:
                    per_image[result['image_id']].append(result)
                for image_id in img_ids:
                    self._predictions[iou_type].setdefault(image_id, per_image[image_id])
                continue
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            if self.backend == "ultrafast":
                merged = {}
                for per_rank in dist_utils.all_gather(self._predictions[iou_type]):
                    for image_id, records in per_rank.items():
                        merged.setdefault(image_id, records)
                image_ids = sorted(merged)
                results = [item for image_id in image_ids for item in merged[image_id]]
                evaluator = self.coco_eval[iou_type]
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluator.cocoDt = self.coco_gt.loadRes(results)
                    evaluator.params.imgIds = image_ids
                    evaluator.evaluate()
                self._predictions[iou_type].clear()
                continue
            if not self.eval_imgs[iou_type]:
                params = self.coco_eval[iou_type].params
                self.eval_imgs[iou_type].append(np.empty(
                    (len(params.catIds) if params.useCats else 1, len(params.areaRng), 0), dtype=object
                ))
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        encoder = mask_util
        if self.backend == "ultrafast":
            from ultrafast_pycocotools import mask as encoder
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            if self.backend == "ultrafast":
                batch = masks[:, 0].detach().cpu().numpy().transpose(1, 2, 0)
                rles = encoder.encode(np.asfortranarray(batch, dtype=np.uint8)) if len(masks) else []
            else:
                rles = [
                    encoder.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                    for mask in masks
                ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2)
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx].ravel()

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
