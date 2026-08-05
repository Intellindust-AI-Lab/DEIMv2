#!/usr/bin/env python3
"""Evaluate RGB and multiframe models on the phase_11_5_test3 subset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.v2.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import engine.backbone.hgnetv2_multiframe  # noqa: F401
import engine.data.dataset.multiframe_coco_dataset  # noqa: F401
from engine.core import YAMLConfig


TIME_RE = re.compile(r"^(?P<video>.+)_(?P<tenth>\d{4})$")


def normalize_person_annotation(src_path: Path, dst_path: Path, image_dir: Path) -> None:
    data = json.load(src_path.open())
    images_by_name = {}
    old_to_new = {}

    for image in data["images"]:
        name = Path(image["file_name"]).name
        if name not in images_by_name:
            new_id = len(images_by_name)
            item = dict(image)
            item["id"] = new_id
            item["file_name"] = name
            images_by_name[name] = item
        old_to_new[image["id"]] = images_by_name[name]["id"]

    person_category_ids = {
        cat["id"] for cat in data.get("categories", []) if cat.get("name") == "person"
    }
    if not person_category_ids:
        raise ValueError("No person category found in annotation")

    annotations = []
    seen = set()
    for ann in data["annotations"]:
        if ann["category_id"] not in person_category_ids:
            continue
        image_id = old_to_new[ann["image_id"]]
        bbox_key = tuple(round(float(v), 3) for v in ann["bbox"])
        key = (image_id, bbox_key)
        if key in seen:
            continue
        seen.add(key)
        item = dict(ann)
        item["id"] = len(annotations)
        item["image_id"] = image_id
        item["category_id"] = 0
        item["area"] = float(item["bbox"][2]) * float(item["bbox"][3])
        item["iscrowd"] = int(item.get("iscrowd", 0))
        annotations.append(item)

    normalized = {
        "images": sorted(images_by_name.values(), key=lambda item: item["id"]),
        "annotations": annotations,
        "categories": [{"id": 0, "name": "person"}],
        "info": data.get("info", {}),
    }

    missing = [
        image["file_name"]
        for image in normalized["images"]
        if not (image_dir / image["file_name"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, first: {missing[:3]}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(normalized, dst_path.open("w"), ensure_ascii=False, indent=2)
    print(f"normalized annotation: {dst_path}")
    print(f"images={len(normalized['images'])} person_annotations={len(annotations)}")


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    if "HGNetv2MultiFrame" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2MultiFrame"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().to(device)
            self.postprocessor = cfg.postprocessor.eval().to(device)

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    return Model(), cfg.yaml_cfg["eval_spatial_size"]


def parse_output(outputs):
    if isinstance(outputs, tuple) and len(outputs) == 3:
        labels, boxes, scores = outputs
        return labels[0], boxes[0], scores[0]
    output = outputs[0]
    return output["labels"], output["boxes"], output["scores"]


def load_rgb_tensor(path: Path) -> torch.Tensor:
    return F.pil_to_tensor(Image.open(path).convert("RGB")).float() / 255.0


def diff_channel(current_rgb: torch.Tensor, previous_rgb: torch.Tensor | None) -> torch.Tensor:
    if previous_rgb is None:
        return torch.zeros((1, current_rgb.shape[1], current_rgb.shape[2]), dtype=current_rgb.dtype)
    cur_gray = current_rgb.mean(dim=0, keepdim=True)
    prev_gray = previous_rgb.mean(dim=0, keepdim=True)
    return ((cur_gray - prev_gray) + 1.0).mul(0.5).clamp(0.0, 1.0)


def split_video_and_tenth(file_name: str) -> tuple[str, int]:
    match = TIME_RE.match(Path(file_name).stem)
    if match is None:
        raise ValueError(f"Could not parse frame time from {file_name}")
    return match.group("video"), int(match.group("tenth"))


def build_backgrounds(all_frame_dir: Path, sample_interval_tenth: int) -> dict[str, torch.Tensor]:
    grouped = defaultdict(list)
    for path in sorted(all_frame_dir.glob("*.jpg")):
        video, tenth = split_video_and_tenth(path.name)
        grouped[video].append((tenth, path))

    backgrounds = {}
    for video, items in grouped.items():
        sampled = []
        last_tenth = None
        for tenth, path in sorted(items):
            if last_tenth is None or tenth - last_tenth >= sample_interval_tenth:
                sampled.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
                last_tenth = tenth
        median = np.median(np.stack(sampled, axis=0), axis=0).astype(np.uint8)
        backgrounds[video] = F.pil_to_tensor(Image.fromarray(median, mode="RGB")).float() / 255.0
        print(f"background {video}: {len(sampled)} frames")
    return backgrounds


def make_multiframe_input(
    image_path: Path,
    all_frame_dir: Path,
    previous_offsets_tenth: tuple[int, ...],
    background: torch.Tensor | None = None,
):
    current_rgb = load_rgb_tensor(image_path)
    video, tenth = split_video_and_tenth(image_path.name)
    channels = [current_rgb]
    for offset_tenth in previous_offsets_tenth:
        prev_path = all_frame_dir / f"{video}_{tenth - offset_tenth:04d}.jpg"
        prev_rgb = load_rgb_tensor(prev_path) if prev_path.exists() else None
        channels.append(diff_channel(current_rgb, prev_rgb))
    if background is not None:
        channels.append(diff_channel(current_rgb, background))
    return torch.cat(channels, dim=0)


def mode_channels(mode: str) -> tuple[tuple[int, ...], bool]:
    if mode == "rgb":
        return (), False
    if mode == "ex001":
        return (5,), False
    if mode == "ex003":
        return (5, 10), False
    if mode == "ex004":
        return (), True
    if mode == "ex006":
        return (5, 10), True
    raise ValueError(f"Unsupported mode: {mode}")


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_iou_xywh(box_a, box_b) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def compute_fixed_confidence_metrics(
    annotation_path: Path,
    detection_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict:
    gt_data = json.load(annotation_path.open())
    detections = json.load(detection_path.open())

    gt_by_image = defaultdict(list)
    for ann in gt_data["annotations"]:
        if ann["category_id"] == 0 and int(ann.get("iscrowd", 0)) == 0:
            gt_by_image[int(ann["image_id"])].append(ann["bbox"])

    det_by_image = defaultdict(list)
    for det in detections:
        if det["category_id"] == 0 and float(det["score"]) >= confidence_threshold:
            det_by_image[int(det["image_id"])].append(det)

    tp = fp = fn = 0
    per_image = {}
    image_ids = sorted({int(image["id"]) for image in gt_data["images"]})
    for image_id in image_ids:
        gt_boxes = gt_by_image.get(image_id, [])
        image_dets = sorted(
            det_by_image.get(image_id, []),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        matched_gt = set()
        image_tp = image_fp = 0

        for det in image_dets:
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = box_iou_xywh(det["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                tp += 1
                image_tp += 1
            else:
                fp += 1
                image_fp += 1

        image_fn = len(gt_boxes) - len(matched_gt)
        fn += image_fn
        per_image[image_id] = {
            "gt": len(gt_boxes),
            "detections": len(image_dets),
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
        }

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_gt": tp + fn,
        "num_detections": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_image": per_image,
    }


def run_inference(args, annotation_path: Path) -> Path:
    device = torch.device(args.device)
    image_dir = Path(args.image_dir)
    all_frame_dir = Path(args.all_frame_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, size = load_model(args.config, args.resume, device)
    coco = COCO(str(annotation_path))
    image_infos = coco.loadImgs(coco.getImgIds())
    image_infos.sort(key=lambda item: item["id"])

    previous_offsets_tenth, use_background = mode_channels(args.mode)
    if args.mode == "rgb":
        transforms = T.Compose([T.Resize(size), T.ToTensor()])
        backgrounds = None
    else:
        transforms = None
        backgrounds = build_backgrounds(all_frame_dir, sample_interval_tenth=20) if use_background else {}

    detections = []
    with torch.no_grad():
        for image_info in tqdm(image_infos, desc=f"infer {args.name}"):
            image_path = image_dir / image_info["file_name"]
            if args.mode == "rgb":
                image = Image.open(image_path).convert("RGB")
                tensor = transforms(image)
            else:
                video, _ = split_video_and_tenth(image_info["file_name"])
                tensor = F.resize(
                    make_multiframe_input(
                        image_path,
                        all_frame_dir,
                        previous_offsets_tenth,
                        backgrounds.get(video),
                    ),
                    size,
                    antialias=True,
                )

            width = int(image_info["width"])
            height = int(image_info["height"])
            outputs = model(
                tensor.unsqueeze(0).to(device),
                torch.tensor([[width, height]], dtype=torch.float32, device=device),
            )
            labels, boxes, scores = parse_output(outputs)
            labels = labels.detach().cpu()
            boxes = boxes.detach().cpu()
            scores = scores.detach().cpu()

            keep = (labels == 0) & (scores >= args.min_score)
            for box, score in zip(boxes[keep].tolist(), scores[keep].tolist()):
                detections.append(
                    {
                        "image_id": int(image_info["id"]),
                        "category_id": 0,
                        "bbox": xyxy_to_xywh(box),
                        "score": float(score),
                    }
                )

    det_path = output_dir / f"{args.name}_person_detections.json"
    json.dump(detections, det_path.open("w"), ensure_ascii=False)
    print(f"detections: {det_path} ({len(detections)} boxes)")
    return det_path


def evaluate(annotation_path: Path, detection_path: Path, output_dir: Path, args) -> None:
    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(detection_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.catIds = [0]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    summary = {
        "name": args.name,
        "annotation": str(annotation_path),
        "detections": str(detection_path),
        "stats": {
            "AP": evaluator.stats[0],
            "AP50": evaluator.stats[1],
            "AP75": evaluator.stats[2],
            "AP_small": evaluator.stats[3],
            "AP_medium": evaluator.stats[4],
            "AP_large": evaluator.stats[5],
            "AR1": evaluator.stats[6],
            "AR10": evaluator.stats[7],
            "AR100": evaluator.stats[8],
            "AR_small": evaluator.stats[9],
            "AR_medium": evaluator.stats[10],
            "AR_large": evaluator.stats[11],
        },
        "fixed_confidence": compute_fixed_confidence_metrics(
            annotation_path,
            detection_path,
            confidence_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        ),
    }
    out_path = output_dir / f"{args.name}_person_coco_eval.json"
    json.dump(summary, out_path.open("w"), ensure_ascii=False, indent=2)
    print(f"summary: {out_path}")
    fixed = summary["fixed_confidence"]
    print(
        "fixed confidence "
        f"score>={fixed['confidence_threshold']} IoU>={fixed['iou_threshold']}: "
        f"precision={fixed['precision']:.4f} "
        f"recall={fixed['recall']:.4f} "
        f"f1={fixed['f1']:.4f} "
        f"tp={fixed['tp']} fp={fixed['fp']} fn={fixed['fn']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["rgb", "ex001", "ex003", "ex004", "ex006"], required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("--src-annotation", default="annotations/phase_11_5_test.json")
    parser.add_argument("--annotation", default="annotations/phase_11_5_test3_person_eval_coco.json")
    parser.add_argument("--image-dir", default="/datadrive/phase_11_5_test3/labelstudio_annotation_500")
    parser.add_argument(
        "--all-frame-dir",
        default="/datadrive/phase_11_5_test3/frames_05s_all/phase_11_5_test3_05s_frames",
    )
    parser.add_argument("--output-dir", default="outputs/phase_11_5_test3_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument("--conf-threshold", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    normalize_person_annotation(Path(args.src_annotation), annotation_path, Path(args.image_dir))
    detection_path = run_inference(args, annotation_path)
    evaluate(annotation_path, detection_path, Path(args.output_dir), args)


if __name__ == "__main__":
    main()
