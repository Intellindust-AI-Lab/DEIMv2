"""
Multiframe video inference for EX001/EX003.

Input tensor is RGB + one grayscale frame-difference channel per offset:
current RGB, plus gray(current) - gray(previous frame at --frame-offsets seconds).
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import engine.backbone.hgnetv2_multiframe  # noqa: F401
import engine.data.dataset.multiframe_coco_dataset  # noqa: F401
from engine.core import YAMLConfig


LABEL_MAP = {
    0: "person",
    1: "head",
}

COLORS = {
    0: (255, 80, 80),
    1: (80, 180, 255),
}


def frame_to_rgb_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0


def make_diff_channel(current_rgb, previous_rgb):
    if previous_rgb is None:
        return torch.zeros((1, current_rgb.shape[1], current_rgb.shape[2]), dtype=current_rgb.dtype)

        cur_gray = current_rgb.mean(dim=0, keepdim=True)
    prev_gray = previous_rgb.mean(dim=0, keepdim=True)
    diff = ((cur_gray - prev_gray) + 1.0) * 0.5
    return diff.clamp(0.0, 1.0)


def make_multiframe_input(current_rgb, previous_rgbs, size):
    channels = [current_rgb]
    channels.extend(make_diff_channel(current_rgb, previous_rgb) for previous_rgb in previous_rgbs)
    image = torch.cat(channels, dim=0)
    return F.resize(image, size, antialias=True)


def draw_on_frame(frame_bgr, labels, boxes, scores, threshold):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    keep = scores > threshold
    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    for label, box, score in zip(labels, boxes, scores):
        label_id = int(label.item())
        color = COLORS.get(label_id, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        text = f"{LABEL_MAP.get(label_id, str(label_id))} {float(score):.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_box = draw.textbbox((0, 0), text, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - th - 3)), text, fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def load_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
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


def process_video(model, input_path, output_path, size, threshold, frame_offsets, device, max_frames=None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    offset_frames = [max(1, int(round(float(offset) * fps))) for offset in frame_offsets]
    max_offset_frames = max(offset_frames)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    history = deque(maxlen=max_offset_frames)
    frame_idx = 0

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"fps={fps:.3f}, frames={total}, size={width}x{height}, offset_frames={offset_frames}")

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            current_rgb = frame_to_rgb_tensor(frame_bgr)
            previous_rgbs = [
                history[-frame_offset] if len(history) >= frame_offset else None
                for frame_offset in offset_frames
            ]
            image = make_multiframe_input(current_rgb, previous_rgbs, size).unsqueeze(0).to(device)
            orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)

            outputs = model(image, orig_size)[0]
            vis = draw_on_frame(frame_bgr, outputs["labels"], outputs["boxes"], outputs["scores"], threshold)
            writer.write(vis)

            history.append(current_rgb)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"processed {frame_idx}/{total}")

    cap.release()
    writer.release()
    print(f"done: {frame_idx} frames -> {output_path}")


def iter_inputs(input_path):
    path = Path(input_path)
    if path.is_dir():
        for suffix in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            yield from sorted(path.glob(suffix))
    else:
        yield path


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, size = load_model(args.config, args.resume, device)
    frame_offsets = args.frame_offsets if args.frame_offsets is not None else [args.frame_offset]
    input_channels = 3 + len(frame_offsets)

    output_dir = Path(args.output_dir)
    for input_file in iter_inputs(args.input):
        output_path = output_dir / f"{input_file.stem}_thr{args.threshold:.2f}_{input_channels}ch.mp4"
        process_video(
            model,
            input_file,
            output_path,
            size=size,
            threshold=args.threshold,
            frame_offsets=frame_offsets,
            device=device,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", default="./outputs/inference_phase11_5_test")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--frame-offset", type=float, default=0.5)
    parser.add_argument("--frame-offsets", type=float, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    main(parser.parse_args())
