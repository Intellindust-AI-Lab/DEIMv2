"""
Multiframe video inference for EX001/EX003.

Input tensor is RGB + one grayscale frame-difference channel per offset:
current RGB, plus gray(current) - gray(previous frame at --frame-offsets seconds).
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
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


def make_multiframe_input(current_rgb, previous_rgbs, size, background_rgb=None):
    channels = [current_rgb]
    channels.extend(make_diff_channel(current_rgb, previous_rgb) for previous_rgb in previous_rgbs)
    if background_rgb is not None:
        channels.append(make_diff_channel(current_rgb, background_rgb))
    image = torch.cat(channels, dim=0)
    return F.resize(image, size, antialias=True)


def sample_background_indices(total_frames, fps, sample_interval, max_frames):
    if total_frames <= 0:
        return []
    if sample_interval is not None and sample_interval > 0 and fps > 0:
        step = max(1, int(round(sample_interval * fps)))
        indices = list(range(0, total_frames, step))
    else:
        indices = list(range(total_frames))

    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        sampled = np.linspace(0, len(indices) - 1, max_frames).round().astype(int)
        indices = [indices[i] for i in sorted(set(sampled.tolist()))]
    return indices


def build_median_background(video_path, fps, total_frames, sample_interval, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for background: {video_path}")

    frames = []
    for idx in sample_background_indices(total_frames, fps, sample_interval, max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None, 0

    median_rgb = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    background_rgb = torch.from_numpy(median_rgb).permute(2, 0, 1).float() / 255.0
    print(f"built median background from {len(frames)} frames")
    return background_rgb, len(frames)


def max_rss_mb():
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024 if sys.platform != "darwin" else 1024 ** 2)


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


def process_video(
    model,
    input_path,
    output_path,
    size,
    threshold,
    frame_offsets,
    device,
    max_frames=None,
    diff_source="previous",
    background_sample_interval=None,
    background_max_frames=120,
    duration=None,
    batch_size=1,
    warmup=10,
    metrics_json=None,
    config_path=None,
    checkpoint_path=None,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    offset_frames = [max(1, int(round(float(offset) * fps))) for offset in frame_offsets]
    max_offset_frames = max(offset_frames)
    frame_limit = total if total > 0 else None
    if duration is not None:
        duration_frame_limit = max(1, int(round(duration * fps)))
        frame_limit = min(frame_limit, duration_frame_limit) if frame_limit else duration_frame_limit
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames) if frame_limit else max_frames

    use_background = diff_source in {"background_median", "previous_and_background_median"}
    background_rgb = None
    background_seconds = 0.0
    background_sampled_frames = 0
    if use_background:
        background_start = time.perf_counter()
        background_rgb, background_sampled_frames = build_median_background(
            input_path,
            fps=fps,
            total_frames=total,
            sample_interval=background_sample_interval,
            max_frames=background_max_frames,
        )
        background_seconds = time.perf_counter() - background_start

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    history = deque(maxlen=max_offset_frames)
    frame_idx = 0
    batch_idx = 0
    measured_batches = 0
    measured_frames = 0
    preprocess_cpu_seconds = 0.0
    h2d_ms = 0.0
    inference_ms = 0.0
    draw_write_seconds = 0.0
    wall_start = time.perf_counter()

    cuda_device = device.type == "cuda"
    if cuda_device:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
    else:
        baseline_allocated = 0
        baseline_reserved = 0

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"fps={fps:.3f}, frames={total}, size={width}x{height}, offset_frames={offset_frames}")
    if frame_limit is not None:
        print(f"benchmark frames={frame_limit} ({frame_limit / fps:.2f}s), batch_size={batch_size}, warmup={warmup}")
    if use_background:
        print(f"background build: {background_seconds:.3f}s from {background_sampled_frames} frames")

    try:
        with torch.inference_mode():
            while frame_limit is None or frame_idx < frame_limit:
                batch_frames = []
                batch_images = []

                while len(batch_images) < batch_size and (frame_limit is None or frame_idx + len(batch_images) < frame_limit):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    preprocess_start = time.perf_counter()
                    current_rgb = frame_to_rgb_tensor(frame_bgr)
                    if diff_source == "background_median":
                        previous_rgbs = []
                    else:
                        previous_rgbs = [
                            history[-frame_offset] if len(history) >= frame_offset else None
                            for frame_offset in offset_frames
                        ]
                    image = make_multiframe_input(
                        current_rgb,
                        previous_rgbs,
                        size,
                        background_rgb=background_rgb,
                    )
                    preprocess_cpu_seconds += time.perf_counter() - preprocess_start

                    batch_frames.append(frame_bgr)
                    batch_images.append(image)
                    history.append(current_rgb)

                if not batch_images:
                    break

                current_batch_size = len(batch_images)
                batch_tensor_cpu = torch.stack(batch_images)

                if cuda_device:
                    h2d_start = torch.cuda.Event(enable_timing=True)
                    h2d_end = torch.cuda.Event(enable_timing=True)
                    h2d_start.record()
                    image = batch_tensor_cpu.to(device, non_blocking=True)
                    orig_size = torch.tensor([[width, height]] * current_batch_size, dtype=torch.float32, device=device)
                    h2d_end.record()
                    torch.cuda.synchronize()
                    h2d_batch_ms = h2d_start.elapsed_time(h2d_end)

                    infer_start = torch.cuda.Event(enable_timing=True)
                    infer_end = torch.cuda.Event(enable_timing=True)
                    infer_start.record()
                    outputs = model(image, orig_size)
                    infer_end.record()
                    torch.cuda.synchronize()
                    infer_batch_ms = infer_start.elapsed_time(infer_end)
                else:
                    h2d_start_seconds = time.perf_counter()
                    image = batch_tensor_cpu.to(device)
                    orig_size = torch.tensor([[width, height]] * current_batch_size, dtype=torch.float32, device=device)
                    h2d_batch_ms = (time.perf_counter() - h2d_start_seconds) * 1000.0

                    infer_start_seconds = time.perf_counter()
                    outputs = model(image, orig_size)
                    infer_batch_ms = (time.perf_counter() - infer_start_seconds) * 1000.0

                if batch_idx >= warmup:
                    h2d_ms += h2d_batch_ms
                    inference_ms += infer_batch_ms
                    measured_batches += 1
                    measured_frames += current_batch_size

                draw_start = time.perf_counter()
                for frame_bgr, output in zip(batch_frames, outputs):
                    vis = draw_on_frame(frame_bgr, output["labels"], output["boxes"], output["scores"], threshold)
                    writer.write(vis)
                draw_write_seconds += time.perf_counter() - draw_start

                frame_idx += current_batch_size
                batch_idx += 1
                if frame_idx % 100 < current_batch_size:
                    denominator = frame_limit if frame_limit is not None else total
                    print(f"processed {frame_idx}/{denominator}")
    finally:
        cap.release()
        writer.release()

    if cuda_device:
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_name = device_props.name
        gpu_total_memory_mb = device_props.total_memory / 1024 ** 2
    else:
        peak_allocated = 0
        peak_reserved = 0
        gpu_name = str(device)
        gpu_total_memory_mb = None

    wall_seconds = time.perf_counter() - wall_start
    processed_duration = frame_idx / fps if fps > 0 else None
    preprocess_ms_per_frame = (preprocess_cpu_seconds * 1000.0 / frame_idx) if frame_idx else None
    h2d_ms_per_frame = (h2d_ms / measured_frames) if measured_frames else None
    inference_ms_per_frame = (inference_ms / measured_frames) if measured_frames else None
    draw_write_ms_per_frame = (draw_write_seconds * 1000.0 / frame_idx) if frame_idx else None
    model_fps = (1000.0 / inference_ms_per_frame) if inference_ms_per_frame else None
    end_to_end_fps = frame_idx / wall_seconds if wall_seconds else None

    print(f"done: {frame_idx} frames -> {output_path}")

    metrics = {
        "input": os.path.abspath(str(input_path)),
        "output": os.path.abspath(str(output_path)),
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total,
            "duration_seconds": total / fps if total > 0 and fps > 0 else None,
        },
        "conditions": {
            "config": os.path.abspath(config_path) if config_path else None,
            "checkpoint": os.path.abspath(checkpoint_path) if checkpoint_path else None,
            "batch_size": batch_size,
            "input_size": list(size),
            "input_channels": 3 + (0 if diff_source == "background_median" else len(frame_offsets)) + (1 if use_background else 0),
            "frame_offsets_seconds": list(frame_offsets),
            "diff_source": diff_source,
            "precision": "float32",
            "output_codec": "mp4v",
            "threshold": threshold,
            "warmup_batches": warmup,
            "requested_duration_seconds": duration,
            "background_sample_interval": background_sample_interval,
            "background_max_frames": background_max_frames,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": gpu_name,
            "gpu_total_memory_mb": gpu_total_memory_mb,
            "platform": platform.platform(),
        },
        "results": {
            "processed_frames": frame_idx,
            "processed_video_seconds": processed_duration,
            "measured_batches": measured_batches,
            "measured_frames": measured_frames,
            "background_build_seconds": background_seconds,
            "background_sampled_frames": background_sampled_frames,
            "six_channel_build_total_seconds": preprocess_cpu_seconds,
            "six_channel_build_ms_per_frame": preprocess_ms_per_frame,
            "host_to_device_total_ms": h2d_ms if measured_frames else None,
            "host_to_device_ms_per_frame": h2d_ms_per_frame,
            "model_and_postprocess_total_ms": inference_ms if measured_frames else None,
            "model_and_postprocess_ms_per_frame": inference_ms_per_frame,
            "model_throughput_fps": model_fps,
            "draw_write_total_seconds": draw_write_seconds,
            "draw_write_ms_per_frame": draw_write_ms_per_frame,
            "end_to_end_seconds": wall_seconds,
            "end_to_end_fps": end_to_end_fps,
            "realtime_factor": processed_duration / wall_seconds if processed_duration is not None and wall_seconds else None,
            "gpu_peak_allocated_mb": peak_allocated / 1024 ** 2,
            "gpu_peak_reserved_mb": peak_reserved / 1024 ** 2,
            "gpu_runtime_allocated_increase_mb": max(0, peak_allocated - baseline_allocated) / 1024 ** 2,
            "gpu_runtime_reserved_increase_mb": max(0, peak_reserved - baseline_reserved) / 1024 ** 2,
            "process_max_rss_mb": max_rss_mb(),
        },
    }

    print("\n" + "=" * 72)
    print("  EX006 Multiframe Video Performance Summary")
    print("=" * 72)
    print(f"  GPU / PyTorch        : {gpu_name} / {torch.__version__}")
    print(f"  Input / Batch        : {size[0]}x{size[1]}x{metrics['conditions']['input_channels']} / {batch_size}")
    print(f"  Video processed      : {processed_duration:.2f} s ({frame_idx} frames)")
    print(f"  Background build     : {background_seconds:.3f} s ({background_sampled_frames} frames)")
    print(f"  6ch build            : {preprocess_ms_per_frame:.2f} ms/frame")
    if h2d_ms_per_frame is not None:
        print(f"  Host->device         : {h2d_ms_per_frame:.2f} ms/frame")
    if inference_ms_per_frame is not None:
        print(f"  Model+postprocess    : {inference_ms_per_frame:.2f} ms/frame ({model_fps:.2f} FPS)")
        print(f"  Timed frames         : {measured_frames} (warmup={warmup} batches excluded)")
    else:
        print("  Model+postprocess    : no timed frames; reduce --warmup")
    print(f"  Draw/write           : {draw_write_ms_per_frame:.2f} ms/frame")
    print(f"  End-to-end           : {wall_seconds:.2f} s ({end_to_end_fps:.2f} FPS)")
    print(f"  Realtime factor      : {metrics['results']['realtime_factor']:.2f}x")
    print(f"  GPU peak allocated   : {metrics['results']['gpu_peak_allocated_mb']:.1f} MB")
    print(f"  GPU peak reserved    : {metrics['results']['gpu_peak_reserved_mb']:.1f} MB")
    print(f"  Process max RSS      : {metrics['results']['process_max_rss_mb']:.1f} MB")
    print("=" * 72)

    if metrics_json:
        metrics_path = Path(metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"metrics json: {metrics_path}")

    return metrics


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
    if args.diff_source == "background_median":
        input_channels = 4
    else:
        input_channels = 3 + len(frame_offsets)
        if args.diff_source == "previous_and_background_median":
            input_channels += 1

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
            diff_source=args.diff_source,
            background_sample_interval=args.background_sample_interval,
            background_max_frames=args.background_max_frames,
            duration=args.duration,
            batch_size=args.batch_size,
            warmup=args.warmup,
            metrics_json=args.metrics_json,
            config_path=args.config,
            checkpoint_path=args.resume,
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
    parser.add_argument(
        "--diff-source",
        choices=["previous", "background_median", "previous_and_background_median"],
        default="previous",
    )
    parser.add_argument("--background-sample-interval", type=float, default=None)
    parser.add_argument("--background-max-frames", type=int, default=120)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    main(args)
