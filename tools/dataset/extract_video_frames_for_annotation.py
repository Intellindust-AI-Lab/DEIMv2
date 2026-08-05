#!/usr/bin/env python3
"""Extract fixed-interval video frames and a smaller annotation subset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def iter_videos(input_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for suffix in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        videos.extend(sorted(input_dir.glob(suffix)))
    return sorted(videos)


def frame_name(video: Path, seconds: float) -> str:
    tenth = int(round(seconds * 10))
    return f"{video.stem}_{tenth:04d}.jpg"


def planned_times(duration: float, interval: float) -> list[float]:
    count = int(duration / interval)
    return [i * interval for i in range(count)]


def read_frame_at(cap: cv2.VideoCapture, fps: float, seconds: float):
    frame_idx = max(0, int(round(seconds * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame, frame_idx


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "video", "time_sec", "frame_idx", "fps", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--all-output-dir", required=True)
    parser.add_argument("--sample-output-dir", required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_output_dir = Path(args.all_output_dir)
    sample_output_dir = Path(args.sample_output_dir)
    all_output_dir.mkdir(parents=True, exist_ok=True)
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    videos = iter_videos(input_dir)
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")

    all_rows: list[dict] = []
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for seconds in planned_times(args.duration, args.interval):
            ok, frame, frame_idx = read_frame_at(cap, fps, seconds)
            if not ok:
                continue
            name = frame_name(video, seconds)
            out_path = all_output_dir / name
            cv2.imwrite(str(out_path), frame, encode_params)
            all_rows.append(
                {
                    "image": name,
                    "video": video.name,
                    "time_sec": f"{seconds:.1f}",
                    "frame_idx": frame_idx,
                    "fps": f"{fps:.6f}",
                    "width": width,
                    "height": height,
                }
            )

        cap.release()
        print(f"extracted {video.name}")

    write_manifest(all_output_dir / "frames_manifest.csv", all_rows)

    if len(all_rows) <= args.sample_count:
        sample_rows = all_rows
    else:
        indices = np.linspace(0, len(all_rows) - 1, args.sample_count).round().astype(int)
        sample_rows = [all_rows[i] for i in sorted(set(indices.tolist()))]

    for row in sample_rows:
        shutil.copy2(all_output_dir / row["image"], sample_output_dir / row["image"])

    write_manifest(sample_output_dir / "annotation_sample_manifest.csv", sample_rows)
    print(f"wrote {len(all_rows)} frames to {all_output_dir}")
    print(f"wrote {len(sample_rows)} annotation samples to {sample_output_dir}")


if __name__ == "__main__":
    main()
