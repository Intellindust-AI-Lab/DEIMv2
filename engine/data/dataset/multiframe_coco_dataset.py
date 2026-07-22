"""
COCO dataset variant that appends configurable grayscale frame-difference channels.
"""

import bisect
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2.functional as F
from PIL import Image as PILImage

from .coco_dataset import CocoDetection, mscoco_category2label
from .._misc import Image, convert_to_tv_tensor
from ...core import register

torchvision.disable_beta_transforms_warning()
PILImage.MAX_IMAGE_PIXELS = None

__all__ = ["MultiFrameCocoDetection"]


_FRAME_RE = re.compile(
    r"^(?P<scene>.+)_(?P<md>\d{3,4})_(?P<hour>\d{1,2})_"
    r"(?P<minute>\d{1,2})_(?P<second>\d{1,2})_(?P<frac>\d+)$"
)


def _parse_frame_key(file_name: str) -> Optional[Tuple[str, float]]:
    stem = Path(file_name).stem
    match = _FRAME_RE.match(stem)
    if match is None:
        return None

    md = match.group("md").zfill(4)
    month = int(md[:2])
    day = int(md[2:])
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    frac_text = match.group("frac")
    frac = int(frac_text) / (10 ** len(frac_text))

    timestamp = (((month * 31 + day) * 24 + hour) * 60 + minute) * 60 + second + frac
    return match.group("scene"), timestamp


@register()
class MultiFrameCocoDetection(CocoDetection):
    """
    Adds one channel per requested previous-frame offset.

    The output is CxHxW float image data in [0, 1]. For example:
    frame_offsets: [0.5] -> RGB + gray(current - previous_0.5s), 4 channels
    frame_offsets: [0.5, 1.0] -> RGB + two difference channels, 5 channels
    """

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        frame_offsets=(0.5,),
        frame_tolerance=0.05,
        diff_mode="signed",
        missing_frame="zeros",
        diff_source="previous",
        video_folder=None,
        video_duration=120.0,
        background_max_frames=120,
        background_sample_interval=None,
    ):
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        self.frame_offsets = [float(v) for v in frame_offsets]
        self.frame_tolerance = float(frame_tolerance)
        self.diff_mode = diff_mode
        self.missing_frame = missing_frame
        self.diff_source = diff_source
        self.video_folder = video_folder or self._infer_video_folder(img_folder)
        self.video_duration = float(video_duration)
        self.background_max_frames = int(background_max_frames)
        self.background_sample_interval = None if background_sample_interval is None else float(background_sample_interval)
        self._frame_index = self._build_frame_index()
        self._video_index = self._build_video_index()
        self._background_groups = self._build_background_groups()
        self._background_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    @staticmethod
    def _infer_video_folder(img_folder):
        img_path = Path(img_folder)
        return str(img_path.parent / "videos" / img_path.name)

    def _build_frame_index(self) -> Dict[str, List[Tuple[float, str]]]:
        index: Dict[str, List[Tuple[float, str]]] = {}
        for image_info in self.coco.dataset["images"]:
            file_name = image_info["file_name"]
            parsed = _parse_frame_key(file_name)
            if parsed is None:
                continue
            scene, timestamp = parsed
            index.setdefault(scene, []).append((timestamp, file_name))

        for frames in index.values():
            frames.sort(key=lambda item: item[0])
        return index

    def _build_video_index(self) -> Dict[str, List[Tuple[float, float, str]]]:
        index: Dict[str, List[Tuple[float, float, str]]] = {}
        video_dir = Path(self.video_folder)
        if not video_dir.exists():
            return index

        for path in sorted(video_dir.glob("*.mp4")):
            parsed = _parse_frame_key(path.name)
            if parsed is None:
                continue
            scene, start = parsed
            index.setdefault(scene, []).append((start, start + self.video_duration, path.stem))

        for videos in index.values():
            videos.sort(key=lambda item: item[0])
        return index

    def _background_group_key(self, file_name: str) -> Optional[Tuple[str, str]]:
        parsed = _parse_frame_key(file_name)
        if parsed is None:
            return None

        scene, timestamp = parsed
        for start, end, video_key in self._video_index.get(scene, []):
            if start <= timestamp < end:
                return scene, video_key

        bucket = int(timestamp // self.video_duration)
        return scene, f"{scene}_bucket_{bucket}"

    def _build_background_groups(self) -> Dict[Tuple[str, str], List[str]]:
        groups: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
        for image_info in self.coco.dataset["images"]:
            file_name = image_info["file_name"]
            parsed = _parse_frame_key(file_name)
            key = self._background_group_key(file_name)
            if parsed is None or key is None:
                continue
            _, timestamp = parsed
            groups.setdefault(key, []).append((timestamp, file_name))

        return {
            key: [file_name for _, file_name in sorted(items, key=lambda item: item[0])]
            for key, items in groups.items()
        }

    def _previous_file(self, file_name: str, offset: float) -> Optional[str]:
        parsed = _parse_frame_key(file_name)
        if parsed is None:
            return None
        scene, timestamp = parsed
        frames = self._frame_index.get(scene)
        if not frames:
            return None

        target_time = timestamp - offset
        times = [item[0] for item in frames]
        pos = bisect.bisect_left(times, target_time)
        candidates = []
        if pos < len(frames):
            candidates.append(frames[pos])
        if pos > 0:
            candidates.append(frames[pos - 1])
        if not candidates:
            return None

        best_time, best_file = min(candidates, key=lambda item: abs(item[0] - target_time))
        if abs(best_time - target_time) <= self.frame_tolerance:
            return best_file
        return None

    def _sample_background_files(self, files: List[str]) -> List[str]:
        sampled = files
        if self.background_sample_interval is not None and self.background_sample_interval > 0:
            sampled = []
            last_time = None
            for file_name in files:
                parsed = _parse_frame_key(file_name)
                if parsed is None:
                    continue
                _, timestamp = parsed
                if last_time is None or timestamp - last_time >= self.background_sample_interval:
                    sampled.append(file_name)
                    last_time = timestamp

        if self.background_max_frames <= 0 or len(sampled) <= self.background_max_frames:
            return sampled
        indices = np.linspace(0, len(sampled) - 1, self.background_max_frames).round().astype(int)
        return [sampled[i] for i in sorted(set(indices.tolist()))]

    def _load_background_rgb(self, file_name: str, current_rgb: torch.Tensor) -> torch.Tensor:
        key = self._background_group_key(file_name)
        if key is None:
            return current_rgb
        if key in self._background_cache:
            return self._background_cache[key]

        files = self._sample_background_files(self._background_groups.get(key, []))
        if not files:
            self._background_cache[key] = current_rgb
            return current_rgb

        arrays = []
        target_size = (current_rgb.shape[2], current_rgb.shape[1])
        for bg_file in files:
            path = os.path.join(self.img_folder, bg_file)
            image = PILImage.open(path).convert("RGB")
            if image.size != target_size:
                image = image.resize(target_size, resample=PILImage.BILINEAR)
            arrays.append(np.asarray(image, dtype=np.uint8))

        median = np.median(np.stack(arrays, axis=0), axis=0).astype(np.uint8)
        background = F.pil_to_tensor(PILImage.fromarray(median, mode="RGB")).float() / 255.0
        self._background_cache[key] = background
        print(f"Built median background for {key[1]} from {len(arrays)} frames.")
        return background

    def _load_rgb_tensor(self, file_name: str, size=None) -> torch.Tensor:
        path = os.path.join(self.img_folder, file_name)
        image = PILImage.open(path).convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, resample=PILImage.BILINEAR)
        return F.pil_to_tensor(image).float() / 255.0

    def _rgb_diff_channel(self, current_rgb: torch.Tensor, prev_rgb: torch.Tensor) -> torch.Tensor:
        cur_gray = current_rgb.mean(dim=0, keepdim=True)
        prev_gray = prev_rgb.mean(dim=0, keepdim=True)
        diff = cur_gray - prev_gray
        if self.diff_mode == "abs":
            diff = diff.abs()
        elif self.diff_mode == "signed":
            diff = (diff + 1.0) * 0.5
        else:
            raise ValueError(f"Unsupported diff_mode: {self.diff_mode}")
        return diff.clamp(0.0, 1.0)

    def _make_background_diff_channel(self, current_rgb: torch.Tensor, file_name: str) -> torch.Tensor:
        background_rgb = self._load_background_rgb(file_name, current_rgb)
        return self._rgb_diff_channel(current_rgb, background_rgb)

    def _make_diff_channel(self, current_rgb: torch.Tensor, file_name: str, offset: float) -> torch.Tensor:
        if self.diff_source == "background_median":
            return self._make_background_diff_channel(current_rgb, file_name)

        prev_file = self._previous_file(file_name, offset)
        if prev_file is None:
            if self.missing_frame == "current":
                prev_rgb = current_rgb
            else:
                return torch.zeros((1, current_rgb.shape[1], current_rgb.shape[2]), dtype=current_rgb.dtype)
        else:
            prev_rgb = self._load_rgb_tensor(prev_file, size=(current_rgb.shape[2], current_rgb.shape[1]))

        return self._rgb_diff_channel(current_rgb, prev_rgb)

    def load_item(self, idx):
        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        current_rgb = self._load_rgb_tensor(file_name)
        channels = [current_rgb]
        channels.extend(self._make_diff_channel(current_rgb, file_name, offset) for offset in self.frame_offsets)
        if self.diff_source == "previous_and_background_median":
            channels.append(self._make_background_diff_channel(current_rgb, file_name))
        image = Image(torch.cat(channels, dim=0))

        target = {"image_id": image_id, "annotations": target}
        category2label = mscoco_category2label if self.remap_mscoco_category else self.category2label
        _, target = self.prepare(PILImage.new("RGB", (current_rgb.shape[2], current_rgb.shape[1])), target, category2label=category2label)
        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=(current_rgb.shape[1], current_rgb.shape[2]))
        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target
