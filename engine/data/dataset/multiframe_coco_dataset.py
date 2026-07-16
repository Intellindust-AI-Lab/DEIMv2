"""
COCO dataset variant that appends configurable grayscale frame-difference channels.
"""

import bisect
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    ):
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        self.frame_offsets = [float(v) for v in frame_offsets]
        self.frame_tolerance = float(frame_tolerance)
        self.diff_mode = diff_mode
        self.missing_frame = missing_frame
        self._frame_index = self._build_frame_index()

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

    def _load_rgb_tensor(self, file_name: str, size=None) -> torch.Tensor:
        path = os.path.join(self.img_folder, file_name)
        image = PILImage.open(path).convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, resample=PILImage.BILINEAR)
        return F.pil_to_tensor(image).float() / 255.0

    def _make_diff_channel(self, current_rgb: torch.Tensor, file_name: str, offset: float) -> torch.Tensor:
        prev_file = self._previous_file(file_name, offset)
        if prev_file is None:
            if self.missing_frame == "current":
                prev_rgb = current_rgb
            else:
                return torch.zeros((1, current_rgb.shape[1], current_rgb.shape[2]), dtype=current_rgb.dtype)
        else:
            prev_rgb = self._load_rgb_tensor(prev_file, size=(current_rgb.shape[2], current_rgb.shape[1]))

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

    def load_item(self, idx):
        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))

        current_rgb = self._load_rgb_tensor(file_name)
        channels = [current_rgb]
        channels.extend(self._make_diff_channel(current_rgb, file_name, offset) for offset in self.frame_offsets)
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
