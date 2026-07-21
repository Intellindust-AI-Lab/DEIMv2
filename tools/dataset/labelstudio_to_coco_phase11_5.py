"""
Convert the phase_11_5 Label Studio exports to COCO detection JSON.
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image


LABEL_MAP = {
    "person": 0,
    "head": 1,
}


def image_path_from_task(task, image_root):
    raw = task["data"]["image"]
    marker = "data_phase11_5/"
    if marker in raw:
        rel = raw.split(marker, 1)[1]
    else:
        rel = os.path.basename(raw)
    return Path(image_root) / rel


def convert(input_json, image_root, output_json):
    with open(input_json, "r") as f:
        tasks = json.load(f)

    images = []
    annotations = []
    ann_id = 1

    for image_id, task in enumerate(tasks, start=1):
        image_path = image_path_from_task(task, image_root)
        with Image.open(image_path) as img:
            width, height = img.size

        file_name = image_path.name
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                value = result["value"]
                labels = value.get("rectanglelabels", [])
                if not labels:
                    continue
                label = labels[0]
                if label not in LABEL_MAP:
                    continue

                x = value["x"] * width / 100.0
                y = value["y"] * height / 100.0
                w = value["width"] * width / 100.0
                h = value["height"] * height / 100.0
                if w <= 0 or h <= 0:
                    continue

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": LABEL_MAP[label],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in LABEL_MAP.items()],
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--image-root", default="/datadrive/data_phase11_5")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    convert(args.input_json, args.image_root, args.output_json)
