"""Convert YOLO-style txt annotations to a single COCO-format json file.

Expected layout for every dataset under ``--data-root`` (only the ``cropped_masked``
sub-folder of each dataset is used)::

    <data_root>/<dataset_name>/cropped_masked/
        images/{train,val}/*.jpg
        labels/{train,val}/*.txt     # rows: class x_center y_center w h occlusion (normalized)
        {train,val}.txt              # image paths relative to the cropped_masked folder

Each label row is ``class_index x_center y_center width height occlusion`` with
coordinates normalized to [0, 1]. Occlusion (0/1) is stored as an ``occlusion``
attribute on every COCO annotation.

Usage::

    python tools/dataset/yolo_to_coco.py --data-root /path/to/product_detection --split train
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

CATEGORIES = {
    0: 'product',
    1: 'cellphone',
    2: 'wallet',
    3: 'credit_card',
    4: 'other_item',
    5: 'inseparable',
    6: 'bag',
}
SUBFOLDER = 'cropped_masked'


@dataclass
class ImageRecord:
    """A single image with its annotations, ready to be merged into the COCO json."""

    file_name: str  # relative to data root
    width: int
    height: int
    boxes: list  # list of (category_id, x, y, w, h, occlusion) in absolute pixels


def parse_label_file(label_path: Path, width: int, height: int) -> list:
    """Parse one YOLO label file, returning absolute-pixel boxes clamped to the image.

    :param label_path: Path to the ``.txt`` label file.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :returns: List of ``(category_id, x, y, w, h, occlusion)`` tuples.
    """
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        category_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        occlusion = int(float(parts[5])) if len(parts) > 5 else 0
        x1 = max((cx - w / 2) * width, 0.0)
        y1 = max((cy - h / 2) * height, 0.0)
        x2 = min((cx + w / 2) * width, width)
        y2 = min((cy + h / 2) * height, height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((category_id, x1, y1, x2 - x1, y2 - y1, occlusion))
    return boxes


def process_image(args: 'tuple[Path, Path, str]') -> 'ImageRecord | None':
    """Read image size and labels for one image listed in a split file.

    :param args: ``(data_root, dataset_dir, image_rel_path)`` where ``image_rel_path``
        is the path from the split txt file, relative to the ``cropped_masked`` folder.
    :returns: An :class:`ImageRecord`, or ``None`` if the image or label file is missing.
    """
    data_root, dataset_dir, image_rel_path = args
    image_path = (dataset_dir / SUBFOLDER / image_rel_path).resolve()
    label_path = Path(str(image_path).replace('/images/', '/labels/')).with_suffix('.txt')
    if not image_path.is_file() or not label_path.is_file():
        return None
    with Image.open(image_path) as image:
        width, height = image.size
    return ImageRecord(
        file_name=str(image_path.relative_to(data_root.resolve())),
        width=width,
        height=height,
        boxes=parse_label_file(label_path, width, height),
    )


def collect_split_tasks(data_root: Path, split: str, exclude: 'list[str]') -> list:
    """Gather per-image tasks for every dataset that has a ``cropped_masked/<split>.txt``.

    :param data_root: Root folder containing the dataset sub-folders.
    :param split: Either ``train`` or ``val``.
    :param exclude: Dataset folder names to skip.
    :returns: List of argument tuples for :func:`process_image`.
    """
    tasks = []
    for dataset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if dataset_dir.name in exclude:
            print(f'excluding {dataset_dir.name}')
            continue
        split_file = dataset_dir / SUBFOLDER / f'{split}.txt'
        if not split_file.is_file():
            print(f'skipping {dataset_dir.name}: no {SUBFOLDER}/{split}.txt')
            continue
        rel_paths = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        tasks.extend((data_root, dataset_dir, rel_path) for rel_path in rel_paths)
        print(f'{dataset_dir.name}: {len(rel_paths)} images')
    return tasks


def build_coco_dict(records: 'list[ImageRecord]') -> dict:
    """Assemble the COCO-format dictionary from processed image records.

    :param records: Image records with absolute-pixel boxes.
    :returns: COCO-format dictionary with ``images``, ``annotations`` and ``categories``.
    """
    images, annotations = [], []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        images.append({
            'id': image_id,
            'file_name': record.file_name,
            'width': record.width,
            'height': record.height,
        })
        for category_id, x, y, w, h, occlusion in record.boxes:
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                'area': round(w * h, 2),
                'iscrowd': 0,
                'occlusion': occlusion,
            })
            annotation_id += 1
    categories = [{'id': idx, 'name': name} for idx, name in CATEGORIES.items()]
    return {'images': images, 'annotations': annotations, 'categories': categories}


def convert_split(data_root: Path, split: str, output_file: Path, num_workers: int, exclude: 'list[str]') -> None:
    """Convert one split across all datasets and write the merged COCO json.

    :param data_root: Root folder containing the dataset sub-folders.
    :param split: Either ``train`` or ``val``.
    :param output_file: Destination json path.
    :param num_workers: Number of processes for reading image headers.
    :param exclude: Dataset folder names to skip.
    """
    tasks = collect_split_tasks(data_root, split, exclude)
    print(f'processing {len(tasks)} images for split "{split}" with {num_workers} workers...')
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(process_image, tasks, chunksize=256))
    records = [record for record in results if record is not None]
    missing = len(results) - len(records)
    if missing:
        print(f'warning: skipped {missing} entries with missing image or label file')
    coco = build_coco_dict(records)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        json.dump(coco, f)
    print(f'wrote {output_file}: {len(coco["images"])} images, {len(coco["annotations"])} annotations')


def main() -> None:
    """Parse command line arguments and run the conversion."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, required=True, help='folder containing the dataset sub-folders')
    parser.add_argument('--split', choices=['train', 'val'], required=True, help='which split to convert')
    parser.add_argument('--output', type=Path, default=None,
                        help='output json path (default: <data-root>/annotations/<split>.json)')
    parser.add_argument('--num-workers', type=int, default=16, help='processes used to read image headers')
    parser.add_argument('--exclude', nargs='*', default=[], help='dataset folder names to skip')
    args = parser.parse_args()

    output = args.output or args.data_root / 'annotations' / f'{args.split}.json'
    convert_split(args.data_root, args.split, output, args.num_workers, args.exclude)


if __name__ == '__main__':
    main()
