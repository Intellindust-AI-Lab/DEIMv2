import builtins
import copy
import json

import numpy as np
import pytest
import torch
import yaml
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

from engine.core import YAMLConfig, yaml_utils
from engine.data.dataset import coco_eval
from engine.data.dataset.coco_eval import CocoEvaluator

pytest.importorskip("ultrafast_pycocotools")


def test_merge_keeps_matching_records_for_unique_sorted_image_ids(monkeypatch):
    image_ids = [[2, 1], [1, 3]]
    records = [np.arange(16).reshape(2, 4, 2), np.arange(16, 32).reshape(2, 4, 2)]
    gathered = iter([image_ids, [[records[0]], [records[1]]]])
    monkeypatch.setattr(coco_eval.dist_utils, "all_gather", lambda value: next(gathered))
    ids, values = coco_eval.merge(image_ids[0], [records[0]])
    assert ids == [1, 2, 3]
    expected = np.concatenate(records, axis=2)[..., [1, 0, 3]].ravel()
    np.testing.assert_array_equal(values, expected)


def _inputs():
    from faster_coco_eval import COCO

    keypoints = [[3 + i % 4, 4 + i // 4, 2] for i in range(17)]
    ann = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": [2, 2, 12, 12],
        "area": 144,
        "segmentation": [[2, 2, 14, 2, 14, 14, 2, 14]],
        "keypoints": np.asarray(keypoints).flatten().tolist(),
        "num_keypoints": 17,
    }
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": i, "width": 32, "height": 32} for i in (1, 2, 3)],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [ann, dict(ann, id=2, image_id=2, iscrowd=1)],
    }
    for image in gt.dataset["images"]:
        image.update(neg_category_ids=[], not_exhaustive_category_ids=[])
    for category in gt.dataset["categories"]:
        category["frequency"] = "c"
    gt.createIndex()
    masks = torch.zeros((3, 1, 32, 32))
    masks[0, :, 18:30, 18:30] = 1
    masks[1:, :, 2:14, 2:14] = 1
    pose = torch.tensor(keypoints, dtype=torch.float32)
    shifted_pose = pose.clone()
    shifted_pose[:, :2] += 16
    prediction = {
        "boxes": torch.tensor([[18.0, 18.0, 30.0, 30.0], [2.0, 2.0, 14.0, 14.0], [2.0, 2.0, 14.0, 14.0]]),
        "scores": torch.tensor([0.9, 0.8, 0.8]),
        "labels": torch.ones(3, dtype=torch.int64),
        "masks": masks,
        "keypoints": torch.stack([shifted_pose, pose, pose]),
    }
    return gt, {1: prediction, 2: {}, 3: {}}


def _finish(evaluator):
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()


def _assert_equal(candidate, reference):
    for iou_type in reference.iou_types:
        actual, expected = candidate.coco_eval[iou_type], reference.coco_eval[iou_type]
        assert actual.params.imgIds == expected.params.imgIds
        for key in ("precision", "recall", "scores"):
            np.testing.assert_array_equal(actual.eval[key], expected.eval[key])
        np.testing.assert_array_equal(actual.stats, expected.stats)
        np.testing.assert_array_equal(actual.all_stats, expected.all_stats)
        assert actual.stats_as_dict == expected.stats_as_dict


@pytest.mark.parametrize("iou_types", [["bbox"], ["segm"], ["keypoints"], ["bbox", "segm", "keypoints"]])
@pytest.mark.parametrize("cap", [None, 500])
def test_arrays_and_epoch_cleanup(iou_types, cap):
    gt, predictions = _inputs()
    original = copy.deepcopy(gt.dataset)
    reference = FasterCocoEvaluator(gt, iou_types)
    candidates = [CocoEvaluator(gt, iou_types, backend=backend) for backend in ("faster_coco_eval", "ultrafast")]
    for epoch in range(2):
        for evaluator in (reference, *candidates):
            evaluator.cleanup()
            assert evaluator.img_ids == []
            if cap:
                for kind, metric in evaluator.coco_eval.items():
                    metric.params.maxDets = [cap] if kind == "keypoints" else [cap, 1, 10]
                    if kind != "keypoints":
                        metric.params.areaRng = [[0, 1e10], [0, 100], [100, 400], [400, 1e10]]
            prediction = predictions[1] if epoch == 0 else {k: v[:1] for k, v in predictions[1].items()}
            evaluator.update({1: prediction})
            evaluator.update({2: {}, 3: {}})
            evaluator.update({1: {k: v[:1] for k, v in predictions[1].items()}})
            _finish(evaluator)
        for candidate in candidates:
            _assert_equal(candidate, reference)
            candidate.accumulate()
            candidate.summarize()
            _assert_equal(candidate, reference)
    assert gt.dataset == original


@pytest.mark.parametrize("backend", ["faster_coco_eval", "ultrafast"])
def test_empty_predictions(backend):
    gt, predictions = _inputs()
    candidate = CocoEvaluator(gt, ["bbox", "segm", "keypoints"], backend=backend)
    reference = FasterCocoEvaluator(gt, candidate.iou_types)
    for evaluator in (reference, candidate):
        evaluator.update({image_id: {} for image_id in predictions})
        _finish(evaluator)
    _assert_equal(candidate, reference)


def test_backend_errors(monkeypatch):
    gt, _ = _inputs()
    with pytest.raises(ValueError, match="Unknown COCO backend"):
        CocoEvaluator(gt, ["bbox"], backend="invalid")
    original_import = builtins.__import__

    def no_ultrafast(name, *args, **kwargs):
        if name.startswith("ultrafast_pycocotools"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ultrafast)
    with pytest.raises(ModuleNotFoundError, match="ultrafast-pycocotools>=0.1.11"):
        CocoEvaluator(gt, ["bbox"], backend="ultrafast")
    assert CocoEvaluator(gt, ["bbox"]).backend == "faster_coco_eval"


def _distributed_worker(rank, init_file, empty_rank):
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        gt, predictions = _inputs()
        reference = FasterCocoEvaluator(gt, ["bbox", "segm", "keypoints"])
        reference.world_size = 1
        reference.update(predictions)
        _finish(reference)
        for backend in ("faster_coco_eval", "ultrafast"):
            candidate = CocoEvaluator(gt, reference.iou_types, backend=backend)
            if empty_rank:
                local = predictions if rank == 0 else {}
            elif rank == 0:
                local = {1: predictions[1]}
            else:
                local = dict(predictions)
                local[1] = {key: value[:1] for key, value in predictions[1].items()}
            candidate.update(local)
            _finish(candidate)
            _assert_equal(candidate, reference)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("empty_rank", [False, True])
def test_two_rank_merge(tmp_path, empty_rank):
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / "gloo"), empty_rank), nprocs=2, join=True)


def test_yaml_config_and_engine(tmp_path):
    from engine.solver.det_engine import evaluate

    gt, predictions = _inputs()
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(gt.dataset))
    path = tmp_path / "evaluation.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox", "segm"]},
                "val_dataloader": {
                    "type": "DataLoader",
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": 0,
                    "dataset": {
                        "type": "CocoDetection",
                        "img_folder": str(tmp_path),
                        "ann_file": str(annotations),
                        "transforms": None,
                    },
                },
            }
        )
    )
    cfg = YAMLConfig(str(path), **yaml_utils.parse_cli(["evaluator.backend=ultrafast"]))
    candidate = cfg.evaluator
    assert candidate.backend == "ultrafast"
    assert cfg.evaluator is candidate
    reference = CocoEvaluator(gt, ["bbox", "segm"])
    samples = torch.tensor([[1], [2], [3]])
    targets = [{"image_id": torch.tensor(i), "orig_size": torch.tensor([32, 32])} for i in (1, 2, 3)]
    for epoch in range(2):
        current = copy.deepcopy(predictions)
        if epoch:
            current[1] = {k: v[:1] for k, v in current[1].items()}

        def postprocess(outputs, sizes):
            return [current[int(row[0])] for row in outputs]

        results = []
        for evaluator in (reference, candidate):
            stats, returned = evaluate(
                torch.nn.Identity(),
                torch.nn.Identity(),
                postprocess,
                [(samples, targets)],
                evaluator,
                torch.device("cpu"),
            )
            assert returned is evaluator
            results.append(stats)
        assert results[0] == results[1]
        _assert_equal(candidate, reference)
