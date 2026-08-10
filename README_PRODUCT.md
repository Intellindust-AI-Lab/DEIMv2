# DEIMv2 + DINOv2 for Product Detection

Our fork of [DEIMv2](README.md) that replaces the DINOv3 backbone with **DINOv2** (Apache 2.0
license, unlike DINOv3's restrictive one) and trains it on the product detection dataset.

## What was changed vs. upstream

- `engine/backbone/dinov2/` — vendored minimal DINOv2 ViT (from the official Meta repo, Apache 2.0),
  loads official checkpoints with `strict=True`.
- `engine/backbone/dinov3_adapter.py` — the STA adapter accepts `dinov2_*` backbone names and
  rescales its input so the patch-14 token grid matches the stride-16 feature pyramid
  (640 px image -> 560 px ViT input -> 40x40 tokens).
- `engine/solver/` — `amp_dtype` config key selects the autocast dtype. The DINOv2 configs use
  **bfloat16**: fp16 overflows on DINOv2's high-norm activations after enough fine-tuning and
  corrupts the weights with NaNs (happened at epoch 30 with fp16).
- `engine/misc/dist_utils.py` — `reduce_dict` gathers the union of loss keys across ranks before
  the all-reduce. The loss dict has data-dependent keys (`loss_ddf*`), which deadlocked NCCL.
- `engine/data/dataset/coco_eval.py` — evaluation results are gathered on rank 0 only instead of
  every rank (the 80k-image val set caused host-RAM OOM), and the final stats are broadcast back.
- `engine/data/dataset/coco_dataset.py` — optional `remove_empty_annotations` flag drops
  background-only images from training.
- `train.py` — ClearML logging is on by default; opt out with `--avoid-clearml`.
- `tools/deployment/convert_onnx_fp16.py`, `tools/benchmark/onnx_fps.py` — new scripts (below).

## Environment

Python 3.12 virtualenv at `.venv` (managed with rye/uv), torch 2.5.1+cu124. Set up with `rye sync`.
The deployment extras (`onnx`, `onnxsim`, `onnxruntime-gpu`, `onnxconverter-common`) were installed
ad hoc with `uv pip install` and are **not** in the lockfile; re-add them after a fresh sync.

## Dataset

COCO-format detection dataset at `/home/rbezen/Projects/datasets/product_detection/`:

```
product_detection/
├── annotations/train.json    # 341,932 images / 960,230 boxes (225,915 images with objects)
├── annotations/val.json      #  79,651 images / 205,415 boxes ( 50,566 images with objects)
└── aldi/ coophu/ dm/ ...     # image folders, one per retailer, referenced by file_name
```

7 classes: `product, cellphone, wallet, credit_card, other_item, inseparable, bag`
(`num_classes: 7`, no COCO category remapping). About a third of the images are background-only;
training runs with `remove_empty_annotations=True` to skip them (val keeps everything). Paths and
classes are defined in `configs/dataset/product_detection.yml`.

## Pretrained backbone weights

Official DINOv2 checkpoints (with registers) go into `./ckpts/`:

```bash
wget -P ckpts https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth
wget -P ckpts https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
```

## Configs

| Config | Backbone | Scale |
|---|---|---|
| `configs/deimv2/deimv2_dinov2_product.yml` | DINOv2 ViT-S/14 + registers | DEIMv2-L recipe (hidden_dim 224) |
| `configs/deimv2/deimv2_dinov2_b_product.yml` | DINOv2 ViT-B/14 + registers | DEIMv2-X recipe (hidden_dim 256) |

Key settings (ViT-S config): 48 epochs (augmentation stops at 36), `total_batch_size: 40`
(5 GPUs x 8), input 640x640, ImageNet normalization (DINOv2 expects it), backbone fine-tuned at
LR 1.25e-5 with the rest at 5e-4, `amp_dtype: bfloat16`.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --master_port=7777 --nproc_per_node=5 train.py \
  -c configs/deimv2/deimv2_dinov2_product.yml \
  -u train_dataloader.dataset.remove_empty_annotations=True
```

- **Resume** after an interruption: add `-r outputs/deimv2_dinov2_product/last.pth`.
- **ClearML** is on by default (project `DEIMv2`, task named after the config file); add
  `--avoid-clearml` to disable, `--clearml-task <name>` to rename.
- **Evaluate only**: `train.py --test-only -c <config> -r <checkpoint>`.

### Output folder (`outputs/deimv2_dinov2_product/`)

| File | Meaning |
|---|---|
| `last.pth` | full training state, written every epoch (for resume) |
| `checkpointNNNN.pth` | snapshot every `checkpoint_freq` (5) epochs |
| `best_stg1.pth` / `best_stg2.pth` | best AP checkpoint during / after the augmented phase |
| `log.txt` | one JSON line per epoch: train losses + 12 COCO eval stats |
| `summary/` | TensorBoard events (mirrored to ClearML) |
| `eval/` | raw `COCOeval` precision/recall arrays |

Checkpoints store model + EMA + optimizer state (~500 MB); **the EMA weights are what gets
evaluated and shipped**. Use `best_stg2.pth` for deployment.

### Result (ViT-S, 48 epochs)

Best AP50-95 **0.4794** / AP50 **0.6429** on the 80k-image val set (epoch 39, kept in
`best_stg2.pth`).

## Export to ONNX

```bash
python tools/deployment/export_onnx.py --check --simplify \
  -c configs/deimv2/deimv2_dinov2_product.yml \
  -r outputs/deimv2_dinov2_product/best_stg2.pth
```

Writes `best_stg2.onnx`: EMA weights, re-param blocks fused, postprocessor baked in. Inputs are
`images` (Nx3x640x640, float32, ImageNet-normalized) and `orig_target_sizes` (Nx2, int64); outputs
are `labels`, `boxes` (xyxy in original image coordinates), `scores` (300 per image). Batch is
dynamic; resolution is fixed at 640. Do not shrink the batch-32 dummy input in the export script to
1 — a `memory.shape[0] > 1` branch in the decoder would bake in the wrong path.

Float16 variant (halves the file to 63 MiB; also fixes Cast nodes the stock converter mistypes):

```bash
python tools/deployment/convert_onnx_fp16.py -i outputs/deimv2_dinov2_product/best_stg2.onnx
```

Verified against fp32 on real val images: every detection above 0.4 confidence recovered with
IoU > 0.95, no NaNs.

## Benchmarks (A10, 640x640, batch 1)

```bash
python tools/benchmark/get_info.py -c configs/deimv2/deimv2_dinov2_product.yml   # FLOPs/params
CUDA_VISIBLE_DEVICES=0 python tools/benchmark/onnx_fps.py --onnx outputs/deimv2_dinov2_product/best_stg2.onnx
```

| Metric | fp32 | fp16 |
|---|---|---|
| MACs / FLOPs | 48.03 GMACs / 96.23 GFLOPs | same |
| Parameters | 32.59 M | same |
| ONNX file size | 125.6 MiB | 63.0 MiB |
| FPS batch 1 (onnxruntime CUDA) | 41.7 (24 ms) | 20.0 (50 ms) |
| Throughput batch 8 | 51.0 img/s | 50.0 img/s |

fp16 is *slower* under onnxruntime's CUDA provider: the graph's many CPU-assigned shape ops force
extra Cast/Memcpy at every boundary, which dominates at small batches. For real fp16 gains build a
TensorRT engine (`trtexec --onnx=... --fp16`, TensorRT >= 10.6); expect roughly 2-3x the fp32 FPS.
For onnxruntime deployments, use the fp32 model.

## Gotchas learned the hard way

1. **Never train DINOv2 with fp16 AMP** — activations overflow around epoch 30 and every weight
   becomes NaN. `amp_dtype: bfloat16` is set in both DINOv2 configs; keep it.
2. A NaN crash leaves a `NaN.pth` debug dump in the repo root (written by `det_engine.py`); the
   weights inside tell you whether corruption happened before the crash. `last.pth` from the
   previous epoch is usually clean — resume from it after fixing the cause.
3. `decoder.anchors` containing `inf` in checkpoints is **by design** (invalid-anchor mask), not
   corruption. The fp16 ONNX converter clamps it to 10000, which is saturating-equivalent.
4. The ONNX export needs the position-embedding cache in
   `engine/backbone/dinov2/vision_transformer.py`: DINOv2's antialiased bicubic resize
   (`aten::_upsample_bicubic2d_aa`) has no ONNX symbolic in torch 2.5. The cache turns it into a
   graph constant (the export script's warm-up forward fills it).
