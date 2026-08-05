# DEIMv2 - ACES カスタム学習・評価ガイド

DEIMv2をACESのデータセットで学習・評価するためのガイドです。

---

## S3リソース構成

```
s3://aces-project-denver-11/
├── dataset/
│   └── （画像データ .jpg, YOLO用 .json）
│
├── datasets/
│   └── new-domain-phase11/
│       └── annotations/
│           ├── train_coco_clean.json
│           ├── val_coco_clean.json
│           └── test_coco_clean.json
│
└── models/
    └── deimv2_m_phase11_20260611/
        ├── weights/
        │   └── best_stg2.pth
        └── configs/
            ├── deimv2_hgnetv2_m_phase11.yml
            ├── phase11_detection.yml
            ├── runtime.yml
            └── base/
                ├── dataloader.yml
                ├── optimizer.yml
                └── deimv2.yml
```

---

## 環境セットアップ

```bash
git clone https://github.com/KotaIzawa/DEIMv2.git
cd DEIMv2
pip install -r requirements.txt
```

---

## S3からファイルを取得・配置

git clone後、以下のコマンドでファイルをダウンロードして所定の場所に配置してください。

```bash
# ① モデルの重み → DEIMv2/outputs/ 以下に配置
mkdir -p outputs/deimv2_m_phase11_20260611
aws s3 cp s3://aces-project-denver-11/models/deimv2_m_phase11_20260611/weights/best_stg2.pth \
  outputs/deimv2_m_phase11_20260611/best_stg2.pth

# ② configファイル → DEIMv2/configs/ 以下に上書き配置
aws s3 sync s3://aces-project-denver-11/models/deimv2_m_phase11_20260611/configs/ \
  configs/

# ③ アノテーションJSON → 任意の場所に配置（例: /datadrive/annotations/）
mkdir -p /datadrive/annotations
aws s3 sync s3://aces-project-denver-11/datasets/new-domain-phase11/annotations/ \
  /datadrive/annotations/

# ④ 画像データ → 任意の場所に配置（例: /datadrive/data/）
mkdir -p /datadrive/data
aws s3 sync s3://aces-project-denver-11/dataset/ \
  /datadrive/data/
```

ダウンロード後のリポジトリ構成：

```
DEIMv2/
├── configs/
│   ├── deimv2/
│   │   └── deimv2_hgnetv2_m_phase11.yml   ← ② でダウンロード
│   ├── dataset/
│   │   └── phase11_detection.yml           ← ② でダウンロード
│   ├── runtime.yml                         ← ② でダウンロード
│   └── base/
│       ├── dataloader.yml                  ← ② でダウンロード
│       ├── optimizer.yml                   ← ② でダウンロード
│       └── deimv2.yml                      ← ② でダウンロード
└── outputs/
    └── deimv2_m_phase11_20260611/
        └── best_stg2.pth                   ← ① でダウンロード

/datadrive/
├── data/          ← ④ 画像ファイル群
└── annotations/   ← ③ train/val/test_coco_clean.json
```

---

## 編集が必要なファイル

画像・アノテーションの配置場所を変えた場合は以下を編集してください：

**`configs/dataset/phase11_detection.yml`**

```yaml
train_dataloader:
  dataset:
    img_folder: /datadrive/data/              ← 画像フォルダのパスに変更
    ann_file: /datadrive/annotations/train_coco_clean.json  ← JSONのパスに変更

val_dataloader:
  dataset:
    img_folder: /datadrive/data/              ← 同上
    ann_file: /datadrive/annotations/val_coco_clean.json    ← 同上
```

---

## アノテーションJSONの生成（新規データの場合）

画像ごとのJSONアノテーションからCOCO形式に変換します。

```
データ構造:
/your/data/
├── image001.jpg
├── image001.json   ← アノテーション
├── image002.jpg
├── image002.json
...
/your/cfg/
├── label.txt       ← クラス名（1行1クラス）
├── train.txt       ← 学習画像のファイル名一覧（拡張子なし）
├── val.txt
└── test.txt
```

```bash
# convert_to_coco.py の DATA_DIR / CFG_DIR / OUT_DIR を編集してから実行
python3 convert_to_coco.py
```

---

## 学習

```bash
torchrun --nproc_per_node=1 train.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml
```

---

## 定量評価（クラス別 TP / FP / FN / mAP）

```bash
python3 tools/eval_per_class.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml \
  -r outputs/deimv2_m_phase11_20260611/best_stg2.pth \
  -a /your/annotations/test_coco_clean.json \
  -i /your/data/ \
  -l /your/cfg/label.txt
```

---

## 定性評価（BBox可視化・推論速度・GPUメモリ計測）

```bash
python3 tools/inference/torch_inf_vis2.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml \
  -r outputs/deimv2_m_phase11_20260611/best_stg2.pth \
  -d /your/data/ \
  -o /your/output/ \
  -l /your/cfg/label.txt \
  -a /your/annotations/test_coco_clean.json \
  --batch_size 1
```

---

## 動画推論

```bash
python3 tools/inference/torch_inf_video.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11.yml \
  -r outputs/deimv2_m_phase11_20260611/best_stg2.pth \
  -i /your/input.mp4 \
  -o /your/output.mp4 \
  -l /your/cfg/label.txt
```

---

## EX001 / EX002 / EX003 / EX004 / EX006 再現コマンド

実装はGitHubの該当コミット・ブランチを参照してください。

- GitHub: https://github.com/KotaIzawa/DEIMv2

以下はリポジトリルートで実行します。

```bash
cd /home/kotaizawa/DEIMv2
```

### 共通パス

```bash
PHASE11_5_VIDEO_DIR=/home/kotaizawa/phase_11_5_test3
PHASE11_5_FRAME_ALL=/datadrive/phase_11_5_test3/frames_05s_all/phase_11_5_test3_05s_frames
PHASE11_5_FRAME_SAMPLE=/datadrive/phase_11_5_test3/labelstudio_annotation_500
PHASE11_5_TEST_ANN=annotations/phase_11_5_test.json
PHASE11_5_PERSON_EVAL_ANN=annotations/phase_11_5_test3_person_eval_coco.json
LABELS_EX002=experiments/ex002/configs/dataset/phase11_5_labels.txt
INIT_WEIGHT=checkpoints/best_stg2.pth
```

### 動画のフレーム処理

0.5秒間隔で全フレームを切り出し、アノテーション用に500枚をサンプリングします。

```bash
python3 tools/dataset/extract_video_frames_for_annotation.py \
  --input-dir "$PHASE11_5_VIDEO_DIR" \
  --all-output-dir "$PHASE11_5_FRAME_ALL" \
  --sample-output-dir "$PHASE11_5_FRAME_SAMPLE" \
  --interval 0.5 \
  --duration 120 \
  --sample-count 500 \
  --quality 95
```

### 学習コマンド

EX001: RGB + 0.5秒前との差分、4ch。

```bash
python3 train_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_multiframe.yml \
  -t "$INIT_WEIGHT" \
  --seed 0 \
  --output-dir outputs/EX001_4ch_phase11_5_retry
```

EX002: RGB、3ch。

```bash
python3 train.py \
  -c experiments/ex002/configs/deimv2/deimv2_hgnetv2_m_phase11_5_EX002.yml \
  -t "$INIT_WEIGHT" \
  --seed 0 \
  --output-dir outputs/EX002_rgb_phase11_5
```

EX003: RGB + 0.5秒前/1.0秒前との差分、5ch。

```bash
python3 train_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_multiframe_ex003.yml \
  -t "$INIT_WEIGHT" \
  --seed 0 \
  --output-dir outputs/EX003_5ch_phase11_5
```

EX004: RGB + median background差分、4ch。

```bash
python3 train_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_background_ex004.yml \
  -t "$INIT_WEIGHT" \
  --seed 0 \
  --output-dir outputs/EX004_background_median_4ch_phase11_5
```

EX006: RGB + 0.5秒前/1.0秒前との差分 + median background差分、6ch。

```bash
python3 train_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_combined_ex006.yml \
  -t "$INIT_WEIGHT" \
  --seed 0 \
  --output-dir outputs/EX006_combined_6ch_phase11_5
```

### 推論コマンド（定量評価）

EX001。

```bash
python3 tools/eval_phase11_5_test3.py \
  --name EX001_4ch_phase11_5_retry \
  --mode ex001 \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_multiframe.yml \
  -r outputs/EX001_4ch_phase11_5_retry/best_stg2.pth \
  --src-annotation "$PHASE11_5_TEST_ANN" \
  --annotation "$PHASE11_5_PERSON_EVAL_ANN" \
  --image-dir "$PHASE11_5_FRAME_SAMPLE" \
  --all-frame-dir "$PHASE11_5_FRAME_ALL" \
  --output-dir outputs/phase_11_5_test3_eval \
  --conf-threshold 0.45 \
  --iou-threshold 0.5
```

EX002。

```bash
python3 tools/eval_phase11_5_test3.py \
  --name EX002_rgb_phase11_5 \
  --mode rgb \
  -c experiments/ex002/configs/deimv2/deimv2_hgnetv2_m_phase11_5_EX002.yml \
  -r outputs/EX002_rgb_phase11_5/best_stg2.pth \
  --src-annotation "$PHASE11_5_TEST_ANN" \
  --annotation "$PHASE11_5_PERSON_EVAL_ANN" \
  --image-dir "$PHASE11_5_FRAME_SAMPLE" \
  --all-frame-dir "$PHASE11_5_FRAME_ALL" \
  --output-dir outputs/phase_11_5_test3_eval \
  --conf-threshold 0.45 \
  --iou-threshold 0.5
```

EX003。

```bash
python3 tools/eval_phase11_5_test3.py \
  --name EX003_5ch_phase11_5 \
  --mode ex003 \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_multiframe_ex003.yml \
  -r outputs/EX003_5ch_phase11_5/best_stg2.pth \
  --src-annotation "$PHASE11_5_TEST_ANN" \
  --annotation "$PHASE11_5_PERSON_EVAL_ANN" \
  --image-dir "$PHASE11_5_FRAME_SAMPLE" \
  --all-frame-dir "$PHASE11_5_FRAME_ALL" \
  --output-dir outputs/phase_11_5_test3_eval \
  --conf-threshold 0.45 \
  --iou-threshold 0.5
```

EX004。

```bash
python3 tools/eval_phase11_5_test3.py \
  --name EX004_background_median_4ch_phase11_5 \
  --mode ex004 \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_background_ex004.yml \
  -r outputs/EX004_background_median_4ch_phase11_5/best_stg2.pth \
  --src-annotation "$PHASE11_5_TEST_ANN" \
  --annotation "$PHASE11_5_PERSON_EVAL_ANN" \
  --image-dir "$PHASE11_5_FRAME_SAMPLE" \
  --all-frame-dir "$PHASE11_5_FRAME_ALL" \
  --output-dir outputs/phase_11_5_test3_eval \
  --conf-threshold 0.45 \
  --iou-threshold 0.5
```

EX006。

```bash
python3 tools/eval_phase11_5_test3.py \
  --name EX006_combined_6ch_phase11_5 \
  --mode ex006 \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_combined_ex006.yml \
  -r outputs/EX006_combined_6ch_phase11_5/best_stg2.pth \
  --src-annotation "$PHASE11_5_TEST_ANN" \
  --annotation "$PHASE11_5_PERSON_EVAL_ANN" \
  --image-dir "$PHASE11_5_FRAME_SAMPLE" \
  --all-frame-dir "$PHASE11_5_FRAME_ALL" \
  --output-dir outputs/phase_11_5_test3_eval \
  --conf-threshold 0.45 \
  --iou-threshold 0.5
```

### 定性評価（推論結果のBBoxを動画へ描画）

EX001。

```bash
python3 tools/inference/torch_inf_video_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_multiframe.yml \
  -r outputs/EX001_4ch_phase11_5_retry/best_stg2.pth \
  -i "$PHASE11_5_VIDEO_DIR" \
  -o outputs/inference_phase11_5_test_EX001_3 \
  --threshold 0.45 \
  --frame-offsets 0.5 \
  --diff-source previous
```

EX002。

```bash
mkdir -p outputs/inference_phase11_5_test_EX002_rgb

for video in "$PHASE11_5_VIDEO_DIR"/*.mp4
do
  base=$(basename "$video" .mp4)
  python3 tools/inference/torch_inf_video.py \
    -c experiments/ex002/configs/deimv2/deimv2_hgnetv2_m_phase11_5_EX002.yml \
    -r outputs/EX002_rgb_phase11_5/best_stg2.pth \
    -i "$video" \
    -o "outputs/inference_phase11_5_test_EX002_rgb/${base}_thr0.45_rgb.mp4" \
    -l "$LABELS_EX002" \
    --threshold 0.45
done
```

EX003。

```bash
python3 tools/inference/torch_inf_video_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_multiframe_ex003.yml \
  -r outputs/EX003_5ch_phase11_5/best_stg2.pth \
  -i "$PHASE11_5_VIDEO_DIR" \
  -o outputs/inference_phase11_5_test_EX003_5ch_test3 \
  --threshold 0.45 \
  --frame-offsets 0.5 1.0 \
  --diff-source previous
```

EX004。

```bash
python3 tools/inference/torch_inf_video_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_background_ex004.yml \
  -r outputs/EX004_background_median_4ch_phase11_5/best_stg2.pth \
  -i "$PHASE11_5_VIDEO_DIR" \
  -o outputs/inference_phase_11_5_test3_EX004_bg_median_thr045 \
  --threshold 0.45 \
  --frame-offsets 0.5 \
  --diff-source background_median \
  --background-sample-interval 2.0 \
  --background-max-frames 120
```

EX006。

```bash
python3 tools/inference/torch_inf_video_multiframe.py \
  -c configs/deimv2/deimv2_hgnetv2_m_phase11_5_combined_ex006.yml \
  -r outputs/EX006_combined_6ch_phase11_5/best_stg2.pth \
  -i "$PHASE11_5_VIDEO_DIR" \
  -o outputs/inference_phase_11_5_test3_EX006_combined_6ch_thr045 \
  --threshold 0.45 \
  --frame-offsets 0.5 1.0 \
  --diff-source previous_and_background_median \
  --background-sample-interval 2.0 \
  --background-max-frames 120
```


