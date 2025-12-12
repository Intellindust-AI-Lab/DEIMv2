<h2 align="center">
  DEIMv2 On Pose Estimation
</h2>


  
## 1. Results Comparation

| Model | size | Dataset | mAP | AP50 | #Params | GFLOPs | Latency-4090 (ms) | Latency-T4 (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:------------:|:------------:|
| **YOLO11x-pose** | 640 | COCO | **69.5** | **91.1** | 58.8M | 202.8 |     2.18->3.24    |     12.7->16.1     |
| **DETRPose-X** | 640 | COCO | **73.3** | **90.5** | 73.3M | 239.5 |     4.18     |     20.7     |
| **DEIMv2_Pose-X** | 640 | COCO | **74.2** | **91.8** | 53.9M | 175.1 |     3.31     |     16.8     |

```
Note: The latency values for YOLO11x-pose (2.18->3.24 ms and 12.7->16.1 ms) include TensorRT's EfficientNMS operator processing time to provide more realistic inference metrics for practical deployment scenarios.
```



**Get Our Model**

| Model | mAP | AP50 | #Params | GFLOPs | Checkpoint | Log |
| :---: | :---: | :---: | :---: | :---: | :--------: | :---: |
| **DEIMv2_Pose-L** | **72.0** | **90.7** | 35.0M | 111.0 | [Google](https://drive.google.com/file/d/1H5_ejiuPU8w8uiOJBhFrNVlV5Vp-MZJh/view?usp=drive_link) | [Google](https://drive.google.com/file/d/1XtWjLRu4hBV3W202BbzExjqnIsXmpdwd/view?usp=drive_link) |
| **DEIMv2_Pose-X** | **74.2** | **91.8** | 53.9M | 175.1 | [Google](https://drive.google.com/file/d/1Irq2hszYAwE_gfW-RgdJtya7GahqcEbF/view?usp=drive_link) | [Google](https://drive.google.com/file/d/1TAibYqUiFi57JqCn_8glRFTMMcn0UaEW/view?usp=drive_link) |


## 2. Quick Start

### 2.1 Build Conda Environmnet

```shell
# You can use PyTorch 2.6.1 or 2.5.1. We have not tried other versions, but we recommend that the PyTorch version be 2.0 or higher.

conda create -n deimv2_pose python=3.11 -y
conda activate deimv2_pose
pip install -r requirements.txt
```

### 2.2 Data Preparation

<details open>
<summary> 2.1.1 COCO2017 Dataset </summary>

Follow the steps below to prepare COCO dataset:

1. Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) or [COCO](https://cocodataset.org/#download).
2. Modify paths in [coco_pose.yml](./configs/dataset/coco_pose.yml)

    ```yaml
    train_dataloader:
        img_folder: /data/COCO2017/train2017/
        ann_file: /data/COCO2017/annotations/person_keypoints_train2017.json
    val_dataloader:
        img_folder: /data/COCO2017/val2017/
        ann_file: /data/COCO2017/annotations/person_keypoints_val2017.json
    ```

</details>


### 2.3 Pretrained Weight Preparation

<details open>

- **DEIMv2_Pose-X**: We use DINOv3-S-Plus, as backbone, you can download them following the guide in [DINOv3](https://github.com/facebookresearch/dinov3).

Place dinov3 and vits into ./ckpts folder as:

```shell
ckpts/
├── dinov3_vits16plus.pth
└── ...
```

</details>


## 3. Usage
<details open>
<summary> 3.1 COCO2017 </summary>

1. Training
    ```shell
    # for DINOv3-based variants
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2_pose/deimv2_pose_dinov3_${model}_coco.yml --use-amp --seed=42
    ```

2. Testing
    ```shell
    # for DINOv3-based variants
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2_pose/deimv2_pose_dinov3_${model}_coco.yml --test-only -r model.pth
    ```

3. Tuning
    ```shell
    # for ViT-based variants
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2_pose/deimv2_pose_dinov3_${model}_coco.yml --use-amp --seed=42 -t model.pth

    # for HGNetv2-based variants
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2_pose/deimv2_pose_hgnetv2_${model}_coco.yml --use-amp --seed=42 -t model.pth
    ```
</details>
