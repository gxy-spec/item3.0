# DAIR-V2X-C 数据预处理、检测基线与可视化

本项目用于 DAIR-V2X-C 数据集的本地数据检查、索引构建、坐标转换、检测结果评价与可视化。

当前重点实验是：

```text
实验一：Vehicle LiDAR-only 3D 检测 baseline

vehicle-side LiDAR 点云
-> PointPillars / OpenPCDet 车辆端 3D 检测
-> vehicle LiDAR 坐标系下的 3D box
-> world 坐标系
-> 与 cooperative label_world GT 评价
-> 输出漏检、误检、数量误差、定位误差、分类指标、AP/mAP 和可视化
```

实验一的完整说明见：

- [实验一文档：Vehicle LiDAR-only 3D 检测 baseline](docs/experiment1_vehicle_lidar_baseline.md)

## 项目结构

- `config.py`：数据集根目录和输出路径配置。
- `scripts/`：数据检查、索引构建、统计和预处理脚本。
- `outputs/preprocessed/`：预处理索引和统计结果。
- `outputs/visual_check/`：数据可视化检查结果。
- `outputs/baselines/vehicle_lidar/`：实验一检测、转换、评价和可视化结果。
- `notes/`：官方 DAIR-V2X / MMDetection3D 环境搭建备注。

## 数据路径

默认会自动使用本机路径：

```text
DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure
```

如果数据集放在其他位置，可以设置：

```bash
export DAIR_V2X_C_ROOT=/path/to/cooperative-vehicle-infrastructure
```

## 环境说明

本项目目前用到两个主要环境：

```bash
conda activate dair-baseline
```

用于 item3.0 的数据预处理、坐标转换、评价和画图。需要 `numpy`、`pandas`、`matplotlib`、`tqdm` 等基础包。

```bash
conda activate torch-gpu-test
```

用于本机 OpenPCDet / PointPillars 训练和推理。当前本机 RTX 5060 Ti 可用此环境运行新版 PyTorch/CUDA。

> 注意：DAIR-V2X 官方旧版 MMDetection3D 栈在本机 RTX 5060 Ti 上不适配。旧 PyTorch 可以看到 GPU，但不能正常分配 CUDA tensor。因此本地实验一采用 OpenPCDet 路线。

## 常用数据检查脚本

在项目根目录 `/home/gxy/projects/item3` 下运行：

```bash
conda run -n dair-baseline python dair_v2x_project/scripts/01_check_structure.py
conda run -n dair-baseline python dair_v2x_project/scripts/02_inspect_data_info.py
conda run -n dair-baseline python dair_v2x_project/scripts/03_read_image.py
conda run -n dair-baseline python dair_v2x_project/scripts/04_read_pointcloud.py
conda run -n dair-baseline python dair_v2x_project/scripts/05_read_label.py
conda run -n dair-baseline python dair_v2x_project/scripts/06_read_calib.py
conda run -n dair-baseline python dair_v2x_project/scripts/07_build_pair_index.py
conda run -n dair-baseline python dair_v2x_project/scripts/08_file_size_stats.py
conda run -n dair-baseline python dair_v2x_project/scripts/09_visual_check_batch.py
conda run -n dair-baseline python dair_v2x_project/scripts/11_label_statistics.py
conda run -n dair-baseline python dair_v2x_project/scripts/12_build_device_modality_index.py
```

## 实验一快速流程

### 阶段 1：构建 item3.0 索引和 world GT

```bash
conda run -n dair-baseline python dair_v2x_project/scripts/07_build_pair_index.py
conda run -n dair-baseline python dair_v2x_project/scripts/14_build_world_gt_objects.py
conda run -n dair-baseline python dair_v2x_project/scripts/16_build_vehicle_lidar_baseline_index.py
```

主要输出：

```text
dair_v2x_project/outputs/preprocessed/dair_v2x_pair_index.json
dair_v2x_project/outputs/preprocessed/world_gt_objects.json
dair_v2x_project/outputs/preprocessed/vehicle_lidar_baseline_index.json
```

### 阶段 2：本机 OpenPCDet PointPillars 训练与推理

第一次使用 OpenPCDet 时，先编译：

```bash
conda activate torch-gpu-test
conda install -n torch-gpu-test -y numpy=2.0 scipy=1.14

cd /home/gxy/projects/item3/OpenPCDet

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="12.0"

python setup.py develop
```

验证环境：

```bash
cd /home/gxy/projects/item3/OpenPCDet
OPENPCDET_FORCE_NUMPY_VOXELIZER=1 python -c "import torch, pcdet, spconv; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print(torch.__version__, torch.cuda.is_available()); print(torch.zeros(1, device='cuda')); print('OpenPCDet env ok')"
```

训练 PointPillars：

```bash
cd /home/gxy/projects/item3/OpenPCDet/tools

OPENPCDET_FORCE_NUMPY_VOXELIZER=1 CUDA_VISIBLE_DEVICES=0 python train.py \
  --cfg_file cfgs/custom_models/pointpillar_custom_vehicle.yaml \
  --batch_size 2 \
  --epochs 80 \
  --workers 0 \
  --extra_tag veh_lidar_pointpillar_80e
```

如果显存不够，把 `--batch_size 2` 改成 `--batch_size 1`。

查找 checkpoint：

```bash
find /home/gxy/projects/item3/OpenPCDet/output -type f -name "*.pth" | sort
```

推理并导出 OpenPCDet `result.pkl`：

```bash
cd /home/gxy/projects/item3/OpenPCDet/tools

OPENPCDET_FORCE_NUMPY_VOXELIZER=1 OPENPCDET_SKIP_KITTI_EVAL=1 CUDA_VISIBLE_DEVICES=0 python test.py \
  --cfg_file cfgs/custom_models/pointpillar_custom_vehicle.yaml \
  --ckpt /home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 \
  --workers 0 \
  --extra_tag veh_lidar_pointpillar_80e_eval \
  --save_to_file
```

查找 `result.pkl`：

```bash
find /home/gxy/projects/item3/OpenPCDet/output -type f -name "result.pkl" | sort
```

当前正式 80 epoch 结果路径为：

```text
/home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e_eval/eval/epoch_80/val/default/result.pkl
```

### 阶段 3：导出为 item3.0 统一预测格式

```bash
conda run --no-capture-output -n dair-baseline python dair_v2x_project/export_vehicle_lidar_predictions.py \
  --input /home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e_eval/eval/epoch_80/val/default/result.pkl \
  --source-format openpcdet-result-pkl \
  --score-threshold 0.1
```

输出：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar.json
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar_summary.json
```

### 阶段 4：转换到 world 坐标并评价

```bash
conda run --no-capture-output -n dair-baseline python dair_v2x_project/convert_vehicle_lidar_predictions_to_world.py
conda run --no-capture-output -n dair-baseline python dair_v2x_project/check_predictions_world.py
conda run --no-capture-output -n dair-baseline python dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py
```

主要输出：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_world.json
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_world_summary.json
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_eval.csv
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_eval_summary.json
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_ap_summary.json
dair_v2x_project/outputs/baselines/vehicle_lidar/matched_prediction_gt_pairs.csv
dair_v2x_project/outputs/baselines/vehicle_lidar/per_class_metrics.csv
dair_v2x_project/outputs/baselines/vehicle_lidar/class_confusion_matrix.csv
dair_v2x_project/outputs/baselines/vehicle_lidar/visualization/
```

这些文件的详细解释见 [实验一文档](docs/experiment1_vehicle_lidar_baseline.md)。

## 当前 80 epoch 结果摘要

当前已经完成一轮 Vehicle LiDAR-only PointPillars 80 epoch baseline 评价。

```text
验证样本数: 9
预测框总数: 205
mean_precision: 0.3357
mean_recall: 0.2987
mean_f1: 0.3053
matched classification accuracy: 1.0000
BEV mAP@0.50: 0.2085
3D mAP@0.50: 0.1689
```

当前预测只包含 `Car` 类，因此分类准确率只表示“已匹配目标中 Car 是否被预测成 Car”，不能解释为完整的 Car/Pedestrian/Cyclist 三分类能力。

## 无检测器的工程自检

如果只想验证坐标转换和评价代码，可以用 vehicle-side label 构造 oracle-like prediction：

```bash
conda run -n dair-baseline python dair_v2x_project/export_vehicle_label_oracle_predictions.py
conda run -n dair-baseline python dair_v2x_project/convert_vehicle_lidar_predictions_to_world.py
conda run -n dair-baseline python dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py
```
