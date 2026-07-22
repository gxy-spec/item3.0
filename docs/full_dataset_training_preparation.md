# Full Dataset 重新训练数据与配置准备

## 官方训练轮次

依据 DAIR-V2X 仓库中同步的官方配置：PointPillars 使用 `20` epoch，ImVoxelNet 使用 `12` epoch。官方文档主要提供预训练 checkpoint 和评测流程，因此本项目没有继续使用旧的 `80 epoch` 自定义小样本配置。

## 已完成

### Vehicle LiDAR

```text
/home/gxy/projects/item3/OpenPCDet/data/full_training_vehicle/
```

- train：4813 样本
- full_common_val：1243 样本
- train 有效 3D 标注：51674 个
- 配置：[pointpillar_vehicle_full_20e.yaml](../outputs/full_training/configs/pointpillar_vehicle_full_20e.yaml)

### Infrastructure LiDAR

```text
/home/gxy/projects/item3/OpenPCDet/data/full_training_infrastructure/
```

- train：4813 样本
- full_common_val：1243 样本
- train 有效 3D 标注：76272 个
- 配置：[pointpillar_infrastructure_full_20e.yaml](../outputs/full_training/configs/pointpillar_infrastructure_full_20e.yaml)

### Vehicle Camera

- train：4813 样本，40507 个有效 3D 实例
- val：1783 样本
- full_common_val：1243 样本
- info 文件位于 Full Dataset `vehicle-side/mmdet3d_camera_infos_*.pkl`
- 配置：[imvoxelnet_vehicle_full_train.py](../configs/imvoxelnet_vehicle_full_train.py)

## Infrastructure Camera 当前阻塞

已生成 `full_common_val` 的路侧 Camera info，但 train/val 不能完整生成。原因是 cooperative 映射要求的部分路侧图像在当前 Full Dataset 解压目录中不存在。检查结果见：

```text
outputs/full_training/camera_training_data_readiness.json
```

当前统计：

- train：4813 个请求样本，缺失图像 1471 个
- val：1783 个请求样本，缺失图像 540 个
- full_common_val：使用 cooperative 映射后的样本可生成

因此不能在 Infrastructure Camera 缺失图像的情况下直接训练，否则训练集会被静默改变，无法与其他 baseline 公平比较。需要先补齐对应的 infrastructure-side image 文件或重新核对 Full Dataset 解压包。

## 生成脚本

- `scripts/prepare_full_opencdet_lidar.py`
- `scripts/build_full_opencdet_infos.py --for-training --epochs 20`
- `scripts/build_camera_imvoxelnet_infos.py --with-labels`
- `scripts/check_full_camera_training_readiness.py`

原始 DAIR-V2X 数据未被修改；生成的点云训练文件、info 和配置均放在独立派生目录中。
