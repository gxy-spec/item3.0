# Day2：本机 RTX 5060 Ti 的 Camera-only ImVoxelNet 小样本推理

本文档记录 Vehicle Camera-only 与 Infrastructure Camera-only 的本机推理流程。目标是对 `example-cooperative-split-data.json` 中的验证集各 9 帧运行真实 ImVoxelNet checkpoint，导出统一预测 JSON，转换到 world 坐标，并复用项目已有的统一评价与可视化。

## 1. 实现原则

DAIR-V2X 原始 Camera ImVoxelNet 依赖 `mmdet3d 0.17.1 + PyTorch 1.10`。该旧运行时不能在 RTX 5060 Ti 的 `sm_120` 架构上运行。

本流程使用独立环境 `dair-camera-5060`：

```text
PyTorch 2.11.0+cu128
CUDA Toolkit 12.8
MMDetection3D 1.4.0
MMDetection 3.3.0
MMCV 2.1.0
```

模型结构按新版 MMDetection3D 的 ImVoxelNet 配置表达，但保持 DAIR-V2X 官方 checkpoint 的网络尺寸、anchor 范围、体素尺寸和类别集合。已验证 Vehicle 与 Infrastructure checkpoint 均与新版模型完全匹配：

```text
checkpoint state_dict keys: 394
model state_dict keys: 394
missing keys: 0
unexpected keys: 0
```

因此这是使用官方权重的本机兼容运行时推理，不是 label oracle，也不是从头训练的模型。

## 2. 坐标说明

模型输入只有单张 Camera 图像，因此属于 Camera-only；但 ImVoxelNet 使用相机标定将图像特征投影到 3D voxel，`Anchor3DHead` 最终输出的是 LiDAR 坐标系 3D 框。

所以预测 JSON 的约定是：

| Baseline | `sensor` | `coordinate_system` | world 转换链 |
| --- | --- | --- | --- |
| Vehicle Camera-only | `vehicle_camera` | `vehicle_lidar` | `vehicle_lidar -> lidar_to_novatel -> novatel_to_world -> world` |
| Infrastructure Camera-only | `infrastructure_camera` | `infrastructure_lidar` | `virtuallidar -> virtuallidar_to_world -> world` |

`coordinate_system` 描述的是输出 box 的坐标系，不是模型输入模态。这使 Camera 与 LiDAR baseline 能使用相同的 world 坐标匹配和 cooperative `label_world` 评价。

## 3. 数据准备

Infrastructure 的 `training/image_2`、`training/calib`、`training/label_2` 与 `ImageSets` 由 Day2 Step1 脚本生成。Vehicle 原有 Windows 转换脚本写入过占位标定，因此必须用官方标定重建 Vehicle 的派生 `training/calib`；否则 Camera 特征会被投影到错误的 3D 位置。

下列命令不会修改原始 image、label、calib JSON 或 `data_info.json`：

```bash
conda activate dair-camera-5060
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd /home/gxy/projects/item3/dair_v2x_project

python prepare_vehicle_camera_imvoxelnet_format.py
python prepare_infrastructure_camera_imvoxelnet_format.py

python scripts/build_camera_imvoxelnet_infos.py --sensor vehicle_camera
python scripts/build_camera_imvoxelnet_infos.py --sensor infrastructure_camera
```

生成的 `mmdet3d_camera_infos_train.pkl` 和 `mmdet3d_camera_infos_val.pkl` 位于两个 sensor-side 数据目录中，仅包含 MMD3D 推理所需的图像路径、`cam2img`、`lidar2cam` 和 `lidar2img`。

## 4. 运行真实推理

```bash
cd /home/gxy/projects/item3/dair_v2x_project

CUDA_VISIBLE_DEVICES=0 python run_camera_imvoxelnet_inference.py \
  --sensor vehicle_camera \
  --split val

CUDA_VISIBLE_DEVICES=0 python run_camera_imvoxelnet_inference.py \
  --sensor infrastructure_camera \
  --split val
```

`--max-samples N` 可用于只运行前 N 帧。例如 `--max-samples 1` 用于环境试跑。`--score-threshold` 默认是 `0.1`，与当前模型测试配置保持一致。

输出：

```text
outputs/baselines/vehicle_camera/predictions_vehicle_camera.json
outputs/baselines/vehicle_camera/predictions_vehicle_camera_summary.json
outputs/baselines/infrastructure_camera/predictions_infrastructure_camera.json
outputs/baselines/infrastructure_camera/predictions_infrastructure_camera_summary.json
```

预测类型分别为：

```text
mmdet3d_1x_imvoxelnet_vehicle_camera
mmdet3d_1x_imvoxelnet_infrastructure_camera
```

它们明确表示使用新版 MMD3D 运行时加载 ImVoxelNet 的 Camera-only checkpoint，不会与 LiDAR PointPillars 或工程 label oracle 混淆。

## 5. 格式、world 转换与评价

```bash
python validate_predictions_format.py \
  outputs/baselines/vehicle_camera/predictions_vehicle_camera.json \
  outputs/baselines/infrastructure_camera/predictions_infrastructure_camera.json

python convert_lidar_predictions_to_world.py --sensor vehicle_camera
python convert_lidar_predictions_to_world.py --sensor infrastructure_camera

MPLBACKEND=Agg python evaluate_and_visualize_lidar_baseline.py --baseline vehicle_camera
MPLBACKEND=Agg python evaluate_and_visualize_lidar_baseline.py --baseline infrastructure_camera
```

转换和评价输出分别位于：

```text
outputs/baselines/vehicle_camera/predictions_world.json
outputs/baselines/vehicle_camera/vehicle_camera_eval_summary.json
outputs/baselines/vehicle_camera/visualization/

outputs/baselines/infrastructure_camera/predictions_world.json
outputs/baselines/infrastructure_camera/infrastructure_camera_eval_summary.json
outputs/baselines/infrastructure_camera/visualization/
```

`predictions_world.json` 与 LiDAR baseline 的同名文件保持统一结构：每个目标包括 `center_world`、`box_world`、`corners_bev_world`、`corners_3d_world`、`z_min` 和 `z_max`。

## 6. 当前小样本结果的解释边界

当前 example 验证集仅有 9 帧，且模型 checkpoint 是 Car-only。因此：

- 分类准确率只能说明空间匹配成功框的 Car 类一致性，不能解释为完整三类别分类能力。
- 评价使用 cooperative `label_world` 和当前通用 evaluator 的 5m 中心距离匹配规则。
- 结果可用于比较 Vehicle Camera-only 与 Infrastructure Camera-only 的检测覆盖与定位表现，但不能替代完整测试集的正式 benchmark。

## 7. 原始图像 3D 检测结果可视化

世界坐标评价图用于检查预测框与 cooperative `label_world` 的空间关系；相机模态还应查看 3D 框是否合理地落在原始图像目标上。`visualize_camera_3d_predictions.py` 会将 Camera-only ImVoxelNet 在 LiDAR 坐标系输出的预测框投影回对应相机图像。

脚本不会重新计算匹配关系，而是读取已有的 `<baseline>_eval_details.json`。因此图像框的匹配、漏检和误检状态与 world 坐标评价 CSV/JSON 一致。运行前应已完成推理、world 转换和评价。

```bash
conda activate dair-camera-5060
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/gxy/projects/item3/dair_v2x_project

python visualize_camera_3d_predictions.py --baseline vehicle_camera
python visualize_camera_3d_predictions.py --baseline infrastructure_camera
```

输出目录：

```text
outputs/baselines/vehicle_camera/visualization/image_3d_detection/
outputs/baselines/infrastructure_camera/visualization/image_3d_detection/
```

每张图同时叠加 cooperative world GT 与模型预测框：绿色为匹配成功的 GT，红色为漏检 GT（FN），青色为匹配成功的预测框，橙色为误检预测框（FP），紫红色为位置匹配但类别错误的预测框。

每个 baseline 还会生成：

```text
visualization/summary/camera_3d_detection_contact_sheet.png
visualization/summary/camera_3d_visualization_summary.json
```

联系表便于浏览全部验证样本；summary 记录写出的图像数、GT/预测/匹配总数、每种框的投影绘制数量和跳过样本。若 3D 框完全在相机视锥外或相机后方，它仍计入 world 评价，但会作为 `hidden_*` 记录而不显示在图像上。
- 生成的可视化、pkl info 和 baseline 预测属于实验产物；代码同步时不应默认提交大体积中间文件。
