# Baseline Route Notes

## 当前目标

使用 DAIR-V2X Example 数据集先跑通四个单模态 3D 感知 baseline 的工程流程：

1. Vehicle LiDAR-only / PointPillars
2. Infrastructure LiDAR-only / PointPillars
3. Vehicle Camera-only / ImVoxelNet or official image baseline
4. Infrastructure Camera-only / ImVoxelNet or official image baseline

当前优先实现：

Vehicle LiDAR-only / PointPillars

## 项目分工

### item3.0 / dair_v2x_project

负责：
- 数据检查
- pair index
- world GT objects
- 预测结果导入
- 坐标转换到 world
- baseline 评价
- 漏检可视化

### DAIR-V2X official repo

负责：
- 官方 baseline 配置
- PointPillars / ImVoxelNet 模型
- 训练与推理
- 输出原始预测结果

## 当前检查目标

确认官方仓库中是否存在：

- configs/sv3d-veh
- configs/sv3d-veh/pointpillars
- configs/vic3d/late-fusion-pointcloud/pointpillars
- trainval_config_v.py
- trainval_config_i.py


## Vehicle LiDAR-only baseline 路线修正

当前数据集是 DAIR-V2X-C cooperative-vehicle-infrastructure，不是 DAIR-V2X-V single-vehicle-side。

因此 Vehicle LiDAR-only baseline 不优先使用：

configs/sv3d-veh/pointpillars/trainval_config.py

而优先使用：

configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v.py

原因：
- trainval_config_v.py 对应 DAIR-V2X-C 中的 vehicle-side detector
- trainval_config_i.py 后续可用于 Infrastructure LiDAR-only baseline
- 该路线更适合后续扩展到 Vehicle + Infrastructure LiDAR late fusion