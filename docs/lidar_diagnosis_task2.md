# LiDAR 诊断 Task 2：原始坐标系检查

## 运行命令

当前 DAIR-V2X-C Full Dataset 的 PCD 是 `binary_compressed` 格式，需要在包含 Open3D 的环境中运行：

```bash
cd /home/gxy/projects/item3/dair_v2x_project
conda activate torch-gpu-test
python scripts/run_lidar_diagnosis_task2.py
```

脚本使用 Task 1 通过随机种子 `20260718` 选择的 20 个诊断样本，覆盖 `Bus、Car、Cyclist、Motorcyclist、Pedestrian、Trafficcone、Tricyclist、Truck、Van` 等原始类别，不重新运行模型，不修改原始数据。

## 输出

```text
outputs/diagnosis/sensor_frame_visualization/vehicle/
outputs/diagnosis/sensor_frame_visualization/infrastructure/
outputs/diagnosis/vehicle_local_frame_eval.csv
outputs/diagnosis/infrastructure_local_frame_eval.csv
outputs/diagnosis/vehicle_local_gt_world_alignment.csv
outputs/diagnosis/infrastructure_local_gt_world_alignment.csv
outputs/diagnosis/vehicle_prediction_world_conversion_check.csv
outputs/diagnosis/infrastructure_prediction_world_conversion_check.csv
outputs/diagnosis/lidar_diagnosis_task2_summary.json
```

每个传感器每帧生成两张图：`A_points_local_gt.png` 只显示原始点云和本地 GT，`B_points_local_gt_prediction.png` 额外显示原始坐标系 prediction。绿色为 GT/匹配 GT，红色为未匹配 GT，蓝色为匹配预测，橙色为误检预测。

## 评价含义

`*_local_frame_eval.csv` 使用与现有诊断评价相同的 5 m BEV 中心距离贪心匹配，统计本地坐标系中的 TP、FP、FN、Precision、Recall、F1、匹配框 BEV/3D IoU 和中心误差。它用于判断模型在原始 LiDAR 坐标下是否已经产生错误，不替代官方 AP。

`*_local_gt_world_alignment.csv` 将本地 GT 经过标定转换后与 cooperative `label_world` 比较。文件同时保存：

- `raw_*`：当前项目原始转换链的结果；
- `corrected_*`：对 infrastructure `virtuallidar_to_world.relative_error` 应用 `delta_x/delta_y` 后的结果。

`*_prediction_world_conversion_check.csv` 检查本地 prediction 转换后与现有 `predictions_world.json` 的中心和角点残差。残差接近 0 只能说明两个文件使用了同一转换实现，不能证明该转换使用了正确的标定修正。

## 验证结论

1. Vehicle LiDAR 的本地 GT 转 world 在大多数诊断帧中几乎完全对齐：20 帧平均中心误差为 `0.140 m`，平均 BEV IoU 为 `0.957`，但最大误差为 `4.680 m`，存在少量离群。车端没有发现统一性的整体平移/旋转错位证据；离群需要结合本地/world 标注对象集合和时间差分析。

2. Infrastructure LiDAR 的原始转换链存在明确问题。原代码只使用 `rotation + translation`，没有应用标定文件中的 `relative_error.delta_x/delta_y`。加入该修正后，20 帧平均 GT 中心误差从 `1.022 m` 降至 `0.452 m`，平均匹配 BEV IoU 从 `0.559` 提升至 `0.829`。这说明路侧 world 结果的异常至少部分来自标定误差修正遗漏。

3. 修正后仍存在离群帧，Infrastructure 最大误差为 `4.263 m`，因此不能把全部问题归结为 `relative_error`。还需要检查本地 GT 与 world GT 的目标集合差异、车路时间差，以及是否存在标注生成规则差异。

4. local prediction 与修复前 `predictions_world.json` 的原始转换残差约为 `6.4e-13 m`，说明旧导出文件和旧转换脚本在数值上自洽；但 Infrastructure 应用 `relative_error` 后与旧 world prediction 最大相差 `1.447 m`，证明“自洽”不等于“绝对坐标正确”。修复后 Full Dataset 已重新生成 `predictions_world.json`。

因此，当前两个 LiDAR baseline 的低性能不能再简单解释为“模型训练不好”：Infrastructure 的 world 标定修正遗漏已经修复，但仍有离群帧；Vehicle 的主要问题更接近原始 prediction 本身的检测质量、点云范围/体素配置或少量 GT 配对异常。后续应对修正后的离群样本做逐目标对照。
