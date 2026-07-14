# 实验一：Vehicle LiDAR-only 3D 检测 Baseline

## 1. 实验目标

本实验建立一个只使用车辆端 LiDAR 点云的 3D 检测 baseline。

完整流程是：

```text
vehicle-side LiDAR 点云
-> PointPillars / OpenPCDet 车辆端 3D 检测
-> 输出 vehicle LiDAR 坐标系下的 3D box、类别、置信度
-> 利用标定参数转换到 world 坐标系
-> 与 cooperative label_world GT 进行匹配和评价
-> 输出检测指标、分类指标、AP/mAP 和可视化结果
```

实验关注的问题包括：

- 车辆端 LiDAR-only 模型能检测到多少 cooperative `label_world` 目标。
- 漏检、误检、数量误差有多大。
- 已匹配目标的定位误差是多少。
- 已匹配目标的类别是否预测正确。
- 各类别的 TP、FP、FN、precision、recall、F1 如何。
- AP/mAP 在 BEV 和 3D IoU 下表现如何。
- 预测结果在 world 坐标系中与 GT 的空间关系是否合理。

## 2. 已完成的工作

### 2.1 数据检查与索引构建

使用 item3.0 脚本完成了 DAIR-V2X-C 数据结构检查、样本配对索引构建和 world 坐标 GT 构建。

关键输出：

```text
dair_v2x_project/outputs/preprocessed/dair_v2x_pair_index.json
dair_v2x_project/outputs/preprocessed/world_gt_objects.json
dair_v2x_project/outputs/preprocessed/vehicle_lidar_baseline_index.json
```

这些文件用于确认每个 vehicle-side LiDAR 样本与 cooperative label、标定文件、split 信息之间的对应关系。

### 2.2 Vehicle LiDAR-only 检测模型

本机使用 OpenPCDet 训练并推理 PointPillars 模型。

模型输入：

```text
vehicle-side LiDAR point cloud
```

模型输出：

```text
vehicle LiDAR 坐标系下的 3D box
类别 pred_class
置信度 pred_score
```

当前正式结果来自：

```text
/home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e_eval/eval/epoch_80/val/default/result.pkl
```

### 2.3 坐标转换

OpenPCDet 输出的 3D box 位于 vehicle LiDAR 坐标系。

本项目将其转换为 world 坐标，转换链路为：

```text
vehicle_lidar
-> lidar_to_novatel
-> novatel_to_world
-> world
```

转换后输出：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_world.json
```

### 2.4 评价与可视化

评价脚本：

```text
dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py
```

评价对象：

```text
prediction: predictions_world.json
GT: cooperative label_world
```

评价方式：

- 先在 world 坐标系下进行预测框与 GT 的空间匹配。
- 中心点 BEV 距离小于阈值的目标视为匹配候选。
- 当前匹配阈值为 `5.0 m`。
- 匹配方式为按预测置信度排序后的贪心匹配。
- 匹配本身不强制类别一致。
- 匹配后再统计 `pred_class == gt_class` 是否成立。

这样可以同时得到检测层面的 TP/FP/FN，以及分类层面的混淆矩阵。

## 3. 当前结果概览

当前 80 epoch Vehicle LiDAR-only PointPillars 结果：

```text
验证样本数: 9
预测框总数: 205
GT 总数: 239
空间匹配成功数: 70
matched_cls_correct: 70
matched_cls_wrong: 0
classification_accuracy: 1.0000
mean_precision: 0.3357
mean_recall: 0.2987
mean_f1: 0.3053
BEV mAP@0.50: 0.2085
3D mAP@0.50: 0.1689
```

重要注意事项：

```text
当前预测只包含 Car 类。
因此 classification_accuracy=1.0 不能解释为完整三分类准确率。
它只表示：在已空间匹配成功的目标中，GT 为 Car 的目标也被预测成 Car。
```

如果后续模型支持 `Car / Pedestrian / Cyclist` 多类别输出，则同一套评价脚本会自动计算完整多类别分类指标。

## 4. 结果目录总览

实验一所有结果位于：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/
```

主要文件：

```text
predictions_vehicle_lidar.json
predictions_vehicle_lidar_summary.json
predictions_world.json
predictions_world_summary.json
vehicle_lidar_only_eval.csv
vehicle_lidar_only_eval_summary.json
vehicle_lidar_only_ap_summary.json
vehicle_lidar_only_eval_details.json
matched_prediction_gt_pairs.csv
per_class_metrics.csv
class_confusion_matrix.csv
visualization/
```

下面逐一解释每个文件。

## 5. `predictions_vehicle_lidar.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar.json
```

含义：

这是从 OpenPCDet `result.pkl` 导出的统一预测格式，仍然在 vehicle LiDAR 坐标系下。

该文件已经被固定为所有 baseline 的统一预测输出模板。完整标准见：

```text
dair_v2x_project/docs/predictions_json_format.md
```

统一模板的样本级必填字段为：

```text
sample_id
coordinate_system
prediction_type
pred_objects
```

`vehicle_id` 不是必填字段。`sensor` 和 `source_id` 是推荐字段，缺失时只给 warning，不作为错误。

每个样本记录通常包含：

- `sample_id`：样本 ID。
- `vehicle_id`：车辆端样本 ID。
- `coordinate_system`：当前预测坐标系，一般为 vehicle LiDAR。
- `prediction_type`：预测来源，例如 `openpcdet_vehicle_lidar_pointpillars`。
- `pred_objects`：该帧所有预测目标。

每个预测目标通常包含：

- `class`：预测类别。
- `score`：检测置信度。
- `center_lidar` 或等价字段：vehicle LiDAR 坐标下的 3D box 中心。
- `box_lidar`：vehicle LiDAR 坐标下的 box 尺寸和朝向。

用途：

该文件是检测器输出与 item3.0 坐标转换之间的中间格式。

## 6. `predictions_vehicle_lidar_summary.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar_summary.json
```

含义：

记录 vehicle LiDAR 坐标预测导出的摘要信息。

关键字段：

- `input`：输入的 OpenPCDet `result.pkl` 路径。
- `source_format`：输入格式，当前为 `openpcdet-result-pkl`。
- `score_threshold`：导出时使用的置信度阈值。
- `num_samples`：样本数量。
- `num_objects`：导出的预测框总数。
- `prediction_types`：预测来源类型。

当前结果：

```text
num_samples = 9
num_objects = 205
score_threshold = 0.1
```

## 7. `predictions_world.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_world.json
```

含义：

这是将预测框从 vehicle LiDAR 坐标系转换到 world 坐标系后的结果。

每个预测目标包含：

- `class`：预测类别。
- `score`：置信度。
- `center_world`：world 坐标下的中心点 `[x, y, z]`。
- `box_world`：world 坐标下的 3D box 信息。
- `corners_bev_world`：BEV 平面四个角点。
- `corners_3d_world`：如果可用，则为 3D 八角点。
- `z_min` / `z_max`：box 在 z 方向上的上下边界。

用途：

后续所有评价和可视化都基于此文件完成。

## 8. `predictions_world_summary.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_world_summary.json
```

关键字段：

- `num_input_samples`：输入预测样本数。
- `num_output_samples`：成功转换到 world 的样本数。
- `num_skipped_samples`：因为缺少标定等原因跳过的样本数。
- `num_world_objects`：成功转换后的预测目标总数。
- `coordinate_transform`：坐标转换链路。

当前结果：

```text
num_input_samples = 9
num_output_samples = 9
num_skipped_samples = 0
num_world_objects = 205
coordinate_transform = vehicle_lidar -> lidar_to_novatel -> novatel_to_world -> world
```

## 9. `vehicle_lidar_only_eval.csv`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_eval.csv
```

含义：

逐帧评价结果表。每一行对应一个 sample。

字段解释：

- `sample_id`：样本 ID。
- `vehicle_id`：车辆端样本 ID。
- `num_gt`：该帧 cooperative `label_world` GT 数量。
- `num_pred`：该帧预测框数量。
- `num_match`：空间匹配成功的预测-GT 对数量。
- `false_negative`：漏检数量，等于未匹配 GT 数量。
- `false_positive`：误检数量，等于未匹配预测数量。
- `count_error`：数量误差，等于 `abs(num_pred - num_gt)`。
- `mean_loc_error_bev`：匹配目标在 BEV 平面的平均中心距离，单位米。
- `mean_loc_error_3d`：匹配目标在 3D 空间的平均中心距离，单位米。
- `recall`：该帧召回率，`num_match / num_gt`。
- `precision`：该帧精确率，`num_match / num_pred`。
- `f1`：该帧 F1-score，`2 * precision * recall / (precision + recall)`。
- `matched_cls_correct`：匹配目标中类别预测正确的数量。
- `matched_cls_wrong`：匹配目标中类别预测错误的数量。
- `classification_accuracy`：匹配目标分类准确率，`matched_cls_correct / num_match`。
- `gt_car`：该帧 GT 中 Car 数量。
- `gt_pedestrian`：该帧 GT 中 Pedestrian 数量。
- `gt_cyclist`：该帧 GT 中 Cyclist 数量。
- `pred_car`：该帧预测中 Car 数量。
- `pred_pedestrian`：该帧预测中 Pedestrian 数量。
- `pred_cyclist`：该帧预测中 Cyclist 数量。

注意：

这里的 `num_match` 是基于中心距离的空间匹配，不是 IoU AP 里的 TP。

## 10. `matched_prediction_gt_pairs.csv`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/matched_prediction_gt_pairs.csv
```

含义：

逐个匹配对的明细表。每一行对应一个成功匹配的 prediction-GT pair。

字段解释：

- `sample_id`：样本 ID。
- `pred_index`：该帧中预测框的索引。
- `gt_index`：该帧中 GT 框的索引。
- `pred_class`：预测类别。
- `gt_class`：GT 类别。
- `pred_score`：预测置信度。
- `center_distance_bev`：预测中心与 GT 中心的 BEV 距离，单位米。
- `center_distance_3d`：预测中心与 GT 中心的 3D 距离，单位米。
- `class_correct`：预测类别是否与 GT 类别一致。

用途：

该文件用于定位具体哪些目标匹配成功、哪些匹配对存在类别错误、置信度和定位误差如何。

## 11. `per_class_metrics.csv`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/per_class_metrics.csv
```

含义：

按类别汇总检测指标。

字段解释：

- `class`：类别名，包含 `Car`、`Pedestrian`、`Cyclist`。
- `gt_count`：该类别 GT 总数。
- `pred_count`：该类别预测总数。
- `TP`：该类别 true positive 数量。
- `FP`：该类别 false positive 数量。
- `FN`：该类别 false negative 数量。
- `precision`：该类别精确率，`TP / (TP + FP)`。
- `recall`：该类别召回率，`TP / (TP + FN)`。
- `f1`：该类别 F1-score。

当前结果：

```text
Car:
  gt_count = 239
  pred_count = 205
  TP = 70
  FP = 135
  FN = 169
  precision = 0.3415
  recall = 0.2929
  f1 = 0.3153

Pedestrian / Cyclist:
  当前 GT 和预测均为 0，因此 precision / recall / f1 为空。
```

## 12. `class_confusion_matrix.csv`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/class_confusion_matrix.csv
```

含义：

匹配成功目标的类别混淆矩阵。

矩阵定义：

- 行：GT 类别。
- 列：预测类别。
- 单元格数值：空间匹配成功后，GT 为行类别且预测为列类别的数量。

当前结果：

```text
GT Car -> Pred Car = 70
其他类别均为 0
```

注意：

混淆矩阵只统计匹配成功的目标。漏检目标不会进入混淆矩阵，误检目标也不会进入混淆矩阵。

## 13. `vehicle_lidar_only_eval_summary.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_eval_summary.json
```

含义：

实验一评价的总摘要文件。

关键字段：

- `num_samples`：参与评价的样本数量。
- `output_csv`：逐帧评价 CSV 路径。
- `details_json`：详细匹配记录 JSON 路径。
- `matched_pairs_csv`：匹配对明细 CSV 路径。
- `per_class_metrics_path`：各类别指标 CSV 路径。
- `confusion_matrix_path`：混淆矩阵 CSV 路径。
- `visualization_world_dir`：逐帧 world 对比图目录。
- `visualization_summary_dir`：汇总图目录。
- `matching_method`：匹配方法说明。
- `distance_threshold_meter`：中心距离匹配阈值。
- `gt_classes_present`：GT 中实际出现的类别。
- `pred_classes_present`：预测中实际出现的类别。
- `matched_cls_correct`：匹配目标中类别正确数量。
- `matched_cls_wrong`：匹配目标中类别错误数量。
- `classification_accuracy`：匹配目标分类准确率。
- `per_class_metrics`：各类别指标的 JSON 形式。
- `mean_precision`：逐帧 precision 的均值。
- `mean_recall`：逐帧 recall 的均值。
- `mean_f1`：逐帧 F1 的均值。
- `map_bev_iou_0.50`：BEV mAP@0.50。
- `map_3d_iou_0.50`：3D mAP@0.50。
- `evaluation_scope_note`：当前评价范围说明。

当前重要字段：

```text
mean_precision = 0.3357
mean_recall = 0.2987
mean_f1 = 0.3053
classification_accuracy = 1.0000
matched_cls_correct = 70
matched_cls_wrong = 0
map_bev_iou_0.50 = 0.2085
map_3d_iou_0.50 = 0.1689
```

`evaluation_scope_note` 中已经说明当前结果为 Car-only detection evaluation。

## 14. `vehicle_lidar_only_ap_summary.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_ap_summary.json
```

含义：

基于 IoU PR 曲线计算的 AP/mAP 结果。

与 `vehicle_lidar_only_eval.csv` 中的中心距离匹配不同，这里使用 IoU 匹配：

- `bev`：BEV 2D box IoU。
- `3d`：3D box IoU。
- IoU 阈值：`0.25`、`0.50`、`0.70`。

每个类别每个阈值包含：

- `num_gt`：该类别 GT 总数。
- `num_pred`：该类别预测总数。
- `num_tp`：在该 IoU 阈值下的 TP 数量。
- `ap`：该类别 AP。

`map` 字段表示各类别 AP 的平均值。当前由于只有 Car 类有效，mAP 实际等于 Car AP。

当前结果示例：

```text
BEV AP@0.50 Car = 0.2085
3D AP@0.50 Car = 0.1689
```

## 15. `vehicle_lidar_only_eval_details.json`

路径：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/vehicle_lidar_only_eval_details.json
```

含义：

逐帧详细评价信息，适合程序化调试。

每个样本包含：

- `sample_id`
- `vehicle_id`
- `label_world_path`
- `matches`
- `matched_pairs`
- `missed_gt_indices`
- `false_pred_indices`
- `metrics`

如果需要追踪某一帧某一个 GT 是否被匹配，可以查看此文件。

## 16. 可视化结果目录

可视化目录：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/visualization/
```

分为两类：

```text
visualization/world_gt_pred_compare/
visualization/summary/
```

## 17. 逐帧 world 对比图

路径示例：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/visualization/world_gt_pred_compare/sample_000_015372_world_compare.png
```

每张图对应一个样本。

图中元素含义：

- 绿色实线框：匹配成功的 GT。
- 红色实线框：漏检 GT。
- 蓝色虚线框：匹配成功的预测框。
- 橙色虚线框：误检预测框。
- 灰色虚线：匹配预测中心与 GT 中心的连线。
- 文本 `GT-Car`：GT 类别。
- 文本 `P-Car 0.xx`：预测类别和置信度。

标题中包含：

- `GT`：GT 数量。
- `Pred`：预测数量。
- `Match`：空间匹配数量。
- `Missed`：漏检数量。
- `False Pred`：误检数量。
- `Matched classification accuracy`：该帧已匹配目标的分类准确率。
- `Precision`：该帧 precision。
- `Recall`：该帧 recall。

用途：

用来直观看预测框是否和 cooperative `label_world` 对齐，定位误差是否合理，以及漏检/误检出现在什么区域。

## 18. `metric_summary_bar.png`

路径：

```text
visualization/summary/metric_summary_bar.png
```

含义：

整体平均指标柱状图。

包含：

- `false_negative`：平均每帧漏检数量。
- `false_positive`：平均每帧误检数量。
- `count_error`：平均数量误差。
- `mean_loc_error_bev`：平均 BEV 定位误差。
- `recall`：平均召回率。
- `precision`：平均精确率。
- `f1`：平均 F1。
- `classification_accuracy`：匹配目标平均分类准确率。

用途：

快速查看 baseline 的总体表现。

## 19. `per_sample_count_curve.png`

路径：

```text
visualization/summary/per_sample_count_curve.png
```

含义：

逐帧数量曲线。

曲线包括：

- GT count：每帧 GT 数量。
- Prediction count：每帧预测数量。
- Matched count：每帧匹配成功数量。

用途：

判断模型是否系统性少检或多检，以及哪些帧数量偏差较大。

## 20. `per_sample_error_curve.png`

路径：

```text
visualization/summary/per_sample_error_curve.png
```

含义：

逐帧错误数量曲线。

曲线包括：

- False negative：漏检数量。
- False positive：误检数量。
- Count error：数量误差。

用途：

观察错误是否集中在某几帧。

## 21. `per_sample_localization_error.png`

路径：

```text
visualization/summary/per_sample_localization_error.png
```

含义：

逐帧平均 BEV 定位误差。

只统计匹配成功的目标。

单位：

```text
meter
```

用途：

判断坐标转换是否稳定，以及模型定位是否在某些帧明显变差。

## 22. `class_confusion_matrix.png`

路径：

```text
visualization/summary/class_confusion_matrix.png
```

含义：

匹配成功目标的类别混淆矩阵图。

读法：

- 纵轴为 GT 类别。
- 横轴为预测类别。
- 数字越大表示该 GT 类别被预测成该预测类别的次数越多。

如果出现非对角线数值，说明存在类别混淆。

当前结果只有 `Car -> Car`，因为预测和 GT 当前都只出现 Car。

## 23. `per_class_precision_recall_f1.png`

路径：

```text
visualization/summary/per_class_precision_recall_f1.png
```

含义：

各类别 precision、recall、F1 柱状图。

横轴：

```text
Car / Pedestrian / Cyclist
```

纵轴：

```text
precision / recall / F1
```

用途：

比较不同类别的检测表现。

当前 Pedestrian 和 Cyclist 没有 GT 和预测，因此对应指标为空，在图中显示为 0。

## 24. `gt_pred_class_distribution.png`

路径：

```text
visualization/summary/gt_pred_class_distribution.png
```

含义：

GT 和预测的类别数量分布图。

每个类别有两根柱：

- GT：GT 中该类别数量。
- Prediction：预测中该类别数量。

用途：

查看模型输出类别分布是否和 GT 分布一致。

当前图中 Car 有数量，Pedestrian / Cyclist 为 0。

## 25. `score_distribution_by_match.png`

路径：

```text
visualization/summary/score_distribution_by_match.png
```

含义：

预测置信度分布图。

分为两类：

- Matched predictions：匹配成功预测的 score 分布。
- False predictions：误检预测的 score 分布。

用途：

观察置信度是否能区分正确检测和误检。

如果误检预测也集中在高分区域，说明单纯提高 score threshold 可能会损失召回，但未必能明显减少误检。

如果误检集中在低分区域，可以考虑提高 score threshold。

## 26. 指标公式汇总

### 26.1 空间匹配指标

```text
num_match = 成功匹配的 prediction-GT 对数量
false_negative = num_gt - num_match
false_positive = num_pred - num_match
count_error = abs(num_pred - num_gt)
```

### 26.2 Precision / Recall / F1

逐帧：

```text
precision = num_match / num_pred
recall = num_match / num_gt
F1 = 2 * precision * recall / (precision + recall)
```

按类别：

```text
precision_class = TP_class / (TP_class + FP_class)
recall_class = TP_class / (TP_class + FN_class)
F1_class = 2 * precision_class * recall_class / (precision_class + recall_class)
```

### 26.3 定位误差

```text
center_distance_bev = sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2)
center_distance_3d = sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2 + (pred_z - gt_z)^2)
```

逐帧定位误差为所有匹配目标距离的平均值。

### 26.4 分类准确率

```text
class_correct = pred_class == gt_class
classification_accuracy = matched_cls_correct / num_match
```

注意：

只对匹配成功的目标统计分类准确率。

### 26.5 AP / mAP

AP/mAP 使用 IoU PR 匹配计算。

```text
BEV AP: 使用 BEV 2D box IoU
3D AP: 使用 3D box IoU
IoU thresholds: 0.25 / 0.50 / 0.70
```

AP 与中心距离匹配的 precision/recall 是两套不同评价：

- 中心距离匹配用于直观统计漏检、误检、定位误差和分类明细。
- IoU AP 用于标准检测性能评价。

## 27. 当前结果的解释边界

当前使用的结果为 Vehicle LiDAR-only PointPillars baseline，预测类别只有 Car。

因此：

- 可以解释 Car 类检测性能。
- 可以解释 Car 类定位误差。
- 可以解释 Car-only 情况下的漏检、误检、AP/mAP。
- 不应解释为完整多类别分类模型。
- `classification_accuracy=1.0` 不代表完整三分类准确率。

后续如果模型输出 Car、Pedestrian、Cyclist 三类，本评价脚本已经支持：

- 多类别 matched pair 记录。
- 多类别 per-class TP/FP/FN。
- 多类别 confusion matrix。
- 多类别 precision/recall/F1。

## 28. 复现实验一的最短命令

如果已经有 OpenPCDet `result.pkl`，只需要运行：

```bash
cd /home/gxy/projects/item3

conda run --no-capture-output -n dair-baseline python dair_v2x_project/export_vehicle_lidar_predictions.py \
  --input /home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e_eval/eval/epoch_80/val/default/result.pkl \
  --source-format openpcdet-result-pkl \
  --score-threshold 0.1

conda run --no-capture-output -n dair-baseline python dair_v2x_project/convert_vehicle_lidar_predictions_to_world.py
conda run --no-capture-output -n dair-baseline python dair_v2x_project/check_predictions_world.py
conda run --no-capture-output -n dair-baseline python dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py
```

运行结束后查看：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/
```
