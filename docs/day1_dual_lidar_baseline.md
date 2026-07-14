# Day1：双 LiDAR 3D 检测 Baseline

## 1. 目标

Day1 在同一组 DAIR-V2X-C 验证样本上建立并比较两个单模态 3D 检测基线：

- Vehicle LiDAR-only：仅输入车辆端点云。
- Infrastructure LiDAR-only：仅输入路侧端点云。

两端的预测框均转换到 world 坐标系，并与 cooperative `label_world` 使用同一套匹配和评价逻辑。因此汇总结果可用于比较两类传感器视角的检测能力；它不等同于多传感器融合结果。

## 2. 已完成流程

1. 构建车辆端和路侧端 LiDAR 样本索引，检查点云、标注、标定和 cooperative world GT 文件。
2. 使用 OpenPCDet PointPillars 对两端点云执行真实模型推理，导出 `result.pkl`。
3. 将 `result.pkl` 转换为项目统一的 `predictions_*.json` 格式，并用 `validate_predictions_format.py` 校验。
4. 使用 `convert_lidar_predictions_to_world.py` 将预测从各自 LiDAR 坐标系转换到 world 坐标系。
5. 使用 `evaluate_and_visualize_lidar_baseline.py` 对两端执行统一 world GT 匹配、分类分析、AP/mAP 统计与可视化。
6. 使用 `generate_day1_lidar_baseline_summary.py` 汇总两端结果并生成对比图。

统一预测格式见 [predictions_json_format.md](predictions_json_format.md)。

## 3. 可复现命令

以下命令均应在项目上一级目录 `/home/gxy/projects/item3` 执行。

### 3.1 构建路侧端索引

```bash
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/scripts/build_infrastructure_lidar_baseline_index.py
```

### 3.2 导出并校验两端预测

车辆端预测导出命令见 [实验一文档](experiment1_vehicle_lidar_baseline.md)。路侧端真实模型导出示例：

```bash
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/export_infrastructure_lidar_predictions.py \
  --input-result-pkl /path/to/infrastructure/result.pkl \
  --source-format openpcdet-result-pkl \
  --score-threshold 0.1

conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/validate_predictions_format.py \
  dair_v2x_project/outputs/baselines/infrastructure_lidar/predictions_infrastructure_lidar.json
```

真实路侧模型必须使用 `prediction_type = openpcdet_infrastructure_lidar_pointpillars` 或 `official_infrastructure_pointpillars`。若使用标注构造工程验证预测，必须保留 `label_oracle_engineering_validation` 标记，不能将其解释为真实检测性能。

### 3.3 转换与评价

```bash
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/convert_lidar_predictions_to_world.py --sensor vehicle_lidar
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/convert_lidar_predictions_to_world.py --sensor infrastructure_lidar

conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/evaluate_and_visualize_lidar_baseline.py --baseline vehicle_lidar
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/evaluate_and_visualize_lidar_baseline.py --baseline infrastructure_lidar

conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/generate_day1_lidar_baseline_summary.py
```

## 4. 当前对比结果

当前结果均来自真实 OpenPCDet PointPillars 预测，使用 9 个验证样本、239 个 cooperative world GT 目标。

| Baseline | prediction_type | 预测数 | 匹配数 | FN | FP | Precision | Recall | F1 | BEV mAP@0.50 | 3D mAP@0.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vehicle LiDAR-only | `openpcdet_vehicle_lidar_pointpillars` | 205 | 70 | 169 | 135 | 0.3357 | 0.2987 | 0.3053 | 0.2085 | 0.1689 |
| Infrastructure LiDAR-only | `openpcdet_infrastructure_lidar_pointpillars` | 164 | 141 | 98 | 23 | 0.8582 | 0.5888 | 0.6979 | 0.3898 | 0.2741 |

路侧端在当前验证集上具有更高的 Precision、Recall、F1 和 AP/mAP。两端的 `classification_accuracy` 当前均为 1.0；应结合类别配置解读，不能单独视为完整的多类别分类能力证明。

## 5. 汇总输出说明

运行汇总脚本后生成：

- `outputs/baselines/day1_lidar_baseline_summary.csv`：两行对比表，适合论文表格或后处理。
- `outputs/baselines/day1_lidar_baseline_summary.json`：保留完整数值、输入 summary 路径、图像路径及 AP 缺失警告。
- `outputs/baselines/day1_lidar_summary_visualization/day1_lidar_precision_recall_f1.png`：两端 Precision、Recall 和 F1 的并列柱状图。
- `outputs/baselines/day1_lidar_summary_visualization/day1_lidar_fn_fp_comparison.png`：两端漏检数 FN 和误检数 FP 对比。
- `outputs/baselines/day1_lidar_summary_visualization/day1_lidar_count_error_comparison.png`：逐样本预测数量偏差的平均值对比。
- `outputs/baselines/day1_lidar_summary_visualization/day1_lidar_loc_error_comparison.png`：已匹配目标的平均 BEV 与 3D 中心定位误差对比。
- `outputs/baselines/day1_lidar_summary_visualization/day1_lidar_ap_comparison.png`：BEV mAP@0.50 与 3D mAP@0.50 对比。

AP summary 不存在或不可读时，汇总 CSV/JSON 的对应 AP 字段会写为 `N/A`，其他指标和所有图仍会生成。最终比较前应检查 `prediction_type`：出现 `label_oracle_engineering_validation` 时，该行仅能用于工程链路验证，不可和真实模型结果混合比较。

## 6. 版本控制范围

仓库提交代码、中文文档、统一格式规范以及轻量的 Day1 汇总 CSV/JSON。原始数据、OpenPCDet checkpoint、`result.pkl`、全量预测 JSON、逐帧可视化图和其他中间产物不提交；它们体积较大且可由上述流程重新生成。
