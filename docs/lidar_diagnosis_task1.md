# LiDAR 诊断 Task 1：实验冻结与样本配对

## 目标

本任务不重新推理、不改变模型阈值，也不修改原始数据。它冻结当前 Full Dataset 的 Vehicle LiDAR-only 与 Infrastructure LiDAR-only 实验状态，并建立一组可审计的诊断样本，为后续检查坐标链、预测导出、标定和模型误差提供固定输入。

## 运行

```bash
cd /home/gxy/projects/item3/dair_v2x_project

python3 scripts/prepare_lidar_diagnosis_task1.py
```

默认读取 `outputs/full_baselines/` 中的 `full_common_val` 结果，选择 20 个诊断样本，并计算关键工件的 SHA256。快速预览可使用 `--no-hash`，但正式诊断应使用默认哈希模式。

抽样使用可复现随机种子，默认值为 `20260718`。更换 seed 会得到另一组样本，但相同 seed 始终得到相同结果：

```bash
python3 scripts/prepare_lidar_diagnosis_task1.py --seed 20260719 --sample-count 20
```

## 输出

```text
outputs/diagnosis/baseline_snapshot.json
outputs/diagnosis/lidar_diagnosis_manifest.json
outputs/diagnosis/lidar_sample_pairing.csv
```

`baseline_snapshot.json` 固定 Git commit、工作区状态、运行配置、checkpoint、`result.pkl`、local/world 预测、评价文件、AP 文件及其 SHA256。它区分以下三个控制量：OpenPCDet NMS IoU、AP IoU 和 5 m 中心匹配距离。

`lidar_diagnosis_manifest.json` 保存完整的验证结论和嵌套样本信息。每个样本包含 cooperative ID、车端/路侧帧 ID、点云/标注/标定路径、预测 JSON 及 sample key、两侧点云时间戳、GT 距离统计、两条 baseline 的帧级指标和选择原因。

`lidar_sample_pairing.csv` 是便于人工检查的扁平表。`vehicle_frame_id` 与 `infrastructure_frame_id` 数值不同是数据集的正常配对形式，不能作为错误；配对有效性由固定的 `full_common_val_pair_manifest.json` 和 `cooperative/data_info.json` 共同验证。

## 抽样规则

抽样由 seed 控制，既可复现又可以更换样本。样本来源包括：

- 3 个固定随机对照帧；
- 4 个类别组合更丰富的帧；
- 按稀有原始标签类别进行覆盖，例如 `Pedestrian`、`Cyclist`、`Motorcyclist`、`Truck`、`Bus`、`Van`、`Trafficcone` 和 `Tricyclist`；
- Vehicle 明显差于 Infrastructure 的帧；
- Infrastructure 明显差于 Vehicle 的帧；
- 高 FN 率与高 FP/GT 帧；
- 两端均较远的场景；
- GT 稀疏与 GT 稠密场景；
- 车路点云时间差较大的场景；
- 如有重复，再补充按 sample ID 排序的确定性覆盖帧。

类别使用原始 local label 类型统计，不强行映射到 AP 的三类 taxonomy。它们只用于定性和坐标链诊断，不替代 1243 帧全量的统计结果。

## 当前验证结论

当前正式运行确认：

- `full_common_val` 共 1243 个配对，全部可访问；
- Vehicle 与 Infrastructure 的索引、预测、评价记录均完整；
- 未发现重复 vehicle ID 或 infrastructure ID；
- 跨传感器点云绝对时间差中位数为 14.852 ms，95 分位为 28.541 ms，最大为 29.975 ms；
- 时间差已记录为潜在混杂因素，尚不能据此认定它是误差来源；
- 当前 Vehicle 配置使用范围 `[0, -39.68, -3, 69.12, 39.68, 1]`、体素 `[0.16, 0.16, 4]`；Infrastructure 配置从基础配置继承范围 `[0, -50, -5, 220, 50, 5]`、体素 `[0.32, 0.32, 10]`。

下一任务应在这 20 帧上执行 local label -> world 与 cooperative `label_world` 的对齐检查，并分别报告车端 `lidar -> novatel -> world` 与路侧 `virtuallidar -> world` 的回环误差。
