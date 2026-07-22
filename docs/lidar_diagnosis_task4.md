# LiDAR 诊断 Task 4：评价器、配置与训练证据检查

## 运行命令

```bash
cd /home/gxy/projects/item3/dair_v2x_project
conda activate torch-gpu-test
python scripts/run_lidar_diagnosis_task4.py \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
```

输出：

```text
outputs/diagnosis/task4_evaluator_config_summary.json
```

## 验证结果

### 1. 评价器单元测试

5/5 通过：完全重合、完全不重合、两个预测匹配一个 GT、同位置错类别、无预测。测试确认了 TP/FP/FN、贪心唯一匹配和 `class_correct` 的基本行为。

### 2. 阈值扫描

当前评价使用 5 m BEV 中心匹配，阈值扫描结果如下：

| Baseline | score | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Vehicle LiDAR | 0.1 | 0.152 | 0.253 | 0.190 |
| Vehicle LiDAR | 0.3 | 0.339 | 0.163 | 0.220 |
| Vehicle LiDAR | 0.5 | 0.464 | 0.116 | 0.185 |
| Infrastructure LiDAR | 0.1 | 0.378 | 0.408 | 0.393 |
| Infrastructure LiDAR | 0.2 | 0.428 | 0.366 | 0.394 |
| Infrastructure LiDAR | 0.5 | 0.536 | 0.292 | 0.378 |

提高 score threshold 可以减少误检、提高 precision，但同时降低 recall。Infrastructure 在 `0.2` 附近取得略高的 F1；这只是当前中心距离诊断指标下的阈值敏感性，不替代官方 AP/mAP 选择。

### 3. 配置与结果审计

两套 baseline 都有 1243 个 Full Dataset common-val 预测样本，result.pkl 和运行时 YAML 均存在，预测类型分别为：

- `openpcdet_vehicle_lidar_pointpillars`
- `openpcdet_infrastructure_lidar_pointpillars`

### 4. 模型是否在完整数据集上训练足够轮次

当前证据不能证明这一点。两个训练日志都显示训练到 `80/80 epoch`，但日志中的 evaluation 数据量为 `9 samples`，不是 Full Dataset 的 1243 个样本。因此可以确认：

- checkpoint 确实来自 OpenPCDet PointPillars 训练流程；
- checkpoint 训练轮数为 80 epoch；
- 当前日志不能证明训练集是完整数据集，也不能证明已经在 Full Dataset 上训练了足够轮次；
- 当前 Full Dataset 结果是用该 checkpoint 对 1243 个验证样本进行推理，并不等于该模型在 1243 个样本对应的完整训练集上训练过。

因此正式实验前应补充 Full Dataset train split 的训练日志、训练样本数、配置中的 `INFO_PATH.train` 和训练 checkpoint 来源。没有这些证据，不应把当前模型称为“Full Dataset 充分训练模型”。
