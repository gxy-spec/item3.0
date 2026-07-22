# DAIR-V2X-C Full Dataset：四个真实单模态 Baseline

## 1. 范围与原则

Full Dataset 根目录为：

```text
/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full
```

整理后的标准数据根目录为：

```text
/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
```

当前数据包包含官方 cooperative split 中全部 `train=4813` 和 `val=1783` 样本，但不包含 cooperative test 样本。由于其中 540 个 val 配对缺少路侧图像，四个单模态 baseline 的公平共同验证集是 `full_common_val=1243`；官方全部 1783 val 保留给 LiDAR-only 的覆盖分析，不可与四模态共同表混用。

所有结果保存到：

```text
outputs/full_baselines/<baseline>/
```

不覆盖 `outputs/baselines/` 中的小样本结果。LiDAR 只接受真实 OpenPCDet `result.pkl`，Camera 只接受真实 ImVoxelNet checkpoint 推理；任一模型失败时必须保留命令日志，不能用 label、模拟框或工程验证预测填补。

## 2. 数据整理与检查

先生成清单，再创建四个目录软链接。软链接将标准目录的 `image`、`velodyne` 指向并列解压目录，不移动、不复制、不删除原始数据：

```bash
cd /home/gxy/projects/item3/dair_v2x_project

python scripts/prepare_full_dataset_layout.py \
  --dataset-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full

python scripts/prepare_full_dataset_layout.py \
  --dataset-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full \
  --apply
```

清单和快照位于：

```text
outputs/full_baselines/dataset_preparation/full_dataset_layout_file_manifest.jsonl
outputs/full_baselines/dataset_preparation/full_dataset_layout_manifest.json
outputs/full_baselines/dataset_preparation/full_dataset_layout_backup_snapshot.json
```

构建共同的 train/val ImageSets：

```bash
python scripts/build_full_dataset_splits.py \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
```

该脚本使用官方 `cooperative-split-data.json`。若一个 `vehicle_id` 对应多个路侧帧，默认选择 `cooperative/data_info.json` 的首个记录，并在 `full_split_summary.json` 中完整记录。随后会额外生成四模态文件均可用的 `full_common_val` ImageSets 和 pair manifest；该策略是可复现的工程约定，后续应进行 `first/last` 敏感性分析。

执行完整性检查：

```bash
python scripts/check_full_dataset_readiness.py \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
```

输出 `outputs/full_baselines/full_dataset_readiness_summary.json`。只有 `ready=true` 且引用错误为 0 时，才进入全量推理。

## 3. Vehicle / Infrastructure LiDAR-only

先生成与小样本目录隔离的 Full val OpenPCDet 输入。以下命令在 `torch-gpu-test` 环境运行；PCD 解码使用真实 x/y/z/intensity，不产生预测。

```bash
conda activate torch-gpu-test
cd /home/gxy/projects/item3/dair_v2x_project

python scripts/build_full_lidar_baseline_indexes.py \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure \
  --pair-manifest outputs/full_baselines/dataset_preparation/full_common_val_pair_manifest.json \
  --split full_common_val

python scripts/prepare_full_opencdet_lidar.py \
  --sensor vehicle_lidar --split full_common_val \
  --pair-manifest outputs/full_baselines/dataset_preparation/full_common_val_pair_manifest.json \
  --target-root /home/gxy/projects/item3/OpenPCDet/data/full_vehicle

python scripts/prepare_full_opencdet_lidar.py \
  --sensor infrastructure_lidar --split full_common_val \
  --pair-manifest outputs/full_baselines/dataset_preparation/full_common_val_pair_manifest.json \
  --target-root /home/gxy/projects/item3/OpenPCDet/data/full_infrastructure
```

生成 OpenPCDet info 和运行时配置。这里复用当前确认过的 PointPillars 网络配置，只替换数据路径和 val info；请在最终报告中记录 checkpoint SHA256 和训练来源。当前 checkpoint 是小样本训练模型，因此“Full val 推理”不应被表述为“Full Dataset 训练结果”。

```bash
python scripts/build_full_opencdet_infos.py \
  --sensor vehicle_lidar \
  --data-path /home/gxy/projects/item3/OpenPCDet/data/full_vehicle \
  --split full_common_val \
  --output-config outputs/full_baselines/vehicle_lidar/runtime_configs/pointpillar_full_vehicle.yaml

python scripts/build_full_opencdet_infos.py \
  --sensor infrastructure_lidar \
  --data-path /home/gxy/projects/item3/OpenPCDet/data/full_infrastructure \
  --split full_common_val \
  --output-config outputs/full_baselines/infrastructure_lidar/runtime_configs/pointpillar_full_infrastructure.yaml
```

在 OpenPCDet 中运行真实 checkpoint，并保留日志：

```bash
cd /home/gxy/projects/item3/OpenPCDet/tools

OPENPCDET_FORCE_NUMPY_VOXELIZER=1 OPENPCDET_SKIP_KITTI_EVAL=1 CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --cfg_file /home/gxy/projects/item3/dair_v2x_project/outputs/full_baselines/vehicle_lidar/runtime_configs/pointpillar_full_vehicle.yaml \
  --ckpt /home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_vehicle/veh_lidar_pointpillar_80e/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 --workers 0 --extra_tag full_val --save_to_file \
  2>&1 | tee /home/gxy/projects/item3/dair_v2x_project/outputs/full_baselines/vehicle_lidar/inference.log
```

路侧端使用 `pointpillar_full_infrastructure.yaml` 和 `infra_lidar_pointpillar_80e/ckpt/checkpoint_epoch_80.pth`，日志写入 `outputs/full_baselines/infrastructure_lidar/inference.log`。只有 `result.pkl` 成功生成后，才导出预测：

```bash
cd /home/gxy/projects/item3/dair_v2x_project

python export_vehicle_lidar_predictions.py \
  --input /path/to/vehicle/result.pkl --source-format openpcdet-result-pkl \
  --output outputs/full_baselines/vehicle_lidar/predictions_vehicle_lidar.json

python export_infrastructure_lidar_predictions.py \
  --index outputs/full_baselines/infrastructure_lidar/infrastructure_lidar_baseline_index.json \
  --input-result-pkl /path/to/infrastructure/result.pkl \
  --output outputs/full_baselines/infrastructure_lidar/predictions_infrastructure_lidar.json
```

## 4. Vehicle / Infrastructure Camera-only

Camera-only 直接从标准 raw image/calib 构建 Full val info，不复制图像。使用 `dair-camera-5060` 环境：

```bash
conda activate dair-camera-5060
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/gxy/projects/item3/dair_v2x_project

python scripts/build_camera_imvoxelnet_infos.py \
  --sensor vehicle_camera --splits full_common_val --input-format raw_dair \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure

python scripts/build_camera_imvoxelnet_infos.py \
  --sensor infrastructure_camera --splits full_common_val --input-format raw_dair \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
```

运行真实 ImVoxelNet checkpoint，并将错误持久化到日志：

```bash
CUDA_VISIBLE_DEVICES=0 python run_camera_imvoxelnet_inference.py \
  --sensor vehicle_camera --split full_common_val \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure \
  --output outputs/full_baselines/vehicle_camera/predictions_vehicle_camera.json \
  2>&1 | tee outputs/full_baselines/vehicle_camera/inference.log

CUDA_VISIBLE_DEVICES=0 python run_camera_imvoxelnet_inference.py \
  --sensor infrastructure_camera --split full_common_val \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure \
  --output outputs/full_baselines/infrastructure_camera/predictions_infrastructure_camera.json \
  2>&1 | tee outputs/full_baselines/infrastructure_camera/inference.log
```

## 5. 统一 world 转换、评价和可视化

每个 baseline 均复用当前小样本的转换和评价实现。下面以 `vehicle_camera` 为例，替换 sensor、输入预测和输出目录即可：

```bash
DATA_ROOT=/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
ROOT=outputs/full_baselines/vehicle_camera

python convert_lidar_predictions_to_world.py \
  --sensor vehicle_camera --data-root "$DATA_ROOT" \
  --input "$ROOT/predictions_vehicle_camera.json" \
  --output "$ROOT/predictions_world.json" \
  --summary "$ROOT/predictions_world_summary.json"

MPLBACKEND=Agg python evaluate_and_visualize_lidar_baseline.py \
  --baseline vehicle_camera --data-root "$DATA_ROOT" --output-root "$ROOT" \
  --world-vis-mode none

python visualize_camera_3d_predictions.py \
  --baseline vehicle_camera --data-root "$DATA_ROOT" \
  --prediction-path "$ROOT/predictions_vehicle_camera.json" \
  --details-path "$ROOT/eval_details.json" \
  --output-dir "$ROOT/visualization"
```

LiDAR baseline 不运行最后一条图像投影命令，但同样生成 world BEV、逐样本曲线、分类指标、AP/mAP、FN、FP、数量误差和定位误差。每个 Full baseline 的紧凑输出为：

```text
predictions_<baseline>.json
predictions_world.json
eval.csv
eval_summary.json
ap_summary.json
matched_prediction_gt_pairs.csv
per_class_metrics.csv
class_confusion_matrix.csv
visualization/
```

最后汇总四路真实结果：

```bash
python generate_full_single_modal_summary.py
```

输出 `outputs/full_baselines/full_single_modal_summary.csv`。缺少真实模型结果时对应行会标为 `missing` 并保留错误日志路径，不会生成虚假指标。

## 6. 完整测试集可视化

对 1243 帧逐帧绘制 world 图会产生大量难以审阅的文件。因此 Full Dataset 评价推荐使用 `--world-vis-mode none`，再由下列脚本从每个 `eval.csv` 自动选择有诊断价值的代表帧：高 FN、高 FP、高 3D 定位误差、低 Recall 和高 F1。选择原因会写入 CSV，不会依赖人工挑图。

四条 baseline 的 world 转换和评价都完成后运行：

```bash
conda activate dair-baseline
cd /home/gxy/projects/item3/dair_v2x_project

python generate_full_baseline_visualizations.py \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure \
  --max-representatives 12
```

输出：

```text
outputs/full_baselines/visualization/summary/full_precision_recall_f1.png
outputs/full_baselines/visualization/summary/full_fn_fp.png
outputs/full_baselines/visualization/summary/full_ap.png
outputs/full_baselines/visualization/summary/full_localization_error.png
outputs/full_baselines/visualization/summary/full_global_micro_precision_recall_f1.png
outputs/full_baselines/visualization/summary/full_frame_macro_precision_recall_f1.png
outputs/full_baselines/visualization/summary/full_global_precision_recall_tradeoff.png
outputs/full_baselines/visualization/summary/full_normalized_detection_errors.png
outputs/full_baselines/visualization/summary/full_prediction_count_bias.png
outputs/full_baselines/visualization/summary/full_car_ap_iou_sweep.png
outputs/full_baselines/visualization/summary/full_single_modal_academic_comparison.csv
outputs/full_baselines/visualization/summary/full_single_modal_evaluation_protocol.md
outputs/full_baselines/<baseline>/visualization/representative_samples.csv
outputs/full_baselines/<baseline>/visualization/representative_world/
outputs/full_baselines/full_baseline_visualization_summary.json
```

其中 `full_global_micro_precision_recall_f1.png` 使用全数据汇总的 TP、FP、FN，回答总体检测性能；`full_frame_macro_precision_recall_f1.png` 是逐帧等权平均，描述帧间稳定性。两者不可互换。`full_normalized_detection_errors.png` 使用 `FN/GT`、`FP/GT` 和 `|Pred-GT|/GT`，避免原始 FN/FP 受样本规模影响。`full_car_ap_iou_sweep.png` 分别展示 Car 的 BEV/3D AP 在 IoU=0.25、0.50、0.70 下的变化，用于观察几何约束收紧后的性能衰减。

当前 `full_common_val` 在标准评测类 Car/Pedestrian/Cyclist 中仅有 Car GT；原始标签另有 2 个 `barrow` 和 2 个 `unknowns_movable`，它们保留在总 GT 计数中，但不进入三类 AP 或分类矩阵。因此，`classification_accuracy` 仅在已匹配对象中统计，不能解释为完整多类别分类能力；AP 图也应解读为 Car 类比较。完整口径和该限制写入 `full_single_modal_evaluation_protocol.md`。

两个 Camera baseline 还可将同一批代表帧投影回原始图像。以 Vehicle Camera 为例：

```bash
conda activate dair-camera-5060
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/gxy/projects/item3/dair_v2x_project

python visualize_camera_3d_predictions.py \
  --baseline vehicle_camera \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure \
  --prediction-path outputs/full_baselines/vehicle_camera/predictions_vehicle_camera.json \
  --details-path outputs/full_baselines/vehicle_camera/eval_details.json \
  --sample-id-file outputs/full_baselines/vehicle_camera/visualization/representative_sample_ids.txt \
  --output-dir outputs/full_baselines/vehicle_camera/visualization
```

路侧端将 `vehicle_camera` 和相关路径替换为 `infrastructure_camera`。该命令只生成代表帧原图和联系表，不会覆盖 world 图或重新运行模型。

完整的路侧 Camera 原图投影命令为：

```bash
conda activate dair-camera-5060
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/gxy/projects/item3/dair_v2x_project

python visualize_camera_3d_predictions.py \
  --baseline infrastructure_camera \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure \
  --prediction-path outputs/full_baselines/infrastructure_camera/predictions_infrastructure_camera.json \
  --details-path outputs/full_baselines/infrastructure_camera/eval_details.json \
  --sample-id-file outputs/full_baselines/infrastructure_camera/visualization/representative_sample_ids.txt \
  --output-dir outputs/full_baselines/infrastructure_camera/visualization
```

原图投影的颜色语义：绿色为已匹配 GT，红色为漏检 GT，青色为已匹配预测，橙色为误检预测，紫色为类别错误的已匹配预测。每张图包含该帧的 GT、预测、匹配数、Precision 和 Recall。输出位于：

```text
outputs/full_baselines/<camera_baseline>/visualization/image_3d_detection/
outputs/full_baselines/<camera_baseline>/visualization/summary/camera_3d_detection_contact_sheet.png
outputs/full_baselines/<camera_baseline>/visualization/summary/camera_3d_visualization_summary.json
```

## 7. LiDAR 诊断准备

当 Full Dataset 的 LiDAR 结果异常时，不应直接更换模型或修改阈值。先运行 Task 1 冻结当前 checkpoint、配置、预测、world 转换和评价工件，并建立固定诊断样本集：

```bash
python3 scripts/prepare_lidar_diagnosis_task1.py
```

详细字段、抽样策略和当前验证结论见 [LiDAR 诊断 Task 1 文档](lidar_diagnosis_task1.md)。
诊断样本支持 `--seed`，更换随机种子可获得另一组类别分布不同但仍可复现的样本。

Task 2 的原始坐标系检查、点云/本地 GT/原始预测图和标定修正对照见 [LiDAR 诊断 Task 2 文档](lidar_diagnosis_task2.md)。
