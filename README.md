# DAIR-V2X-C 数据预处理、检测基线与可视化

本项目用于 DAIR-V2X-C 数据集的本地数据检查、索引构建、坐标转换、检测结果评价与可视化。

当前版本已完成 Day1 双 LiDAR 3D 检测 baseline，以及 Day2 双 Camera-only ImVoxelNet 小样本推理。它们共同构成后续多模态融合和语义通信研究的可审计单模态参照系：

```text
实验一：Vehicle LiDAR-only 3D 检测 baseline

vehicle-side LiDAR 点云
-> PointPillars / OpenPCDet 车辆端 3D 检测
-> vehicle LiDAR 坐标系下的 3D box
-> world 坐标系
-> 与 cooperative label_world GT 评价
-> 输出漏检、误检、数量误差、定位误差、分类指标、AP/mAP 和可视化
```

Day2 已补充本机 RTX 5060 Ti 可运行的 Camera-only ImVoxelNet 小样本推理：

```text
Vehicle Camera image -> ImVoxelNet -> vehicle LiDAR frame 3D box -> world -> cooperative label_world evaluation
Infrastructure Camera image -> ImVoxelNet -> infrastructure LiDAR frame 3D box -> world -> cooperative label_world evaluation
```

```text
Day1 扩展：Infrastructure LiDAR-only 3D 检测 baseline

infrastructure-side LiDAR 点云
-> PointPillars / OpenPCDet 路侧端 3D 检测
-> infrastructure LiDAR 坐标系下的 3D box
-> world 坐标系
-> 与同一 cooperative label_world GT 评价
-> 与 Vehicle LiDAR-only 进行统一指标对比
```

实验一的完整说明见：

- [实验一文档：Vehicle LiDAR-only 3D 检测 baseline](docs/experiment1_vehicle_lidar_baseline.md)
- [Day1 文档：双 LiDAR baseline 构建、评价与对比](docs/day1_dual_lidar_baseline.md)
- [Day2 文档：本机 Camera-only ImVoxelNet 小样本推理](docs/day2_camera_imvoxelnet_local_inference.md)
- [统一预测 JSON 格式](docs/predictions_json_format.md)

## 当前研究状态与科学问题

本仓库的目标不是只运行某个检测器，而是为车路协同感知建立可复现、可比较、可追溯的研究链路。当前版本固定了以下共同约定：

1. Vehicle LiDAR、Infrastructure LiDAR、Vehicle Camera 和 Infrastructure Camera 均输出各自传感器关联 LiDAR 坐标系中的 3D 检测框。
2. 所有预测使用标定链转换至 world 坐标系，并与同一 cooperative `label_world` GT 进行评价。
3. 所有 baseline 使用统一预测 JSON、统一格式校验、统一匹配逻辑和统一可视化入口。
4. 预测来源、坐标系和工程验证属性由 `prediction_type` 明确记录，避免将 label oracle、真实检测器和不同运行时混为一谈。

由此可以分别研究四个尚未混合的因素：车端与路侧视角差异、点云与图像模态差异、world 坐标对齐误差，以及后续融合或通信机制带来的增益。当前阶段的结果是后续方法的比较基线，不是最终融合性能结论。

### 已完成工作

| 工作包 | 当前完成内容 | 可审计产物 |
| --- | --- | --- |
| 数据与标定 | cooperative 样本索引、world GT 构建、关键文件缺失检查、车端/路侧 LiDAR 索引、Camera KITTI-style 派生格式 | `outputs/preprocessed/`、索引 summary JSON |
| LiDAR baseline | 两端真实 OpenPCDet PointPillars 推理、`result.pkl` 导出、统一预测 JSON、world 转换、评价与对比 | `outputs/baselines/{vehicle,infrastructure}_lidar/` |
| Camera baseline | 在 RTX 5060 Ti 上以新版 MMD3D 兼容运行时加载 DAIR-V2X ImVoxelNet 官方权重；车端和路侧各完成 9 帧真实推理 | `outputs/baselines/{vehicle,infrastructure}_camera/` |
| 统一评价 | 匹配对记录、FN/FP、数量误差、中心定位误差、分类分析、混淆矩阵、每类 Precision/Recall/F1、BEV/3D mAP | `<baseline>_eval.csv`、`<baseline>_eval_summary.json`、AP summary |
| 可视化 | world 平面 GT/预测对比、逐样本指标曲线、类别指标图、Day1 LiDAR 对比图、相机图像平面 3D 框投影 | `visualization/` 与 `day1_lidar_summary_visualization/` |

### 当前结果快照

以下数值均来自当前 `example-cooperative-split-data.json` 的 9 个验证样本和 239 个 cooperative world GT。Precision、Recall、F1 和定位误差为逐帧均值；mAP 使用 IoU 阈值 0.50。它们适合检查工程链路和形成研究假设，但样本量不足以作为正式 benchmark 或统计显著性结论。

| Baseline | 真实预测类型 | 预测/匹配/FN/FP | Precision | Recall | F1 | BEV mAP@0.50 | 3D mAP@0.50 | 匹配框 BEV/3D 中心误差 (m) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vehicle LiDAR-only | `openpcdet_vehicle_lidar_pointpillars` | 205 / 70 / 169 / 135 | 0.3357 | 0.2987 | 0.3053 | 0.2085 | 0.1689 | 0.5207 / 0.5745 |
| Infrastructure LiDAR-only | `openpcdet_infrastructure_lidar_pointpillars` | 164 / 141 / 98 / 23 | 0.8582 | 0.5888 | 0.6979 | 0.3898 | 0.2741 | 0.7254 / 0.7609 |
| Vehicle Camera-only | `mmdet3d_1x_imvoxelnet_vehicle_camera` | 39 / 37 / 202 / 2 | 0.9500 | 0.1567 | 0.2673 | 0.0168 | 0.0000 | 0.9147 / 1.2661 |
| Infrastructure Camera-only | `mmdet3d_1x_imvoxelnet_infrastructure_camera` | 67 / 67 / 172 / 0 | 1.0000 | 0.2793 | 0.4360 | 0.1381 | 0.0000 | 0.8608 / 1.2924 |

`classification_accuracy = 1.0` 不列为主要结论：当前四条链路有效 GT 和预测均只有 Car，且该指标只在已经空间匹配的对象上计算。因此它仅表示“已匹配对象的 Car 标签一致”，不能证明完整的 Car/Pedestrian/Cyclist 多类别分类能力。

### 对结果的审慎解读

**达到预期的部分：**

- 两类 LiDAR 与两类 Camera 都已走通“真实模型预测 -> world 坐标 -> cooperative GT -> 统一评价”的完整闭环；没有将工程 label oracle 当成真实模型结果。
- Day1 中路侧 LiDAR 的 Recall、F1 和 mAP 均高于车辆端 LiDAR，且 FP 更少。这与固定路侧视角可能覆盖交叉口更多目标的研究假设相容，但尚不能据此断言“路侧传感器普遍优于车端”。值得注意的是，车辆端的匹配框中心误差反而更低，说明覆盖能力和匹配后定位精度是不同维度。
- Camera-only 图像投影结果在原始图像上能看到青色匹配预测与绿色 GT 的重合区域，且红色漏检框可定位到具体目标。这证明预测框坐标、相机标定和 world 评价索引可连通，为后续误差归因提供了可检查证据，而非只依赖汇总数值。

**当前不理想且需要进一步验证的部分：**

- 两条 Camera-only 链路的 Recall 较低，3D mAP@0.50 均为 0。它表明在当前协议下，预测中心落入 5 m 匹配半径不代表 3D IoU 达到 0.50；尺寸、朝向、深度或标定误差中的任一项都可能降低严格 IoU。不能仅根据该现象直接归因于“图像模态不如点云”。
- Camera world 评价对全部 cooperative world GT 计数，而单个相机只覆盖其中一部分视野。图像投影 summary 中，Vehicle Camera 有 121 个漏检 GT、Infrastructure Camera 有 94 个漏检 GT 完全未落入对应图像可见区域。这些对象仍应保留在当前“协同场景全目标”评价中，但会使单传感器的 Recall 同时混合检测失败与不可见目标两种因素。后续必须增加 FOV/遮挡/距离分层评价。
- LiDAR 结果的车路差异也不能简单归因为安装位置。候选解释包括视场与遮挡、训练数据与类别分布、点云密度、模型 checkpoint 与预处理配置、分数阈值和 NMS 设置。需要受控实验逐一固定这些因素。
- Camera 兼容运行时已完成 checkpoint 的 394 个参数键完全匹配，但“权重可加载”不等于与旧版官方运行时数值完全等价。仍需要在兼容旧硬件或已保存参考输出上进行逐框/逐层近似一致性测试。

### 重点可视化输出与阅读方法

| 输出 | 研究用途 | 应如何解读 |
| --- | --- | --- |
| `visualization/world_gt_pred_compare/` | 在 world BEV 中检查统一坐标、匹配、漏检和误检 | 可定位误检集中区域、漏检空间分布和跨端覆盖差异；不能单凭橙色框判定是模型训练不足、单模态歧义或标定误差，必须结合原图/点云、IoU 和标定残差。 |
| `summary/metric_summary_bar.png`、`per_sample_count_curve.png`、`per_sample_error_curve.png` | 比较逐帧 Precision/Recall、数量偏差和定位偏差 | 关注离群帧，而非只看均值；离群帧应回查传感器视场、场景密度、遮挡和时间同步信息。 |
| `summary/per_sample_localization_error.png` | 分析已匹配对象的 BEV/3D 中心误差 | 仅对匹配对象计算，不能用于说明漏检对象的定位能力；应与 IoU、尺寸、朝向误差联合报告。 |
| `summary/class_confusion_matrix.png`、`per_class_precision_recall_f1.png` | 检查类别混淆和类别不均衡 | 当前仅 Car 有效，图中 Pedestrian/Cyclist 的 N/A 是“无样本/无预测”，不是零性能。 |
| `image_3d_detection/*.jpg` | 将 3D GT 和预测投影回车辆端或路侧原始图像 | 绿色为匹配 GT，红色为漏检 GT，青色为匹配预测，橙色为 FP，紫红色为类别错误。用于判断框是否贴合可见目标，并识别视锥外的 `hidden_*` 对象。 |
| `camera_3d_detection_contact_sheet.png` | 快速扫描 9 帧相机检测质量 | 适合发现系统性深度偏移、固定方向误差或特定拥挤场景退化；发现模式后必须用定量分桶统计验证。 |
| `day1_lidar_summary_visualization/` | 将车端与路侧 LiDAR 的 P/R/F1、FN/FP、数量误差、定位误差、AP 并列比较 | 两行结果使用同一 GT 和相同样本，具备初步可比性；但应报告样本数量、置信区间和场景组成后再作方法优劣结论。 |

### 下一步研究路线

**短期目标（先保证比较公平和结论可信）：**

1. 冻结数据版本、split、类别映射、坐标约定和评分阈值，补充每个 baseline 的 checkpoint、配置哈希和运行环境记录。
2. 扩展至正式训练/验证/测试划分，按距离、遮挡、目标尺度、交通密度、昼夜/天气和传感器 FOV 做分层统计，并报告均值、方差或 bootstrap 置信区间。
3. 在严格 IoU 指标之外保留中心匹配诊断；新增 BEV/3D IoU、尺寸误差、航向误差和标定残差，以区分“发现目标”与“框质量”。
4. 为 Camera-only 评价增加可见性掩码，分别报告全 cooperative 场景目标和当前传感器可见目标的 Recall；同时验证新版运行时与官方参考输出的数值一致性。
5. 完成 Car/Pedestrian/Cyclist 多类别模型与评估，进行分数校准和 PR 曲线分析，避免单一 `score_threshold=0.1` 主导结论。

**长期目标（面向车路协同融合与语义通信）：**

1. 建立同一 world GT 下的早期、特征级、目标级和后期融合对照组，明确每一种协同增益来自补盲、定位修正还是类别判别。
2. 在融合前注入可控的时间延迟、外参扰动、丢包和带宽约束，量化融合系统对现实通信条件的鲁棒性。
3. 将语义通信定义为任务导向的率失真问题：比较原始点云/图像、压缩特征、候选框和任务相关语义消息在相同带宽下的检测性能、时延和可靠性。
4. 以单模态强基线、无通信融合、理想通信融合和受限通信融合为完整消融链，报告感知指标、通信代价、计算开销和鲁棒性，而不仅报告单一 mAP。

### 可复现性与版本控制范围

GitHub 仅提交源码、配置、中文文档和轻量汇总表。原始 DAIR-V2X 数据、模型 checkpoint、OpenPCDet `result.pkl`、预测 JSON、逐帧图像和联系表均通过 `.gitignore` 排除；这些产物可由 README 和对应 Day1/Day2 文档中的命令重新生成。提交前应运行语法检查、预测格式校验以及目标 baseline 的最小样本推理。

## 项目结构

- `config.py`：数据集根目录和输出路径配置。
- `scripts/`：数据检查、索引构建、统计和预处理脚本。
- `outputs/preprocessed/`：预处理索引和统计结果。
- `outputs/visual_check/`：数据可视化检查结果。
- `outputs/baselines/vehicle_lidar/`：实验一检测、转换、评价和可视化结果。
- `outputs/baselines/infrastructure_lidar/`：路侧 LiDAR-only 检测、转换、评价和可视化结果。
- `outputs/baselines/vehicle_camera/`：Vehicle Camera-only ImVoxelNet 预测、world 转换、评价和可视化结果。
- `outputs/baselines/infrastructure_camera/`：Infrastructure Camera-only ImVoxelNet 预测、world 转换、评价和可视化结果。
- `outputs/baselines/day1_lidar_baseline_summary.*`：双 LiDAR 的最终可审计汇总表。
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

```bash
conda activate dair-camera-5060
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
```

用于 Day2 本机 Camera-only ImVoxelNet 推理。该环境使用新版 MMDetection3D 运行时加载 DAIR-V2X Camera ImVoxelNet checkpoint，避免旧 PyTorch 对 RTX 5060 Ti `sm_120` 的不兼容。

完整命令见 [Day2 Camera-only 文档](docs/day2_camera_imvoxelnet_local_inference.md)。

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

所有 baseline 的 `predictions_*.json` 都必须遵循同一套预测 JSON 标准。标准格式见：

```text
dair_v2x_project/docs/predictions_json_format.md
```

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

格式校验：

```bash
conda run --no-capture-output -n dair-baseline python dair_v2x_project/validate_predictions_format.py \
  dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar.json
```

注意：`vehicle_id` 不是统一模板必填字段。`sensor` 和 `source_id` 是推荐字段，缺失时只给 warning，不作为错误。

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

## Day1 双 LiDAR 结果摘要

当前已完成 Vehicle LiDAR-only 与 Infrastructure LiDAR-only 的真实 OpenPCDet PointPillars 80 epoch 推理、world 坐标转换和统一评价。两端均使用相同的 9 个验证样本和 cooperative `label_world` GT。

```text
Vehicle LiDAR-only
  num_pred / num_match / FN / FP: 205 / 70 / 169 / 135
  precision / recall / F1: 0.3357 / 0.2987 / 0.3053
  BEV mAP@0.50 / 3D mAP@0.50: 0.2085 / 0.1689

Infrastructure LiDAR-only
  num_pred / num_match / FN / FP: 164 / 141 / 98 / 23
  precision / recall / F1: 0.8582 / 0.5888 / 0.6979
  BEV mAP@0.50 / 3D mAP@0.50: 0.3898 / 0.2741
```

最终汇总命令：

```bash
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/generate_day1_lidar_baseline_summary.py
```

输出文件：

```text
dair_v2x_project/outputs/baselines/day1_lidar_baseline_summary.csv
dair_v2x_project/outputs/baselines/day1_lidar_baseline_summary.json
dair_v2x_project/outputs/baselines/day1_lidar_summary_visualization/
```

详细流程、指标解释与文件说明见 [Day1 双 LiDAR baseline 文档](docs/day1_dual_lidar_baseline.md)。

## Day2 Step1：Infrastructure Camera ImVoxelNet 数据格式

DAIR-V2X-C 的 vehicle-side 已有官方转换后的 `training/` 目录，而 infrastructure-side 原始数据默认没有。下面的脚本使用 `example-cooperative-split-data.json` 中的 infrastructure split，为路侧相机生成 MMDetection3D / KITTI-style 派生目录：

```bash
conda run --no-capture-output -n dair-baseline python \
  dair_v2x_project/prepare_infrastructure_camera_imvoxelnet_format.py
```

生成后可检查：

```bash
cd /home/gxy/projects/item3/DAIR-V2X
ls data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/training
```

预期目录为：

```text
calib/
image_2/
label_2/
```

脚本复用官方 `tools/dataset_converter/gen_kitti/` 中的标签坐标转换、KITTI 标签/标定和 ImageSets 生成逻辑。它仅创建派生目录：`image_2` 内的图片是指向原始 `image/` 的软链接；原始 image、label、calib 和 `data_info.json` 不会被修改。当前 example split 生成 46 个样本，其中训练集 37 个、验证集 9 个。摘要写入：

```text
dair_v2x_project/outputs/preprocessed/infrastructure_camera_imvoxelnet_format_summary.json
```

## 无检测器的工程自检

如果只想验证坐标转换和评价代码，可以用 vehicle-side label 构造 oracle-like prediction：

```bash
conda run -n dair-baseline python dair_v2x_project/export_vehicle_label_oracle_predictions.py
conda run -n dair-baseline python dair_v2x_project/convert_vehicle_lidar_predictions_to_world.py
conda run -n dair-baseline python dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py
```
