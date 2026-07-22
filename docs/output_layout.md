# 输出目录说明

`outputs/` 按实验阶段组织，活跃结果与归档结果分开存放。

## 活跃目录

| 目录 | 用途 | 是否可用于当前实验 |
|---|---|---|
| `outputs/full_baselines/<baseline>/` | Full Dataset 单模态推理、world 转换、评价与可视化 | 是 |
| `outputs/common_multimodal_dataset/` | 四模态公共样本集合、split 与训练配置 | 是 |
| `outputs/full_training/` | Full Dataset 训练准备、数据完整性检查 | 是 |
| `outputs/preprocessed/` | 小样本与索引预处理结果 | 是 |
| `outputs/diagnosis/` | LiDAR 坐标、标定与 evaluator 诊断证据 | 是 |
| `outputs/baselines/` | Day1 小样本实验结果，保留用于历史对照 | 是 |

## Full Dataset baseline 目录

每个 `outputs/full_baselines/<baseline>/` 的活跃文件约定为：

```text
predictions_<baseline>.json          本地传感器坐标预测
predictions_world.json               统一 world 坐标预测
eval.csv / eval_summary.json         项目诊断指标与逐帧匹配结果
official_local_aligned_ap_*.json/csv 官方 DAIR-V2X local-coordinate AP
visualization/                       world/BEV 代表帧与指标图
visualization_camera_image/          Camera 原始图像 3D 框投影图
train_1e_fixed/epoch_1.pth           当前阶段性 Camera checkpoint
```

`official_local_aligned_ap_*` 是与 DAIR-V2X 官方 `eval_utils.py` 对齐的 AP 输出；
`eval_*` 是项目用于定位误差、漏检、误检和分类分析的 world 坐标诊断输出，两者不可混用。

## 归档目录

`outputs/archive/` 只存放不再参与当前实验的中间产物，采用移动而不是删除：

```text
invalid_empty_gt_camera/  空标注训练产生的无效 checkpoint
camera_smoke_tests/       8 样本、3 epoch 等 smoke test checkpoint
provenance/               修复前零预测和底面中心 box 的可追溯副本
organization_manifest.json 每次整理的来源、目标、原因和大小
```

执行整理前先预览：

```bash
python scripts/organize_project_outputs.py
```

确认后执行移动：

```bash
python scripts/organize_project_outputs.py --apply
```

该脚本不会删除任何文件，也不会移动 Full Dataset 正式预测、评价、当前 checkpoint、数据集、环境或源码。
