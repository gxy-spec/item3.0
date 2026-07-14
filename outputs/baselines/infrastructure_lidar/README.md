# Infrastructure LiDAR-only Baseline 输出说明

本目录用于后续保存 Infrastructure LiDAR-only baseline 的预测、坐标转换、评价和可视化结果。

后续 Infrastructure LiDAR-only 检测器必须导出统一预测文件：

```text
outputs/baselines/infrastructure_lidar/predictions_infrastructure_lidar.json
```

该文件必须遵循统一 baseline 预测 JSON 格式：

```text
dair_v2x_project/docs/predictions_json_format.md
```

## 必填样本级字段

每个样本记录必须包含：

```text
sample_id
coordinate_system
prediction_type
pred_objects
```

其中：

- `sample_id`：统一样本 ID，用于连接标定、GT 和评价流程。
- `coordinate_system`：Infrastructure LiDAR-only 阶段建议为 `infrastructure_lidar`。
- `prediction_type`：建议明确写出模型和模态，例如 `openpcdet_infrastructure_lidar_pointpillars`。
- `pred_objects`：该帧预测目标列表。

## 推荐字段

以下字段推荐提供，但缺失时只产生 warning，不作为格式错误：

```text
sensor
source_id
```

Infrastructure LiDAR-only 推荐写法：

```json
{
  "sample_id": "015372",
  "coordinate_system": "infrastructure_lidar",
  "prediction_type": "openpcdet_infrastructure_lidar_pointpillars",
  "sensor": "infrastructure_lidar",
  "source_id": "006315",
  "pred_objects": []
}
```

注意：

```text
vehicle_id 不是统一模板必填字段。
```

Infrastructure baseline 可以使用 `infrastructure_id` 或 `source_id` 表示路侧原始帧 ID。

## 格式校验

生成 `predictions_infrastructure_lidar.json` 后，运行：

```bash
cd /home/gxy/projects/item3
conda run -n dair-baseline python dair_v2x_project/validate_predictions_format.py \
  dair_v2x_project/outputs/baselines/infrastructure_lidar/predictions_infrastructure_lidar.json
```

校验通过后，再进行后续坐标转换和 world 坐标评价。

## 当前真实 PointPillars 结果

当前目录中的 `predictions_infrastructure_lidar.json` 已由真实 OpenPCDet
PointPillars 模型导出，不是标注工程验证版。模型训练使用 37 个 `train` 样本，
并只在 9 个 `val` 样本上推理。输入点特征为原始 PCD 的：

```text
x, y, z, intensity
```

正式 checkpoint：

```text
/home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_infrastructure/
infra_lidar_pointpillar_intensity_80e/ckpt/checkpoint_epoch_80.pth
```

对应真实推理结果：

```text
/home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_infrastructure/
infra_lidar_pointpillar_intensity_80e_eval/eval/epoch_80/val/default/result.pkl
```

从项目根目录运行的完整复现流程如下：

```bash
conda run -n torch-gpu-test python OpenPCDet/prepare_infrastructure_custom_dataset.py

conda run -n torch-gpu-test bash -lc '
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python /home/gxy/projects/item3/OpenPCDet/create_custom_infrastructure_infos.py --workers 2
'

conda run -n torch-gpu-test bash -lc '
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/gxy/projects/item3/OpenPCDet/tools
OPENPCDET_FORCE_NUMPY_VOXELIZER=1 OPENPCDET_SKIP_KITTI_EVAL=1 CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --cfg_file cfgs/custom_models/pointpillar_custom_infrastructure.yaml \
  --batch_size 1 --epochs 80 --workers 0 \
  --extra_tag infra_lidar_pointpillar_intensity_80e
'

conda run -n torch-gpu-test bash -lc '
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/gxy/projects/item3/OpenPCDet/tools
OPENPCDET_FORCE_NUMPY_VOXELIZER=1 OPENPCDET_SKIP_KITTI_EVAL=1 CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --cfg_file cfgs/custom_models/pointpillar_custom_infrastructure.yaml \
  --ckpt /home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_infrastructure/infra_lidar_pointpillar_intensity_80e/ckpt/checkpoint_epoch_80.pth \
  --batch_size 1 --workers 0 \
  --extra_tag infra_lidar_pointpillar_intensity_80e_eval --save_to_file
'

conda run -n dair-baseline python dair_v2x_project/export_infrastructure_lidar_predictions.py \
  --input-result-pkl /home/gxy/projects/item3/OpenPCDet/output/custom_models/pointpillar_custom_infrastructure/infra_lidar_pointpillar_intensity_80e_eval/eval/epoch_80/val/default/result.pkl \
  --score-threshold 0.1

conda run -n dair-baseline python dair_v2x_project/convert_lidar_predictions_to_world.py \
  --sensor infrastructure_lidar
```

只有在显式传入 `--label-oracle` 时，导出脚本才会生成工程验证版。
真实模型结果的 `prediction_type` 固定为：

```text
openpcdet_infrastructure_lidar_pointpillars
```
