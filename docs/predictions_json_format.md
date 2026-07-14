# Baseline 统一预测 JSON 格式说明

本文档定义所有 baseline 检测器输出到 item3.0 评价流程前必须遵循的统一 JSON 格式。

当前标准以实验一已经生成的文件为模板：

```text
dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar.json
```

后续所有 baseline，例如：

```text
Vehicle LiDAR-only
Infrastructure LiDAR-only
Vehicle Camera-only
Infrastructure Camera-only
```

都应导出为同一类 `predictions_*.json` 文件。这样坐标转换、评价、分类分析和可视化脚本可以复用同一套读取逻辑。

## 1. 顶层结构

统一预测文件必须是一个 JSON list。

每个元素对应一个样本：

```json
[
  {
    "sample_id": "015372",
    "coordinate_system": "vehicle_lidar",
    "prediction_type": "openpcdet_vehicle_lidar_pointpillars",
    "sensor": "vehicle_lidar",
    "source_id": "015372",
    "pred_objects": []
  }
]
```

## 2. 样本级字段

### 2.1 必填字段

以下字段为必填：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `sample_id` | string | 样本 ID，是评价流程中连接预测、标定和 GT 的主键。 |
| `coordinate_system` | string | 当前预测框所在坐标系，例如 `vehicle_lidar`、`infrastructure_lidar`、`vehicle_camera`、`world`。 |
| `prediction_type` | string | 预测来源和模型类型，例如 `openpcdet_vehicle_lidar_pointpillars`。 |
| `pred_objects` | list | 当前样本的预测目标列表。没有预测时应为空列表 `[]`。 |

注意：

```text
vehicle_id 不作为必填字段。
```

原因是不同 baseline 的 source ID 命名方式可能不同，且 Vehicle LiDAR-only 的统一输出模板不依赖 `vehicle_id` 才能完成评价。因此统一格式不应强制要求 `vehicle_id`。

### 2.2 推荐字段

以下字段推荐提供，但缺失时只产生 warning，不作为格式错误：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `sensor` | string | 传感器或模态名称，例如 `vehicle_lidar`、`infrastructure_lidar`。 |
| `source_id` | string | 原始数据源 ID，例如 vehicle frame id 或 infrastructure frame id。 |

推荐使用方式：

```json
{
  "sample_id": "015372",
  "sensor": "vehicle_lidar",
  "source_id": "015372"
}
```

当前 Vehicle LiDAR-only 导出脚本会自动补充：

```json
{
  "sensor": "vehicle_lidar",
  "source_id": "<sample_id>"
}
```

如果是旧版本输出文件缺少这两个字段，可以重新运行导出脚本补齐；验证脚本仍然只会把缺失视为 warning。

如果某个 baseline 存在更明确的原始 ID，例如 infrastructure-side frame id，则可以写：

```json
{
  "sample_id": "015372",
  "sensor": "infrastructure_lidar",
  "source_id": "006315"
}
```

### 2.3 可选字段

以下字段可以根据需要提供：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `vehicle_id` | string | 车辆端 ID。仅当该 baseline 明确有 vehicle-side frame id 时使用。 |
| `infrastructure_id` | string | 路侧端 ID。仅当该 baseline 明确有 infrastructure-side frame id 时使用。 |
| `source_file` | string | 预测来源文件，例如 OpenPCDet `result.pkl` 路径。 |
| `metadata` | object | 额外元信息，例如模型 checkpoint、score threshold、split 名称等。 |

## 3. `pred_objects` 目标级字段

`pred_objects` 中每个元素对应一个预测目标。

推荐格式：

```json
{
  "class": "Car",
  "score": 0.5589513778686523,
  "center_lidar": [24.99, 32.44, -0.78],
  "box_lidar": {
    "dx": 4.59,
    "dy": 1.93,
    "dz": 1.70,
    "heading": 1.95
  },
  "source": "openpcdet_pointpillars",
  "corners_3d_lidar": []
}
```

### 3.1 必填字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `class` | string | 预测类别。统一类别建议使用 `Car`、`Pedestrian`、`Cyclist`。 |
| `score` | number | 预测置信度。通常范围为 `[0, 1]`。 |

此外，每个目标应至少提供一组中心点字段和一组 box 字段。

中心点字段根据坐标系命名：

```text
center_lidar
center_world
center_sensor
center_camera
```

box 字段根据坐标系命名：

```text
box_lidar
box_world
box_sensor
box_camera
```

当前 Vehicle LiDAR-only baseline 使用：

```text
center_lidar
box_lidar
```

world 坐标转换后使用：

```text
center_world
box_world
```

### 3.2 box 字段要求

对于 LiDAR 3D box，`box_lidar` 推荐包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `dx` | number | box 在局部 x 方向的长度。 |
| `dy` | number | box 在局部 y 方向的宽度。 |
| `dz` | number | box 高度。 |
| `heading` | number | LiDAR 坐标系下 yaw 角，单位为弧度。 |

对于 world 3D box，`box_world` 推荐包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `dx` 或 `length` | number | box 长度。 |
| `dy` 或 `width` | number | box 宽度。 |
| `dz` 或 `height` | number | box 高度。 |
| `heading_world` 或 `yaw` | number | world 坐标系下 yaw 角，单位为弧度。 |

## 4. 文件命名规范

每个 baseline 应输出一个 `predictions_*.json` 文件。

推荐命名：

```text
Vehicle LiDAR-only:
outputs/baselines/vehicle_lidar/predictions_vehicle_lidar.json

Infrastructure LiDAR-only:
outputs/baselines/infrastructure_lidar/predictions_infrastructure_lidar.json

Vehicle Camera-only:
outputs/baselines/vehicle_camera/predictions_vehicle_camera.json

Infrastructure Camera-only:
outputs/baselines/infrastructure_camera/predictions_infrastructure_camera.json
```

## 5. 当前 Vehicle LiDAR-only 示例

当前 `predictions_vehicle_lidar.json` 的样本级字段为：

```json
{
  "sample_id": "015372",
  "coordinate_system": "vehicle_lidar",
  "prediction_type": "openpcdet_vehicle_lidar_pointpillars",
  "source_file": "/path/to/result.pkl",
  "pred_objects": []
}
```

该文件缺少推荐字段：

```text
sensor
source_id
```

这不是错误。校验脚本只会给出 warning。

## 6. 格式校验

使用：

```bash
cd /home/gxy/projects/item3
conda run -n dair-baseline python dair_v2x_project/validate_predictions_format.py \
  dair_v2x_project/outputs/baselines/vehicle_lidar/predictions_vehicle_lidar.json
```

如果不指定路径，脚本会默认扫描：

```text
dair_v2x_project/outputs/baselines/**/predictions_*.json
```

校验规则：

- 缺少必填字段时报 error。
- 字段类型错误时报 error。
- `sensor` 或 `source_id` 缺失时报 warning。
- `vehicle_id` 缺失不报错。
