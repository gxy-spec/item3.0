# LiDAR 诊断 Task 3：逐级验证到 world 的坐标转换

## 运行命令

```bash
cd /home/gxy/projects/item3/dair_v2x_project
conda activate torch-gpu-test
python scripts/run_lidar_diagnosis_task3.py \
  --data-root /mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure
```

输出：

```text
outputs/diagnosis/coordinate_transform_task3_summary.json
```

## 检查内容

脚本对 Task 1 选取的 20 个多类别诊断样本检查：

1. Vehicle：`vehicle_lidar -> lidar_to_novatel -> novatel_to_world -> world`。
2. Infrastructure：`virtuallidar -> virtuallidar_to_world -> world`，并应用 `relative_error.delta_x/delta_y`。
3. 旋转矩阵正交性、行列式、齐次矩阵末行和有限数值。
4. world -> LiDAR 逆变换后的点、GT 中心、box 角点和 box 尺寸闭环误差。
5. 本地 GT 转 world 后与 cooperative `label_world` 的 5 m 唯一中心匹配误差。

## 本次验证结果

两侧 20/20 个样本的变换矩阵均为有效刚体变换。

| 指标 | Vehicle | Infrastructure |
|---|---:|---:|
| 最大点往返误差 | `1.44e-12 m` | `1.53e-12 m` |
| 最大中心往返误差 | `1.28e-12 m` | `2.13e-12 m` |
| 最大角点往返误差 | `1.58e-12 m` | `2.15e-12 m` |
| 最大 box 尺寸往返误差 | `1.25e-12 m` | `1.36e-12 m` |
| 匹配 GT 平均中心误差 | `0.142 m` | `0.485 m` |
| 匹配 GT 最大中心误差 | `4.704 m` | `4.323 m` |

## 结论

矩阵结构和正逆变换数值闭环均通过，误差约为浮点计算误差量级。因此没有发现矩阵乘法顺序、逆矩阵实现、box corner 变换或固定整体旋转/平移错误。Infrastructure 使用 `relative_error` 后与 world GT 的平均对齐误差明显降低，但仍有少量离群样本；这些离群更可能与标注对象匹配、车路时间差、局部标注规则或单帧数据质量有关，不能仅凭 Task 3 归因于坐标转换实现错误。

往返误差验证的是“变换实现内部自洽”，GT 对齐误差验证的是“绝对坐标是否与 cooperative world 一致”，二者需要分别解读。
