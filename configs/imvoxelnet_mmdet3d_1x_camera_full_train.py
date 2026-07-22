"""Full Dataset ImVoxelNet training config.

Use ``DAIR_CAMERA_DATA_ROOT`` to select vehicle-side or infrastructure-side.
The same config is intentionally shared so both camera baselines use the same
12-epoch official 1x schedule.
"""

_base_ = "./imvoxelnet_mmdet3d_1x_camera.py"

import os

data_root = os.environ.get("DAIR_CAMERA_DATA_ROOT", ".")
if not data_root.endswith("/"):
    data_root += "/"
class_names = ["Car"]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=False, use_camera=True)

train_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=None),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="Resize", scale=(960, 540), keep_ratio=True),
        dict(type="Pack3DDetInputs", keys=["img", "gt_bboxes_3d", "gt_labels_3d"], meta_keys=(
        "img_id", "img_path", "ori_shape", "img_shape", "scale_factor",
        "cam2img", "lidar2cam", "lidar2img", "box_type_3d", "box_mode_3d",
    )),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file="mmdet3d_camera_infos_train.pkl",
        data_prefix=dict(img="image"),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        filter_empty_gt=False,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=None,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file="mmdet3d_camera_infos_val.pkl",
        data_prefix=dict(img="image"),
        pipeline=[dict(type="LoadImageFromFileMono3D", backend_args=None), dict(type="Resize", scale=(960, 540), keep_ratio=True), dict(type="Pack3DDetInputs", keys=["img"])],
        modality=input_modality,
        test_mode=True,
        filter_empty_gt=False,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=None,
    ),
)
test_dataloader = val_dataloader
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
val_evaluator = dict(type="KittiMetric", metric="bbox")
test_evaluator = val_evaluator
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]
