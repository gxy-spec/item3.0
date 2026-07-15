"""MMDetection3D 1.x ImVoxelNet configuration for DAIR-V2X-C Camera-only.

The model receives a single camera image, but follows the original DAIR-V2X
ImVoxelNet convention and predicts 3D boxes in the corresponding LiDAR frame.
This makes its output directly comparable with the existing LiDAR baselines.
"""

import os


default_scope = "mmdet3d"
data_root = os.environ.get("DAIR_CAMERA_DATA_ROOT", ".")
if not data_root.endswith("/"):
    data_root += "/"

class_names = ["Car"]
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 92.16, 39.68, 1]

model = dict(
    type="ImVoxelNet",
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="mmdet.ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        style="pytorch",
    ),
    neck=dict(
        type="mmdet.FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4,
    ),
    neck_3d=dict(type="OutdoorImVoxelNeck", in_channels=64, out_channels=256),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="AlignedAnchor3DRangeGenerator",
            ranges=[[0, -39.68, -1.78, 92.16, 39.68, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(
            type="mmdet.FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    n_voxels=[248, 288, 12],
    coord_type="LIDAR",
    prior_generator=dict(
        type="AlignedAnchor3DRangeGenerator",
        ranges=[[0, -39.68, -3.08, 92.16, 39.68, 0.76]],
        rotations=[0.0],
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    ),
)

test_pipeline = [
    dict(type="LoadImageFromFileMono3D", backend_args=None),
    dict(type="Resize", scale=(960, 540), keep_ratio=True),
    dict(type="Pack3DDetInputs", keys=["img"]),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file="mmdet3d_camera_infos_val.pkl",
        data_prefix=dict(img="training/image_2"),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d="LiDAR",
        backend_args=None,
    ),
)

visualizer = dict(type="Det3DLocalVisualizer", vis_backends=[dict(type="LocalVisBackend")])
