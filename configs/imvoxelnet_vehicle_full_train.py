_base_ = "./imvoxelnet_mmdet3d_1x_camera_full_train.py"

data_root = "/mnt/d/python/study/item3.0/datasets/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure/vehicle-side/"
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file="mmdet3d_camera_infos_train.pkl"))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file="mmdet3d_camera_infos_full_common_val.pkl"))
test_dataloader = val_dataloader
