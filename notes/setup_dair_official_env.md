# DAIR-V2X Official Detector Environment

This note is for Stage 2 of Experiment 1: running the official DAIR-V2X Vehicle LiDAR-only PointPillars model.

The item3.0 environment `dair-baseline` is only for preprocessing, coordinate conversion, and evaluation. It is not expected to run `DAIR-V2X/v2x/eval.py`.

## What The Official Code Needs

The local DAIR-V2X documentation states:

- `mmdetection3d==0.17.1`
- a modified Python 3 compatible `pypcd`

The official detector runtime also needs a compatible old OpenMMLab stack:

- Python 3.8 is a safer base than Python 3.10.
- PyTorch 1.x is a safer base than PyTorch 2.x.
- `mmcv-full`, not plain `mmcv`, is needed because the detector uses compiled CUDA/C++ ops.
- `mmdet`, `mmseg`, `mmdet3d`, `scipy`, `pyquaternion`, and `pypcd` must import.
- `open3d` is also required by the official dataset file I/O code.
- GPU access must work before running PointPillars inference.

Current local observations:

- `dair-baseline` is ready for item3.0 preprocessing/evaluation, but not for official detector inference.
- `torch-gpu-test` has a very new Torch stack; it is too new for `mmdetection3d==0.17.1`.
- `dair-official-test` now imports `torch/scipy/mmcv/mmdet/mmseg/mmdet3d/open3d/pypcd`, and `mmcv.ops` plus `mmdet3d.ops` import.
- The local RTX 5060 Ti is not compatible with the old official PyTorch 1.10/CUDA 11.3 runtime. PyTorch reports the GPU as `sm_120`, while the installed PyTorch build supports up to `sm_86`; a real CUDA tensor allocation times out. This means official DAIR-V2X MMDetection3D inference cannot be completed on this GPU with the historical official stack.

## Preflight Check

From any candidate official environment:

```bash
cd /home/gxy/projects/item3
conda activate <candidate-env>
bash dair_v2x_project/scripts/check_official_env.sh
```

The environment is ready only when the script ends with:

```text
[RESULT] READY
```

## Recommended Environment Shape

Use a dedicated environment instead of reusing `dair-baseline` or `torch-gpu-test`.

```bash
conda create -n dair-mmdet3d python=3.8 -y
conda activate dair-mmdet3d
```

Then install a PyTorch 1.x build that matches your available CUDA driver/runtime. Choose the exact CUDA build according to your machine.

Examples:

```bash
# Example for CUDA 11.1 era environments.
# Adjust this if your machine uses a different CUDA/PyTorch combination.
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
```

Install the OpenMMLab stack compatible with MMDetection3D 0.17.1. The exact `mmcv-full` wheel must match both PyTorch and CUDA.

Example pattern:

```bash
pip install -U openmim
mim install mmcv-full==1.3.17
pip install mmdet==2.14.0 mmsegmentation==0.14.1
pip install mmdet3d==0.17.1
pip install scipy pyquaternion pypcd
```

If `pypcd` fails under Python 3, use the modified version mentioned by DAIR-V2X:

```bash
cd /tmp
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```

## If Only `mmdet3d` Is Missing

Stay in the official detector environment that already passes the other checks, then install MMDetection3D 0.17.1:

```bash
cd /home/gxy/projects/item3
conda activate dair-official-test
pip install mmdet3d==0.17.1
python -c "import mmdet3d; print(mmdet3d.__version__)"
bash dair_v2x_project/scripts/check_official_env.sh
```

If the wheel install is unavailable or the import still fails, install the matching source tag:

```bash
cd /tmp
git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.17.1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d-v0171
cd /tmp/mmdetection3d-v0171
pip install -v -e .
```

If `git clone` fails, do not run `pip install -v -e .` from `/tmp`; there is no Python project there. Fix the clone first, then enter the cloned directory.

Then rerun:

```bash
cd /home/gxy/projects/item3
bash dair_v2x_project/scripts/check_official_env.sh
```

After installation:

```bash
cd /home/gxy/projects/item3
bash dair_v2x_project/scripts/check_official_env.sh
```

## GPU Check

Before running official inference, this must work:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch; x=torch.zeros(1, device='cuda'); torch.cuda.synchronize(); print(x.device)"
```

Expected:

```text
True 1
cuda:0
```

or a positive GPU count.

If `nvidia-smi` says GPU access is blocked by the operating system, fix WSL/Linux GPU access first. Package installation alone will not make official inference run.

If `torch.cuda.is_available()` is true but CUDA allocation hangs or fails, the GPU/runtime pair is still not usable. This is the current local RTX 5060 Ti situation with PyTorch 1.10/CUDA 11.3.

## Run Official Vehicle LiDAR-only Inference

Once the preflight check passes:

```bash
cd /home/gxy/projects/item3/DAIR-V2X/v2x
mkdir -p ../cache/vic-vehonly-lidar/result ../cache/vic-vehonly-lidar/veh/lidar

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --input ../data/DAIR-V2X/cooperative-vehicle-infrastructure \
  --output ../cache/vic-vehonly-lidar \
  --model veh_only \
  --dataset vic-async \
  --k 0 \
  --split val \
  --split-data-path ../data/split_datas/example-cooperative-split-data.json \
  --veh-config-path ../configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v.py \
  --veh-model-path ../configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth \
  --device 0 \
  --pred-classes car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1 \
  --overwrite-cache
```

Then return to `dair-baseline` for Stage 3 and Stage 4:

```bash
cd /home/gxy/projects/item3
conda activate dair-baseline

python dair_v2x_project/export_vehicle_lidar_predictions.py \
  --input DAIR-V2X/cache/vic-vehonly-lidar \
  --source-format dair-official-pkl \
  --score-threshold 0.1

python dair_v2x_project/convert_vehicle_lidar_predictions_to_world.py
python dair_v2x_project/check_predictions_world.py
python dair_v2x_project/evaluate_and_visualize_vehicle_lidar_baseline.py
```
