#!/usr/bin/env bash
set -u

echo "============================================================"
echo "DAIR-V2X official detector environment check"
echo "============================================================"

if command -v conda >/dev/null 2>&1; then
  echo "[conda]"
  conda info --envs
else
  echo "[WARN] conda command not found"
fi

echo
echo "[python]"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[FAIL] neither python nor python3 is available"
    exit 1
  fi
fi

echo "using: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo
echo "[gpu]"
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/tmp/dair_v2x_nvidia_smi.log 2>&1; then
    head -n 20 /tmp/dair_v2x_nvidia_smi.log
  else
    echo "[FAIL] nvidia-smi failed:"
    cat /tmp/dair_v2x_nvidia_smi.log
  fi
else
  echo "[FAIL] nvidia-smi command not found"
fi

echo
echo "[python packages]"
"$PYTHON_BIN" - <<'PY'
import importlib
import sys

print("python:", sys.version.replace("\n", " "))

packages = [
    ("torch", "torch"),
    ("scipy", "scipy"),
    ("mmcv", "mmcv"),
    ("mmdet", "mmdet"),
    ("mmseg", "mmseg"),
    ("mmdet3d", "mmdet3d"),
    ("open3d", "open3d"),
    ("pypcd", "pypcd"),
]

failed = []

for display_name, module_name in packages:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {display_name}: {version}")
    except Exception as exc:
        print(f"[FAIL] {display_name}: {exc}")
        failed.append(display_name)

try:
    import torch
    print("[torch cuda] compiled cuda:", torch.version.cuda)
    print("[torch cuda] available:", torch.cuda.is_available())
    print("[torch cuda] device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        import subprocess

        code = (
            "import torch; "
            "x=torch.zeros(1, device='cuda'); "
            "torch.cuda.synchronize(); "
            "print(str(x.device))"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            print("[FAIL] torch cuda allocation: timed out")
            failed.append("torch_cuda_alloc")
        else:
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                print("[OK] torch cuda allocation:", output)
            else:
                print("[FAIL] torch cuda allocation:", output)
                failed.append("torch_cuda_alloc")
except Exception as exc:
    print("[FAIL] torch cuda check:", exc)
    failed.append("torch_cuda")

try:
    import mmcv.ops  # noqa: F401
    print("[OK] mmcv.ops import")
except Exception as exc:
    print("[FAIL] mmcv.ops import:", exc)
    failed.append("mmcv.ops")

try:
    import mmdet3d.ops.iou3d.iou3d_cuda  # noqa: F401
    import mmdet3d.ops.voxel.voxel_layer  # noqa: F401
    print("[OK] mmdet3d.ops import")
except Exception as exc:
    print("[FAIL] mmdet3d.ops import:", exc)
    failed.append("mmdet3d.ops")

print("------------------------------------------------------------")
if failed:
    print("[RESULT] NOT READY")
    print("Missing or broken:", ", ".join(failed))
    print("Use dair_v2x_project/notes/setup_dair_official_env.md to build the official detector environment.")
    sys.exit(1)

print("[RESULT] READY")
PY
