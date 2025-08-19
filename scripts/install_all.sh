# scripts/install_all.sh
#!/usr/bin/env bash
set -euo pipefail

WITH_POSE=${WITH_POSE:-"false"}   # set WITH_POSE=true to install OpenMMLab stack
TORCH_CUDA=${TORCH_CUDA:-"cu121"} # cu118|cu121|cpu
PYTHON_BIN=${PYTHON_BIN:-"python3"}

$PYTHON_BIN -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# Base deps (PyPI default index)
pip install -r requirements.txt

# --- Clone third-party repos ---
mkdir -p third_party
cd third_party
[ -d Person-in-WiFi-3D-repo ] || git clone https://github.com/aiotgroup/Person-in-WiFi-3D-repo
[ -d NeRF2 ] || git clone https://github.com/XPengZhao/NeRF2
[ -d 3D_wifi_scanner ] || git clone https://github.com/Neumi/3D_wifi_scanner
cd ..

# --- Optional: OpenMMLab stack for Person-in-WiFi-3D ---
if [[ "$WITH_POSE" == "true" ]]; then
  # 1) Install Torch from PyTorch index ONLY for torch packages
  case "$TORCH_CUDA" in
    cu118) TORCH_SPEC="torch==2.2.2+cu118 torchvision==0.17.2+cu118";;
    cu121) TORCH_SPEC="torch==2.2.2+cu121 torchvision==0.17.2+cu121";;
    cpu)   TORCH_SPEC="torch==2.2.2+cpu torchvision==0.17.2+cpu";;
    *) echo "Unknown TORCH_CUDA: $TORCH_CUDA"; exit 1;;
  esac
  pip install $TORCH_SPEC --index-url https://download.pytorch.org/whl/$TORCH_CUDA

  # 2) Now install openmim from default PyPI (NO index-url)
  pip install --upgrade openmim

  # 3) Install mmengine/mmcv/mmdet via mim (compatible 2.x)
  mim install "mmengine>=0.10.3"
  mim install "mmcv>=2.0.0"
  mim install "mmdet>=3.2.0"
fi

echo "Install complete."
echo "Tips:"
echo "  WITH_POSE=true TORCH_CUDA=cu121 bash scripts/install_all.sh   # to enable pose bridge"
