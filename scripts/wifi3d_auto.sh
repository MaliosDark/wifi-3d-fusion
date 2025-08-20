#!/usr/bin/env bash
# ============================================================================
# wifi3d_auto.sh — One-shot automation for Person-in-WiFi-3D in this project
#
# It will (idempotent):
#   1) Download the Person-in-WiFi-3D dataset (SDP8 mirror) if missing.
#   2) Link it into third_party/Person-in-WiFi-3D-repo/data/wifipose.
#   3) Ensure a local Python venv + base deps + OpenMMLab pose deps.
#   4) Train if no checkpoint is found (work_dirs/petr_wifi/*.pth).
#   5) Run inference with the found checkpoint.
#
# Usage (run from project root):
#   bash scripts/wifi3d_auto.sh
#   bash scripts/wifi3d_auto.sh --no-train          # skip training (if a .pth already exists)
#   bash scripts/wifi3d_auto.sh --no-infer          # stop after preparing/training
#   bash scripts/wifi3d_auto.sh --dataset-url URL   # override dataset mirror
#   bash scripts/wifi3d_auto.sh --dataset-dir DIR   # override local dataset path
#
# Env:
#   TORCH_CUDA=cu121|cu118|cpu   (default: cu121)
#   PYTHON_BIN=python3           (default)
# ============================================================================

set -euo pipefail

# ---- Defaults / Config ----
REPO="third_party/Person-in-WiFi-3D-repo"
CFG="${REPO}/config/wifi/petr_wifi.py"
WORK_DIR="${REPO}/work_dirs/petr_wifi"
CKPT_DEFAULT="${WORK_DIR}/latest.pth"

DATA_URL_DEFAULT="http://sdp8.comp.nus.edu.sg/dataset/Person-in-WiFi-3D/"
DATA_DIR_DEFAULT="datasets/person_in_wifi3d"
SYMLINK="${REPO}/data/wifipose"

TORCH_CUDA="${TORCH_CUDA:-cu121}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_URL="${DATA_URL_DEFAULT}"
DATA_DIR="${DATA_DIR_DEFAULT}"
DO_TRAIN=true
DO_INFER=true

# ---- Helpers ----
info(){ echo "[INFO] $*"; }
warn(){ echo "[WARN] $*"; }
die(){  echo "[ERROR] $*" >&2; exit 1; }

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-train) DO_TRAIN=false; shift;;
    --no-infer) DO_INFER=false; shift;;
    --dataset-url) DATA_URL="$2"; shift 2;;
    --dataset-dir) DATA_DIR="$2"; shift 2;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) die "Unknown arg: $1 (use --help)";;
  esac
done

# ---- Validate project layout ----
[[ -f "scripts/install_all.sh" ]] || die "Missing scripts/install_all.sh (run from project root)."
[[ -d "third_party" ]] || die "Missing third_party directory."
[[ -d "$REPO" ]] || die "Missing repo: $REPO"
[[ -f "$CFG"  ]] || die "Missing config: $CFG"

# ---- Ensure wget ----
ensure_wget(){
  if ! command -v wget >/dev/null 2>&1; then
    warn "wget not found; attempting apt-get install..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -y && sudo apt-get install -y wget
    else
      die "wget not available and cannot auto-install on this OS."
    fi
  fi
}

# ---- Download dataset (idempotent) ----
download_dataset(){
  ensure_wget
  mkdir -p "$DATA_DIR"
  pushd "$DATA_DIR" >/dev/null
  info "Mirroring dataset from: $DATA_URL"
  wget \
    --recursive \
    --no-parent \
    --no-host-directories \
    --cut-dirs=2 \
    --continue \
    --timestamping \
    --reject "index.html*" \
    --retry-connrefused \
    --timeout=30 \
    --tries=5 \
    "$DATA_URL"
  popd >/dev/null

  local n
  n=$(find "$DATA_DIR" -type f | wc -l | tr -d ' ')
  [[ "$n" -ge 5 ]] || die "Dataset looks empty or blocked (files: $n). Check URL/network."
  info "Dataset ready at: $DATA_DIR (files: $n)"
}

# ---- Link dataset into third_party repo ----
link_dataset(){
  mkdir -p "${REPO}/data"
  if [[ -L "$SYMLINK" ]]; then
    info "Symlink already present: $SYMLINK"
  elif [[ -e "$SYMLINK" && ! -L "$SYMLINK" ]]; then
    warn "$SYMLINK exists and is not a symlink; leaving as-is."
  else
    ln -s "../../${DATA_DIR}" "$SYMLINK"
    info "Linked: $SYMLINK -> $DATA_DIR"
  fi
}

# ---- Ensure venv + install pose stack ----
ensure_env(){
  if [[ ! -d ".venv" ]]; then
    info "Creating .venv with ${PYTHON_BIN} ..."
    ${PYTHON_BIN} -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip -q install -U pip setuptools wheel
  info "Installing base + pose deps (WITH_POSE=true, TORCH_CUDA=${TORCH_CUDA}) ..."
  WITH_POSE=true TORCH_CUDA="${TORCH_CUDA}" bash scripts/install_all.sh
}

# ---- Train if no checkpoint is available ----
train_if_needed(){
  if ls "${WORK_DIR}"/*.pth >/dev/null 2>&1; then
    info "Checkpoint found in ${WORK_DIR} — skipping training."
    return 0
  fi
  $DO_TRAIN || { info "--no-train set; skipping training."; return 0; }

  info "Launching training with config: ${CFG}"
  pushd "$REPO" >/dev/null
  if [[ -f "tools/train.py" ]]; then
    python tools/train.py "${CFG}"
  elif [[ -f "opera/.mim/tools/train.py" ]]; then
    python opera/.mim/tools/train.py "${CFG}"
  else
    popd >/dev/null
    die "Training script not found (expected tools/train.py or opera/.mim/tools/train.py)."
  fi
  popd >/dev/null
  info "Training process launched."
}

# ---- Resolve checkpoint path ----
find_ckpt(){
  if [[ -f "$CKPT_DEFAULT" ]]; then
    echo "$CKPT_DEFAULT"; return
  fi
  if [[ -d "$WORK_DIR" ]]; then
    local cand
    cand=$(ls -1t "${WORK_DIR}"/*.pth 2>/dev/null | head -n1 || true)
    if [[ -n "$cand" ]]; then echo "$cand"; return; fi
  fi
  if [[ -f "env/weights/pwifi3d.pth" ]]; then
    echo "env/weights/pwifi3d.pth"; return
  fi
  echo ""
}

# ---- Run inference via bridge ----
run_inference(){
  $DO_INFER || { info "--no-infer set; done."; return 0; }
  local ckpt; ckpt="$(find_ckpt)"
  [[ -n "$ckpt" ]] || die "No checkpoint found for inference (train first or place a .pth)."

  info "Running inference with checkpoint: $ckpt"
  if [[ -d ".venv" ]]; then source .venv/bin/activate; fi
  python -m src.bridges.pwifi3d_runner "$REPO" "$CFG" "$ckpt"
}

# ---- Main flow ----
info "WiFi-3D auto pipeline starting..."
info "TORCH_CUDA=${TORCH_CUDA}  PYTHON_BIN=${PYTHON_BIN}"
info "DATASET_URL=${DATA_URL}  DATASET_DIR=${DATA_DIR}"

download_dataset
link_dataset
ensure_env
train_if_needed
run_inference

info "All done. Adapter prepared, model ready, inference executed."
