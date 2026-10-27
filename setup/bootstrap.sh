#!/usr/bin/env bash
# One-time workspace setup on Ubuntu 22.04 (native or WSL2), laptop or cloud instance.
# An NVIDIA GPU is optional; without one, simulations run the planner on CPU.
# Idempotent: safe to re-run after a failure.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.env"

log() { echo -e "\n\033[1;34m[bootstrap]\033[0m $*"; }

log "System packages"
sudo apt-get update -y
sudo apt-get install -y git tmux htop unzip aria2 build-essential libgl1 libglib2.0-0

log "Data directories under $NUPLAN_HOME"
mkdir -p "$NUPLAN_HOME" 2>/dev/null || { sudo mkdir -p "$NUPLAN_HOME" && sudo chown "$USER":"$USER" "$NUPLAN_HOME"; }
mkdir -p "$NUPLAN_DATA_ROOT"/nuplan-v1.1/{trainval,test,mini} "$NUPLAN_MAPS_ROOT" "$NUPLAN_EXP_ROOT"

log "Conda (miniforge)"
if ! command -v conda >/dev/null && [ ! -x "$HOME/miniforge3/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  "$HOME/miniforge3/bin/conda" init bash
fi
CONDA_BASE="$(command -v conda >/dev/null && conda info --base || echo "$HOME/miniforge3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda env list | grep -q "^$CONDA_ENV " || conda create -y -n "$CONDA_ENV" python=3.9
conda activate "$CONDA_ENV"
# pip >= 24.1 rejects omegaconf 2.1.0rc1's metadata, which nuplan-devkit's hydra-core==1.1.0rc1 pin requires.
pip install "pip<24.1"

clone_pinned() {  # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" fetch --quiet origin && git -C "$2" checkout --quiet "$3"
}
mkdir -p "$EXT_DIR"

log "nuplan-devkit @ $NUPLAN_DEVKIT_COMMIT"
clone_pinned https://github.com/motional/nuplan-devkit.git "$NUPLAN_DEVKIT_ROOT" "$NUPLAN_DEVKIT_COMMIT"
pip install -e "$NUPLAN_DEVKIT_ROOT"
pip install -r "$NUPLAN_DEVKIT_ROOT/requirements.txt"

log "Diffusion-Planner (this repository)"
pip install -e "$DP_ROOT"
# Upstream requirements_torch.txt points at a Tsinghua PyPI mirror (slow from AWS); same pins, official index.
# Blackwell GPUs (RTX 50xx, compute capability 12.x) have no kernels in torch 2.0 / CUDA 11.8 and need
# torch >= 2.7 built for CUDA 12.8. Everything else (including no GPU) keeps upstream's pins.
GPU_CC_MAJOR="$( (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true) | head -1 | cut -d. -f1)"
if [ "${GPU_CC_MAJOR:-0}" -ge 12 ]; then
  log "Compute capability ${GPU_CC_MAJOR}.x GPU: installing torch 2.7.1 (CUDA 12.8)"
  pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
else
  pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
fi
pip install pytorch_lightning==2.0.1 tensorboard==2.11.2 timm==1.0.10 mmengine

log "tuplan_garage @ $TUPLAN_COMMIT (PDM-Closed reference baseline)"
clone_pinned https://github.com/autonomousvision/tuplan_garage.git "$TUPLAN_ROOT" "$TUPLAN_COMMIT"
# --no-deps: its requirements pin nuplan-devkit from PyPI, which would shadow the editable install above.
pip install -e "$TUPLAN_ROOT" --no-deps
pip install scikit-learn==1.2.2 positional-encodings==6.0.1

log "Diffusion Planner checkpoint"
mkdir -p "$DP_ROOT/checkpoints"
for f in args.json model.pth; do
  [ -s "$DP_ROOT/checkpoints/$f" ] || wget -q --show-progress -P "$DP_ROOT/checkpoints" \
    "https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/$f"
done

log "Freezing environment to setup/pip-freeze.txt"
pip freeze > "$PROJ_ROOT/setup/pip-freeze.txt"

log "Sanity checks"
python - <<'EOF'
import torch, nuplan, diffusion_planner, tuplan_garage
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
if torch.cuda.is_available():
    # is_available() is True even when the build has no kernels for this GPU; a real op catches that
    x = torch.randn(256, 256, device="cuda")
    print("cuda matmul ok:", torch.isfinite(x @ x).all().item())
sd = torch.load(f"{__import__('os').environ['DP_ROOT']}/checkpoints/model.pth", map_location="cpu", weights_only=False)
print("checkpoint loaded, top-level keys:", list(sd.keys())[:5] if isinstance(sd, dict) else type(sd))
EOF

log "Done. Next: download nuPlan data with setup/download_nuplan.sh"
