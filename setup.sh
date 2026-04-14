#!/bin/bash
# =============================================================================
#  Double Pendulum RL Project — Environment Setup
#  Leonardo Luigi Pepe, 2157562
#
#  Usage (from project root, where src/ lives):
#    chmod +x setup.sh && ./setup.sh
#
#  Works on: Linux x86_64, macOS (Intel + Apple Silicon)
#  Requires: conda (preferred) OR python3 + pip
# =============================================================================

set -e

CONDA_ENV_NAME="double_pendulum"
PYTHON_VERSION="3.10"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
section() { echo -e "\n${GREEN}=== $1 ===${NC}"; }

echo "============================================================"
echo "   Double Pendulum RL Project — Setup Script"
echo "============================================================"

# ── Sanity check: must be run from project root ───────────────────────────────
if [ ! -d "./src" ]; then
    echo -e "${RED}[✗] 'src/' not found. Run this script from the project root.${NC}"
    exit 1
fi
info "Project root OK — src/ found."

# ── STEP 1: Python environment ────────────────────────────────────────────────
section "STEP 1/3 — Python environment"

if command -v conda &> /dev/null; then
    info "conda found."
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if conda env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
        warn "Conda env '${CONDA_ENV_NAME}' already exists — skipping creation."
    else
        info "Creating conda env '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}..."
        conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
    fi

    conda activate "$CONDA_ENV_NAME"
    info "Activated: ${CONDA_ENV_NAME}"

else
    warn "conda not found — using Python venv."
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[✗] python3 not found. Install Python ${PYTHON_VERSION}+.${NC}"
        exit 1
    fi
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    else
        warn ".venv already exists — reusing."
    fi
    source .venv/bin/activate
    info "Activated: .venv"
fi

pip install --upgrade pip --quiet

# ── STEP 2: Install double_pendulum from local src/ ───────────────────────────
section "STEP 2/3 — double_pendulum (local src/)"

info "Installing double_pendulum base package from ./src ..."
pip install --quiet -e "./src"
info "double_pendulum installed."

# ── STEP 3: Install project-specific dependencies ─────────────────────────────
section "STEP 3/3 — Project dependencies"

# PyTorch — auto-detect CUDA or CPU
if python -c "import torch" &>/dev/null; then
    warn "torch already installed — skipping."
else
    if nvidia-smi &>/dev/null 2>&1; then
        info "CUDA GPU detected — installing PyTorch (CUDA 12.1)..."
        pip install --quiet torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu121
    else
        info "No GPU — installing PyTorch (CPU)..."
        pip install --quiet torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cpu
    fi
fi

info "Installing RL packages..."
pip install --quiet \
    "gymnasium==1.2.3" \
    "stable-baselines3==2.7.1" \
    "tensorboard" \
    "cloudpickle" \
    "dill"

info "Installing evolutionary optimization..."
pip install --quiet \
    "evotorch==0.6.1" \
    "cma" \
    "ray[default]==2.54.1"

info "Installing utilities..."
pip install --quiet \
    "rich" \
    "pydot" \
    "pandas" \
    "tqdm"

# ── Final verification ────────────────────────────────────────────────────────
section "Verification"

python - <<'PYCHECK'
import sys

checks = {
    "double_pendulum": "double_pendulum (local src/)",
    "torch":           "torch",
    "numpy":           "numpy",
    "scipy":           "scipy",
    "matplotlib":      "matplotlib",
    "gymnasium":       "gymnasium",
    "stable_baselines3": "stable_baselines3",
    "ray":             "ray",
    "evotorch":        "evotorch",
    "cma":             "cma",
    "cv2":             "opencv-python",
    "rich":            "rich",
    "tqdm":            "tqdm",
}

all_ok = True
for module, label in checks.items():
    try:
        __import__(module)
        print(f"  ✓  {label}")
    except ImportError:
        print(f"  ✗  {label}  <-- MISSING", file=sys.stderr)
        all_ok = False

if not all_ok:
    print("\n[!] Some packages missing — check errors above.", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll packages OK.")
PYCHECK

echo ""
echo "============================================================"
echo "   Setup complete!"
echo ""
echo "   Next steps:"
if command -v conda &> /dev/null; then
echo "     conda activate ${CONDA_ENV_NAME}"
else
echo "     source .venv/bin/activate"
fi
echo "     python evaluate.py"
echo "     python baseline.py"
echo "============================================================"