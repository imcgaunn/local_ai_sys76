#!/usr/bin/env bash
# Build llama.cpp with CUDA into a separate build-cuda/ tree (so the existing
# CPU build under llama.cpp/build/ keeps working for the Gemma-4-26B-A4B path),
# download Qwen2.5-7B-Instruct Q4_K_M, and ensure the venv has the PDF deps.
#
# Companion to summarize_pdf_setup.sh (CPU/Gemma-4 path). The 5070 has 8 GB
# VRAM, which fits Qwen2.5-7B Q4_K_M (~4.4 GB) plus a 32k KV cache (~1.8 GB)
# with comfortable headroom for activations.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LCPP_DIR="$REPO_DIR/llama.cpp"
BUILD_DIR="$LCPP_DIR/build-cuda"
MODELS_DIR="$REPO_DIR/models"
VENV_DIR="$REPO_DIR/.venv"

GGUF_REPO="${GGUF_REPO:-bartowski/Qwen2.5-7B-Instruct-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen2.5-7B-Instruct-Q4_K_M.gguf}"

echo "== 1) Pre-flight checks"
if ! command -v nvcc >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: nvcc not found. Install the CUDA toolkit before running this script.
The NVIDIA driver alone is not enough to build llama.cpp with CUDA.

On Ubuntu, either:
    sudo apt install nvidia-cuda-toolkit          # distro version (may lag)
or follow https://developer.nvidia.com/cuda-downloads for a current version
matching your driver (you have CUDA 13.2 runtime per nvidia-smi).

Then re-run this script.
EOF
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "== 2) Clone llama.cpp (if needed) and build with CUDA"
if [[ ! -d "$LCPP_DIR" ]]; then
    git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$LCPP_DIR"
fi

# Separate build dir so we don't clobber the CPU build at llama.cpp/build/.
cmake -S "$LCPP_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)" \
    --target llama-cli llama-bench llama-server

LLAMA_SERVER="$BUILD_DIR/bin/llama-server"
[[ -x "$LLAMA_SERVER" ]] || { echo "ERROR: llama-server not built at $LLAMA_SERVER"; exit 1; }
echo "  built: $LLAMA_SERVER"

echo
echo "== 3) Fetch Qwen2.5-7B-Instruct GGUF"
mkdir -p "$MODELS_DIR"
TARGET="$MODELS_DIR/$GGUF_FILE"
if [[ -f "$TARGET" ]]; then
    echo "  cached: $TARGET"
else
    if [[ -x "$VENV_DIR/bin/python" ]] && \
       "$VENV_DIR/bin/python" -c "import huggingface_hub" 2>/dev/null; then
        "$VENV_DIR/bin/python" - <<PY
from huggingface_hub import hf_hub_download
import shutil, pathlib
p = hf_hub_download(repo_id="$GGUF_REPO", filename="$GGUF_FILE")
dest = pathlib.Path("$TARGET")
shutil.copy(p, dest)
print(f"  downloaded -> {dest}")
PY
    else
        URL="https://huggingface.co/$GGUF_REPO/resolve/main/$GGUF_FILE"
        echo "  wget $URL"
        wget -O "$TARGET" "$URL"
    fi
fi
ls -lh "$TARGET"

echo
echo "== 4) Ensure venv has PDF deps"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null
pip install pymupdf requests

echo
echo "== 5) Smoke test (10-token greedy on GPU)"
"$BUILD_DIR/bin/llama-cli" -m "$TARGET" -p "Hello, " -n 10 \
    -ngl 99 --no-warmup 2>&1 | tail -20

echo
echo "Done. Run a summary with:"
echo "    $VENV_DIR/bin/python summarize_pdf_gpu.py path/to/doc.pdf"
