#!/usr/bin/env bash
# Build llama.cpp from source (CPU, native-tuned for the 275HX) and download
# a Gemma-4-26B-A4B-it GGUF. Targets the "exploit my 96 GB RAM with an MoE"
# track — companion to npu_setup.sh which targets the NPU track.
#
# QUICK ALTERNATIVE (already-installed Ollama, opaque defaults):
#     ollama pull gemma4:26b && ollama run gemma4:26b
# This script gives you explicit quant control + comparable benchmarks instead.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LCPP_DIR="$REPO_DIR/llama.cpp"
MODELS_DIR="$REPO_DIR/models"
# Default: Q4_K_M is the standard sweet spot. ~14 GB for 25.2B params at Q4.
# Bump to Q5_K_M (~17.6 GB) or Q6_K (~20.7 GB) for higher quality — your RAM
# can handle any of them comfortably.
GGUF_REPO="${GGUF_REPO:-bartowski/google_gemma-4-26B-A4B-it-GGUF}"
GGUF_FILE="${GGUF_FILE:-google_gemma-4-26B-A4B-it-Q4_K_M.gguf}"

echo "== 1) Clone & build llama.cpp =="
if [[ ! -d "$LCPP_DIR" ]]; then
    git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "$LCPP_DIR"
else
    echo "  $LCPP_DIR exists; pulling latest"
    git -C "$LCPP_DIR" pull --ff-only
fi

# GGML_NATIVE=ON picks up AVX-VNNI / SHA-NI / etc on the 275HX automatically.
# CUDA is OFF by default — the 5070's 8 GB VRAM helps less than you'd think for
# a 14 GB MoE, and avoiding CUDA dependencies keeps this script standalone.
# To rebuild with partial GPU offload later:
#     cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake -S "$LCPP_DIR" -B "$LCPP_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$LCPP_DIR/build" --config Release -j"$(nproc)" --target llama-cli llama-bench llama-server

LLAMA_CLI="$LCPP_DIR/build/bin/llama-cli"
[[ -x "$LLAMA_CLI" ]] || { echo "ERROR: llama-cli not built at $LLAMA_CLI"; exit 1; }
echo "  built: $LLAMA_CLI"

echo
echo "== 2) Fetch Gemma-4-26B-A4B-it GGUF =="
mkdir -p "$MODELS_DIR"
TARGET="$MODELS_DIR/$GGUF_FILE"
if [[ -f "$TARGET" ]]; then
    echo "  cached: $TARGET"
else
    # Use huggingface_hub from the venv if present, else fall back to wget.
    if [[ -x "$REPO_DIR/.venv/bin/python" ]] && \
       "$REPO_DIR/.venv/bin/python" -c "import huggingface_hub" 2>/dev/null; then
        "$REPO_DIR/.venv/bin/python" - <<PY
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
echo "== 3) Smoke test (10-token greedy) =="
"$LLAMA_CLI" -m "$TARGET" -p "Hello, " -n 10 --no-warmup -t "$(nproc)" 2>&1 \
    | tail -20

echo
echo "Done. Run a real benchmark with:"
echo "    python gemma4_run.py"
echo "Override quant by re-running with e.g.:"
echo "    GGUF_FILE=google_gemma-4-26B-A4B-it-Q5_K_M.gguf ./gemma4_setup.sh"
