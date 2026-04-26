#!/usr/bin/env bash
# Set up an OpenVINO GenAI environment for benchmarking LLMs across CPU / iGPU / NPU.
#
# Steps that DO NOT need sudo: venv creation, pip installs.
# Steps that DO need sudo: Intel iGPU runtime (apt), Intel NPU driver (Intel GitHub releases).
# This script does the no-sudo work itself and prints the sudo commands for you to inspect+run.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "== 1) Python venv at $VENV_DIR =="
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

echo "== 2) Python packages (openvino-genai stack) =="
pip install \
    'openvino>=2026.1.0' \
    'openvino-genai>=2026.1.0' \
    'huggingface_hub[cli]>=0.27' \
    'optimum-intel[openvino]>=1.21'

echo "== 3) Device runtime check =="
missing_igpu=0
missing_npu=0

# Intel iGPU: needs Level Zero loader + Intel L0 GPU implementation + (optional) OpenCL ICD.
if ! dpkg -s libze1 >/dev/null 2>&1 || ! dpkg -s libze-intel-gpu1 >/dev/null 2>&1; then
    missing_igpu=1
fi

# Intel NPU: kernel side is /dev/accel/accel0 (already present), userspace is from Intel GitHub.
# Heuristic: the userspace ships libze_intel_vpu.so under /usr or /opt; check for that.
if ! ldconfig -p 2>/dev/null | grep -q 'libze_intel_vpu\|level_zero_npu' \
   && ! find /usr /opt -maxdepth 5 -name 'libze_intel_vpu.so*' 2>/dev/null | grep -q .; then
    missing_npu=1
fi

if [[ $missing_igpu -eq 1 ]]; then
    cat <<'EOF'

  iGPU runtime is NOT installed. To enable OpenVINO device="GPU" (Intel Arc iGPU), run:

      sudo apt install -y intel-opencl-icd libze1 libze-intel-gpu1

EOF
fi

if [[ $missing_npu -eq 1 ]]; then
    cat <<'EOF'

  NPU userspace driver is NOT installed. The kernel module is loaded (/dev/accel/accel0
  is present) but you still need Intel's userspace L0 NPU plugin + compiler + firmware.
  These ship as .deb files from https://github.com/intel/linux-npu-driver/releases
  Pick the latest release matching your Ubuntu (24.04+/26.04). Example install:

      cd /tmp && mkdir npu-driver && cd npu-driver
      wget https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-driver-compiler-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb
      wget https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-fw-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb
      wget https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-level-zero-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb
      sudo dpkg -i ./*.deb
      sudo apt -f install -y   # resolves any missing deps

  ^ Verify the version/URL on the GitHub releases page first; the example above may
  be older than what's current. After install, you may need to log out + log back in
  (or reboot) for the L0 loader to find the new plugin.

EOF
fi

if [[ $missing_igpu -eq 0 && $missing_npu -eq 0 ]]; then
    echo "  iGPU + NPU runtimes look present."
fi

echo
echo "== 4) Verify OpenVINO sees the devices =="
python - <<'PY'
import openvino as ov
core = ov.Core()
print("  available devices:", core.available_devices)
for d in core.available_devices:
    try:
        print(f"    {d:6} -> {core.get_property(d, 'FULL_DEVICE_NAME')}")
    except Exception as e:
        print(f"    {d:6} -> (FULL_DEVICE_NAME query failed: {e})")
PY

echo
echo "Done. Activate the venv with:  source $VENV_DIR/bin/activate"
echo "Then run:                       python npu_run.py"
