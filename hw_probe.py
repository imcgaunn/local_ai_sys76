#!/usr/bin/env python3
"""Probe local hardware and recommend LLM model sizes that should fit.

Stdlib only. Run: python3 hw_probe.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field


GIB = 1024 ** 3

# GGUF quant byte cost per 1B parameters (empirical, dense models).
# Q4_K_M is the default sweet spot for most practical local inference.
QUANT_GB_PER_B = {
    "Q4_K_M": 0.55,
    "Q5_K_M": 0.70,
    "Q8_0":   1.05,
    "FP16":   2.00,
}

# KV-cache cost varies with arch + context. 20% headroom on weights is a
# reasonable conservative buffer at 4-8k context for 7-70B class models.
HEADROOM = 0.20


@dataclass
class GPU:
    name: str
    vram_gb: float
    backend: str  # "cuda", "intel-igpu", "intel-npu"


@dataclass
class Profile:
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_flags: list[str] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_avail_gb: float = 0.0
    swap_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpus: list[GPU] = field(default_factory=list)
    npu_present: bool = False
    has_ollama: bool = False
    has_llama_cpp: bool = False
    has_openvino: bool = False


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def probe() -> Profile:
    p = Profile()

    # CPU
    lscpu = _run(["lscpu"])
    for line in lscpu.splitlines():
        if line.startswith("Model name:"):
            p.cpu_model = line.split(":", 1)[1].strip()
        elif line.startswith("CPU(s):") and p.cpu_cores == 0:
            try:
                p.cpu_cores = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Flags:"):
            p.cpu_flags = line.split(":", 1)[1].strip().split()

    # Memory
    try:
        with open("/proc/meminfo") as f:
            mem = f.read()
        def kb(key: str) -> float:
            m = re.search(rf"^{key}:\s+(\d+)\s+kB", mem, re.M)
            return int(m.group(1)) * 1024 / GIB if m else 0.0
        p.ram_total_gb = kb("MemTotal")
        p.ram_avail_gb = kb("MemAvailable")
        p.swap_gb = kb("SwapTotal")
    except OSError:
        pass

    # Disk free on the user's home filesystem
    try:
        st = os.statvfs(os.path.expanduser("~"))
        p.disk_free_gb = st.f_bavail * st.f_frsize / GIB
    except OSError:
        pass

    # NVIDIA GPUs
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits"])
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                try:
                    p.gpus.append(GPU(parts[0], float(parts[1]) / 1024, "cuda"))
                except ValueError:
                    pass

    # Intel iGPU + NPU via lspci
    lspci = _run(["lspci"])
    for line in lspci.splitlines():
        low = line.lower()
        is_display = "vga compatible controller" in low or "display controller" in low
        if "intel" in low and is_display:
            p.gpus.append(GPU(line.split(": ", 1)[-1].strip(), 0.0, "intel-igpu"))
        if "npu" in low and "intel" in low:
            p.npu_present = True
    if not p.npu_present and os.path.exists("/dev/accel/accel0"):
        p.npu_present = True

    # Tooling
    p.has_ollama = bool(shutil.which("ollama"))
    p.has_llama_cpp = bool(shutil.which("llama-cli") or shutil.which("llama-server"))
    try:
        import importlib.util
        p.has_openvino = importlib.util.find_spec("openvino") is not None
    except ImportError:
        p.has_openvino = False

    return p


def fit_size_b(budget_gb: float, quant: str = "Q4_K_M") -> float:
    """Max dense-model parameter count (in B) that fits a memory budget."""
    return (budget_gb * (1 - HEADROOM)) / QUANT_GB_PER_B[quant]


# Concrete 2026-vintage model picks per size band. Dense unless marked MoE.
DENSE_PICKS = [
    (1.0,   "Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct"),
    (3.0,   "Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, Phi-3.5-mini (3.8B)"),
    (7.0,   "Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct"),
    (8.0,   "Llama-3.1-8B-Instruct (the workhorse 8B)"),
    (12.0,  "Gemma-3-12B-it, Phi-4 (14B)"),
    (14.0,  "Qwen2.5-14B-Instruct, Phi-4 (14B)"),
    (24.0,  "Mistral-Small-3-24B-Instruct, Gemma-3-27B-it"),
    (32.0,  "Qwen2.5-32B-Instruct, QwQ-32B"),
    (70.0,  "Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct"),
]

# MoE picks: total params drive RAM, active params drive decode speed.
# Listed as (total_b, active_b, name).
MOE_PICKS = [
    (30.0,   3.0,  "Qwen3-30B-A3B  (30B total, 3B active — lightning on CPU)"),
    (109.0, 17.0,  "Llama-4-Scout-17B-16E  (~109B total, 17B active)"),
    (120.0,  5.0,  "GPT-OSS-120B  (120B total, ~5B active — decode like a 5B)"),
    (235.0, 22.0,  "Qwen3-235B-A22B  (235B total, 22B active — Q4 needs >96GB; Q3 marginal)"),
]


def fmt_gb(x: float) -> str:
    return f"{x:6.1f} GB"


def report(p: Profile) -> None:
    print("=" * 72)
    print("HARDWARE")
    print("=" * 72)
    print(f"  CPU      : {p.cpu_model} ({p.cpu_cores} threads)")
    avx_vnni = "avx_vnni" in p.cpu_flags
    avx512 = any(f.startswith("avx512") for f in p.cpu_flags)
    print(f"             AVX-VNNI={avx_vnni}  AVX-512={avx512}  "
          f"SHA-NI={'sha_ni' in p.cpu_flags}")
    print(f"  RAM      : {fmt_gb(p.ram_total_gb)} total, "
          f"{fmt_gb(p.ram_avail_gb)} available, {fmt_gb(p.swap_gb)} swap")
    print(f"  Disk free: {fmt_gb(p.disk_free_gb)} (~{int(p.disk_free_gb // 50)} "
          f"big models @ 50GB each)")
    for g in p.gpus:
        v = f"{g.vram_gb:.1f} GB VRAM" if g.vram_gb else "shared system RAM"
        print(f"  GPU      : [{g.backend:10}] {g.name}  ({v})")
    print(f"  NPU      : {'present (Intel, /dev/accel/accel0)' if p.npu_present else 'none detected'}")
    print(f"  Tooling  : ollama={p.has_ollama}  llama.cpp={p.has_llama_cpp}  "
          f"openvino={p.has_openvino}")
    print()

    # Per-backend memory budgets. We're conservative: assume the user wants
    # to keep the desktop usable, so reserve some RAM for the OS.
    print("=" * 72)
    print("MEMORY BUDGETS  (Q4_K_M, 20% headroom for KV cache + activations)")
    print("=" * 72)

    cuda_gpu = next((g for g in p.gpus if g.backend == "cuda"), None)
    if cuda_gpu:
        b = cuda_gpu.vram_gb
        print(f"  CUDA (full offload): {fmt_gb(b)}  →  up to ~{fit_size_b(b):.1f}B dense")
        print( "                       (use llama.cpp -ngl 999 or vLLM if it fits)")

    cpu_budget = max(p.ram_total_gb - 8, 8)  # leave 8 GB for the OS
    print(f"  CPU + system RAM   : {fmt_gb(cpu_budget)}  →  up to ~{fit_size_b(cpu_budget):.1f}B dense")
    print( "                       (this is your real superpower — MoE shines here)")

    if cuda_gpu:
        hybrid = cuda_gpu.vram_gb + cpu_budget
        print(f"  CUDA + CPU split   : {fmt_gb(hybrid)}  →  up to ~{fit_size_b(hybrid):.1f}B dense")
        print( "                       (llama.cpp -ngl N, partial offload)")

    if p.npu_present:
        print("  Intel NPU          : limited; OpenVINO GenAI supports a curated set")
        print("                       (Llama-3 8B, Phi-3, TinyLlama at INT4 — see below)")
    print()

    # Concrete recommendations
    print("=" * 72)
    print("DENSE MODELS THAT FIT  (Q4_K_M GGUF)")
    print("=" * 72)
    for size_b, name in DENSE_PICKS:
        weights_gb = size_b * QUANT_GB_PER_B["Q4_K_M"]
        budget_gb = weights_gb * (1 + HEADROOM)
        gpu_ok = cuda_gpu and budget_gb <= cuda_gpu.vram_gb
        cpu_ok = budget_gb <= cpu_budget
        marker = "GPU" if gpu_ok else ("CPU" if cpu_ok else " — ")
        print(f"  [{marker}]  ~{size_b:5.1f}B  ({fmt_gb(weights_gb)})  {name}")
    print()

    print("=" * 72)
    print("MoE MODELS  (RAM-heavy, but decode speed scales with ACTIVE params)")
    print("=" * 72)
    print("  These are the models where your 96 GB really pays off.")
    print()
    for total_b, active_b, name in MOE_PICKS:
        weights_gb = total_b * QUANT_GB_PER_B["Q4_K_M"]
        budget_gb = weights_gb * (1 + HEADROOM)
        ok = budget_gb <= cpu_budget
        marker = "FITS" if ok else "TIGHT"
        print(f"  [{marker}]  total={total_b:5.1f}B  active={active_b:4.1f}B  "
              f"({fmt_gb(weights_gb)} @ Q4)  {name}")
    print()

    print("=" * 72)
    print("SUGGESTED FIRST EXPERIMENTS")
    print("=" * 72)
    print("  1. GPU baseline (fast, familiar):")
    print("       ollama run llama3.1:8b-instruct-q4_K_M")
    print("  2. RAM flex (slow but capable — does the 96 GB actually deliver?):")
    print("       ollama run llama3.3:70b-instruct-q4_K_M")
    print("  3. MoE on CPU (the interesting one — should feel fast despite size):")
    print("       ollama run gpt-oss:120b   # or qwen3:30b-a3b for a milder test")
    print("  4. NPU exploration (most exotic, narrowest model selection):")
    print("       pip install openvino-genai  # then load an INT4 OV model")
    print("       https://huggingface.co/OpenVINO has prebuilt NPU-targeted models")


if __name__ == "__main__":
    report(probe())
