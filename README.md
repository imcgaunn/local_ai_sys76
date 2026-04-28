# local_ai_sys76

Personal scratch repo for exploring local LLM inference on a System76 Adder WS.
The goal is to figure out, hands-on, what each of the machine's three
accelerators is actually good for — not to build production tooling.

## Hardware

- Intel Core Ultra 9 275HX (Arrow Lake), 24 cores, AVX2/AVX-VNNI/SHA-NI
- 96 GB RAM
- NVIDIA RTX 5070 Mobile, 8 GB VRAM
- Intel Arc iGPU + Intel Arrow Lake NPU (~13 TOPS class)

## The three accelerator paths

Each path has a different sweet spot. The scripts here let me compare them on
the same machine without committing to any one stack.

| Path | Backend | Sweet spot | Cap |
|---|---|---|---|
| **CUDA** | llama.cpp + RTX 5070 | Fast 7–8B Q4 inference | ~8 GB VRAM limits fully-offloaded model size |
| **CPU + RAM** | llama.cpp on the 275HX | Big models that don't fit anywhere else | Slow per-token but can hold 70B Q4 / 120B Q4 / 26B MoE |
| **NPU** | OpenVINO GenAI on Arrow Lake NPU | Low-power inference, narrow model selection | Mostly INT4 ports of TinyLlama / Phi-3 / Llama-3 8B |

## Contents

### Hardware probing
- `hw_probe.py` — stdlib hardware probe + per-backend model-fit recommendations.

### NPU track (OpenVINO GenAI)
- `npu_setup.sh` — venv + OpenVINO stack + driver checks.
- `npu_run.py` — benchmarks TinyLlama-1.1B-int4 across CPU / iGPU / NPU.

### CPU + RAM track (Gemma-4-26B-A4B MoE)
- `gemma4_setup.sh` — builds llama.cpp (CPU, native-tuned) and downloads
  Gemma-4-26B-A4B-it Q4_K_M (~14 GB).
- `gemma4_run.py` — benchmarks the MoE: 25.2B total params, 3.8B activate per
  token. Tests the "big MoE on big RAM" hypothesis.

### PDF summarization (map-reduce over llama-server)
- `summarize_pdf_setup.sh` + `summarize_pdf.py` — uses the CPU build with
  Gemma-4-26B-A4B. Quality-leaning path; slow but capacious.
- `summarize_pdf_gpu_setup.sh` + `summarize_pdf_gpu.py` — builds llama.cpp with
  CUDA into a separate `build-cuda/` tree and uses Qwen2.5-7B-Instruct Q4_K_M
  fully offloaded to the GPU. Speed-leaning path.
- `high-performance-git-summary.md` — example output from the CPU/Gemma-4
  pipeline summarizing a long technical PDF.

## Conventions

- Each setup script is idempotent: re-running it is safe and just verifies
  state. Models cache to `models/`, llama.cpp clones into `llama.cpp/`, and
  Python deps live in a single shared `.venv/`.
- llama.cpp has two build trees: `llama.cpp/build/` (CPU, native) and
  `llama.cpp/build-cuda/` (CUDA). They coexist so the CPU and GPU paths can
  be exercised side-by-side.
- `models/`, `llama.cpp/`, `.venv/`, and `*.log` are gitignored — everything
  is regenerable from the setup scripts.

## Quickstart

```sh
python3 hw_probe.py                        # see what should fit

./gemma4_setup.sh                          # CPU/RAM path (big MoE)
.venv/bin/python summarize_pdf.py doc.pdf

./summarize_pdf_gpu_setup.sh               # GPU path (Qwen2.5-7B)
.venv/bin/python summarize_pdf_gpu.py doc.pdf

./npu_setup.sh                             # NPU path
.venv/bin/python npu_run.py
```

The GPU setup script requires the CUDA toolkit (`nvcc`); the driver alone is
not enough.
