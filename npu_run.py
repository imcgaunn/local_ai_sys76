#!/usr/bin/env python3
"""Benchmark an OpenVINO LLM across CPU / iGPU / NPU on this machine.

Downloads a pre-converted OpenVINO IR model from HuggingFace, then runs the
same prompt on each available device, measuring time-to-first-token (prefill)
and decode tokens/sec.

Run after `npu_setup.sh`:
    source .venv/bin/activate
    python npu_run.py                              # default: TinyLlama-1.1B int4
    python npu_run.py --model OpenVINO/Phi-3.5-mini-instruct-int4-ov
    python npu_run.py --devices CPU,NPU            # subset
    python npu_run.py --tokens 256 --prompt "Why is the sky blue?"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download


# Defaults sized for a first NPU run: TinyLlama is the canonical "definitely
# fits on NPU" model. Phi-3.5-mini and Llama-3.1-8B (gated) are nearby steps up.
DEFAULT_MODEL = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
DEFAULT_PROMPT = "In one paragraph, explain how a transformer language model generates text."
MODELS_DIR = Path(__file__).parent / "models"

# NPU LLM compile needs static prompt/response budgets. Conservative defaults.
NPU_KW = {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 128}


def fetch_model(repo_id: str) -> Path:
    local_dir = MODELS_DIR / repo_id.replace("/", "__")
    if not local_dir.exists() or not any(local_dir.iterdir()):
        print(f"  downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir),
                          allow_patterns=["*.xml", "*.bin", "*.json", "*.txt",
                                          "*.model", "tokenizer*"])
    else:
        print(f"  cached at {local_dir}")
    return local_dir


class TimingStreamer:
    """Captures per-token timestamps so we can compute TTFT and decode rate."""
    def __init__(self) -> None:
        self.first_token_t: float | None = None
        self.last_token_t: float | None = None
        self.token_count = 0
        self.text_chunks: list[str] = []

    def __call__(self, chunk: str) -> bool:
        now = time.perf_counter()
        if self.first_token_t is None:
            self.first_token_t = now
        self.last_token_t = now
        self.token_count += 1
        self.text_chunks.append(chunk)
        return False  # keep generating


def run_one(model_path: Path, device: str, prompt: str, max_new_tokens: int) -> dict:
    print(f"\n--- {device} ---")
    kwargs = NPU_KW if device == "NPU" else {}
    t_load = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(model_path), device, **kwargs)
    load_s = time.perf_counter() - t_load
    print(f"  pipeline ready in {load_s:.1f} s (compile + load)")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False  # greedy; deterministic for benchmarking

    streamer = TimingStreamer()
    t_start = time.perf_counter()
    pipe.generate(prompt, cfg, streamer)
    t_end = time.perf_counter()

    ttft = (streamer.first_token_t - t_start) if streamer.first_token_t else float("nan")
    decode_s = (streamer.last_token_t - streamer.first_token_t) \
        if streamer.first_token_t and streamer.last_token_t and streamer.token_count > 1 \
        else 0.0
    decode_tps = (streamer.token_count - 1) / decode_s if decode_s > 0 else float("nan")

    print(f"  TTFT (prefill)   : {ttft*1000:7.1f} ms")
    print(f"  decode tokens/sec: {decode_tps:7.2f}  ({streamer.token_count} tokens in {decode_s:.2f}s)")
    print(f"  total wall time  : {(t_end-t_start):7.2f} s")
    return {
        "device": device, "load_s": load_s, "ttft_ms": ttft * 1000,
        "decode_tps": decode_tps, "tokens": streamer.token_count,
        "total_s": t_end - t_start,
        "preview": "".join(streamer.text_chunks)[:120].replace("\n", " "),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HF repo id of an OpenVINO IR model (default: {DEFAULT_MODEL})")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokens", type=int, default=128, help="max new tokens")
    ap.add_argument("--devices", default="auto",
                    help="comma-separated subset of CPU,GPU,NPU (default: auto = all available)")
    args = ap.parse_args()

    available = ov.Core().available_devices
    print("OpenVINO sees:", available)

    if args.devices == "auto":
        wanted = [d for d in ("CPU", "GPU", "NPU") if d in available]
    else:
        wanted = [d.strip().upper() for d in args.devices.split(",") if d.strip()]
        for d in wanted:
            if d not in available:
                print(f"  WARN: device {d} not present; will skip")
        wanted = [d for d in wanted if d in available]

    if not wanted:
        raise SystemExit("No usable devices.")

    print(f"\nFetching model: {args.model}")
    model_path = fetch_model(args.model)

    results = []
    for d in wanted:
        try:
            results.append(run_one(model_path, d, args.prompt, args.tokens))
        except Exception as e:
            print(f"  FAILED on {d}: {type(e).__name__}: {e}")
            results.append({"device": d, "error": str(e)})

    print("\n" + "=" * 72)
    print(f"SUMMARY  model={args.model}  prompt_len={len(args.prompt)} chars  max_tokens={args.tokens}")
    print("=" * 72)
    print(f"{'device':6}  {'load(s)':>8}  {'TTFT(ms)':>9}  {'tok/s':>7}  preview")
    for r in results:
        if "error" in r:
            print(f"{r['device']:6}  {'ERROR':>8}  {'-':>9}  {'-':>7}  {r['error'][:60]}")
        else:
            print(f"{r['device']:6}  {r['load_s']:8.1f}  {r['ttft_ms']:9.1f}  "
                  f"{r['decode_tps']:7.2f}  {r['preview']}")


if __name__ == "__main__":
    main()
