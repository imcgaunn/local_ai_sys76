#!/usr/bin/env python3
"""Benchmark Gemma-4-26B-A4B (MoE) via llama.cpp on this machine.

Companion to npu_run.py — that one tests NPU/iGPU/CPU on a small dense model;
this one stress-tests the "big MoE on big RAM" hypothesis: 25.2B total params
sit in ~14 GB of Q4 weights, but only 3.8B activate per token. Decode tokens/sec
should feel closer to a 4B model than a 25B one. That's the experiment.

Run after `./gemma4_setup.sh`:
    python gemma4_run.py
    python gemma4_run.py --threads 16          # try different thread counts
    python gemma4_run.py --tokens 256 --prompt "Why is the sky blue?"
    python gemma4_run.py --gguf models/<other-quant>.gguf
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent
DEFAULT_LLAMA_CLI = REPO / "llama.cpp" / "build" / "bin" / "llama-cli"
DEFAULT_GGUF = REPO / "models" / "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_PROMPT = "In one paragraph, explain how a transformer language model generates text."

PERF_RE = re.compile(
    r"^llama_perf_context_print:\s+(?P<key>[a-z ]+?)\s+=\s+"
    r"(?P<ms>[\d.]+)\s+ms"
    r"(?:\s+/\s+(?P<count>\d+)\s+(?:tokens|runs))?"
    r"(?:\s+\(.*?(?P<tps>[\d.]+)\s+tokens per second.*\))?",
    re.M,
)


def parse_perf(stderr: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in PERF_RE.finditer(stderr):
        key = m.group("key").strip().replace(" ", "_")
        out[f"{key}_ms"] = float(m.group("ms"))
        if m.group("count"):
            out[f"{key}_count"] = float(m.group("count"))
        if m.group("tps"):
            out[f"{key}_tps"] = float(m.group("tps"))
    return out


def run_llama_cli(cli: Path, gguf: Path, prompt: str, n_tokens: int,
                  threads: int, ctx: int) -> dict:
    cmd = [
        str(cli),
        "-m", str(gguf),
        "-p", prompt,
        "-n", str(n_tokens),
        "-t", str(threads),
        "-c", str(ctx),
        "--no-warmup",
        "--no-display-prompt",
        "-no-cnv",   # one-shot, no chat loop
    ]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print("  STDERR (tail):")
        for line in proc.stderr.splitlines()[-15:]:
            print(f"    {line}")
        raise SystemExit(f"llama-cli exited {proc.returncode}")
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "perf": parse_perf(proc.stderr),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cli", default=str(DEFAULT_LLAMA_CLI))
    ap.add_argument("--gguf", default=str(DEFAULT_GGUF))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokens", type=int, default=128, help="max tokens to generate")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 24,
                    help="llama.cpp -t value (default: all logical CPUs)")
    ap.add_argument("--ctx", type=int, default=4096, help="-c context size")
    args = ap.parse_args()

    cli = Path(args.llama_cli)
    if not cli.exists():
        sys.exit(f"llama-cli not found at {cli}. Run ./gemma4_setup.sh first.")
    gguf = Path(args.gguf)
    if not gguf.exists():
        sys.exit(f"GGUF not found at {gguf}. Run ./gemma4_setup.sh first.")

    size_gb = gguf.stat().st_size / (1024 ** 3)
    print(f"model : {gguf.name}  ({size_gb:.1f} GB on disk)")
    print(f"cli   : {cli}")
    print(f"prompt: {args.prompt!r}  | max_new={args.tokens}  | threads={args.threads}  | ctx={args.ctx}")

    result = run_llama_cli(cli, gguf, args.prompt, args.tokens, args.threads, args.ctx)
    perf = result["perf"]

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    if "load_time_ms" in perf:
        print(f"  load (mmap+warm) : {perf['load_time_ms']/1000:7.2f} s")
    if "prompt_eval_time_tps" in perf:
        print(f"  prefill          : {perf['prompt_eval_time_tps']:7.1f} tok/s "
              f"({int(perf.get('prompt_eval_time_count',0))} prompt tokens)")
    if "eval_time_tps" in perf:
        print(f"  decode           : {perf['eval_time_tps']:7.1f} tok/s "
              f"({int(perf.get('eval_time_count',0))} generated tokens)")
        print()
        print("  ^ Decode rate is the headline number. For a 25.2B/3.8B-active MoE")
        print("    on a 24-core Arrow Lake at Q4, 'good' is roughly 8-15 tok/s.")
        print("    If you're below that, the model isn't really MoE-resident in cache —")
        print("    consider lowering --threads (try 12) or pinning to P-cores.")
    if "total_time_ms" in perf:
        print(f"  total wall time  : {perf['total_time_ms']/1000:7.2f} s")

    # Show the actual generated text so you can sanity-check coherence.
    print("\n--- generated ---")
    print(result["stdout"].strip()[:600])


if __name__ == "__main__":
    main()
