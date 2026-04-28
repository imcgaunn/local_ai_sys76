#!/usr/bin/env python3
"""Summarize a long PDF locally with Gemma-4-26B-A4B via llama-server.

Map-reduce pipeline:
  1. Extract text from the PDF (PyMuPDF)
  2. Split into ~chunk-tokens chunks on paragraph boundaries
  3. Map: summarize each chunk via /v1/chat/completions
  4. Reduce: combine chunk summaries into a final summary,
     recursively if combined summaries don't fit the context window

By default the script spawns llama-server itself, runs the pipeline, and
shuts the server down on exit. Pass --server-url to reuse a running instance.

Usage:
    ./summarize_pdf_setup.sh                                # one-time
    .venv/bin/python summarize_pdf.py path/to/doc.pdf
    .venv/bin/python summarize_pdf.py doc.pdf -o summary.md
    .venv/bin/python summarize_pdf.py doc.pdf --chunk-tokens 4000 --target-words 800
    .venv/bin/python summarize_pdf.py doc.pdf --server-url http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests

REPO = Path(__file__).parent
DEFAULT_LLAMA_SERVER = REPO / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_GGUF = REPO / "models" / "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_PORT = 8765

# Conservative chars-per-token for English; Gemma's tokenizer averages ~3.7.
# Used for chunking math; we never enforce a server-side token limit, so a
# slight over-estimate just means smaller chunks (safer).
CHARS_PER_TOKEN = 3.5

MAP_PROMPT = """The text below is one section of a longer document. Please write a concise summary of it that preserves the key facts, arguments, names, numbers, dates, and technical terms. Do not invent details. Do not say "this section" or "the text" — write as if your summary will be concatenated with summaries of other sections into a single document summary.

Section text:

{chunk}"""

REDUCE_PROMPT = """The texts below are summaries of consecutive sections of a single document, in order. Please combine them into one coherent summary of the whole document of roughly {target_words} words. Preserve overall structure (introduction, main points, conclusion) where visible. Keep specific details. Do not invent content.

Section summaries:

{summaries}"""


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF naming
        except ImportError:
            sys.exit("pymupdf not installed. Run ./summarize_pdf_setup.sh")
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages)


def split_into_chunks(text: str, target_tokens: int) -> list[str]:
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for para in paragraphs:
        if len(para) > target_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_chars = [], 0
            for i in range(0, len(para), target_chars):
                chunks.append(para[i:i + target_chars])
            continue
        if current_chars + len(para) > target_chars and current:
            chunks.append("\n\n".join(current))
            current, current_chars = [para], len(para)
        else:
            current.append(para)
            current_chars += len(para) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chat(url: str, prompt: str, max_tokens: int, temperature: float = 0.3) -> str:
    # Gemma-4-26B-A4B-it is a reasoning model: by default it emits a
    # chain-of-thought trace before the answer, which would burn token budget
    # and 10x the wall time per chunk. We don't need the reasoning artifact
    # for summarization, so disable it via the chat template kwarg.
    r = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


@contextmanager
def llama_server(server_bin: Path, gguf: Path, ctx: int, port: int,
                 threads: int, log_path: Path):
    cmd = [
        str(server_bin),
        "-m", str(gguf),
        "-c", str(ctx),
        "-t", str(threads),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-warmup",
    ]
    print(f"  starting: {' '.join(cmd)}")
    print(f"  log:      {log_path}")
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            if proc.poll() is not None:
                log_f.close()
                tail = "\n  ".join(log_path.read_text().splitlines()[-30:])
                sys.exit(f"llama-server died during startup. Tail:\n  {tail}")
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            sys.exit("llama-server did not become healthy within 5 minutes")
        print(f"  ready at {url}")
        yield url
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log_f.close()


def map_step(url: str, chunks: list[str], summary_tokens: int) -> list[str]:
    summaries: list[str] = []
    n = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        t0 = time.time()
        s = chat(url, MAP_PROMPT.format(chunk=chunk), summary_tokens)
        dt = time.time() - t0
        out_tokens = estimate_tokens(s) if s else 0
        print(f"  [{i:3d}/{n}] ~{estimate_tokens(chunk):5d} in -> "
              f"{len(s):5d} chars / ~{out_tokens:4d} tokens in {dt:5.1f}s")
        if not s.strip():
            print("\n  !! Model returned an empty summary. Stopping the map step")
            print("     so you don't burn time on the remaining chunks.")
            print("     Inspect the server log (summarize_pdf.llama-server.log)")
            print("     and try a curl against /v1/chat/completions to verify")
            print("     the chat template is producing non-empty output.")
            sys.exit(2)
        summaries.append(s)
    return summaries


def reduce_step(url: str, summaries: list[str], ctx: int,
                summary_tokens: int, target_words: int) -> str:
    safety = 1024
    while True:
        joined = "\n\n---\n\n".join(summaries)
        prompt = REDUCE_PROMPT.format(summaries=joined, target_words=target_words)
        if estimate_tokens(prompt) + summary_tokens + safety <= ctx:
            print(f"  reducing {len(summaries)} summaries -> final")
            return chat(url, prompt, max_tokens=summary_tokens, temperature=0.4)

        budget_chars = int((ctx - summary_tokens - safety) * CHARS_PER_TOKEN)
        scaffold = REDUCE_PROMPT.format(summaries="", target_words=target_words)
        budget_chars -= len(scaffold)
        groups: list[list[str]] = []
        cur: list[str] = []
        cur_chars = 0
        for s in summaries:
            tlen = len(s) + 7  # "---" separator + newlines
            if cur_chars + tlen > budget_chars and cur:
                groups.append(cur)
                cur, cur_chars = [s], tlen
            else:
                cur.append(s)
                cur_chars += tlen
        if cur:
            groups.append(cur)
        print(f"  {len(summaries)} summaries exceed context "
              f"-> {len(groups)} intermediate groups")
        next_round: list[str] = []
        for j, group in enumerate(groups, 1):
            p = REDUCE_PROMPT.format(
                summaries="\n\n---\n\n".join(group),
                target_words=max(target_words // len(groups), 200),
            )
            print(f"    intermediate {j}/{len(groups)}: {len(group)} summaries")
            next_round.append(chat(url, p, max_tokens=summary_tokens))
        summaries = next_round


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pdf", type=Path, help="PDF file to summarize")
    ap.add_argument("-o", "--out", type=Path,
                    help="write final summary to this file (also prints)")
    ap.add_argument("--chunk-tokens", type=int, default=5000,
                    help="approximate tokens per map-step chunk (default: 5000)")
    ap.add_argument("--summary-tokens", type=int, default=500,
                    help="max tokens per chunk summary (default: 500)")
    ap.add_argument("--target-words", type=int, default=600,
                    help="approximate length of the final summary in words")
    ap.add_argument("--ctx", type=int, default=32768,
                    help="server context size / KV cache budget (default: 32768)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 24)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--server-url",
                    help="reuse a llama-server already running at this URL "
                         "(skips spawning one)")
    ap.add_argument("--save-chunks", type=Path,
                    help="optional: write per-chunk summaries to this file")
    args = ap.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    print(f"== 1) Extracting text from {args.pdf.name}")
    text = extract_pdf_text(args.pdf)
    est = estimate_tokens(text)
    print(f"  {len(text):,} chars (~{est:,} tokens)")

    print(f"\n== 2) Chunking (~{args.chunk_tokens} tokens each)")
    chunks = split_into_chunks(text, args.chunk_tokens)
    print(f"  {len(chunks)} chunks")

    log_path = REPO / "summarize_pdf.llama-server.log"

    def _run(url: str) -> str:
        print(f"\n== 3) Map: summarizing {len(chunks)} chunks")
        summaries = map_step(url, chunks, args.summary_tokens)
        if args.save_chunks:
            args.save_chunks.write_text(
                "\n\n=== CHUNK BREAK ===\n\n".join(summaries))
            print(f"  wrote per-chunk summaries to {args.save_chunks}")
        print(f"\n== 4) Reduce")
        # Final summary may want a bit more headroom than chunk summaries.
        return reduce_step(url, summaries, args.ctx,
                           args.summary_tokens * 2, args.target_words)

    t0 = time.time()
    if args.server_url:
        final = _run(args.server_url.rstrip("/"))
    else:
        if not args.llama_server.exists():
            sys.exit(f"llama-server not found: {args.llama_server}\n"
                     f"Run ./gemma4_setup.sh first.")
        if not args.gguf.exists():
            sys.exit(f"GGUF not found: {args.gguf}\n"
                     f"Run ./gemma4_setup.sh first.")
        with llama_server(args.llama_server, args.gguf, args.ctx,
                          args.port, args.threads, log_path) as url:
            final = _run(url)
    dt = time.time() - t0

    print("\n" + "=" * 72)
    print(f"FINAL SUMMARY  (total wall time: {dt/60:.1f} min)")
    print("=" * 72)
    print(final)
    if args.out:
        args.out.write_text(final + "\n")
        print(f"\n  wrote to {args.out}")


if __name__ == "__main__":
    main()
