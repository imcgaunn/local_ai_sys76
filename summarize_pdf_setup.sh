#!/usr/bin/env bash
# Add PDF extraction support to the repo's .venv. Reuses the venv created by
# npu_setup.sh; creates it if absent. The summarizer itself talks to
# llama-server (already built by gemma4_setup.sh) over HTTP, so the only new
# Python dependency is PyMuPDF.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "== Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null
pip install pymupdf requests

echo
echo "Done. Make sure llama-server is built (./gemma4_setup.sh), then run:"
echo "    $VENV_DIR/bin/python summarize_pdf.py path/to/doc.pdf"
