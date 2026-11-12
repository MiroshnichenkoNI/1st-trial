#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt --index-url https://pypi.org/simple
.venv/bin/python -m pip check
mkdir -p out
.venv/bin/python prepare.py
.venv/bin/python -m pip freeze > out/installed.txt
printf '%s\n' 'Setup complete. Run: ./run.sh --input /path/to/videos --output out/run1'
