#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -B run.py "$@"
