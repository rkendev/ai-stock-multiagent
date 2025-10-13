#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501
