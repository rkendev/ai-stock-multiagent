#!/usr/bin/env bash
set -euo pipefail
TICKER="${1:-ASML.AS}"
SINCE="${2:-2023-10-01}"

export PYTHONPATH=src

python -m cli --ticker "$TICKER" --since "$SINCE"
python -m agents.technical --ticker "$TICKER" --data-dir data --out-dir data
python -m agents.visualizer --ticker "$TICKER" --data-dir data --out-dir output
python -m reporter.report_generator --ticker "$TICKER" --data-dir data --out-dir output

echo
echo "Artifacts:"
echo "  data/$TICKER/prices.parquet"
echo "  data/$TICKER/technical.json"
echo "  output/$TICKER/charts/price_ma50.png"
echo "  output/$TICKER/report.md"
