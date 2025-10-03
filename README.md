# ai-stock-multiagent

**Multi-Agent Stock Market Analyst (portfolio-ready capstone)**

This project orchestrates a small team of AI agents to analyze a stock end-to-end and generate a Markdown research report. It’s designed to showcase modern AI orchestration, data engineering, and practical ML/NLP:

- **Ingestion** – prices (Yahoo Finance), fundamentals (yfinance), and news (RSS).
- **Agents**
  - **Researcher**: collects recent headlines (RSS).
  - **Sentiment**: labels headlines with a finance-tuned sentiment pipeline (Transformers).
  - **Fundamental**: computes simple growth/ratio metrics from quarterly data.
  - **Technical**: computes MA50 and RSI(14), flags overbought.
  - **Reporter**: synthesizes everything (LLM). Default: OpenAI; pluggable interface for local LLMs.
- **Orchestrator** – runs the whole pipeline and writes a final report.

---

## Contents

- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start (Docker)](#quick-start-docker)
- [Quick Start (Local Ubuntu)](#quick-start-local-ubuntu)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap / Next Steps](#roadmap--next-steps)
- [License](#license)

---

## Architecture



ingest/
prices -> data/<TICKER>/prices.parquet
fundamentals -> data/<TICKER>/fundamentals.csv
news -> data/<TICKER>/news/rss_*.json

agents/
researcher -> output/researcher/<TICKER>/news_.json
sentiment -> output/sentiment/<TICKER>/sentiment_.json
fundamental-> output/fundamental/<TICKER>/fundamental_.json
technical -> output/technical/<TICKER>/technical_.json
reporter -> output/<TICKER>/reporter/report_*.md (LLM synthesis)

orchestrator (sequential): ingest -> researcher/sentiment -> fundamental/technical -> reporter


---

## Requirements

- **Docker** & **Docker Compose** (recommended)
- Optional GPU acceleration (NVIDIA + CUDA) for faster Transformers
- Or run **locally** with **Python 3.10+**
- **OpenAI API key** for the Reporter agent (if using OpenAI)

> Data and outputs are ignored by git via `.gitignore`. Do **not** commit secrets or `.env`.

---

## Quick Start (Docker)

1) Create a `.env` at repo root (or export variables in your shell):

```ini
# .env
OPENAI_API_KEY=sk-...           # Required for OpenAI Reporter
# Optional overrides:
# OUTPUT_ROOT=output
# TRANSFORMERS_CACHE=/root/.cache/huggingface


Build & start a dev shell:

docker compose build
docker compose run dev


Inside the container:

# Run end-to-end for a ticker with a start date
PYTHONPATH=src python src/agents/orchestrator.py AAPL 2024-06-01
# Final report: output/AAPL/reporter/report_<timestamp>.md

Quick Start (Local Ubuntu)

Tested on Ubuntu 24.04 (works in VMs like VirtualBox):

sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: speed up Hugging Face downloads for the Sentiment agent
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"

# OpenAI key if using the Reporter
export OPENAI_API_KEY=sk-...

# Verify
PYTHONPATH=src pytest -q

# Run end-to-end
PYTHONPATH=src python src/agents/orchestrator.py AAPL 2024-06-01

Environment Variables
Variable	Required	Default	Purpose
OPENAI_API_KEY	Yes*	—	Needed for OpenAIReporter (Reporter agent).
OUTPUT_ROOT	No	output	Root folder where agents write their outputs.
TRANSFORMERS_CACHE	No	HF default	Cache directory for Hugging Face models (speeds up Sentiment agent).
SENTIMENT_MODEL	No	internal	(Optional) Override the Transformers model for Sentiment experiments.

*Only required if you run the Reporter with OpenAI. The reporter is pluggable—swap in a local model via the same interface if preferred.

Usage
Run individual components
# Ingest
PYTHONPATH=src python src/ingest/prices.py --ticker AAPL --since 2024-06-01
PYTHONPATH=src python src/ingest/fundamentals.py --ticker AAPL
PYTHONPATH=src python src/ingest/news.py --ticker AAPL --limit 20

# Agents
PYTHONPATH=src python src/agents/researcher.py --ticker AAPL --limit 20
PYTHONPATH=src python src/agents/sentiment.py  --ticker AAPL --limit 20
PYTHONPATH=src python src/agents/fundamental.py --ticker AAPL
PYTHONPATH=src python src/agents/technical.py   --ticker AAPL

# Reporter (OpenAI by default; requires OPENAI_API_KEY)
PYTHONPATH=src python src/agents/reporter.py --ticker AAPL
# -> output/<TICKER>/reporter/report_<timestamp>.md

### Unified CLI (optional)

You can drive everything via a single CLI:

```bash
# Help
PYTHONPATH=src python -m cli --help

# Ingest
PYTHONPATH=src python -m cli ingest prices --ticker AAPL --since 2024-06-01
PYTHONPATH=src python -m cli ingest fundamentals --ticker AAPL
PYTHONPATH=src python -m cli ingest news --ticker AAPL --limit 20

# Agents
PYTHONPATH=src python -m cli agent researcher --ticker AAPL --limit 20
PYTHONPATH=src python -m cli agent sentiment  --ticker AAPL --limit 20
PYTHONPATH=src python -m cli agent fundamental --ticker AAPL
PYTHONPATH=src python -m cli agent technical   --ticker AAPL
PYTHONPATH=src python -m cli agent reporter    --ticker AAPL --model gpt-4o-mini --temp 0.3

# Full pipeline
PYTHONPATH=src python -m cli run AAPL 2024-06-01

Full pipeline
PYTHONPATH=src python src/agents/orchestrator.py AAPL 2024-06-01


Notes:

The orchestrator accepts a ticker and optional since date.

It runs ingestion, then analysis agents, then the reporter.

All components respect OUTPUT_ROOT where applicable.

Tests
# Docker container shell or local venv
PYTHONPATH=src pytest -q


First run may download a large Transformers model for the Sentiment agent. Use TRANSFORMERS_CACHE to avoid repeated downloads.

Project Structure
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── README.md
├── src
│   ├── ingest
│   │   ├── prices.py
│   │   ├── fundamentals.py
│   │   └── news.py
│   ├── agents
│   │   ├── researcher.py
│   │   ├── sentiment.py
│   │   ├── fundamental.py
│   │   ├── technical.py
│   │   ├── reporter.py
│   │   └── orchestrator.py
│   └── tests
│       ├── test_prices.py
│       ├── test_fundamentals.py
│       ├── test_news.py
│       ├── test_researcher.py
│       ├── test_sentiment.py
│       ├── test_fundamental_agent.py
│       ├── test_technical_agent.py
│       └── test_orchestrator.py
└── output/         # generated artifacts (ignored by git)

Troubleshooting

OpenAI: APIRemovedInV1 / openai.ChatCompletion errors
We use the new openai>=1.0 client (from openai import OpenAI). If you see legacy API errors, upgrade:

pip install --upgrade openai


Sentiment model download is slow
Set a cache dir and keep it between runs:

export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"


GPU not used

In Docker, ensure --gpus all is enabled by compose (NVIDIA Container Toolkit installed).

In local, verify PyTorch sees the GPU:

import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))


ModuleNotFoundError: ingest when running tests
Always run with PYTHONPATH=src:

PYTHONPATH=src pytest -q


FileNotFoundError: data/<TICKER>/prices.parquet
Run price ingestion first (or the orchestrator), e.g.:

PYTHONPATH=src python src/ingest/prices.py --ticker AAPL --since 2024-06-01

Roadmap / Next Steps

Prompt engineering for the Reporter (few-shot, structured outputs).

Visualization agent: embed charts (price vs. MA50, sentiment over time) into the Markdown.

Parallelization: run ingest steps concurrently to speed up end-to-end time.

Local LLMs: add a LocalLlamaReporter implementing the same BaseReporter interface.

Streamlit UI: simple viewer for latest report + charts.

CI: GitHub Actions to run PYTHONPATH=src pytest -q on PRs.

License

MIT © 2025 Your Name


---

Want me to also add a minimal **GitHub Actions** CI workflow or a tiny **Streamlit** viewer stub in another branch?
::contentReference[oaicite:0]{index=0}