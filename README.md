# AI Stock Multi-Agent

Multi-Agent Stock Market Analyst (portfolio-ready capstone)

This project orchestrates a team of modular AI agents to analyze a stock end-to-end — from data ingestion to report generation. It demonstrates modern AI orchestration, data pipelines, and LLM integration using OpenAI and local LLaMA backends.

# Overview

The pipeline:

Ingests prices, fundamentals, and news.

Analyzes data using specialized agents (Researcher, Sentiment, Fundamental, Technical).

Synthesizes insights into a Markdown research report via the Reporter (LLM).

Optionally includes:

Prompt-v2 flow for modular LLM orchestration.

Local LLaMA backend for offline execution.

Parallel ingestion for faster data retrieval.

Visualization agent to embed charts in reports.

# Architecture
ingest/
 ├── prices.py        → data/<TICKER>/prices.parquet
 ├── fundamentals.py  → data/<TICKER>/fundamentals.csv
 └── news.py          → data/<TICKER>/news/rss_*.json

agents/
 ├── researcher.py    → output/researcher/<TICKER>/
 ├── sentiment.py     → output/sentiment/<TICKER>/
 ├── fundamental.py   → output/fundamental/<TICKER>/
 ├── technical.py     → output/technical/<TICKER>/
 ├── reporter.py      → output/<TICKER>/reporter/report_*.md
 ├── visualizer.py    → output/<TICKER>/charts/
 ├── prompt_manager.py
 ├── prompt_executor.py
 └── orchestrator.py  → Orchestrates agents and pipelines

orchestrator modes:
  - classic sequential
  - parallel ingest (`--parallel-ingest`)
  - prompt-v2 orchestrated LLM flow

# Requirements

Python 3.10+ or Docker

Optional GPU (NVIDIA + CUDA)

OpenAI API key (for cloud LLMs)

Local LLaMA optional backend (--use-local-llama)

# Quick Start (Local)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# Verify
PYTHONPATH=src pytest -q

# Run full pipeline
PYTHONPATH=src python -m cli run AAPL 2024-06-01

# New CLI Features
# Help
PYTHONPATH=src python -m cli --help

# Full pipeline with OpenAI
PYTHONPATH=src python -m cli run AAPL 2024-06-01

# Use Prompt-v2 orchestration
PYTHONPATH=src python -m cli run AAPL 2024-06-01 --use-prompt-v2

# Use Local LLaMA (offline)
PYTHONPATH=src python -m cli run AAPL 2024-06-01 --use-prompt-v2 --local-llama

# Enable parallel ingestion for speed
PYTHONPATH=src python -m cli run AAPL 2024-06-01 --parallel-ingest

# Prompt-v2 and Local LLaMA

PromptManager defines reusable templates for agents.

PromptExecutor handles prompt execution for OpenAI or local backends.

You can run the pipeline offline with a local LLaMA stub or integration (future full model).

Example:

from agents.prompt_executor import PromptExecutor
executor = PromptExecutor(use_local_llama=True)
print(executor.execute_prompt("Summarize TSLA outlook"))

# Visualization Agent

Adds plots (price trends, sentiment over time) to the final report.

PYTHONPATH=src python -m cli run AAPL 2024-06-01 --visualize


Outputs charts to:

output/<TICKER>/charts/

# Testing
# Unit tests
PYTHONPATH=src pytest -q -m "not integration"

# Full test suite
PYTHONPATH=src pytest -q

# Roadmap (Next Steps)
Plugin API for custom agents

Real local LLaMA integration (llama-cpp / HuggingFace)

Real-time / streaming ingestion

Dashboard (Streamlit / FastAPI)

Caching + retry logic for robustness

# License
MIT © 2025 rkendev