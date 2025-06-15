# ai-stock-multiagent

**Multi-Agent Stock Market Analyst**

A proof-of-concept AI pipeline that orchestrates multiple agents to perform end-to-end stock analysis for any ticker:

- **Data Ingestion**: Fetches historical price data, fundamentals, and news.
- **Agent Analysis**:
  - Researcher: Aggregates and stores RSS news items.
  - Sentiment Analyst: Labels headlines using a finance-specific LLM pipeline.
  - Fundamental Analyst: Computes key financial metrics and growth rates.
  - Technical Analyst: Derives moving averages and RSI, flags overbought/oversold.
- **Reporter**: Synthesizes all agent outputs into a Markdown research report via the OpenAI API.
- **Orchestrator**: Single-script workflow to run all agents sequentially and generate the final report.

## Features

- **Modular** architecture with clear separation of concerns.
- **OO Design** for the Reporter to support multiple LLM backends (OpenAI, local models, etc.).
- **Fully tested** codebase with pytest for each component.
- **Dockerized** dev environment with GPU support through NVIDIA CUDA on WSL2.

## Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + CUDA drivers (optional, for accelerated inference)
- (Optional) Python 3.10+ and virtualenv for local development
- OpenAI API key (for the Reporter agent)

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/ai-stock-multiagent.git
   cd ai-stock-multiagent
   ```
2. Create a `.env` file at the project root:
   ```ini
   OPENAI_API_KEY=sk-...
   ```
3. Build and spin up the dev container:
   ```bash
   docker compose build
   docker compose run dev
   ```

## Usage

### Ingest & Agents

Inside the container shell:

```bash
# Fetch prices, fundamentals, news
PYTHONPATH=src python src/ingest/prices.py --ticker AAPL --since 2024-01-01
PYTHONPATH=src python src/ingest/fundamentals.py --ticker AAPL
PYTHONPATH=src python src/ingest/news.py --ticker AAPL --limit 20

# Run specialized agents
PYTHONPATH=src python src/agents/researcher.py --ticker AAPL --limit 20
PYTHONPATH=src python src/agents/sentiment.py  --ticker AAPL --limit 20
PYTHONPATH=src python src/agents/fundamental.py --ticker AAPL
PYTHONPATH=src python src/agents/technical.py   --ticker AAPL
``` 

### Generating the Report

```bash
PYTHONPATH=src python src/agents/reporter.py --ticker AAPL
# Output saved at output/reporter/AAPL/report_<timestamp>.md
```

### Full Pipeline with Orchestrator

```bash
PYTHONPATH=src python orchestrator.py AAPL 2024-01-01
```

## Running Tests

```bash
# From inside the container (or with PYTHONPATH=src locally)
pytest -q
```

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── orchestrator.py
├── .env.example    # template for your .env
├── src
│   ├── ingest      # data ingestion scripts
│   ├── agents      # multi-agent analysis modules
│   └── tests       # pytest unit tests
└── output          # generated data & reports (ignored)
```

## Next Steps

- Prompt engineering & few-shot examples for richer reports
- Embed charts & visualizations in the Markdown
- Experiment with local LLM backends
- Parallelize ingestion and analysis for speed

## License

MIT © Your Name
