import argparse
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime

import openai


# -----------------------------------------------------------------------------
# Abstract base class for reporters
# -----------------------------------------------------------------------------
class BaseReporter(ABC):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        openai.api_key = self.api_key

    @abstractmethod
    def generate_report(self, context: dict) -> str:
        """
        Given a context dict with keys
        ['fundamental','technical','sentiment','researcher'],
        returns the generated report as text.
        """
        pass


# -----------------------------------------------------------------------------
# OpenAI-backed reporter
# -----------------------------------------------------------------------------
class OpenAIReporter(BaseReporter):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def generate_report(self, context: dict) -> str:
        prompt = self._build_prompt(context)
        # Use the v1 openai-python interface
        resp = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst."},
                {"role": "user",   "content": prompt}
            ],
            temperature=self.temperature,
        )
        # Response structure remains similar
        return resp.choices[0].message.content.strip()

    def _build_prompt(self, c: dict) -> str:
        return (
            f"Analyze the following stock analysis for {c['fundamental']['ticker']}:\n\n"
            f"=== Fundamentals ===\n{json.dumps(c['fundamental'], indent=2)}\n\n"
            f"=== Technical ===\n{json.dumps(c['technical'], indent=2)}\n\n"
            f"=== Sentiment ===\n{json.dumps(c['sentiment'], indent=2)}\n\n"
            f"=== Top News Titles ===\n"
            + "\n".join(f"- {item['title']}" for item in c['researcher']) +
            "\n\n"
            "Please produce a Markdown report with sections:\n"
            "1. Executive Summary\n"
            "2. Key Findings\n"
            "3. Investment Recommendation\n"
            "Use bullet points and a final one-sentence recommendation with a confidence score."
        )


# -----------------------------------------------------------------------------
# Orchestrator-style wrapper for the reporter
# -----------------------------------------------------------------------------
class ReporterAgent:
    # Allow tests or env to override where agent outputs live
    OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output")

    def __init__(self, reporter: BaseReporter):
        self.reporter = reporter

    def run(self, ticker: str) -> str:
        context = {}
        for agent in ("fundamental", "technical", "sentiment", "researcher"):
            dirpath = os.path.join(self.OUTPUT_ROOT, agent, ticker)
            files = sorted(f for f in os.listdir(dirpath) if f.lower().endswith(".json"))
            if not files:
                raise FileNotFoundError(f"No output for {agent} at {dirpath}")
            latest = files[-1]
            with open(os.path.join(dirpath, latest), encoding="utf8") as f:
                context[agent] = json.load(f)

        report_md = self.reporter.generate_report(context)

        out_dir = os.path.join(self.OUTPUT_ROOT, "reporter", ticker)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(out_dir, f"report_{ts}.md")
        with open(out_path, "w", encoding="utf8") as f:
            f.write(report_md + "\n")
        print(f"[Reporter] Wrote report to {out_path}")
        return out_path


def main():
    p = argparse.ArgumentParser(description="Reporter agent: synthesize all analyses")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    p.add_argument("--model",   default="gpt-4o", help="OpenAI model name")
    p.add_argument("--temp",    type=float, default=0.7, help="Sampling temperature")
    args = p.parse_args()

    reporter = OpenAIReporter(model=args.model, temperature=args.temp)
    agent = ReporterAgent(reporter)
    agent.run(args.ticker)

if __name__ == "__main__":
    main()
