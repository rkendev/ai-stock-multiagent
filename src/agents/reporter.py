import argparse
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from typing import Optional
from pathlib import Path


try:
    # New-style OpenAI client (openai>=1.0)
    from openai import OpenAI
    _HAS_OPENAI_V1 = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    _HAS_OPENAI_V1 = False


# -----------------------------------------------------------------------------
# Abstract base class for reporters (LLM-agnostic)
# -----------------------------------------------------------------------------
class BaseReporter(ABC):
    """Interface for any reporter implementation (OpenAI, local LLM, etc.)."""

    @abstractmethod
    def generate_report(self, context: dict) -> str:
        """
        Given a context dict with keys:
          ['fundamental','technical','sentiment','researcher'],
        return a Markdown report string.
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# OpenAI-backed reporter (uses openai>=1.0 client)
# -----------------------------------------------------------------------------
class OpenAIReporter(BaseReporter):
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        if not _HAS_OPENAI_V1:
            raise RuntimeError(
                "openai>=1.0 is required for OpenAIReporter. "
                "Install with: pip install --upgrade openai"
            )
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided (set OPENAI_API_KEY or pass api_key=).")
        self.client = OpenAI(api_key=self.api_key)

    def generate_report(self, context: dict) -> str:
        prompt = self._build_prompt(context)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional financial analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

    def _build_prompt(self, c: dict) -> str:
        return (
            f"Analyze the following stock analysis for {c['fundamental']['ticker']}:\n\n"
            f"=== Fundamentals ===\n{json.dumps(c['fundamental'], indent=2)}\n\n"
            f"=== Technical ===\n{json.dumps(c['technical'], indent=2)}\n\n"
            f"=== Sentiment ===\n{json.dumps(c['sentiment'], indent=2)}\n\n"
            "=== Top News Titles ===\n"
            + "\n".join(f"- {item.get('title','(no title)')}" for item in c.get("researcher", []))
            + "\n\n"
            "Please produce a Markdown report with sections:\n"
            "1. Executive Summary\n"
            "2. Key Findings\n"
            "3. Investment Recommendation\n"
            "Use bullet points and a final one-sentence recommendation with a confidence score."
        )


# -----------------------------------------------------------------------------
# Reporter Agent (reads prior agent outputs, calls a reporter, writes .md)
# -----------------------------------------------------------------------------
class ReporterAgent:
    def __init__(self, reporter: BaseReporter, output_root: Optional[str] = None):
        """
        :param reporter: a BaseReporter implementation (OpenAI, local, etc.)
        :param output_root: override root folder for agent outputs.
                            Defaults to env OUTPUT_ROOT or 'output'.
        """
        self.reporter = reporter
        self.output_root = output_root or os.getenv("OUTPUT_ROOT") or "output"

    def _read_latest_json(self, *parts: str):
        """Read the lexicographically latest .json under output_root/<parts...>."""
        dirpath = os.path.join(self.output_root, *parts)
        if not os.path.isdir(dirpath):
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        files = sorted(
            f for f in os.listdir(dirpath)
            if f.lower().endswith(".json") and os.path.isfile(os.path.join(dirpath, f))
        )
        if not files:
            raise FileNotFoundError(f"No JSON files in: {dirpath}")
        latest = files[-1]
        with open(os.path.join(dirpath, latest), "r", encoding="utf8") as f:
            return json.load(f)

    def run(self, ticker: str) -> str:
        # Collect context from prior agents
        context = {
            "fundamental": self._read_latest_json("fundamental", ticker),
            "technical":   self._read_latest_json("technical", ticker),
            "sentiment":   self._read_latest_json("sentiment", ticker),
            "researcher":  self._read_latest_json("researcher", ticker),
        }

        # Generate report text
        report_md = self.reporter.generate_report(context)

        # ---- Write under OUTPUT_ROOT/<ticker>/reporter/ to satisfy tests ----
        out_dir = os.path.join(self.output_root, ticker, "reporter")
        os.makedirs(out_dir, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(out_dir, f"report_{ts}.md")

        with open(out_path, "w", encoding="utf8") as f:
            f.write(report_md + "\n")

        print(f"[Reporter] Wrote report to {out_path}")
        return out_path

def _maybe_append_charts_section(report_md: str, ticker: str, report_path: Path) -> str:
    """
    If output/visuals/<ticker> has known charts, append a '## Charts' section.
    Compute RELATIVE paths from the report file to the images so that they render
    when opening the markdown in-place.
    """
    visuals = Path("output") / "visuals" / ticker
    cand = [visuals / "price_ma50.png", visuals / "sentiment.png"]
    imgs = [p for p in cand if p.exists()]
    if not imgs:
        return report_md

    lines = ["", "## Charts", ""]
    for p in imgs:
        rel = os.path.relpath(p, start=report_path.parent)
        lines.append(f"![{p.name}]({rel})")
    return report_md + "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Reporter agent: synthesize all analyses.")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    p.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    p.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    p.add_argument(
        "--output-root",
        default=None,
        help="Override the output root (defaults to $OUTPUT_ROOT or 'output').",
    )
    args = p.parse_args()

    reporter = OpenAIReporter(model=args.model, temperature=args.temp)
    agent = ReporterAgent(reporter, output_root=args.output_root)
    agent.run(args.ticker)


if __name__ == "__main__":
    main()
