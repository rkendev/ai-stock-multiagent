# orchestrator.py
import subprocess
import sys

# Define the sequence of scripts and their arguments
AGENTS = [
    ("src/ingest/prices.py",      ["--ticker", "{ticker}", "--since", "{since}"]),
    ("src/ingest/fundamentals.py",["--ticker", "{ticker}"]),
    ("src/ingest/news.py",        ["--ticker", "{ticker}", "--limit", "20"]),
    ("src/agents/researcher.py",  ["--ticker", "{ticker}", "--limit", "20"]),
    ("src/agents/sentiment.py",   ["--ticker", "{ticker}", "--limit", "20"]),
    ("src/agents/fundamental.py", ["--ticker", "{ticker}"]),
    ("src/agents/technical.py",   ["--ticker", "{ticker}"]),
    ("src/agents/reporter.py",    ["--ticker", "{ticker}"]),
]

def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <TICKER> [SINCE_DATE]")
        sys.exit(1)
    ticker = sys.argv[1]
    since  = sys.argv[2] if len(sys.argv) > 2 else "2024-06-01"

    for script, args in AGENTS:
        # Build the full command
        cmd = [sys.executable, script] + [a.format(ticker=ticker, since=since) for a in args]
        run(cmd)


if __name__ == "__main__":
    main()
