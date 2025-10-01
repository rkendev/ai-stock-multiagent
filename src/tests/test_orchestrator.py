from pathlib import Path
import os

import pytest

# Import the orchestrator module from the package path
from agents import orchestrator as orch


def test_build_cmd_expands_placeholders(tmp_path, monkeypatch):
    """
    Ensure {py}, {ticker}, {since} get expanded and command is a list[str].
    Accept both 'python' and 'python3' executables for portability.
    """
    cmd = orch.build_cmd(
        "{py} src/ingest/prices.py --ticker {ticker} --since {since}",
        ticker="AAPL",
        since="2024-06-01",
    )

    assert isinstance(cmd, list)
    assert cmd[0].endswith(("python", "python3"))  # env-agnostic
    assert cmd[1] == "src/ingest/prices.py"
    assert cmd[2] == "--ticker"
    assert cmd[3] == "AAPL"
    assert cmd[4] == "--since"
    assert cmd[5] == "2024-06-01"


def test_mkdirp(tmp_path):
    """mkdirp should create nested directories idempotently."""
    p = tmp_path / "a" / "b" / "c"
    orch.mkdirp(str(p))
    assert p.is_dir()
    # call again to ensure idempotence
    orch.mkdirp(str(p))
    assert p.is_dir()


def test_env_passthrough(monkeypatch, tmp_path):
    """
    If orchestrator relies on environment variables, ensure the helper
    can see them (e.g., for OUTPUT_ROOT propagation).
    """
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    assert os.environ.get("OUTPUT_ROOT") == str(tmp_path / "output")


@pytest.mark.parametrize(
    "template,kwargs,expected_tail",
    [
        (
            "{py} -m agents.researcher --ticker {ticker} --limit {limit}",
            {"ticker": "MSFT", "limit": 5},
            ["-m", "agents.researcher", "--ticker", "MSFT", "--limit", "5"],
        ),
        (
            "{py} src/agents/technical.py --ticker {ticker}",
            {"ticker": "NVDA"},
            ["src/agents/technical.py", "--ticker", "NVDA"],
        ),
    ],
)
def test_build_cmd_various_cases(template, kwargs, expected_tail):
    cmd = orch.build_cmd(template, **kwargs)
    # First arg is interpreter (python/python3), rest must match expected tail
    assert cmd[0].endswith(("python", "python3"))
    assert cmd[1:] == expected_tail
