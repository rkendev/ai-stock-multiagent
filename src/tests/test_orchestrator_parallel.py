# src/tests/test_orchestrator_parallel.py

import pytest
from unittest.mock import patch, call

from agents.orchestrator import orchestrate

@pytest.mark.parametrize("parallel", [False, True])
def test_orchestrate_parallel_runs_ingest_tasks(parallel, tmp_path):
    # Prepare dummy ticker and since
    ticker = "TEST"
    since = "2025-01-01"
    news_limit = 5

    # Patch subprocess.run to avoid real execution
    with patch("agents.orchestrator.run_cmd") as mocked_run_cmd:
        # Run orchestration with parallel_ingest flag
        orchestrate(
            ticker=ticker,
            since=since,
            news_limit=news_limit,
            skip_news=False,
            skip_prices=False,
            skip_fundamentals=False,
            parallel_ingest=parallel,
        )

        # Check that at least ingest commands were called
        # There should be calls for prices, fundamentals, news, reporter, etc.
        calls = mocked_run_cmd.call_args_list
        called_cmds = [c[0][0] for c in calls]  # first arg of each call

        assert any("prices" in cmd for cmd in called_cmds)
        assert any("fundamentals" in cmd for cmd in called_cmds)
        assert any("news" in cmd for cmd in called_cmds)
        assert any("reporter" in cmd for cmd in called_cmds)

        # Reporter should be last call (or at least exist)
        assert any("reporter" in cmd for cmd in called_cmds[-1:])


        # If parallel, ingest commands may appear in any order
        if parallel:
            assert len(called_cmds) >= 4
        else:
            # In sequential mode, order is deterministic:
            # first price, fundamentals, news
            assert "prices.py" in called_cmds[0]

