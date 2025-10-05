import pytest
from agents.orchestrator import orchestrate_with_prompt_flow

def test_orchestrator_prompt_flow_monkeypatch_run_cmd(monkeypatch, capsys):
    calls = []
    def fake_run_cmd(cmd):
        calls.append(cmd)
    monkeypatch.setattr("agents.orchestrator.run_cmd", fake_run_cmd)

    orchestrate_with_prompt_flow("MSFT", "2025-01-01", use_prompt_v2=True)
    out = capsys.readouterr().out
    assert "prompts:" in out
    assert any("src/agents/researcher.py" in cmd for cmd in calls)
    # ensure some agent runs were invoked
