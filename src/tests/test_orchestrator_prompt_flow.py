# src/tests/test_orchestrator_prompt_flow.py
from agents.orchestrator import orchestrate_with_prompt_flow

def test_orchestrator_prompt_flow_monkeypatch_run_cmd(monkeypatch):
    calls = []
    def fake_run_cmd(cmd_args):
        calls.append(cmd_args)
    monkeypatch.setattr("agents.orchestrator.run_cmd", fake_run_cmd)

    # Run with prompt mode
    orchestrate_with_prompt_flow("MSFT", "2025-01-01", use_prompt_v2=True, use_local_llama=True)

    out = "\n".join(str(c) for c in calls)
    # It should include calls to some agents, e.g. researcher
    assert any("src/agents/researcher.py" in str(cmd) for cmd in calls)
