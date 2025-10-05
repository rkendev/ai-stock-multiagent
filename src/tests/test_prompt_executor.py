from agents.prompt_executor import StubExecutor

def test_stub_executor_returns_expected():
    executor = StubExecutor()
    resp = executor.execute("researcher", "Do something about AAPL", {"ticker": "AAPL"})
    assert "[researcher response]" in resp
    assert "AAPL" in resp
