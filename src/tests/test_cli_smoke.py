# src/tests/test_cli_smoke.py
def test_cli_imports():
    # Ensure CLI imports cleanly (no heavy side effects)
    import cli  # noqa: F401
