def test_streamlit_app_imports():
    """
    Health check: importing the Streamlit app should not raise.
    This catches missing deps / bad top-level code paths in CI.
    """
    import importlib
    m = importlib.import_module("ui.app")
    assert hasattr(m, "main")
