# src/agents/__init__.py
# Lazy submodule loader so `from agents import technical` still works
import importlib

__all__ = [
    "orchestrator",
    "technical",
    "fundamental",
    "sentiment",
    "researcher",
    "visualizer",
]

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
