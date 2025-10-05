from __future__ import annotations
from typing import Protocol, Any, Dict, runtime_checkable


@runtime_checkable
class PromptExecutor(Protocol):
    """
    Interface to “execute” a prompt and return a response (string).
    In production, this might call OpenAI, LLaMA, etc. Here we stub.
    """

    def execute(self, agent: str, prompt: str, context: Dict[str, Any]) -> str:
        ...


class StubExecutor:
    """
    A prompt executor stub for testing / development.
    Simply returns an echo or fixed response per agent.
    """

    def execute(self, agent: str, prompt: str, context: Dict[str, Any]) -> str:
        # simple stub behavior
        return f"[{agent} response] based on prompt: {prompt}"
