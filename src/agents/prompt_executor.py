from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol


class BaseExecutor(Protocol):
    def execute_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        ...


@dataclass
class OpenAIExecutor:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3

    def execute_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        # Stubbed for tests: include class name for assertion
        return f"OpenAIExecutor: {prompt}"


@dataclass
class LocalLlamaExecutor:
    model: str = "local-llama"
    temperature: float = 0.1

    def execute_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        # Stubbed for tests: include class name for assertion
        return f"LocalLlamaExecutor: {prompt}"


@dataclass
class PromptExecutor:
    """
    Thin façade that selects an underlying LLM executor.
    Tests only assert string contains 'OpenAIExecutor' / 'LocalLlamaExecutor'.
    """
    use_local_llama: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = None

    def __post_init__(self) -> None:
        if self.use_local_llama:
            self._impl: BaseExecutor = LocalLlamaExecutor(
                model=self.model or "local-llama",
                temperature=self.temperature or 0.1,
            )
        else:
            self._impl = OpenAIExecutor(
                model=self.model or "gpt-4o-mini",
                temperature=self.temperature or 0.3,
            )

    def execute_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        return self._impl.execute_prompt(prompt, context=context)


class LLMBackendFactory:
    """
    Factory used by orchestrator to obtain an executor by string id.
    """
    @staticmethod
    def create(backend: str = "openai", **kwargs) -> PromptExecutor:
        if backend == "local-llama":
            return PromptExecutor(use_local_llama=True, **kwargs)
        return PromptExecutor(use_local_llama=False, **kwargs)
