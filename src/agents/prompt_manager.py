from __future__ import annotations
from typing import Dict, Any, List


class PromptManager:
    """
    Manages prompt templates for different agents, and produces
    completed prompt strings by substituting variables.
    """

    def __init__(self):
        self.templates: Dict[str, str] = {
            "researcher": "Research {ticker} since {since}. Focus on {topic}.",
            "sentiment": "Given recent news, rate sentiment for {ticker}.",
            "fundamental": "Summarize fundamentals for {ticker}.",
            "reporter": "Write a report for {ticker} based on inputs."
        }

    def render(self, agent: str, **kwargs: Any) -> str:
        template = self.templates.get(agent)
        if template is None:
            raise ValueError(f"No template for agent {agent}")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e.args[0]} in prompt render")

    def list_agents(self) -> List[str]:
        return list(self.templates.keys())
