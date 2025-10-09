# src/agents/prompt_manager.py
from __future__ import annotations
from typing import Any, Dict, List


class PromptManager:
    """
    Manage templates for various agents and render prompt text
    given a context.
    """
    def __init__(self):
        # You can expand or externalize this later
        self.templates = {
            "researcher": "Research {ticker} since {since}",
            "reporter": (
                "Summarize findings for {ticker}.\n"
                "Technical: {technical}\nFundamental: {fundamental}\n"
                "Sentiment: {sentiment}\nResearch: {researcher}"
            ),
        }
       
    def render(self, agent: str, **kwargs: Any) -> str:
        template = self.templates.get(agent)
        if template is None:
            raise ValueError(f"No prompt template for agent '{agent}'")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable '{e.args[0]}' for agent '{agent}'")

    def list_agents(self) -> List[str]:
        return list(self.templates.keys())
