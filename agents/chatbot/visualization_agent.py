"""Visualization planning agent for chart recommendations."""

from __future__ import annotations

from agents.chatbot.common import get_llm, safe_llm_invoke


class VisualizationAgent:
    """Suggests charts and dashboard visuals for communicating results."""

    def __init__(self) -> None:
        self.llm = get_llm()

    def run(self, query: str, analysis: str) -> str:
        prompt = f"""
You are a data visualization expert.
Task: {query}
Analysis:
{analysis}

Suggest the best charts to explain model and data insights.
Include:
- chart type
- what each axis should represent
- why it helps decision making
"""
        return safe_llm_invoke(
            self.llm,
            prompt,
            fallback="- Use a feature importance bar chart and a prediction-vs-actual scatter plot.",
        )

