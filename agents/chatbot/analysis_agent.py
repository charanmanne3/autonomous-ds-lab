"""Analysis agent for deep reasoning over research output."""

from __future__ import annotations

from agents.chatbot.common import get_llm, safe_llm_invoke


class AnalysisAgent:
    """Analyzes collected research and produces actionable reasoning."""

    def __init__(self) -> None:
        self.llm = get_llm()

    def run(self, query: str, research_notes: str) -> str:
        prompt = f"""
You are an ML analysis specialist.
Query: {query}
Research notes:
{research_notes}

Produce:
- Problem framing
- Recommended modeling approach
- Validation strategy
- Production considerations
"""
        return safe_llm_invoke(
            self.llm,
            prompt,
            fallback="Analysis: frame the prediction goal, choose robust baseline + tree model, and validate with holdout metrics.",
        )

