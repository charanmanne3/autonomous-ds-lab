"""Research agent for collecting context about user queries."""

from __future__ import annotations

from agents.chatbot.common import get_llm, safe_llm_invoke


class ResearchAgent:
    """Collects high-level facts and relevant context for a user query."""

    def __init__(self) -> None:
        self.llm = get_llm()

    def run(self, query: str, similar_context: str = "") -> str:
        prompt = f"""
You are a research assistant for data/ML questions.
User query: {query}
Similar historical context: {similar_context}

Provide concise research notes:
1) Core objective
2) Relevant domain assumptions
3) Key data points to investigate
4) Risks or caveats
"""
        return safe_llm_invoke(
            self.llm,
            prompt,
            fallback="Research notes: define objective, gather key data signals, and document assumptions.",
        )

