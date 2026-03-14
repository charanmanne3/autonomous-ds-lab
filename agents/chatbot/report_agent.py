"""Final report agent for composing user-facing chatbot responses."""

from __future__ import annotations

from agents.chatbot.common import get_llm, safe_llm_invoke


class ChatReportAgent:
    """Produces final explanation from upstream agent outputs."""

    def __init__(self) -> None:
        self.llm = get_llm()

    def run(
        self,
        query: str,
        research: str,
        analysis: str,
        code: str,
        visualization: str,
        memory_hint: str,
    ) -> str:
        prompt = f"""
You are the final response agent in a multi-agent ML assistant.
User query: {query}
Memory hint: {memory_hint}

Research:
{research}

Analysis:
{analysis}

Code:
{code}

Visualization ideas:
{visualization}

Generate a polished final response with sections:
1) Recommendation
2) Why this approach
3) Suggested implementation snippet
4) Visualization plan
5) Next steps
"""
        return safe_llm_invoke(
            self.llm,
            prompt,
            fallback=(
                "Recommendation: start with RandomForest and compare against LinearRegression baseline.\n\n"
                "Why: tree ensembles capture nonlinear patterns in tabular housing-style data.\n\n"
                "Next steps: run holdout evaluation and inspect feature importance."
            ),
        )

