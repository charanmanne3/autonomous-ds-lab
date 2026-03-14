"""Code generation agent for executable ML snippets."""

from __future__ import annotations

from agents.chatbot.common import get_llm, safe_llm_invoke


class CodeAgent:
    """Generates Python code suggestions based on analysis decisions."""

    def __init__(self) -> None:
        self.llm = get_llm()

    def run(self, query: str, analysis: str) -> str:
        prompt = f"""
You are a Python ML engineer.
User request: {query}
Analysis summary:
{analysis}

Write a concise Python code snippet that demonstrates the recommended approach.
Use sklearn/pandas style code and include model training + evaluation.
"""
        return safe_llm_invoke(
            self.llm,
            prompt,
            fallback=(
                "```python\n"
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.ensemble import RandomForestRegressor\n"
                "# ... load dataframe, split, fit, evaluate\n"
                "```\n"
            ),
        )

