"""Final report agent for composing user-facing chatbot responses."""

from __future__ import annotations

from agents.chatbot.common import get_llm, safe_llm_invoke


def _synthesize_from_context(
    query: str,
    research: str,
    analysis: str,
    code: str,
    visualization: str,
    memory_hint: str,
) -> str:
    """When LLM is unavailable, compile agent context into a readable response.
    Never include memory_hint (prior Q&A) in output—that would repeat stored responses.
    """
    sections = []
    if research:
        sections.append(f"## Research\n{research}")
    if analysis:
        sections.append(f"## Analysis\n{analysis}")
    if code:
        sections.append(f"## Suggested Implementation\n```\n{code}\n```")
    if visualization:
        sections.append(f"## Visualization Ideas\n{visualization}")

    if not sections:
        return "LLM unavailable. Set OPENAI_API_KEY to enable tailored recommendations."

    return "\n\n".join(sections) + "\n\n---\n*Set OPENAI_API_KEY for tailored recommendations.*"


class ChatReportAgent:
    """Produces final explanation from upstream agent outputs using an LLM.
    Always generates a fresh LLM response; retrieved memory is never returned directly.
    """

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
You are an AI data science assistant. Generate a helpful, tailored response to the user's question.

User question:
{query}

Context from previous agents:

Research:
{research}

Analysis:
{analysis}

Code:
{code}

Visualization ideas:
{visualization}

Relevant previous conversation (for reference only—do NOT copy or repeat):
{memory_hint}

CRITICAL: Use the previous conversation only as background context. You MUST generate a completely NEW, tailored answer. Do NOT copy, repeat, or paraphrase the previous responses. The user expects a fresh response every time.

Provide a polished final response with sections:
1) Recommendation
2) Why this approach
3) Suggested implementation snippet
4) Visualization plan
5) Next steps

Base your answer on the user's question and the context above. Do not use generic or hard-coded recommendations.
"""
        return safe_llm_invoke(
            self.llm,
            prompt,
            fallback=_synthesize_from_context(
                query, research, analysis, code, visualization, memory_hint
            ),
        )

