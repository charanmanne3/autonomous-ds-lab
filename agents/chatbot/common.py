"""Shared LLM utilities for chatbot agents."""

from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_llm() -> Optional[ChatOpenAI]:
    """Create a shared ChatOpenAI client for all chatbot agents."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True,
        api_key=api_key,
    )


def safe_llm_invoke(llm: Optional[ChatOpenAI], prompt: str, fallback: Optional[str] = None) -> str:
    """Invoke the LLM with graceful fallback if API is unavailable."""
    if llm is None:
        return fallback or "LLM unavailable. Set OPENAI_API_KEY to enable model responses."
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        if fallback:
            return f"{fallback}\n\n(LLM unavailable: {exc})"
        return f"LLM call failed: {exc}"

