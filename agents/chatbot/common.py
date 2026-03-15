"""Shared LLM utilities for chatbot agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env from cwd and project root so OPENAI_API_KEY is found regardless of startup location
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# Placeholder patterns that indicate the key was not replaced
_PLACEHOLDER_PATTERNS = ("your-actual", "your-api", "your_key", "placeholder", "-key-here")


def _is_valid_api_key(key: Optional[str]) -> bool:
    """Return True if the key looks like a real API key, not a placeholder."""
    if not key or not key.strip():
        return False
    key_lower = key.strip().lower()
    return not any(p in key_lower for p in _PLACEHOLDER_PATTERNS)


def get_llm() -> Optional[ChatOpenAI]:
    """Create a shared ChatOpenAI client for all chatbot agents.
    Uses OPENAI_API_KEY from environment only. Rejects placeholder keys.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not _is_valid_api_key(api_key):
        return None
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        streaming=True,
        api_key=api_key.strip(),
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

