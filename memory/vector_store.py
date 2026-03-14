"""Vector memory for chatbot question/response history."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


class ChatVectorStore:
    """Stores and retrieves prior chatbot interactions using ChromaDB."""

    def __init__(self, persist_dir: str = "storage/memory/chat_chroma") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True
        self._fallback: List[Dict[str, str]] = []

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = self._client.get_or_create_collection(
                name="chat_memory",
                embedding_function=ef,
            )
        except Exception as exc:  # noqa: BLE001
            self._enabled = False
            self.logger.warning("Chat vector store fallback enabled: %s", exc)

    def add_message(self, question: str, response: str) -> None:
        metadata = {"question": question, "response": response}
        if self._enabled:
            self._collection.add(
                ids=[f"chat_{uuid4()}"],
                documents=[f"Question: {question}\nResponse: {response}"],
                metadatas=[metadata],
            )
        else:
            self._fallback.append(metadata)

    def query_similar(self, question: str, limit: int = 3) -> List[Dict[str, Any]]:
        if self._enabled:
            result = self._collection.query(query_texts=[question], n_results=limit)
            metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
            return [dict(md) for md in metadatas]

        question_lower = question.lower()
        return [
            record
            for record in self._fallback
            if question_lower in record.get("question", "").lower()
        ][:limit]

