"""Experiment memory system backed by local ChromaDB."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


class ExperimentMemory:
    """Stores and retrieves experiment knowledge for AI-assisted recommendations."""

    def __init__(self, persist_dir: str = "storage/memory/chroma") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True
        self._fallback_records: List[Dict[str, Any]] = []

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = self._client.get_or_create_collection(
                name="experiment_memory",
                embedding_function=sentence_transformer_ef,
            )
            self.logger.info("Experiment memory initialized at %s", self.persist_dir)
        except Exception as exc:  # noqa: BLE001
            self._enabled = False
            self.logger.warning("ChromaDB unavailable, using in-memory fallback: %s", exc)

    @staticmethod
    def _to_document(record: Dict[str, Any]) -> str:
        return (
            f"Task: {record.get('task', '')}. "
            f"Dataset: {record.get('dataset_name', '')}. "
            f"Best model: {record.get('best_model', '')}. "
            f"RMSE: {record.get('rmse', '')}, MAE: {record.get('mae', '')}, R2: {record.get('r2', '')}."
        )

    def save_experiment(self, record: Dict[str, Any]) -> None:
        """Persist one experiment summary into vector memory."""
        clean_record = {
            "task": str(record.get("task", "")),
            "dataset_name": str(record.get("dataset_name", "unknown")),
            "best_model": str(record.get("best_model", "unknown")),
            "rmse": float(record.get("rmse", 0.0)),
            "mae": float(record.get("mae", 0.0)),
            "r2": float(record.get("r2", 0.0)),
            "timestamp": str(record.get("timestamp", datetime.utcnow().isoformat())),
        }
        if self._enabled:
            self._collection.add(
                ids=[f"exp_{uuid4()}"],
                documents=[self._to_document(clean_record)],
                metadatas=[clean_record],
            )
        else:
            self._fallback_records.append(clean_record)

    def query_similar(self, task: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Return semantically similar experiments for a new user task."""
        if not task.strip():
            return []

        if self._enabled:
            result = self._collection.query(query_texts=[task], n_results=limit)
            metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
            distances = result.get("distances", [[]])[0] if result.get("distances") else []
            similar = []
            for idx, md in enumerate(metadatas):
                row = dict(md)
                row["similarity_score"] = float(1 / (1 + distances[idx])) if idx < len(distances) else None
                similar.append(row)
            return similar

        task_lower = task.lower()
        fallback = [r for r in self._fallback_records if task_lower in r.get("task", "").lower()]
        return fallback[:limit]

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return latest experiment history."""
        if self._enabled:
            result = self._collection.get(include=["metadatas"])
            rows = [dict(md) for md in result.get("metadatas", [])]
        else:
            rows = list(self._fallback_records)

        rows.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return rows[:limit]

    def recommend_models(self, task: str, limit: int = 3) -> List[str]:
        """Recommend models from similar successful experiments."""
        similar = self.query_similar(task, limit=limit)
        models = [row.get("best_model", "") for row in similar if row.get("best_model")]
        if not models:
            return ["XGBoost", "RandomForest", "LinearRegression"]
        ranked = Counter(models).most_common()
        return [name for name, _ in ranked]

    def generate_insight(self, task: str) -> str:
        """Produce a concise AI insight from memory."""
        similar = self.query_similar(task, limit=1)
        if not similar:
            return "No prior experiments found. Run the pipeline to build memory."
        top = similar[0]
        return (
            f"Based on previous experiments for similar tasks, {top.get('best_model', 'XGBoost')} "
            f"performed best with R2={float(top.get('r2', 0.0)):.3f}."
        )
