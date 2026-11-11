"""Unit tests for the pluggable TransformerIntent adapter."""

from __future__ import annotations

import numpy as np
import pytest

from eq_layer.transformer_intent import (
    DEFAULT_TRANSFORMER_MODEL,
    TransformerIntent,
    _missing_transformer,
)
from eq_layer._deps import OptionalDependencyError


class MockTransformerEmbedder:
    """Deterministic mock embedder for offline testing."""

    def __init__(self, dim: int = 312) -> None:
        self.dim = dim

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for text in texts:
            # Deterministic pseudo-embedding from text hash
            val = sum(ord(c) for c in text)
            np.random.seed(val % 10000)
            vectors.append(np.random.randn(self.dim))
        return np.vstack(vectors)


def test_missing_transformer_dependency_error() -> None:
    exc = _missing_transformer("TestTransformer", ImportError("No module named 'torch'"))
    assert isinstance(exc, OptionalDependencyError)
    assert "transformers and torch" in str(exc)
    assert 'pip install "eq-layer[transformers]"' in str(exc)


def test_transformer_intent_fit_and_infer_end_to_end() -> None:
    records = [
        {"text": "Исправь эту ошибку", "label": "action_request"},
        {"text": "Сделай коммит", "label": "action_request"},
        {"text": "Добавь файл в репозиторий", "label": "action_request"},
        {"text": "Какой сейчас статус сборки?", "label": "status_check"},
        {"text": "Что по статусу деплоя?", "label": "status_check"},
        {"text": "Сколько еще осталось?", "label": "status_check"},
        {"text": "Почему падает сервер?", "label": "explanation"},
        {"text": "Как работает этот алгоритм?", "label": "explanation"},
    ]

    mock_embedder = MockTransformerEmbedder()
    adapter = TransformerIntent(embedder=mock_embedder)
    adapter.fit(records)

    assert adapter.training_examples == len(records)
    assert len(adapter.classes_) == 3

    # Inference on action request
    state = adapter.infer([{"role": "user", "content": "Исправь ошибку"}])
    assert state.kind in ("action_request", "status_check", "explanation", "unknown")
    assert state.confidence > 0.0
    assert state.belief is not None
    assert state.risk_decision is not None
    assert any("transformer_prediction" in ev for ev in state.evidence)


def test_transformer_intent_empty_turn() -> None:
    adapter = TransformerIntent(embedder=MockTransformerEmbedder())
    adapter.classifier = object()  # fitted stub
    state = adapter.infer([{"role": "user", "content": "   "}])
    assert state.kind == "unknown"
    assert state.needs_clarification is True
    assert "transformer:empty_user_turn" in state.evidence


def test_transformer_intent_unfitted_raises() -> None:
    adapter = TransformerIntent()
    with pytest.raises(RuntimeError, match="is not fitted"):
        adapter.infer([{"role": "user", "content": "тест"}])
