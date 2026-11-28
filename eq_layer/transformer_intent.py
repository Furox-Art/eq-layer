"""Transformer-based intent belief adapter for EQ-Layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

from ._deps import missing_joblib, OptionalDependencyError
from .intent import HeuristicIntent, IntentState
from .intent_belief import IntentBelief, decide_intent_action
from .model_state import FITTED, validate_state

DEFAULT_TRANSFORMER_MODEL = "cointegrated/rubert-tiny2"


def _missing_transformer(purpose: str, cause: BaseException) -> OptionalDependencyError:
    return OptionalDependencyError(
        purpose,
        package="transformers and torch",
        cause=cause,
        install_command='pip install "eq-layer[transformers]"',
    )


class TransformerEmbeddingPipeline:
    """Extracts dense sentence embeddings using HuggingFace Transformers and PyTorch."""

    def __init__(self, model_name: str = DEFAULT_TRANSFORMER_MODEL, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise _missing_transformer("TransformerEmbeddingPipeline", exc) from exc

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load transformer model {model_name!r}: {exc}. "
                "Ensure model weights are available locally or cached."
            ) from exc

    def embed(self, texts: Sequence[str], batch_size: int = 32):
        import torch

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Mean pooling over attention mask
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = (
                    encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
                all_embeddings.append(mean_pooled)

        import numpy as np
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 312))


@dataclass
class TransformerIntent:
    """Intent belief state classifier powered by lightweight transformer embeddings."""

    name: str = "transformer-intent"
    model_name: str = DEFAULT_TRANSFORMER_MODEL
    classifier: object | None = None
    embedder: TransformerEmbeddingPipeline | None = None
    classes_: list[str] = field(default_factory=list)
    training_examples: int = 0
    fallback: HeuristicIntent = field(default_factory=HeuristicIntent)

    @classmethod
    def from_bundled(
        cls,
        model_name: str = DEFAULT_TRANSFORMER_MODEL,
    ) -> "TransformerIntent":
        data = resources.files("eq_layer").joinpath("data", "intent_train.jsonl")
        with data.open("r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        adapter = cls(model_name=model_name)
        adapter.fit(records)
        return adapter

    def fit(self, records: Iterable[dict]) -> "TransformerIntent":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            from ._deps import missing_ml_dependency
            raise missing_ml_dependency("TransformerIntent", exc) from exc

        rows = list(records)
        texts = [str(r["text"]) for r in rows]
        labels = [str(r["label"]) for r in rows]
        if len(set(labels)) < 2:
            raise ValueError("Intent training requires at least two labels.")

        if self.embedder is None:
            self.embedder = TransformerEmbeddingPipeline(model_name=self.model_name)

        features = self.embedder.embed(texts)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        clf.fit(features, labels)

        self.classifier = clf
        self.classes_ = [str(c) for c in clf.classes_]
        self.training_examples = len(rows)
        return self

    def infer(self, messages: list[dict]) -> IntentState:
        if self.classifier is None or self.embedder is None:
            raise RuntimeError("TransformerIntent is not fitted.")

        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        current = user_turns[-1].strip() if user_turns else ""
        low = self.fallback._normalise(current)
        constraints = self.fallback._constraints(low)

        if not current:
            return IntentState(
                kind="unknown",
                canonical_request=self.fallback._canonical("unknown", current, constraints),
                response_mode="clarify",
                confidence=0.0,
                explicit=False,
                needs_clarification=True,
                constraints=constraints,
                evidence=("transformer:empty_user_turn",),
            )

        feat = self.embedder.embed([current])
        probabilities = self.classifier.predict_proba(feat)[0]
        posterior = {
            str(label): float(prob)
            for label, prob in zip(self.classes_, probabilities, strict=True)
        }
        belief = IntentBelief.from_mapping(posterior)
        decision = decide_intent_action(belief)

        kind = decision.selected_intent or "unknown"
        explicit = kind in {"action_request", "status_check", "explanation", "question"}
        needs_clarification = decision.action == "clarify"
        response_mode = {
            "action_request": "execute",
            "status_check": "direct",
            "explanation": "explain",
            "question": "direct",
            "statement": "contextual",
            "unknown": "clarify",
        }[kind]

        return IntentState(
            kind=kind,
            canonical_request=self.fallback._canonical(kind, current, constraints),
            response_mode=response_mode,
            confidence=belief.top_probability,
            explicit=explicit,
            needs_clarification=needs_clarification,
            constraints=constraints,
            evidence=(
                f"transformer_prediction:{kind}",
                f"top_probability:{belief.top_probability:.4f}",
                f"model:{self.model_name}",
            ),
            belief=belief,
            risk_decision=decision,
        )
