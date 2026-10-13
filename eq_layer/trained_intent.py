from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

from ._deps import missing_joblib, missing_ml_dependency
from .intent import HeuristicIntent, IntentState
from .intent_belief import IntentBelief, decide_intent_action, fuse_dialogue_evidence
from .model_state import FITTED, validate_state


def build_pipeline():
    """Build the deterministic text classifier used by TrainedIntent."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError as exc:
        raise missing_ml_dependency("TrainedIntent", exc) from exc

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=0,
        C=10.0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def load_jsonl_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bundled_training_records() -> list[dict]:
    data = resources.files("eq_layer").joinpath("data", "intent_train.jsonl")
    with data.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bundled_records_path() -> Path:
    """Locate the bundled corpus on the filesystem, when it has a path.

    The cache key needs path+mtime so an edited corpus invalidates the memo.
    Zipped/egg distributions expose no real path, in which case this raises and
    the cache falls back to a stable key (a zipped corpus cannot change under
    the running process anyway).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate in (parent / "eq_layer" / "data", parent / "data"):
            artifact = candidate / "intent_train.jsonl"
            if artifact.is_file():
                return artifact
    raise FileNotFoundError("Bundled intent training corpus has no filesystem path.")


def _bundled_cache_key(
    confidence_threshold: float,
    margin_threshold: float,
) -> tuple[str, int, float, float]:
    """Cache key for the bundled fit: artifact path + mtime + thresholds."""
    try:
        artifact = bundled_records_path()
    except FileNotFoundError:
        return (":resources:", 0, float(confidence_threshold), float(margin_threshold))
    return (
        str(artifact),
        artifact.stat().st_mtime_ns,
        float(confidence_threshold),
        float(margin_threshold),
    )


@lru_cache(maxsize=8)
def _fit_bundled(
    cls: type["TrainedIntent"],
    artifact: str,
    mtime_ns: int,
    confidence_threshold: float,
    margin_threshold: float,
) -> "TrainedIntent":
    """Fit once per (artifact, mtime, thresholds); cached for the process."""
    adapter = cls(
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    adapter.fit(bundled_training_records())
    return adapter


def clear_bundled_cache() -> None:
    """Drop the memoised bundled fits (used by tests and long-lived processes)."""
    _fit_bundled.cache_clear()


@dataclass
class TrainedIntent:
    """Trainable intent adapter.

    The bundled model is deliberately lightweight: word and character TF-IDF
    features feed a balanced multinomial logistic regression classifier.
    It is a learned classifier, not a claim of deep semantic understanding.
    """

    confidence_threshold: float = 0.40
    margin_threshold: float = 0.08
    model: object | None = None
    training_examples: int = 0
    fallback: HeuristicIntent = field(default_factory=HeuristicIntent)
    name: str = "trained-tfidf-logreg"

    @classmethod
    def from_bundled(
        cls,
        confidence_threshold: float = 0.40,
        margin_threshold: float = 0.08,
    ) -> "TrainedIntent":
        """Return the bundled intent adapter, fitting it at most once.

        The fit is deterministic but expensive (~seconds), so it is memoised
        per process on (artifact path, artifact mtime, both thresholds). A
        repeated call is O(1) and returns the *same* adapter instance, which
        therefore has to be treated as read-only: callers must not call
        ``fit`` on it or they would mutate every other user of the cache.
        """
        return _fit_bundled(
            cls,
            *_bundled_cache_key(confidence_threshold, margin_threshold),
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        confidence_threshold: float = 0.40,
        margin_threshold: float = 0.08,
    ) -> "TrainedIntent":
        adapter = cls(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        adapter.fit(load_jsonl_records(path))
        return adapter

    def fit(self, records: Iterable[dict]) -> "TrainedIntent":
        rows = list(records)
        texts = [str(row["text"]) for row in rows]
        labels = [str(row["label"]) for row in rows]
        if len(set(labels)) < 2:
            raise ValueError("Intent training requires at least two labels.")
        self.model = build_pipeline()
        self.model.fit(texts, labels)
        self.training_examples = len(rows)
        return self

    def infer(self, messages: list[dict]) -> IntentState:
        if self.model is None:
            raise RuntimeError("TrainedIntent is not fitted.")

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
                evidence=("trained:empty_user_turn",),
            )

        probabilities = self.model.predict_proba([current])[0]
        classes = self.model.classes_
        posterior = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities, strict=True)
        }
        belief = IntentBelief.from_mapping(posterior)
        decision = decide_intent_action(belief)

        predicted = belief.top_kind
        confidence = belief.top_probability
        margin = belief.margin

        # Bayes-risk routing replaces the old confidence+margin gate. The
        # legacy thresholds remain serialized for backwards compatibility and
        # diagnostics, but they do not decide the production route.
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
            confidence=round(confidence, 4),
            explicit=explicit,
            needs_clarification=needs_clarification,
            constraints=constraints,
            evidence=(
                f"trained_prediction:{predicted}",
                f"probability:{confidence:.4f}",
                f"margin:{margin:.4f}",
                f"belief_entropy:{belief.normalized_entropy:.4f}",
                f"routing:{decision.rationale}",
                f"training_examples:{self.training_examples}",
            ),
            belief=belief,
            risk_decision=decision,
        )

    def reconcile_with_dialogue_evidence(
        self,
        state: IntentState,
        messages: list[dict],
        signals: object | None,
    ) -> IntentState:
        """Fuse dialogue-act/structural evidence without discarding raw posterior."""
        if state.belief is None:
            return state

        user_turns = [
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        ]
        current = user_turns[-1].strip() if user_turns else ""

        act = getattr(signals, "act", None) if signals is not None else None
        act_confidence = (
            float(getattr(signals, "act_confidence", 0.0))
            if signals is not None
            else 0.0
        )
        fused, fusion_evidence = fuse_dialogue_evidence(
            state.belief,
            text=current,
            dialogue_act=act,
            act_confidence=act_confidence,
        )
        if not fusion_evidence:
            return state

        decision = decide_intent_action(fused)
        kind = decision.selected_intent or "unknown"
        explicit = kind in {
            "action_request",
            "status_check",
            "explanation",
            "question",
        }
        needs_clarification = decision.action == "clarify"
        response_mode = {
            "action_request": "execute",
            "status_check": "direct",
            "explanation": "explain",
            "question": "direct",
            "statement": "contextual",
            "unknown": "clarify",
        }[kind]
        constraints = state.constraints

        return replace(
            state,
            kind=kind,
            canonical_request=self.fallback._canonical(
                kind,
                current,
                constraints,
            ),
            response_mode=response_mode,
            confidence=round(fused.top_probability, 4),
            explicit=explicit,
            needs_clarification=needs_clarification,
            evidence=state.evidence + tuple(
                f"belief_fusion:{item}" for item in fusion_evidence
            ) + (
                f"fused_routing:{decision.rationale}",
            ),
            belief=fused,
            risk_decision=decision,
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("TrainedIntent is not fitted.")
        try:
            import joblib
        except ImportError as exc:
            raise missing_joblib("Saving a trained intent model", exc) from exc
        joblib.dump(
            {
                "model": self.model,
                "confidence_threshold": self.confidence_threshold,
                "margin_threshold": self.margin_threshold,
                "training_examples": self.training_examples,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainedIntent":
        try:
            import joblib
        except ImportError as exc:
            raise missing_joblib("Loading a trained intent model", exc) from exc
        payload = joblib.load(path)
        validate_state(
            payload,
            model="TrainedIntent",
            path=path,
            required={"model": FITTED, "confidence_threshold": (int, float)},
            optional={
                "margin_threshold": (int, float),
                "training_examples": (int,),
            },
        )
        return cls(
            confidence_threshold=float(payload["confidence_threshold"]),
            margin_threshold=float(payload.get("margin_threshold", 0.08)),
            model=payload["model"],
            training_examples=int(payload.get("training_examples", 0)),
        )
