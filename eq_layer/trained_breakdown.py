from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ._deps import missing_joblib, missing_ml_dependency
from .model_state import FITTED, validate_state


def build_breakdown_pipeline():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError as exc:
        raise missing_ml_dependency("TrainedBreakdown", exc) from exc

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    lowercase=True,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=2e-6,
        max_iter=100,
        tol=1e-4,
        class_weight="balanced",
        random_state=0,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


@dataclass(frozen=True)
class BreakdownDecision:
    probability: float
    is_breakdown: bool
    evidence: tuple[str, ...] = ()


@dataclass
class TrainedBreakdown:
    model: object | None = None
    training_examples: int = 0
    positive_examples: int = 0
    threshold: float = 0.75
    name: str = "trained-dbdc3-breakdown"

    def fit(self, records: Iterable[object]) -> "TrainedBreakdown":
        rows = list(records)
        if not rows:
            raise ValueError("Breakdown training requires at least one record.")

        texts = [str(row.model_text) for row in rows]
        labels = [1 if bool(row.hard_breakdown) else 0 for row in rows]
        positives = sum(labels)
        if positives < 100:
            raise ValueError(f"Too few hard-breakdown examples: {positives}")

        self.model = build_breakdown_pipeline()
        self.model.fit(texts, labels)
        self.training_examples = len(rows)
        self.positive_examples = positives
        return self

    def predict(
        self,
        system_text: str,
        context: list[str] | tuple[str, ...] = (),
    ) -> BreakdownDecision:
        if self.model is None:
            raise RuntimeError("TrainedBreakdown is not fitted.")

        context_text = "\n".join(str(x) for x in context if str(x).strip())
        model_text = (
            f"{context_text}\n[SYSTEM]\n{system_text.strip()}"
            if context_text
            else system_text.strip()
        )
        probability = float(self.model.predict_proba([model_text])[0][1])
        return BreakdownDecision(
            probability=round(probability, 4),
            is_breakdown=probability >= self.threshold,
            evidence=(
                f"trained_breakdown_probability:{probability:.4f}",
                f"threshold:{self.threshold:.2f}",
            ),
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("TrainedBreakdown is not fitted.")
        try:
            import joblib
        except ImportError as exc:
            raise missing_joblib("Saving breakdown model", exc) from exc

        joblib.dump(
            {
                "model": self.model,
                "training_examples": self.training_examples,
                "positive_examples": self.positive_examples,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainedBreakdown":
        try:
            import joblib
        except ImportError as exc:
            raise missing_joblib("Loading breakdown model", exc) from exc

        try:
            payload = joblib.load(path)
        except ImportError as exc:
            raise missing_ml_dependency("Loading breakdown model", exc) from exc
        validate_state(
            payload,
            model="TrainedBreakdown",
            path=path,
            required={"model": FITTED},
            optional={
                "training_examples": (int,),
                "positive_examples": (int,),
                "threshold": (int, float),
            },
        )
        return cls(
            model=payload["model"],
            training_examples=int(payload.get("training_examples", 0)),
            positive_examples=int(payload.get("positive_examples", 0)),
            threshold=float(payload.get("threshold", 0.75)),
        )
