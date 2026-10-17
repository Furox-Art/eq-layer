from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ._deps import missing_joblib, missing_ml_dependency
from .model_state import FITTED, validate_state

COARSE_DISCOURSE_ARCHIVE_SHA256 = "33cc25e906e677881c0031c1e134460b1e389fac707f2e52fe59b484dc6d0b61"


def build_disagreement_pipeline():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError as exc:
        raise missing_ml_dependency("TrainedDisagreement", exc) from exc

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
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
                    max_features=50000,
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
class DisagreementDecision:
    probability: float
    is_disagreement: bool
    evidence: tuple[str, ...] = ()


@dataclass
class TrainedDisagreement:
    model: object | None = None
    training_examples: int = 0
    positive_examples: int = 0
    threshold: float = 0.70
    name: str = "trained-coarse-discourse-disagreement"

    def fit(self, records: Iterable[object]) -> TrainedDisagreement:
        rows = list(records)
        if not rows:
            raise ValueError("Disagreement training requires at least one record.")

        texts = [str(row.model_text) for row in rows]
        labels = [1 if str(row.label).lower() == "disagreement" else 0 for row in rows]
        positives = sum(labels)
        if positives < 100:
            raise ValueError(f"Too few disagreement examples for training: {positives}")

        self.model = build_disagreement_pipeline()
        self.model.fit(texts, labels)
        self.training_examples = len(rows)
        self.positive_examples = positives
        return self

    def predict(
        self,
        reply_text: str,
        context_text: str = "",
    ) -> DisagreementDecision:
        if self.model is None:
            raise RuntimeError("TrainedDisagreement is not fitted.")

        model_text = (
            f"{context_text.strip()}\n[REPLY]\n{reply_text.strip()}"
            if context_text.strip()
            else reply_text.strip()
        )
        probability = float(self.model.predict_proba([model_text])[0][1])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("classifier probability must be finite and in [0, 1]")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or not 0.0 <= self.threshold <= 1.0
        ):
            raise ValueError("classifier threshold must be finite and in [0, 1]")
        return DisagreementDecision(
            probability=round(probability, 4),
            is_disagreement=probability >= self.threshold,
            evidence=(
                f"trained_disagreement_probability:{probability:.4f}",
                f"threshold:{self.threshold:.2f}",
            ),
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("TrainedDisagreement is not fitted.")
        try:
            import joblib
        except ImportError as exc:
            raise missing_joblib("Saving disagreement model", exc) from exc

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
    def load(cls, path: str | Path) -> TrainedDisagreement:
        try:
            import joblib
        except ImportError as exc:
            raise missing_joblib("Loading disagreement model", exc) from exc

        try:
            payload = joblib.load(path)
        except ImportError as exc:
            raise missing_ml_dependency("Loading disagreement model", exc) from exc
        validate_state(
            payload,
            model="TrainedDisagreement",
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
            threshold=float(payload.get("threshold", 0.70)),
        )
