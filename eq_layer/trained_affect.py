from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .affect import HeuristicAffect
from .policies import AffectState

EMOBANK_COMMIT = "248ce2a43e165a66d31aeaed83cff9641d6654e0"
EMOBANK_URL = (
    "https://raw.githubusercontent.com/JULIELab/EmoBank/"
    f"{EMOBANK_COMMIT}/corpus/emobank.csv"
)
EMOBANK_LICENSE = "CC-BY-SA-4.0"


def build_affect_pipeline():
    """Build the deterministic VAD regression pipeline."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError as exc:
        raise ImportError(
            "TrainedAffect requires the optional ML dependencies. "
            'Install with: pip install "eq-layer[ml]"'
        ) from exc

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40000,
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
    regressor = Ridge(alpha=8.0)
    return Pipeline([("features", features), ("regressor", regressor)])


def load_emobank(path: str | Path, split: str | None = None) -> list[dict]:
    """Load official EmoBank CSV rows without introducing a pandas dependency."""
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"split", "V", "A", "D", "text"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"EmoBank is missing required columns: {sorted(missing)}")

        for row in reader:
            if split is not None and row["split"] != split:
                continue
            text = row["text"].strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "V": float(row["V"]),
                    "A": float(row["A"]),
                    "D": float(row["D"]),
                    "split": row["split"],
                }
            )
    return rows


@dataclass(frozen=True)
class DimensionalAffect:
    """Continuous VAD prediction in EmoBank's original 1..5 scale."""

    valence: float
    arousal: float
    dominance: float


@dataclass
class TrainedAffect:
    """Learned affect adapter trained on continuous VAD supervision.

    The regression model estimates Valence, Arousal and Dominance. EQ-Layer's
    policy state consumes normalized valence/arousal while structural subtext
    and stance remain explicit, inspectable signals.
    """

    model: object | None = None
    training_examples: int = 0
    structural: HeuristicAffect = field(default_factory=HeuristicAffect)
    name: str = "trained-emobank-tfidf-ridge"
    training_source: str = f"EmoBank@{EMOBANK_COMMIT}"
    training_license: str = EMOBANK_LICENSE

    def fit(self, records: Iterable[dict]) -> "TrainedAffect":
        rows = list(records)
        if not rows:
            raise ValueError("Affect training requires at least one record.")

        texts = [str(row["text"]) for row in rows]
        targets = [
            [float(row["V"]), float(row["A"]), float(row["D"])]
            for row in rows
        ]

        self.model = build_affect_pipeline()
        self.model.fit(texts, targets)
        self.training_examples = len(rows)
        return self

    @classmethod
    def from_emobank(
        cls,
        path: str | Path,
        split: str = "train",
    ) -> "TrainedAffect":
        return cls().fit(load_emobank(path, split=split))

    def predict_dimensional(self, text: str) -> DimensionalAffect:
        if self.model is None:
            raise RuntimeError("TrainedAffect is not fitted.")
        prediction = self.model.predict([text])[0]
        return DimensionalAffect(
            valence=self._clamp(float(prediction[0]), 1.0, 5.0),
            arousal=self._clamp(float(prediction[1]), 1.0, 5.0),
            dominance=self._clamp(float(prediction[2]), 1.0, 5.0),
        )

    def infer(
        self,
        messages: list[dict],
        turn_index: int = 0,
        annotated: dict | None = None,
    ) -> AffectState:
        if self.model is None:
            raise RuntimeError("TrainedAffect is not fitted.")

        annotated = annotated or {}
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not user_turns:
            return AffectState(
                valence=0.0,
                arousal=0.0,
                escalation_delta=0.0,
                stance="unknown",
                subtext="statement",
                turn_index=turn_index,
                history=(),
            )

        predictions = self.model.predict(user_turns)
        history: list[AffectState] = []
        previous_arousal = 0.0

        for i, prediction in enumerate(predictions):
            valence = self._normalise_valence(float(prediction[0]))
            arousal = self._normalise_arousal(float(prediction[1]))
            delta = arousal - previous_arousal if i else 0.0
            history.append(
                AffectState(
                    valence=round(valence, 4),
                    arousal=round(arousal, 4),
                    escalation_delta=round(delta, 4),
                    stance="unknown",
                    subtext="",
                    turn_index=i,
                )
            )
            previous_arousal = arousal

        current = user_turns[-1]
        state = history[-1]
        return AffectState(
            valence=state.valence,
            arousal=state.arousal,
            escalation_delta=state.escalation_delta,
            stance=self.structural._stance(current, annotated),
            subtext=self.structural._subtext(user_turns, annotated),
            turn_index=turn_index,
            history=tuple(history),
        )

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("TrainedAffect is not fitted.")
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Saving a trained affect model requires joblib.") from exc

        joblib.dump(
            {
                "model": self.model,
                "training_examples": self.training_examples,
                "training_source": self.training_source,
                "training_license": self.training_license,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainedAffect":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Loading a trained affect model requires joblib.") from exc
        payload = joblib.load(path)
        return cls(
            model=payload["model"],
            training_examples=int(payload.get("training_examples", 0)),
            training_source=str(payload.get("training_source", "unknown")),
            training_license=str(payload.get("training_license", "unknown")),
        )

    @staticmethod
    def _normalise_valence(value: float) -> float:
        # EmoBank V: 1 (negative) .. 5 (positive) -> -1 .. +1
        return TrainedAffect._clamp((value - 3.0) / 2.0, -1.0, 1.0)

    @staticmethod
    def _normalise_arousal(value: float) -> float:
        # EmoBank A: 1 (calm) .. 5 (excited) -> 0 .. 1
        return TrainedAffect._clamp((value - 1.0) / 4.0, 0.0, 1.0)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
