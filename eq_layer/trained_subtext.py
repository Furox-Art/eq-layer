from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .affect import CORRECTION_MARKERS, DISCLOSURE_MARKERS, HeuristicAffect
from .policies import AffectState

XDAILYDIALOG_COMMIT = "6e7ecf54c9f169215b4b8c18995c7aac74117127"
XDAILYDIALOG_BASE = (
    "https://raw.githubusercontent.com/liuzeming01/XDailyDialog/"
    f"{XDAILYDIALOG_COMMIT}/data"
)
XDAILYDIALOG_URLS = {
    "train": f"{XDAILYDIALOG_BASE}/en_train_human.txt",
    "dev": f"{XDAILYDIALOG_BASE}/en_dev_human.txt",
    "test": f"{XDAILYDIALOG_BASE}/en_test_human.txt",
}

ACT_MAP = {
    "1": "inform",
    "2": "question",
    "3": "directive",
    "4": "commissive",
}

EMOTION_MAP = {
    "0": "neutral",
    "1": "anger",
    "2": "disgust",
    "3": "fear",
    "4": "happiness",
    "5": "sadness",
    "6": "surprise",
}


@dataclass(frozen=True)
class DialogueSignals:
    act: str
    emotion: str
    act_confidence: float
    emotion_confidence: float


@dataclass(frozen=True)
class SubtextDecision:
    label: str
    confidence: float
    signals: DialogueSignals | None
    evidence_level: str = "unknown"
    evidence: tuple[str, ...] = ()


def load_xdailydialog(path: str | Path) -> list[dict]:
    """Parse XDailyDialog's combined human split into utterance-level records."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"XDailyDialog line {line_number} has {len(parts)} fields; expected 4."
                )

            dialogue, topic, acts_raw, emotions_raw = parts
            utterances = [u.strip() for u in dialogue.split("__eou__") if u.strip()]
            acts = acts_raw.strip().split()
            emotions = emotions_raw.strip().split()

            if not (len(utterances) == len(acts) == len(emotions)):
                raise ValueError(
                    "XDailyDialog alignment error on line "
                    f"{line_number}: utterances={len(utterances)} "
                    f"acts={len(acts)} emotions={len(emotions)}"
                )

            for turn_index, (text, act_code, emotion_code) in enumerate(
                zip(utterances, acts, emotions, strict=True)
            ):
                if act_code not in ACT_MAP:
                    raise ValueError(f"Unknown XDailyDialog act code: {act_code}")
                if emotion_code not in EMOTION_MAP:
                    raise ValueError(f"Unknown XDailyDialog emotion code: {emotion_code}")
                rows.append(
                    {
                        "text": text,
                        "act": ACT_MAP[act_code],
                        "emotion": EMOTION_MAP[emotion_code],
                        "topic": topic,
                        "turn_index": turn_index,
                    }
                )
    return rows


def build_feature_extractor():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
    except ImportError as exc:
        raise ImportError(
            "TrainedSubtext requires the optional ML dependencies. "
            'Install with: pip install "eq-layer[ml]"'
        ) from exc

    return FeatureUnion(
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


def build_classifier():
    try:
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:
        raise ImportError(
            "TrainedSubtext requires the optional ML dependencies. "
            'Install with: pip install "eq-layer[ml]"'
        ) from exc

    return SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=80,
        tol=1e-4,
        class_weight="balanced",
        random_state=0,
    )


@dataclass
class TrainedSubtext:
    """Learn dialogue-act/emotion signals and derive EQ-Layer subtext conservatively.

    XDailyDialog directly supervises dialogue act and basic emotion, not
    EQ-Layer's full subtext taxonomy. Challenge, demand and exhaustion are
    therefore explicit compositions of learned signals and learned V/A state,
    not falsely advertised as directly annotated labels.
    """

    features: object | None = None
    act_model: object | None = None
    emotion_model: object | None = None
    training_examples: int = 0
    act_confidence_threshold: float = 0.40
    emotion_confidence_threshold: float = 0.40
    structural: HeuristicAffect = field(default_factory=HeuristicAffect)
    name: str = "trained-xdailydialog-signals"

    def fit(self, records: Iterable[dict]) -> "TrainedSubtext":
        rows = list(records)
        if not rows:
            raise ValueError("Subtext training requires at least one record.")

        texts = [str(row["text"]) for row in rows]
        act_labels = [str(row["act"]) for row in rows]
        emotion_labels = [str(row["emotion"]) for row in rows]

        self.features = build_feature_extractor()
        matrix = self.features.fit_transform(texts)

        self.act_model = build_classifier()
        self.act_model.fit(matrix, act_labels)

        self.emotion_model = build_classifier()
        self.emotion_model.fit(matrix, emotion_labels)

        self.training_examples = len(rows)
        return self

    @classmethod
    def from_xdailydialog(cls, path: str | Path) -> "TrainedSubtext":
        return cls().fit(load_xdailydialog(path))

    def predict_signals(self, text: str) -> DialogueSignals:
        if self.features is None or self.act_model is None or self.emotion_model is None:
            raise RuntimeError("TrainedSubtext is not fitted.")

        matrix = self.features.transform([text])
        act_probs = self.act_model.predict_proba(matrix)[0]
        emotion_probs = self.emotion_model.predict_proba(matrix)[0]

        act_index = int(act_probs.argmax())
        emotion_index = int(emotion_probs.argmax())

        return DialogueSignals(
            act=str(self.act_model.classes_[act_index]),
            emotion=str(self.emotion_model.classes_[emotion_index]),
            act_confidence=round(float(act_probs[act_index]), 4),
            emotion_confidence=round(float(emotion_probs[emotion_index]), 4),
        )

    def infer(
        self,
        messages: list[dict],
        affect: AffectState,
        annotated: dict | None = None,
    ) -> SubtextDecision:
        annotated = annotated or {}
        user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
        current = user_turns[-1].strip() if user_turns else ""

        if annotated.get("subtext"):
            return SubtextDecision(
                label=str(annotated["subtext"]),
                confidence=1.0,
                signals=None,
                evidence_level="verified",
                evidence=("annotation",),
            )

        if not current:
            return SubtextDecision(
                label="statement",
                confidence=0.0,
                signals=None,
                evidence_level="fallback",
                evidence=("empty_user_turn",),
            )

        if self.structural._has(current, CORRECTION_MARKERS):
            return SubtextDecision(
                label="correction",
                confidence=1.0,
                signals=None,
                evidence_level="structural",
                evidence=("structural:correction_marker",),
            )
        if self.structural._has(current, DISCLOSURE_MARKERS):
            return SubtextDecision(
                label="disclosure_request",
                confidence=1.0,
                signals=None,
                evidence_level="structural",
                evidence=("structural:disclosure_marker",),
            )
        if self.structural._escalating(user_turns):
            return SubtextDecision(
                label="escalating",
                confidence=0.95,
                signals=None,
                evidence_level="structural",
                evidence=("structural:repeated_question_run",),
            )
        if self.structural._demand_pattern(user_turns):
            return SubtextDecision(
                label="demand",
                confidence=0.95,
                signals=None,
                evidence_level="structural",
                evidence=("structural:stacked_short_demands",),
            )

        signals = self.predict_signals(current)
        act_reliable = signals.act_confidence >= self.act_confidence_threshold
        emotion_reliable = signals.emotion_confidence >= self.emotion_confidence_threshold

        if (
            act_reliable
            and signals.act == "directive"
            and (
                affect.arousal >= 0.55
                or (
                    emotion_reliable
                    and signals.emotion in {"anger", "disgust"}
                )
            )
        ):
            return self._decision(
                "demand",
                signals,
                "derived:directive_plus_heat",
                evidence_level="derived",
            )

        if (
            act_reliable
            and signals.act == "question"
            and emotion_reliable
            and signals.emotion in {"anger", "disgust"}
            and affect.arousal >= 0.45
        ):
            return self._decision(
                "challenge",
                signals,
                "derived:question_plus_negative_high_arousal",
                evidence_level="derived",
            )

        if (
            act_reliable
            and signals.act in {"inform", "commissive"}
            and emotion_reliable
            and signals.emotion == "sadness"
            and affect.valence <= -0.10
            and affect.arousal <= 0.42
        ):
            label = "resignation" if self._drained_run(affect) else "exhaustion"
            return self._decision(
                label,
                signals,
                "derived:sadness_plus_low_valence_low_arousal",
                evidence_level="derived",
            )

        if act_reliable and signals.act == "question":
            return self._decision(
                "question",
                signals,
                "learned_act:question",
                evidence_level="direct_learned",
            )

        # A calm directive is not automatically a "demand". Intent handles the
        # requested action; subtext remains plain unless heat is independently
        # supported.
        if act_reliable and signals.act in {"inform", "directive", "commissive"}:
            return self._decision(
                "statement",
                signals,
                f"learned_act:{signals.act}",
                evidence_level="direct_learned",
            )

        # Low-confidence learned signals fall back to the auditable structural
        # tracker instead of being converted into a confident subtext.
        fallback = self.structural._subtext(user_turns, {})
        return SubtextDecision(
            label=fallback,
            confidence=max(signals.act_confidence, signals.emotion_confidence),
            signals=signals,
            evidence_level="fallback",
            evidence=("low_confidence:fallback_structural",),
        )

    def save(self, path: str | Path) -> None:
        if self.features is None or self.act_model is None or self.emotion_model is None:
            raise RuntimeError("TrainedSubtext is not fitted.")
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Saving a trained subtext model requires joblib.") from exc

        joblib.dump(
            {
                "features": self.features,
                "act_model": self.act_model,
                "emotion_model": self.emotion_model,
                "training_examples": self.training_examples,
                "act_confidence_threshold": self.act_confidence_threshold,
                "emotion_confidence_threshold": self.emotion_confidence_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainedSubtext":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Loading a trained subtext model requires joblib.") from exc

        payload = joblib.load(path)
        return cls(
            features=payload["features"],
            act_model=payload["act_model"],
            emotion_model=payload["emotion_model"],
            training_examples=int(payload.get("training_examples", 0)),
            act_confidence_threshold=float(payload.get("act_confidence_threshold", 0.40)),
            emotion_confidence_threshold=float(
                payload.get("emotion_confidence_threshold", 0.40)
            ),
        )

    @staticmethod
    def _drained_run(affect: AffectState) -> bool:
        recent = affect.history[-2:]
        return len(recent) >= 2 and all(
            state.valence <= -0.10 and state.arousal <= 0.42 for state in recent
        )

    @staticmethod
    def _decision(
        label: str,
        signals: DialogueSignals,
        evidence: str,
        *,
        evidence_level: str,
    ) -> SubtextDecision:
        confidence = min(
            1.0,
            max(signals.act_confidence, signals.emotion_confidence),
        )
        return SubtextDecision(
            label=label,
            confidence=round(confidence, 4),
            signals=signals,
            evidence_level=evidence_level,
            evidence=(evidence,),
        )
