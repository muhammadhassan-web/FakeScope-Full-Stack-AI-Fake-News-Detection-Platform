"""Prediction service — loads ML artifacts and scores news text."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path for shared ML modules
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocess import clean_text  # noqa: E402


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    fake_prob: float
    real_prob: float
    cleaned_tokens: int
    model: str
    verdict: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "fake_prob": self.fake_prob,
            "real_prob": self.real_prob,
            "cleaned_tokens": self.cleaned_tokens,
            "model": self.model,
            "verdict": self.verdict,
            "explanation": self.explanation,
        }


class ModelNotReadyError(RuntimeError):
    """Raised when model artifacts are missing or unloadable."""


class Predictor:
    """Thread-safe-ish wrapper around the trained classifier."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.model_name = "Unknown"
        self.metrics: dict[str, Any] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self, auto_train: bool = True) -> None:
        model_path = self.model_dir / "best_model.pkl"
        vec_path = self.model_dir / "vectorizer.pkl"
        name_path = self.model_dir / "best_model_name.pkl"
        metrics_path = self.model_dir / "metrics_summary.pkl"

        if not model_path.exists() or not vec_path.exists():
            if not auto_train:
                raise ModelNotReadyError(
                    "Model artifacts not found. Run `python train.py` first."
                )
            logger.warning("Models not found — running training pipeline…")
            self._run_training()

        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            self.model_name = joblib.load(name_path) if name_path.exists() else "Unknown"
            if metrics_path.exists():
                raw = joblib.load(metrics_path)
                self.metrics = {
                    name: {
                        k: round(float(v), 4)
                        for k, v in vals.items()
                        if isinstance(v, (int, float)) and k != "time"
                    }
                    for name, vals in raw.items()
                }
            self._ready = True
            logger.info("Loaded model: %s", self.model_name)
        except Exception as exc:
            self._ready = False
            raise ModelNotReadyError(f"Failed to load model artifacts: {exc}") from exc

    def _run_training(self) -> None:
        root = self.model_dir.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import train

        train.main()

    def predict(self, text: str) -> PredictionResult:
        if not self._ready or self.model is None or self.vectorizer is None:
            raise ModelNotReadyError("Model is not loaded.")

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError(
                "Text contained no usable words after cleaning. "
                "Try a longer headline or article excerpt."
            )

        vec = self.vectorizer.transform([cleaned])
        pred = int(self.model.predict(vec)[0])
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec)[0]

        label = "FAKE" if pred == 1 else "REAL"
        if proba is not None:
            fake_p = float(proba[1]) * 100
            real_p = float(proba[0]) * 100
            conf = float(max(proba)) * 100
        else:
            fake_p = 100.0 if pred == 1 else 0.0
            real_p = 0.0 if pred == 1 else 100.0
            conf = 100.0

        verdict = "Likely fabricated" if label == "FAKE" else "Likely authentic"
        if conf < 60:
            explanation = (
                "Low confidence — the text sits near the decision boundary. "
                "Treat this as a weak signal and verify with trusted sources."
            )
        elif label == "FAKE":
            explanation = (
                "Language patterns resemble known misinformation: sensational "
                "framing, unverifiable claims, or conspiratorial cues."
            )
        else:
            explanation = (
                "Language patterns resemble verified reporting: measured tone, "
                "specific details, and fewer sensational markers."
            )

        return PredictionResult(
            label=label,
            confidence=round(conf, 1),
            fake_prob=round(fake_p, 1),
            real_prob=round(real_p, 1),
            cleaned_tokens=len(cleaned.split()),
            model=self.model_name,
            verdict=verdict,
            explanation=explanation,
        )


# Module-level singleton set by the app factory
predictor: Predictor | None = None


def get_predictor() -> Predictor:
    if predictor is None or not predictor.ready:
        raise ModelNotReadyError("Predictor is not initialized.")
    return predictor
