"""Fusion layer: combine spatial Ss and temporal Ts into final fake probability F.

Plan alignment (docs/PROJECT_PLAN_v10.md §9 / §7):
- Preferred: StandardScaler + LogisticRegression pipeline trained on (Ss,Ts).
- Fallback: if temporal is unavailable (<2 frames), return F = Ss (do NOT feed [Ss,0] to LR).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class FusionResult:
    fusion_score: float
    verdict: str  # REAL/FAKE
    used_fallback: bool


class FusionLayer:
    def __init__(
        self, model_path: str | Path = "models/fusion_lr.pkl", threshold: float = 0.5
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self._model: Any | None = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Missing fusion LR model: {self.model_path}. "
                "Train it with training/fit_fusion_lr.py or use fallback only."
            )
        self._model = joblib.load(self.model_path)

    def predict(self, ss: float, ts: float | None, n_frames: int) -> FusionResult:
        ss_f = float(ss)
        if n_frames < 2 or ts is None:
            f = ss_f
            verdict = "FAKE" if f >= self.threshold else "REAL"
            return FusionResult(fusion_score=float(f), verdict=verdict, used_fallback=True)

        if self._model is None:
            self.load()

        X = np.asarray([[ss_f, float(ts)]], dtype=np.float32)
        proba = float(self._model.predict_proba(X)[0, 1])
        verdict = "FAKE" if proba >= self.threshold else "REAL"
        return FusionResult(fusion_score=proba, verdict=verdict, used_fallback=False)
