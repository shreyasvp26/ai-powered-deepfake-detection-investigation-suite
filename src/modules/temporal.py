"""Module 2: temporal consistency score (Ts) from per-frame Xception predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils import load_config

_DEFAULT_WEIGHTS: dict[str, float] = {
    "global_variance": 0.30,
    "sign_flip_rate": 0.25,
    "max_window_variance": 0.25,
    "max_jump": 0.20,
}


def _merge_weights(base: dict[str, float], override: dict[str, Any] | None) -> dict[str, float]:
    if not override:
        return dict(base)
    merged = {**base}
    for k, v in override.items():
        if k in base:
            merged[k] = float(v)
    return merged


class TemporalAnalyzer:
    """Four-feature temporal score over per-frame fake probabilities."""

    def __init__(
        self,
        window_size: int = 30,
        weights: dict[str, float] | None = None,
        inference_config_path: str | Path | None = None,
    ) -> None:
        ws = window_size
        w = dict(_DEFAULT_WEIGHTS)
        if weights:
            w = _merge_weights(w, weights)
        if inference_config_path is not None:
            cfg = load_config(inference_config_path)
            t = cfg.get("temporal") or {}
            if "window_size" in t:
                ws = int(t["window_size"])
            if isinstance(t.get("weights"), dict):
                w = _merge_weights(w, t["weights"])
        self.window_size = ws
        self.weights = w
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def analyze(self, per_frame_predictions: list[float]) -> dict[str, float]:
        preds = np.array(per_frame_predictions, dtype=np.float32)
        n = len(preds)

        if n == 0:
            return {
                "temporal_score": 0.5,
                "global_variance": 0.0,
                "sign_flip_rate": 0.0,
                "max_window_variance": 0.0,
                "max_jump": 0.0,
                "mean_jump": 0.0,
            }

        global_variance = float(np.var(preds))

        binary = (preds > 0.5).astype(int)
        sign_flips = int(np.sum(np.abs(np.diff(binary))))
        sign_flip_rate = sign_flips / max(n - 1, 1)

        if n > 1:
            jumps = np.abs(np.diff(preds))
            max_jump = float(np.max(jumps))
            mean_jump = float(np.mean(jumps))
        else:
            max_jump = 0.0
            mean_jump = 0.0

        if n >= self.window_size:
            ws = self.window_size
            window_vars = [float(np.var(preds[i:i+ws])) for i in range(n - ws + 1)]
            max_window_var = max(window_vars) if window_vars else global_variance
        else:
            max_window_var = global_variance

        raw_score = (
            self.weights["global_variance"] * min(global_variance * 10, 1.0)
            + self.weights["sign_flip_rate"] * min(sign_flip_rate, 1.0)
            + self.weights["max_window_variance"] * min(max_window_var * 10, 1.0)
            + self.weights["max_jump"] * min(max_jump, 1.0)
        )
        ts = float(np.clip(raw_score, 0.0, 1.0))

        return {
            "temporal_score": ts,
            "global_variance": global_variance,
            "sign_flip_rate": sign_flip_rate,
            "max_window_variance": max_window_var,
            "max_jump": max_jump,
            "mean_jump": mean_jump,
        }
