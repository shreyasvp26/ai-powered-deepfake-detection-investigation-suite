"""ReportGenerator: JSON + PDF output, FIX-9 (no blink / Bs)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fpdf")

from src.report.report_generator import ReportGenerator


def test_generate_json_pdf_minimal() -> None:
    gen = ReportGenerator()
    analysis = {
        "verdict": "REAL",
        "fusion_score": 0.31,
        "spatial_score": 0.35,
        "temporal_score": 0.12,
        "metadata": {"frames_analysed": 10},
        "technical": {"device": "cpu", "inference_time_s": 1.25},
    }
    with tempfile.TemporaryDirectory() as d:
        paths = gen.generate(analysis, d)
        jp = Path(paths["json_path"])
        pp = Path(paths["pdf_path"])
        assert jp.is_file() and pp.is_file()
        assert pp.stat().st_size > 100
        loaded = json.loads(jp.read_text(encoding="utf-8"))
        assert loaded["verdict"] == "REAL"
        assert "spatial_score" in loaded and "temporal_score" in loaded


def test_deprecated_blink_keys_stripped() -> None:
    gen = ReportGenerator()
    analysis = {
        "verdict": "FAKE",
        "fusion_score": 0.9,
        "spatial_score": 0.8,
        "temporal_score": "N/A",
        "blink_score": 0.1,
        "Bs": 0.2,
        "metadata": {},
        "technical": {},
    }
    with tempfile.TemporaryDirectory() as d:
        paths = gen.generate(analysis, d)
        loaded = json.loads(Path(paths["json_path"]).read_text(encoding="utf-8"))
        assert "blink_score" not in loaded
        assert "Bs" not in loaded


def test_fake_attribution_section_pdf_does_not_crash() -> None:
    gen = ReportGenerator()
    analysis = {
        "verdict": "FAKE",
        "fusion_score": 0.88,
        "spatial_score": 0.7,
        "temporal_score": 0.5,
        "attribution": {
            "predicted_method": "Deepfakes",
            "class_probabilities": {"Deepfakes": 0.5, "Face2Face": 0.2, "FaceSwap": 0.2, "NeuralTextures": 0.1},
        },
        "metadata": {},
        "technical": {},
    }
    with tempfile.TemporaryDirectory() as d:
        paths = gen.generate(analysis, d)
        assert Path(paths["pdf_path"]).stat().st_size > 200
