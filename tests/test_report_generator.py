"""ReportGenerator: JSON + PDF output, FIX-9 (no blink fields)."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("fpdf")

from src import ENGINE_VERSION
from src.report.checksums import sha256_bytes
from src.report.report_generator import ReportGenerator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
        assert loaded["engine_version"] == ENGINE_VERSION
        assert _SHA256.match(loaded["input_sha256"])
        assert loaded["seed"] == 42
        m = loaded["model_checksums"]
        for k in ("xception_c23", "dsan_v3", "fusion_lr"):
            assert k in m and _SHA256.match(m[k])


def test_deprecated_blink_keys_stripped() -> None:
    gen = ReportGenerator()
    analysis = {
        "verdict": "FAKE",
        "fusion_score": 0.9,
        "spatial_score": 0.8,
        "temporal_score": "N/A",
        "blink_score": 0.1,
        "metadata": {},
        "technical": {},
    }
    with tempfile.TemporaryDirectory() as d:
        paths = gen.generate(analysis, d)
        loaded = json.loads(Path(paths["json_path"]).read_text(encoding="utf-8"))
        assert "blink_score" not in loaded
        assert loaded["engine_version"] == ENGINE_VERSION


def test_v1f03_input_sha256_from_file_and_seed_metadata(tmp_path: Path) -> None:
    gen = ReportGenerator()
    video = tmp_path / "clip.bin"
    payload = b"\xff\x00not-a-real-video-lol"
    video.write_bytes(payload)
    analysis = {
        "verdict": "REAL",
        "fusion_score": 0.1,
        "spatial_score": 0.2,
        "temporal_score": 0.3,
        "metadata": {"input_video_path": str(video), "frames_analysed": 1, "seed": 7},
        "technical": {"device": "cpu"},
    }
    d = str(tmp_path / "out")
    paths = gen.generate(analysis, d)
    loaded = json.loads(Path(paths["json_path"]).read_text(encoding="utf-8"))
    assert loaded["engine_version"] == ENGINE_VERSION
    assert loaded["input_sha256"] == sha256_bytes(payload)
    assert loaded["seed"] == 7
    m = loaded["model_checksums"]
    assert set(m.keys()) == {"xception_c23", "dsan_v3", "fusion_lr"}
    for v in m.values():
        assert _SHA256.match(v)
    assert Path(paths["pdf_path"]).stat().st_size > 200


def test_fake_attribution_section_pdf_does_not_crash() -> None:
    gen = ReportGenerator()
    analysis = {
        "verdict": "FAKE",
        "fusion_score": 0.88,
        "spatial_score": 0.7,
        "temporal_score": 0.5,
        "attribution": {
            "predicted_method": "Deepfakes",
            "class_probabilities": {
                "Deepfakes": 0.5,
                "Face2Face": 0.2,
                "FaceSwap": 0.2,
                "NeuralTextures": 0.1,
            },
        },
        "metadata": {},
        "technical": {},
    }
    with tempfile.TemporaryDirectory() as d:
        paths = gen.generate(analysis, d)
        assert Path(paths["pdf_path"]).stat().st_size > 200
