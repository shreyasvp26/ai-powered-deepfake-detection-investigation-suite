"""Git-tracked fixture layout (no-GPU plan A2)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURE_CROPS = ROOT / "fixtures" / "crops_demo"


def test_crops_demo_has_frame_pngs() -> None:
    paths = sorted(FIXTURE_CROPS.glob("frame_*.png"))
    assert len(paths) >= 2, "Expected tests/fixtures/crops_demo/frame_*.png"


def test_bundled_sample_json_exists() -> None:
    p = ROOT.parent / "app" / "sample_results" / "sample_result.json"
    assert p.is_file()
    from app.api_client import load_bundled_sample_result

    data = load_bundled_sample_result()
    assert data.get("verdict") in ("REAL", "FAKE")
    assert "blink_score" not in data and "Bs" not in data
