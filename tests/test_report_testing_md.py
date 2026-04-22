"""Unit tests for ``scripts/report_testing_md.py`` (dry-run / static YAML; no W&B)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "report_testing_md.py"
FIXTURE = REPO / "tests" / "fixtures" / "testing_md_dryrun.yaml"
TESTING = REPO / "docs" / "TESTING.md"


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "report_testing_md", REPO / "scripts" / "report_testing_md.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_render_contains_fixture_metrics() -> None:
    m = _load_report_module()
    data = m.load_results_yaml(FIXTURE)
    out = m.render_results_block(data)
    assert "0.9410" in out
    assert "86.0 %" in out
    assert "Celeb-DF v2 smoke" in out
    assert "V1F-11 GPU" in out


def test_replace_markers_roundtrip() -> None:
    m = _load_report_module()
    data = m.load_results_yaml(FIXTURE)
    block = m.render_results_block(data)
    before = f"x\n<!-- auto:results:start -->\nOLD\n<!-- auto:results:end -->\ny\n"
    after = m.replace_results_markers(before, block)
    assert "OLD" not in after
    assert "0.9410" in after
    assert after.startswith("x\n<!-- auto:results:start -->")


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_dry_run_cli_on_temp_testing_md(tmp_path: Path) -> None:
    import shutil

    tmd = tmp_path / "T.md"
    shutil.copyfile(TESTING, tmd)
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--testing-md",
            str(tmd),
            "--dry-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    text = tmd.read_text(encoding="utf-8")
    assert "0.9410" in text
    assert "<!-- auto:results:start -->" in text
    assert "<!-- auto:results:end -->" in text
