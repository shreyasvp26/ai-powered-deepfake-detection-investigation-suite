"""Forensic analysis reports: JSON + PDF (fpdf2).

FIX-9: blink score (Bs) is deprecated and must not appear in any report output.

V1F-03: every JSON report includes ``engine_version``, ``input_sha256``,
``model_checksums``, and ``seed``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src import ENGINE_VERSION
from src.report.checksums import build_model_checksums, resolve_input_sha256

_DEPRECATED_TOP_LEVEL = frozenset({"blink_score", "blink", "Bs", "bs"})

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _json_default(o: Any) -> str:
    return str(o)


def _resolve_seed(data: dict[str, Any]) -> int:
    raw = data.get("seed")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    for block in ("metadata", "technical"):
        b = data.get(block)
        if isinstance(b, dict) and "seed" in b:
            try:
                return int(b["seed"])
            except (TypeError, ValueError):
                pass
    return 42


class ReportGenerator:
    """Generate JSON and PDF forensic reports from a pipeline / dashboard result dict."""

    def generate(self, analysis_result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
        """Write JSON and PDF under ``output_dir``.

        Returns paths with keys ``json_path``, ``pdf_path``.
        """
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out / f"report_{ts}.json"
        pdf_path = out / f"report_{ts}.pdf"

        payload = {k: v for k, v in analysis_result.items() if k not in _DEPRECATED_TOP_LEVEL}
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now().isoformat(timespec="seconds")

        models_dir = _REPO_ROOT / "models"
        payload["engine_version"] = ENGINE_VERSION
        payload["input_sha256"] = resolve_input_sha256(payload)
        payload["model_checksums"] = build_model_checksums(models_dir)
        payload["seed"] = _resolve_seed(payload)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default, ensure_ascii=False)

        self._generate_pdf(payload, str(pdf_path))

        return {"json_path": str(json_path), "pdf_path": str(pdf_path)}

    def _generate_pdf(self, result: dict[str, Any], pdf_path: str) -> None:
        """Structured PDF summary: verdict, metadata, Ss/Ts only (no Bs)."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)

        nx, ny = XPos.LMARGIN, YPos.NEXT
        pdf.cell(0, 10, "DeepFake Detection Forensic Report", new_x=nx, new_y=ny, align="C")
        pdf.set_font("Helvetica", size=10)
        ts = result.get("timestamp", datetime.now().isoformat(timespec="seconds"))
        pdf.cell(0, 6, f"Generated: {ts}", new_x=nx, new_y=ny, align="C")
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "1. Verdict", new_x=nx, new_y=ny)
        pdf.set_font("Helvetica", size=11)
        verdict = result.get("verdict", "UNKNOWN")
        f_score = result.get("fusion_score", 0.0)
        try:
            f_num = float(f_score)
            f_line = f"Fusion Score F: {f_num:.4f}"
        except (TypeError, ValueError):
            f_line = f"Fusion Score F: {f_score}"
        pdf.cell(0, 7, f"Verdict:        {verdict}", new_x=nx, new_y=ny)
        pdf.cell(0, 7, f_line, new_x=nx, new_y=ny)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "2. Video Metadata", new_x=nx, new_y=ny)
        pdf.set_font("Helvetica", size=11)
        meta = result.get("metadata", {})
        if isinstance(meta, dict):
            for key, val in meta.items():
                pdf.cell(0, 7, f"  {key}: {val}", new_x=nx, new_y=ny)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "3. Detection Breakdown", new_x=nx, new_y=ny)
        pdf.set_font("Helvetica", size=11)
        ss = result.get("spatial_score", "N/A")
        ts_score = result.get("temporal_score", "N/A")
        pdf.cell(0, 7, f"  Spatial Score Ss:  {ss}", new_x=nx, new_y=ny)
        pdf.cell(0, 7, f"  Temporal Score Ts: {ts_score}", new_x=nx, new_y=ny)
        pdf.ln(3)

        if str(result.get("verdict")) == "FAKE" and "attribution" in result:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "4. Attribution", new_x=nx, new_y=ny)
            pdf.set_font("Helvetica", size=11)
            attr = result["attribution"]
            if isinstance(attr, dict):
                pred_method = attr.get("predicted_method", "Unknown")
                pdf.cell(0, 7, f"  Predicted Method: {pred_method}", new_x=nx, new_y=ny)
                probs = attr.get("class_probabilities", {})
                if isinstance(probs, dict):
                    for method, prob in sorted(probs.items(), key=lambda x: str(x[0])):
                        try:
                            p = float(prob)
                            line = f"    {method}: {p:.2%}"
                        except (TypeError, ValueError):
                            line = f"    {method}: {prob}"
                        pdf.cell(0, 7, line, new_x=nx, new_y=ny)
            pdf.ln(3)

        heatmaps = result.get("heatmap_paths")
        if heatmaps:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, "5. Explainability (Grad-CAM++)", new_x=nx, new_y=ny)
            pdf.set_font("Helvetica", size=11)
            if isinstance(heatmaps, dict):
                for label, path in heatmaps.items():
                    pdf.cell(0, 7, f"  {label}:", new_x=nx, new_y=ny)
                    pth = Path(str(path))
                    try:
                        if pth.is_file():
                            pdf.image(str(pth), w=80)
                            pdf.ln(2)
                        else:
                            pdf.cell(0, 7, f"  [Heatmap not found: {path}]", new_x=nx, new_y=ny)
                    except (OSError, RuntimeError, ValueError):
                        pdf.cell(0, 7, f"  [Heatmap not available: {path}]", new_x=nx, new_y=ny)
            pdf.ln(3)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "6. Technical Details", new_x=nx, new_y=ny)
        pdf.set_font("Helvetica", size=11)
        tech = result.get("technical", {})
        if isinstance(tech, dict):
            for key, val in tech.items():
                pdf.cell(0, 7, f"  {key}: {val}", new_x=nx, new_y=ny)

        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 8)
        eng = str(result.get("engine_version", ""))
        inh = str(result.get("input_sha256", ""))
        trunc = f"{inh[:16]}..." if len(inh) > 16 else inh
        pdf.cell(
            0,
            5,
            f"Engine {eng} | input_sha256: {trunc}",
            new_x=nx,
            new_y=ny,
            align="C",
        )

        pdf.output(pdf_path)
