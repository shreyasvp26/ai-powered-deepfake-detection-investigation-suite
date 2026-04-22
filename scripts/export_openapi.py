#!/usr/bin/env python3
"""
Regenerate ``api/openapi.json`` from the live FastAPI app (V2A-09).

Run from the repository root (or anywhere with ``PYTHONPATH=`` pointing at the repo)::

  DATABASE_URL=sqlite:///:memory: SYNC_RQ=1 MOCK_ENGINE=1 python scripts/export_openapi.py

CI fails if the committed file does not match this output (``sort_keys`` for stable diffs).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("SYNC_RQ", "1")
    os.environ.setdefault("MOCK_ENGINE", "1")
    os.environ.setdefault("GIT_SHA", "export-openapi")
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

    from api.main import create_app

    app = create_app()
    spec = app.openapi()
    out = _REPO / "api" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} ({len(text)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
