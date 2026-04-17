# Environment verification — run on local and remote before development (PROJECT_PLAN_v10 §4.4)
from __future__ import annotations

import importlib
import sys


def _try(mod: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        if attr:
            getattr(m, attr)
    except Exception as e:
        return False, f"{mod}: {e}"
    return True, f"{mod}: OK"


def main() -> None:
    print(f"Python: {sys.version}\n")

    ok_all = True
    for mod, attr in (
        ("torch", None),
        ("cv2", None),
        ("yaml", None),
        ("timm", None),
        ("PIL.Image", None),
    ):
        ok, msg = _try(mod, attr)
        ok_all = ok_all and ok
        print(msg)

    try:
        import torch
    except ImportError:
        print("PyTorch details: (not importable)")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, "mps"):
            print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Face / UI (optional on minimal CI images)
    for label, mod, attr in (
        ("MediaPipe (optional for some scripts)", "mediapipe", None),
        ("facenet-pytorch / MTCNN", "facenet_pytorch", "MTCNN"),
        ("Streamlit (dashboard)", "streamlit", None),
    ):
        ok, msg = _try(mod, attr)
        if not ok:
            print(f"{label}: SKIP — {msg}")
        else:
            print(f"{label}: {msg}")

    if ok_all:
        print("\n--- Core dependencies verified ---")
    else:
        print("\n--- Core dependency check reported failures ---")
        sys.exit(1)


if __name__ == "__main__":
    main()
