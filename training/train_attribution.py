#!/usr/bin/env python3
"""DSAN v3 training entrypoint (plan §10.11). Use ``--dry-run`` on CPU without data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from src.attribution.attribution_model import DSANv3
from src.attribution.losses import DSANLoss
from src.utils import load_config


def _dry_run(cfg: dict[str, Any], device: torch.device, *, pretrained: bool) -> None:
    m = cfg["attribution"]["model"]
    model = DSANv3(
        num_classes=int(m["num_classes"]),
        fused_dim=int(m["fused_dim"]),
        pretrained=pretrained,
    ).to(device)
    model.train()
    lc = cfg["attribution"]["loss"]
    criterion = DSANLoss(
        alpha=float(lc["alpha"]),
        beta=float(lc["beta"]),
        temperature=float(lc["temperature"]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(0)
    b = 8
    rgb = torch.randn(b, 3, 224, 224, device=device)
    srm = torch.clamp(torch.randn(b, 3, 224, 224, device=device), -1.0, 1.0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=device)

    logits, emb = model(rgb, srm)
    loss, l_ce, l_con = criterion(logits, emb, labels)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    assert logits.shape == (b, int(m["num_classes"]))
    assert emb.shape == (b, int(m["fused_dim"]))
    assert torch.isfinite(loss).item()
    print(
        "dry-run ok:",
        {
            "loss": float(loss.detach()),
            "l_ce": float(l_ce.detach()),
            "l_con": float(l_con.detach()),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train DSAN v3 (use --dry-run locally without GPU data).")
    p.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    p.add_argument("--dry-run", action="store_true", help="One forward+backward on random tensors (CPU/GPU).")
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="With --dry-run: load ImageNet weights (needs network). Default dry-run uses random init.",
    )
    p.add_argument("--device", type=str, default=None, help="cpu | cuda (default: cuda if available else cpu).")
    args = p.parse_args()

    cfg = load_config(args.config)
    if "attribution" not in cfg:
        print("Config must contain top-level 'attribution' key (V8-01).", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        _dry_run(cfg, device, pretrained=bool(args.pretrained))
        return

    print(
        "Full training loop is intended for the GPU host with FF++ crops and W&B. "
        "Locally, use:  python training/train_attribution.py --dry-run",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
