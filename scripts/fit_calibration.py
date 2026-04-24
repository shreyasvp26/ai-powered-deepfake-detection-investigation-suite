#!/usr/bin/env python3
"""Temperature-scaling calibration for the DSAN v3.1 classifier (S-10c).

Inputs:
  * ``--ckpt``    — v3.1 checkpoint (best, SWA, EMA, or winner).
  * ``--config``  — ``configs/train_config_max.yaml`` (or an override).
  * ``--eval-split`` — ``val`` (standard) or ``test`` (audit).

Writes:
  * ``--out-json`` — JSON with ``{"temperature": float, "ece_before": float, "ece_after": float}``.
  * ``--reliability-plot`` — optional PNG with a reliability diagram.

Method:
  * Fit a single scalar ``T`` that minimises NLL on the held-out split (L-BFGS).
  * Report ECE (expected calibration error, 15 bins) before and after scaling.

Reference: Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.attribution.attribution_model_v31 import DSANv31
from src.attribution.dataset_v31 import DSANv31Dataset
from src.utils import load_config
from training.train_attribution_v31 import METHODS, build_model, load_labeled_videos


@torch.no_grad()
def _collect_logits(
    model: DSANv31, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for batch in loader:
        rgb, srm, _, y, cls_mask, _ = [t.to(device) for t in batch]
        keep = cls_mask > 0
        if not keep.any():
            continue
        logits, _, _ = model(rgb[keep], srm[keep])
        logits_all.append(logits.detach().cpu())
        labels_all.append(y[keep].detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accs = (preds == labels).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confs)
    out = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        if mask.any():
            bin_acc = float(accs[mask].mean())
            bin_conf = float(confs[mask].mean())
            out += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(out)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def _maybe_plot(
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    labels: np.ndarray,
    out_png: Path,
    n_bins: int = 15,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"matplotlib unavailable; skipping {out_png}")
        return

    def _reliability(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        accs = (preds == labels).astype(np.float32)
        edges = np.linspace(0, 1, n_bins + 1)
        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (confs > lo) & (confs <= hi)
            if m.any():
                bin_acc[i] = accs[m].mean()
                bin_conf[i] = confs[m].mean()
        return bin_conf, bin_acc

    fig, ax = plt.subplots(figsize=(5, 5))
    cb, ab = _reliability(probs_before)
    ca, aa = _reliability(probs_after)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(cb, ab, "o-", label="before")
    ax.plot(ca, aa, "s-", label="after (T-scaled)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (DSAN v3.1)")
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit temperature scaling for DSAN v3.1.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train_config_max.yaml"))
    ap.add_argument("--eval-split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--reliability-plot", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    dev = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a = cfg["attribution"]
    data = a["data"]
    crop_dir = (project_root / str(data.get("crop_dir", "data/processed/faces"))).resolve()
    split_file = project_root / str(data[f"{args.eval_split}_split"])
    methods = list(data.get("methods", list(METHODS)))
    fpv = int(data.get("frames_per_video", 30))
    image_size = int(data.get("image_size", 380))
    ids, ys = load_labeled_videos(split_file, crop_dir, methods)
    ds = DSANv31Dataset(
        ids, ys, str(crop_dir),
        augment=False, frames_per_video=fpv, methods=methods,
        image_size=image_size,
        mask_out_size=int(a["model"].get("mask_head", {}).get("out_size", 64)),
        sbi_ratio=0.0,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(cfg, dev, pretrained=False)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd.get("model", sd), strict=False)

    logits, labels = _collect_logits(model, loader, dev)

    probs_before = F.softmax(logits, dim=1).numpy()
    ece_before = _ece(probs_before, labels.numpy())

    T = _fit_temperature(logits, labels)
    probs_after = F.softmax(logits / T, dim=1).numpy()
    ece_after = _ece(probs_after, labels.numpy())

    result = {
        "temperature": float(T),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "eval_split": args.eval_split,
        "n_samples": int(len(labels)),
        "ckpt": str(args.ckpt),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))

    if args.reliability_plot is not None:
        _maybe_plot(probs_before, probs_after, labels.numpy(), args.reliability_plot)


if __name__ == "__main__":
    main()
