#!/usr/bin/env python3
"""CUDA smoke test for DSAN v3.1 before a Kaggle run.

This script is intentionally small: it builds the DSAN model with the same
config override machinery as ``training/train_attribution_v31.py``, runs one
synthetic forward/backward pass, reports CUDA memory, and optionally checks
that two real batches can be read from a local FF++ crop subset.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _apply_smoke_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    attr = cfg.setdefault("attribution", {})
    model = attr.setdefault("model", {})
    training = attr.setdefault("training", {})
    data = attr.setdefault("data", {})
    sbi = attr.setdefault("sbi", {})

    model["rgb_backbone"] = "efficientnet_b4"
    model["freq_backbone"] = "resnet18"
    model["pretrained"] = False
    data["image_size"] = 224
    data["frames_per_video"] = 2
    training["batch_size"] = 1
    training["num_workers"] = 0
    training["pin_memory"] = False
    sbi["enabled"] = False
    return cfg


def _make_dummy_batch(
    batch_size: int, image_size: int, mask_size: int, num_classes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.randn(batch_size, 3, image_size, image_size, device=device)
    srm = torch.randn(batch_size, 3, image_size, image_size, device=device)
    mask_gt = torch.zeros(batch_size, 1, mask_size, mask_size, device=device)
    labels = torch.arange(batch_size, device=device, dtype=torch.long) % num_classes
    cls_mask = torch.ones(batch_size, device=device)
    mask_mask = torch.zeros(batch_size, device=device)
    return rgb, srm, mask_gt, labels, cls_mask, mask_mask


def _first_grad_norm(model: torch.nn.Module) -> tuple[str, float]:
    for name, param in model.named_parameters():
        if param.grad is not None:
            return name, float(param.grad.detach().norm().item())
    raise RuntimeError("No gradients found after backward pass")


def _candidate_video_dirs(data_root: Path, method: str) -> list[Path]:
    direct = data_root / method
    candidates: list[Path] = []
    if not direct.is_dir():
        return candidates
    for child in sorted(direct.iterdir()):
        if child.is_dir() and child.name in {"c23", "c40"}:
            candidates.extend([p for p in sorted(child.iterdir()) if p.is_dir()])
        elif child.is_dir():
            candidates.append(child)
    return candidates


def _check_dataloader(cfg: dict[str, Any], data_root: Path, batches: int) -> None:
    from src.attribution.dataset_v31 import DSANv31Dataset

    attr = cfg["attribution"]
    data_cfg = attr.get("data", {})
    model_cfg = attr.get("model", {})
    methods = list(data_cfg.get("methods", METHODS))
    image_size = int(data_cfg.get("image_size", 224))
    frames_per_video = int(data_cfg.get("frames_per_video", 2))
    mask_cfg = model_cfg.get("mask_head", {}) or {}
    mask_out_size = int(mask_cfg.get("out_size", 64))

    video_ids: list[str] = []
    labels: list[int] = []
    for label, method in enumerate(methods):
        for video_dir in _candidate_video_dirs(data_root, method):
            if list(video_dir.glob("frame_*.png")):
                video_ids.append(str(video_dir.relative_to(data_root)))
                labels.append(label)
                break

    if not video_ids:
        raise FileNotFoundError(
            f"No frame_*.png files found under {data_root}; expected "
            "<root>/<method>/<video>/frame_*.png or "
            "<root>/<method>/<compression>/<video>/frame_*.png"
        )

    dataset = DSANv31Dataset(
        video_ids=video_ids,
        labels=labels,
        crop_dir=str(data_root),
        masks_crop_dir=None,
        augment=False,
        frames_per_video=frames_per_video,
        methods=methods,
        image_size=image_size,
        mask_out_size=mask_out_size,
        sbi_ratio=0.0,
        seed=SEED,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    for idx, batch in enumerate(loader, start=1):
        rgb, srm, mask_gt, labels_t, _cls_mask, _mask_mask = batch
        print(
            "DataLoader batch "
            f"{idx}: rgb={tuple(rgb.shape)} srm={tuple(srm.shape)} "
            f"mask={tuple(mask_gt.shape)} labels={tuple(labels_t.shape)}"
        )
        if idx >= batches:
            return
    raise RuntimeError(f"DataLoader produced only {idx if 'idx' in locals() else 0} batches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local CUDA smoke test for DSAN v3.1.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config_max.yaml"))
    parser.add_argument("--override", action="append", default=[], help="key.path=value overrides")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument(
        "--data-root", type=Path, default=None, help="Optional FF++ crop subset root"
    )
    parser.add_argument("--dataloader-batches", type=int, default=2)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Load pretrained backbone weights; default is off to avoid network and reduce "
            "startup time."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from src.attribution.losses import DSANv31Loss
        from src.utils import load_config
        from training.train_attribution_v31 import _apply_overrides, build_model, set_seed

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")

        set_seed(SEED)
        cfg = load_config(args.config)
        if "attribution" not in cfg:
            raise ValueError("Config must contain top-level 'attribution' key")
        cfg = _apply_smoke_defaults(cfg)
        if args.pretrained:
            cfg["attribution"]["model"]["pretrained"] = True
        cfg = _apply_overrides(cfg, list(args.override or []))

        device = torch.device(args.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        print("Building DSAN smoke model")
        print(
            "Backbones: "
            f"rgb={cfg['attribution']['model'].get('rgb_backbone')} "
            f"freq={cfg['attribution']['model'].get('freq_backbone')}"
        )
        model = build_model(
            cfg,
            device,
            pretrained=bool(cfg["attribution"]["model"].get("pretrained", False)),
        )
        model.train()
        total, trainable = _count_parameters(model)
        print(f"Parameters: total={total:,} trainable={trainable:,}")

        data_cfg = cfg["attribution"].get("data", {})
        model_cfg = cfg["attribution"].get("model", {})
        loss_cfg = cfg["attribution"].get("loss", {})
        image_size = int(data_cfg.get("image_size", 224))
        num_classes = int(model_cfg.get("num_classes", 4))
        mask_cfg = model_cfg.get("mask_head", {}) or {}
        mask_size = int(mask_cfg.get("out_size", 64))
        batch_size = 1
        rgb, srm, mask_gt, labels, cls_mask, mask_mask = _make_dummy_batch(
            batch_size, image_size, mask_size, num_classes, device
        )

        criterion = DSANv31Loss(
            alpha=float(loss_cfg.get("alpha", 1.0)),
            beta=float(loss_cfg.get("beta", 0.2)),
            lambda_mask=float(mask_cfg.get("lambda_mask", 0.3)),
            temperature=float(loss_cfg.get("temperature", 0.15)),
            mask_pos_weight=float(loss_cfg.get("mask_pos_weight", 2.0)),
        ).to(device)

        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits, embeddings, mask_logits = model(rgb, srm)
            loss, ce, supcon, mask_bce = criterion(
                logits, embeddings, labels, mask_logits, mask_gt, cls_mask, mask_mask
            )
        torch.cuda.synchronize(device)
        print(f"Output shape: logits={tuple(logits.shape)} embedding={tuple(embeddings.shape)}")
        if mask_logits is not None:
            print(f"Mask output shape: {tuple(mask_logits.shape)}")
        print(
            "Loss: "
            f"total={float(loss.detach().item()):.6f} "
            f"ce={float(ce.detach().item()):.6f} "
            f"supcon={float(supcon.detach().item()):.6f} "
            f"mask_bce={float(mask_bce.detach().item()):.6f}"
        )
        print(
            "CUDA memory after forward: "
            f"allocated={torch.cuda.memory_allocated(device) / 1024**2:.1f} MiB "
            f"reserved={torch.cuda.memory_reserved(device) / 1024**2:.1f} MiB "
            f"peak_allocated={torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB"
        )

        loss.backward()
        grad_name, grad_norm = _first_grad_norm(model)
        print(f"Gradient check: {grad_name} grad_norm={grad_norm:.6e}")

        if args.data_root is None:
            print("WARNING: --data-root not provided; skipping real DataLoader check.")
        else:
            _check_dataloader(cfg, args.data_root.resolve(), int(args.dataloader_batches))

        print("SUMMARY: PASS")
        return 0
    except Exception as exc:
        print(f"SUMMARY: FAIL - {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
