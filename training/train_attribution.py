#!/usr/bin/env python3
"""DSAN v3 training (plan §10.11): full loop with W&B, AMP, early stopping, checkpointing.

* ``--dry-run``: one forward+backward on random tensors (no dataloader, no W&B).
* ``--smoke-train``: one epoch, two train batches, CPU/tiny data (CI; no W&B I/O by default).
* Full training: requires crop tree + split JSON; intended for the GPU host (L4), but **runs on CPU**
  (slow) if data paths resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from src.attribution.attribution_model import DSANv3
from src.attribution.dataset import DSANDataset
from src.attribution.losses import DSANLoss, SupConLoss
from src.attribution.samplers import StratifiedBatchSampler
from src.utils import load_config

# --- seeds (project policy) ---
SEED = 42

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _load_pairs(split_json: Path) -> list[tuple[str, str]]:
    with split_json.open(encoding="utf-8") as f:
        raw = json.load(f)
    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]).strip(), str(item[1]).strip()))
    if not pairs:
        raise ValueError(f"No [src, tgt] pairs in {split_json}")
    return pairs


def _stem_from_pair(src: str, tgt: str) -> str:
    return f"{src.strip()}_{tgt.strip()}"


def load_labeled_videos(
    split_json: Path, crop_dir: Path, methods: list[str] | tuple[str, ...]
) -> tuple[list[str], list[int]]:
    """Return (video_id, label) for **fake** 4-class samples found on disk (nested tree)."""
    pairs = _load_pairs(split_json)
    video_ids: list[str] = []
    labels: list[int] = []
    for method in methods:
        mi = methods.index(method)
        for src, tgt in pairs:
            stem = _stem_from_pair(src, tgt)
            rel = f"{method}/{stem}"
            p = crop_dir / method / stem
            if not p.is_dir():
                continue
            if not list(p.glob("frame_*.png")):
                continue
            video_ids.append(rel)
            labels.append(mi)
    return video_ids, labels


def build_dataloaders(
    cfg: dict[str, Any],
    project_root: Path,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Train DataLoader (Stratified batch sampler); val with regular batching."""
    a = cfg["attribution"]
    data = a["data"]
    tr = a["training"]
    crop_dir = (project_root / str(data.get("crop_dir", "data/processed/faces"))).resolve()
    train_split = project_root / str(data["train_split"])
    val_split = project_root / str(data["val_split"])
    methods = list(data.get("methods", list(METHODS)))
    fpv = int(data.get("frames_per_video", 30))

    tr_ids, tr_y = load_labeled_videos(train_split, crop_dir, methods)
    va_ids, va_y = load_labeled_videos(val_split, crop_dir, methods)
    if not tr_ids or not va_ids:
        raise FileNotFoundError(
            f"No training or val samples under {crop_dir} for the given splits. "
            "Add crops or use --smoke-train (synthetic) / --dry-run."
        )

    tr_ds = DSANDataset(
        tr_ids, tr_y, str(crop_dir), augment=True, frames_per_video=fpv, methods=methods
    )
    va_ds = DSANDataset(
        va_ids, va_y, str(crop_dir), augment=False, frames_per_video=fpv, methods=methods
    )
    tcfg = a["training"]
    batch_size = int(tcfg["batch_size"])
    num_workers = int(tcfg.get("num_workers", 0))
    pin = bool(tcfg.get("pin_memory", False)) and device.type == "cuda"
    sampler_mode = str(tcfg.get("sampler", "stratified_batch"))
    if sampler_mode == "stratified_batch":
        ytrain = np.asarray(tr_ds.labels, dtype=np.int64)
        bsz = int(batch_size)
        batch_sampler = StratifiedBatchSampler(ytrain, batch_size=bsz, min_per_class=2)
        t_loader = DataLoader(
            tr_ds,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            pin_memory=pin,
            **({"prefetch_factor": int(tcfg.get("prefetch_factor", 2)), "persistent_workers": num_workers > 0} if num_workers > 0 else {}),
        )
    else:
        t_loader = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            **({"prefetch_factor": int(tcfg.get("prefetch_factor", 2)), "persistent_workers": num_workers > 0} if num_workers > 0 else {}),
        )
    v_loader = DataLoader(
        va_ds,
        batch_size=min(batch_size, 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        **({"prefetch_factor": int(tcfg.get("prefetch_factor", 2)), "persistent_workers": num_workers > 0} if num_workers > 0 else {}),
    )
    return t_loader, v_loader


def build_model(
    cfg: dict[str, Any], device: torch.device, *, pretrained: bool | None = None
) -> DSANv3:
    m = cfg["attribution"]["model"]
    p = bool(m.get("pretrained", True)) if pretrained is None else bool(pretrained)
    model = DSANv3(
        num_classes=int(m["num_classes"]),
        fused_dim=int(m["fused_dim"]),
        pretrained=p,
    )
    return model.to(device)


def build_optim(
    cfg: dict[str, Any], model: nn.Module
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None, list[float]]:
    """AdamW (backbone vs head) + cosine after warmup (scheduler stepped per epoch post-warmup)."""
    ocfg = cfg["attribution"]["optimizer"]
    scfg = cfg["attribution"]["scheduler"]
    tcfg = cfg["attribution"]["training"]
    epochs = int(tcfg["epochs"])
    wu = int(scfg.get("warmup_epochs", 0))
    min_lr = float(scfg.get("min_lr", 1e-7))
    wd = float(ocfg.get("weight_decay", 0.0))
    b_lr = float(ocfg.get("backbone_lr", 1e-5))
    h_lr = float(ocfg.get("head_lr", 3e-4))

    backbone: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "classifier" in n:
                head.append(p)
            else:
                backbone.append(p)
    param_groups: list[dict[str, Any]] = [
        {"params": backbone, "lr": b_lr, "name": "backbone"},
        {"params": head, "lr": h_lr, "name": "head"},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=wd)
    t_cos = max(1, epochs - wu)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t_cos, eta_min=min_lr, last_epoch=-1
    )
    base_lrs = [b_lr, h_lr]
    return opt, sched, base_lrs


def build_losses(
    cfg: dict[str, Any],
) -> tuple[nn.CrossEntropyLoss, SupConLoss, float, float]:
    """CE + SupCon with alpha/beta weighting in the training loop (same weighting as ``DSANLoss``)."""
    lc = cfg["attribution"]["loss"]
    ce = nn.CrossEntropyLoss()
    sc = SupConLoss(temperature=float(lc["temperature"]))
    return ce, sc, float(lc["alpha"]), float(lc["beta"])


def _autocast_device(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    return "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    supcon_loss: SupConLoss,
    alpha: float,
    beta: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool,
    grad_accum: int,
    epoch: int,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    ac_dev = _autocast_device(device)

    run_loss = 0.0
    run_n = 0.0
    step = -1
    for step, (rgb, srm, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        rgb = rgb.to(device, non_blocking=True)
        srm = srm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(ac_dev, enabled=use_amp):
            logits, emb = model(rgb, srm)
            l_ce = ce_loss(logits, y)
            l_con = supcon_loss(emb, y)
            loss = alpha * l_ce + beta * l_con

        run_loss += float(loss.detach().cpu().item())
        run_n += 1.0

        if use_amp and scaler is not None:
            scaler.scale(loss / float(grad_accum)).backward()
        else:
            (loss / float(grad_accum)).backward()
        if (step + 1) % int(grad_accum) == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if os.environ.get("WANDB_MODE", "online") != "disabled" and (step + 1) % int(grad_accum) == 0:
                try:  # noqa: S110
                    import wandb  # type: ignore[import-not-found]

                    wandb.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/l_ce": float(l_ce.item()),
                            "train/l_con": float(l_con.item()),
                            "step": step,
                            "epoch": epoch,
                        }
                    )
                except Exception:
                    pass
    if step >= 0 and (step + 1) % int(grad_accum) != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    out = {
        "loss": run_loss / max(run_n, 1.0),
    }
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | dict[int, float]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for rgb, srm, y in loader:
        rgb = rgb.to(device)
        srm = srm.to(device)
        logits, _ = model(rgb, srm)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(int(x) for x in y.cpu().numpy().tolist())
        y_pred.extend(int(x) for x in pred.tolist())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    oa = float(accuracy_score(y_true, y_pred))
    rec = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2, 3]
    )
    per_class = {i: float(rec[i]) for i in range(len(rec))}
    return {
        "macro_f1": macro_f1,
        "accuracy": oa,
        "per_class_recall": per_class,
    }


def _dry_run(
    cfg: dict[str, Any], device: torch.device, *, pretrained: bool, seed: int
) -> None:
    set_seed(seed)
    m = cfg["attribution"]["model"]
    model = DSANv3(
        num_classes=int(m["num_classes"]),
        fused_dim=int(m["fused_dim"]),
        pretrained=pretrained,
    ).to(device)
    model.train()
    lc = cfg["attribution"]["loss"]
    crit = DSANLoss(
        alpha=float(lc["alpha"]),
        beta=float(lc["beta"]),
        temperature=float(lc["temperature"]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accum = max(1, int(cfg["attribution"]["training"].get("gradient_accumulation_steps", 1)))
    b = 8
    torch.manual_seed(seed)
    rgb = torch.randn(b, 3, 224, 224, device=device)
    srm = torch.clamp(torch.randn(b, 3, 224, 224, device=device), -1.0, 1.0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=device)
    opt.zero_grad(set_to_none=True)
    logits, emb = model(rgb, srm)
    loss, l_ce, l_con = crit(logits, emb, labels)
    (loss / float(accum)).backward()
    opt.step()
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


def _write_smoke_crops(target: Path) -> None:
    """8 train ``.jpg`` + 4 val ``v####.jpg`` in one dir (``flat_jpg`` layout)."""
    from PIL import Image

    target.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        im = Image.new("RGB", (32, 32), (i * 5 % 255, 50, 100))
        im.save(target / f"{i:04d}.jpg", format="JPEG")
    for i in range(4):
        im = Image.new("RGB", (32, 32), (10 + i, 60, 120))
        im.save(target / f"v{i:04d}.jpg", format="JPEG")


def _smoke_dataloaders(
    crop: Path, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    """8 train, 4 val; batch 4 => 2 train batches; 1 val batch. CPU-safe."""
    train_ids = [f"{i:04d}" for i in range(8)]
    train_y = [0, 0, 1, 1, 2, 2, 3, 3]
    tr_ds = DSANDataset(
        train_ids, train_y, str(crop), augment=True, frames_per_video=1, crop_layout="flat_jpg"
    )
    ytrain = np.asarray(tr_ds.labels, dtype=np.int64)
    t_loader = DataLoader(
        tr_ds,
        num_workers=0,
        batch_sampler=StratifiedBatchSampler(ytrain, batch_size=4, min_per_class=1),
    )
    v_ids = [f"v{i:04d}" for i in range(4)]
    v_y = [0, 1, 2, 3]
    va_ds = DSANDataset(
        v_ids, v_y, str(crop), augment=False, frames_per_video=1, crop_layout="flat_jpg"
    )
    v_loader = DataLoader(va_ds, batch_size=4, shuffle=False, num_workers=0)
    return t_loader, v_loader


def run_smoke_train(device: torch.device) -> None:
    """1 epoch, 2 train mini-batches, CPU, synthetic crops; W&B off."""
    os.environ["WANDB_MODE"] = "disabled"
    set_seed(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        crop = Path(tmp) / "crops"
        _write_smoke_crops(crop)
        t_loader, v_loader = _smoke_dataloaders(crop, device)
        cfg: dict[str, Any] = {
            "attribution": {
                "model": {
                    "num_classes": 4,
                    "fused_dim": 512,
                    "pretrained": False,
                },
                "loss": {"alpha": 1.0, "beta": 0.2, "temperature": 0.15},
                "training": {
                    "epochs": 1,
                    "gradient_accumulation_steps": 1,
                    "mixed_precision": False,
                },
                "optimizer": {
                    "backbone_lr": 1e-4,
                    "head_lr": 1e-3,
                    "weight_decay": 0.0,
                },
                "scheduler": {
                    "warmup_epochs": 0,
                    "min_lr": 1e-7,
                },
            }
        }
        model = build_model(cfg, device, pretrained=False)
        opt, sched, base_lrs = build_optim(cfg, model)
        for pg, b0 in zip(opt.param_groups, base_lrs):
            pg["lr"] = b0
        ce, sc, al, be = build_losses(cfg)
        stats = train_one_epoch(
            model,
            t_loader,
            ce,
            sc,
            al,
            be,
            opt,
            device,
            False,
            1,
            0,
            max_batches=2,
        )
        if not np.isfinite(stats["loss"]):
            raise RuntimeError("non-finite loss in smoke")
        ev = evaluate(model, v_loader, device)
        if not np.isfinite(float(ev["macro_f1"])):
            raise RuntimeError("non-finite macro_f1 in smoke")
        out_dir = Path(tmp) / "ckpt"
        out_dir.mkdir()
        ck = {"model": model.state_dict(), "config": {"smoke": True}}
        torch.save(ck, out_dir / "attribution_dsan_v3_epoch1.pt")
        torch.save(ck, out_dir / "attribution_dsan_v3_best.pt")
    print("smoke-train ok:", {"train_loss": stats["loss"], "val_macro_f1": ev["macro_f1"]})


def run_training(
    cfg: dict[str, Any],
    project_root: Path,
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> None:
    set_seed(seed)
    tr_loader, val_loader = build_dataloaders(cfg, project_root, device)
    tcfg = cfg["attribution"]["training"]
    epochs = int(tcfg["epochs"])
    wu = int(cfg["attribution"]["scheduler"].get("warmup_epochs", 0))
    mixed = bool(tcfg.get("mixed_precision", False)) and device.type == "cuda"
    gaccum = int(tcfg.get("gradient_accumulation_steps", 1))
    es = tcfg.get("early_stopping", {}) or {}
    es_on = bool(es.get("enabled", False))
    patience = int(es.get("patience", 7))
    best_metric = float("-inf")
    wait = 0
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg, device)
    opt, sched, base_lrs = build_optim(cfg, model)
    ce, sc, al, be = build_losses(cfg)

    if os.environ.get("WANDB_MODE", "online") != "disabled":
        try:  # noqa: S110
            import wandb  # type: ignore[import-not-found]

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "deepfake-v1-fix"),
                config=cfg,
            )
        except Exception:
            pass

    for epoch in range(epochs):
        if wu > 0 and epoch < wu:
            wf = (epoch + 1) / wu
            for pg, b0 in zip(opt.param_groups, base_lrs):
                pg["lr"] = b0 * wf
        tr_stats = train_one_epoch(
            model,
            tr_loader,
            ce,
            sc,
            al,
            be,
            opt,
            device,
            mixed,
            gaccum,
            epoch,
        )
        if epoch >= wu and sched is not None:
            sched.step()
        ev = evaluate(model, val_loader, device)
        m_f1 = float(ev["macro_f1"])
        if os.environ.get("WANDB_MODE", "online") != "disabled":
            try:  # noqa: S110
                import wandb  # type: ignore[import-not-found]

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": tr_stats["loss"],
                        "val/macro_f1": m_f1,
                        "val/acc": ev["accuracy"],
                    }
                )
            except Exception:
                pass
        path_e = out_dir / f"attribution_dsan_v3_epoch{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "val_macro_f1": m_f1,
            },
            path_e,
        )
        if m_f1 > best_metric:
            best_metric = m_f1
            wait = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "val_macro_f1": m_f1,
                },
                out_dir / "attribution_dsan_v3_best.pt",
            )
        else:
            wait += 1
        if es_on and wait >= patience:
            print(f"Early stopping at epoch {epoch} (no val macro_f1 improve for {patience}).", file=sys.stderr)
            break
    print("training finished", {"best_val_macro_f1": best_metric})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train DSAN v3 (use --dry-run or --smoke-train on CPU; full run on L4 with crops)."
    )
    p.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="One forward+backward on random tensors (no dataset).",
    )
    p.add_argument(
        "--smoke-train",
        action="store_true",
        help="1 epoch, 2 train batches, synthetic jpg crops, CPU; no W&B I/O.",
    )
    p.add_argument("--pretrained", action="store_true", help="ImageNet backbones in dry-run (needs network).")
    p.add_argument(
        "--device", type=str, default=None, help="cpu | cuda (default: auto)."
    )
    p.add_argument("--output-dir", type=Path, default=Path("models"), help="Checkpoints and best.pt")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    if "attribution" not in cfg:
        print("Config must contain top-level 'attribution' key (V8-01).", file=sys.stderr)
        sys.exit(1)

    if args.smoke_train:
        # Spec: 1 epoch, 2 batches on CPU; ignore CUDA even if available.
        dev = torch.device("cpu")
    elif args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        _dry_run(cfg, dev, pretrained=bool(args.pretrained), seed=int(args.seed))
        return
    if args.smoke_train:
        run_smoke_train(dev)
        return

    try:
        run_training(cfg, project_root, dev, args.output_dir.resolve(), int(args.seed))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        print(
            "Use --dry-run, --smoke-train, or place FF++ crops and splits per configs/train_config.yaml.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()