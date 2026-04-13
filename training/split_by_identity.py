#!/usr/bin/env python3
"""Build identity-safe train/val/test JSON from official FF++ pair splits (UI2, V5-23, V8-06)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Set, Tuple


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/splits/train.json"),
        help="Official FF++ train split (list of [src, tgt] pairs).",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=Path("data/splits/test.json"),
        help="Official FF++ test split (for V8-06 cross-reference).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for train/val/test *_identity_safe.json files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_pairs = load_pairs(args.train_json)
    if not train_pairs:
        raise SystemExit(f"No pairs loaded from {args.train_json}")

    source_ids = sorted({p[0] for p in train_pairs})
    random.seed(args.seed)
    random.shuffle(source_ids)

    n = len(source_ids)
    n_tr = int(0.8 * n)
    n_va = int(0.9 * n)
    train_sources: Set[str] = set(source_ids[:n_tr])
    val_sources: Set[str] = set(source_ids[n_tr:n_va])
    test_sources: Set[str] = set(source_ids[n_va:])

    assert train_sources.isdisjoint(test_sources)
    assert train_sources.isdisjoint(val_sources)
    assert val_sources.isdisjoint(test_sources)

    real_train = sorted(train_sources)
    real_test = sorted(test_sources)
    assert set(real_train).isdisjoint(
        set(real_test)
    ), "Real video identity leak: train/test overlap"

    official_test_pairs = load_pairs(args.test_json)
    official_test_sources = {p[0] for p in official_test_pairs}
    train_official_overlap = train_sources & official_test_sources
    if train_official_overlap:
        raise ValueError(
            f"Identity leak detected: {len(train_official_overlap)} source IDs appear in both "
            f"your training split AND the official FF++ test split.\n"
            f"Overlapping IDs: {sorted(train_official_overlap)}\n"
            f"Remove these IDs from train_sources before proceeding."
        )
    print("Official FF++ test set cross-reference: PASSED — zero overlap with train_sources")

    val_official_overlap = val_sources & official_test_sources
    if val_official_overlap:
        print(
            f"NOTE: {len(val_official_overlap)} source IDs overlap between val_sources and "
            f"official FF++ test set. Document as known limitation in evaluation section."
        )

    def filter_pairs(sources: Set[str]) -> List[List[str]]:
        return [[a, b] for a, b in train_pairs if a in sources]

    def build_payload(sources: Set[str]) -> dict:
        return {
            "pairs": filter_pairs(sources),
            "real_source_ids": sorted(sources),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, src_set in (
        ("train_identity_safe.json", train_sources),
        ("val_identity_safe.json", val_sources),
        ("test_identity_safe.json", test_sources),
    ):
        payload = build_payload(src_set)
        out_path = args.out_dir / name
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"Wrote {out_path} — fake pairs: {len(payload['pairs'])}, "
            f"reals: {len(payload['real_source_ids'])}"
        )

    print(f"Fake train pairs: {len(filter_pairs(train_sources))}")
    print(f"Real train IDs:   {len(real_train)}")


if __name__ == "__main__":
    main()
