"""
Validate train/val/test manifest splits and optional feature file presence.

Usage (from repo root):
  python scripts/validate_manifests.py
  python scripts/validate_manifests.py --check-features --feature-samples 100
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _id_series(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for key in ("filename", "id", "sample_id"):
        if key in df.columns:
            return df[key].astype(str)
    raise ValueError("Manifest needs a filename, id, or sample_id column.")


def validate(
    train_path: str,
    val_path: str,
    test_path: str,
    feature_dir: str | None,
    feature_samples: int,
    check_features: bool,
) -> int:
    paths = {"train": train_path, "val": val_path, "test": test_path}
    frames = {}
    ids = {}
    for name, p in paths.items():
        if not os.path.isfile(p):
            print(f"ERROR: missing file: {p}", file=sys.stderr)
            return 1
        frames[name] = pd.read_csv(p)
        ids[name] = set(_id_series(frames[name]))

    print("Row counts:")
    for name in paths:
        print(f"  {name:5s}: {len(frames[name])}")

    overlaps = []
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            inter = ids[a] & ids[b]
            if inter:
                overlaps.append((a, b, len(inter), list(sorted(inter))[:5]))

    if overlaps:
        print("\nERROR: ID overlaps between splits (leakage risk):", file=sys.stderr)
        for a, b, n, sample in overlaps:
            print(f"  {a} ∩ {b}: {n} ids (e.g. {sample})", file=sys.stderr)
        return 2

    print("\nOK: no overlapping IDs between train / val / test.")

    if not check_features or not feature_dir:
        return 0

    text_d = os.path.join(feature_dir, "text")
    audio_d = os.path.join(feature_dir, "audio")
    video_d = os.path.join(feature_dir, "video")
    for d, label in ((text_d, "text"), (audio_d, "audio"), (video_d, "video")):
        if not os.path.isdir(d):
            print(f"ERROR: missing feature subdir: {d}", file=sys.stderr)
            return 3

    missing_total = 0
    for split_name, p in paths.items():
        df = frames[split_name]
        n = min(feature_samples, len(df))
        sid_list = _id_series(df).head(n).tolist()
        for sid in sid_list:
            for sub in ("text", "audio", "video"):
                fp = os.path.join(feature_dir, sub, f"{sid}.pt")
                if not os.path.isfile(fp):
                    print(f"ERROR: missing {fp} (split={split_name})", file=sys.stderr)
                    missing_total += 1
                    if missing_total >= 20:
                        print("... (stopping after 20 missing files)", file=sys.stderr)
                        return 4
    if missing_total:
        return 4

    n_tr = min(feature_samples, len(frames["train"]))
    n_va = min(feature_samples, len(frames["val"]))
    n_te = min(feature_samples, len(frames["test"]))
    print(f"OK: spot-checked features for train (first {n_tr}), val (first {n_va}), test (first {n_te}) — all .pt files present.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Validate manifest splits and optional features.")
    p.add_argument("--train", default="data/manifest_train.csv")
    p.add_argument("--val", default="data/manifest_val.csv")
    p.add_argument("--test", default="data/manifest_test.csv")
    p.add_argument("--feature-dir", default="data/features")
    p.add_argument("--check-features", action="store_true", help="Verify .pt files exist for sample IDs")
    p.add_argument("--feature-samples", type=int, default=100, help="Rows per split to check (from top)")
    args = p.parse_args()
    code = validate(
        args.train,
        args.val,
        args.test,
        args.feature_dir,
        args.feature_samples,
        args.check_features,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
