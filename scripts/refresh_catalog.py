#!/usr/bin/env python3
"""Regenerate data/catalog.tsv from the live dataset on the Hub.

Lists every config (subset) of the dataset and, for each, reads split sizes and
sums clip durations to recompute hours. This keeps the static catalog in sync
when the source dataset changes.

Usage:
    python scripts/refresh_catalog.py [--dataset DATASET_ID] [--token HF_TOKEN]

Note: this streams each split to total durations, so it touches the whole
dataset and can take a while. The committed catalog.tsv was produced from the
maintainer-published table; run this only when the dataset is updated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afrispeech_selector.catalog import COUNTRY_NAMES, DATASET_ID, load_catalog  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--token", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "catalog.tsv"))
    args = ap.parse_args()

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    # Preserve language/iso/country metadata from the existing catalog where we
    # can (the Hub configs don't always carry a clean language label).
    known = {e.subset: e for e in load_catalog()}

    configs = get_dataset_config_names(args.dataset, token=args.token)
    print(f"{len(configs)} configs found in {args.dataset}", file=sys.stderr)

    rows = []
    for cfg in sorted(configs):
        meta = known.get(cfg)
        splits = get_dataset_split_names(args.dataset, cfg, token=args.token)
        sizes = {"train": 0, "val": 0, "test": 0}
        seconds = 0.0
        clips = 0
        for sp in splits:
            ds = load_dataset(args.dataset, cfg, split=sp, token=args.token, streaming=True)
            n = 0
            for ex in ds:
                n += 1
                a = ex.get("audio")
                if isinstance(a, dict) and a.get("array") is not None and a.get("sampling_rate"):
                    seconds += len(a["array"]) / a["sampling_rate"]
                elif ex.get("duration"):
                    seconds += float(ex["duration"])
            key = "val" if sp in ("val", "validation", "dev") else sp
            sizes[key] = sizes.get(key, 0) + n
            clips += n
        rows.append({
            "subset": cfg,
            "language": meta.language if meta else cfg,
            "iso": meta.iso if meta else "",
            "country": meta.country if meta else "",
            "clips": clips,
            "hours": round(seconds / 3600, 2),
            "train": sizes["train"], "val": sizes["val"], "test": sizes["test"],
        })
        print(f"  {cfg}: {clips} clips, {rows[-1]['hours']}h", file=sys.stderr)

    fields = ["subset", "language", "iso", "country", "clips", "hours", "train", "val", "test"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
