#!/usr/bin/env python3
"""Permanently remove columns from the source dataset on the Hugging Face Hub.

Removes ``id``, ``source`` and ``jw_code`` from every config of
``AfriSpeech/african-speech-public_v1`` while KEEPING ``audio``, ``text``,
``duration`` and ``iso639_3``.

It rewrites each parquet shard surgically with pyarrow (audio bytes are copied,
never re-decoded), fixes the parquet's embedded HF feature metadata, and finally
updates the dataset card's ``dataset_info`` so the declared schema matches.

Safety / practicality:
  * The previous version stays in the repo's git history — roll back by loading
    an earlier ``revision`` if needed.
  * Shards are processed one config at a time and deleted locally after upload,
    so disk stays bounded.
  * Resumable: completed configs are recorded in a state file and skipped on
    re-run (use --force to redo).
  * Dry-run by default; pass --execute to actually upload.

Usage:
  python scripts/clean_source_dataset.py                 # dry run, all configs
  python scripts/clean_source_dataset.py --only luvale_lue --execute
  python scripts/clean_source_dataset.py --execute       # full rewrite
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download

REPO = "AfriSpeech/african-speech-public_v1"
DROP = ["id", "source", "jw_code"]
KEEP_HINT = ["audio", "text", "duration", "iso639_3"]
HF_META_KEY = b"huggingface"
_HERE = Path(__file__).resolve().parent
STATE_FILE = _HERE / ".clean_source_state.json"
# Pristine copy of the dataset card, captured before any cleaning. The card is
# always regenerated from this so it reflects exactly the set of cleaned configs.
README_SNAPSHOT = _HERE / ".readme_original.md"


def _load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()).get("done", []))
    return set()


def _save_state(done: set[str]) -> None:
    STATE_FILE.write_text(json.dumps({"done": sorted(done)}, indent=2))


def _fix_parquet_metadata(schema):
    """Return new schema metadata with DROP columns removed from HF features."""
    md = dict(schema.metadata or {})
    if HF_META_KEY in md:
        hf = json.loads(md[HF_META_KEY])
        feats = hf.get("info", {}).get("features") or hf.get("features")
        if isinstance(feats, dict):
            for d in DROP:
                feats.pop(d, None)
        md[HF_META_KEY] = json.dumps(hf).encode()
    return md


def _rewrite_shard(local_in: str, local_out: str) -> tuple[bool, list[str]]:
    """Drop DROP columns from one parquet file. Returns (changed, dropped)."""
    pf = pq.ParquetFile(local_in)
    table = pf.read()
    present = [c for c in DROP if c in table.column_names]
    if not present:
        return False, []
    table = table.drop_columns(present)
    table = table.replace_schema_metadata(_fix_parquet_metadata(table.schema))
    # Preserve the original compression codec.
    codec = pf.metadata.row_group(0).column(0).compression.lower()
    if codec == "uncompressed":
        codec = "none"
    pq.write_table(table, local_out, compression=codec)
    return True, present


def _config_of(path: str) -> str:
    return path.split("/", 1)[0]


def clean_dataset(execute: bool, only: list[str] | None, force: bool) -> int:
    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    by_config: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_config[_config_of(f)].append(f)

    configs = sorted(by_config)
    if only:
        configs = [c for c in configs if c in only]
        if not configs:
            print(f"No matching configs for {only}", file=sys.stderr)
            return 1

    done = _load_state()
    print(f"{'EXECUTE' if execute else 'DRY-RUN'}: {len(configs)} config(s); "
          f"dropping {DROP}; keeping {KEEP_HINT}\n")

    for ci, cfg in enumerate(configs, 1):
        if cfg in done and not force:
            print(f"[{ci}/{len(configs)}] {cfg}: already done, skipping")
            continue
        shards = sorted(by_config[cfg])
        print(f"[{ci}/{len(configs)}] {cfg}: {len(shards)} shard(s)")
        ops = []
        tmpdir = Path(tempfile.mkdtemp(prefix=f"clean_{cfg}_"))
        try:
            for shard in shards:
                local_in = hf_hub_download(REPO, shard, repo_type="dataset")
                out = tmpdir / Path(shard).name
                changed, dropped = _rewrite_shard(local_in, str(out))
                if not changed:
                    print(f"    {shard}: already clean")
                    continue
                print(f"    {shard}: dropped {dropped}")
                if execute:
                    from huggingface_hub import CommitOperationAdd
                    ops.append(CommitOperationAdd(path_in_repo=shard, path_or_fileobj=str(out)))
            if execute and ops:
                api.create_commit(
                    REPO, repo_type="dataset", operations=ops,
                    commit_message=f"Drop {', '.join(DROP)} from {cfg}",
                )
                print(f"    committed {len(ops)} shard(s)")
            done.add(cfg)
            if execute and ops:
                _save_state(done)
                _update_readme(api, done)
        finally:
            for p in tmpdir.glob("*"):
                p.unlink(missing_ok=True)
            tmpdir.rmdir()

    if not execute:
        print("\nDry run complete. Re-run with --execute to apply.")
    return 0


def _update_readme(api: HfApi, done: set[str]) -> None:
    """Regenerate the dataset card from the snapshot, stripping DROP columns
    from exactly the configs in ``done`` (so the declared schema always matches
    the data on the Hub)."""
    if not README_SNAPSHOT.exists():
        print("  (no README snapshot; skipping card update)")
        return
    text = README_SNAPSHOT.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm)
    infos = meta.get("dataset_info")
    if infos is None:
        return
    infos = infos if isinstance(infos, list) else [infos]
    for info in infos:
        if info.get("config_name") in done:
            feats = info.get("features")
            if isinstance(feats, list):
                info["features"] = [f for f in feats if f.get("name") not in DROP]
    new_fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False)
    new_text = f"---\n{new_fm}---{body}"
    api.upload_file(
        path_or_fileobj=new_text.encode("utf-8"),
        path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
        commit_message=f"Sync dataset card: {len(done)} config(s) cleaned",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--only", nargs="*", help="limit to these config names")
    ap.add_argument("--force", action="store_true", help="redo configs already in state file")
    args = ap.parse_args()
    return clean_dataset(execute=args.execute, only=args.only, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
