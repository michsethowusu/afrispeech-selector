"""Export a built dataset: a downloadable archive, or a push to the HF Hub."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def export_archive(dataset, out_dir: str | Path | None = None, name: str = "afrispeech_selection"):
    """Save the dataset to disk (load_from_disk format) and zip it.

    Returns the path to the ``.zip``. The archive round-trips with
    ``datasets.load_from_disk`` after unzipping, so audio stays decoded and
    training-ready.
    """
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = out_dir / name
    dataset.save_to_disk(str(ds_dir))
    zip_base = str(out_dir / name)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(ds_dir))
    return zip_path


def export_parquet(dataset, out_dir: str | Path | None = None, name: str = "afrispeech_selection"):
    """Write the dataset to a single Parquet file (audio embedded as bytes).

    Returns the path to the ``.parquet``. Good for a one-file download that
    loads with ``datasets.Dataset.from_parquet`` or pandas/pyarrow.
    """
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    dataset.to_parquet(str(path))
    return str(path)


def export_metadata_csv(dataset, out_dir: str | Path | None = None, name: str = "afrispeech_selection"):
    """Write a manifest CSV of everything except the audio bytes."""
    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="afrispeech_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_manifest.csv"
    cols = [c for c in dataset.column_names if c != "audio"]
    dataset.remove_columns([c for c in dataset.column_names if c not in cols]).to_csv(str(path))
    return str(path)


def push_to_hub(
    dataset,
    repo_id: str,
    *,
    token: str,
    private: bool = True,
    split: str = "train",
):
    """Push the built dataset to a user-owned HF dataset repo.

    The repo is created if it does not exist. ``split`` names the split the
    selection is uploaded as. Returns the repo URL.
    """
    if not token:
        raise ValueError("A Hugging Face token with write access is required to push.")
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must look like 'username/dataset-name'.")

    from datasets import DatasetDict

    payload = dataset if hasattr(dataset, "keys") else DatasetDict({split: dataset})
    payload.push_to_hub(repo_id, token=token, private=private)
    return f"https://huggingface.co/datasets/{repo_id}"
