"""Tests for TTS exporters and the training-schema adapter (offline, fake data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

datasets = pytest.importorskip("datasets")

from afrispeech_selector import apply_schema, export_tts


def _fake_ds(n=3, sr=16000):
    rows = {
        "audio": [{"array": np.zeros(sr, dtype=np.float32), "sampling_rate": sr} for _ in range(n)],
        "text": [f"sentence {i}" for i in range(n)],
        "language": ["Twi"] * n,
        "country": ["GH"] * n,
        "length": [1.0] * n,
        "iso": ["twi"] * n,
        "subset": ["twi_twi"] * n,
    }
    return datasets.Dataset.from_dict(rows).cast_column("audio", datasets.Audio(sampling_rate=sr))


def test_apply_schema_resample_and_rename():
    ds = apply_schema(_fake_ds(), target_sampling_rate=22050, schema="whisper")
    assert ds.column_names == ["audio", "sentence"]
    assert ds.features["audio"].sampling_rate == 22050
    asr = apply_schema(_fake_ds(), schema="asr")
    assert asr.column_names == ["audio", "text"]


def test_export_ljspeech(tmp_path):
    base = export_tts(_fake_ds(3), out_dir=str(tmp_path), name="d", fmt="ljspeech", sampling_rate=22050)
    base = Path(base)
    assert (base / "metadata.csv").exists()
    wavs = list((base / "wavs").glob("*.wav"))
    assert len(wavs) == 3
    line = (base / "metadata.csv").read_text().splitlines()[0]
    parts = line.split("|")
    assert len(parts) == 3 and parts[1] == parts[2]  # id|text|text


def test_export_piper_and_vits_and_melo(tmp_path):
    piper = Path(export_tts(_fake_ds(2), out_dir=str(tmp_path), name="p", fmt="piper"))
    assert (piper / "metadata.csv").read_text().splitlines()[0].split("|") == ["twi_twi_000000", "Twi", "sentence 0"]

    vits = Path(export_tts(_fake_ds(2), out_dir=str(tmp_path), name="v", fmt="vits"))
    assert (vits / "speakers.txt").exists()
    fl = (vits / "filelist.txt").read_text().splitlines()[0]
    assert fl.startswith("wavs/twi_twi_000000.wav|0|")

    melo = Path(export_tts(_fake_ds(2), out_dir=str(tmp_path), name="m", fmt="melo"))
    parts = (melo / "metadata.list").read_text().splitlines()[0].split("|")
    assert parts[0].startswith("wavs/") and parts[2] == "TWI"  # path|spk|LANG|text


def test_export_tts_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        export_tts(_fake_ds(1), out_dir=str(tmp_path), fmt="nope")
