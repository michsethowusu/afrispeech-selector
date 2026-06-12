"""Standardisation test using a fake in-memory subset (no network).

Verifies that build_subset maps the real source schema
(id, audio, text, duration, source, jw_code, iso639_3) onto the standard
schema and drops id/source/jw_code.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

datasets = pytest.importorskip("datasets")

from afrispeech_selector import builder
from afrispeech_selector.catalog import by_subset


def _fake_subset():
    sr = 16000
    rows = {
        "id": ["w_1", "w_2", "w_3"],
        "audio": [
            {"array": np.zeros(sr, dtype=np.float32), "sampling_rate": sr},        # 1.0s
            {"array": np.zeros(sr * 2, dtype=np.float32), "sampling_rate": sr},    # 2.0s
            {"array": np.zeros(sr // 2, dtype=np.float32), "sampling_rate": sr},   # 0.5s
        ],
        "text": ["mbë órò", "ebá ewu", "jɔ vivi"],
        "duration": [1.0, 2.0, 0.5],
        "source": ["w_202509_01", "w_202509_01", "w_202509_02"],
        "jw_code": ["aba", "aba", "aba"],
        "iso639_3": ["aba", "aba", "aba"],
    }
    return datasets.Dataset.from_dict(rows).cast_column("audio", datasets.Audio(sampling_rate=sr))


def test_build_subset_schema_and_dropped_columns(monkeypatch):
    fake = _fake_subset()
    monkeypatch.setattr(builder, "load_dataset", lambda *a, **k: fake, raising=False)
    # build_subset imports load_dataset locally, so patch the symbol it resolves:
    import datasets as _d
    monkeypatch.setattr(_d, "load_dataset", lambda *a, **k: fake)

    entry = by_subset("abbey_aba")
    out = builder.build_subset(entry, split="train", per_language=None)

    assert out.column_names == ["audio", "text", "language", "country", "length", "iso", "subset"]
    for gone in ("id", "source", "jw_code", "iso639_3"):
        assert gone not in out.column_names

    row = out[0]
    assert row["language"] == "Abbey"
    assert row["country"] == "CI"
    assert row["iso"] == "aba"
    assert row["subset"] == "abbey_aba"
    assert row["length"] == 1.0          # taken from the source duration column
    assert row["audio"] is not None      # decoded audio (dict or AudioDecoder)


def test_length_falls_back_to_audio_when_no_duration(monkeypatch):
    fake = _fake_subset().remove_columns(["duration"])
    import datasets as _d
    monkeypatch.setattr(_d, "load_dataset", lambda *a, **k: fake)
    out = builder.build_subset(by_subset("abbey_aba"), split="train", per_language=None)
    # lengths now computed from the audio itself (1.0s, 2.0s, 0.5s clips)
    assert sorted(out["length"]) == [0.5, 1.0, 2.0]


def test_per_language_cap(monkeypatch):
    fake = _fake_subset()
    import datasets as _d
    monkeypatch.setattr(_d, "load_dataset", lambda *a, **k: fake)
    out = builder.build_subset(by_subset("abbey_aba"), split="train", per_language=2)
    assert len(out) == 2


def test_clip_length_filter(monkeypatch):
    # fake clips have durations 1.0s, 2.0s, 0.5s
    fake = _fake_subset()
    import datasets as _d
    monkeypatch.setattr(_d, "load_dataset", lambda *a, **k: fake)
    # keep only clips between 0.8s and 1.5s -> just the 1.0s clip
    out = builder.build_subset(by_subset("abbey_aba"), split="train",
                               min_clip_seconds=0.8, max_clip_seconds=1.5)
    assert sorted(out["length"]) == [1.0]
    # min only
    out2 = builder.build_subset(by_subset("abbey_aba"), split="train", min_clip_seconds=1.0)
    assert sorted(out2["length"]) == [1.0, 2.0]
    # max only
    out3 = builder.build_subset(by_subset("abbey_aba"), split="train", max_clip_seconds=1.0)
    assert sorted(out3["length"]) == [0.5, 1.0]
