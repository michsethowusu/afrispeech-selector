# 🌍 AfriSpeech Selector

A point-and-click tool for building training sets from
[`AfriSpeech/african-speech-public_v1`](https://huggingface.co/datasets/AfriSpeech/african-speech-public_v1).
Clone it, run one command, and a **Gradio UI opens in your browser**: rank
African languages by recorded **hours** (strength), pick a country-balanced
top-N (or hand-pick specific languages), size the sample, and either **download**
the result or **push it to your own Hugging Face dataset repo** — all in a fixed,
training-ready schema.

It runs **locally** so there are no server timeouts — pull as much audio as you
want.

## Quickstart

```bash
git clone https://github.com/AfriSpeech/afrispeech-selector.git
cd afrispeech-selector
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
python app.py
```

`python app.py` launches the UI and opens http://127.0.0.1:7860 in your browser.

Optional environment variables:

| var | default | meaning |
|-----|---------|---------|
| `PORT` | `7860` | port to serve on |
| `HOST` | `127.0.0.1` | bind address (use `0.0.0.0` to expose on your LAN) |
| `SHARE` | `0` | `1` → also create a temporary public `gradio.live` link |
| `NO_BROWSER` | `0` | `1` → don't auto-open the browser |

## What it does

- **Rank by hours.** Hours of audio is the strength signal. Pick the *top-N*
  languages, or hand-pick specific ones.
- **Balance across countries.** By default the top-N takes one strong language
  per country first, then fills seconds by hours — so a "top 10" spans 10
  countries, not 10 variants of one. Cap it (e.g. max 2/country) or switch to a
  pure hours ranking.
- **Filter the pool.** Drop languages below a **min hours** / above a **max
  hours** threshold, require a minimum clip count, or restrict to chosen
  countries.
- **Right-size the sample**, per language:
  - **Max samples per language** — clip-count cap.
  - **Max hours per language** — duration budget (decimals OK, e.g. `0.5`):
    accumulates clips until the summed audio is closest to the target.
  - **Sample length window** (min/max seconds) — a *precondition*: out-of-range
    clips are skipped while picking, so the target is filled from in-range clips.
- **Standard output schema** for every selection:

  | column | meaning |
  |--------|---------|
  | `audio` | decoded waveform (HF `Audio`: `array` + `sampling_rate`) |
  | `text` | transcription |
  | `language` | language label |
  | `country` | ISO 3166-1 alpha-2 code |
  | `length` | clip duration in seconds |
  | `iso`, `subset` | ISO 639-3 code and source config (for traceability) |

- **Two ways out.** Download a **zip** (round-trips with `load_from_disk`) or a
  single **parquet** file, *and/or* **push to the Hub** with your token. A
  manifest CSV is always included.

Capped pulls **stream** from the Hub and only transfer the samples you ask for,
so they're fast. An uncapped "full build" downloads whole shards (the dataset is
~65 GB) and must be explicitly enabled.

## Use it as a library / in your own training code

The selection logic is a small importable package — no UI required:

```python
from afrispeech_selector import filter_catalog, select_top, build_dataset

# Top 10 languages, country-balanced, >=10h each; 30 min/language, 3–20s clips
pool  = filter_catalog(min_hours=10, split="train")
langs = select_top(pool, 10, proportional=True, max_per_country=2)

ds = build_dataset(
    langs, split="train",
    max_seconds=1800,                 # 30 min per language
    min_clip_seconds=3, max_clip_seconds=20,
    streaming=True,
)
# ds: datasets.Dataset with audio/text/language/country/length/iso/subset
ds = ds.train_test_split(test_size=0.1)   # feed straight to your trainer
```

Load a downloaded selection later:

```python
from datasets import load_from_disk
ds = load_from_disk("afrispeech_selection")           # from the zip
# or: Dataset.from_parquet("afrispeech_selection.parquet")
```

## Tests

```bash
pip install pytest
pytest tests/        # selection-logic + standardisation tests (mostly offline)
```

## Keeping the catalog current

`data/catalog.tsv` is the static strength table the app reads (fast, offline).
When the source dataset changes, regenerate it:

```bash
python scripts/refresh_catalog.py --token "$HF_TOKEN"
```

## Project layout

```
app.py                       Gradio UI (run this)
afrispeech_selector/
  catalog.py                 load the language table; country names
  selector.py                ranking, filtering, country-proportional top-N, plan
  builder.py                 pull subsets from the Hub → standard schema (streaming)
  export.py                  zip / parquet / manifest / push_to_hub
data/catalog.tsv             strength table (hours, clips, splits per subset)
scripts/refresh_catalog.py   regenerate the table from the live dataset
tests/                       selection + builder tests
```

## License

CC-BY-4.0
