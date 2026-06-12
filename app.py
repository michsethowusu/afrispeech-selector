"""AfriSpeech Selector — Hugging Face Space UI.

A friendly interface over ``AfriSpeech/african-speech-public_v1``: rank African
languages by recorded hours, pick a country-balanced top-N (or hand-pick
specific languages), cap the sample size per language, and either download the
selection or push it to your own HF dataset repo.
"""

from __future__ import annotations

import pandas as pd
import gradio as gr

from afrispeech_selector import (
    DATASET_ID,
    build_dataset,
    countries,
    export_archive,
    export_metadata_csv,
    export_parquet,
    filter_catalog,
    load_catalog,
    plan_samples,
    push_to_hub,
    select_top,
)
from afrispeech_selector.catalog import COUNTRY_NAMES

CATALOG = load_catalog()
SPLITS = ["train", "val", "test", "all"]

# Map "Twi — Ghana (twi_twi, 50.3h)" -> subset for the hand-pick dropdown.
LANG_LABELS = {
    f"{e.language} — {e.country_name} ({e.subset}, {e.hours:.1f}h)": e.subset
    for e in sorted(CATALOG, key=lambda x: x.language)
}
COUNTRY_CHOICES = [f"{c} — {COUNTRY_NAMES.get(c, c)}" for c in countries()]


def _code(c: str) -> str:
    return c.split(" — ")[0]


def _plan_to_df(plan: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(plan)
    cols = ["language", "country_name", "iso", "hours", "clips",
            "available", "planned", "planned_hours", "subset"]
    df = df[[c for c in cols if c in df.columns]]
    return df.rename(columns={"country_name": "country", "available": "in_split",
                              "planned": "to_pull", "planned_hours": "est_hours"})


def _caps(per_language, max_hours):
    """Normalise the two cap inputs into (clip_cap, seconds_cap)."""
    cap = int(per_language) if per_language else None
    secs = float(max_hours) * 3600 if max_hours else None
    return cap, secs


def _clip_range(min_len, max_len):
    lo = float(min_len) if min_len else None
    hi = float(max_len) if max_len else None
    return lo, hi


def _summary(plan, split, cap, secs, lo=None, hi=None):
    total_clips = sum(p["planned"] for p in plan)
    total_hours = round(sum(p.get("planned_hours", 0) for p in plan), 2)
    limit = []
    if cap:
        limit.append(f"≤{cap} clips/lang")
    if secs:
        limit.append(f"≤{secs/3600:g} h/lang")
    if not limit:
        limit.append("all available")
    if lo is not None or hi is not None:
        limit.append(f"clip {lo or 0:g}–{hi if hi is not None else '∞'}s")
    return (f"~**{total_clips}** clips / **~{total_hours} h** total to pull "
            f"(split=`{split}`, limit: {', '.join(limit)}).")


# --------------------------------------------------------------------------- #
# Selection callbacks (no heavy I/O — just the catalog)
# --------------------------------------------------------------------------- #
def preview_top(n, proportional, max_per_country, min_hours, max_hours,
                min_clips, restrict_countries, split, per_language, max_hours_per_lang,
                min_len, max_len):
    pool = filter_catalog(
        min_hours=float(min_hours or 0),
        max_hours=float(max_hours) if max_hours else None,
        min_clips=int(min_clips or 0),
        countries=[_code(c) for c in restrict_countries] if restrict_countries else None,
        split=split,
    )
    if not pool:
        return None, {}, "No languages match these filters. Loosen the hours/clip limits."
    chosen = select_top(
        pool, int(n),
        proportional=bool(proportional),
        max_per_country=int(max_per_country) if max_per_country else None,
    )
    cap, secs = _caps(per_language, max_hours_per_lang)
    lo, hi = _clip_range(min_len, max_len)
    plan = plan_samples(chosen, per_language=cap, max_seconds=secs,
                        min_clip_seconds=lo, max_clip_seconds=hi, split=split)
    n_countries = len({p["country"] for p in plan})
    msg = f"Selected **{len(chosen)}** languages across **{n_countries}** countries — " + _summary(plan, split, cap, secs, lo, hi)
    sel = {"subsets": [e.subset for e in chosen], "split": split, "cap": cap,
           "secs": secs, "min_len": lo, "max_len": hi}
    return _plan_to_df(plan), sel, msg


def preview_pick(picked_labels, split, per_language, max_hours_per_lang, min_len, max_len):
    if not picked_labels:
        return None, {}, "Pick at least one language."
    subsets = [LANG_LABELS[l] for l in picked_labels]
    chosen = [e for e in CATALOG if e.subset in subsets]
    cap, secs = _caps(per_language, max_hours_per_lang)
    lo, hi = _clip_range(min_len, max_len)
    plan = plan_samples(chosen, per_language=cap, max_seconds=secs,
                        min_clip_seconds=lo, max_clip_seconds=hi, split=split)
    msg = f"Selected **{len(chosen)}** languages — " + _summary(plan, split, cap, secs, lo, hi)
    sel = {"subsets": subsets, "split": split, "cap": cap,
           "secs": secs, "min_len": lo, "max_len": hi}
    return _plan_to_df(plan), sel, msg


# --------------------------------------------------------------------------- #
# Build + export (heavy: pulls audio from the Hub)
# --------------------------------------------------------------------------- #
# An uncapped pull downloads whole shards (the full 65 GB dataset) — slow and
# disk-heavy. Require a per-language limit unless the user opts into a full build.
UNCAPPED_GUARD = 2000

def build_and_export(sel, out_format, hf_token, allow_full, progress=gr.Progress()):
    if not sel or not sel.get("subsets"):
        raise gr.Error("Nothing selected. Click a Preview button first.")
    subsets = sel["subsets"]
    split = sel.get("split", "train")
    cap = sel.get("cap")
    secs = sel.get("secs")
    min_clip = sel.get("min_len")
    max_clip = sel.get("max_len")
    if min_clip is not None and max_clip is not None and min_clip > max_clip:
        raise gr.Error(f"Min sample length ({min_clip}s) is greater than max ({max_clip}s).")

    if cap is None and secs is None and not allow_full:
        raise gr.Error(
            "No per-language limit set. An uncapped pull downloads whole shards (the "
            "full ~65 GB dataset). Set 'Max samples per language' or 'Max hours per "
            "language' for a fast streamed pull, or tick 'Allow uncapped full build' "
            "if you really want everything."
        )
    if cap is not None and cap > UNCAPPED_GUARD and not allow_full:
        raise gr.Error(
            f"Cap of {cap}/language is large and may be slow. "
            f"Lower it to ≤{UNCAPPED_GUARD}, or tick 'Allow uncapped full build'."
        )

    # Stream when a limit is set (transfers only what's needed); full download otherwise.
    use_streaming = cap is not None or secs is not None
    n_done = {"i": 0}

    def _p(msg):
        n_done["i"] += 1
        frac = 0.05 + 0.8 * min(1.0, n_done["i"] / max(1, len(subsets)))
        progress(frac, desc=msg)

    progress(0.02, desc=f"Pulling {len(subsets)} languages (streaming={use_streaming})…")
    ds = build_dataset(
        subsets, split=split, per_language=cap, max_seconds=secs,
        min_clip_seconds=min_clip, max_clip_seconds=max_clip,
        token=hf_token or None, streaming=use_streaming, progress=_p,
    )
    progress(0.9, desc="Packaging download…")

    files = []
    if out_format in ("zip", "both"):
        files.append(export_archive(ds))
    if out_format in ("parquet", "both"):
        files.append(export_parquet(ds))
    files.append(export_metadata_csv(ds))

    summary = (f"Built **{len(ds)}** clips, {len(subsets)} languages (split=`{split}`). "
               f"Columns: `{', '.join(ds.column_names)}`.")
    return ds, files, summary


def do_push(state_ds, repo_id, hf_token, private, split):
    if state_ds is None:
        raise gr.Error("Build the dataset first (it's held for this session).")
    if not hf_token:
        raise gr.Error("Provide a Hugging Face token with write access.")
    url = push_to_hub(state_ds, repo_id, token=hf_token, private=bool(private), split=split)
    return f"✅ Pushed to [{repo_id}]({url})"


LOAD_SNIPPET = '''```python
# After downloading & unzipping the archive:
from datasets import load_from_disk
ds = load_from_disk("afrispeech_selection")

# Or from the single parquet file:
from datasets import Dataset
ds = Dataset.from_parquet("afrispeech_selection.parquet")

# Columns: audio (decoded), text, language, country, length, iso, subset
print(ds[0])
```'''


with gr.Blocks(title="AfriSpeech Selector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# 🌍 AfriSpeech Selector\n"
        f"Build training sets from **`{DATASET_ID}`** by selecting African languages — "
        f"ranked by recorded **hours** (strength), balanced across countries.\n\n"
        f"Output schema: `audio · text · language · country · length`. "
        f"Download it, or push it to your own HF dataset repo."
    )

    sel_state = gr.State({})        # {"subsets": [...], "split": str, "cap": int|None}
    ds_state = gr.State(None)       # built datasets.Dataset (per session)

    with gr.Tab("🏆 Top languages across Africa"):
        with gr.Row():
            n = gr.Number(value=10, label="How many languages (top-N)", precision=0, minimum=1)
            split_a = gr.Dropdown(SPLITS, value="train", label="Pull from split")
            per_lang_a = gr.Number(label="Max samples per language (blank = all)", precision=0)
            max_hr_a = gr.Number(label="Max hours per language (blank = no limit, e.g. 0.5)", minimum=0)
        with gr.Row():
            min_len_a = gr.Number(label="Min sample length (sec, blank = no min)", minimum=0)
            max_len_a = gr.Number(label="Max sample length (sec, blank = no max)", minimum=0)
        gr.Markdown(
            "_Sample length window is a precondition: out-of-range clips are skipped "
            "while picking, so the sample/hour target is filled from in-range clips only._"
        )
        with gr.Row():
            proportional = gr.Checkbox(value=True, label="Balance across countries (one per country first, then fill)")
            max_pc = gr.Number(label="Max languages per country (blank = no cap)", precision=0)
        with gr.Accordion("Strength filters", open=True):
            with gr.Row():
                min_h = gr.Number(value=0, label="Min hours", minimum=0)
                max_h = gr.Number(label="Max hours (blank = no cap)")
                min_c = gr.Number(value=0, label="Min clips", precision=0, minimum=0)
            restrict_c = gr.Dropdown(COUNTRY_CHOICES, label="Restrict to countries (optional)", multiselect=True)
        btn_top = gr.Button("Preview selection", variant="primary")

    with gr.Tab("🎯 Pick specific languages"):
        picked = gr.Dropdown(list(LANG_LABELS), label="Languages", multiselect=True)
        with gr.Row():
            split_b = gr.Dropdown(SPLITS, value="train", label="Pull from split")
            per_lang_b = gr.Number(label="Max samples per language (blank = all)", precision=0)
            max_hr_b = gr.Number(label="Max hours per language (blank = no limit, e.g. 0.5)", minimum=0)
        with gr.Row():
            min_len_b = gr.Number(label="Min sample length (sec, blank = no min)", minimum=0)
            max_len_b = gr.Number(label="Max sample length (sec, blank = no max)", minimum=0)
        btn_pick = gr.Button("Preview selection", variant="primary")

    gr.Markdown("### Selection")
    status = gr.Markdown()
    table = gr.Dataframe(label="What will be pulled", interactive=False, wrap=True)

    gr.Markdown("### Build & export")
    with gr.Row():
        out_format = gr.Radio(["zip", "parquet", "both"], value="zip", label="Download format")
        token = gr.Textbox(label="HF token (needed to push or for gated access)", type="password", placeholder="hf_…")
    allow_full = gr.Checkbox(
        value=False,
        label="Allow uncapped / large full build (slow — downloads whole shards, lots of disk)",
    )
    gr.Markdown(
        "_Tip: set a per-language sample cap or hour budget for a fast streamed pull. "
        "Capped pulls only transfer the samples you ask for._"
    )
    btn_build = gr.Button("⬇️ Build & prepare download", variant="primary")
    build_status = gr.Markdown()
    files_out = gr.File(label="Download", file_count="multiple")

    with gr.Accordion("🚀 Push to a Hugging Face dataset repo (optional)", open=False):
        with gr.Row():
            repo_id = gr.Textbox(label="Target repo id", placeholder="your-username/my-afrispeech-subset")
            push_split = gr.Dropdown(SPLITS[:-1], value="train", label="Upload as split")
            private = gr.Checkbox(value=True, label="Private repo")
        btn_push = gr.Button("Push to Hub")
        push_status = gr.Markdown()

    with gr.Accordion("📋 How to load the result for training", open=False):
        gr.Markdown(LOAD_SNIPPET)

    # Wiring
    btn_top.click(
        preview_top,
        [n, proportional, max_pc, min_h, max_h, min_c, restrict_c, split_a, per_lang_a,
         max_hr_a, min_len_a, max_len_a],
        [table, sel_state, status],
    )
    btn_pick.click(
        preview_pick, [picked, split_b, per_lang_b, max_hr_b, min_len_b, max_len_b],
        [table, sel_state, status]
    )
    # Build reads the split/cap captured in the selection state by whichever
    # Preview button was last clicked.
    btn_build.click(
        build_and_export,
        [sel_state, out_format, token, allow_full],
        [ds_state, files_out, build_status],
    )
    btn_push.click(
        do_push, [ds_state, repo_id, token, private, push_split], [push_status]
    )


if __name__ == "__main__":
    import os

    # Runs locally: opens the UI in your browser at http://127.0.0.1:7860.
    # The queue keeps long build/push jobs alive (heartbeats). Set SHARE=1 for a
    # temporary public gradio.live link, or PORT to change the port.
    demo.queue(default_concurrency_limit=2, max_size=16).launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("SHARE", "0") == "1",
        inbrowser=os.environ.get("NO_BROWSER", "0") != "1",
        show_error=True,
    )
