"""Load and represent the AfriSpeech language catalog.

The catalog is a static table (``data/catalog.tsv``) describing every subset
(language variant) in the source dataset: its HF config name, language label,
ISO 639-3 code, country, clip count, total hours and split sizes.

Hours is the primary "strength" signal used for ranking and filtering. The
table can be regenerated from the live dataset with ``scripts/refresh_catalog.py``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# The source dataset on the Hugging Face Hub. Each catalog row's ``subset`` is a
# config name within this dataset.
DATASET_ID = "AfriSpeech/african-speech-public_v1"

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = _DATA_DIR / "catalog.tsv"

# ISO 3166-1 alpha-2 -> display name, for the countries present in the catalog.
COUNTRY_NAMES = {
    "AO": "Angola", "BF": "Burkina Faso", "BI": "Burundi", "BJ": "Benin",
    "CD": "DR Congo", "CF": "Central African Republic", "CG": "Congo",
    "CI": "Côte d'Ivoire", "CM": "Cameroon", "CV": "Cabo Verde",
    "DZ": "Algeria", "ET": "Ethiopia", "GH": "Ghana", "GQ": "Equatorial Guinea",
    "GW": "Guinea-Bissau", "KE": "Kenya", "LR": "Liberia", "MG": "Madagascar",
    "MU": "Mauritius", "MW": "Malawi", "MZ": "Mozambique", "NA": "Namibia",
    "NG": "Nigeria", "RE": "Réunion", "RW": "Rwanda", "SC": "Seychelles",
    "SL": "Sierra Leone", "SN": "Senegal", "SS": "South Sudan", "TG": "Togo",
    "TZ": "Tanzania", "UG": "Uganda", "ZA": "South Africa", "ZM": "Zambia",
    "ZW": "Zimbabwe",
}


@dataclass(frozen=True)
class LanguageEntry:
    """One subset (language variant) of the dataset."""

    subset: str          # HF config name, e.g. "twi_twi"
    language: str        # human label, e.g. "Twi"
    iso: str             # ISO 639-3 code, e.g. "twi"
    country: str         # ISO 3166-1 alpha-2 code, e.g. "GH"
    clips: int
    hours: float
    train: int
    val: int
    test: int

    @property
    def country_name(self) -> str:
        return COUNTRY_NAMES.get(self.country, self.country)

    def split_size(self, split: str) -> int:
        """Number of clips available in a given split (or all splits)."""
        if split == "all":
            return self.clips
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["country_name"] = self.country_name
        return d


@lru_cache(maxsize=1)
def load_catalog(path: str | Path = CATALOG_PATH) -> list[LanguageEntry]:
    """Read the catalog TSV into a list of :class:`LanguageEntry`."""
    entries: list[LanguageEntry] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            entries.append(
                LanguageEntry(
                    subset=row["subset"].strip(),
                    language=row["language"].strip(),
                    iso=row["iso"].strip(),
                    country=row["country"].strip(),
                    clips=int(row["clips"]),
                    hours=float(row["hours"]),
                    train=int(row["train"]),
                    val=int(row["val"]),
                    test=int(row["test"]),
                )
            )
    return entries


def countries(entries: Iterable[LanguageEntry] | None = None) -> list[str]:
    """Sorted list of country codes present in the catalog."""
    entries = entries or load_catalog()
    return sorted({e.country for e in entries})


def by_subset(subset: str) -> LanguageEntry | None:
    for e in load_catalog():
        if e.subset == subset:
            return e
    return None
