"""
data_loader.py

Responsible for ONE thing: reading the three raw Rossmann CSVs into memory and
validating that their structure matches what the rest of the pipeline expects.

Deliberately does NOT clean, transform, or merge anything -- that is preprocessing.py's
job (Notebook 02). Keeping "load" and "clean" as separate, single-responsibility
functions makes each easier to test, debug, and reuse independently. It also means
the Streamlit dashboard can call load_all() later and get the exact same guarantees
the notebooks did, without re-implementing validation logic.
"""

from typing import Tuple

import pandas as pd

from src.config import (
    CLEANED_DATA_FILE,
    EXPECTED_STORE_COLUMNS,
    EXPECTED_TEST_COLUMNS,
    EXPECTED_TRAIN_COLUMNS,
    FEATURED_DATA_FILE,
    STORE_FILE,
    TEST_FILE,
    TRAIN_FILE,
)


def _validate_columns(df: pd.DataFrame, expected: set, name: str) -> None:
    """
    Raise a clear, early error if a dataset's columns don't match expectations.

    Why this matters: without this check, a corrupted download or a wrong file
    swapped into data/raw/ would fail silently -- possibly not until a KeyError
    deep inside feature engineering, far from the actual root cause. Failing loudly
    at load time, with a specific missing-column name, saves real debugging time.
    """
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(f"{name}: missing expected column(s): {sorted(missing)}")
    if extra:
        print(f"Warning - {name}: unexpected extra column(s) found: {sorted(extra)}")


def load_train() -> pd.DataFrame:
    """
    Load train.csv and validate its schema.

    low_memory=False forces pandas to infer each column's dtype from the entire
    file at once, rather than chunk-by-chunk. Verified in Notebook 01: for this
    dataset, that alone is sufficient to prevent the StateHoliday column from
    being read as a mixed int/string type -- reading with default settings
    reproduces the mixed-type warning, but low_memory=False resolves it cleanly.
    """
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    _validate_columns(df, EXPECTED_TRAIN_COLUMNS, "train.csv")
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv and validate its schema."""
    df = pd.read_csv(TEST_FILE, low_memory=False)
    _validate_columns(df, EXPECTED_TEST_COLUMNS, "test.csv")
    return df


def load_store() -> pd.DataFrame:
    """Load store.csv and validate its schema."""
    df = pd.read_csv(STORE_FILE)
    _validate_columns(df, EXPECTED_STORE_COLUMNS, "store.csv")
    return df


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function: load and validate all three raw datasets at once."""
    return load_train(), load_test(), load_store()


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the cleaned/merged dataset produced by Notebook 02.

    IMPORTANT: uses low_memory=False and explicit Date parsing deliberately.
    Verified while building Notebook 03: even though StateHoliday was a clean,
    consistent string type in memory when Notebook 02 saved this file, CSVs
    don't preserve pandas dtype metadata -- reading it back with default
    settings silently reintroduces the exact same mixed int/string bug from
    Notebook 01, because pandas re-infers types from scratch on every read.
    This function exists so every notebook from here on reads the cleaned data
    safely by default, instead of relying on everyone remembering the flag.
    """
    df = pd.read_csv(CLEANED_DATA_FILE, low_memory=False, parse_dates=["Date"])
    return df


def load_featured_data() -> pd.DataFrame:
    """
    Load the feature-engineered dataset produced by Notebook 04.

    Same low_memory=False + explicit Date parsing as load_cleaned_data(), for
    the identical reason: CSVs don't preserve dtype metadata, so every fresh
    read needs to guard against StateHoliday's mixed-type re-emergence.
    """
    df = pd.read_csv(FEATURED_DATA_FILE, low_memory=False, parse_dates=["Date"])
    return df
