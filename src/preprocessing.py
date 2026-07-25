"""
preprocessing.py

Responsible for cleaning and merging the raw Rossmann datasets, acting directly on
the findings documented in Notebook 01's Data Quality Report. Every function makes
exactly one decision and documents why in its docstring -- so six months from now,
anyone (including us) can see not just *what* was done but *why*, without having to
re-derive the reasoning from scratch.

Deliberately does NOT engineer new predictive features (date decomposition, lag or
rolling features) -- that is feature_engineering.py's job (Notebook 04). Cleaning
fixes what's wrong with the data as given; feature engineering adds new information
on top of already-correct data. Keeping these concerns separate means each can be
tested, reasoned about, and reused independently.
"""

import pandas as pd


def parse_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Convert a text date column to a real datetime dtype. Returns a copy."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def merge_store_metadata(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join store.csv onto a sales DataFrame (train or test) via the Store key.

    A left join is used -- not inner -- because we verified in Notebook 01 that
    every Store ID in train/test also exists in store.csv, so no rows should be
    lost. Left join also protects us if that ever changes (e.g. a future data
    refresh): a sales row would be kept with NaN store metadata instead of being
    silently dropped, which is far easier to notice and debug.
    """
    merged = sales_df.merge(store_df, on="Store", how="left")
    return merged


def clean_store_data(store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve the two genuine missing-value gaps identified in store.csv.

    1. CompetitionDistance (a few stores, ~0.3%): imputed with the column median,
       which is robust to the right-skew typical of distance data (a few very
       large outlier distances shouldn't pull a mean-based fill upward). A
       boolean flag column records exactly which rows were imputed, so this
       assumption is fully reversible and visible to any model or analyst that
       wants to discount it.

    2. CompetitionOpenSinceMonth / CompetitionOpenSinceYear (354 stores, ~31.8%):
       same median-imputation + flag approach. This is a much larger share of
       stores, which is exactly why the flag matters here -- a tree-based model
       can learn to treat "unknown competition open date" as its own signal
       rather than silently trusting a fabricated date for a third of the stores.

    Promo2SinceWeek / Promo2SinceYear / PromoInterval are deliberately NOT
    touched here. Notebook 01 confirmed these are missing precisely when
    Promo2 == 0 -- they are structurally "not applicable," not missing data.
    Imputing them would fabricate a fake recurring-promotion history for stores
    that never ran one. The existing Promo2 column already serves as the
    complete signal needed; downstream code should read PromoInterval only when
    Promo2 == 1.
    """
    df = store_df.copy()

    df["CompetitionDistance_was_missing"] = df["CompetitionDistance"].isnull()
    median_distance = df["CompetitionDistance"].median()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_distance)

    df["CompetitionOpenSince_was_missing"] = df["CompetitionOpenSinceMonth"].isnull()
    median_month = df["CompetitionOpenSinceMonth"].median()
    median_year = df["CompetitionOpenSinceYear"].median()
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(median_month)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(median_year)

    return df


def flag_suspicious_zero_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean flag for the "open but zero sales" edge case found in Notebook 01.

    These rows are NOT the same as closed-store rows (Open == 0, Sales == 0 is
    tautological and uninformative). A row where the store was OPEN yet sold
    nothing is unusual enough to be worth flagging rather than silently dropping
    or silently keeping -- flagging preserves the row (it may be real signal, e.g.
    a genuine demand shock) while making it easy for later analysis or modeling
    to isolate and test the effect of these rows explicitly.
    """
    df = df.copy()
    df["Suspicious_Zero_Sales"] = (df["Open"] == 1) & (df["Sales"] == 0)
    return df


def clean_and_merge(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the full Notebook 02 cleaning pipeline in one call:
    parse dates -> clean store metadata -> merge -> flag suspicious zero-sales rows.

    Deliberately does NOT filter out Open == 0 rows. That exclusion is a
    modeling-time decision (made explicitly in Notebook 05, right before the
    train/validation split), not a data-quality fix -- Open == 0 rows are
    truthful, not wrong, and EDA in Notebook 03 may still want to analyze
    closed-store patterns (e.g. holiday-driven closures).
    """
    train_parsed = parse_dates(train_df)
    store_clean = clean_store_data(store_df)
    merged = merge_store_metadata(train_parsed, store_clean)
    merged = flag_suspicious_zero_sales(merged)
    return merged
