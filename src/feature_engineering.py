"""
feature_engineering.py

Builds predictive features on top of the already-cleaned dataset from Notebook 02.
Every function documents its business rationale and, critically, whether the
resulting feature is safe to use for real forecasting -- see LEAKAGE_EXCLUDED_COLUMNS
at the bottom, which is the single source of truth for what Notebook 05 is allowed
to feed a model.
"""

from typing import List

import numpy as np
import pandas as pd

# Custom month-abbreviation map, NOT Python's datetime %b format.
# Verified against store.csv's actual PromoInterval values: the dataset spells
# September as "Sept" (4 letters), while Python's strftime("%b") produces "Sep"
# (3 letters). Using the standard format would silently fail to match every
# September row against an active recurring promotion -- caught by comparing
# the two directly before writing this function, not assumed.
MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec",
}


def add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Add calendar-derived features: Year, Month, Day, WeekOfYear, Quarter, IsWeekend.

    All of these derive only from `Date`/`DayOfWeek`, both of which are present
    in test.csv -- fully safe for real forecasting, no leakage risk.

    IsWeekend uses DayOfWeek (Saturday=6, Sunday=7). Per Notebook 03's finding,
    this does NOT mean "weekend = higher sales" -- Saturday was actually the
    softest reliable day. The flag still lets a model learn whatever the true
    (even counter-intuitive) relationship is, rather than us assuming one.
    """
    df = df.copy()
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["WeekOfYear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Quarter"] = df[date_col].dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7])
    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsHoliday: a simple boolean summary of StateHoliday.

    StateHoliday already carries the detailed holiday type ('a'/'b'/'c'), which
    stays in the dataset. IsHoliday is an additional, simpler binary signal for
    models/visualizations that just need "holiday or not," directly motivated
    by Notebook 03's Holiday Impact chart.
    """
    df = df.copy()
    df["IsHoliday"] = df["StateHoliday"] != "0"
    return df


def add_promo2_active_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsPromo2Active: whether a store's ONGOING recurring promotion (Promo2)
    is actually active in the given row's month -- not just whether the store
    is EVER enrolled in the program.

    Why this matters: the raw `Promo2` column only says a store participates in
    the recurring-promotion program at all; it says nothing about *when*. Two
    stores with Promo2==1 could be in completely different points of their promo
    calendar on any given date. `PromoInterval` (e.g. "Feb,May,Aug,Nov") specifies
    which months the recurring promotion actually runs. This feature checks the
    row's own month against that list, so the model gets a real activity signal
    instead of a static enrollment flag. This directly follows up on Notebook 03's
    ambiguous Promo2 finding (correlation was negative, likely a selection
    effect) -- this feature gives a model the chance to find the true, more
    precise relationship instead of relying on the coarser proxy.

    Deliberately NOT using Python's strftime("%b") to get the month name -- the
    dataset spells September as "Sept" (4 letters), which strftime does not
    produce. A custom mapping is used instead (see MONTH_ABBR).
    """
    df = df.copy()
    month_name = df["Month"].map(MONTH_ABBR)

    def _is_active(row_month_name: str, promo_interval) -> bool:
        if pd.isna(promo_interval):
            return False
        months = promo_interval.split(",")
        return row_month_name in months

    is_active = [
        _is_active(m, interval)
        for m, interval in zip(month_name, df["PromoInterval"])
    ]
    df["IsPromo2Active"] = pd.Series(is_active, index=df.index) & (df["Promo2"] == 1)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PrevDaySales, PrevWeekSales, RollingMean7, RollingMean30 -- all computed
    per store, in chronological order, using ONLY past days.

    Leakage-safety note: pandas' .rolling() includes the CURRENT row by default.
    Using it directly on Sales would leak the current day's own target value
    into its own feature. Every rolling/lag calculation below is built on
    Sales.shift(1) first (i.e. "yesterday and earlier"), so the feature for any
    given row only ever reflects days strictly before it.

    These features naturally contain NaN for each store's earliest rows (no
    history yet exists) -- left as NaN here rather than silently filled, so
    Notebook 05 makes an explicit, visible decision about how to handle them
    (e.g. drop those rows only when these specific features are used).
    """
    df = df.sort_values(["Store", "Date"]).copy()
    grouped_sales = df.groupby("Store")["Sales"]

    df["PrevDaySales"] = grouped_sales.shift(1)
    df["PrevWeekSales"] = grouped_sales.shift(7)

    shifted = grouped_sales.shift(1)
    df["RollingMean7"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["RollingMean30"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(30, min_periods=1).mean())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates the full Notebook 04 feature engineering pipeline, in order."""
    df = add_date_features(df)
    df = add_holiday_flag(df)
    df = add_promo2_active_flag(df)
    df = add_lag_and_rolling_features(df)
    return df


# ----------------------------------------------------------------------------
# Feature-leakage audit: the single source of truth for what Notebook 05 may
# actually feed a model. Verified directly against test.csv's real columns
# (Notebook 04), not assumed from documentation.
# ----------------------------------------------------------------------------

LEAKAGE_EXCLUDED_COLUMNS: List[str] = [
    "Customers",               # Not present in test.csv -- unavailable at real forecast time,
                                # despite being the single strongest correlate with Sales (0.82).
    "Suspicious_Zero_Sales",   # Derived directly from Sales itself (Notebook 02) -- circular,
                                # cannot be computed for genuine future/unseen rows.
]
