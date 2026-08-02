"""
forecasting.py

Generates genuine multi-day-ahead forecasts using the trained Random Forest
model, directly addressing the caveat raised in Notebook 06: lag/rolling
features account for ~77% of the model's predictive power, but those features
need real Sales history that doesn't exist for future dates.

The core idea -- RECURSIVE (walk-forward) forecasting: predict one day at a
time, in chronological order, and feed each day's own prediction back in as
"history" for computing the next day's lag/rolling features. This is standard
practice for lag-feature-dependent time series models, and it comes with a
known, honest limitation: errors can compound over the horizon, since day 10's
forecast partly depends on day 9's forecast being right, not on a real
observed value. That's a property of the problem, not a bug in this code.
"""

from typing import List

import pandas as pd

from src.feature_engineering import add_date_features, add_holiday_flag, add_promo2_active_flag
from src.preprocessing import clean_store_data, merge_store_metadata, parse_dates
from src.model import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_history_pivot(cleaned_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Date x Store pivot table of actual historical Sales.

    A pivot table (rather than repeated row-by-row lookups) makes every
    lag/rolling calculation in the forecast loop a single vectorized index
    operation instead of a per-store Python loop -- this is what makes
    forecasting 856 stores x 48 days tractable in this sandbox's single-core
    environment.
    """
    pivot = cleaned_train_df.pivot_table(index="Date", columns="Store", values="Sales", aggfunc="first")
    return pivot


def prepare_forecast_input(test_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the known, given inputs for the forecast period: merges store
    metadata, engineers date/holiday/promo2 features -- everything EXCEPT the
    lag/rolling features, which can only be computed inside the day-by-day
    recursive loop (build_forecast()) since they depend on prior days'
    predictions.

    Store 622 has 11 rows with a missing Open value in the last few days of
    the raw test.csv (a known quirk, confirmed in Notebook 01/04) -- imputed
    to Open=1 here, based on that store's own historical pattern of being
    open on 96.4% of weekdays/Saturdays, verified directly before choosing
    this value rather than assumed.
    """
    df = test_df.copy()
    df = parse_dates(df)
    df["Open"] = df["Open"].fillna(1)

    store_clean = clean_store_data(store_df)
    df = merge_store_metadata(df, store_clean)
    df = add_date_features(df)
    df = add_holiday_flag(df)
    df = add_promo2_active_flag(df)
    return df


def _encode_row_batch(day_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    One-hot encode a single day's rows and align them onto the exact same
    feature columns the model was trained on (from src/model.py's saved
    feature_cols) -- critical, since a single day's data may not contain every
    category the training set did (e.g. only 3 of 4 StoreTypes might appear
    among that day's stores), which would silently produce a misaligned
    feature matrix without this reindex step.
    """
    encoded = pd.get_dummies(day_df, columns=CATEGORICAL_FEATURES, drop_first=True)
    encoded = encoded.reindex(columns=feature_cols, fill_value=0)
    return encoded


def generate_forecast_frozen(
    model,
    forecast_input_df: pd.DataFrame,
    history_pivot: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    RECOMMENDED forecasting strategy, chosen after backtesting against the
    fully recursive approach above.

    Backtest result (Notebook 07, on the known Notebook 05/06 validation
    period so accuracy could actually be measured against real values):
      - Single-step (Notebook 06, using real history):         R^2 = 0.889
      - Fully recursive (this file's generate_forecast):        R^2 = 0.205
      - Frozen history (this function):                         R^2 = 0.728

    Why recursive collapses: lag/rolling features drive ~77% of this model's
    predictions (Notebook 06). Feeding the model's own noisy predictions back
    in as "history" compounds that noise through RollingMean30 especially,
    snowballing over the horizon. Freezing every future day's lag/rolling
    features at the LAST KNOWN REAL values avoids that compounding entirely --
    at the cost of those specific features going stale further into the
    horizon (they stop reflecting how sales actually evolve day to day).

    This is a genuine, evidence-based trade-off, not a shortcut: the
    day-varying features (Promo, DayOfWeek, StateHoliday, SchoolHoliday,
    calendar features) still change correctly per forecast day, so the model
    still responds to promotions and weekday patterns -- only the
    sales-history-derived features are held constant across the horizon.
    """
    last_date = history_pivot.index.max()
    df = forecast_input_df.copy()
    stores = df["Store"].unique()

    prev_day_frozen = history_pivot.loc[last_date].reindex(stores)
    prev_week_frozen = history_pivot.loc[last_date - pd.Timedelta(days=6)].reindex(stores) \
        if (last_date - pd.Timedelta(days=6)) in history_pivot.index else pd.Series(index=stores, dtype=float)
    window7 = history_pivot.loc[(history_pivot.index > last_date - pd.Timedelta(days=7)) & (history_pivot.index <= last_date)]
    window30 = history_pivot.loc[(history_pivot.index > last_date - pd.Timedelta(days=30)) & (history_pivot.index <= last_date)]
    rolling7_frozen = window7.reindex(columns=stores).mean()
    rolling30_frozen = window30.reindex(columns=stores).mean()

    df["PrevDaySales"] = df["Store"].map(prev_day_frozen)
    df["PrevWeekSales"] = df["Store"].map(prev_week_frozen)
    df["RollingMean7"] = df["Store"].map(rolling7_frozen)
    df["RollingMean30"] = df["Store"].map(rolling30_frozen)

    open_mask = df["Open"] == 1
    df["PredictedSales"] = 0.0
    if open_mask.any():
        X = _encode_row_batch(df.loc[open_mask], feature_cols)
        df.loc[open_mask, "PredictedSales"] = model.predict(X)

    return df[["Store", "Date", "PredictedSales", "Open"]]


def generate_forecast(
    model,
    forecast_input_df: pd.DataFrame,
    history_pivot: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    The recursive walk-forward forecast loop -- one day at a time, in
    chronological order.

    For each date: look up each store's most recent (real-or-already-forecast)
    sales history from `history_pivot` to build that day's lag/rolling
    features, predict Sales for Open==1 rows (Open==0 rows are trivially 0,
    consistent with how the entire pipeline has treated closed-store rows
    since Notebook 02), then APPEND that day's results back into
    `history_pivot` so the next iteration can use them. This is what makes it
    "recursive" -- day 2 onward partially depends on the model's own prior
    predictions, not solely on real observed data.
    """
    history_pivot = history_pivot.copy()
    results = []

    dates = sorted(forecast_input_df["Date"].unique())

    for current_date in dates:
        day_df = forecast_input_df[forecast_input_df["Date"] == current_date].copy()
        stores = day_df["Store"].values

        prev_day = current_date - pd.Timedelta(days=1)
        prev_week = current_date - pd.Timedelta(days=7)
        window7_start = current_date - pd.Timedelta(days=7)
        window30_start = current_date - pd.Timedelta(days=30)

        prev_day_sales = history_pivot.reindex(index=[prev_day], columns=stores).iloc[0]
        prev_week_sales = history_pivot.reindex(index=[prev_week], columns=stores).iloc[0]

        window7 = history_pivot.loc[(history_pivot.index >= window7_start) & (history_pivot.index < current_date)]
        window30 = history_pivot.loc[(history_pivot.index >= window30_start) & (history_pivot.index < current_date)]
        rolling7 = window7.reindex(columns=stores).mean()
        rolling30 = window30.reindex(columns=stores).mean()

        day_df["PrevDaySales"] = prev_day_sales.values
        day_df["PrevWeekSales"] = prev_week_sales.values
        day_df["RollingMean7"] = rolling7.values
        day_df["RollingMean30"] = rolling30.values

        open_mask = day_df["Open"] == 1
        day_df["PredictedSales"] = 0.0

        if open_mask.any():
            X_day = _encode_row_batch(day_df.loc[open_mask], feature_cols)
            preds = model.predict(X_day)
            day_df.loc[open_mask, "PredictedSales"] = preds

        results.append(day_df[["Store", "Date", "PredictedSales", "Open"]])

        # Feed this day's results (predicted where open, 0 where closed) back
        # into history so the NEXT date's lag/rolling features can see them.
        # Batch-assigned in one call rather than looping per store -- with up
        # to ~48 forecast days x ~850 stores, a per-store Python loop here
        # would add real overhead for no benefit.
        day_sales_series = day_df.set_index("Store")["PredictedSales"]
        if current_date not in history_pivot.index:
            history_pivot.loc[current_date] = pd.Series(dtype=float)
        history_pivot.loc[current_date, day_sales_series.index] = day_sales_series.values

    forecast_df = pd.concat(results, ignore_index=True)
    return forecast_df
