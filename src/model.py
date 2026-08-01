"""
model.py

Prepares the model-ready feature matrix, performs the mandatory chronological
train/validation split, and trains each of the PRD's required regression models.

Design note on Store ID: raw Store ID is deliberately EXCLUDED from the feature
set for both models, for a fair, apples-to-apples comparison. One-hot encoding
1,115 stores would add 1,115 columns -- expensive and prone to overfitting for
a plain linear model. Store-level *characteristics* (StoreType, Assortment,
CompetitionDistance, Promo2 status) are kept instead and carry most of the
genuinely predictive store-level signal found in Notebook 03. Real trade-off,
documented honestly: two different stores with identical characteristics will
receive identical predictions. Target-encoding Store ID is a reasonable future
improvement, intentionally out of scope here to avoid scope creep.
"""

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import MODELS_DIR, RANDOM_SEED

# Explicit ALLOW-list of features actually fed to a model -- safer than a
# deny-list here, since several columns (Date, PromoInterval, Store, Id) need
# excluding for reasons other than leakage (wrong type, high cardinality,
# already captured by a more specific engineered feature).
CATEGORICAL_FEATURES: List[str] = ["DayOfWeek", "StateHoliday", "StoreType", "Assortment"]

NUMERIC_FEATURES: List[str] = [
    "Year", "Month", "Day", "WeekOfYear", "Quarter",
    "Promo", "Promo2", "SchoolHoliday", "IsWeekend", "IsHoliday", "IsPromo2Active",
    "CompetitionDistance", "CompetitionDistance_was_missing",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "CompetitionOpenSince_was_missing",
    "PrevDaySales", "PrevWeekSales", "RollingMean7", "RollingMean30",
]

TARGET = "Sales"


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to Open==1 rows (closed-day Sales==0 is trivial, not learnable
    signal -- decision made in Notebook 02, applied here at modeling time as
    planned), and drops the small number of rows with missing lag/rolling
    features (each store's earliest history, no prior days to look back on yet).
    """
    df = df[df["Open"] == 1].copy()
    lag_cols = ["PrevDaySales", "PrevWeekSales", "RollingMean7", "RollingMean30"]
    before = len(df)
    df = df.dropna(subset=lag_cols)
    after = len(df)
    print(f"Filtered to Open==1 and dropped {before - after:,} rows with missing "
          f"lag/rolling history ({(before - after) / before:.3%})")
    return df


def chronological_split(df: pd.DataFrame, validation_days: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by DATE, not randomly -- the last `validation_days` days become the
    validation set, mirroring the ~6-week horizon of Rossmann's original Kaggle
    test period (a deliberate choice from the architecture review). Everything
    strictly before that cutoff is training data. Random shuffling would let
    the model validate on dates chronologically earlier than some of its own
    training data -- leaking future-relative-to-some-rows information.
    """
    cutoff = df["Date"].max() - pd.Timedelta(days=validation_days)
    train_df = df[df["Date"] <= cutoff].copy()
    val_df = df[df["Date"] > cutoff].copy()
    return train_df, val_df


def encode_features(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    One-hot encode the categorical features, fit on TRAIN only, then align the
    validation set onto the exact same columns.

    Fitting on train and reindexing validation (rather than one-hot encoding
    the combined data) guards against the validation set silently introducing
    categories the model never trained on, or the two sets ending up with
    mismatched column sets/order -- both of which would break prediction.
    """
    train_encoded = pd.get_dummies(train_df, columns=CATEGORICAL_FEATURES)
    val_encoded = pd.get_dummies(val_df, columns=CATEGORICAL_FEATURES)

    dummy_cols = [c for c in train_encoded.columns
                  if any(c.startswith(cat + "_") for cat in CATEGORICAL_FEATURES)]
    feature_cols = NUMERIC_FEATURES + dummy_cols

    train_encoded = train_encoded.reindex(columns=feature_cols, fill_value=0)
    val_encoded = val_encoded.reindex(columns=feature_cols, fill_value=0)

    return train_encoded, val_encoded, feature_cols


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Baseline model. Fast, fully interpretable (coefficients have direct
    meaning), and a necessary sanity-check floor: if Random Forest can't beat
    this, something in the pipeline needs investigating before trusting the
    more complex model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestRegressor:
    """
    Mandatory tree-based model. Hyperparameters below are deliberately modest
    (shallow-ish depth, sqrt feature sampling, a minimum leaf size) -- this
    sandbox has a single CPU core, and unrestricted defaults (sklearn's
    RandomForestRegressor uses ALL features per split by default, unlike the
    classifier) were benchmarked and found to take far too long at this row
    count. This is a real, documented compute constraint, not a modeling
    preference -- a production environment with more cores would reasonably
    support deeper trees, more estimators, and proper hyperparameter search.
    """
    defaults = dict(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=30,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    model = RandomForestRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def save_model(model, filename: str) -> Path:
    """Persist a trained model to models/ via joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path
