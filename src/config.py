"""
config.py

Central location for filesystem paths and dataset schema definitions.

Why this file exists: hardcoding paths like "../data/raw/train.csv" directly inside
notebooks or scripts is a common beginner mistake -- it silently breaks the moment a
file is run from a different working directory (e.g. the Streamlit dashboard, which
runs from the project root, not from notebooks/). Defining every path once, here,
relative to this file's own location, means it resolves correctly no matter where
the calling code lives.
"""

from pathlib import Path

# This file lives at <project_root>/src/config.py, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Directories ---
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# --- Raw input files ---
TRAIN_FILE = DATA_RAW_DIR / "train.csv"
TEST_FILE = DATA_RAW_DIR / "test.csv"
STORE_FILE = DATA_RAW_DIR / "store.csv"

# --- Processed output (created in Notebook 02) ---
CLEANED_DATA_FILE = DATA_PROCESSED_DIR / "cleaned_sales_data.csv"

# --- Expected schemas ---
# Used by data_loader.py to catch a corrupted/mismatched download early and loudly,
# instead of failing confusingly three steps later in feature engineering.
EXPECTED_TRAIN_COLUMNS = {
    "Store", "DayOfWeek", "Date", "Sales", "Customers",
    "Open", "Promo", "StateHoliday", "SchoolHoliday",
}

EXPECTED_TEST_COLUMNS = {
    "Id", "Store", "DayOfWeek", "Date",
    "Open", "Promo", "StateHoliday", "SchoolHoliday",
}

EXPECTED_STORE_COLUMNS = {
    "Store", "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
}

RANDOM_SEED = 42
