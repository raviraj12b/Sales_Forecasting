import pandas as pd


def parse_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def merge_store_metadata(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    merged = sales_df.merge(store_df, on="Store", how="left")
    return merged


def clean_store_data(store_df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    df["Suspicious_Zero_Sales"] = (df["Open"] == 1) & (df["Sales"] == 0)
    return df


def clean_and_merge(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    train_parsed = parse_dates(train_df)
    store_clean = clean_store_data(store_df)
    merged = merge_store_metadata(train_parsed, store_clean)
    merged = flag_suspicious_zero_sales(merged)
    return merged
