"""
utils.py

Small, genuinely reusable helper functions shared across notebooks and the
dashboard. Anything used in more than one place belongs here, not copy-pasted.
"""

import pandas as pd


def missing_report(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Return a DataFrame summarizing missing-value counts and percentages,
    sorted descending, for every column that has at least one missing value.

    Also prints a small labeled summary for quick reading in a notebook.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)

    label = f": {name}" if name else ""
    print(f"--- Missing values{label} ---")
    print(report if not report.empty else "No missing values.")
    print()

    return report


def save_chart(fig, filename: str, charts_dir) -> None:
    """
    Save a matplotlib figure to the outputs/charts/ directory as a PNG.

    Centralizing this (rather than repeating fig.savefig(...) in every EDA cell)
    means the DPI, format, and bounding-box settings only need to be decided once.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    filepath = charts_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {filepath.name}")
