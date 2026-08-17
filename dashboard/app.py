"""
dashboard/app.py

Streamlit dashboard for the Sales Forecasting & Business Analytics Platform.

Architecture principle (from the project's architecture review): the dashboard
is a SERVING layer only -- it reads the artifacts the notebooks already
produced (cleaned data, trained model, predictions.csv) and never retrains or
re-cleans anything live. If those artifacts don't exist yet (e.g. a fresh
clone before running notebooks 01-07), pages fail gracefully with a clear
message rather than crashing.

UI layer notes (Phase 10 redesign):
- This file only changes presentation. Every data/model computation reuses
  the same functions and produces the same numbers as before the redesign.
- Visual design system lives in the CSS block below `# --- design tokens ---`.
  Colors are defined once in DESIGN_TOKENS and reused for both the injected
  CSS and the Plotly chart theming, so the two never drift out of sync.
- Reusable UI components (render_kpi_card, render_section_header,
  render_chart_container, render_insight_card, render_empty_state,
  render_metric_card, render_sidebar) live in the "COMPONENTS" section and are
  used by every page below.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CHARTS_DIR, DATA_SAMPLE_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import load_cleaned_data, load_featured_data
from src.model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    chronological_split,
    encode_features,
    prepare_model_data,
)
from src.utils import missing_report

# ============================================================
# Page config -- must be the first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Sales Forecasting & Business Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Design tokens -- single source of truth for both the injected
# CSS below and the Plotly chart theming further down. Change a
# color here and both the UI chrome and the charts follow.
# ============================================================
BG = "#0B0E14"            # page background
PANEL_BG = "#141926"      # sidebar / card / code background
BORDER = "#262C3A"        # subtle borders
TEXT_PRIMARY = "#E8EAED"
TEXT_MUTED = "#8B93A7"
ACCENT = "#E8A33D"        # single restrained accent -- forecast, selection, highlights
ACCENT_SOFT = "rgba(232, 163, 61, 0.12)"
POSITIVE = "#4FAE7C"
NEGATIVE = "#D9614F"
BLUE = "#5B7FDB"           # "actual / historical" series -- contrasts with the accent

PLOT_COLORWAY = [BLUE, ACCENT, POSITIVE, NEGATIVE, "#8B93A7"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
}}
footer {{ visibility: hidden; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER};
}}
.brand-block {{ padding: 0.2rem 0 1.1rem 0; }}
.brand-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}
.brand-subtitle {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    margin-top: 0.1rem;
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.03em;
    color: {TEXT_MUTED};
    margin-top: 0.7rem;
}}
.status-dot {{ width: 6px; height: 6px; border-radius: 50%; }}
.status-dot.ok {{ background: {POSITIVE}; }}
.status-dot.warn {{ background: {ACCENT}; }}

/* nav buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.1rem;
    border-radius: 8px;
    transition: background 0.12s ease;
}}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border: 1px solid transparent !important;
    color: {TEXT_MUTED} !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
    background: rgba(255,255,255,0.045) !important;
    color: {TEXT_PRIMARY} !important;
}}
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
    background: {ACCENT_SOFT} !important;
    border: 1px solid {ACCENT} !important;
    color: {ACCENT} !important;
    box-shadow: inset 3px 0 0 {ACCENT} !important;
    font-weight: 600;
}}

/* ---------- page header ---------- */
.kicker {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.35rem;
}}
.page-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
    line-height: 1.15;
}}
.page-subtitle {{
    font-size: 0.98rem;
    color: {TEXT_MUTED};
    margin-top: 0.3rem;
    max-width: 62rem;
}}
.section-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 0.2rem 0 0.1rem 0;
}}
.section-subtitle {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.4rem;
}}

/* ---------- status row (overview snapshot) ---------- */
.status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 2.2rem;
    padding: 0.9rem 0;
    border-top: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    margin: 1rem 0 1.4rem 0;
}}
.status-item-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
}}
.status-item-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin-top: 0.15rem;
}}

/* ---------- KPI cards ---------- */
.kpi-card {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.05rem 1.2rem 0.95rem 1.2rem;
    height: 100%;
}}
.kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.15;
    margin-top: 0.3rem;
}}
.kpi-context {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    margin-top: 0.25rem;
}}
.kpi-delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    font-weight: 600;
    margin-top: 0.5rem;
}}
.kpi-delta-pos {{ color: {POSITIVE}; }}
.kpi-delta-neg {{ color: {NEGATIVE}; }}

/* ---------- insight cards ---------- */
.insight-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.98rem;
    color: {TEXT_PRIMARY};
    margin-bottom: 0.35rem;
}}
.insight-body {{
    font-size: 0.86rem;
    color: {TEXT_MUTED};
    line-height: 1.55;
}}
.insight-body b {{ color: {TEXT_PRIMARY}; font-weight: 600; }}

/* ---------- chart panel title ---------- */
.chart-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    color: {TEXT_PRIMARY};
}}
.chart-subtitle {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.4rem;
}}

/* ---------- empty / error state ---------- */
.state-card {{
    background: {PANEL_BG};
    border: 1px dashed {BORDER};
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    text-align: left;
}}
.state-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    font-size: 1rem;
}}
.state-detail {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    margin-top: 0.3rem;
    line-height: 1.5;
}}

/* ---------- st.metric polish ---------- */
div[data-testid="stMetric"] {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.85rem 1rem;
}}
div[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* misc */
.small-caption {{ font-size: 0.8rem; color: {TEXT_MUTED}; }}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached data / model loading -- the ONLY place these are read
# ============================================================
def _missing(path: Path) -> bool:
    return not path.exists()


@st.cache_data(show_spinner="Loading cleaned dataset...")
def get_cleaned_data():
    return load_cleaned_data()


@st.cache_data(show_spinner="Loading sample dataset...")
def get_sample_data():
    path = DATA_SAMPLE_DIR / "sales_sample.csv"
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_data(show_spinner="Loading predictions...")
def get_predictions():
    return pd.read_csv(OUTPUTS_DIR / "predictions.csv", parse_dates=["Date"])


@st.cache_resource(show_spinner="Loading Random Forest model...")
def get_rf_model():
    return joblib.load(MODELS_DIR / "random_forest.pkl")


@st.cache_resource(show_spinner="Loading Linear Regression model...")
def get_lr_model():
    return joblib.load(MODELS_DIR / "linear_regression.pkl")


@st.cache_data(show_spinner="Preparing validation data for live metrics...")
def get_validation_data():
    featured = load_featured_data()
    ready = prepare_model_data(featured)
    train_df, val_df = chronological_split(ready)
    X_train, X_val, feature_cols = encode_features(train_df, val_df)
    y_train, y_val = train_df[TARGET], val_df[TARGET]
    return X_val, y_val, feature_cols


def pipeline_ready() -> bool:
    return not (
        _missing(OUTPUTS_DIR / "predictions.csv")
        or _missing(MODELS_DIR / "random_forest.pkl")
        or _missing(MODELS_DIR / "linear_regression.pkl")
    )


# ============================================================
# COMPONENTS -- reusable UI building blocks used by every page
# ============================================================
def render_section_header(title: str, subtitle: str | None = None, kicker: str | None = None):
    if kicker:
        st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_subsection(title: str, subtitle: str | None = None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_kpi_card(col, label: str, value: str, context: str = "", delta: str | None = None, positive: bool = True):
    delta_html = ""
    if delta:
        arrow = "▲" if positive else "▼"
        cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    col.markdown(
        f"""<div class="kpi-card">
<div class="kpi-label">{label}</div>
<div class="kpi-value">{value}</div>
<div class="kpi-context">{context}</div>
{delta_html}
</div>""",
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: list[dict]):
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        render_kpi_card(col, **kpi)


def render_metric_card(col, label: str, value: str, delta: str | None = None, delta_color: str = "normal"):
    """Thin wrapper around st.metric so every metric on the platform is
    labeled and formatted the same way, and picks up the .stMetric CSS."""
    col.metric(label, value, delta=delta, delta_color=delta_color)


def render_status_row(items: list[tuple[str, str]]):
    cells = "".join(
        f'<div><div class="status-item-label">{label}</div>'
        f'<div class="status-item-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="status-row">{cells}</div>', unsafe_allow_html=True)


def render_insight_card(title: str, body_html: str):
    with st.container(border=True):
        st.markdown(f'<div class="insight-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-body">{body_html}</div>', unsafe_allow_html=True)


def render_empty_state(title: str, detail: str):
    st.markdown(
        f"""<div class="state-card">
<div class="state-title">{title}</div>
<div class="state-detail">{detail}</div>
</div>""",
        unsafe_allow_html=True,
    )


@contextmanager
def render_chart_container(title: str, subtitle: str | None = None):
    """Usage: with render_chart_container("Sales Trend"): st.plotly_chart(fig, width="stretch")"""
    with st.container(border=True):
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        yield


def style_fig(fig, height: int = 380, show_legend: bool = True):
    """Applies the shared dark chart theme. Called on every Plotly figure so
    charts read as one product instead of a pile of default-themed plots."""
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        colorway=PLOT_COLORWAY,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL_BG, font_color=TEXT_PRIMARY, font_family="Inter", bordercolor=BORDER),
        xaxis=dict(gridcolor=BORDER, zeroline=False, showline=False),
        yaxis=dict(gridcolor=BORDER, zeroline=False, showline=False),
    )
    return fig


# ============================================================
# Sidebar navigation (session-state driven so the active page
# gets an obvious, CSS-controlled highlighted state)
# ============================================================
NAV_ITEMS = ["Overview", "Data", "Sales Analytics", "Machine Learning", "Forecast", "Business Insights", "About"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"


def render_sidebar():
    st.sidebar.markdown(
        """<div class="brand-block">
<div class="brand-title">🏪 Sales Forecasting</div>
<div class="brand-subtitle">Business Analytics Platform</div>
</div>""",
        unsafe_allow_html=True,
    )
    for name in NAV_ITEMS:
        active = st.session_state.page == name
        if st.sidebar.button(name, key=f"nav_{name}", width="stretch", type="primary" if active else "secondary"):
            st.session_state.page = name
            st.rerun()

    st.sidebar.markdown("---")
    ready = pipeline_ready()
    dot_cls = "ok" if ready else "warn"
    status_text = "Pipeline ready" if ready else "Artifacts missing"
    st.sidebar.markdown(
        f'<div class="status-pill"><span class="status-dot {dot_cls}"></span>{status_text}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Data: Rossmann Store Sales (Kaggle) · Model: Random Forest")


# ============================================================
# OVERVIEW
# ============================================================
def render_overview():
    render_section_header(
        "Sales Forecasting",
        subtitle="Analyze historical store performance, evaluate forecasting models, "
        "and generate data-driven sales forecasts.",
        kicker="Machine Learning Powered Retail Demand Intelligence",
    )

    if not pipeline_ready():
        render_empty_state(
            "Pipeline artifacts not found",
            "Run Notebooks 01–07 in order first to generate the cleaned data, trained "
            "models, and forecast this dashboard reads (data/, models/, outputs/).",
        )
        return

    cleaned = get_cleaned_data()
    predictions = get_predictions()

    render_status_row([
        ("Dataset", "Rossmann Store Sales"),
        ("Stores", f"{cleaned['Store'].nunique():,}"),
        ("Historical Records", f"{len(cleaned):,}"),
        ("Model", "Random Forest"),
        ("Forecast Status", "Ready"),
    ])

    # ---- KPIs ----
    rf_model = get_rf_model()
    X_val, y_val, feature_cols = get_validation_data()
    r2 = r2_score(y_val, rf_model.predict(X_val))

    open_df = cleaned[cleaned["Open"] == 1]
    type_avg = open_df.groupby("StoreType")["Sales"].mean()
    best_type = type_avg.idxmax()

    pred_stores = predictions["Store"].unique()
    pred_open = predictions[predictions["Open"] == 1]
    avg_forecast = pred_open["Sales"].mean()
    horizon_days = pred_open["Date"].nunique()
    last_date = cleaned["Date"].max()
    trailing = cleaned[
        (cleaned["Date"] > last_date - pd.Timedelta(days=horizon_days))
        & (cleaned["Store"].isin(pred_stores))
        & (cleaned["Open"] == 1)
    ]
    avg_actual_trailing = trailing["Sales"].mean()
    growth_pct = (avg_forecast / avg_actual_trailing - 1) * 100 if avg_actual_trailing else 0.0

    render_kpi_row([
        dict(label="Total Sales", value=f"€{cleaned['Sales'].sum() / 1e6:.1f}M", context="Full historical period"),
        dict(label="Model R²", value=f"{r2:.1%}", context="Random Forest, validation set"),
        dict(label="Best Performing Category", value=f"Type {best_type.upper()}",
             context=f"€{type_avg.max():,.0f} avg / open day"),
        dict(label="Forecast Growth", value=f"{growth_pct:+.1f}%", context="vs. trailing actuals",
             delta="vs recent history", positive=growth_pct >= 0),
    ])

    # ---- dominant chart: Actual vs Forecast ----
    daily_actual = (
        cleaned[cleaned["Store"].isin(pred_stores)].groupby("Date")["Sales"].sum().reset_index()
    )
    tail_actual = daily_actual[daily_actual["Date"] >= last_date - pd.Timedelta(days=120)]
    daily_forecast = predictions.groupby("Date")["Sales"].sum().reset_index()

    with render_chart_container(
        "Actual vs. Forecast",
        f"Combined daily sales for the {len(pred_stores)} stores covered by the current forecast window.",
    ):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tail_actual["Date"], y=tail_actual["Sales"], name="Actual",
                                  mode="lines", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=daily_forecast["Date"], y=daily_forecast["Sales"], name="Forecast",
                                  mode="lines", line=dict(color=ACCENT, width=2, dash="dash")))
        fig.add_vline(x=last_date, line=dict(color=BORDER, width=1, dash="dot"))
        style_fig(fig, height=440)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        render_subsection("Project Lifecycle")
        st.markdown("""
- ✅ Data Understanding & Cleaning
- ✅ Exploratory Data Analysis
- ✅ Feature Engineering (with a leakage audit)
- ✅ Model Training — Linear Regression & Random Forest
- ✅ Model Evaluation & Comparison
- ✅ Forecasting — recursive vs. frozen-history, backtested
- ✅ Interactive Dashboard *(you are here)*
        """)
    with col_right:
        render_subsection("Key Design Decisions")
        render_insight_card(
            "Product analysis proxy",
            "This dataset has no SKU-level data — <b>StoreType</b> / <b>Assortment</b> "
            "stand in for product analysis throughout.",
        )
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        render_insight_card(
            "Forecast strategy",
            "Frozen-history lag features, chosen after backtesting beat a naive "
            "recursive approach (R² 0.73 vs. 0.21).",
        )


# ============================================================
# DATA
# ============================================================
def render_data():
    render_section_header("Data", subtitle="Dataset composition, quality, and a preview of the cleaned records.")

    sample_path = DATA_SAMPLE_DIR / "sales_sample.csv"
    if _missing(sample_path):
        render_empty_state("Sample dataset not found", f"Expected a file at {sample_path}.")
        return

    sample = get_sample_data()
    have_cleaned = not _missing(OUTPUTS_DIR / "predictions.csv")

    render_subsection("Dataset Summary")
    if have_cleaned:
        cleaned = get_cleaned_data()
        num_cols = cleaned.select_dtypes(include="number").columns
        cat_cols = cleaned.select_dtypes(exclude="number").columns
        render_kpi_row([
            dict(label="Rows", value=f"{len(cleaned):,}"),
            dict(label="Columns", value=f"{cleaned.shape[1]}"),
            dict(label="Stores", value=f"{cleaned['Store'].nunique():,}"),
            dict(label="Date Range",
                 value=f"{cleaned['Date'].min():%b %Y} – {cleaned['Date'].max():%b %Y}"),
        ])
        st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
        render_kpi_row([
            dict(label="Missing Values", value=f"{int(cleaned.isna().sum().sum()):,}"),
            dict(label="Numerical Features", value=f"{len(num_cols)}"),
            dict(label="Categorical Features", value=f"{len(cat_cols)}"),
            dict(label="Duplicate Rows", value=f"{int(cleaned.duplicated().sum()):,}"),
        ])

        render_subsection("Data Quality", "Computed live from the full cleaned dataset.")
        anomaly_open_zero = int(((cleaned["Open"] == 1) & (cleaned["Sales"] == 0)).sum())
        anomaly_closed_sales = int(((cleaned["Open"] == 0) & (cleaned["Sales"] > 0)).sum())
        c1, c2 = st.columns(2)
        with c1:
            report = missing_report(cleaned, "cleaned dataset")
            with render_chart_container("Missing Values by Column"):
                if report.empty:
                    st.success("No missing values in the cleaned dataset.")
                else:
                    st.dataframe(report, width="stretch", hide_index=True)
                st.caption(
                    "Remaining gaps in `Promo2SinceWeek/Year` and `PromoInterval` are structurally "
                    "'not applicable' when `Promo2 == 0` — not data quality defects."
                )
        with c2:
            with render_chart_container("Potential Anomalies"):
                render_insight_card(
                    "Open with zero sales",
                    f"<b>{anomaly_open_zero:,}</b> rows are flagged open with €0 in sales.",
                )
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                render_insight_card(
                    "Closed with recorded sales",
                    f"<b>{anomaly_closed_sales:,}</b> rows are flagged closed but show sales > €0.",
                )
    else:
        render_empty_state(
            "Full dataset quality report unavailable",
            "Run the pipeline notebooks first to see row counts, missing-value, and anomaly checks "
            "for the complete cleaned dataset. The sample preview below still works.",
        )

    render_subsection("Inspect the Data")
    with st.expander("Preview rows, missing values, and statistics", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Dataset Preview", "Missing Values", "Statistics"])
        with tab1:
            st.markdown(f"Random sample ({len(sample):,} rows) of the cleaned, merged dataset.")
            st.dataframe(sample.head(50), width="stretch")
        with tab2:
            if have_cleaned:
                cleaned = get_cleaned_data()
                report = missing_report(cleaned, "cleaned dataset")
                if report.empty:
                    st.success("No missing values.")
                else:
                    st.dataframe(report, width="stretch")
            else:
                st.warning("Run the pipeline notebooks first to see the full missing-value report.")
        with tab3:
            st.markdown("Descriptive statistics — numeric columns only (sample):")
            numeric_sample = sample.select_dtypes(include="number")
            st.dataframe(numeric_sample.describe(), width="stretch")


# ============================================================
# SALES ANALYTICS
# ============================================================
def render_sales_analytics():
    render_section_header("Sales Analytics", subtitle="What happened to sales, historically?")

    if _missing(OUTPUTS_DIR / "predictions.csv") or _missing(MODELS_DIR / "random_forest.pkl"):
        render_empty_state("Analytics unavailable", "Run Notebooks 01–02 first to generate the cleaned dataset this page needs.")
        return

    cleaned = get_cleaned_data()
    open_df = cleaned[cleaned["Open"] == 1]

    tab_trend, tab_patterns, tab_stores = st.tabs(["Trends", "Patterns & Promotions", "Store Performance"])

    with tab_trend:
        daily_total = cleaned.groupby("Date")["Sales"].sum().reset_index()
        daily_total["Rolling30"] = daily_total["Sales"].rolling(30, min_periods=1).mean()
        with render_chart_container("Sales Trend", "Daily total sales with a 30-day rolling average."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_total["Date"], y=daily_total["Sales"], name="Daily total",
                                      line=dict(color=BORDER, width=1)))
            fig.add_trace(go.Scatter(x=daily_total["Date"], y=daily_total["Rolling30"], name="30-day avg",
                                      line=dict(color=BLUE, width=2.5)))
            style_fig(fig, height=380)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

        daily_total["Month"] = daily_total["Date"].dt.month
        monthly = daily_total.groupby("Month")["Sales"].mean().reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["MonthName"] = monthly["Month"].apply(lambda m: month_names[m - 1])
        dec_share = cleaned.groupby(cleaned["Date"].dt.month)["Sales"].sum()
        dec_pct = dec_share.get(12, 0) / dec_share.sum() * 100
        with render_chart_container("Monthly Pattern", f"December accounts for {dec_pct:.1f}% of total annual sales."):
            fig = px.bar(monthly, x="MonthName", y="Sales", color_discrete_sequence=[BLUE])
            style_fig(fig, height=340, show_legend=False)
            fig.update_layout(xaxis_title="", yaxis_title="Avg Daily Sales")
            st.plotly_chart(fig, width="stretch")

    with tab_patterns:
        col1, col2 = st.columns(2)
        with col1:
            dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
            dow = open_df.groupby("DayOfWeek")["Sales"].mean().reset_index()
            dow["Day"] = dow["DayOfWeek"].map(dow_names)
            with render_chart_container("Weekly Pattern", "Average sales on open-store days."):
                fig = px.bar(dow, x="Day", y="Sales", color_discrete_sequence=[ACCENT])
                style_fig(fig, height=340, show_legend=False)
                fig.update_layout(xaxis_title="", yaxis_title="Avg Sales (Open Stores)")
                st.plotly_chart(fig, width="stretch")
                st.caption("Sunday's small sample (<0.5% of open-store days) makes it unreliable.")

        with col2:
            promo_avg = open_df.groupby("Promo")["Sales"].mean()
            lift_pct = (promo_avg.get(1, 0) / promo_avg.get(0, 1) - 1) * 100 if 0 in promo_avg.index else 0
            promo_plot = promo_avg.reset_index()
            promo_plot["Label"] = promo_plot["Promo"].map({0: "No Promotion", 1: "Promotion"})
            with render_chart_container("Promotion Impact", f"Promotional days show a computed {lift_pct:+.0f}% average sales lift."):
                fig = px.bar(promo_plot, x="Label", y="Sales", color="Label",
                             color_discrete_map={"No Promotion": BLUE, "Promotion": ACCENT})
                style_fig(fig, height=340, show_legend=False)
                fig.update_layout(xaxis_title="", yaxis_title="Avg Sales")
                st.plotly_chart(fig, width="stretch")

    with tab_stores:
        col3, col4 = st.columns(2)
        with col3:
            storetype = open_df.groupby("StoreType")["Sales"].mean().reset_index()
            with render_chart_container("Product Performance", "StoreType / Assortment proxy — see Overview for the design rationale."):
                fig = px.bar(storetype, x="StoreType", y="Sales", color_discrete_sequence=[BLUE])
                style_fig(fig, height=320, show_legend=False)
                fig.update_layout(yaxis_title="Avg Sales")
                st.plotly_chart(fig, width="stretch")

        with col4:
            top10 = open_df.groupby("Store")["Sales"].mean().sort_values(ascending=False).head(10).reset_index()
            top10["Store"] = top10["Store"].astype(str)
            with render_chart_container("Store Performance", "Top 10 stores by average sales."):
                fig = px.bar(top10, x="Sales", y="Store", orientation="h", color_discrete_sequence=[ACCENT])
                style_fig(fig, height=320, show_legend=False)
                fig.update_layout(yaxis=dict(autorange="reversed", gridcolor=BORDER))
                st.plotly_chart(fig, width="stretch")


# ============================================================
# MACHINE LEARNING
# ============================================================
def render_machine_learning():
    render_section_header("Machine Learning", subtitle="How the forecasting model performs, and what drives its predictions.")

    if _missing(MODELS_DIR / "random_forest.pkl") or _missing(MODELS_DIR / "linear_regression.pkl"):
        render_empty_state("Models unavailable", "Run Notebook 05 first to train and save the models.")
        return

    rf_model = get_rf_model()
    lr_model = get_lr_model()
    X_val, y_val, feature_cols = get_validation_data()

    render_subsection("Selected Model: Random Forest", "Chosen over Linear Regression — outperforms it on every metric below.")

    preds_rf = rf_model.predict(X_val)
    preds_lr = lr_model.predict(X_val)

    def _metrics(y_true, y_pred):
        return dict(
            MAE=mean_absolute_error(y_true, y_pred),
            RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
            R2=r2_score(y_true, y_pred),
        )

    m_rf, m_lr = _metrics(y_val, preds_rf), _metrics(y_val, preds_lr)

    render_subsection("Model Performance")
    c1, c2, c3 = st.columns(3)
    render_metric_card(c1, "MAE (Random Forest)", f"€{m_rf['MAE']:.0f}",
                        delta=f"{m_rf['MAE'] - m_lr['MAE']:.0f} vs LR", delta_color="inverse")
    render_metric_card(c2, "RMSE (Random Forest)", f"€{m_rf['RMSE']:.0f}",
                        delta=f"{m_rf['RMSE'] - m_lr['RMSE']:.0f} vs LR", delta_color="inverse")
    render_metric_card(c3, "R² (Random Forest)", f"{m_rf['R2']:.1%}",
                        delta=f"{(m_rf['R2'] - m_lr['R2']) * 100:.1f} pts vs LR")

    col_a, col_b = st.columns(2)
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)

    with col_a:
        name_lower = importances.index.str.lower()
        lag_share = importances[name_lower.str.contains("lag") | name_lower.str.contains("roll")].sum() / importances.sum() * 100
        promo_share = importances[name_lower.str.contains("promo")].sum() / importances.sum() * 100
        comp_share = importances[name_lower.str.contains("competition")].sum() / importances.sum() * 100
        explanation_parts = []
        if lag_share > 15:
            explanation_parts.append(f"recent sales history (lag/rolling features) makes up {lag_share:.0f}% of the top-12 importance")
        if promo_share > 5:
            explanation_parts.append(f"promotion-related features contribute {promo_share:.0f}%")
        if comp_share > 5:
            explanation_parts.append(f"competition-distance features contribute {comp_share:.0f}%")
        explanation = "What Drives Sales? " + ("; ".join(explanation_parts) + "." if explanation_parts else
                                                 "Importance is spread relatively evenly across the top features.")
        with render_chart_container("What Drives Sales?", explanation):
            fig = px.bar(x=importances.values[::-1], y=importances.index[::-1], orientation="h",
                         color_discrete_sequence=[BLUE])
            style_fig(fig, height=420, show_legend=False)
            fig.update_layout(xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

    with col_b:
        sample_idx = np.random.RandomState(42).choice(len(y_val), size=min(3000, len(y_val)), replace=False)
        with render_chart_container("Actual vs Predicted", "Validation set, 3,000-point sample."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_val.values[sample_idx], y=preds_rf[sample_idx], mode="markers",
                marker=dict(color=ACCENT, size=4, opacity=0.4), name="Predictions"
            ))
            max_val = max(y_val.max(), preds_rf.max())
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                      line=dict(color=TEXT_MUTED, dash="dash"), name="Perfect prediction"))
            style_fig(fig, height=420)
            fig.update_layout(xaxis_title="Actual Sales", yaxis_title="Predicted Sales")
            st.plotly_chart(fig, width="stretch")

    with st.expander("Technical Details — Model Configuration"):
        rf_params = rf_model.get_params()
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"""
**Selected model:** Random Forest Regressor
**Number of features:** {len(feature_cols)}
**n_estimators:** {rf_params.get('n_estimators', '—')}
**max_depth:** {rf_params.get('max_depth', '—')}
            """)
        with d2:
            st.markdown(f"""
**Training approach:** supervised regression on engineered lag/rolling and
categorical features, one-hot encoded with `drop_first=True` to avoid the
dummy-variable trap.
**Validation approach:** chronological split — the model is validated on the
most recent slice of dates only, never a random shuffle, to avoid time
leakage from future information into training.
**Validation rows:** {len(y_val):,}
            """)


# ============================================================
# FORECAST
# ============================================================
def render_forecast():
    render_section_header("Sales Forecast", subtitle="Predict future retail demand using the trained forecasting model.")

    if _missing(OUTPUTS_DIR / "predictions.csv"):
        render_empty_state("Forecast unavailable", "Run Notebook 07 first to generate outputs/predictions.csv.")
        return

    predictions = get_predictions()
    cleaned = get_cleaned_data()
    store_meta = cleaned[["Store", "StoreType", "Assortment"]].drop_duplicates(subset="Store")

    st.caption(
        "Forecast uses a frozen-history strategy for lag features, validated against a "
        "fully recursive approach in Notebook 07 (R² 0.73 vs 0.21) — see Business Insights."
    )

    horizon_label = st.segmented_control("Forecast Horizon", ["7 Days", "14 Days", "30 Days"], default="7 Days")
    horizon_days = {"7 Days": 7, "14 Days": 14, "30 Days": 30}.get(horizon_label, 7)

    col_filter, col_select = st.columns(2)
    with col_filter:
        store_types = ["All"] + sorted(store_meta["StoreType"].unique().tolist())
        selected_type = st.selectbox("Filter by Product Category (Store Type proxy)", store_types)

    available_stores = predictions["Store"].unique()
    eligible = store_meta[store_meta["Store"].isin(available_stores)]
    if selected_type != "All":
        eligible = eligible[eligible["StoreType"] == selected_type]

    with col_select:
        selected_store = st.selectbox("Select Store", sorted(eligible["Store"].unique().tolist()))

    store_info = store_meta[store_meta["Store"] == selected_store].iloc[0]
    st.markdown(
        f"**Store {selected_store}** — Type `{store_info['StoreType']}`, Assortment `{store_info['Assortment']}`"
    )

    store_forecast = predictions[predictions["Store"] == selected_store].sort_values("Date")
    open_forecast = store_forecast[store_forecast["Open"] == 1]

    if open_forecast.empty:
        render_empty_state("No open-day forecasts", "This store has no open-day forecasts in the current window.")
        return

    first_date = open_forecast["Date"].min()
    horizon_window = open_forecast[open_forecast["Date"] <= first_date + pd.Timedelta(days=horizon_days - 1)]

    store_hist = cleaned[(cleaned["Store"] == selected_store) & (cleaned["Open"] == 1)].sort_values("Date")
    trailing = store_hist.tail(len(horizon_window))
    avg_trailing = trailing["Sales"].mean() if not trailing.empty else np.nan
    avg_forecast_store = horizon_window["Sales"].mean()
    growth = (avg_forecast_store / avg_trailing - 1) * 100 if avg_trailing else 0.0

    hi_row = horizon_window.loc[horizon_window["Sales"].idxmax()]
    lo_row = horizon_window.loc[horizon_window["Sales"].idxmin()]

    render_subsection("Forecast Summary", f"{horizon_label} window starting {first_date:%d %b %Y}.")
    render_kpi_row([
        dict(label="Forecasted Sales", value=f"€{horizon_window['Sales'].sum():,.0f}", context=f"Sum over {horizon_label.lower()}"),
        dict(label="Average Forecast", value=f"€{avg_forecast_store:,.0f}", context="Per open day"),
        dict(label="Highest Forecast Day", value=f"€{hi_row['Sales']:,.0f}", context=f"{hi_row['Date']:%d %b}"),
        dict(label="Lowest Forecast Day", value=f"€{lo_row['Sales']:,.0f}", context=f"{lo_row['Date']:%d %b}"),
    ])
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    growth_col = st.columns(4)[0]
    render_kpi_card(
        growth_col, label="Expected Growth", value=f"{growth:+.1f}%",
        context="vs. same-length trailing actuals",
        delta="vs recent history", positive=growth >= 0,
    )

    with render_chart_container("Forecasted Sales", "Recent actual sales followed by the forecast window. Dips to zero reflect predicted closed days, not forecast error."):
        recent_actual = store_hist[store_hist["Date"] >= first_date - pd.Timedelta(days=60)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent_actual["Date"], y=recent_actual["Sales"], name="Actual",
                                  mode="lines+markers", line=dict(color=BLUE), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=store_forecast["Date"], y=store_forecast["Sales"], name="Forecast",
                                  mode="lines+markers", line=dict(color=ACCENT), marker=dict(size=4)))
        fig.add_vline(x=first_date, line=dict(color=BORDER, width=1, dash="dot"))
        style_fig(fig, height=400)
        fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Sales")
        st.plotly_chart(fig, width="stretch")


# ============================================================
# BUSINESS INSIGHTS
# ============================================================
def render_business_insights():
    render_section_header("Business Insights", subtitle="Translating the analysis and the model's output into business language.")

    if _missing(OUTPUTS_DIR / "predictions.csv"):
        render_empty_state("Insights unavailable", "Run the pipeline notebooks first — this page reads the same cleaned data as the rest of the platform.")
        return

    cleaned = get_cleaned_data()
    open_df = cleaned[cleaned["Open"] == 1]

    dow_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
    dow_avg = open_df.groupby("DayOfWeek")["Sales"].mean()
    best_dow = dow_names[dow_avg.idxmax()]
    weakest_reliable_dow = dow_names[dow_avg.drop(index=7, errors="ignore").idxmin()]

    monthly = cleaned.groupby(cleaned["Date"].dt.month)["Sales"].sum()
    dec_pct = monthly.get(12, 0) / monthly.sum() * 100

    promo_avg = open_df.groupby("Promo")["Sales"].mean()
    lift_pct = (promo_avg.get(1, 0) / promo_avg.get(0, 1) - 1) * 100 if 0 in promo_avg.index else 0

    best_storetype = open_df.groupby("StoreType")["Sales"].mean().idxmax()

    col1, col2 = st.columns(2)
    with col1:
        render_insight_card(
            "Demand Trend",
            f"Revenue is <b>seasonally stable</b> rather than structurally growing or shrinking. "
            f"December alone accounts for <b>{dec_pct:.1f}%</b> of annual sales — the genuine "
            f"seasonal peak — while the rest of the year is fairly flat once seasonally adjusted.",
        )
    with col2:
        render_insight_card(
            "Peak Demand",
            f"<b>{best_dow}</b> is the strongest reliable weekday for sales; <b>{weakest_reliable_dow}</b> "
            f"is actually the softest of the reliably-sampled days — weekends are not stronger for "
            f"sales at this company.",
        )

    col3, col4 = st.columns(2)
    with col3:
        render_insight_card(
            "Store Performance",
            f"Store <b>Type {best_storetype.upper()}</b> leads on average sales per open day. "
            f"No SKU-level data exists in this dataset — <b>StoreType</b> / <b>Assortment</b> serve as "
            f"a documented proxy for product analysis throughout.",
        )
    with col4:
        render_insight_card(
            "Promotion Effect",
            f"Same-day promotions carry a computed <b>{lift_pct:+.0f}%</b> average sales lift. "
            f"<b>Promo2</b> enrollment shows lower average sales — most consistent with a selection "
            f"effect (struggling stores opt in) rather than the program itself hurting sales.",
        )

    render_subsection("Forecast Signal")
    render_insight_card(
        "What the model expects",
        "A naive recursive forecast (feeding predictions back in as 'history') collapsed accuracy "
        "from <b>R²=0.889 to R²≈0.21</b> — caught by backtesting before shipping. The "
        "<b>frozen-history approach</b> (holding lag features at last-known real values) recovered "
        "most of that accuracy (<b>R²≈0.73</b>) and is what powers the Forecast page. Confidence "
        "should be read as narrowing the further out you look — a measured limitation, not hidden.",
    )

    render_subsection("From Modeling")
    st.markdown("""
- **Random Forest (R²=88.9%) outperforms Linear Regression (R²=82.8%)** on every metric —
  consistent with genuinely non-linear, interactive retail sales patterns.
- Prediction error is **~3x wider for high-volume days** — forecast uncertainty should
  scale with the size of the prediction, not be communicated as a flat number.
- **77% of the model's predictive power comes from recent sales history**
  (lag/rolling features) — this model will be markedly less reliable for brand-new
  stores with no sales history.
    """)


# ============================================================
# ABOUT
# ============================================================
def render_about():
    render_section_header("About This Project", subtitle="Sales Forecasting & Business Analytics Platform")
    st.markdown("""
An end-to-end data science portfolio project built on the
[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) dataset (Kaggle).
    """)

    col1, col2 = st.columns(2)
    with col1:
        render_insight_card(
            "Tech Stack",
            "Python 3.12 · pandas, numpy · matplotlib, seaborn, Plotly · "
            "scikit-learn (Linear Regression, Random Forest) · Streamlit · joblib",
        )
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        render_insight_card(
            "Methodology",
            "Data Understanding → Cleaning → EDA → Feature Engineering (with a leakage audit) → "
            "Model Training → Model Evaluation → Forecasting (recursive vs. frozen-history, "
            "backtested) → this Dashboard.",
        )
    with col2:
        render_insight_card(
            "Known Limitations",
            "No SKU/product-level data — StoreType/Assortment used as a documented proxy. Raw "
            "Store ID is excluded from modeling (avoids 1,115 one-hot columns) — two stores with "
            "identical characteristics receive identical predictions. Multi-week forecasts are "
            "measurably less accurate than same-day predictions (R² 0.73 vs 0.889).",
        )
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        render_insight_card(
            "Links",
            '<a href="https://www.kaggle.com/competitions/rossmann-store-sales" target="_blank">'
            "Dataset source</a> · Project repository: add your GitHub URL here",
        )


# ============================================================
# Router
# ============================================================
PAGES = {
    "Overview": render_overview,
    "Data": render_data,
    "Sales Analytics": render_sales_analytics,
    "Machine Learning": render_machine_learning,
    "Forecast": render_forecast,
    "Business Insights": render_business_insights,
    "About": render_about,
}

render_sidebar()
PAGES[st.session_state.page]()