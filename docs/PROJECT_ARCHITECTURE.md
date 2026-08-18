# Project Architecture & Roadmap

**Status:** Approved (Product Analysis proxy decision confirmed in Phase 2).

This document is the senior-architect-level analysis of the PRD, produced before
pipeline code was written. It covers: roadmap, milestones, folder architecture,
dataset requirements, ML pipeline design, dashboard architecture, git strategy,
risks, timeline, and definition of done per milestone.

## 1. Project Roadmap

| Phase | Name | Status |
|---|---|---|
| 0 | PRD & Master Prompt review | Done |
| 1 | Environment & folder scaffolding | Done |
| 2 | Data Understanding (Notebook 01) | Done |
| 3 | Data Cleaning (Notebook 02) | Done |
| 4 | EDA — 12 business-driven visualizations (Notebook 03) | Done |
| 5 | Feature Engineering (Notebook 04) | Done |
| 6 | Model Training (Notebook 05) | Done |
| 7 | Model Evaluation & Comparison (Notebook 06) | Done |
| 8 | Forecast Module (Notebook 07) | Done |
| 9 | Streamlit Dashboard | Done |
| 10 | Documentation, screenshots, final repo polish | Done |

## 2. Development Milestones

M1 Scaffold -> M2 Data Validated -> M3 Data Cleaned -> M4 EDA Complete ->
M5 Features Engineered -> M6 Models Trained -> M7 Models Evaluated ->
M8 Forecasts Generated -> M9 Dashboard Live -> M10 Repo Portfolio-Ready.

## 3. Folder Architecture

```
Sales-Forecasting/
├── data/{raw,processed,sample}/
├── notebooks/01_...07_...ipynb
├── src/{data_loader,preprocessing,feature_engineering,model,forecasting,utils,config}.py
├── dashboard/app.py
├── models/
├── outputs/{predictions.csv,charts/,reports/}
├── docs/
├── screenshots/
├── requirements.txt, .gitignore, README.md, LICENSE
```

Principle: notebooks and the dashboard both import from `src/` — one source of
truth for loading, cleaning, feature engineering, and model logic. `src/config.py`
holds all path constants to avoid hardcoded paths anywhere in the codebase.

## 4. Dataset Requirements — Rossmann Store Sales

- `train.csv`: Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
- `test.csv`: same minus Sales/Customers, plus Id
- `store.csv`: Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSince{Month,Year}, Promo2, Promo2Since{Week,Year}, PromoInterval

**Key architectural decision (approved):** Rossmann has no SKU/product-level
data. `StoreType` and `Assortment` serve as a documented proxy for the PRD's
"Product Analysis" requirement throughout EDA and the dashboard.

Known data-quality findings, verified in Notebook 01/02: `StateHoliday` mixed
dtype (resolved by `low_memory=False`, verified not to recur when reading raw
files, but DOES recur on naive re-reads of processed CSVs — see
`load_cleaned_data()`), `Sales==0` when `Open==0` (tautological, excluded from
modeling), `Promo2`-related NaNs are "not applicable" not "missing" (544 stores
with `Promo2==0`), 3 stores missing `CompetitionDistance`, 354 stores missing
`CompetitionOpenSinceMonth/Year`, `PromoInterval` spells September as "Sept"
(not standard "Sep" — a real parsing gotcha caught and handled in Notebook 04).

## 5. ML Pipeline

`data_loader -> preprocessing (merge, dtype fix, filter Open==1 at modeling
time) -> feature_engineering (date + business features, leakage audit) ->
chronological train/validation split (last 42 days held out) -> model.py
(Linear Regression + Random Forest, one-hot encoding with drop_first=True) ->
evaluation (MAE/MSE/RMSE/R² + diagnostics) -> joblib serialization ->
forecasting.py (frozen-history strategy, chosen after backtesting beat a fully
recursive approach) -> outputs/predictions.csv -> dashboard (reads artifacts,
never retrains live)`

**Feature-leakage audit (Notebook 04):** `Customers` (strongest raw correlate
with Sales, r=0.82) is not present in `test.csv` and is excluded from
modeling. `Suspicious_Zero_Sales` is derived from `Sales` itself and is also
excluded. Both are enforced via `LEAKAGE_EXCLUDED_COLUMNS` in
`src/feature_engineering.py`, not just documented.

**Forecasting strategy (Notebook 07):** a fully recursive walk-forward
approach was tested first and found to collapse validation accuracy from
R²=0.889 to R²≈0.21, because ~77% of Random Forest's predictive power comes
from lag/rolling features and compounding predictions as "history" snowballs
error. A frozen-history approach (holding lag features at last-known real
values, letting only day-varying features like Promo/holidays change) was
backtested and recovered accuracy to R²≈0.73 — the approach used in production.

## 6. Dashboard Architecture

Streamlit, sidebar navigation: Overview, Data, Sales Analytics, Machine
Learning, Forecast, Business Insights, About. `@st.cache_data` /
`@st.cache_resource` for performance. Dashboard consumes pre-trained model +
pre-generated predictions/cleaned data; no live retraining or re-cleaning.
Every page checks for required artifacts and shows a clear message (not a
crash) if the pipeline hasn't been run yet.

## 7. Git Strategy

Milestone-scoped imperative commit messages. Trunk-based on `main`. Raw data,
models, and generated outputs remain gitignored; `data/sample/` is committed
for repo browsability without the full Kaggle download.

## 8. Risks and Challenges (resolved)

No product-level data (mitigated via documented proxy) · StateHoliday dtype
trap (resolved, and re-verified not to silently recur on CSV round-trips) ·
Open==0 handling (deferred to modeling time as planned) · recursive forecast
compounding (caught via backtest, fixed with frozen-history) · one-hot
"dummy variable trap" in `encode_features` (caught while building model
interpretation, fixed with `drop_first=True`, models retrained) · single-core
sandbox compute constraints (addressed via tuned Random Forest hyperparameters,
documented as a real trade-off).

## 9. Timeline

All 10 phases complete, built incrementally with verification (execution,
AppTest, or backtesting) at every phase before moving to the next.

## 10. Definition of Done — met for every milestone

Each milestone required: working, executed code committed; notebook markdown
fully explaining concept/output/business insight; every claim verified against
real computed output before being written; no PRD requirement silently
skipped; every discovered bug fixed and disclosed rather than hidden.
