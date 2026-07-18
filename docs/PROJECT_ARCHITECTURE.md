# Project Architecture & Roadmap

**Status:** Approved pending sign-off on the Product Analysis proxy decision (see Dataset Requirements).

This document is the senior-architect-level analysis of the PRD, produced before any
pipeline code was written. It covers: roadmap, milestones, folder architecture, dataset
requirements, ML pipeline design, dashboard architecture, git strategy, risks, timeline,
and definition of done per milestone.

## 1. Project Roadmap

| Phase | Name | Status |
|---|---|---|
| 0 | PRD & Master Prompt review | Done |
| 1 | Environment & folder scaffolding | Done |
| 2 | Data Understanding (Notebook 01) | Blocked on dataset upload |
| 3 | Data Cleaning (Notebook 02) | Not started |
| 4 | EDA — 10+ business-driven visualizations (Notebook 03) | Not started |
| 5 | Feature Engineering (Notebook 04) | Not started |
| 6 | Model Training (Notebook 05) | Not started |
| 7 | Model Evaluation & Comparison (Notebook 06) | Not started |
| 8 | Forecast Module (Notebook 07) | Not started |
| 9 | Streamlit Dashboard | Not started |
| 10 | Documentation, screenshots, final repo polish | Not started |

## 2. Development Milestones

M1 Scaffold (done) -> M2 Data Validated -> M3 Data Cleaned -> M4 EDA Complete ->
M5 Features Engineered -> M6 Models Trained -> M7 Models Evaluated ->
M8 Forecasts Generated -> M9 Dashboard Live -> M10 Repo Portfolio-Ready.

See main chat discussion (or project history) for full milestone descriptions.

## 3. Folder Architecture

```
Sales-Forecasting/
├── data/{raw,processed,sample}/
├── notebooks/01_...07_...ipynb
├── src/{data_loader,preprocessing,feature_engineering,model,evaluation,forecasting,utils,config}.py
├── dashboard/app.py
├── models/
├── outputs/{predictions.csv,charts/,reports/}
├── docs/
├── screenshots/
├── requirements.txt, .gitignore, README.md, LICENSE
```

Principle: notebooks import from `src/`, never redefine logic. `src/config.py` will
hold path constants to avoid hardcoded paths anywhere in the codebase.

## 4. Dataset Requirements — Rossmann Store Sales

- `train.csv`: Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
- `test.csv`: same minus Sales/Customers, plus Id
- `store.csv`: Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSince{Month,Year}, Promo2, Promo2Since{Week,Year}, PromoInterval

**Key architectural decision (pending approval):** Rossmann has no SKU/product-level
data. `StoreType` and `Assortment` will serve as a documented proxy for the PRD's
"Product Analysis" requirement.

Known data-quality considerations to verify on ingestion: `StateHoliday` mixed dtype,
`Sales==0` when `Open==0`, `Promo2`-related NaNs are "not applicable" not "missing",
possible missing `CompetitionDistance` for a few stores.

## 5. ML Pipeline

`data_loader -> preprocessing (merge, dtype fix, filter Open==0) -> feature_engineering
(date + business features) -> chronological train/validation split -> model.py
(Linear Regression + Random Forest mandatory, XGBoost/LightGBM optional) ->
evaluation.py (MAE/MSE/RMSE/R² + diagnostics) -> joblib serialization ->
forecasting.py -> outputs/predictions.csv -> dashboard (reads artifacts, never retrains live)`

Time-aware (chronological) split is mandatory per PRD — prevents future-data leakage
inherent to random shuffling of time series.

## 6. Dashboard Architecture

Streamlit, sidebar navigation: Home, Data Overview, Sales Analytics, Machine Learning,
Forecast, Insights, About (About added per Master Prompt, non-conflicting with PRD).
`@st.cache_data` / `@st.cache_resource` for performance. Dashboard consumes pre-trained
model + pre-generated predictions; no live retraining.

## 7. Git Strategy

Milestone-scoped imperative commit messages per PRD examples. Trunk-based on `main`.
Tag `v0.1` after core pipeline works end-to-end, `v1.0` at project completion.
Raw data, models, and generated outputs remain gitignored.

## 8. Risks and Challenges

No product-level data (mitigated via documented proxy) · StateHoliday dtype trap ·
Open==0 handling · limited ~2.5yr history weakens YoY seasonality claims ·
tree-model overfitting risk · optional-feature scope creep · Streamlit caching
required for performance.

## 9. Timeline (session-based, not calendar deadlines)

Scaffold done · Data Understanding 1 · Cleaning 1-2 · EDA 2-3 · Feature Engineering 1-2
· Model Training 1-2 · Evaluation 1 · Forecasting 1 · Dashboard 2-3 · Polish 1.
Total ≈ 12-18 sessions.

## 10. Definition of Done — see per-milestone criteria in main project discussion.
Each milestone requires: working code committed, notebook markdown fully explaining
concept/output/business insight, and no PRD requirement silently skipped.
