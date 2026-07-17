# Sales Forecasting & Business Analytics Platform

> End-to-end Machine Learning system that forecasts retail sales and surfaces
> business insights through an interactive Streamlit dashboard.
>
> **Status:** 🚧 Work in progress — built incrementally, phase by phase.

## Project Overview

Retailers struggle to predict demand accurately, which leads to overstocking,
stockouts, and inefficient marketing spend. This project builds a regression-based
forecasting pipeline on real-world retail data (Rossmann Store Sales), covering the
full data science lifecycle: cleaning → EDA → feature engineering → model
training/comparison → evaluation → forecasting → dashboard.

## Dataset

**Source:** [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) (Kaggle)

Because of Kaggle's competition data-redistribution rules, the raw CSVs are **not**
committed to this repo. To reproduce:

1. Create a free Kaggle account and accept the competition rules.
2. Download `train.csv`, `test.csv`, and `store.csv`.
3. Place them in `data/raw/`.

## Tech Stack

- **Language:** Python 3.12+
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **ML:** scikit-learn, xgboost, lightgbm
- **Dashboard:** Streamlit
- **Persistence:** joblib

## Project Structure

```
Sales-Forecasting/
├── data/
│   ├── raw/          # Original Kaggle CSVs (gitignored)
│   ├── processed/     # Cleaned/merged datasets (gitignored)
│   └── sample/        # Small committed sample for quick demo
├── notebooks/          # Numbered, tutorial-style analysis notebooks
├── src/                 # Reusable, importable pipeline modules
├── dashboard/          # Streamlit application
├── models/              # Trained model artifacts (gitignored)
├── outputs/            # Predictions, charts, reports (gitignored)
├── screenshots/        # Dashboard screenshots for this README
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd Sales-Forecasting
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

_To be added as modules are completed._

## Roadmap

- [x] Project scaffolding & environment setup
- [ ] Data understanding & cleaning
- [ ] Exploratory Data Analysis (10+ business-driven visualizations)
- [ ] Feature engineering
- [ ] Model training (Linear Regression, Random Forest, XGBoost, LightGBM)
- [ ] Model evaluation & comparison
- [ ] Forecast generation
- [ ] Streamlit dashboard
- [ ] Screenshots & final documentation polish

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
