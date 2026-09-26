# Property Price Estimator

> An end-to-end machine learning project for estimating residential property prices from property characteristics and location-level signals.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-brightgreen)

## 1. Executive Summary

**Business problem:** Property buyers, sellers, analysts, and real-estate teams need a consistent first-pass estimate of a property's market value before conducting a detailed valuation.

**Objective:** Build a reproducible regression pipeline that predicts `price_crore` from property attributes such as area, bedrooms, bathrooms, location, age, amenities, and distance to metro.

**Primary success metrics**
- **RMSE** — penalizes larger pricing errors.
- **MAE** — expresses the typical absolute error in the target's units.
- **R²** — measures explained variance.

**Modeling strategy**
1. Establish a simple Linear Regression baseline.
2. Build a nonlinear tree-based model.
3. Compare models on an untouched test set.
4. Inspect residuals and segment-level errors.
5. Save the final preprocessing + model pipeline for inference.

> **Data note:** The included dataset is synthetic and was generated for portfolio/learning purposes. It is intentionally shaped like a Dhaka residential-property dataset; it is **not** a record of actual market transactions and should not be used for real financial decisions.

## 2. Project Goals

- Produce a clean, reproducible ML workflow.
- Demonstrate disciplined data auditing and leakage prevention.
- Explain which property characteristics are associated with price.
- Deliver a reusable prediction pipeline.
- Provide a lightweight Streamlit demo for stakeholders.

## 3. Repository Structure

```text
property-price-estimator/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── property_prices.csv
│   └── processed/
├── docs/
│   ├── data_dictionary.md
│   ├── methodology.md
│   └── model_card.md
├── models/
├── notebooks/
│   ├── 01_problem_and_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_model_training_and_evaluation.ipynb
│   └── 06_final_report.ipynb
├── reports/
│   ├── figures/
│   └── final_report.md
├── src/
│   └── property_price_estimator/
│       ├── data.py
│       ├── features.py
│       ├── modeling.py
│       └── evaluation.py
├── tests/
│   └── test_pipeline.py
├── slides/
│   └── demo_day_outline.md
├── .gitignore
├── requirements.txt
└── README.md
```

## 4. Quick Start

### Windows

```bash
git clone <your-github-repo-url>
cd property-price-estimator

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python -m src.property_price_estimator.modeling
streamlit run app/streamlit_app.py
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m src.property_price_estimator.modeling
streamlit run app/streamlit_app.py
```

## 5. Notebook Workflow

| Notebook | Purpose |
|---|---|
| `01_problem_and_data_audit` | Frame the business problem, define target, inspect schema, missingness, duplicates, and data risks. |
| `02_eda` | Build purposeful charts and document what drives price. |
| `03_feature_engineering` | Define transformations and leakage-safe preprocessing. |
| `04_baseline_model` | Establish a transparent Linear Regression baseline. |
| `05_model_training_and_evaluation` | Train improved models, compare metrics, inspect residuals and segment errors. |
| `06_final_report` | Turn technical results into an executive-ready narrative. |

## 6. Key EDA Questions

The analysis should answer:

1. How does property size relate to price?
2. Which locations command higher prices?
3. How does property age affect price?
4. Does distance to metro matter?
5. How do bedrooms and bathrooms relate to value?
6. Do amenities such as lift, security, and parking add signal?
7. Where does the model make its largest errors?

## 7. Modeling Design

### Target

`price_crore`

### Candidate features

- `location`
- `property_type`
- `area_sqft`
- `bedrooms`
- `bathrooms`
- `age_years`
- `floor`
- `total_floors`
- `parking_spaces`
- `has_lift`
- `has_security`
- `distance_to_metro_km`
- `property_condition`

### Leakage controls

The model must only use information available when the estimate is requested. The target itself and any post-sale information are excluded.

The train/test split occurs **before fitting learned preprocessing**. Encoding and imputation are implemented inside a scikit-learn `Pipeline` / `ColumnTransformer`.

## 8. Expected Model Comparison

Do not hard-code a model as "best" before running the notebook. The notebook computes the metrics from the current dataset and records the selected model based on validation performance.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Baseline Linear Regression | generated at run time | generated at run time | generated at run time |
| Random Forest Regressor | generated at run time | generated at run time | generated at run time |
| HistGradientBoosting Regressor | generated at run time | generated at run time | generated at run time |

## 9. Business Interpretation

A model is not a valuation certificate. The output should be treated as a **decision-support estimate**.

For a production deployment, the next steps would include:
- real transaction/listing data,
- time-aware validation,
- neighborhood/geospatial features,
- market-regime variables,
- calibration and prediction intervals,
- monitoring for drift,
- human review for high-value properties.

## 10. Testing

Run:

```bash
pytest -q
```

Tests cover:
- required columns,
- duplicate handling,
- pipeline fitting,
- prediction shape,
- positive/finite predictions.

## 11. AI Usage & Integrity

AI tools may be used for:
- understanding unfamiliar APIs,
- debugging,
- improving documentation structure,
- generating alternative explanations.

Human responsibilities:
- verify every claim,
- understand every submitted code block,
- reproduce the results,
- cite external data sources,
- never present synthetic data as real market data.

## 12. Demo Day

Suggested 5-minute story:

1. Problem and stakeholder
2. Data and risks
3. What the EDA revealed
4. Baseline vs improved model
5. Error analysis
6. Demo of one prediction
7. Limitations and next steps

See `slides/demo_day_outline.md`.

## 13. Author

**Aspiring Researcher & Data Scientist**  
Government Officer | Associate Faculty @ BTI (Research & Consultancy) | Entrepreneurship Development

---
Built as a portfolio-grade applied machine learning case study.
