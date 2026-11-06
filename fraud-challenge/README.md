# Explainable Fraud Detection

An end-to-end fraud-detection project for the Sparkov credit-card transaction
dataset. The project includes exploratory analysis, leakage-safe feature
engineering, supervised and unsupervised models, cost-based thresholding,
SHAP explanations, a Streamlit dashboard, and a FastAPI service.

The pipeline scripts are designed to be run in order:

1. Load and explore the data.
2. Build features and chronological train/validation/test splits.
3. Compare imbalance strategies and train models.
4. Evaluate models and select a business-aware alert threshold.
5. Generate global and local SHAP explanations.

## Project structure

```text
fraud-challenge/
├── api.py                       # FastAPI prediction service
├── app.py                       # Streamlit dashboard
├── config.py                    # Paths and business/model parameters
├── requirements.txt             # Python dependencies
├── run_01_eda.py                # Module 1: EDA
├── run_02_features.py           # Module 2: features and split
├── run_03_train.py              # Modules 2-3: imbalance and models
├── run_04_evaluate.py           # Module 4: metrics and threshold
├── run_05_explain.py            # Module 5: SHAP explanations
├── src/
│   ├── data.py                  # Raw CSV loading
│   ├── explain.py               # SHAP and reason codes
│   ├── features.py              # Static and behavioral features
│   ├── metrics.py               # Evaluation and cost functions
│   └── models.py                # Model and resampling helpers
├── tools/
│   └── make_synthetic_data.py   # Local smoke-test data generator
├── data/                        # Input data and generated feature tables
├── artifacts/                   # Models, scores, metrics, and metadata
└── figures/                     # EDA, evaluation, and SHAP plots
```

## Requirements

- Python 3.10 or newer
- At least 8 GB of RAM for the full dataset
- The Sparkov `fraudTrain.csv` and `fraudTest.csv` files

The full training pipeline can take several minutes. LOF and One-Class SVM
are the slowest models.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fraud-challenge
```

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents script activation, run this once for the current user,
then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the data

Download the Sparkov credit-card fraud dataset from Kaggle and place these
two files directly in the project's `data/` directory:

```text
data/
├── fraudTrain.csv
└── fraudTest.csv
```

The loader uses these columns from each CSV:

```text
trans_date_trans_time, cc_num, merchant, category, amt, gender, state,
lat, long, city_pop, job, dob, merch_lat, merch_long, trans_num, is_fraud
```

The original CSV files are not included in this repository. Do not commit
the dataset or other sensitive transaction data.

## Quick smoke test without Kaggle data

To verify the complete pipeline locally, generate a small synthetic dataset.
Synthetic results are intentionally easy to classify and must not be used in
a report.

### Windows PowerShell

```powershell
$env:FRAUD_SMOKE = "1"
python tools\make_synthetic_data.py
```

### macOS/Linux

```bash
export FRAUD_SMOKE=1
python tools/make_synthetic_data.py
```

The generator writes synthetic `fraudTrain.csv` and `fraudTest.csv` files to
`data/`. Run the five pipeline steps below. Before switching to real data,
remove the generated CSV/parquet files and artifacts, then unset
`FRAUD_SMOKE`:

```text
data/*.csv
data/*.parquet
artifacts/*
figures/*
```

On Windows PowerShell, unset the environment variable with:

```powershell
Remove-Item Env:FRAUD_SMOKE
```

On macOS/Linux:

```bash
unset FRAUD_SMOKE
```

## Run the pipeline

Run every step from the project root and in the order shown.

### 1. Exploratory data analysis

```bash
python run_01_eda.py
```

This merges and sorts both CSV files chronologically, prints dataset
statistics, and creates:

- `data/transactions.parquet`
- `figures/eda_fraud_rate_monthly.png`
- `figures/eda_fraud_rate_hour.png`
- `figures/eda_fraud_rate_category.png`
- `figures/eda_amount_by_class.png`

### 2. Feature engineering and data split

```bash
python run_02_features.py
```

Features include:

- Cyclic time features, weekday/weekend, and night-time indicators
- Log transaction amount, customer age, city population, and gender
- Haversine distance between the customer and merchant
- Category one-hot features learned from the training split only
- Past-only card behavior: transaction counts over one and 24 hours,
  amount relative to the card's 30-day mean, and time since the previous
  transaction

The data is split chronologically into 60% train, 20% validation, and 20%
test. This avoids using future transactions to predict past transactions.
The step creates:

- `data/features.parquet`
- `artifacts/feature_names.json`
- `artifacts/categories.json`

Review the printed leakage check. The first transaction for a card should
have no transaction history.

### 3. Train models and compare imbalance strategies

```bash
python run_03_train.py
```

The script compares no resampling, class weighting, and several
undersampling ratios on the validation set. It then trains:

- XGBoost
- Weighted logistic regression
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

Generated files include:

- `artifacts/imbalance_comparison.csv`
- `artifacts/xgb.joblib`
- `artifacts/scores.parquet`
- `artifacts/train_summary.json`

### 4. Evaluate models and choose the alert threshold

```bash
python run_04_evaluate.py
```

The threshold is selected on validation data using:

```text
business cost = missed fraudulent amount + alert cost × number of alerts
```

The selected threshold is then evaluated on the test set. Reported metrics
include PR-AUC, ROC-AUC, precision@k, recall@k, F2, MCC, alert count, and
business cost.

Generated files include:

- `artifacts/model_comparison.csv`
- `artifacts/thresholds.json`
- `figures/pr_curves_test.png`
- `figures/cost_curve_val_xgb.png`

### 5. Generate SHAP explanations

```bash
python run_05_explain.py
```

This creates global feature-importance plots and a local explanation for one
high-scoring transaction:

- `figures/shap_beeswarm.png`
- `figures/shap_bar.png`
- `figures/shap_waterfall_example.png`
- `artifacts/example_reason_codes.json`

## Run the Streamlit dashboard

Run the pipeline through at least step 4 first. Run step 5 as well to show
the SHAP plots in the dashboard.

```bash
streamlit run app.py
```

Open the URL printed by Streamlit, normally
`http://localhost:8501`. The dashboard includes tabs for:

- Data and exploratory analysis
- Chronological split and imbalance comparison
- Model comparison and precision-recall curves
- Threshold, analyst budget, alerts, precision, recall, F2, and cost
- SHAP explanation and reason codes for an alert

## Run the FastAPI service

Run steps 1-4 first so the model and threshold artifacts exist:

```bash
python -m uvicorn api:app --reload
```

Open the interactive API documentation at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

Available endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

Example request with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ts": "2020-09-14T23:42:10",
    "amt": 742.5,
    "category": "shopping_net",
    "gender": "F",
    "lat": 40.7,
    "long": -74.0,
    "merch_lat": 34.0,
    "merch_long": -118.2,
    "city_pop": 8000000,
    "dob": "1985-03-02",
    "n_tx_1h": 3,
    "n_tx_24h": 4,
    "amt_ratio_30d": 9.5,
    "secs_since_prev": 420
  }'
```

The response contains a fraud score, alert decision, configured threshold,
and reason codes. In production, the caller or a feature store must provide
the four behavioral features because the API does not have access to the
card's transaction history.

## Configuration

Edit [`config.py`](config.py) to change:

- `ALERT_COST`: cost of reviewing one alert
- `ALERTS_PER_DAY`: analyst alert capacity
- `UNDERSAMPLE_RATIOS` and `UNDERSAMPLE_SEEDS`
- `UNSUP_TRAIN_SAMPLE`
- `XGB_PARAMS`

Optional environment variables:

- `FRAUD_WORK`: store `data/`, `artifacts/`, and `figures/` outside the
  repository
- `FRAUD_SMOKE=1`: use smaller model settings for the synthetic smoke test

For example:

```powershell
$env:FRAUD_WORK = "C:\path\to\fraud-work"
python run_01_eda.py
```

## Troubleshooting

- **Missing `data/fraudTrain.csv`:** put both required CSV files directly
  inside `data/`, or generate synthetic files for a smoke test.
- **Artifacts not found in Streamlit/API:** run steps 1-4 first; run step 5
  for SHAP images.
- **Out of memory:** close other applications, reduce `UNSUP_TRAIN_SAMPLE`
  in `config.py`, or use a smaller/synthetic dataset.
- **Training is too slow:** reduce `n_estimators` in `XGB_PARAMS` and
  `UNSUP_TRAIN_SAMPLE`. LOF is usually the slowest stage.
- **PowerShell activation is blocked:** use the `Set-ExecutionPolicy`
  command in the installation section, or activate the environment from
  Command Prompt with `.venv\Scripts\activate.bat`.

## Suggested analysis checklist

- Explain why the split is chronological and why behavioral features use
  only past transactions.
- Compare imbalance strategies using validation data.
- Compare at least one supervised and two unsupervised models.
- Include PR-AUC, ROC-AUC, precision@k, recall@k, F2, and MCC.
- Justify `ALERT_COST` and `ALERTS_PER_DAY`.
- Show that the threshold is selected on validation and confirmed on test.
- Include global SHAP importance, a waterfall plot, and reason codes.
