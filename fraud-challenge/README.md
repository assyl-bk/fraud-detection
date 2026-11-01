# Explainable Fraud Detection - Data Mining Challenge 2 (ENSI)

Dataset: **Credit Card Transactions Fraud Detection (Sparkov)** - `fraudTrain.csv` + `fraudTest.csv` from Kaggle.
One script per module, run them in order. Each step prints what to look at and what to write in your report.

```
fraud-challenge/
  config.py              <- business parameters (alert cost, analyst budget...) - JUSTIFY these
  run_01_eda.py          Module 1  data + exploratory analysis
  run_02_features.py     Module 2  features + chronological split
  run_03_train.py        Modules 2-3  imbalance comparison + supervised & unsupervised models
  run_04_evaluate.py     Module 4  metrics, PR curves, cost-based threshold
  run_05_explain.py      Module 5  SHAP (global + one alert)
  app.py                 Streamlit web app (all modules, live sliders)
  api.py                 Module 6  FastAPI
  src/                   reusable code (features, metrics, models, explain)
  tools/make_synthetic_data.py   fake Sparkov-shaped data to test the pipeline
  data/  artifacts/  figures/    inputs / saved models & scores / plots
```

## 0. Setup (once)

```bash
cd fraud-challenge
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `fraudTrain.csv` and `fraudTest.csv` from Kaggle (Sparkov credit card fraud) into `data/`.
Need RAM? ~8 GB is enough. **Quick dry run first** (30 s, fake data, no Kaggle needed):

```bash
# Linux/Mac                                   # Windows PowerShell
export FRAUD_SMOKE=1                          # $env:FRAUD_SMOKE="1"
python tools/make_synthetic_data.py
```
Then run steps 1-5 below, and **delete `data/*.csv data/*.parquet artifacts/* figures/*` and unset `FRAUD_SMOKE`** before using the real data.
(The synthetic data is trivially separable, so metrics look near-perfect. That is expected; never report them.)

## Step 1 - Data & EDA (Module 1)

```bash
python run_01_eda.py
```
Merges both files, sorts by time, prints fraud rate / cards / amounts per class, previews the 60/20/20 split and saves
4 plots in `figures/`. **Report:** fraud rate (~0.5%), amount distribution fraud vs legit, fraud rate over time / by hour / by category,
and the observation that the provided train/test files are merged to make *your own* chronological split.

## Step 2 - Features & split (Module 2)

```bash
python run_02_features.py
```
- Time: hour as sin/cos (cyclic), weekday, weekend, night flag.
- Other: log amount, home-merchant distance (haversine), age, log city population, gender, category one-hot.
- **Behavioral (per card, past only):** transactions in the previous 1h and 24h, amount / card's 30-day mean, seconds since previous transaction.
- **Split:** chronological by position, 60/20/20. Categories learned on train only.

Check the *leakage check* printout: the first transaction of a card must show `0 / 0 / 1.0 / 2592000` (no history).
**Report:** why chronological, why velocity features use only the past, the 3 splits' dates and fraud rates (drift!).

## Step 3 - Imbalance + models (Modules 2-3)

```bash
python run_03_train.py        # a few minutes on the full dataset
```
1. **Imbalance strategies compared on validation** (XGBoost, PR-AUC): none, class weights, undersampling 1:1 / 1:5 / 1:10 / 1:20
   (3 seeds each; resampling touches the training set only). The best one is kept -> `artifacts/imbalance_comparison.csv`.
   Course order: start with *no resampling + threshold*, then class weights, then undersampling; SMOTE only if validation proves it helps.
   Undersampling inflates probabilities, so they are corrected with the Bayes formula (slide 156).
2. Supervised: XGBoost (main), weighted logistic regression (baseline).
3. Unsupervised: Isolation Forest, LOF, One-Class SVM (LOF/OCSVM trained on a subsample of normal transactions, standardised).
   Unsupervised scores are converted to percentiles of the validation distribution so they live in [0, 1] like probabilities.

**Report:** the imbalance table and your justification, why unsupervised models are weaker (unusual != fraudulent), subsampling for LOF/OCSVM.
Optional extra: add an autoencoder (`MLPRegressor`, slide 178) as a 4th unsupervised method.

## Step 4 - Evaluation & threshold (Module 4)

```bash
python run_04_evaluate.py
```
- Metrics per model on val and test: **PR-AUC, ROC-AUC, precision@k, recall@k, F2, MCC**.
- **Budget:** `k = ALERTS_PER_DAY x number of days` (config.py). Justify the number (analyst capacity vs ~13 frauds/day).
- **Cost:** `C(tau) = sum(amounts of missed frauds) + Ca x number of alerts` (`ALERT_COST` in config.py). Justify Ca.
- `tau` is chosen **on validation**, then evaluated on test; column `test_oracle_cost` shows what test alone would have chosen (confirmation).
- Figures: overlaid PR curves, cost vs number of alerts.

**Report:** comparison table (all methods), PR curves, chosen Ca and k, savings vs "no model", validation vs test cost.
Try several Ca values (1, 5, 20, 50) and show how the optimal threshold moves.

## Step 5 - Explainability (Module 5)

```bash
python run_05_explain.py
```
SHAP beeswarm + bar (global), SHAP waterfall + **reason codes** (top 3-5 features pushing one alert towards fraud).
**Report:** top features and whether they make business sense (night, distance, amount vs history, velocity), one explained alert.
SHAP values are in log-odds of the XGBoost model.

## Step 6 - Web app

```bash
streamlit run app.py
```
5 tabs: Data / Split & imbalance / Models (table + PR curves) / **Threshold & budget** (two sliders, live alerts, precision, recall, F2, cost)
/ Explain an alert (pick an alert -> waterfall + reason codes). Take screenshots for the report.

## Step 7 - API (Module 6)

```bash
uvicorn api:app --reload
```
Open http://127.0.0.1:8000/docs. Endpoints: `GET /health`, `GET /model-info`, `POST /predict`
(returns `fraud_score`, `alert`, `threshold`, `reason_codes`). Velocity features are sent by the caller
(in production they come from a feature store). The PDF brief cuts off at "Module 6": check the exact API requirements with your teacher.

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "ts":"2020-09-14T23:42:10","amt":742.5,"category":"shopping_net","gender":"F",
  "lat":40.7,"long":-74.0,"merch_lat":34.0,"merch_long":-118.2,"city_pop":8000000,"dob":"1985-03-02",
  "n_tx_1h":3,"n_tx_24h":4,"amt_ratio_30d":9.5,"secs_since_prev":420}'
```

## Rubric checklist

- [ ] Chronological split, behavioral features without leakage (Step 2)
- [ ] Imbalance strategy chosen by a validation comparison, resampling on train only (Step 3)
- [ ] >= 1 supervised + >= 2 unsupervised incl. Isolation Forest, comparison table (Steps 3-4)
- [ ] PR-AUC, ROC-AUC, precision@k, recall@k, F2, MCC; overlaid PR curves (Step 4)
- [ ] Threshold minimising a cost in amounts, chosen on val, confirmed on test (Step 4)
- [ ] Live threshold + budget sliders in the app (Step 6)
- [ ] SHAP global + waterfall + reason codes (Step 5)
- [ ] FastAPI (Step 7)

## Troubleshooting

- *Out of memory:* close other apps; in `config.py` lower `UNSUP_TRAIN_SAMPLE`; or work on the last N months only.
- *Step 3 too slow:* lower `n_estimators` in `XGB_PARAMS`, or `UNSUP_TRAIN_SAMPLE`.
- *`Missing data/fraudTrain.csv`:* the files must be directly in `data/`.
- *Streamlit says artifacts not found:* run steps 1-4 (and 5 for the SHAP images) from the project folder.
