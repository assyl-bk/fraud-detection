"""Central configuration. Edit the business parameters here and JUSTIFY them in your report."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# FRAUD_WORK lets you keep data/artifacts/figures outside the code folder (optional)
WORK = Path(os.environ.get("FRAUD_WORK", ROOT))
DATA_DIR = WORK / "data"
ART_DIR = WORK / "artifacts"
FIG_DIR = WORK / "figures"
for _d in (DATA_DIR, ART_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# FRAUD_SMOKE=1 shrinks everything for a quick sanity run (used on synthetic data)
SMOKE = os.environ.get("FRAUD_SMOKE") == "1"

SEED = 42
RAW_FILES = ["fraudTrain.csv", "fraudTest.csv"]
SPLIT = (0.6, 0.2, 0.2)            # chronological train / val / test

# ---- Business parameters (Module 4) ---------------------------------------
ALERT_COST = 5.0                   # Ca: cost of analysing ONE alert, in currency units
ALERTS_PER_DAY = 5 if SMOKE else 30  # analyst budget -> k = ALERTS_PER_DAY * number of days

# ---- Imbalance strategies compared on validation (Module 2) ---------------
UNDERSAMPLE_RATIOS = [1, 5, 10, 20]   # normals kept per fraud
UNDERSAMPLE_SEEDS = [42, 43, 44]      # result depends on the draw -> repeat

# ---- Models (Module 3) ----------------------------------------------------
UNSUP_TRAIN_SAMPLE = 2000 if SMOKE else 10000   # LOF / One-Class SVM are slow: subsample normals
XGB_PARAMS = dict(
    n_estimators=60 if SMOKE else 300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    eval_metric="aucpr",
)

MODEL_NAMES = ["xgb", "logreg", "iforest", "lof", "ocsvm"]
MODEL_LABELS = {
    "xgb": "XGBoost (supervised)",
    "logreg": "Logistic regression (supervised)",
    "iforest": "Isolation Forest (unsup.)",
    "lof": "LOF (unsup.)",
    "ocsvm": "One-Class SVM (unsup.)",
}
