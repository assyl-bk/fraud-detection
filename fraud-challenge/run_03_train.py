"""STEP 3 - Modules 2 (imbalance comparison) and 3 (modelling).

    python run_03_train.py

1. Compares imbalance strategies on VALIDATION with XGBoost (PR-AUC), keeps the best.
2. Trains the final supervised models (XGBoost, weighted logistic regression).
3. Trains 3 unsupervised detectors (Isolation Forest, LOF, One-Class SVM).
4. Scores validation + test with every model -> artifacts/scores.parquet

Takes a few minutes on the full dataset (LOF scoring is the slow part).
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config as C
from src import metrics as M
from src import models as MD

warnings.filterwarnings("ignore")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


feat = pd.read_parquet(C.DATA_DIR / "features.parquet")
names = json.loads((C.ART_DIR / "feature_names.json").read_text())
tr, va, te = (feat[feat.split == s] for s in ("train", "val", "test"))
Xtr, ytr = tr[names], tr.is_fraud.to_numpy()
Xva, yva, amtva = va[names], va.is_fraud.to_numpy(), va.amt.to_numpy()
k_val, days_val = M.alerts_budget(va.ts, C.ALERTS_PER_DAY)
log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,} | frauds in train: {int(ytr.sum()):,} | "
    f"budget k_val = {k_val:,} alerts ({C.ALERTS_PER_DAY}/day x {days_val} days)")

# ---------------------------------------------------------------------------
# 1) Imbalance strategies compared on VALIDATION (resampling touches TRAIN only)
# ---------------------------------------------------------------------------
rows = []
for strat in MD.strategy_list():
    seeds = C.UNDERSAMPLE_SEEDS if strat.startswith("under") else [C.SEED]
    prs, rocs, pks, rks = [], [], [], []
    ts0 = time.time()
    for sd in seeds:
        m, _ = MD.fit_strategy(Xtr, ytr, strat, seed=sd)
        s = m.predict_proba(Xva)[:, 1]
        prs.append(average_precision_score(yva, s))
        rocs.append(roc_auc_score(yva, s))
        p_k, r_k = M.precision_recall_at_k(yva, s, k_val)
        pks.append(p_k); rks.append(r_k)
    rows.append(dict(strategy=strat, val_pr_auc=np.mean(prs), val_pr_auc_std=np.std(prs),
                     val_roc_auc=np.mean(rocs), val_precision_at_k=np.mean(pks),
                     val_recall_at_k=np.mean(rks), n_seeds=len(seeds), seconds=round(time.time() - ts0, 1)))
    log(f"strategy {strat:13s} val PR-AUC = {np.mean(prs):.4f} (+/- {np.std(prs):.4f})")
comp = pd.DataFrame(rows)
comp.to_csv(C.ART_DIR / "imbalance_comparison.csv", index=False)
best = comp.sort_values("val_pr_auc", ascending=False).iloc[0]["strategy"]
log(f"-> best strategy on validation: {best}")

# ---------------------------------------------------------------------------
# 2) Final supervised models
# ---------------------------------------------------------------------------
xgb, beta = MD.fit_strategy(Xtr, ytr, best, seed=C.SEED)
joblib.dump({"model": xgb, "beta": beta, "strategy": best}, C.ART_DIR / "xgb.joblib")
log(f"XGBoost trained (strategy={best}, share of normals kept beta={beta:.4f})")

logreg = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=200))
logreg.fit(Xtr, ytr)
log("Logistic regression trained")

# ---------------------------------------------------------------------------
# 3) Unsupervised detectors
# ---------------------------------------------------------------------------
rng = np.random.default_rng(C.SEED)
normal_idx = np.flatnonzero(ytr == 0)
sample = rng.choice(normal_idx, size=min(C.UNSUP_TRAIN_SAMPLE, len(normal_idx)), replace=False)
scaler = StandardScaler().fit(Xtr.iloc[sample])
Xs = scaler.transform(Xtr.iloc[sample])

iforest = IsolationForest(n_estimators=100, max_samples=256, random_state=C.SEED, n_jobs=-1).fit(Xtr.to_numpy())
log("Isolation Forest trained (on the full unlabeled train set)")
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(Xs)
log(f"LOF trained on {len(Xs):,} normal transactions")
ocsvm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale").fit(Xs)
log(f"One-Class SVM trained on {len(Xs):,} normal transactions")

# ---------------------------------------------------------------------------
# 4) Score validation + test with every model (higher = more suspect)
# ---------------------------------------------------------------------------
ev = pd.concat([va, te])
Xev = ev[names]
Xev_np, Xev_sc = Xev.to_numpy(), scaler.transform(Xev)

p_xgb = xgb.predict_proba(Xev)[:, 1]
if beta < 1:
    p_xgb = MD.correct_proba(p_xgb, beta)          # undo probability inflation from undersampling
raw = {
    "xgb": p_xgb,
    "logreg": logreg.predict_proba(Xev)[:, 1],
    "iforest": -iforest.score_samples(Xev_np),
    "lof": MD.chunked(lambda a: -lof.score_samples(a), Xev_sc),
    "ocsvm": MD.chunked(lambda a: -ocsvm.decision_function(a), Xev_sc),
}
log("scored validation + test with all models")

is_val = (ev.split == "val").to_numpy()
out = ev[["row_id", "split", "ts", "is_fraud"]].copy()
out["amt"] = ev["amt"].to_numpy()
for name, sc in raw.items():
    if name in ("xgb", "logreg"):                   # already probabilities in [0, 1]
        out[name] = sc
    else:                                           # anomaly scores -> percentile w.r.t. validation distribution
        ref = np.sort(sc[is_val])
        out[name] = np.searchsorted(ref, sc, side="right") / (len(ref) + 1)
out.to_parquet(C.ART_DIR / "scores.parquet", index=False)

summary = {"best_strategy": best, "beta": beta, "k_val": k_val, "days_val": days_val,
           "alerts_per_day": C.ALERTS_PER_DAY, "alert_cost": C.ALERT_COST,
           "unsup_train_sample": int(len(Xs))}
(C.ART_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
log("done -> artifacts/scores.parquet, xgb.joblib, imbalance_comparison.csv")
print("\nImbalance comparison (validation):")
print(comp.round(4).to_string(index=False))
