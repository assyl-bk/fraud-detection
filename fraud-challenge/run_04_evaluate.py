"""STEP 4 - Module 4: evaluation metrics, PR curves, cost-based threshold.

    python run_04_evaluate.py

The threshold is chosen on VALIDATION (minimising missed-fraud amounts + Ca x alerts),
then CONFIRMED on TEST. Outputs: artifacts/model_comparison.csv, thresholds.json, figures/
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import config as C
from src import metrics as M

sc = pd.read_parquet(C.ART_DIR / "scores.parquet")
va, te = sc[sc.split == "val"], sc[sc.split == "test"]
k = {"val": M.alerts_budget(va.ts, C.ALERTS_PER_DAY), "test": M.alerts_budget(te.ts, C.ALERTS_PER_DAY)}
ca = C.ALERT_COST
print(f"Alert cost Ca = {ca} | budget = {C.ALERTS_PER_DAY} alerts/day -> k_val = {k['val'][0]:,} ({k['val'][1]} days), "
      f"k_test = {k['test'][0]:,} ({k['test'][1]} days)\n")

rows, thresholds = [], {}
for name in C.MODEL_NAMES:
    yv, av, sv = va.is_fraud.to_numpy(), va.amt.to_numpy(), va[name].to_numpy()
    yt, at, st = te.is_fraud.to_numpy(), te.amt.to_numpy(), te[name].to_numpy()
    tau, n_star, _ = M.optimal_threshold(yv, av, sv, ca)           # chosen on validation
    _, _, oracle_cost = M.optimal_threshold(yt, at, st, ca)         # what test would have chosen (for comparison only)
    thresholds[name] = {"tau": tau, "n_alerts_val": n_star}
    for split, (y, a, s) in {"val": (yv, av, sv), "test": (yt, at, st)}.items():
        r = M.evaluate(y, a, s, tau, k[split][0], ca)
        r.update(model=name, split=split, tau=tau,
                 savings_pct=100 * (1 - r["cost"] / r["cost_no_model"]),
                 test_oracle_cost=oracle_cost if split == "test" else np.nan)
        rows.append(r)

table = pd.DataFrame(rows)
cols = ["model", "split", "pr_auc", "roc_auc", "precision_at_k", "recall_at_k", "f2", "mcc",
        "n_alerts", "precision", "recall", "tau", "cost", "cost_no_model", "savings_pct", "test_oracle_cost"]
table = table[cols]
table.to_csv(C.ART_DIR / "model_comparison.csv", index=False)

meta = {"alert_cost": ca, "alerts_per_day": C.ALERTS_PER_DAY,
        "days": {"val": k["val"][1], "test": k["test"][1]}, "k": {"val": k["val"][0], "test": k["test"][0]},
        "models": thresholds}
(C.ART_DIR / "thresholds.json").write_text(json.dumps(meta, indent=2))

pd.set_option("display.width", 220)
print("TEST results (threshold chosen on validation):")
print(table[table.split == "test"].drop(columns=["split"]).round(4).to_string(index=False))

# ---- Figure 1: overlaid precision-recall curves (test) ----------------------
fig, ax = plt.subplots(figsize=(7, 5))
for name in C.MODEL_NAMES:
    p, r, _ = precision_recall_curve(te.is_fraud, te[name])
    ap = table[(table.model == name) & (table.split == "test")].pr_auc.iloc[0]
    ax.plot(r, p, label=f"{C.MODEL_LABELS[name]} (AP={ap:.3f})")
ax.axhline(te.is_fraud.mean(), ls="--", c="gray", label=f"random ({te.is_fraud.mean():.3%})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves (test set)")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(C.FIG_DIR / "pr_curves_test.png", dpi=140); plt.close()

# ---- Figure 2: business cost vs number of alerts (XGBoost, validation) ------
yv, av, sv = va.is_fraud.to_numpy(), va.amt.to_numpy(), va["xgb"].to_numpy()
_, cost = M.cost_curve(yv, av, sv, ca)
n_star = thresholds["xgb"]["n_alerts_val"]
upto = min(len(cost) - 1, max(5 * k["val"][0], 5 * n_star, 1000))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(upto + 1), cost[:upto + 1])
ax.axhline(cost[0], ls="--", c="gray", label="no model (all frauds pass)")
ax.axvline(n_star, c="red", ls=":", label=f"optimum: {n_star:,} alerts")
ax.axvline(k["val"][0], c="green", ls=":", label=f"budget k = {k['val'][0]:,}")
ax.set_xlabel("Number of alerts raised (top scores)"); ax.set_ylabel("Business cost")
ax.set_title("Cost vs alerts - XGBoost, validation"); ax.legend(); plt.tight_layout()
plt.savefig(C.FIG_DIR / "cost_curve_val_xgb.png", dpi=140); plt.close()
print(f"\nSaved artifacts/model_comparison.csv, artifacts/thresholds.json, figures/pr_curves_test.png, figures/cost_curve_val_xgb.png")
