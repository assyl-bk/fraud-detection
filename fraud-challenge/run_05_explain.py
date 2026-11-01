"""STEP 5 - Module 5: Explainable AI (SHAP global + one alert explained).

    python run_05_explain.py

Outputs: figures/shap_beeswarm.png, shap_bar.png, shap_waterfall_example.png, artifacts/example_reason_codes.json
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config as C
from src import explain as E
from src.features import label

names = json.loads((C.ART_DIR / "feature_names.json").read_text())
bundle = joblib.load(C.ART_DIR / "xgb.joblib")
model = bundle["model"]
thr = json.loads((C.ART_DIR / "thresholds.json").read_text())
tau = thr["models"]["xgb"]["tau"]

feat = pd.read_parquet(C.DATA_DIR / "features.parquet", filters=[("split", "==", "test")]).set_index("row_id")
scores = pd.read_parquet(C.ART_DIR / "scores.parquet").set_index("row_id")
feat["score"] = scores.loc[feat.index, "xgb"]

explainer = E.get_explainer(model)

# ---- Global explanation: sample of test rows (all alerts are over-represented on purpose) ----
rng = np.random.default_rng(C.SEED)
n = len(feat)
sample_idx = rng.choice(n, size=min(5000, n), replace=False)
Xs = feat.iloc[sample_idx][names]
sv = explainer.shap_values(Xs)
Xs_lab = Xs.rename(columns={c: label(c) for c in names})

plt.figure(); shap.summary_plot(sv, Xs_lab, show=False, max_display=15)
plt.tight_layout(); plt.savefig(C.FIG_DIR / "shap_beeswarm.png", dpi=140, bbox_inches="tight"); plt.close()
plt.figure(); shap.summary_plot(sv, Xs_lab, plot_type="bar", show=False, max_display=15)
plt.tight_layout(); plt.savefig(C.FIG_DIR / "shap_bar.png", dpi=140, bbox_inches="tight"); plt.close()

# ---- Local explanation: highest-scored true fraud in the test set ----------------
frauds = feat[feat.is_fraud == 1].sort_values("score", ascending=False)
row = frauds.iloc[0] if len(frauds) else feat.sort_values("score", ascending=False).iloc[0]
x = row[names].to_numpy(dtype=float)
sv1 = explainer.shap_values(row[names].to_frame().T.astype(float))[0]

plt.figure()
shap.plots.waterfall(E.explanation(sv1, x, names, explainer), show=False, max_display=12)
plt.tight_layout(); plt.savefig(C.FIG_DIR / "shap_waterfall_example.png", dpi=140, bbox_inches="tight"); plt.close()

codes = E.reason_codes(sv1, x, names, top=5)
print(f"Alert example: amount={row['amt']:.2f}, score={row['score']:.3f}, threshold={tau:.3f}, is_fraud={int(row['is_fraud'])}")
print("Reason codes:")
for c in codes:
    print("  -", c["text"])
(C.ART_DIR / "example_reason_codes.json").write_text(json.dumps(codes, indent=2))
print("\nSaved figures/shap_beeswarm.png, shap_bar.png, shap_waterfall_example.png")
