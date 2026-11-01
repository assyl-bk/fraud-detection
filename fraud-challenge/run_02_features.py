"""STEP 2 - Module 2: features + chronological split.

    python run_02_features.py

Outputs: data/features.parquet, artifacts/feature_names.json, artifacts/categories.json
"""
import json

import numpy as np
import pandas as pd

import config as C
from src import features as F
from src.data import load_raw

path = C.DATA_DIR / "transactions.parquet"
df = pd.read_parquet(path) if path.exists() else load_raw()
df = df.sort_values("ts", kind="stable").reset_index(drop=True)

# 1) Chronological split by position: 60% train / 20% val / 20% test
df["split"] = F.assign_split(len(df))

# 2) Categories learned from TRAIN only
categories = sorted(df.loc[df.split == "train", "category"].astype(str).unique())
names = F.feature_names(categories)

# 3) Features (behavioral ones only look at each card's past)
sta = F.static_features(df, categories)
beh = F.behavioral_features(df)
feat = pd.concat([sta, beh], axis=1)[names]

meta = df[["ts", "cc_num", "trans_num", "category", "is_fraud", "split"]].copy()
meta["category"] = meta["category"].astype(str)
out = pd.concat([meta, feat], axis=1)
out.insert(0, "row_id", np.arange(len(out)))
out.to_parquet(C.DATA_DIR / "features.parquet", index=False)

(C.ART_DIR / "feature_names.json").write_text(json.dumps(names))
(C.ART_DIR / "categories.json").write_text(json.dumps(categories))

# ---- Sanity checks (look at them, and mention them in your report) ----------
print(f"{len(names)} features: {names}\n")
print("Split summary:")
for s in ("train", "val", "test"):
    p = out[out.split == s]
    print(f"  {s:5s} {p.ts.min().date()} -> {p.ts.max().date()}  rows {len(p):>9,}  "
          f"frauds {int(p.is_fraud.sum()):>6,}  rate {p.is_fraud.mean():.4%}")
print("\nNaN count in features:", int(feat.isna().sum().sum()))

card = df.cc_num.iloc[0]
cols = ["ts", "amt", "n_tx_1h", "n_tx_24h", "amt_ratio_30d", "secs_since_prev"]
chk = out.loc[out.cc_num == card, cols].head(6)
print(f"\nLeakage check - first transactions of card {card} (first row must show 0 / 0 / 1.0 / 2592000):")
print(chk.to_string(index=False))
print("\nMean velocity features, fraud vs legit:")
print(out.groupby("is_fraud")[["n_tx_1h", "n_tx_24h", "amt_ratio_30d", "dist_km"]].mean().round(2))
