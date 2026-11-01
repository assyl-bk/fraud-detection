"""STEP 1 - Module 1: load the data + exploratory analysis.

    python run_01_eda.py

Outputs: data/transactions.parquet (merged, time-sorted) and figures/eda_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from src.data import load_raw

df = load_raw()
df.to_parquet(C.DATA_DIR / "transactions.parquet", index=False)

# ---- Key facts -------------------------------------------------------------
print(f"Rows: {len(df):,} | cards: {df.cc_num.nunique():,} | merchants: {df.merchant.nunique():,}")
print(f"Period: {df.ts.min()} -> {df.ts.max()}")
print(f"Fraud rate: {df.is_fraud.mean():.4%} ({int(df.is_fraud.sum()):,} frauds)")
per_card = df.groupby("cc_num").size()
print(f"Transactions per card: median {per_card.median():.0f} (min {per_card.min()}, max {per_card.max()})")
print("\nAmount by class:")
print(df.groupby("is_fraud")["amt"].describe().round(2))
print(f"\nTotal amount lost to fraud: {df.loc[df.is_fraud == 1, 'amt'].sum():,.0f}")

# Preview of the chronological split (the real split is done in step 2)
n = len(df)
i1, i2 = int(0.6 * n), int(0.8 * n)
print("\nChronological 60/20/20 preview:")
for name, part in (("train", df.iloc[:i1]), ("val", df.iloc[i1:i2]), ("test", df.iloc[i2:])):
    print(f"  {name:5s} {part.ts.min().date()} -> {part.ts.max().date()}  "
          f"rows {len(part):>9,}  fraud rate {part.is_fraud.mean():.4%}")

# ---- Figures ---------------------------------------------------------------
monthly = df.set_index("ts")["is_fraud"].resample("MS").mean()
ax = monthly.plot(marker="o", title="Fraud rate per month")
ax.set_ylabel("fraud rate")
plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_fraud_rate_monthly.png", dpi=130); plt.close()

ax = df.groupby(df.ts.dt.hour)["is_fraud"].mean().plot(kind="bar", title="Fraud rate by hour of day")
ax.set_ylabel("fraud rate")
plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_fraud_rate_hour.png", dpi=130); plt.close()

ax = df.groupby("category", observed=True)["is_fraud"].mean().sort_values().plot(kind="barh", title="Fraud rate by category")
ax.set_xlabel("fraud rate")
plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_fraud_rate_category.png", dpi=130); plt.close()

fig, ax = plt.subplots()
for lab, grp in df.groupby("is_fraud"):
    ax.hist(grp["amt"].clip(upper=1500), bins=60, alpha=0.6, density=True, label="fraud" if lab else "legit")
ax.set_title("Amount distribution by class (clipped at 1500)")
ax.set_xlabel("amt"); ax.legend()
plt.tight_layout(); plt.savefig(C.FIG_DIR / "eda_amount_by_class.png", dpi=130); plt.close()

print(f"\nSaved {C.DATA_DIR / 'transactions.parquet'} and figures in {C.FIG_DIR}/")
