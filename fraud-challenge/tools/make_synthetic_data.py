"""Generate a SYNTHETIC dataset with the same columns as the Kaggle Sparkov files.
Only meant to check that the whole pipeline runs (or to practise) - NOT to report results on.

    python tools/make_synthetic_data.py --rows 120000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

CATS = ["gas_transport", "grocery_pos", "home", "shopping_pos", "kids_pets", "shopping_net", "entertainment",
        "food_dining", "personal_care", "health_fitness", "misc_pos", "misc_net", "grocery_net", "travel"]
JOBS = ["Engineer", "Teacher", "Nurse", "Lawyer", "Designer", "Accountant", "Chef", "Pilot"]
STATES = ["NY", "TX", "CA", "FL", "PA", "OH", "IL", "MI"]


def main(rows, n_cards, fraud_rate, seed):
    rng = np.random.default_rng(seed)
    cc = np.unique(rng.integers(10**15, 10**16, n_cards * 2, dtype=np.int64))[:n_cards]
    home_lat, home_lon = rng.uniform(26, 48, n_cards), rng.uniform(-123, -70, n_cards)
    gender = rng.choice(["M", "F"], n_cards)
    dob = pd.to_datetime("1940-01-01") + pd.to_timedelta(rng.integers(0, 60 * 365, n_cards), unit="D")
    pop = rng.lognormal(9, 1.5, n_cards).astype(int) + 100
    job = rng.choice(JOBS, n_cards)
    state = rng.choice(STATES, n_cards)

    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31 23:59:59")
    span = (end - start).total_seconds()

    # --- legitimate transactions
    n_fraud = int(rows * fraud_rate)
    n_legit = rows - n_fraud
    ci = rng.integers(0, n_cards, n_legit)
    ts = start + pd.to_timedelta(rng.uniform(0, span, n_legit), unit="s")
    hour_bias = rng.random(n_legit) < 0.8                        # mostly daytime
    ts = ts.where(~hour_bias, ts.normalize() + pd.to_timedelta(rng.uniform(8 * 3600, 21 * 3600, n_legit), unit="s"))
    legit = pd.DataFrame({
        "ts": ts, "ci": ci, "amt": np.round(rng.lognormal(3.4, 0.9, n_legit) + 1, 2),
        "category": rng.choice(CATS, n_legit),
        "merch_lat": home_lat[ci] + rng.normal(0, 0.3, n_legit), "merch_long": home_lon[ci] + rng.normal(0, 0.3, n_legit),
        "is_fraud": 0})

    # --- fraud episodes: bursts on a compromised card, at night, large amounts, far from home, risky categories
    fr = []
    while len(fr) < n_fraud:
        c = int(rng.integers(0, n_cards))
        t0 = start + pd.to_timedelta(rng.uniform(0, span), unit="s")
        t0 = t0.normalize() + pd.to_timedelta(rng.choice([22, 23, 0, 1, 2, 3]) * 3600 + rng.uniform(0, 3000), unit="s")
        for j in range(int(rng.integers(1, 7))):
            t0 = t0 + pd.to_timedelta(rng.uniform(60, 1200), unit="s")
            fr.append({"ts": t0, "ci": c, "amt": round(float(rng.lognormal(5.8, 0.6)), 2),
                       "category": rng.choice(["shopping_net", "misc_net", "grocery_pos", "gas_transport", "shopping_pos"]),
                       "merch_lat": home_lat[c] + rng.normal(0, 6), "merch_long": home_lon[c] + rng.normal(0, 6), "is_fraud": 1})
    fraud = pd.DataFrame(fr[:n_fraud])

    df = pd.concat([legit, fraud], ignore_index=True).sort_values("ts").reset_index(drop=True)
    ci = df["ci"].to_numpy()
    out = pd.DataFrame({
        "Unnamed: 0": np.arange(len(df)),
        "trans_date_trans_time": df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc[ci], "merchant": "fraud_" + pd.Series(rng.integers(0, 600, len(df))).astype(str),
        "category": df["category"], "amt": df["amt"], "first": "A", "last": "B", "gender": gender[ci],
        "street": "1 Main St", "city": "Town", "state": state[ci], "zip": 10000,
        "lat": home_lat[ci], "long": home_lon[ci], "city_pop": pop[ci], "job": job[ci],
        "dob": dob[ci].strftime("%Y-%m-%d"),
        "trans_num": [f"{x:032x}" for x in rng.integers(0, 2**62, len(df))],
        "unix_time": (df["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds().astype(int),
        "merch_lat": df["merch_lat"], "merch_long": df["merch_long"], "is_fraud": df["is_fraud"],
    })
    cut = pd.Timestamp("2020-06-21")
    mask = df["ts"] < cut
    out[mask].to_csv(C.DATA_DIR / "fraudTrain.csv", index=False)
    out[~mask].to_csv(C.DATA_DIR / "fraudTest.csv", index=False)
    print(f"Wrote synthetic fraudTrain.csv ({mask.sum():,} rows) and fraudTest.csv ({(~mask).sum():,} rows) in {C.DATA_DIR}")
    print(f"Fraud rate: {out.is_fraud.mean():.3%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--cards", type=int, default=400)
    ap.add_argument("--fraud-rate", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.rows, a.cards, a.fraud_rate, a.seed)
