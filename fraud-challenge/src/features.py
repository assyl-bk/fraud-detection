"""Feature engineering (Module 2). Everything here is leak-free:
- time / distance / age features only use the row itself
- behavioral (velocity) features only use the PAST of the same card
"""
import numpy as np
import pandas as pd

import config as C

CAP_SECS = 30 * 86400  # cap for "time since previous transaction" (also used for a card's first transaction)

STATIC = ["amt", "log_amt", "hour_sin", "hour_cos", "is_night", "dow", "is_weekend",
          "dist_km", "age", "log_city_pop", "gender_m"]
BEHAV = ["n_tx_1h", "n_tx_24h", "amt_ratio_30d", "secs_since_prev"]

FEATURE_LABELS = {
    "amt": "Amount", "log_amt": "Log amount", "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
    "is_night": "Night-time (22h-4h)", "dow": "Day of week", "is_weekend": "Weekend",
    "dist_km": "Distance home-merchant (km)", "age": "Customer age", "log_city_pop": "Log city population",
    "gender_m": "Gender = M", "n_tx_1h": "Card transactions in previous 1h",
    "n_tx_24h": "Card transactions in previous 24h", "amt_ratio_30d": "Amount / card's 30-day mean",
    "secs_since_prev": "Seconds since card's previous transaction",
}


def feature_names(categories):
    return STATIC + BEHAV + [f"cat_{c}" for c in categories]


def label(name: str) -> str:
    if name.startswith("cat_"):
        return "Category = " + name[4:]
    return FEATURE_LABELS.get(name, name)


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def assign_split(n: int) -> np.ndarray:
    """Chronological split BY POSITION (df must already be sorted by time)."""
    i1, i2 = int(C.SPLIT[0] * n), int((C.SPLIT[0] + C.SPLIT[1]) * n)
    s = np.empty(n, dtype=object)
    s[:i1], s[i1:i2], s[i2:] = "train", "val", "test"
    return s


def static_features(df: pd.DataFrame, categories) -> pd.DataFrame:
    """Features computable from a single raw transaction (used by the pipeline AND the API)."""
    ts = df["ts"]
    hour = (ts.dt.hour + ts.dt.minute / 60).to_numpy(float)
    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"].astype(float)
    out["log_amt"] = np.log1p(out["amt"])
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)      # cyclic encoding: 23h is close to 0h
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_night"] = ((hour >= 22) | (hour < 4)).astype(int)
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["dist_km"] = haversine_km(df["lat"].to_numpy(float), df["long"].to_numpy(float),
                                  df["merch_lat"].to_numpy(float), df["merch_long"].to_numpy(float))
    out["age"] = (ts - df["dob"]).dt.days / 365.25
    out["log_city_pop"] = np.log1p(df["city_pop"].astype(float))
    out["gender_m"] = (df["gender"].astype(str) == "M").astype(int)
    cat = df["category"].astype(str)
    for c in categories:
        out[f"cat_{c}"] = (cat == c).astype(int)
    return out


def behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity features per card, using ONLY earlier transactions (current row excluded).
    df must be sorted by time. Vectorised per card with searchsorted (fast on 1.8M rows)."""
    n = len(df)
    secs = ((df["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    amt = df["amt"].to_numpy(float)
    card = df["cc_num"].to_numpy()

    n1, n24 = np.zeros(n), np.zeros(n)
    ratio, since = np.ones(n), np.full(n, float(CAP_SECS))

    order = np.lexsort((secs, card))                  # by card, then time
    cs_card = card[order]
    bounds = np.flatnonzero(np.r_[True, cs_card[1:] != cs_card[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        t, x, m = secs[idx], amt[idx], b - a
        i = np.arange(m)
        lo1 = np.searchsorted(t, t - 3600, side="left")
        lo24 = np.searchsorted(t, t - 86400, side="left")
        lo30 = np.searchsorted(t, t - 30 * 86400, side="left")
        cum = np.concatenate([[0.0], np.cumsum(x)])
        cnt30 = i - lo30
        sum30 = cum[i] - cum[lo30]
        r = np.ones(m)
        ok = cnt30 > 0
        r[ok] = x[ok] / np.maximum(sum30[ok] / cnt30[ok], 1e-6)
        n1[idx], n24[idx], ratio[idx] = i - lo1, i - lo24, r
        since[idx] = np.minimum(np.concatenate([[CAP_SECS], np.diff(t)]), CAP_SECS)

    return pd.DataFrame({"n_tx_1h": n1, "n_tx_24h": n24, "amt_ratio_30d": ratio,
                         "secs_since_prev": since}, index=df.index)
