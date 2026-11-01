"""Loading the raw Sparkov CSV files."""
from pathlib import Path

import pandas as pd

import config as C

RAW_COLS = [
    "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "state", "lat", "long", "city_pop", "job", "dob", "merch_lat", "merch_long",
    "trans_num", "is_fraud",
]


def load_raw(data_dir: Path = C.DATA_DIR) -> pd.DataFrame:
    """Merge fraudTrain + fraudTest, sort chronologically, keep light dtypes."""
    parts = []
    for name in C.RAW_FILES:
        path = Path(data_dir) / name
        if not path.exists():
            raise SystemExit(f"Missing {path}. Download fraudTrain.csv and fraudTest.csv from Kaggle into {data_dir}/")
        parts.append(pd.read_csv(path, usecols=RAW_COLS, parse_dates=["trans_date_trans_time", "dob"]))
    df = pd.concat(parts, ignore_index=True).rename(columns={"trans_date_trans_time": "ts"})
    # Chronological order is the backbone of everything (split + velocity features)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["is_fraud"] = df["is_fraud"].astype("int8")
    for c in ("merchant", "category", "gender", "state", "job"):
        df[c] = df[c].astype("category")
    return df
