"""Model builders and imbalance strategies (Modules 2-3)."""
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

import config as C


def make_xgb(scale_pos_weight=1.0, seed=C.SEED):
    return XGBClassifier(**C.XGB_PARAMS, scale_pos_weight=scale_pos_weight, random_state=seed)


def strategy_list():
    return ["none", "class_weight"] + [f"under_1:{r}" for r in C.UNDERSAMPLE_RATIOS]


def fit_strategy(X, y, strategy, seed=C.SEED):
    """Fit XGBoost with an imbalance strategy applied to the TRAINING set only.
    Returns (model, beta) where beta = share of normal transactions kept (1.0 if none removed)."""
    y = np.asarray(y)
    if strategy == "none":
        return make_xgb(1.0, seed).fit(X, y), 1.0
    if strategy == "class_weight":
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        return make_xgb(spw, seed).fit(X, y), 1.0
    if strategy.startswith("under_1:"):
        r = int(strategy.split(":")[1])                    # r normals per fraud
        rus = RandomUnderSampler(sampling_strategy=1 / r, random_state=seed)
        Xr, yr = rus.fit_resample(X, y)
        beta = (np.asarray(yr) == 0).sum() / (y == 0).sum()
        return make_xgb(1.0, seed).fit(Xr, yr), float(beta)
    raise ValueError(strategy)


def correct_proba(ps, beta):
    """Undo the probability inflation caused by undersampling (Bayes, slide 156):
    p = beta*ps / (beta*ps - ps + 1)."""
    ps = np.asarray(ps)
    return beta * ps / (beta * ps - ps + 1)


def chunked(fn, X, size=50_000):
    return np.concatenate([fn(X[i:i + size]) for i in range(0, len(X), size)])
