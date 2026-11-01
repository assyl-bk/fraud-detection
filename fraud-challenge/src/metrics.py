"""Evaluation metrics for rare events + business-cost threshold (Module 4)."""
import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, roc_auc_score


def alerts_budget(ts, alerts_per_day: float):
    """Budget k = alerts per day x number of days covered by the split."""
    days = (ts.max() - ts.min()).days + 1
    return int(alerts_per_day * days), int(days)


def precision_recall_at_k(y, s, k):
    k = int(min(k, len(s)))
    top = np.argsort(-s, kind="stable")[:k]
    tp = y[top].sum()
    return float(tp / max(k, 1)), float(tp / max(y.sum(), 1))


def cost_curve(y, amt, s, ca):
    """cost[n] = cost when the n highest-scored transactions raise an alert:
    C = (sum of amounts of frauds NOT alerted) + Ca * n."""
    order = np.argsort(-s, kind="stable")
    fraud_amt = (amt * y)[order]
    cum = np.concatenate([[0.0], np.cumsum(fraud_amt)])
    n = np.arange(len(s) + 1)
    return order, fraud_amt.sum() - cum + ca * n


def optimal_threshold(y, amt, s, ca):
    """Threshold tau minimising the business cost (alert iff score >= tau)."""
    order, cost = cost_curve(y, amt, s, ca)
    n_star = int(np.argmin(cost))
    s_sorted = s[order]
    tau = float(s_sorted[n_star - 1]) if n_star > 0 else float(s_sorted[0]) + 1e-9
    return tau, n_star, float(cost[n_star])


def cost_at_threshold(y, amt, s, tau, ca):
    alerts = s >= tau
    missed = amt[(y == 1) & ~alerts].sum()
    return float(missed + ca * alerts.sum())


def evaluate(y, amt, s, tau, k, ca):
    alerts = s >= tau
    pred = alerts.astype(int)
    tp = int(((y == 1) & alerts).sum())
    p_k, r_k = precision_recall_at_k(y, s, k)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision_at_k": p_k,
        "recall_at_k": r_k,
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "n_alerts": int(alerts.sum()),
        "precision": tp / max(int(alerts.sum()), 1),
        "recall": tp / max(int(y.sum()), 1),
        "cost": cost_at_threshold(y, amt, s, tau, ca),
        "cost_no_model": float((amt * y).sum()),
    }
