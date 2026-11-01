"""Explainability helpers (Module 5): SHAP + reason codes."""
import numpy as np
import shap

from src.features import label


def get_explainer(model):
    return shap.TreeExplainer(model)


def base_value(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def reason_codes(shap_row, x_row, names, top=5, min_codes=3):
    """The 3-5 features that pushed this transaction the most towards 'fraud'."""
    shap_row = np.asarray(shap_row, dtype=float)
    order = np.argsort(-shap_row)
    picked = [i for i in order[:top] if shap_row[i] > 0]
    if len(picked) < min_codes:                    # always return at least min_codes
        picked = list(order[:min_codes])
    out = []
    for i in picked:
        v = x_row[i]
        out.append({
            "feature": names[i],
            "label": label(names[i]),
            "value": float(v),
            "shap": float(shap_row[i]),
            "text": f"{label(names[i])} = {v:.2f} (impact {shap_row[i]:+.2f})",
        })
    return out


def explanation(shap_row, x_row, names, explainer):
    return shap.Explanation(values=np.asarray(shap_row, dtype=float), base_values=base_value(explainer),
                            data=np.asarray(x_row, dtype=float), feature_names=[label(n) for n in names])
