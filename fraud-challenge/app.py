"""Interactive fraud-detection app (Streamlit).

    streamlit run app.py

Run steps 1-5 first: the app only loads the artifacts they produce.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve

import config as C
from src import explain as E
from src import metrics as M
from src.features import label

st.set_page_config(page_title="Explainable fraud detection", layout="wide")
st.title("Explainable fraud detection")

needed = [C.ART_DIR / f for f in ("scores.parquet", "model_comparison.csv", "thresholds.json", "xgb.joblib")]
if not all(p.exists() for p in needed) or not (C.DATA_DIR / "features.parquet").exists():
    st.error("Artifacts not found. Run `run_01_eda.py` ... `run_04_evaluate.py` first (see README).")
    st.stop()


# ---------------------------------------------------------------- loaders ---
@st.cache_data
def load_meta():
    return pd.read_parquet(C.DATA_DIR / "features.parquet", columns=["ts", "category", "amt", "is_fraud", "split"])


@st.cache_data
def load_scores():
    return pd.read_parquet(C.ART_DIR / "scores.parquet")


@st.cache_data
def load_json(name):
    return json.loads((C.ART_DIR / name).read_text())


@st.cache_resource
def load_model():
    b = joblib.load(C.ART_DIR / "xgb.joblib")
    return b["model"], b["beta"], E.get_explainer(b["model"])


@st.cache_data
def load_test_features():
    names = load_json("feature_names.json")
    return pd.read_parquet(C.DATA_DIR / "features.parquet", filters=[("split", "==", "test")],
                           columns=["row_id", "ts", "category", "is_fraud"] + names).set_index("row_id")


meta, scores, thr = load_meta(), load_scores(), load_json("thresholds.json")
comp = pd.read_csv(C.ART_DIR / "model_comparison.csv")
names = load_json("feature_names.json")

tabs = st.tabs(["1. Data", "2. Split & imbalance", "3. Models", "4. Threshold & budget", "5. Explain an alert"])

# ------------------------------------------------------------ 1. Data -------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(meta):,}")
    c2.metric("Fraud rate", f"{meta.is_fraud.mean():.3%}")
    c3.metric("Frauds", f"{int(meta.is_fraud.sum()):,}")
    c4.metric("Period", f"{meta.ts.min().date()} -> {meta.ts.max().date()}")
    a, b = st.columns(2)
    monthly = meta.set_index("ts")["is_fraud"].resample("MS").mean().reset_index()
    a.plotly_chart(px.line(monthly, x="ts", y="is_fraud", markers=True, title="Fraud rate per month"), width="stretch")
    hourly = meta.groupby(meta.ts.dt.hour)["is_fraud"].mean().reset_index().rename(columns={"ts": "hour"})
    b.plotly_chart(px.bar(hourly, x="hour", y="is_fraud", title="Fraud rate by hour"), width="stretch")
    a, b = st.columns(2)
    cat = meta.groupby("category")["is_fraud"].mean().sort_values().reset_index()
    a.plotly_chart(px.bar(cat, x="is_fraud", y="category", orientation="h", title="Fraud rate by category"), width="stretch")
    samp = meta.sample(min(len(meta), 200_000), random_state=0).assign(cls=lambda d: d.is_fraud.map({0: "legit", 1: "fraud"}))
    b.plotly_chart(px.histogram(samp[samp.amt < 1500], x="amt", color="cls", barmode="overlay", histnorm="probability density",
                                nbins=60, title="Amount distribution by class"), width="stretch")

# ------------------------------------------------ 2. Split & imbalance ------
with tabs[1]:
    st.subheader("Chronological split (60 / 20 / 20)")
    sp = meta.groupby("split").agg(start=("ts", "min"), end=("ts", "max"), rows=("is_fraud", "size"),
                                   frauds=("is_fraud", "sum"), fraud_rate=("is_fraud", "mean"))
    st.dataframe(sp.loc[["train", "val", "test"]], width="stretch")
    st.subheader("Imbalance strategies compared on validation (training set only is resampled)")
    imb = pd.read_csv(C.ART_DIR / "imbalance_comparison.csv")
    st.dataframe(imb.style.highlight_max(subset=["val_pr_auc"], color="#c8f7c5"), width="stretch")
    st.caption(f"Best strategy on validation PR-AUC: **{load_json('train_summary.json')['best_strategy']}**")

# ------------------------------------------------------ 3. Models ----------
with tabs[2]:
    split = st.radio("Evaluation set", ["test", "val"], horizontal=True, key="m_split")
    t = comp[comp.split == split].copy()
    t["model"] = t["model"].map(C.MODEL_LABELS)
    st.dataframe(t.drop(columns=["split"]).set_index("model").round(4), width="stretch")
    fig = go.Figure()
    sub = scores[scores.split == split]
    for name in C.MODEL_NAMES:
        p, r, _ = precision_recall_curve(sub.is_fraud, sub[name])
        step = max(1, len(p) // 2000)
        fig.add_trace(go.Scatter(x=r[::step], y=p[::step], mode="lines", name=C.MODEL_LABELS[name]))
    fig.add_hline(y=sub.is_fraud.mean(), line_dash="dash", annotation_text="random")
    fig.update_layout(title=f"Precision-recall curves ({split})", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------- 4. Threshold & budget ---------
with tabs[3]:
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", C.MODEL_NAMES, format_func=C.MODEL_LABELS.get)
    split = c2.radio("Evaluate on", ["test", "val"], horizontal=True, key="t_split")
    ca = c3.number_input("Cost of analysing one alert (Ca)", min_value=0.0, value=float(thr["alert_cost"]), step=1.0)
    tau_opt = thr["models"][model]["tau"]

    sub = scores[scores.split == split]
    y, amt, s = sub.is_fraud.to_numpy(), sub.amt.to_numpy(), sub[model].to_numpy()
    days = thr["days"][split]
    order = np.argsort(-s, kind="stable")
    cum_tp = np.concatenate([[0], np.cumsum(y[order])])
    cum_fa = np.concatenate([[0.0], np.cumsum((y * amt)[order])])
    s_neg_sorted = -s[order]
    total_fraud_amt = float((y * amt).sum())

    c1, c2 = st.columns(2)
    tau = c1.slider("Alert threshold (score)", 0.0, 1.0, float(np.clip(tau_opt, 0, 1)), 0.001, key=f"tau_{model}")
    budget = c2.slider("Analyst budget (alerts per day)", 1, 300, int(thr["alerts_per_day"]), key="budget")
    st.caption(f"Cost-optimal threshold chosen on validation: **{tau_opt:.4f}**. "
               f"Alerts are limited by BOTH the threshold and the budget ({budget} x {days} days = {budget * days:,}).")

    n_thr = int(np.searchsorted(s_neg_sorted, -tau, side="right"))
    n = min(n_thr, int(budget * days))
    tp = int(cum_tp[n])
    prec = tp / n if n else 0.0
    rec = tp / max(int(y.sum()), 1)
    f2 = 5 * prec * rec / (4 * prec + rec) if (prec + rec) else 0.0
    cost = total_fraud_amt - cum_fa[n] + ca * n

    m = st.columns(6)
    m[0].metric("Alerts", f"{n:,}")
    m[1].metric("Precision", f"{prec:.3f}")
    m[2].metric("Recall", f"{rec:.3f}")
    m[3].metric("F2", f"{f2:.3f}")
    m[4].metric("Cost", f"{cost:,.0f}", delta=f"{cost - total_fraud_amt:,.0f} vs no model", delta_color="inverse")
    m[5].metric("No-model cost", f"{total_fraud_amt:,.0f}")

    n_all = np.arange(len(s) + 1)
    cost_curve = total_fraud_amt - cum_fa + ca * n_all
    upto = min(len(s), max(5 * int(budget * days), 2000))
    step = max(1, upto // 3000)
    fig = go.Figure(go.Scatter(x=n_all[:upto:step], y=cost_curve[:upto:step], mode="lines", name="cost"))
    fig.add_vline(x=n, line_color="red", annotation_text="current")
    fig.add_vline(x=int(np.argmin(cost_curve)), line_color="green", line_dash="dot", annotation_text=f"optimum (Ca={ca:g})")
    fig.add_hline(y=total_fraud_amt, line_dash="dash", annotation_text="no model")
    fig.update_layout(title="Business cost vs number of alerts", xaxis_title="alerts", yaxis_title="cost")
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------ 5. Explain an alert -------
with tabs[4]:
    st.subheader("Global explanation")
    img_b, img_r = C.FIG_DIR / "shap_beeswarm.png", C.FIG_DIR / "shap_bar.png"
    if img_b.exists():
        a, b = st.columns(2)
        a.image(str(img_b), caption="SHAP beeswarm (test sample)")
        b.image(str(img_r), caption="Mean |SHAP|") if img_r.exists() else None
    else:
        st.info("Run run_05_explain.py to generate the global SHAP figures.")

    st.subheader("Explain one alert (XGBoost)")
    model_x, beta, explainer = load_model()
    feats = load_test_features()
    sx = scores[scores.split == "test"].set_index("row_id")["xgb"]
    tau_x = thr["models"]["xgb"]["tau"]
    alerts = feats.assign(score=sx.reindex(feats.index)).query("score >= @tau_x").sort_values("score", ascending=False).head(200)
    if alerts.empty:
        st.warning("No alert at the current threshold.")
    else:
        view = alerts[["ts", "category", "amt", "score", "is_fraud"]].reset_index()
        st.dataframe(view, width="stretch", height=250)
        pick = st.selectbox("Choose an alert (row_id)", view["row_id"].tolist())
        row = alerts.loc[pick]
        x = row[names].to_numpy(dtype=float)
        sv = explainer.shap_values(row[names].to_frame().T.astype(float))[0]
        c1, c2 = st.columns([3, 2])
        with c1:
            plt.figure()
            shap.plots.waterfall(E.explanation(sv, x, names, explainer), show=False, max_display=12)
            st.pyplot(plt.gcf(), clear_figure=True)
        with c2:
            st.metric("Fraud score", f"{row['score']:.3f}", help=f"Alert threshold = {tau_x:.3f}")
            st.markdown(f"**Truth:** {'FRAUD' if row['is_fraud'] == 1 else 'legit'}")
            st.markdown("**Reason codes**")
            for c in E.reason_codes(sv, x, names, top=5):
                st.markdown(f"- {c['text']}")
