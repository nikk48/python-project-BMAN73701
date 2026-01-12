
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local imports work when running: streamlit run python_project/streamlit_app.py
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import AED_FILE_PATH, AED_RANDOM_SEED, AED_SAMPLE_N, TASK6_TARGET_RECALL, AUDIT_LOG_PATH

from lp.solver import (
    build_default_instance,
    solve_task1_min_cost,
    solve_task2_lexicographic,
    solve_task3_min_cost_with_skills,
)
from lp.analysis import schedule_to_dataframe, assert_coverage, compute_weekly_hours, compute_delta_from_weekly_hours

from aed.data_management import AEDDataManager
from aed.analysis import _make_breach_flag  # reuse target cleaning

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

# Reuse the management-friendly summaries from the AED analysis module
from aed.analysis import (
    _management_numeric_summary as _aed_numeric_summary,
    _management_categorical_summary as _aed_categorical_summary,
)


# ----------------------------
# Plot helpers
# ----------------------------
def _fig_heatmap(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    data = df.values.astype(float)
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    # annotate
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def _fig_bar(series: pd.Series, title: str, xlabel: str = "", ylabel: str = "") -> plt.Figure:
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def _fig_corr(corr: pd.DataFrame, title: str = "Correlation heatmap") -> plt.Figure:
    """Simple correlation heatmap for numeric variables (management-friendly)."""
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values.astype(float), aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(corr.index)
    # annotate values
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def _read_audit_log(path: Path, max_lines: int = 5000) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return [ln.rstrip("\n") for ln in lines]


# ----------------------------
# LP Tasks (1–3)
# ----------------------------
@st.cache_data(show_spinner=False)
def _solve_lp_all() -> Dict[str, Any]:
    operators, days, hours_per_day = build_default_instance()

    # Task 1
    t1 = solve_task1_min_cost(operators, days, hours_per_day)
    df1 = schedule_to_dataframe(t1.hours, days)
    assert_coverage(df1, days, hours_per_day)
    weekly1 = pd.Series(compute_weekly_hours(t1.hours, days)).sort_values(ascending=False)
    delta1 = compute_delta_from_weekly_hours(dict(weekly1), t1.avg_hours)

    # Task 2 (fairness under cost cap)
    t2 = solve_task2_lexicographic(fairness_metric="delta", operators=operators, days=days, hours_per_day=hours_per_day, baseline_cost=t1.total_cost, cost_cap=1.018 * t1.total_cost, model_prefix="Task2_ScenarioI_DELTA")
    df2 = schedule_to_dataframe(t2.hours, days)
    assert_coverage(df2, days, hours_per_day)
    weekly2 = pd.Series(compute_weekly_hours(t2.hours, days)).sort_values(ascending=False)
    delta2 = compute_delta_from_weekly_hours(dict(weekly2), t2.avg_hours)

    # Task 3 (skill constraints)
    skill_sets = {
        "Programming": {"E. Khan", "Y. Chen", "R. Perez", "C. Santos"},
        "Troubleshooting": {"A. Taylor", "R. Zidane", "C. Santos"},
    }
    t3 = solve_task3_min_cost_with_skills(
        operators=operators,
        days=days,
        hours_per_day=hours_per_day,
        baseline_cost=t1.total_cost,
        skill_sets=skill_sets,
        daily_skill_req=6.0,
    )
    df3 = schedule_to_dataframe(t3.hours, days)
    assert_coverage(df3, days, hours_per_day)
    weekly3 = pd.Series(compute_weekly_hours(t3.hours, days)).sort_values(ascending=False)
    delta3 = compute_delta_from_weekly_hours(dict(weekly3), t3.avg_hours)

    return {
        "operators": [op.name for op in operators],
        "days": days,
        "hours_per_day": hours_per_day,
        "task1": {"res": t1, "df": df1, "weekly": weekly1, "delta": float(delta1)},
        "task2": {"res": t2, "df": df2, "weekly": weekly2, "delta": float(delta2)},
        "task3": {"res": t3, "df": df3, "weekly": weekly3, "delta": float(delta3), "skill_sets": skill_sets},
    }


def _skill_coverage_table(hours: Dict[str, Dict[str, float]], days: list[str], skill_sets: Dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for d in days:
        for skill, ops in skill_sets.items():
            total = 0.0
            for op, per_day in hours.items():
                if op in ops:
                    total += float(per_day.get(d, 0.0))
            rows.append({"Day": d, "Skill": skill, "TotalHours": total})
    out = pd.DataFrame(rows)
    return out.pivot(index="Day", columns="Skill", values="TotalHours")


# ----------------------------
# AED Tasks (4–5) + Data Mgmt (7–8)
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_aed(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def _aed_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(int(n), len(df))
    return df.sample(n=n, random_state=int(seed)).copy()


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _breach_flag(df: pd.DataFrame) -> pd.Series:
    if "Breachornot" not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return _make_breach_flag(df, "Breachornot")


# ----------------------------
# ML Task (6)
# ----------------------------
def _run_task6_ml(df: pd.DataFrame, seed: int, target_recall: float) -> Dict[str, Any]:
    if "Breachornot" not in df.columns:
        raise ValueError("Dataset must include a 'Breachornot' column for Task 6.")
    df = df.copy()
    df["Breachornot"] = df["Breachornot"].astype(str).str.strip().str.lower()
    y = (df["Breachornot"] == "breach").astype(int)

    drop_cols = ["Breachornot", "LoS", "ID"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y if y.nunique() > 1 else None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp if y_temp.nunique() > 1 else None
    )

    clf.fit(X_train, y_train)

    # Choose threshold on validation to hit target recall (or best achievable)
    val_proba = clf.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    candidates = []
    best_t, best_recall = 0.5, -1.0
    for t in thresholds:
        y_hat = (val_proba >= t).astype(int)
        tp = int(((y_hat == 1) & (y_val == 1)).sum())
        fn = int(((y_hat == 0) & (y_val == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall > best_recall:
            best_recall, best_t = recall, float(t)
        if recall >= float(target_recall):
            alerts = int((y_hat == 1).sum())
            candidates.append((alerts, float(t), recall))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[1]))
        threshold = candidates[0][1]
    else:
        threshold = best_t

    # Evaluate
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else float("nan")
    pr_auc = average_precision_score(y_test, test_proba) if y_test.nunique() > 1 else float("nan")
    cm = confusion_matrix(y_test, test_pred)
    report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, test_proba) if y_test.nunique() > 1 else (np.array([0,1]), np.array([0,1]), None)
    prec, rec, _ = precision_recall_curve(y_test, test_proba) if y_test.nunique() > 1 else (np.array([1,0]), np.array([0,1]), None)

    # ROC fig
    fig_roc, ax1 = plt.subplots()
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_title("ROC curve (test)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    fig_roc.tight_layout()

    # PR fig
    fig_pr, ax2 = plt.subplots()
    ax2.plot(rec, prec)
    ax2.set_title("Precision–Recall curve (test)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    fig_pr.tight_layout()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc) if roc_auc==roc_auc else None,
        "pr_auc": float(pr_auc) if pr_auc==pr_auc else None,
        "confusion_matrix": cm,
        "report": report,
        "fig_roc": fig_roc,
        "fig_pr": fig_pr,
        "sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "rates": {"train": float(y_train.mean()), "val": float(y_val.mean()), "test": float(y_test.mean())},
    }


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="BMAN73701 — Tasks 1–8 Visual App", layout="wide")

st.title("BMAN73701 — Tasks 1–8 (Visual Streamlit App)")
st.caption("LP scheduling (Tasks 1–3), A&E analysis (Tasks 4–6), data management + audit trail (Tasks 7–8).")

with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Go to",
        [
            "Task 1 — Min cost schedule",
            "Task 2 — Fairness schedule (≤1.8% cost increase)",
            "Task 3 — Skill coverage schedule",
            "Task 4 — A&E sample summary",
            "Task 5 — A&E drivers & breakdowns",
            "Task 6 — ML breach prediction",
            "Task 7 — Data management (CRUD)",
            "Task 8 — Audit trail",
        ],
        index=0,
    )

    st.divider()
    st.header("A&E dataset")
    upload = st.file_uploader("Upload AED CSV (optional)", type=["csv"])
    if upload is not None:
        aed_path = upload
        aed_path_label = "Uploaded file"
    else:
        aed_path = AED_FILE_PATH
        aed_path_label = str(AED_FILE_PATH)

    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=int(AED_RANDOM_SEED), step=1)
    n_sample = st.number_input("Sample size (Tasks 4–5)", min_value=50, max_value=10_000, value=int(AED_SAMPLE_N), step=50)

    st.divider()
    target_recall = st.slider("Task 6 target recall (breach)", min_value=0.50, max_value=0.99, value=float(TASK6_TARGET_RECALL), step=0.01)

# --- Load shared computations
lp_all = _solve_lp_all()

# --- Helper: load AED (uploaded or default)
if upload is not None:
    aed_df = pd.read_csv(upload)
else:
    aed_df = _load_aed(AED_FILE_PATH)

# ----------------------------
# TASK 1
# ----------------------------
if section.startswith("Task 1"):
    t = lp_all["task1"]
    res, df, weekly, delta = t["res"], t["df"], t["weekly"], t["delta"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", res.status)
    c2.metric("Total cost", f"£{res.total_cost:,.2f}")
    c3.metric("Max deviation from average (Δ)", f"{delta:.2f} h")

    st.subheader("Schedule table (hours)")
    st.dataframe(df, use_container_width=True)

    st.subheader("Heatmap: hours per operator × day")
    fig = _fig_heatmap(df[lp_all["days"]], "Task 1 — hours allocation")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Weekly hours per operator")
    fig2 = _fig_bar(weekly, "Task 1 — weekly hours", ylabel="Hours")
    st.pyplot(fig2, clear_figure=True)

# ----------------------------
# TASK 2
# ----------------------------
elif section.startswith("Task 2"):
    t1 = lp_all["task1"]
    t2 = lp_all["task2"]

    res1, df1, weekly1 = t1["res"], t1["df"], t1["weekly"]
    res2, df2, weekly2, delta2 = t2["res"], t2["df"], t2["weekly"], t2["delta"]

    cost_increase = ((res2.total_cost / res1.total_cost) - 1) * 100 if res1.total_cost else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline cost (Task 1)", f"£{res1.total_cost:,.2f}")
    c2.metric("Task 2 cost", f"£{res2.total_cost:,.2f}", delta=f"{cost_increase:.2f}%")
    c3.metric("Fairness (Δ)", f"{delta2:.2f} h")

    st.subheader("Task 2 schedule table (hours)")
    st.dataframe(df2, use_container_width=True)

    st.subheader("Compare weekly hours (Task 1 vs Task 2)")
    comp = pd.DataFrame({"Task 1": weekly1, "Task 2": weekly2}).fillna(0)
    st.dataframe(comp, use_container_width=True)

    fig = _fig_bar(comp["Task 1"] - comp["Task 2"], "Difference in weekly hours (Task 1 − Task 2)", ylabel="Hours")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Heatmap: Task 2 hours per operator × day")
    fig2 = _fig_heatmap(df2[lp_all["days"]], "Task 2 — fairer allocation")
    st.pyplot(fig2, clear_figure=True)

# ----------------------------
# TASK 3
# ----------------------------
elif section.startswith("Task 3"):
    t1 = lp_all["task1"]
    t3 = lp_all["task3"]

    res1 = t1["res"]
    res3, df3, weekly3, delta3, skill_sets = t3["res"], t3["df"], t3["weekly"], t3["delta"], t3["skill_sets"]

    cost_increase = ((res3.total_cost / res1.total_cost) - 1) * 100 if res1.total_cost else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline cost (Task 1)", f"£{res1.total_cost:,.2f}")
    c2.metric("Task 3 cost", f"£{res3.total_cost:,.2f}", delta=f"{cost_increase:.2f}%")
    c3.metric("Fairness (Δ)", f"{delta3:.2f} h")

    st.subheader("Task 3 schedule table (hours)")
    st.dataframe(df3, use_container_width=True)

    st.subheader("Heatmap: Task 3 hours per operator × day")
    fig = _fig_heatmap(df3[lp_all["days"]], "Task 3 — hours allocation")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Skill coverage by day (must be ≥ 6h each)")
    skill_tbl = _skill_coverage_table(res3.hours, lp_all["days"], skill_sets)
    st.dataframe(skill_tbl, use_container_width=True)

    # quick check visuals
    for col in skill_tbl.columns:
        st.write(f"**{col}:** min={skill_tbl[col].min():.1f}h, max={skill_tbl[col].max():.1f}h")

# ----------------------------
# TASK 4
# ----------------------------
elif section.startswith("Task 4"):
    st.write(f"Using dataset: **{aed_path_label}** (rows={len(aed_df):,})")

    sample = _aed_sample(aed_df, n=int(n_sample), seed=int(seed))
    sample = _coerce_numeric(sample, ["Age", "LoS"])

    st.subheader("Random sample")
    st.dataframe(sample.head(30), use_container_width=True)

    # Download the sampled rows (useful for submission appendix / reproducibility)
    st.download_button(
        "Download sample (CSV)",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name=f"AED_sample_{int(n_sample)}_seed_{int(seed)}.csv",
        mime="text/csv",
    )

    c1, c2, c3 = st.columns(3)
    if "Age" in sample.columns:
        c1.metric("Age (mean)", f"{sample['Age'].mean():.1f}")
    if "LoS" in sample.columns:
        c2.metric("LoS minutes (mean)", f"{sample['LoS'].mean():.1f}")
        c3.metric("LoS minutes (median)", f"{sample['LoS'].median():.1f}")

    st.subheader("Distributions")
    colA, colB = st.columns(2)
    if "Age" in sample.columns:
        fig_age, ax = plt.subplots()
        ax.hist(sample["Age"].dropna(), bins=20)
        ax.set_title("Age distribution (sample)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        fig_age.tight_layout()
        colA.pyplot(fig_age, clear_figure=True)

    if "LoS" in sample.columns:
        fig_los, ax = plt.subplots()
        ax.hist(sample["LoS"].dropna(), bins=20)
        ax.set_title("Length of stay distribution (sample)")
        ax.set_xlabel("LoS (minutes)")
        ax.set_ylabel("Count")
        fig_los.tight_layout()
        colB.pyplot(fig_los, clear_figure=True)

    # ----------------------------
    # Management-style summary tables (matches Task 4 wording)
    # ----------------------------
    with st.expander("Management summary tables (numeric + categorical)", expanded=True):
        num_summary = _aed_numeric_summary(sample)
        st.markdown("**Numeric summary (mean, median, quartiles, range, missingness)**")
        st.dataframe(num_summary, use_container_width=True)
        st.download_button(
            "Download numeric summary (CSV)",
            data=num_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"AED_numeric_summary_seed_{int(seed)}.csv",
            mime="text/csv",
        )

        cat_summary = _aed_categorical_summary(sample, top_k=12)
        st.markdown("**Categorical summaries (top 12 categories; includes NaNs)**")
        cat_col = st.selectbox("Choose a categorical column", options=list(cat_summary.keys()), index=0)
        st.dataframe(cat_summary[cat_col], use_container_width=True)

    # ----------------------------
    # Interesting relationships (quick exploration)
    # ----------------------------
    st.subheader("Interesting relationships (explorer)")
    numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = sample.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        st.markdown("**Correlation between numeric variables**")
        corr = sample[numeric_cols].corr(numeric_only=True).fillna(0.0)
        st.pyplot(_fig_corr(corr, "Correlation heatmap (sample)"), clear_figure=True)

        st.markdown("**Scatter plot (pick two numeric variables)**")
        c1, c2 = st.columns(2)
        xvar = c1.selectbox("X variable", options=numeric_cols, index=0)
        yvar = c2.selectbox("Y variable", options=numeric_cols, index=1)
        tmp_sc = sample[[xvar, yvar]].dropna()
        fig_sc, ax = plt.subplots()
        ax.scatter(tmp_sc[xvar], tmp_sc[yvar], s=12)
        ax.set_title(f"{yvar} vs {xvar} (sample)")
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        fig_sc.tight_layout()
        st.pyplot(fig_sc, clear_figure=True)

    if cat_cols and "LoS" in sample.columns:
        st.markdown("**Boxplot: LoS by a chosen category**")
        group_col = st.selectbox("Group by", options=cat_cols, index=0)
        tmp = sample[[group_col, "LoS"]].dropna()
        if not tmp.empty:
            fig_bp, ax = plt.subplots()
            tmp.boxplot(column="LoS", by=group_col, rot=45, ax=ax)
            ax.set_title(f"Length of Stay by {group_col} (sample)")
            plt.suptitle("")
            ax.set_xlabel(group_col)
            ax.set_ylabel("LoS (minutes)")
            fig_bp.tight_layout()
            st.pyplot(fig_bp, clear_figure=True)

    if "Breachornot" in sample.columns and "LoS" in sample.columns:
        st.subheader("LoS by breach status")
        bf = _breach_flag(sample)
        tmp = sample.copy()
        tmp["BreachFlag"] = bf
        grp = tmp.groupby("BreachFlag")["LoS"].agg(["count", "mean", "median", "min", "max"])
        grp.index = grp.index.map({0: "Non-breach", 1: "Breach"})
        st.dataframe(grp, use_container_width=True)

# ----------------------------
# TASK 5
# ----------------------------
elif section.startswith("Task 5"):
    st.write(f"Using dataset: **{aed_path_label}** (rows={len(aed_df):,})")
    sample = _aed_sample(aed_df, n=int(n_sample), seed=int(seed))
    sample = _coerce_numeric(sample, ["Age", "LoS"])
    if "Breachornot" not in sample.columns:
        st.error("Column 'Breachornot' not found — cannot compute breach rates.")
    else:
        sample["BreachFlag"] = _breach_flag(sample)
        st.metric("Overall breach rate (sample)", f"{sample['BreachFlag'].mean()*100:.2f}%")

        # Prolonged >= 4 hours (240 min)
        if "LoS" in sample.columns:
            sample["Prolonged_4h"] = (sample["LoS"] >= 240).astype(int)

        # Download the sampled rows used for Task 5 (reproducibility)
        st.download_button(
            "Download Task 5 sample (CSV)",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name=f"AED_task5_sample_{int(n_sample)}_seed_{int(seed)}.csv",
            mime="text/csv",
        )

        st.subheader("Breach rate by category")
        cats = [c for c in ["DayofWeek", "HourBand", "ArrivalMode", "Outcome", "TriageCategory"] if c in sample.columns]
        if not cats:
            st.info("No known categorical columns found in this dataset (expected one of DayofWeek/HourBand/ArrivalMode/Outcome/TriageCategory).")
        else:
            pick = st.selectbox("Choose a dimension", cats, index=0)
            rate = (sample.groupby(pick)["BreachFlag"].mean().sort_values(ascending=False) * 100).rename("BreachRate%")
            st.dataframe(rate.to_frame(), use_container_width=True)
            st.pyplot(_fig_bar(rate, f"Breach rate by {pick} (sample)", ylabel="Breach rate (%)"), clear_figure=True)

        # Prolonged-stay rates (≥4 hours) by the same or another category
        if "Prolonged_4h" in sample.columns:
            st.subheader("Prolonged stay (≥4h) rate by category")
            cats2 = [c for c in ["DayofWeek", "HourBand", "ArrivalMode", "Outcome", "TriageCategory", "HRG"] if c in sample.columns]
            if cats2:
                pick2 = st.selectbox("Choose a dimension for prolonged stays", cats2, index=0)
                rate_p = (sample.groupby(pick2)["Prolonged_4h"].mean().sort_values(ascending=False) * 100).rename("ProlongedRate%")
                st.dataframe(rate_p.to_frame(), use_container_width=True)
                st.pyplot(_fig_bar(rate_p, f"Prolonged (≥4h) rate by {pick2} (sample)", ylabel="Rate (%)"), clear_figure=True)
            else:
                st.info("No suitable categorical columns found to break down prolonged stays.")

        # Relationship explorer: crowding / complexity vs LoS
        if "LoS" in sample.columns:
            st.subheader("Relationships with Length of Stay (LoS)")
            rel_num = [c for c in ["noofpatients", "noofinvestigation", "nooftreatment", "Age", "Period"] if c in sample.columns]
            if rel_num:
                x_pick = st.selectbox("Select a numeric driver (X)", rel_num, index=0)
                tmp = sample[[x_pick, "LoS"]].dropna()
                if not tmp.empty:
                    fig_sc, ax = plt.subplots()
                    ax.scatter(tmp[x_pick], tmp["LoS"], s=12)
                    ax.set_title(f"LoS vs {x_pick} (sample)")
                    ax.set_xlabel(x_pick)
                    ax.set_ylabel("LoS (minutes)")
                    fig_sc.tight_layout()
                    st.pyplot(fig_sc, clear_figure=True)
                    st.caption(f"Correlation({x_pick}, LoS) = {tmp[x_pick].corr(tmp['LoS']):.3f}")

        # LoS distribution by a chosen categorical variable (helpful 'relationships' output)
        if "LoS" in sample.columns and cats:
            st.subheader("Length of Stay by category (boxplot)")
            group_col = st.selectbox("Group LoS by", cats, index=0, key="los_group")
            tmp = sample[[group_col, "LoS"]].dropna()
            if not tmp.empty:
                fig_bp, ax = plt.subplots()
                tmp.boxplot(column="LoS", by=group_col, rot=45, ax=ax)
                ax.set_title(f"LoS by {group_col} (sample)")
                plt.suptitle("")
                ax.set_xlabel(group_col)
                ax.set_ylabel("LoS (minutes)")
                fig_bp.tight_layout()
                st.pyplot(fig_bp, clear_figure=True)

        if "Age" in sample.columns:
            st.subheader("Breach rate by age band")
            bins = [0, 18, 35, 50, 65, 80, 200]
            labels = ["0–17", "18–34", "35–49", "50–64", "65–79", "80+"]
            sample["AgeBand"] = pd.cut(sample["Age"], bins=bins, labels=labels, right=False)
            rate_age = (sample.groupby("AgeBand")["BreachFlag"].mean() * 100).rename("BreachRate%")
            st.dataframe(rate_age.to_frame(), use_container_width=True)
            st.pyplot(_fig_bar(rate_age, "Breach rate by age band (sample)", ylabel="Breach rate (%)"), clear_figure=True)

        # Crowd/pressure relationship: noofpatients vs LoS (if available)
        if "noofpatients" in sample.columns and "LoS" in sample.columns:
            st.subheader("Departmental pressure vs LoS")
            tmp = sample[["noofpatients", "LoS"]].dropna()
            if not tmp.empty:
                fig_sc, ax = plt.subplots()
                ax.scatter(tmp["noofpatients"], tmp["LoS"], s=12)
                ax.set_title("noofpatients vs LoS (sample)")
                ax.set_xlabel("noofpatients")
                ax.set_ylabel("LoS (minutes)")
                fig_sc.tight_layout()
                st.pyplot(fig_sc, clear_figure=True)
                st.caption(f"Correlation(noofpatients, LoS) = {tmp['noofpatients'].corr(tmp['LoS']):.3f}")

# ----------------------------
# TASK 6
# ----------------------------
elif section.startswith("Task 6"):
    st.write(f"Using dataset: **{aed_path_label}** (rows={len(aed_df):,})")
    try:
        results = _run_task6_ml(aed_df, seed=int(seed), target_recall=float(target_recall))
    except Exception as e:
        st.error(str(e))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Threshold (chosen on val)", f"{results['threshold']:.3f}")
        c2.metric("ROC AUC (test)", f"{results['roc_auc']:.3f}" if results["roc_auc"] is not None else "n/a")
        c3.metric("PR AUC (test)", f"{results['pr_auc']:.3f}" if results["pr_auc"] is not None else "n/a")
        c4.metric("Test breach rate", f"{results['rates']['test']*100:.2f}%")

        st.subheader("Train/Val/Test sizes")
        st.json(results["sizes"])

        st.subheader("Curves")
        col1, col2 = st.columns(2)
        col1.pyplot(results["fig_roc"], clear_figure=True)
        col2.pyplot(results["fig_pr"], clear_figure=True)

        st.subheader("Confusion matrix (test)")
        cm = results["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Classification report (test)")
        rep = pd.DataFrame(results["report"]).T
        st.dataframe(rep, use_container_width=True)

# ----------------------------
# TASK 7
# ----------------------------
elif section.startswith("Task 7"):
    st.write(f"File-backed data manager: **{AED_FILE_PATH}**")
    mgr = AEDDataManager(AED_FILE_PATH, id_col="ID", audit_log_path=AUDIT_LOG_PATH)
    info = mgr.load()

    st.success(f"Loaded {info.n_rows:,} rows × {info.n_cols:,} columns. Audit log: {AUDIT_LOG_PATH}")

    st.subheader("Lookup patient by ID")
    pid = st.text_input("Patient ID", value="")
    if pid:
        try:
            out = mgr.get_patient(pid)
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(str(e))

    st.subheader("Filter numeric range")
    num_cols = [c for c in mgr.df.columns if pd.api.types.is_numeric_dtype(mgr.df[c]) or c in ["Age", "LoS"]]
    col = st.selectbox("Column", options=sorted(set(num_cols)), index=0 if num_cols else None)
    c1, c2, c3 = st.columns(3)
    min_v = c1.number_input("Min (optional)", value=0.0)
    max_v = c2.number_input("Max (optional)", value=0.0)
    use_min = c1.checkbox("Use min", value=False)
    use_max = c2.checkbox("Use max", value=False)
    if st.button("Apply filter"):
        try:
            out = mgr.filter_range(col, min_value=min_v if use_min else None, max_value=max_v if use_max else None)
            st.dataframe(out.head(200), use_container_width=True)
            st.caption(f"Showing first 200 of {len(out):,} matches.")
        except Exception as e:
            st.error(str(e))

    st.subheader("Update a single field (by ID)")
    up_pid = st.text_input("ID to update", value="", key="upd_id")
    if up_pid:
        upd_col = st.selectbox("Column to update", options=list(mgr.df.columns), index=0, key="upd_col")
        new_val = st.text_input("New value (stored as string; CSV types may coerce on reload)", value="", key="upd_val")
        if st.button("Update", key="upd_btn"):
            try:
                n = mgr.update_patient_field(up_pid, upd_col, new_val)
                st.success(f"Updated {n} row(s).")
            except Exception as e:
                st.error(str(e))

    st.subheader("Delete patient (by ID)")
    del_pid = st.text_input("ID to delete", value="", key="del_id")
    if del_pid and st.button("Delete", key="del_btn"):
        try:
            deleted = mgr.delete_patient(del_pid)
            st.warning(f"Deleted {deleted} row(s).")
        except Exception as e:
            st.error(str(e))

    st.subheader("Preview current in-memory dataframe")
    st.dataframe(mgr.df.head(50), use_container_width=True)

# ----------------------------
# TASK 8
# ----------------------------
elif section.startswith("Task 8"):
    st.write(f"Audit log file: **{AUDIT_LOG_PATH}**")
    lines = _read_audit_log(AUDIT_LOG_PATH)
    if not lines:
        st.info("No audit log found yet. Go to Task 7 and perform an action (load/filter/update/delete/save).")
    else:
        st.subheader("Recent audit entries")
        st.code("\n".join(lines[-200:]))

        # Parse action counts
        actions = []
        for ln in lines:
            m = None
            # format: timestamp - action=... | id=... | details=...
            if "action=" in ln:
                try:
                    part = ln.split("action=", 1)[1]
                    action = part.split("|", 1)[0].strip()
                    actions.append(action)
                except Exception:
                    pass
        if actions:
            s = pd.Series(actions).value_counts()
            st.subheader("Action counts")
            st.dataframe(s.to_frame("count"), use_container_width=True)
            st.pyplot(_fig_bar(s, "Audit actions frequency", ylabel="Count"), clear_figure=True)