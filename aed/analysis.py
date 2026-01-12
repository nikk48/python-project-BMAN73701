from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
    STATS_MODELS_OK = True
except Exception:
    STATS_MODELS_OK = False


# Task 4 — Descriptive summaries + plots
def _management_numeric_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute management-friendly summary statistics for numeric columns."""
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for c in num_cols:
        s = df_in[c]
        rows.append({
            "Variable": c,
            "N": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Mean": float(s.mean()) if s.notna().any() else np.nan,
            "Median": float(s.median()) if s.notna().any() else np.nan,
            "Std": float(s.std()) if s.notna().any() else np.nan,
            "Min": float(s.min()) if s.notna().any() else np.nan,
            "P25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "P75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "Max": float(s.max()) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _management_categorical_summary(df_in: pd.DataFrame, top_k: int = 12) -> Dict[str, pd.DataFrame]:
    """Return top-k frequency tables for non-numeric columns (incl. NaNs)."""
    cat_cols = df_in.select_dtypes(exclude=[np.number]).columns.tolist()
    out: Dict[str, pd.DataFrame] = {}
    for c in cat_cols:
        s = df_in[c]
        vc = s.value_counts(dropna=False)
        total = len(s)
        tab = pd.DataFrame({"Count": vc.values, "Percent": (vc.values / total * 100.0)}, index=vc.index)
        tab.index.name = c
        out[c] = tab.head(top_k)
    return out


def _hist(series: pd.Series, title: str, xlabel: str, bins: int = 20) -> None:
    plt.figure()
    series.dropna().hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def _bar_counts(series: pd.Series, title: str, xlabel: str) -> None:
    plt.figure()
    series.value_counts(dropna=False).plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.show()


# Task 5 — Breaches / prolonged stay drivers
def _make_breach_flag(df: pd.DataFrame, col: str = "Breachornot") -> pd.Series:
    """Convert Breachornot into a robust 0/1 flag."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0).astype(int) == 1).astype(int)

    ss = s.astype(str).str.strip().str.lower()
    is_non = ss.str.contains("non") & ss.str.contains("breach")
    is_breach = ss.str.contains("breach") & ~is_non
    is_breach = is_breach | ss.isin(["1", "yes", "true", "y"])
    return is_breach.astype(int)


def _add_age_band(df: pd.DataFrame, age_col: str = "Age") -> None:
    if age_col not in df.columns:
        return
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    bins = [-np.inf, 15, 35, 55, 75, np.inf]
    labels = ["0–15", "16–35", "36–55", "56–75", "76+"]
    df["AgeBand"] = pd.cut(df[age_col], bins=bins, labels=labels)


def _box_by_group(df: pd.DataFrame, y: str, group: str, title: str) -> None:
    plt.figure()
    df.boxplot(column=y, by=group, rot=45)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel(group)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def _rate_bar(rate_series: pd.Series, title: str, ylabel: str) -> None:
    plt.figure()
    rate_series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def run_task5_aed_drivers(sample: pd.DataFrame) -> None:
    """Task 5: explore potential drivers of breach and prolonged stays."""
    df = sample.copy()

    print("\n" + "=" * 90)
    print("TASK 5 — BREACHES & PROLONGED STAYS: POTENTIAL CONTRIBUTING FACTORS (SAMPLE)")
    print("=" * 90)

    if "LoS" in df.columns:
        df["LoS"] = pd.to_numeric(df["LoS"], errors="coerce")
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    if "Breachornot" in df.columns:
        df["BreachFlag"] = _make_breach_flag(df, "Breachornot")
        print(f"Overall breach rate: {df['BreachFlag'].mean() * 100:.2f}%")
    else:
        df["BreachFlag"] = np.nan
        print("NOTE: 'Breachornot' not found — breach analysis limited.")

    if "LoS" in df.columns:
        df["Prolonged_4h"] = (df["LoS"] >= 240).astype(int)
        q80 = df["LoS"].quantile(0.80)
        df["Prolonged_Top20"] = (df["LoS"] >= q80).astype(int)
        print(f"Prolonged (LoS ≥ 240 mins): {df['Prolonged_4h'].mean() * 100:.2f}%")
        print(f"Prolonged (Top 20% LoS; threshold={q80:.1f} mins): {df['Prolonged_Top20'].mean() * 100:.2f}%")
    else:
        df["Prolonged_4h"] = np.nan
        df["Prolonged_Top20"] = np.nan
        print("NOTE: 'LoS' not found — prolonged stay analysis limited.")

    _add_age_band(df, "Age")

    if "LoS" in df.columns and df["BreachFlag"].notna().any():
        _box_by_group(df, "LoS", "BreachFlag", "Length of Stay by Breach Status (0=non-breach, 1=breach)")
        print("\nLoS summary by breach flag:")
        print(
            df.groupby("BreachFlag")["LoS"].agg(["count", "mean", "median", "min", "max"]).to_string(
                float_format=lambda x: f"{x:,.2f}"
            )
        )

    if "DayofWeek" in df.columns:
        if df["BreachFlag"].notna().any():
            rate = df.groupby("DayofWeek")["BreachFlag"].mean().sort_values(ascending=False) * 100
            print("\nBreach rate (%) by Day of Week:")
            print(rate.to_string(float_format=lambda x: f"{x:,.2f}"))
            _rate_bar(rate, "Breach rate by Day of Week (sample)", "Breach rate (%)")

        if df["Prolonged_4h"].notna().any():
            rate = df.groupby("DayofWeek")["Prolonged_4h"].mean().sort_values(ascending=False) * 100
            print("\nProlonged (≥4h) rate (%) by Day of Week:")
            print(rate.to_string(float_format=lambda x: f"{x:,.2f}"))
            _rate_bar(rate, "Prolonged stay (≥4h) rate by Day of Week (sample)", "Rate (%)")

        if "LoS" in df.columns:
            _box_by_group(df, "LoS", "DayofWeek", "Length of Stay by Day of Week (sample)")

    if "AgeBand" in df.columns:
        if df["BreachFlag"].notna().any():
            rate = df.groupby("AgeBand")["BreachFlag"].mean() * 100
            print("\nBreach rate (%) by Age Band:")
            print(rate.to_string(float_format=lambda x: f"{x:,.2f}"))
            _rate_bar(rate, "Breach rate by Age Band (sample)", "Breach rate (%)")

        if df["Prolonged_4h"].notna().any():
            rate = df.groupby("AgeBand")["Prolonged_4h"].mean() * 100
            print("\nProlonged (≥4h) rate (%) by Age Band:")
            print(rate.to_string(float_format=lambda x: f"{x:,.2f}"))
            _rate_bar(rate, "Prolonged stay (≥4h) rate by Age Band (sample)", "Rate (%)")

        if "LoS" in df.columns:
            _box_by_group(df, "LoS", "AgeBand", "Length of Stay by Age Band (sample)")

    if "Age" in df.columns and "LoS" in df.columns:
        tmp = df[["Age", "LoS"]].dropna()
        plt.figure()
        plt.scatter(tmp["Age"], tmp["LoS"], s=12)
        plt.title("Age vs Length of Stay (sample)")
        plt.xlabel("Age (years)")
        plt.ylabel("LoS (minutes)")
        plt.tight_layout()
        plt.show()
        print(f"\nCorrelation(Age, LoS): {tmp['Age'].corr(tmp['LoS']):.3f}")

    if not STATS_MODELS_OK:
        print("\n(statsmodels not installed) Skipping logistic regression odds ratios.")
        print("Install with:  pip install statsmodels")
        return

    if df["BreachFlag"].notna().any() and df["BreachFlag"].nunique(dropna=True) > 1:
        terms = []
        if "DayofWeek" in df.columns:
            terms.append("C(DayofWeek)")
        if "AgeBand" in df.columns:
            terms.append("C(AgeBand)")
        if terms:
            model_df = df.dropna(subset=["BreachFlag"]).copy()
            try:
                m = smf.logit("BreachFlag ~ " + " + ".join(terms), data=model_df).fit(disp=False)
                or_table = pd.DataFrame({"OR": np.exp(m.params), "p_value": m.pvalues})
                print("\nLogistic regression (Breach) — Odds Ratios:")
                print(or_table.to_string(float_format=lambda x: f"{x:,.3f}"))
            except Exception as e:
                print("\nCould not fit breach logistic model (often too few breaches).")
                print("Error:", e)

    if df["Prolonged_4h"].notna().any() and df["Prolonged_4h"].nunique(dropna=True) > 1:
        terms = []
        if "DayofWeek" in df.columns:
            terms.append("C(DayofWeek)")
        if "AgeBand" in df.columns:
            terms.append("C(AgeBand)")
        if terms:
            model_df = df.dropna(subset=["Prolonged_4h"]).copy()
            try:
                m = smf.logit("Prolonged_4h ~ " + " + ".join(terms), data=model_df).fit(disp=False)
                or_table = pd.DataFrame({"OR": np.exp(m.params), "p_value": m.pvalues})
                print("\nLogistic regression (Prolonged LoS ≥ 4h) — Odds Ratios:")
                print(or_table.to_string(float_format=lambda x: f"{x:,.3f}"))
            except Exception as e:
                print("\nCould not fit prolonged-stay logistic model.")
                print("Error:", e)


def run_aed_analysis(file_path: str, seed: int = 42, n: int = 400) -> None:
    """Task 4: sample the AED data and produce summary tables + plots; then run Task 5 drivers."""
    df = pd.read_csv(file_path)
    if n > len(df):
        n = len(df)
    sample = df.sample(n=n, random_state=seed).copy()

    print("\n" + "=" * 90)
    print(f"A&E ANALYSIS — Random sample of {n} patients (seed={seed})")
    print("=" * 90)
    print("Columns:", list(sample.columns))

    for col in ["Age", "LoS"]:
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")

    if "Breachornot" in sample.columns:
        sample["Breachornot"] = sample["Breachornot"].astype(str).str.strip().str.lower()

    if "DayofWeek" in sample.columns:
        sample["DayofWeek"] = sample["DayofWeek"].astype(str).str.strip()

    # ---- Task 4 summaries ----
    num_summary = _management_numeric_summary(sample)
    print("\nNUMERIC SUMMARY (sample):")
    print(num_summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    cat_summary = _management_categorical_summary(sample, top_k=12)
    print("\nCATEGORICAL SUMMARIES (top categories, incl. NaN):")
    for col, tab in cat_summary.items():
        print("\n" + "-" * 90)
        print(col)
        print(tab.to_string(float_format=lambda x: f"{x:,.2f}"))

    if "Age" in sample.columns:
        _hist(sample["Age"], f"Age distribution (sample of {n})", "Age (years)", bins=20)
    if "LoS" in sample.columns:
        _hist(sample["LoS"], f"Length of stay distribution (sample of {n})", "LoS (minutes)", bins=20)
    if "DayofWeek" in sample.columns:
        _bar_counts(sample["DayofWeek"], f"Attendances by day of week (sample of {n})", "Day of week")

    if "LoS" in sample.columns and "Breachornot" in sample.columns:
        plt.figure()
        sample.boxplot(column="LoS", by="Breachornot")
        plt.title("Length of stay by breach status (sample)")
        plt.suptitle("")
        plt.xlabel("Breach status")
        plt.ylabel("LoS (minutes)")
        plt.tight_layout()
        plt.show()

        grp = sample.groupby("Breachornot")["LoS"].agg(["count", "mean", "median", "min", "max"])
        print("\nLoS by breach status:")
        print(grp.to_string(float_format=lambda x: f"{x:,.2f}"))

    if "LoS" in sample.columns and "DayofWeek" in sample.columns:
        plt.figure()
        sample.boxplot(column="LoS", by="DayofWeek", rot=45)
        plt.title("Length of stay by day of week (sample)")
        plt.suptitle("")
        plt.xlabel("Day of week")
        plt.ylabel("LoS (minutes)")
        plt.tight_layout()
        plt.show()

        dow_stats = sample.groupby("DayofWeek")["LoS"].agg(["count", "mean", "median"])
        print("\nLoS by day of week:")
        print(dow_stats.sort_values("mean", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

    if "Age" in sample.columns and "LoS" in sample.columns:
        tmp = sample[["Age", "LoS"]].dropna()
        plt.figure()
        plt.scatter(tmp["Age"], tmp["LoS"], s=12)
        plt.title("Age vs Length of stay (sample)")
        plt.xlabel("Age (years)")
        plt.ylabel("LoS (minutes)")
        plt.tight_layout()
        plt.show()

        corr = tmp["Age"].corr(tmp["LoS"])
        print(f"\nCorrelation(Age, LoS) in sample: {corr:,.3f}")

    # ---- Task 5 drivers ----
    run_task5_aed_drivers(sample)

    print("\nA&E analysis complete.\n")
