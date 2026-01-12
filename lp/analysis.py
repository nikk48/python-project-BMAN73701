from __future__ import annotations
from typing import Dict, List
import pandas as pd

def schedule_to_dataframe(hours: Dict[str, Dict[str, float]], days: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(hours).T
    df = df[days]
    df["WeeklyTotal"] = df.sum(axis=1)
    return df

def compute_weekly_hours(hours: Dict[str, Dict[str, float]], days: List[str]) -> Dict[str, float]:
    return {op: sum(hours[op][d] for d in days) for op in hours.keys()}

def compute_delta_from_weekly_hours(weekly_hours: Dict[str, float], mu: float) -> float:
    return max(abs(h - mu) for h in weekly_hours.values())

def assert_coverage(df: pd.DataFrame, days: List[str], hours_per_day: float) -> None:
    tol = 1e-6
    totals = df[days].sum(axis=0)
    for d in days:
        if abs(totals[d] - hours_per_day) > tol:
            raise ValueError(f"Coverage violated for {d}: expected {hours_per_day}, got {totals[d]}")
