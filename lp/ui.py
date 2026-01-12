# lp/ui.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, Set

from cli_utils import print_header, ask_int, ask_str, pause
from lp.analysis import schedule_to_dataframe, compute_weekly_hours, compute_delta_from_weekly_hours, assert_coverage
from lp.solver import (
    build_default_instance,
    solve_task1_min_cost,
    solve_task2_lexicographic,
    solve_task3_min_cost_with_skills,
)
from lp.models import SolveResult
import pandas as pd

@dataclass
class LPContext:
    operators: list
    days: list
    hours_per_day: float
    baseline: Optional[SolveResult] = None
    scenario_i: Optional[SolveResult] = None
    scenario_ii: Optional[SolveResult] = None
    skills: Optional[SolveResult] = None
    metric: Optional[str] = None

_CTX: Optional[LPContext] = None

def _choose_metric() -> str:
    print_header("Choose Task 2 fairness metric")
    print("1) Range: Hmax − Hmin")
    print("2) Delta: max_i |H_i − μ|")
    c = ask_int("Choose 1 or 2 (default=1): ", default=1, valid={1, 2})
    return "range" if c == 1 else "delta"

def _require_ctx() -> LPContext:
    global _CTX
    if _CTX is None or _CTX.baseline is None:
        raise RuntimeError("Run LP optimisation first (option 1).")
    return _CTX

def _pick_solution(ctx: LPContext) -> Tuple[str, SolveResult]:
    options: Dict[str, SolveResult] = {
        "Task 1 (baseline)": ctx.baseline,
        "Task 2 (Scenario i)": ctx.scenario_i,
        "Task 2 (Scenario ii)": ctx.scenario_ii,
        "Task 3 (skills)": ctx.skills,
    }
    names = [k for k, v in options.items() if v is not None]
    print_header("Select schedule to inspect")
    for i, n in enumerate(names, 1):
        print(f"{i}) {n}")
    idx = ask_int("Choose: ", default=1, valid=set(range(1, len(names) + 1)))
    key = names[idx - 1]
    return key, options[key]

def _run_lp_all() -> None:
    global _CTX
    operators, days, hours_per_day = build_default_instance()

    task1 = solve_task1_min_cost(operators, days, hours_per_day)
    if task1.status != "Optimal":
        raise RuntimeError(f"Task 1 not optimal. Status={task1.status}")

    weekly = compute_weekly_hours(task1.hours, days)
    delta = compute_delta_from_weekly_hours(weekly, task1.avg_hours)
    task1 = SolveResult(
        status=task1.status, hours=task1.hours, total_cost=task1.total_cost,
        avg_hours=task1.avg_hours, delta=delta, cost_increase_pct=0.0
    )

    metric = _choose_metric()
    Z1 = task1.total_cost
    cap = 1.018 * Z1

    scen_i = solve_task2_lexicographic(
        fairness_metric=metric, operators=operators, days=days,
        hours_per_day=hours_per_day, baseline_cost=Z1,
        cost_cap=cap, model_prefix=f"Task2_ScenarioI_{metric.upper()}"
    )

    scen_ii = solve_task2_lexicographic(
        fairness_metric=metric, operators=operators, days=days,
        hours_per_day=hours_per_day, baseline_cost=Z1,
        cost_cap=None, model_prefix=f"Task2_ScenarioII_{metric.upper()}"
    )

    skill_sets = {
        "Programming": {"E. Khan", "Y. Chen", "R. Perez", "C. Santos"},
        "Troubleshooting": {"A. Taylor", "R. Zidane", "C. Santos"},
    }
    task3 = solve_task3_min_cost_with_skills(
        operators=operators, days=days, hours_per_day=hours_per_day,
        baseline_cost=Z1, skill_sets=skill_sets, daily_skill_req=6.0
    )

    _CTX = LPContext(
        operators=operators, days=days, hours_per_day=hours_per_day,
        baseline=task1, scenario_i=scen_i, scenario_ii=scen_ii, skills=task3, metric=metric
    )

def lp_menu() -> None:
    while True:
        print_header("LP SCHEDULING MENU (Task 1/2/3)")
        print("1) Run / refresh optimisation")
        print("2) Show overall cost")
        print("3) Show weekly hours for all operators")
        print("4) Show schedule for one operator")
        print("5) Show schedule for a day")
        print("6) Show overall schedule table")
        print("7) Report-ready summary")
        print("0) Back")

        c = ask_int("Choose: ", default=1, valid={0,1,2,3,4,5,6,7})
        if c == 0:
            return

        if c == 1:
            _run_lp_all()
            print("\nLP results computed and cached.")
            pause()
            continue

        ctx = _require_ctx()
        name, sol = _pick_solution(ctx)
        if sol.status != "Optimal":
            print_header(f"{name}")
            print(f"Status: {sol.status} (no schedule to display)")
            pause()
            continue

        df = schedule_to_dataframe(sol.hours, ctx.days)

        if c == 2:
            print_header(f"Overall cost — {name}")
            print(f"Total cost: £{sol.total_cost:,.2f}")
            if sol.delta is not None:
                print(f"Fairness value: {sol.delta:.4f}")
                print(f"Cost increase vs Task 1: {sol.cost_increase_pct:.3f}%")
            pause()

        elif c == 3:
            print_header(f"Weekly hours — {name}")
            weekly = compute_weekly_hours(sol.hours, ctx.days)
            s = pd.Series(weekly).sort_values(ascending=False)
            print(s.to_string(float_format=lambda x: f"{x:0.2f}"))
            pause()

        elif c == 4:
            ops = [op.name for op in ctx.operators]
            op_name = ask_str("Enter operator name: ", valid=set(ops))
            print_header(f"Schedule for {op_name} — {name}")
            row = df.loc[op_name, ctx.days + ["WeeklyTotal"]]
            print(row.to_string(float_format=lambda x: f"{x:0.2f}"))
            pause()

        elif c == 5:
            day = ask_str(f"Enter day {ctx.days}: ", valid=set(ctx.days))
            print_header(f"Schedule on {day} — {name}")
            col = df[day].sort_values(ascending=False)
            print(col.to_string(float_format=lambda x: f"{x:0.2f}"))
            print(f"\nTotal: {col.sum():0.2f} (required {ctx.hours_per_day:0.2f})")
            pause()

        elif c == 6:
            print_header(f"Overall schedule — {name}")
            assert_coverage(df, ctx.days, ctx.hours_per_day)
            print(df.to_string(float_format=lambda x: f"{x:0.2f}"))
            pause()

        elif c == 7:
            print_header("REPORT-READY SUMMARY (LP)")
            print(f"Task 1 cost: £{ctx.baseline.total_cost:,.2f}")
            print(f"Task 1 inequity (delta): {ctx.baseline.delta:.4f}")
            print(f"Task 2 metric: {ctx.metric.upper()}")
            if ctx.scenario_i:
                print(f"Scenario (i): status={ctx.scenario_i.status} | fairness={ctx.scenario_i.delta} | cost=£{ctx.scenario_i.total_cost:,.2f} | +{ctx.scenario_i.cost_increase_pct:.3f}%")
            if ctx.scenario_ii:
                print(f"Scenario (ii): status={ctx.scenario_ii.status} | fairness={ctx.scenario_ii.delta} | cost=£{ctx.scenario_ii.total_cost:,.2f} | +{ctx.scenario_ii.cost_increase_pct:.3f}%")
            if ctx.skills:
                print(f"Task 3: status={ctx.skills.status} | cost=£{ctx.skills.total_cost:,.2f} | +{ctx.skills.cost_increase_pct:.3f}%")
            pause()
