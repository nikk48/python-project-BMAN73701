from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
import pulp

from lp.models import Operator, SolveResult

def get_solver(msg: bool = False):
    return pulp.HiGHS_CMD(msg=msg)

def build_default_instance() -> Tuple[List[Operator], List[str], float]:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    hours_per_day = 14.0

    ops = [
        Operator("E. Khan",   25, 8, {"Mon": 6, "Tue": 0, "Wed": 6, "Thu": 0, "Fri": 6}),
        Operator("Y. Chen",   26, 8, {"Mon": 0, "Tue": 6, "Wed": 0, "Thu": 6, "Fri": 0}),
        Operator("A. Taylor", 24, 8, {"Mon": 4, "Tue": 8, "Wed": 4, "Thu": 0, "Fri": 4}),
        Operator("R. Zidane", 23, 8, {"Mon": 5, "Tue": 5, "Wed": 5, "Thu": 0, "Fri": 5}),
        Operator("R. Perez",  28, 7, {"Mon": 3, "Tue": 0, "Wed": 3, "Thu": 8, "Fri": 0}),
        Operator("C. Santos", 30, 7, {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 6, "Fri": 2}),
    ]
    return ops, days, hours_per_day


def build_decision_variables(operators: List[Operator], days: List[str]):
    return pulp.LpVariable.dicts(
        "hours",
        ((op.name, d) for op in operators for d in days),
        lowBound=0,
        cat="Continuous",
    )

def add_base_constraints(model: pulp.LpProblem, x, operators: List[Operator], days: List[str], hours_per_day: float) -> None:
    for d in days:
        model += (pulp.lpSum(x[(op.name, d)] for op in operators) == hours_per_day, f"Coverage_{d}")

    for op in operators:
        for d in days:
            model += (x[(op.name, d)] <= op.availability.get(d, 0.0), f"Avail_{op.name}_{d}")
        model += (pulp.lpSum(x[(op.name, d)] for d in days) >= op.min_weekly_hours, f"MinWeek_{op.name}")

def extract_solution(x, operators: List[Operator], days: List[str]) -> Dict[str, Dict[str, float]]:
    return {op.name: {d: float(pulp.value(x[(op.name, d)]) or 0.0) for d in days} for op in operators}

def total_cost_expression(x, operators: List[Operator], days: List[str]):
    return pulp.lpSum(op.wage * x[(op.name, d)] for op in operators for d in days)

# solve_task1_min_cost
def solve_task1_min_cost(operators: List[Operator], days: List[str], hours_per_day: float) -> SolveResult:
    total_weekly_hours = hours_per_day * len(days)
    mu = total_weekly_hours / len(operators)

    model = pulp.LpProblem("Task1_MinCost", pulp.LpMinimize)
    x = build_decision_variables(operators, days)

    cost_expr = total_cost_expression(x, operators, days)
    model += cost_expr
    add_base_constraints(model, x, operators, days, hours_per_day)

    status_code = model.solve(get_solver(msg=False))
    status = pulp.LpStatus[status_code]

    hours = extract_solution(x, operators, days)
    total_cost = float(pulp.value(cost_expr) or 0.0)

    return SolveResult(status=status, hours=hours, total_cost=total_cost, avg_hours=mu, delta=None, cost_increase_pct=0.0)

# solve_task2
def _build_fairness_model(
    fairness_metric: str,
    operators: List[Operator],
    days: List[str],
    hours_per_day: float,
    cost_cap: Optional[float],
    model_name: str,
):
    total_weekly_hours = hours_per_day * len(days)
    mu = total_weekly_hours / len(operators)

    model = pulp.LpProblem(model_name, pulp.LpMinimize)
    x = build_decision_variables(operators, days)

    add_base_constraints(model, x, operators, days, hours_per_day)

    cost_expr = total_cost_expression(x, operators, days)
    if cost_cap is not None:
        model += (cost_expr <= cost_cap, "CostCap")

    if fairness_metric == "delta":
        fair_var = pulp.LpVariable("Delta", lowBound=0, cat="Continuous")
        for op in operators:
            H_i = pulp.lpSum(x[(op.name, d)] for d in days)
            model += (H_i - mu <= fair_var, f"Fair_Above_{op.name}")
            model += (mu - H_i <= fair_var, f"Fair_Below_{op.name}")

    elif fairness_metric == "range":
        weekly = {op.name: pulp.lpSum(x[(op.name, d)] for d in days) for op in operators}
        Hmax = pulp.LpVariable("Hmax", lowBound=0, cat="Continuous")
        Hmin = pulp.LpVariable("Hmin", lowBound=0, cat="Continuous")
        fair_var = pulp.LpVariable("FairnessRange", lowBound=0, cat="Continuous")
        for op in operators:
            model += (Hmax >= weekly[op.name], f"MaxDef_{op.name}")
            model += (Hmin <= weekly[op.name], f"MinDef_{op.name}")
        model += (fair_var == Hmax - Hmin, "RangeDef")
    else:
        raise ValueError("fairness_metric must be 'range' or 'delta'")

    return model, x, fair_var, cost_expr, mu


def solve_task2_lexicographic(
    fairness_metric: str,
    operators: List[Operator],
    days: List[str],
    hours_per_day: float,
    baseline_cost: float,
    cost_cap: Optional[float],
    model_prefix: str
) -> SolveResult:
    mA, xA, fairA, costA, mu = _build_fairness_model(
        fairness_metric=fairness_metric,
        operators=operators,
        days=days,
        hours_per_day=hours_per_day,
        cost_cap=cost_cap,
        model_name=f"{model_prefix}_A_MinFairness"
    )
    mA += fairA
    statusA = pulp.LpStatus[mA.solve(get_solver(msg=False))]
    if statusA != "Optimal":
        return SolveResult(status=statusA, hours={}, total_cost=0.0, avg_hours=mu, delta=None, cost_increase_pct=0.0)

    fair_star = float(pulp.value(fairA) or 0.0)

    mB, xB, fairB, costB, muB = _build_fairness_model(
        fairness_metric=fairness_metric,
        operators=operators,
        days=days,
        hours_per_day=hours_per_day,
        cost_cap=cost_cap,
        model_name=f"{model_prefix}_B_MinCostGivenFairness"
    )
    mB += (fairB <= fair_star + 1e-6, "FairnessLock")
    mB += costB

    statusB = pulp.LpStatus[mB.solve(get_solver(msg=False))]

    hours = extract_solution(xB, operators, days)
    total_cost = float(pulp.value(costB) or 0.0)
    fairness_val = float(pulp.value(fairB) or 0.0)
    increase_pct = ((total_cost - baseline_cost) / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0

    return SolveResult(status=statusB, hours=hours, total_cost=total_cost, avg_hours=muB, delta=fairness_val, cost_increase_pct=increase_pct)

# solve_task3_min_cost_with_skills
def solve_task3_min_cost_with_skills(
    operators: List[Operator],
    days: List[str],
    hours_per_day: float,
    baseline_cost: float,
    skill_sets: Dict[str, Set[str]],
    daily_skill_req: float = 6.0
) -> SolveResult:
    total_weekly_hours = hours_per_day * len(days)
    mu = total_weekly_hours / len(operators)

    model = pulp.LpProblem("Task3_MinCost_WithSkills", pulp.LpMinimize)
    x = build_decision_variables(operators, days)
    cost_expr = total_cost_expression(x, operators, days)
    model += cost_expr

    add_base_constraints(model, x, operators, days, hours_per_day)

    for d in days:
        for skill_name, skilled_ops in skill_sets.items():
            model += (pulp.lpSum(x[(op_name, d)] for op_name in skilled_ops) >= daily_skill_req, f"Skill_{skill_name}_{d}")

    status = pulp.LpStatus[model.solve(get_solver(msg=False))]
    hours = extract_solution(x, operators, days)
    total_cost = float(pulp.value(cost_expr) or 0.0)
    increase_pct = ((total_cost - baseline_cost) / baseline_cost) * 100.0 if baseline_cost > 0 else 0.0

    return SolveResult(
        status=status,
        hours=hours if status == "Optimal" else {},
        total_cost=total_cost if status == "Optimal" else 0.0,
        avg_hours=mu,
        delta=None,
        cost_increase_pct=increase_pct if status == "Optimal" else 0.0
    )


