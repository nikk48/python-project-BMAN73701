from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class Operator:
    name: str
    wage: float
    min_weekly_hours: float
    availability: Dict[str, float]

@dataclass(frozen=True)
class SolveResult:
    status: str
    hours: Dict[str, Dict[str, float]]
    total_cost: float
    avg_hours: float
    delta: Optional[float]
    cost_increase_pct: float
