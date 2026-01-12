from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

from cli_utils import print_header
from exceptions import DataValidationError
from ml.breach_prediction import run_task6_breach_prediction


def ml_menu(file_path, seed: int, target_recall: float) -> None:
    """Menu entry for Task 6 ML breach prediction."""

    print_header("TASK 6 — A&E breach prediction (ML)")
    p = Path(file_path).expanduser().resolve()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"AED CSV not found: {file_path}")

    df = pd.read_csv(file_path)
    if "Breachornot" not in df.columns:
        raise DataValidationError("Task 6 requires 'Breachornot' column in AED dataset.")

    run_task6_breach_prediction(file_path=file_path, seed=seed, target_recall=target_recall)
