from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from audit_log import audit, get_audit_logger


@dataclass
class LoadResult:
    file_path: Path
    n_rows: int
    n_cols: int
    id_col: str


class AEDDataManager:
    """Interactive data manager for the AED CSV.

    Supports:
      - Look up a patient by ID
      - Range filtering on a chosen variable
      - Deleting or modifying records
      - Saving changes back to CSV (with automatic backups)
      - Audit logging of user actions
    """

    def __init__(self, file_path: str | Path, *, id_col: str = "ID", audit_log_path: Optional[Path] = None):
        self.file_path = Path(file_path).expanduser().resolve()
        self.id_col = id_col
        self.df: pd.DataFrame = pd.DataFrame()
        self.dirty: bool = False

        if audit_log_path is None:
            audit_log_path = self.file_path.parent / "logs" / "audit.log"
        self.logger = get_audit_logger(audit_log_path)

    # ------------------------- loading / saving -------------------------
    def load(self) -> LoadResult:
        if not self.file_path.exists():
            raise FileNotFoundError(f"AED CSV not found: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        if self.id_col not in self.df.columns:
            raise ValueError(
                f"ID column '{self.id_col}' not found. Available columns: {list(self.df.columns)}"
            )
        # Normalize IDs to string (but keep other columns unchanged)
        self.df[self.id_col] = self.df[self.id_col].astype(str)
        self.dirty = False
        audit(self.logger, "load", details=f"file={self.file_path}")
        return LoadResult(file_path=self.file_path, n_rows=len(self.df), n_cols=self.df.shape[1], id_col=self.id_col)

    def _backup_current_file(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.file_path.with_suffix(self.file_path.suffix + f".bak_{ts}")
        shutil.copy2(self.file_path, backup_path)
        return backup_path

    def save(self, *, out_path: Optional[str | Path] = None) -> Path:
        if self.df.empty:
            raise ValueError("No data loaded.")

        if out_path is None:
            backup = self._backup_current_file()
            target = self.file_path
            audit(self.logger, "backup", details=f"backup={backup}")
        else:
            target = Path(out_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(target, index=False)
        self.dirty = False
        audit(self.logger, "save", details=f"target={target}")
        return target

    # ------------------------- queries -------------------------
    def get_patient(self, patient_id: str) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("No data loaded.")
        pid = str(patient_id).strip()
        out = self.df.loc[self.df[self.id_col] == pid].copy()
        audit(self.logger, "get_patient", patient_id=pid, details=f"matches={len(out)}")
        return out

    def filter_range(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("No data loaded.")
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found.")

        s = pd.to_numeric(self.df[column], errors="coerce")
        mask = pd.Series(True, index=self.df.index)
        if min_value is not None:
            mask &= s >= float(min_value)
        if max_value is not None:
            mask &= s <= float(max_value)

        out = self.df.loc[mask].copy()
        audit(self.logger, "filter_range", details=f"col={column}, min={min_value}, max={max_value}, matches={len(out)}")
        return out

    # ------------------------- mutations -------------------------
    def delete_patient(self, patient_id: str) -> Tuple[int, int]:
        """Delete all rows for a given patient ID.

        Returns (deleted_rows, remaining_rows).
        """
        if self.df.empty:
            raise ValueError("No data loaded.")

        pid = str(patient_id).strip()
        before = len(self.df)
        self.df = self.df.loc[self.df[self.id_col] != pid].copy()
        after = len(self.df)
        deleted = before - after
        if deleted > 0:
            self.dirty = True
        audit(self.logger, "delete_patient", patient_id=pid, details=f"deleted={deleted}")
        return deleted, after

    def update_patient_field(self, patient_id: str, column: str, new_value: str) -> int:
        """Update *column* to *new_value* for the given patient_id.

        Returns number of rows updated (usually 1).
        """
        if self.df.empty:
            raise ValueError("No data loaded.")
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found.")

        pid = str(patient_id).strip()
        mask = self.df[self.id_col] == pid
        n_match = int(mask.sum())
        if n_match == 0:
            audit(self.logger, "update_patient_field", patient_id=pid, details=f"col={column}, updated=0 (id not found)")
            return 0

        # Cast to numeric if existing column is numeric-like
        col = self.df[column]
        v: object = new_value
        if pd.api.types.is_numeric_dtype(col):
            v = pd.to_numeric(pd.Series([new_value]), errors="coerce").iloc[0]
        else:
            # attempt numeric cast if the column looks numeric in content
            if pd.to_numeric(col, errors="coerce").notna().mean() > 0.8:
                v = pd.to_numeric(pd.Series([new_value]), errors="coerce").iloc[0]
        self.df.loc[mask, column] = v
        self.dirty = True
        audit(self.logger, "update_patient_field", patient_id=pid, details=f"col={column}, updated={n_match}, value={new_value}")
        return n_match


def format_df_for_cli(df: pd.DataFrame, *, max_rows: int = 10) -> str:
    """Pretty-print helper for CLI outputs."""
    if df.empty:
        return "(no rows)"
    shown = df.head(max_rows)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        txt = shown.to_string(index=False)
    if len(df) > max_rows:
        txt += f"\n... ({len(df) - max_rows} more rows)"
    return txt
