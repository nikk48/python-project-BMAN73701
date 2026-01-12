from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional


def get_audit_logger(log_path: Path, name: str = "audit") -> logging.Logger:
    """Return a configured logger that writes an audit trail to *log_path*.

    - Ensures the parent directory exists.
    - Avoids adding duplicate handlers if called multiple times.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Ensure directory exists (fall back to a temp directory if the chosen path isn't writable)
    log_path = Path(log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_dir = Path(tempfile.gettempdir()) / "python_project_logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        log_path = fallback_dir / log_path.name

    # If already configured, just return it
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path:
            return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def audit(
    logger: logging.Logger,
    action: str,
    *,
    patient_id: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    """Convenience helper to write a structured audit message."""

    parts = [f"action={action}"]
    if patient_id:
        parts.append(f"id={patient_id}")
    if details:
        parts.append(f"details={details}")
    logger.info(" | ".join(parts))
