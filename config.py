from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AED_FILE_PATH = BASE_DIR / "AED4weeks.csv"
AED_RANDOM_SEED = 42
AED_SAMPLE_N = 400

TASK6_TARGET_RECALL = 0.80

# Audit trail for interactive data management
LOG_DIR = BASE_DIR / "logs"
AUDIT_LOG_PATH = LOG_DIR / "audit.log"
