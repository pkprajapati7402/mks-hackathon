"""
Audit trail logging system for tracking every row-level change made during data cleaning.
"""

from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
from src.config import REPORTS_DIR

class AuditLogger:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def log(self, file_name: str, row_key: Any, field: str, raw_value: Any,
            new_value: Any, rule_id: str, reason: str):
        """Record a single modification to a data field."""
        self._records.append({
            "file": file_name,
            "row_key": str(row_key),
            "field": field,
            "raw_value": str(raw_value) if raw_value is not None else "",
            "new_value": str(new_value) if new_value is not None else "",
            "rule_id": rule_id,
            "reason": reason
        })

    def count(self) -> int:
        return len(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=[
                "file", "row_key", "field", "raw_value", "new_value", "rule_id", "reason"
            ])
        return pd.DataFrame(self._records)

    def save(self, output_path: Path = None):
        if output_path is None:
            output_path = REPORTS_DIR / "audit_trail.parquet"
        df = self.to_dataframe()
        df.to_parquet(output_path, index=False)
        return output_path

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return df.groupby(["file", "rule_id", "field"]).size().reset_index(name="modification_count")

    def clear(self):
        self._records.clear()

audit = AuditLogger()
