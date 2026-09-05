from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def dataframe_payload(frame: pd.DataFrame | None) -> dict[str, list]:
    if frame is None or frame.empty:
        return {"columns": [], "rows": []}
    normalized = frame.copy()
    rows = [
        {str(column): json_value(value) for column, value in row.items()}
        for row in normalized.to_dict(orient="records")
    ]
    return {"columns": [str(column) for column in normalized.columns], "rows": rows}


def records_payload(records: list[dict] | None) -> dict[str, list]:
    if not records:
        return {"columns": [], "rows": []}
    columns = list(dict.fromkeys(key for record in records for key in record))
    rows = [json_value(record) for record in records]
    return {"columns": columns, "rows": rows}
