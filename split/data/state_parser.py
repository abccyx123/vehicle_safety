# data/state_parser.py
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional
import numpy as np

def ensure_state(record: Mapping[str, Any]) -> Mapping[str, Any]:
    state = record.get("state_raw")
    if isinstance(state, Mapping):
        return state
    state = record.get("state")
    if isinstance(state, Mapping):
        return state
    return {}

def infer_raw_to_readable(record: Mapping[str, Any]) -> Dict[str, str]:
    raw = record.get("state_raw") or {}
    readable = record.get("state") or {}
    if not isinstance(raw, Mapping) or not isinstance(readable, Mapping):
        return {}
    raw_keys = list(raw.keys())
    rd_keys = list(readable.keys())
    if len(raw_keys) != len(rd_keys):
        return {}
    return {str(rk): str(rd_keys[i]) for i, rk in enumerate(raw_keys)}

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            if np.isnan(float(x)):
                return None
        except Exception:
            pass
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _norm_cat(x: Any) -> str:
    if x is None:
        return "<MISSING>"
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x).strip()