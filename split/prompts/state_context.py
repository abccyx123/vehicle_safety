# prompts/state_context.py
from __future__ import annotations
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from data.state_parser import ensure_state, infer_raw_to_readable, _to_float, _norm_cat
from rules.rule_models import MinedRule, RetrievalResult
from features.state_profile import StateProfile

def _short_value(value: Any, max_len: int = 40) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        x = float(value)
        if math.isnan(x):
            return ""
        return f"{x:.6g}"
    v = str(value).strip()
    return v if len(v) <= max_len else v[:max_len] + "…"

def _display_state_name_value(record: Mapping[str, Any], raw_field: str) -> Tuple[str, str]:
    raw = ensure_state(record)
    readable_map = infer_raw_to_readable(record)
    readable_state = record.get("state") if isinstance(record.get("state"), Mapping) else {}
    rd = readable_map.get(raw_field)
    name = str(rd or raw_field)
    value = readable_state.get(rd) if rd and isinstance(readable_state, Mapping) else raw.get(raw_field)
    return name, _short_value(value)

def _all_state_name_text(record: Mapping[str, Any], max_state_fields: int = 512) -> Dict[str, str]:
    out: Dict[str, str] = {}
    readable_state = record.get("state") if isinstance(record.get("state"), Mapping) else None
    if readable_state:
        for k, v in list(readable_state.items())[:max_state_fields]:
            name = str(k)
            if name in out:
                name = f"{name}#{len(out)}"
            out[name] = _short_value(v)
        return out
    raw = ensure_state(record)
    for k in list(raw.keys())[:max_state_fields]:
        name, value = _display_state_name_value(record, str(k))
        if name in out:
            name = f"{name}({k})"
        out[name] = value
    return out

def _state_field_names(record: Mapping[str, Any], fields: Sequence[str], max_fields: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seq = fields[:max_fields] if max_fields is not None else fields
    for f in seq:
        name, _ = _display_state_name_value(record, str(f))
        if name not in out:
            out.append(name)
    return out

def _unique_state_fields_from_rules(rules: Sequence[MinedRule], max_fields: Optional[int] = None) -> List[str]:
    fields: List[str] = []
    for r in rules:
        for f in r.state_fields:
            if f not in fields:
                fields.append(f)
    if max_fields is not None:
        fields = fields[:max_fields]
    return fields

def build_full_state_context(record: Mapping[str, Any], retrieval: Mapping[str, RetrievalResult],
                             state_profile: Optional[StateProfile] = None, max_focus_fields: int = 24,
                             max_anomaly_fields: int = 8, max_state_fields: int = 512) -> Dict[str, Any]:
    fields: List[str] = []
    for res in retrieval.values():
        for f in _unique_state_fields_from_rules(res.rules):
            if f not in fields:
                fields.append(f)
    fields = fields[:max_focus_fields]
    anomaly_pairs: List[Tuple[str, float]] = []
    if state_profile is not None:
        anomaly_pairs = state_profile.top_anomalies(record, exclude_fields=fields, max_fields=max_anomaly_fields)
    anomaly_fields = [k for k, _ in anomaly_pairs]
    return {
        "all": _all_state_name_text(record, max_state_fields=max_state_fields),
        "focus": _state_field_names(record, fields),
        "rare": _state_field_names(record, anomaly_fields),
    }