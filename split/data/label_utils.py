# data/label_utils.py
from __future__ import annotations
from typing import Any, List, Mapping, Sequence
from utils.constants import RISK_ORDER, ORDER_RISK

def normalize_label(x: Any) -> str:
    s = str(x or "safe").strip().lower()
    if s in {"none", "normal", "low", "0"}:
        return "safe"
    if s in {"mid", "moderate", "1"}:
        return "medium"
    if s in {"severe", "danger", "2"}:
        return "high"
    return s if s in RISK_ORDER else "safe"

def max_risk(a: str, b: str) -> str:
    return ORDER_RISK[max(RISK_ORDER[normalize_label(a)], RISK_ORDER[normalize_label(b)])]

def expected_label(record: Mapping[str, Any], task: str) -> str:
    exp = record.get("expected") or {}
    if task == "vehicle":
        return normalize_label(exp.get("vehicle_risk", "safe"))
    if task == "instruction":
        return normalize_label(exp.get("instruction_risk", "safe"))
    raise ValueError(f"unknown task: {task}")

def task_phrases(record: Mapping[str, Any], task: str) -> List[str]:
    exp = record.get("expected") or {}
    if task == "vehicle":
        label = normalize_label(exp.get("vehicle_risk", "safe"))
        if label == "safe":
            return []
        kws = exp.get("all_vehicle_risk_keywords") or []
    elif task == "instruction":
        label = normalize_label(exp.get("instruction_risk", "safe"))
        if label == "safe":
            return []
        kws = exp.get("all_instruction_risk_keywords") or []
    else:
        raise ValueError(f"unknown task: {task}")
    clean: List[str] = []
    for x in kws:
        s = str(x).strip()
        if s and s not in clean:
            clean.append(s)
    if clean:
        return clean
    tip = str(exp.get("risk_tip", "")).strip()
    return [tip] if tip else []

def _clean_keyword_list(values: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for x in values:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out
