# prompts/answer_builder.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping
from data.label_utils import normalize_label, _clean_keyword_list

def build_answer(record: Mapping[str, Any]) -> Dict[str, Any]:
    exp = record.get("expected") or {}
    v_lab = normalize_label(exp.get("vehicle_risk", "safe"))
    i_lab = normalize_label(exp.get("instruction_risk", "safe"))
    v_kws = _clean_keyword_list(exp.get("all_vehicle_risk_keywords", [])) if v_lab != "safe" else []
    i_kws = _clean_keyword_list(exp.get("all_instruction_risk_keywords", [])) if i_lab != "safe" else []
    all_kws: List[str] = []
    for s in v_kws + i_kws:
        if s not in all_kws:
            all_kws.append(s)
    return {
        "vehicle_risk": v_lab,
        "instruction_risk": i_lab,
        "all_vehicle_risk_keywords": v_kws,
        "all_instruction_risk_keywords": i_kws,
        "all_risk_keywords": all_kws,
    }