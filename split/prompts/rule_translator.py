# prompts/rule_translator.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from rules.rule_models import MinedRule
from prompts.state_context import _display_state_name_value

def _rule_evidence(rule: MinedRule, record: Optional[Mapping[str, Any]] = None, max_conditions: int = 5) -> List[str]:
    conds: List[str] = []
    for x in rule.compact_conditions(max_conditions=max_conditions + 4):
        if x.startswith("query 包含") and x not in conds:
            conds.append(x)
    if record is not None:
        for f in rule.state_fields:
            name, value = _display_state_name_value(record, f)
            item = f"{name}={value}"
            if item not in conds:
                conds.append(item)
            if len(conds) >= max_conditions:
                break
    if not conds:
        conds = [x for x in rule.compact_conditions(max_conditions=max_conditions + 2) if not x.startswith("query 不包含")]
    return (conds or rule.compact_conditions(max_conditions=1))[:max_conditions]

def _rule_to_understanding(rule: MinedRule, record: Optional[Mapping[str, Any]] = None,
                           max_conditions: int = 5, include_phrases: bool = False) -> Dict[str, Any]:
    label_name = "vehicle_risk" if rule.task == "vehicle" else "instruction_risk"
    d: Dict[str, Any] = {
        "tree_understanding": f"这些命中证据在训练树叶中主要对应 {label_name}={rule.label}；这只是候选假设，可被当前完整状态和指令推翻。",
        "evidence": _rule_evidence(rule, record=record, max_conditions=max_conditions),
    }
    if include_phrases and rule.phrases:
        d["phrases"] = rule.phrases[:2]
    return d