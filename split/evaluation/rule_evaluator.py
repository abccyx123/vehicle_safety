# evaluation/rule_evaluator.py
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, Mapping, Sequence
from data.label_utils import expected_label
from data.io import record_id
from rules.rule_bank import DualRiskRuleBank

def evaluate_rule_votes(records: Sequence[Mapping[str, Any]], bank: DualRiskRuleBank,
                        top_k_vehicle: int = 5, top_k_instruction: int = 5) -> Dict[str, Any]:
    rows = []
    cv = Counter(); ci = Counter(); nv = 0; ni = 0
    for r in records:
        ret = bank.retrieve(r, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction)
        gt_v = expected_label(r, "vehicle")
        gt_i = expected_label(r, "instruction")
        pv = ret["vehicle"].predicted_label
        pi = ret["instruction"].predicted_label
        nv += int(pv == gt_v)
        ni += int(pi == gt_i)
        cv[(gt_v, pv)] += 1
        ci[(gt_i, pi)] += 1
        rows.append({"id": record_id(r), "vehicle_true": gt_v, "vehicle_pred": pv, "instruction_true": gt_i, "instruction_pred": pi})
    return {
        "n": len(records),
        "vehicle_accuracy_rule_vote": round(nv / max(1, len(records)), 4),
        "instruction_accuracy_rule_vote": round(ni / max(1, len(records)), 4),
        "vehicle_confusion": {f"{a}->{b}": int(c) for (a, b), c in sorted(cv.items())},
        "instruction_confusion": {f"{a}->{b}": int(c) for (a, b), c in sorted(ci.items())},
        "rows": rows,
    }