# prompts/dataset_builder.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence
from rules.rule_bank import DualRiskRuleBank
from prompts.problem_builder import build_problem
from prompts.answer_builder import build_answer
from features.state_profile import StateProfile

def build_training_record(record: Mapping[str, Any], retrieval: Mapping[str, Any],
                          top_rules_per_task: int = 3, max_conditions_per_rule: int = 5,
                          state_profile: Optional[StateProfile] = None,
                          include_rule_phrases: bool = False) -> Dict[str, str]:
    answer = build_answer(record)
    return {
        "problem": build_problem(record, retrieval, top_rules_per_task=top_rules_per_task,
                                 max_conditions_per_rule=max_conditions_per_rule,
                                 state_profile=state_profile, include_rule_phrases=include_rule_phrases),
        "answer": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
    }

def build_prompts(records: Sequence[Mapping[str, Any]], bank: DualRiskRuleBank, top_k_vehicle: int = 3,
                  top_k_instruction: int = 3, min_precision: float = 0.0,
                  include_label_sidecar: bool = True, state_mode: str = "focused") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in records:
        retrieval = bank.retrieve(r, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction,
                                  min_precision=min_precision)
        out.append(build_training_record(r, retrieval, top_rules_per_task=max(top_k_vehicle, top_k_instruction),
                                         state_profile=getattr(bank, "state_profile", None)))
    return out