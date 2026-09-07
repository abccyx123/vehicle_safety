# prompts/problem_builder.py
from __future__ import annotations
import json
from typing import Mapping, Optional
from prompts.rule_translator import _rule_to_understanding
from prompts.state_context import build_full_state_context
from rules.rule_models import RetrievalResult
from features.state_profile import StateProfile

def build_problem(record: Mapping[str, Any], retrieval: Mapping[str, RetrievalResult],
                  top_rules_per_task: int = 3, max_conditions_per_rule: int = 5,
                  state_profile: Optional[StateProfile] = None, include_rule_phrases: bool = False) -> str:
    vehicle_rules = [_rule_to_understanding(r, record=record, max_conditions=max_conditions_per_rule,
                                            include_phrases=include_rule_phrases)
                     for r in retrieval["vehicle"].rules[:top_rules_per_task]]
    instr_rules = [_rule_to_understanding(r, record=record, max_conditions=max_conditions_per_rule,
                                          include_phrases=include_rule_phrases)
                   for r in retrieval["instruction"].rules[:top_rules_per_task]]
    state_ctx = build_full_state_context(record, retrieval, state_profile=state_profile)
    payload = {
        "query": str(record.get("query", "")),
        "state": state_ctx["all"],
        "focus": state_ctx["focus"],
        "rare": state_ctx["rare"],
        "tree_understanding": {
            "vehicle": vehicle_rules,
            "instruction": instr_rules,
        },
        "required_output": {
            "only_json": True,
            "schema": {
                "vehicle_risk": "safe|medium|high",
                "instruction_risk": "safe|medium|high",
                "all_vehicle_risk_keywords": ["简短的解释一下车辆风险；vehicle_risk=safe 时必须为空数组"],
                "all_instruction_risk_keywords": ["简短的解释一下指令风险；instruction_risk=safe 时必须为空数组"],
                "all_risk_keywords": ["前两类关键词去重后的并集"],
            },
        },
    }
    return (
        "你是车载语音指令安全评估器。请分别判断当前车辆状态 vehicle_risk 与执行驾驶员语音指令的 "
        "instruction_risk。等级只能是 safe、medium、high。\n"
        "输入中的 state 是当前车辆全部状态，用于核查参考规则和发现规则以外的风险；"
        "focus 是供参考的规则中命中的字段名列表，rare 是训练分布中罕见的字段名列表。\n"
        "tree_understanding 是树/森林从训练数据自动归纳出的候选理解，不是最终答案；"
        "当完整状态或 query 与候选理解冲突时，应按照实际状态和 query 判断。\n"
        "最终答案必须且只能是一个 JSON 对象，并严格符合 required_output.schema。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )