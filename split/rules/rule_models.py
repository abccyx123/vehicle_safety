# rules/rule_models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RuleCondition:
    feature_index: int
    feature_name: str
    op: str
    threshold: float
    text: str
    raw_field: Optional[str] = None
    query_fragment: Optional[str] = None
    positive: bool = True

@dataclass
class MinedRule:
    rule_id: str
    task: str
    tree_index: int
    leaf_id: int
    label: str
    support: int
    precision: float
    class_counts: Dict[str, int]
    conditions: List[RuleCondition]
    phrases: List[str] = field(default_factory=list)
    exemplar_ids: List[Any] = field(default_factory=list)
    score_base: float = 0.0

    @property
    def state_fields(self) -> List[str]:
        out: List[str] = []
        for c in self.conditions:
            if c.raw_field and c.raw_field not in out:
                out.append(c.raw_field)
        return out

    @property
    def query_fragments(self) -> List[str]:
        out: List[str] = []
        for c in self.conditions:
            if c.query_fragment and c.positive and c.query_fragment not in out:
                out.append(c.query_fragment)
        return out

    def compact_conditions(self, max_conditions: int = 8) -> List[str]:
        positives: List[str] = []
        state_or_numeric: List[str] = []
        negatives: List[str] = []
        for c in self.conditions:
            if c.query_fragment and c.positive:
                positives.append(c.text)
            elif c.raw_field:
                state_or_numeric.append(c.text)
            else:
                negatives.append(c.text)
        merged: List[str] = []
        for group in (positives, state_or_numeric):
            for x in group:
                if x not in merged:
                    merged.append(x)
                if len(merged) >= max_conditions:
                    return merged
        for x in negatives[:2]:
            if x not in merged:
                merged.append(x)
            if len(merged) >= max_conditions:
                return merged
        return merged

@dataclass
class RetrievalResult:
    task: str
    predicted_label: str
    label_votes: Dict[str, float]
    rules: List[MinedRule]