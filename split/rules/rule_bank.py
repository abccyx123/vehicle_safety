# rules/rule_bank.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import joblib
from rules.rule_miner import AutoRuleMiner
from features.state_profile import StateProfile
from rules.rule_models import RetrievalResult
from utils.constants import SEVERITY_WEIGHT  # not directly used but imported for clarity

class DualRiskRuleBank:
    def __init__(self, vehicle_miner: AutoRuleMiner, instruction_miner: AutoRuleMiner,
                 metadata: Optional[Dict[str, Any]] = None, state_profile: Optional[StateProfile] = None):
        self.vehicle_miner = vehicle_miner
        self.instruction_miner = instruction_miner
        self.metadata = metadata or {}
        self.state_profile = state_profile

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]], n_estimators: int = 80, max_depth: int = 8,
            min_samples_leaf: int = 5, random_state: int = 42, max_query_features: int = 1200,
            min_query_df: int = 2, max_auto_categorical_cardinality: int = 20) -> "DualRiskRuleBank":
        vehicle = AutoRuleMiner(
            "vehicle", n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state, max_query_features=max_query_features, min_query_df=min_query_df,
            max_auto_categorical_cardinality=max_auto_categorical_cardinality,
        ).fit(records)
        instruction = AutoRuleMiner(
            "instruction", n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state + 1009, max_query_features=max_query_features, min_query_df=min_query_df,
            max_auto_categorical_cardinality=max_auto_categorical_cardinality,
        ).fit(records)
        state_profile = StateProfile(max_auto_categorical_cardinality=max_auto_categorical_cardinality).fit(records)
        md = {
            "n_train": len(records),
            "random_state": random_state,
            "uses_fields": ["query", "state_raw/state", "expected labels and per-case phrases for leaf annotation only"],
            "excludes_fields": ["command_family", "scene_name", "vehicle_scenario_type", "instruction_scenario_type"],
            "vehicle_summary": vehicle.train_summary,
            "instruction_summary": instruction.train_summary,
            "state_profile": "unsupervised robust numeric/categorical rarity profile",
        }
        return cls(vehicle, instruction, md, state_profile=state_profile)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DualRiskRuleBank":
        return joblib.load(path)

    def retrieve(self, record: Mapping[str, Any], top_k_vehicle: int = 5, top_k_instruction: int = 5,
                 min_precision: float = 0.0) -> Dict[str, RetrievalResult]:
        return {
            "vehicle": self.vehicle_miner.retrieve(record, top_k=top_k_vehicle, min_precision=min_precision),
            "instruction": self.instruction_miner.retrieve(record, top_k=top_k_instruction, min_precision=min_precision),
        }

    def export_rules(self, path: str | Path, max_rules_per_task: int = 5000) -> None:
        payload = {
            "metadata": self.metadata,
            "vehicle_rules": self.vehicle_miner.rules_as_jsonable()[:max_rules_per_task],
            "instruction_rules": self.instruction_miner.rules_as_jsonable()[:max_rules_per_task],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")