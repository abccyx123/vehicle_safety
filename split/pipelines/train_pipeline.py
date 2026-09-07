# pipelines/train_pipeline.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
from data.io import load_jsonl
from rules.rule_bank import DualRiskRuleBank
from evaluation.rule_evaluator import evaluate_rule_votes

def run_train(train_path: str | Path,
              model_out: str | Path,
              split: Optional[str] = None,
              rules_out: Optional[str | Path] = None,
              summary_out: Optional[str | Path] = None,
              n_estimators: int = 80,
              max_depth: int = 8,
              min_samples_leaf: int = 5,
              random_state: int = 42,
              max_query_features: int = 1200,
              min_query_df: int = 2,
              max_auto_categorical_cardinality: int = 20) -> Dict[str, Any]:
    """Fit and persist a dual-task rule bank, optionally writing rule/summary files."""
    records = load_jsonl(train_path, split=split)
    if not records:
        raise SystemExit("No training records loaded")
    bank = DualRiskRuleBank.fit(
        records,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        max_query_features=max_query_features,
        min_query_df=min_query_df,
        max_auto_categorical_cardinality=max_auto_categorical_cardinality,
    )
    bank.save(model_out)
    if rules_out:
        bank.export_rules(rules_out)
    summary = {"train": bank.metadata, "self_eval": evaluate_rule_votes(records, bank)}
    if summary_out:
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
