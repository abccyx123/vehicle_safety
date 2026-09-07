# pipelines/oof_prompts_pipeline.py
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from data.io import load_jsonl, write_dataset_outputs
from data.label_utils import expected_label
from rules.rule_bank import DualRiskRuleBank
from prompts.dataset_builder import build_prompts
from evaluation.rule_evaluator import evaluate_rule_votes

def run_oof_prompts(train_path: str | Path, output_path: str | Path,
                    split: Optional[str] = None,
                    parquet_output: Optional[str | Path] = None,
                    skip_parquet: bool = False,
                    model_out: Optional[str | Path] = None,
                    rules_out: Optional[str | Path] = None,
                    summary_out: Optional[str | Path] = None,
                    folds: int = 5,
                    top_k_vehicle: int = 3,
                    top_k_instruction: int = 3,
                    min_precision: float = 0.50,
                    n_estimators: int = 80,
                    max_depth: int = 8,
                    min_samples_leaf: int = 5,
                    random_state: int = 42,
                    max_query_features: int = 1200,
                    min_query_df: int = 2,
                    max_auto_categorical_cardinality: int = 20,
                    no_label_sidecar: bool = False,
                    state_mode: str = "focused") -> Dict[str, Any]:
    records = load_jsonl(train_path, split=split)
    if not records:
        raise SystemExit("No training records loaded")
    y_joint = [expected_label(r, "vehicle") + "|" + expected_label(r, "instruction") for r in records]
    n_splits = max(2, folds)
    counts = Counter(y_joint)
    min_count = min(counts.values()) if counts else 2
    if min_count >= 2:
        n_splits = min(n_splits, min_count)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = skf.split(np.zeros(len(records)), y_joint)
    else:
        n_splits = min(n_splits, len(records))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = kf.split(np.zeros(len(records)))
    all_prompts: List[Dict[str, Any]] = [None] * len(records)
    fold_summaries = []
    for fold, (train_idx, hold_idx) in enumerate(split_iter):
        fold_train = [records[i] for i in train_idx]
        fold_hold = [records[i] for i in hold_idx]
        bank = DualRiskRuleBank.fit(
            fold_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold * 37,
            max_query_features=max_query_features,
            min_query_df=min_query_df,
            max_auto_categorical_cardinality=max_auto_categorical_cardinality,
        )
        prompts = build_prompts(fold_hold, bank,
                                top_k_vehicle=top_k_vehicle,
                                top_k_instruction=top_k_instruction,
                                min_precision=min_precision,
                                include_label_sidecar=not no_label_sidecar,
                                state_mode=state_mode)
        for pos, p in zip(hold_idx, prompts):
            all_prompts[int(pos)] = p
        ev = evaluate_rule_votes(fold_hold, bank, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction)
        fold_summaries.append({k: v for k, v in ev.items() if k != "rows"} | {"fold": fold})
    jsonl_path, parquet_path = write_dataset_outputs(
        output_path, all_prompts,
        parquet_path=parquet_output,
        skip_parquet=skip_parquet,
    )
    # Fit final bank on all records for future use
    final_bank = DualRiskRuleBank.fit(
        records,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        max_query_features=max_query_features,
        min_query_df=min_query_df,
        max_auto_categorical_cardinality=max_auto_categorical_cardinality,
    )
    if model_out:
        final_bank.save(model_out)
    if rules_out:
        final_bank.export_rules(rules_out)
    summary = {
        "n_train_prompts": len(all_prompts),
        "folds": n_splits,
        "fold_summaries": fold_summaries,
        "final_self_eval": evaluate_rule_votes(records, final_bank, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction),
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path) if parquet_path else None,
    }
    if summary_out:
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "final_self_eval"}, ensure_ascii=False, indent=2))
    return summary
