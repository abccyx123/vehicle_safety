# pipelines/run_pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
from pipelines.oof_prompts_pipeline import run_oof_prompts
from pipelines.build_prompts_pipeline import run_build_prompts

def run_pipeline(train_path: str | Path,
                 output_dir: str | Path,
                 eval_path: Optional[str | Path] = None,
                 split: Optional[str] = None,
                 eval_split: Optional[str] = None,
                 skip_parquet: bool = False,
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
                 state_mode: str = "focused") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Run OOF prompts
    oof_args = {
        "train_path": train_path,
        "output_path": out / "train_dataset_oof.jsonl",
        "split": split,
        "parquet_output": out / "train_dataset_oof.parquet",
        "skip_parquet": skip_parquet,
        "model_out": out / "rule_bank.joblib",
        "rules_out": out / "rules.json",
        "summary_out": out / "train_oof_summary.json",
        "folds": folds,
        "top_k_vehicle": top_k_vehicle,
        "top_k_instruction": top_k_instruction,
        "min_precision": min_precision,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "max_query_features": max_query_features,
        "min_query_df": min_query_df,
        "max_auto_categorical_cardinality": max_auto_categorical_cardinality,
        "no_label_sidecar": no_label_sidecar,
        "state_mode": state_mode,
    }
    run_oof_prompts(**oof_args)
    # If eval provided, build eval prompts using final bank
    if eval_path:
        bp_args = {
            "model_path": out / "rule_bank.joblib",
            "input_path": eval_path,
            "output_path": out / "eval_dataset.jsonl",
            "split": eval_split,
            "parquet_output": out / "eval_dataset.parquet",
            "skip_parquet": skip_parquet,
            "top_k_vehicle": top_k_vehicle,
            "top_k_instruction": top_k_instruction,
            "min_precision": min_precision,
            "no_label_sidecar": no_label_sidecar,
            "state_mode": state_mode,
            "eval_out": out / "eval_rule_vote_metrics.json",
        }
        run_build_prompts(**bp_args)
