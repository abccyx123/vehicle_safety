# pipelines/build_prompts_pipeline.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from data.io import load_jsonl, write_dataset_outputs
from rules.rule_bank import DualRiskRuleBank
from prompts.dataset_builder import build_prompts
from evaluation.rule_evaluator import evaluate_rule_votes

def run_build_prompts(model_path: str | Path,
                      input_path: str | Path,
                      output_path: str | Path,
                      split: Optional[str] = None,
                      parquet_output: Optional[str | Path] = None,
                      skip_parquet: bool = False,
                      top_k_vehicle: int = 3,
                      top_k_instruction: int = 3,
                      min_precision: float = 0.50,
                      no_label_sidecar: bool = False,
                      state_mode: str = "focused",
                      eval_out: Optional[str | Path] = None) -> None:
    bank = DualRiskRuleBank.load(model_path)
    records = load_jsonl(input_path, split=split)
    prompts = build_prompts(records, bank,
                            top_k_vehicle=top_k_vehicle,
                            top_k_instruction=top_k_instruction,
                            min_precision=min_precision,
                            include_label_sidecar=not no_label_sidecar,
                            state_mode=state_mode)
    jsonl_path, parquet_path = write_dataset_outputs(
        output_path, prompts,
        parquet_path=parquet_output,
        skip_parquet=skip_parquet,
    )
    if eval_out:
        ev = evaluate_rule_votes(records, bank, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction)
        Path(eval_out).parent.mkdir(parents=True, exist_ok=True)
        Path(eval_out).write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_records": len(prompts), "jsonl": str(jsonl_path), "parquet": str(parquet_path) if parquet_path else None}, ensure_ascii=False, indent=2))
